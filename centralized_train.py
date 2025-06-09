import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 指定使用 GPU 0
import json
import yaml
from typing import Union
import wandb # Import wandb
os.environ["WANDB_DISABLE_SYSTEM"] = "true"
from bitsandbytes import nn as bnb_nn
from transformers import EarlyStoppingCallback
import random
import numpy as np  
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
from trl import SFTTrainer

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(self, instruction: str, input: Union[None, str] = None, label: Union[None, str] = None) -> str:
        if input:
            res = self.template["prompt_input"].format(instruction=instruction, input=input)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def centralized_finetune():
    base_output_dir = config["output_dir"] + f"/rank{config['lora_r']}"
    num_epochs = config["num_epochs"]

    # Get quantization config from global config
    load_in_4bit = config.get("load_in_4bit", False)
    load_in_8bit = config.get("load_in_8bit", True)

    # Get Wandb config
    use_wandb = config.get("use_wandb", False)
    wandb_project = config.get("wandb_project", "centralized-finetuning")
    wandb_run_name_prefix = config.get("wandb_run_name_prefix", "run-")
    wandb_entity = config.get("wandb_entity")

    global_model = config["global_model"]
    batch_size = config["batch_size"]
    micro_batch_size = config["micro_batch_size"]
    learning_rate = config["learning_rate"]
    val_set_size = config["val_set_size"]
    cutoff_len = config["cutoff_len"]
    lora_r = config["lora_r"]
    lora_alpha = config["lora_alpha"]
    lora_dropout = config["lora_dropout"]
    lora_target_modules = config["lora_target_modules"]
    train_on_inputs = config["train_on_inputs"]
    group_by_length = config["group_by_length"]
    prompt_template_name = config["prompt_template_name"]
    
    assert global_model, "Please specify a global_model in config.yaml"
    
    print(f"Loading and preparing databricks/databricks-dolly-15k dataset...")
    # Load the dataset directly from Hugging Face Hub
    dataset = datasets.load_dataset("databricks/databricks-dolly-15k")
    data = dataset["train"] # Assuming 'train' split contains the data
    print("First sample from the loaded dataset:")
    print(data[42])
    
    # Get unique categories from the dataset
    unique_categories = set(data["category"])
    print(f"Found categories: {sorted(unique_categories)}")
    
    # Train each category separately
    for category in sorted(unique_categories):
        print(f"\n=== 開始訓練類別: {category} ===")
        
        # Filter data for this category
        category_data = data.filter(lambda x: x["category"] == category)
        print(f"類別 {category} 的樣本數量: {len(category_data)}")
        
        # Create category-specific output directory
        output_dir_for_category = os.path.join(base_output_dir, f"category_{category}")
        if not os.path.exists(output_dir_for_category):
            os.makedirs(output_dir_for_category)
            
        if use_wandb:
            # Category-specific wandb run name
            run_name = f"{wandb_run_name_prefix}-rank{config['lora_r']}-{category}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                entity=wandb_entity, 
                config={**config, "category": category, "category_samples": len(category_data)}, 
                settings=wandb.Settings(x_disable_stats=True) 
            )
            print(f"Wandb initialized for run: {run_name}")

        print(f"類別 {category} 的樣本將被訓練 {num_epochs} 個 epochs")
        print(f"輸出將保存在: {output_dir_for_category}")

        prompter = Prompter(prompt_template_name)
        gradient_accumulation_steps = batch_size // micro_batch_size
        device_map = "auto"

        model_load_args = {
            "torch_dtype": torch.bfloat16, 
            "device_map": device_map,
            "attn_implementation": "flash_attention_2" 
        }

        if load_in_4bit:
            print("Loading model in 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, 
                bnb_4bit_quant_type="nf4" 
            )
            model_load_args["quantization_config"] = quantization_config
        elif load_in_8bit:
            print("Loading model in 8-bit quantization...")
            model_load_args["load_in_8bit"] = True
        else:
            print("Loading model in full precision (bfloat16 as per torch_dtype)...")

        print(f"Loading base model ({global_model}) and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            **model_load_args
        )

        tokenizer = AutoTokenizer.from_pretrained(global_model)
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.padding_side = "left"

        def tokenize(prompt, add_eos_token=True):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            result["labels"] = result["input_ids"].copy()
            return result

        def generate_and_tokenize_prompt(data_point):
            # databricks/databricks-dolly-15k dataset has 'instruction', 'context', 'response'
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point.get("context", None), # Use .get for safety, though 'context' is standard in dolly
                data_point.get("response", None), # Use .get for safety, though 'response' is standard in dolly
            )
            tokenized_full_prompt = tokenize(full_prompt)
            if not train_on_inputs:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point.get("context", None)
                )
                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            return tokenized_full_prompt

        model = prepare_model_for_kbit_training(model)
        
        print(f"Initializing new LoRA configuration...")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        print(f"Trainable parameters: {model.print_trainable_parameters()}")

        if torch.cuda.device_count() > 1:
            print("Using multiple GPUs")
            model.is_parallelizable = True
            model.model_parallel = True
            
        print(f"Tokenizing category {category} dataset...")

        if val_set_size > 0:
            # Ensure val_set_size is an integer if it's a percentage string (e.g., "10%")
            if isinstance(val_set_size, str) and "%" in val_set_size:
                val_set_size = int(val_set_size.replace("%", "")) / 100.0
            
            # Make sure val_set_size is a float for test_size if it's a percentage,
            # or an int if it's an absolute number of samples.
            # The train_test_split function expects test_size to be float between 0.0 and 1.0 or int.
            if isinstance(val_set_size, float) and (val_set_size < 0.0 or val_set_size > 1.0) :
                 raise ValueError("val_set_size as a float must be between 0.0 and 1.0.")
            if isinstance(val_set_size, int) and val_set_size <= 0:
                raise ValueError("val_set_size as an int must be greater than 0.")

            train_val = category_data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            train_dataset = train_val["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            eval_dataset = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            print("First processed sample from the training dataset (after tokenization):")
            print(train_dataset[42] if len(train_dataset) > 42 else train_dataset[0])
        else:
            train_dataset = category_data.shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = None
            print("First processed sample from the training dataset (after tokenization):")
            print(train_dataset[42] if len(train_dataset) > 42 else train_dataset[0])
        print(f"Dataset prepared.")

        print(f"Centralized Finetuning LLM-LoRA with category {category}, for {num_epochs} epochs:\n"
              f"  global_model: {global_model}\n"
              f"  category: {category}\n"
              f"  category_samples: {len(category_data)}\n"
              f"  output_dir: {output_dir_for_category}\n"
              )

        training_args = TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.05,
            weight_decay=0.01,
            num_train_epochs=num_epochs, 
            learning_rate=learning_rate,
            bf16=True,
            #logging_strategy="epoch",
            logging_steps=100,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps", 
            output_dir=output_dir_for_category, 
            save_total_limit=num_epochs, 
            load_best_model_at_end=True,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
            gradient_checkpointing=True,
            lr_scheduler_type="cosine",
            report_to="wandb" if use_wandb else "none" 
        )

        trainer = SFTTrainer(
            model=model, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],
            processing_class=tokenizer,
        )
        
        print(f"Starting training for category {category} for {num_epochs} epochs...")
        trainer.train()
        
        # Save final model and tokenizer
        print(f"Training complete. Saving final model and tokenizer to {output_dir_for_category}")
        trainer.save_model(output_dir_for_category) # Ensure final model is saved
        tokenizer.save_pretrained(output_dir_for_category) # Ensure tokenizer is saved

        print(f"Training for category {category} complete. Model saved in {output_dir_for_category}")
        
        if use_wandb:
            wandb.finish()
            print(f"Wandb run finished for category {category}.")

        # Cleanup after each category
        print(f"Category {category} training completed. Cleaning up model and data...")
        del model
        del tokenizer
        del train_dataset
        if eval_dataset:
            del eval_dataset
        del category_data
        del peft_config 
        del trainer 
        torch.cuda.empty_cache()
        print(f"Memory cleared for category {category}.")
        
    print(f"所有類別訓練完成！")


def seed_all(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 時使用
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[INFO] Global seed set to: {seed}")

if __name__ == "__main__":
    seed_all(config["seed"])
    centralized_finetune() 