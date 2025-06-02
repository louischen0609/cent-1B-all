import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 指定使用 GPU 0
import json
import yaml
from typing import Union
import wandb # Import wandb
os.environ["WANDB_DISABLE_SYSTEM"] = "true"
from bitsandbytes import nn as bnb_nn


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
    # json_data_dir = config["data_path"] # 不再使用本地 JSON 資料夾
    num_epochs = config["num_epochs"] # 從 epochs_per_json 改名

    # if not os.path.isdir(json_data_dir): # 移除資料夾檢查
    #     raise ValueError(f"Configured data_path '{json_data_dir}' is not a directory or does not exist.")

    # json_files = sorted([f for f in os.listdir(json_data_dir) if f.startswith("client_partition_") and f.endswith(".json")]) # 移除檔案列表邏輯
    # if not json_files:
    #     print(f"Warning: No 'client_partition_X.json' files found in '{json_data_dir}'.")
    #     return

    # Get quantization config from global config
    load_in_4bit = config.get("load_in_4bit", False)
    load_in_8bit = config.get("load_in_8bit", True)

    # Get Wandb config
    use_wandb = config.get("use_wandb", False)
    wandb_project = config.get("wandb_project", "centralized-finetuning")
    wandb_run_name_prefix = config.get("wandb_run_name_prefix", "run-")
    wandb_entity = config.get("wandb_entity")

    # for json_file in json_files: # 移除遍歷 JSON 檔案的迴圈
    # current_data_path = os.path.join(json_data_dir, json_file) # 移除
    # current_file_output_dir = os.path.join(base_output_dir, os.path.splitext(json_file)[0]) # 簡化輸出目錄
    
    # 直接使用 base_output_dir，因為不再有每個檔案的子目錄
    output_dir_for_run = base_output_dir 
    if not os.path.exists(output_dir_for_run):
        os.makedirs(output_dir_for_run)
        
    if use_wandb:
        # 更新 wandb run name，移除檔案特定資訊
        run_name = f"{wandb_run_name_prefix}-rank{config['lora_r']}" 
        wandb.init(
            project=wandb_project,
            name=run_name,
            entity=wandb_entity, 
            config=config, 
            settings=wandb.Settings(x_disable_stats=True) 
        )
        print(f"Wandb initialized for run: {run_name}")

    # print(f"Processing data file: {current_data_path} for {num_epochs} epochs.") # 更新日誌
    # print(f"Output for this file will be saved in: {output_dir_for_run}") # 更新日誌
    print(f"Training with databricks/databricks-dolly-15k dataset for {num_epochs} epochs.")
    print(f"Output will be saved in: {output_dir_for_run}")


    global_model = config["global_model"]
    # test_path = config["test_path"] # test_path 暫不在此處使用
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

    print(f"Loading base model ({global_model}) and tokenizer...") # 更新日誌
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
    
    print(f"Initializing new LoRA configuration...") # 更新日誌
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        model.is_parallelizable = True
        model.model_parallel = True
        
    print(f"Loading and tokenizing databricks/databricks-dolly-15k dataset...") # 更新日誌
    # Load the dataset directly from Hugging Face Hub
    dataset = datasets.load_dataset("databricks/databricks-dolly-15k")
    data = dataset["train"] # Assuming 'train' split contains the data
    print("First sample from the loaded dataset:")
    print(data[42])

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


        train_val = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_dataset = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        eval_dataset = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        print("First processed sample from the training dataset (after tokenization):")
        print(train_dataset[0])
    else:
        train_dataset = data.shuffle().map(generate_and_tokenize_prompt)
        eval_dataset = None
        print("First processed sample from the training dataset (after tokenization):")
        print(train_dataset[42])
    print(f"Dataset prepared.") # 更新日誌

    print(f"Centralized Finetuning LLM-LoRA with databricks/databricks-dolly-15k, for {num_epochs} epochs:\\n"
          f"  global_model: {global_model}\\n"
          # f"  data_path: {current_data_path}\\n" # 移除
          f"  output_dir: {output_dir_for_run}\\n"
          )

    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20,
        num_train_epochs=num_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch", 
        output_dir=output_dir_for_run, 
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
        # tokenizer=tokenizer, # SFTTrainer can infer tokenizer from model if not set for some models, but explicit is safer.
                               # However, your original code had processing_class=tokenizer, which is not a valid SFTTrainer arg.
                               # SFTTrainer does not have a processing_class arg.
                               # It typically uses the tokenizer passed to it or infers from the model.
                               # Let's pass the tokenizer explicitly.
        processing_class=tokenizer,
        # max_seq_length=cutoff_len, # Pass cutoff_len as max_seq_length
        # processing_class=tokenizer, # This was an error, SFTTrainer does not have this argument.
    )
    
    print(f"Starting training for {num_epochs} epochs...") # 更新日誌
    trainer.train()
    
    # Save final model and tokenizer
    print(f"Training complete. Saving final model and tokenizer to {output_dir_for_run}")
    trainer.save_model(output_dir_for_run) # Ensure final model is saved
    tokenizer.save_pretrained(output_dir_for_run) # Ensure tokenizer is saved

    print(f"Training for {num_epochs} epochs complete. Model saved in {output_dir_for_run}")
    
    if use_wandb:
        wandb.finish()
        print("Wandb run finished.")

    # Cleanup after all epochs for the current JSON file are done
    print(f"All {num_epochs} epochs completed. Cleaning up model and data...")
    del model
    del tokenizer
    del train_dataset
    if eval_dataset:
        del eval_dataset
    del data
    del dataset # also delete the original dataset object
    del peft_config 
    del trainer 
    torch.cuda.empty_cache()
    print(f"Memory cleared. Training finished.") # 更新日誌


if __name__ == "__main__":
    centralized_finetune() 