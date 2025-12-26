import random

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config_validation import load_config

# Load and validate configuration
config = load_config("config/training_config.yaml")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def formatting_prompts_func(batch: dict[str, list[str]]) -> list[str]:
    """Format batch into Llama 3.1 chat format."""
    output_texts = []
    for i in range(len(batch['instruction'])):
        system_msg = "You are a Senior Cloud Security Architect. Provide detailed, secure, and compliant technical guidance."
        user_msg = f"{batch['instruction'][i]}\nContext: {batch['input'][i]}"
        assistant_msg = batch['output'][i]

        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_msg}<|eot_id|>"
        )
        output_texts.append(text)
    return output_texts

def train():
    set_seed(config.training.seed)

    # 1. Load Data
    data_file = "./data/architect_training_data.jsonl"
    dataset = load_dataset("json", data_files=data_file, split="train")  # nosec B615
    dataset = dataset.train_test_split(test_size=0.1, seed=config.training.seed)

    # 2. Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.model.quantization,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 3. Model
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        config.model.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    # Enable Gradient Checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)  # nosec B615
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. LoRA
    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora.target_modules
    )
    model = get_peft_model(model, peft_config)

    # 5. Training Arguments
    args = SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.grad_accum_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        fp16=False,
        bf16=True,
        logging_steps=config.training.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        load_best_model_at_end=True,
        report_to="wandb" if config.system.use_wandb else "none",
        max_seq_length=config.training.max_seq_length,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=args,
    )

    print("Starting Professional Training Pipeline...")
    trainer.train()

    trainer.model.save_pretrained(config.model.new_model_name)
    tokenizer.save_pretrained(config.model.new_model_name)

if __name__ == "__main__":
    train()
