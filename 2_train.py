import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
# Using Llama 3.1 for superior reasoning and explanation capabilities
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
NEW_MODEL_NAME = "cloud-architect-v1"
DATA_FILE = "./data_architect/architect_training_data.jsonl"

def train():
    # 1. Load Data
    print(f"Loading dataset from {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # 2. Quantization (4-bit NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 3. Load Base Model
    print(f"Loading Base Model: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, 
        bias="none", task_type="CAUSAL_LM", 
        target_modules="all-linear" # Adapts all layers for maximum knowledge retention
    )

    # --- LLAMA 3.1 PROMPT FORMATTING ---
    # Llama 3.1 uses strict special tokens: <|begin_of_text|>, <|start_header_id|>, etc.
    def formatting_prompts_func(batch):
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

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=SFTConfig(
            output_dir="./results_architect",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            fp16=False,
            bf16=True,
            logging_steps=10,
            max_seq_length=2048,
            packing=True, # Packs short docs together for 3x faster training
            report_to="none",
        ),
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving Model...")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)

if __name__ == "__main__":
    train()