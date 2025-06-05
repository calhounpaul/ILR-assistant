import os
import json
import torch
import multiprocessing
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from accelerate import Accelerator

# Global configurations
MODEL_NAME = "Qwen/Qwen3-8B"
#MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
LOAD_IN_4BIT = True  # Set to False for 8-bit or full precision
OUTPUT_DIR = f"output/tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MULTIPLIER = 2 ** 7
LORA_R = MULTIPLIER
LORA_ALPHA = 2 * MULTIPLIER
BATCH_SIZE = 1
GRAD_ACCUMULATION = max(1, int(8 / BATCH_SIZE))
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
WARMUP_RATIO = 0.03
NUM_WORKERS = multiprocessing.cpu_count()
TRUNCATION_LIMIT = 2048
SAVE_LIMIT = 10
SAVE_STEPS = 10
LOGGING_STEPS = 1
MAX_EXAMPLES = -1  # Set to a positive integer to limit dataset size
GRAD_CHECKPOINTING = False  # CRITICAL: Disabled for DDP + LoRA compatibility
WEIGHT_DECAY = 0.01
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    #"gate_proj", "up_proj", "down_proj"
]

def main():
    accelerator = Accelerator()
    device = accelerator.device

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with appropriate quantization
    if LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": accelerator.process_index},  # Critical for multi-GPU + quantization
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map={"": accelerator.process_index},
            trust_remote_code=True,
        )

    # Apply LoRA configuration
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # CRITICAL: Disable gradient checkpointing before applying LoRA
    model.config.use_cache = False
    model.config.gradient_checkpointing = False
    
    model = get_peft_model(model, peft_config)

    # Print trainable parameters for verification
    model.print_trainable_parameters()

    # Load and preprocess dataset
    with open("final_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    if MAX_EXAMPLES > 0:
        data = data[:MAX_EXAMPLES]

    # Convert data to Hugging Face Dataset
    dataset = Dataset.from_list([{"messages": convo} for convo in data])

    # Tokenize dataset using chat template
    def tokenize_function(example):
        messages = example["messages"]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=TRUNCATION_LIMIT,
            return_tensors="pt",
        )
        return {"input_ids": tokenized[0]}

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=["messages"])

    # === SANITY CHECK: De-tokenize one example and save to file ===
    if len(tokenized_dataset) > 0:
        print("=== Performing Sanity Check ===")
        
        # Get the first example
        first_example = tokenized_dataset[0]
        original_messages = data[0]  # Original conversation from JSON
        
        # De-tokenize the tokenized input_ids
        detokenized_text = tokenizer.decode(first_example["input_ids"], skip_special_tokens=False)
        
        # Create debug content
        debug_content = f"""=== SANITY CHECK DEBUG OUTPUT ===
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {MODEL_NAME}
Truncation Limit: {TRUNCATION_LIMIT}

=== ORIGINAL MESSAGES (from JSON) ===
{json.dumps(original_messages, indent=2, ensure_ascii=False)}

=== TOKENIZED AND DE-TOKENIZED TEXT ===
{detokenized_text}

=== TOKEN COUNT ===
Total tokens: {len(first_example["input_ids"])}

=== NOTES ===
- This shows how the chat template transforms the original messages
- Pay attention to whether <think></think> content is preserved or stripped
- The tokenized version is what the model will actually see during training
"""
        
        # Save to file
        with open("debug_example.txt", "w", encoding="utf-8") as f:
            f.write(debug_content)
        
        print(f"✓ Debug example saved to debug_example.txt")
        print(f"✓ Token count for first example: {len(first_example['input_ids'])}")
        print("=" * 50)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments - DDP optimized
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_LIMIT,
        bf16=True,
        gradient_checkpointing=GRAD_CHECKPOINTING,  # Disabled for DDP + LoRA
        weight_decay=WEIGHT_DECAY,
        dataloader_num_workers=NUM_WORKERS,
        report_to="none",
        # DDP specific optimizations
        ddp_find_unused_parameters=False,  # Important for LoRA + DDP
        ddp_bucket_cap_mb=25,  # Reduce bucket size for better memory efficiency
        dataloader_pin_memory=False,  # Disable for quantized models
        remove_unused_columns=False,  # Additional DDP + LoRA stability
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # DDP + LoRA compatibility check
    if accelerator.num_processes > 1:
        print("✓ Multi-GPU DDP training enabled with LoRA")
        print(f"✓ Gradient checkpointing disabled for DDP + LoRA compatibility")
    else:
        print("✓ Single GPU training mode")

    # Start training
    print("=== Starting Training ===")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    print(f"Batch size: {BATCH_SIZE}, Grad accumulation: {GRAD_ACCUMULATION}")
    print(f"Processes: {accelerator.num_processes}")
    print(f"Device: {device}")
    print("=" * 50)
    
    trainer.train()

    # Save the final model
    trainer.save_model(OUTPUT_DIR)
    print(f"✓ Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
