#!/usr/bin/env python3
import unsloth
import os
import json
import torch
import datetime
import gc
import math
from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from unsloth import FastLanguageModel, is_bfloat16_supported, unsloth_train
from trl import SFTTrainer
from accelerate import Accelerator

# Disable tokenizer parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize accelerator for multi-GPU support
accelerator = Accelerator()
device = accelerator.device
local_rank = accelerator.local_process_index

# CONFIGURATION - Optimized for stability
MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048  # Reduced from 4096 for stability
LOAD_IN_4BIT = True
OUTPUT_DIR = f"output/tune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# LoRA Configuration - Conservative for stability
LORA_R = 64  # Reduced from 128
LORA_ALPHA = 64  # Equal to rank for stable scaling
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",  # Include all for better adaptation
]

# Training Configuration - Conservative settings
BATCH_SIZE = 4
GRAD_ACCUMULATION = 2  # Effective batch size = 8
LEARNING_RATE = 3e-5  # More conservative than 2e-4
NUM_EPOCHS = 1
WARMUP_RATIO = 0.1  # Increased warmup for stability
WEIGHT_DECAY = 0.01
MAX_EXAMPLES = 4000

# Logging and saving
LOGGING_STEPS = 1
SAVE_STEPS = 50
SAVE_LIMIT = 5

if accelerator.is_main_process:
    print(f"🚀 Starting stable fine-tuning with {MODEL_NAME}")
    print(f"📊 Config: LR={LEARNING_RATE}, Batch={BATCH_SIZE}x{GRAD_ACCUMULATION}, Seq={MAX_SEQ_LENGTH}")
    print(f"🔧 Using {accelerator.num_processes} GPU(s), local rank: {local_rank}")

# Load model with proper device mapping for multi-GPU + quantization
if accelerator.is_main_process:
    print("📥 Loading model and tokenizer...")

# Critical: Proper device mapping for quantized models in multi-GPU setup
device_map = {"": accelerator.local_process_index}
# Critical: Proper device mapping for quantized models in multi-GPU setup
device_map = {"": accelerator.local_process_index}

with accelerator.main_process_first():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
        device_map=device_map,  # Essential for multi-GPU + quantization
    )

# Configure LoRA with stable settings
if accelerator.is_main_process:
    print("🔧 Configuring LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,  # Better numerical stability
    loftq_config=None,
)

def create_chat_dataset(conversations, tokenizer, max_length, accelerator):
    """
    Simple, robust data preprocessing using standard tokenizer truncation.
    No custom logic - let the tokenizer handle everything properly.
    """
    if accelerator.is_main_process:
        print(f"🔄 Processing {len(conversations)} conversations...")
    
    processed_data = []
    skipped = 0
    
    for i, conversation in enumerate(conversations):
        if i >= MAX_EXAMPLES:
            break
            
        try:
            # Basic validation
            if not conversation or not isinstance(conversation, list):
                skipped += 1
                continue
            
            # Filter valid messages
            messages = []
            for msg in conversation:
                if (isinstance(msg, dict) and 
                    msg.get("content") and 
                    isinstance(msg.get("content"), str) and
                    msg.get("role") in ["user", "assistant", "system"]):
                    messages.append(msg)
            
            # Need at least 2 messages for training
            if len(messages) < 2:
                skipped += 1
                continue
            
            # Apply chat template and let tokenizer handle truncation
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Simple validation
            if not formatted_text or len(formatted_text.strip()) < 10:
                skipped += 1
                continue
                
            processed_data.append({"text": formatted_text})
            
            # Progress update
            if accelerator.is_main_process and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{min(len(conversations), MAX_EXAMPLES)} conversations")
                
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️  Skipping conversation {i}: {e}")
            skipped += 1
            continue
    
    if accelerator.is_main_process:
        print(f"✅ Dataset ready: {len(processed_data)} examples, {skipped} skipped")
        
        # Show sample
        if processed_data:
            sample_tokens = tokenizer(processed_data[0]["text"], return_tensors="pt").input_ids
            print(f"📝 Sample length: {len(sample_tokens[0])} tokens")
            print(f"📄 Sample preview: {processed_data[0]['text'][:200]}...")
            
            # Save sample for inspection
            with open("sample_processed.txt", "w", encoding="utf-8") as f:
                f.write(processed_data[0]["text"])
    
    return processed_data

# Load and process dataset
if accelerator.is_main_process:
    print("📂 Loading dataset...")

with accelerator.main_process_first():
    try:
        with open("final_dataset.json", "r", encoding="utf-8") as f:
            raw_conversations = json.load(f)
        if accelerator.is_main_process:
            print(f"📊 Loaded {len(raw_conversations)} raw conversations")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"❌ Error loading dataset: {e}")
        raise

    # Process data with simple, robust approach
    processed_data = create_chat_dataset(raw_conversations, tokenizer, MAX_SEQ_LENGTH, accelerator)

    if not processed_data:
        raise ValueError("❌ No valid conversations after preprocessing!")

    # Create dataset - let transformers handle tokenization and truncation
    if accelerator.is_main_process:
        print("🔨 Creating Hugging Face dataset...")
    dataset = Dataset.from_list(processed_data)

# Memory cleanup
if accelerator.is_main_process:
    print("🧹 Cleaning up memory...")
del raw_conversations, processed_data
gc.collect()
torch.cuda.empty_cache()

# Gradient monitoring callback for debugging
class StabilityMonitor(TrainerCallback):
    def __init__(self, accelerator):
        self.accelerator = accelerator
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs and model and self.accelerator.is_main_process:
            # Check for NaN/inf in logs
            for key, value in logs.items():
                if math.isnan(value) or math.isinf(value):
                    print(f"🚨 ALERT: {key} = {value} - Training unstable!")
                    control.should_training_stop = True
                    return
            
            # Monitor gradient norm
            grad_norm = logs.get("grad_norm", 0)
            if grad_norm > 10.0:
                print(f"⚠️  Large gradient norm: {grad_norm:.3f}")
            elif grad_norm == 0:
                print("⚠️  Zero gradient norm detected")

# Training arguments optimized for stability
if accelerator.is_main_process:
    print("⚙️  Configuring training arguments...")
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
    
    # Precision settings - explicit and conservative
    fp16=False, #False  # Disable fp16 for maximum stability
    bf16=is_bfloat16_supported(),  # Use bf16 if available
    
    # Optimizer settings
    optim="adamw_torch",
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    
    # Gradient management - critical for stability
    max_grad_norm=1.0,  # Conservative gradient clipping
    
    # Memory and performance
    gradient_checkpointing=True,
    dataloader_num_workers=0,  # Avoid multiprocessing issues
    dataloader_pin_memory=False,
    
    # Debugging
    logging_nan_inf_filter=True,  # Auto-filter NaN/inf from logs
    
    # Reproducibility
    seed=3407,
    data_seed=3407,
    
    # Disable external logging
    report_to="none",
)

# Create trainer with proper settings
if accelerator.is_main_process:
    print("🏗️  Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,  # Let SFTTrainer handle truncation properly
    args=training_args,
    packing=False,  # Disable packing for stability
    callbacks=[StabilityMonitor(accelerator)],
)

# Verify configuration
if accelerator.is_main_process:
    print("🔍 Verification:")
    print(f"  Model max_position_embeddings: {model.config.max_position_embeddings}")
    print(f"  Trainer max_seq_length: {MAX_SEQ_LENGTH}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"  Total training steps: {len(dataset) // (BATCH_SIZE * GRAD_ACCUMULATION) * NUM_EPOCHS}")
    print(f"  Number of processes: {accelerator.num_processes}")

# Start training with the critical fix
if accelerator.is_main_process:
    print("🚀 Starting training with unsloth_train() fix...")
try:
    # CRITICAL: Use unsloth_train() instead of trainer.train()
    # This fixes the gradient accumulation bug that causes loss=0 and grad_norm=NaN
    trainer_stats = unsloth_train(trainer)
    
    if accelerator.is_main_process:
        print("✅ Training completed successfully!")
        print(f"📈 Final loss: {trainer_stats.training_loss:.4f}")
        
        # Save the model
        print("💾 Saving model...")
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        print(f"🎉 Training complete! Model saved to: {final_model_path}")
    
except Exception as e:
    if accelerator.is_main_process:
        print(f"❌ Training failed: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Check GPU memory usage - reduce BATCH_SIZE if OOM")
        print("2. Verify dataset format and content")
        print("3. Try reducing MAX_SEQ_LENGTH if still unstable")
        print("4. Check CUDA/PyTorch installation")
        print("5. Update Unsloth to latest version")
    raise

if accelerator.is_main_process:
    print("\n🏁 Script execution completed!")
