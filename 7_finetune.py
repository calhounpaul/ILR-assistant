#!/usr/bin/env python3
import os
import json
import torch
import multiprocessing
from functools import partial
import tqdm
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import datetime
import gc  # For garbage collection
import os
from accelerate import Accelerator

# Set tokenizers parallelism before import to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_PATH = "final_dataset.json"

now = datetime.datetime.now()

# Convert to timestamp (seconds since epoch)
timestamp = now.timestamp()

# Configuration
MODEL_NAME = "unsloth/Qwen3-14B-unsloth-bnb-4bit"  # Can be adjusted to a different size if needed
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
OUTPUT_DIR = "output/tune_" + str(timestamp)
MULTIPLIER=int((2**7))
LORA_R = MULTIPLIER
LORA_ALPHA = 2*MULTIPLIER
BATCH_SIZE = 1  # Per device batch size - reduce to fit in GPU memory
GRAD_ACCUMULATION = max(1,int(8/BATCH_SIZE))
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
WARMUP_RATIO = 0.03
NUM_WORKERS = multiprocessing.cpu_count() * 2  # Use all available CPU cores
MAX_TOKEN_LENGTH = 1536  # Maximum token length to include (discard outliers)
SAVE_LIMIT = 10
SAVE_STEPS = 2
LOGGING_STEPS = SAVE_STEPS
MAX_EXAMPLES=-1
GRAD_CHECKPOINTING="unsloth"
WEIGHT_DECAY=0.01
LORA_DROPOUT=0

TARGET_MODULES=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

# Initialize accelerator for multi-GPU training
# Important: Initialize accelerator before loading the model
accelerator = Accelerator()
is_main_process = accelerator.is_main_process
device_map = {"": accelerator.local_process_index}  # This maps the model to the correct device for each process

if is_main_process:
    print(f"Starting fine-tuning process with {MODEL_NAME}")
    print(f"Using {torch.cuda.device_count()} GPUs detected")
    print(f"Will use {NUM_WORKERS} CPU workers for preprocessing")
    print(f"Local process index: {accelerator.local_process_index}")
    print(f"Process will use device: cuda:{accelerator.local_process_index}")

# Load and prepare the model
# Make sure the model doesn't get sharded across GPUs yet
with accelerator.main_process_first():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto detect
        load_in_4bit=LOAD_IN_4BIT,
        device_map=device_map,  # This ensures each process loads the model on its assigned GPU
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=GRAD_CHECKPOINTING,  # Uses less VRAM
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

# Function to process a chunk of conversations in parallel
def process_conversation_chunk(chunk):
    processed_chunk = []
    
    for conversation_data in chunk:
        conversation_num, conversation = conversation_data
        
        if not conversation:  # Skip empty conversations
            continue
            
        # Collect all valid messages from the conversation
        messages = []
        for msg in conversation:
            if msg.get("content"):
                messages.append(msg)
        
        # Add the conversation to our processed data
        if messages:
            processed_chunk.append({"messages": messages})
    
    return processed_chunk

# Main preprocessing function with multiprocessing
def preprocess_conversations_parallel(conversations):
    if is_main_process:
        print(f"Starting parallel preprocessing with {NUM_WORKERS} workers")
    
    # Create indexed conversation pairs for easier distribution
    indexed_conversations = list(enumerate(conversations))
    
    # Split the data into chunks for each worker
    chunk_size = max(1, len(indexed_conversations) // NUM_WORKERS)
    chunks = [indexed_conversations[i:i+chunk_size] for i in range(0, len(indexed_conversations), chunk_size)]
    
    # Process in parallel
    process_func = partial(process_conversation_chunk)
    
    processed_conversations = []
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        # Process chunks in parallel and show progress
        results = list(tqdm.tqdm(
            pool.imap(process_func, chunks),
            total=len(chunks),
            desc="Processing conversation chunks",
            disable=not is_main_process
        ))
        
        # Combine results
        for chunk_result in results:
            processed_conversations.extend(chunk_result)
            
    if is_main_process:
        print(f"Processed {len(processed_conversations)} conversations")
    
    if MAX_EXAMPLES > 0:
        processed_conversations = processed_conversations[:MAX_EXAMPLES]
        
    # Apply tokenizer in the main process
    if is_main_process:
        print("Applying tokenizer to processed conversations...")
    processed_data = []
    for item in tqdm.tqdm(processed_conversations, disable=not is_main_process):
        formatted_chat = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        processed_data.append({"text": formatted_chat})
    
    return processed_data

# Load and process the dataset
if is_main_process:
    print("Loading and processing dataset...")
try:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # First, filter out outliers by pre-calculating token lengths
    if is_main_process:
        print("Pre-processing conversations to filter outliers...")
    filtered_conversations = []
    outliers = []
    
    # Process a small batch of conversations to estimate token lengths
    for i, conversation in enumerate(tqdm.tqdm(raw_data, 
                                             desc="Filtering outliers", 
                                             total=len(raw_data), 
                                             disable=not is_main_process)):
        if not conversation:  # Skip empty conversations
            continue
            
        # Extract just the message content for a rough token count estimate
        # This is much faster than full tokenization
        combined_text = ""
        for message in conversation:
            if message.get("content"):
                combined_text += message.get("content", "") + " "
        
        # Rough estimate: 1 token ≈ 4 characters for English text
        estimated_tokens = len(combined_text) // 4
        
        if estimated_tokens <= MAX_TOKEN_LENGTH:
            filtered_conversations.append(conversation)
        else:
            outliers.append((i, estimated_tokens))
    
    if is_main_process:
        print(f"Filtered out {len(outliers)} outliers from {len(raw_data)} conversations")
        print(f"Proceeding with {len(filtered_conversations)} conversations")
    
        if outliers:
            print(f"Examples of outliers (conversation index, estimated tokens):")
            for i, (idx, tokens) in enumerate(outliers[:5]):
                print(f"  #{idx}: ~{tokens} tokens")
            if len(outliers) > 5:
                print(f"  ... and {len(outliers) - 5} more")
    
    # Process the filtered data
    processed_data = preprocess_conversations_parallel(filtered_conversations)
    
    # Create a HF dataset
    dataset = Dataset.from_list(processed_data)
    if is_main_process:
        print(f"Dataset loaded with {len(dataset)} examples")
    
        # Show a sample
        if len(dataset) > 0:
            print("\nSample processed example:")
            print(dataset[0]["text"][:2500] + "...\n")
    
except Exception as e:
    if is_main_process:
        print(f"Error loading dataset: {e}")
    
    # Try alternative dataset loading if the format is different
    try:
        if is_main_process:
            print("Attempting alternative dataset loading...")
        with open("dataset.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # If dataset is a list of lists of conversations
        if isinstance(raw_data, list) and isinstance(raw_data[0], list):
            if is_main_process:
                print("Dataset appears to be list of conversation lists")
            all_conversations = []
            for conversation_list in raw_data:
                all_conversations.extend(conversation_list)
            processed_data = preprocess_conversations_parallel(all_conversations)
        else:
            if is_main_process:
                print("Dataset appears to be a direct list of conversations")
            processed_data = preprocess_conversations_parallel(raw_data)
            
        dataset = Dataset.from_list(processed_data)
        if is_main_process:
            print(f"Dataset loaded with alternative method: {len(dataset)} examples")
    except Exception as e2:
        if is_main_process:
            print(f"Both dataset loading methods failed: {e2}")
        raise

# Free up memory after preprocessing
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Calculate and display token length statistics if main process
if is_main_process:
    print("Calculating token length statistics...")
    token_lengths = []
    for example in tqdm.tqdm(dataset, desc="Analyzing token lengths"):
        # Tokenize the text (only getting the token ids, not the actual tokens)
        tokens = tokenizer(example["text"], return_tensors="pt").input_ids
        token_lengths.append(len(tokens[0]))

    avg_token_length = sum(token_lengths) / len(token_lengths)
    min_token_length = min(token_lengths)
    max_token_length = max(token_lengths)

    print(f"Token length statistics:")
    print(f"  Average: {avg_token_length:.2f} tokens")
    print(f"  Minimum: {min_token_length} tokens")
    print(f"  Maximum: {max_token_length} tokens")
    print(f"  Sequence length: {MAX_SEQ_LENGTH} tokens (configured maximum)")

    # Display histogram-like distribution
    length_ranges = [(0, 512), (513, 1024), (1025, 1536), (1537, 2048), (2049, float('inf'))]
    range_counts = {f"{start}-{end}": 0 for start, end in length_ranges}

    for length in token_lengths:
        for start, end in length_ranges:
            if start <= length <= end:
                range_counts[f"{start}-{end}"] += 1
                break

    print("\nToken length distribution:")
    for range_name, count in range_counts.items():
        percentage = (count / len(token_lengths)) * 100
        print(f"  {range_name} tokens: {count} examples ({percentage:.1f}%)")

    # Check for sequences that exceed the maximum length
    oversize_count = sum(1 for length in token_lengths if length > MAX_SEQ_LENGTH)
    if oversize_count > 0:
        oversize_percentage = (oversize_count / len(token_lengths)) * 100
        print(f"\nWARNING: {oversize_count} examples ({oversize_percentage:.1f}%) exceed the maximum sequence length!")
        print(f"These will be truncated during training, which may affect model performance.")

# Set up training arguments with distributed training parameters
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
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_8bit",
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    seed=3407,
    report_to="none",
    # Add distributed training parameters
    ddp_find_unused_parameters=False,
    dataloader_num_workers=NUM_WORKERS,
    # Specify these flags for DDP training with 4-bit quantized models
    ddp_bucket_cap_mb=50,
    gradient_checkpointing=True,
)

# Set up the trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    packing=False,  # Set to True if you have short sequences to speed up training
)

# Train
if is_main_process:
    print("Starting training...")
trainer.train()

# Save the model (only from the main process)
if is_main_process:
    print("Saving model...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

    # Also save merged model in 16-bit for inference
    print("Saving merged 16-bit model for inference...")
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DIR, "merged_model"), 
        tokenizer, 
        save_method="merged_16bit"
    )

    print("Training complete!")