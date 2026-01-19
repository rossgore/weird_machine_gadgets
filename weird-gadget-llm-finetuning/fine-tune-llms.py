"""
Fine-tune google/flan-t5-small on IEC 61850 + DNP3 weird machine gadgets dataset.
Dataset: data/weird_machine_gadgets.jsonl

OPTIMIZED FOR CPU-ONLY LAPTOPS (no GPU required)

Steps:
1. Load dataset from JSONL
2. Explore and create prompts
3. Subsample for a quick run
4. Tokenize
5. Fine-tune with Seq2SeqTrainer
6. Test on held-out examples
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import torch

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR CPU
# ============================================================================

DATA_FILE = "data/weird_machine_gadgets.jsonl"
OUTPUT_DIR = "checkpoints/flan-t5-small-gadgets"
MODEL_NAME = "google/flan-t5-small"

# Training hyperparameters - CPU-optimized
# (small batch sizes, fewer accumulation steps, more gradual learning)
TRAIN_BATCH_SIZE = 1              # CPU can only handle very small batches
EVAL_BATCH_SIZE = 1               # Even smaller for evaluation
GRADIENT_ACCUMULATION_STEPS = 4   # Simulate batch_size=4 by accumulating gradients
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Dataset sizing
TOTAL_EXAMPLES_TO_USE = 100       # Start with fewer examples on CPU
EVAL_SPLIT = 0.1                  # 10% validation, 90% train

# CPU optimization flags
USE_CPU = True                    # Force CPU usage
MAX_PROCESSES = 2                 # Limit parallel processing on CPU

# ============================================================================
# UTILITY: Check device and print system info
# ============================================================================

def check_device():
    """Check if GPU is available and print device info."""
    print("\n" + "=" * 80)
    print("DEVICE INFORMATION")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print(f" GPU detected: {torch.cuda.get_device_name(0)}")
        print("   However, script is configured to use CPU for consistency.")
    else:
        print("✓ Running on CPU (as expected)")
    
    print(f"✓ CPU cores available: {os.cpu_count()}")
    print(f"✓ Using {MAX_PROCESSES} processes for data loading")
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Force CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    return "cpu"

# ============================================================================
# STEP 1: Check file exists
# ============================================================================

print("=" * 80)
print("STEP 1: Checking dataset file...")
print("=" * 80)

if not Path(DATA_FILE).exists():
    raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")

print(f"✓ Found {DATA_FILE}")
with open(DATA_FILE, "r") as f:
    line_count = sum(1 for _ in f)
print(f"✓ Total lines in JSONL: {line_count}")

# ============================================================================
# STEP 2: Check device
# ============================================================================
"""Check for GPU."""
device = check_device()

# ============================================================================
# STEP 3: Load and explore dataset
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Loading dataset from JSONL...")
print("=" * 80)

print("  Loading...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
print(f"✓ Loaded {len(dataset)} examples")

print("\n Example 0:")
example_0 = dataset[0]
for key in ["instruction", "input", "output"]:
    if key in example_0:
        val = example_0[key]
        if len(str(val)) > 100:
            print(f"  {key}: {str(val)[:100]}...")
        else:
            print(f"  {key}: {val}")

# ============================================================================
# STEP 4: Create prompts and subsample
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Creating prompts and subsampling...")
print("=" * 80)

def make_prompt(example):
    """Create a prompt from instruction and input."""
    example["prompt"] = (
        f"Task: {example['instruction']}\n\n"
        f"Excerpt:\n{example['input']}\n\n"
        f"Answer:"
    )
    return example

# Map over entire dataset
print("  Creating prompts...")
dataset = dataset.map(make_prompt, num_proc=MAX_PROCESSES)
print(f"✓ Added prompt field to all {len(dataset)} examples")

# Subsample for faster iteration on CPU
dataset = dataset.shuffle(seed=42).select(range(min(TOTAL_EXAMPLES_TO_USE, len(dataset))))
print(f"✓ Subsampled to {len(dataset)} examples for training")

print("\n Example prompt:")
print(dataset[0]["prompt"])
print("\n Example output:")
print(dataset[0]["output"])

# ============================================================================
# STEP 5: Split into train/validation
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Splitting into train/validation...")
print("=" * 80)

split = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
train_ds = split["train"]
val_ds = split["test"]

print(f"✓ Train: {len(train_ds)} examples")
print(f"✓ Validation: {len(val_ds)} examples")

# ============================================================================
# STEP 6: Load model and tokenizer
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Loading model and tokenizer...")
print("=" * 80)

print(f"  Model: {MODEL_NAME}")
print(f"  Max input length: {MAX_INPUT_LENGTH}")
print(f"  Max target length: {MAX_TARGET_LENGTH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"✓ Model loaded: {type(model).__name__}")
print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
print(f"  Model parameters: {model.num_parameters():,}")

# ============================================================================
# STEP 7: Tokenize datasets
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Tokenizing datasets...")
print("=" * 80)

def preprocess(example):
    """Tokenize input (prompt) and output (target)."""
    # Tokenize input (prompt)
    model_inputs = tokenizer(
        example["prompt"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,  # We'll pad in the collate_fn
    )
    
    # Tokenize target (output) as labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )["input_ids"]
    
    model_inputs["labels"] = labels
    return model_inputs

print("  Tokenizing train dataset...")
tokenized_train = train_ds.map(
    preprocess,
    batched=True,
    remove_columns=train_ds.column_names,
    num_proc=MAX_PROCESSES,
    desc="Tokenizing train",
)

print("  Tokenizing validation dataset...")
tokenized_val = val_ds.map(
    preprocess,
    batched=True,
    remove_columns=val_ds.column_names,
    num_proc=MAX_PROCESSES,
    desc="Tokenizing validation",
)

print(f"✓ Tokenized train: {len(tokenized_train)} examples")
print(f"✓ Tokenized validation: {len(tokenized_val)} examples")

# ============================================================================
# STEP 8: Define training arguments (CPU-optimized)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Setting up training arguments (CPU-optimized)...")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    # Batch sizes
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    # Learning
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=10,
    weight_decay=0.01,
    # Evaluation (using eval_steps instead of evaluation_strategy="epoch")
    eval_strategy="steps",          # Changed from "epoch"
    eval_steps=50,                  # Evaluate every N steps
    # Saving
    save_strategy="steps",          # Changed from "epoch"
    save_steps=50,                  # Save every N steps
    save_total_limit=2,
    # Logging
    logging_strategy="steps",       # Explicit logging strategy
    logging_steps=2,
    # Generation and device
    predict_with_generate=True,
    fp16=False,
    dataloader_num_workers=0,
    report_to=[],
    seed=42,
    optim="adamw_torch",
    max_grad_norm=1.0,
    do_train=True,                  # Explicitly enable training
    do_eval=True,                   # Explicitly enable evaluation
)

print(f"✓ Training arguments configured:")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Train batch size: {TRAIN_BATCH_SIZE}")
print(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Eval batch size: {EVAL_BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Device: CPU (no GPU)")

# ============================================================================
# STEP 9: Create data collator
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Creating data collator...")
print("=" * 80)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
)

print(f"✓ Data collator created (pads to multiples of 8 for CPU efficiency)")

# ============================================================================
# STEP 10: Create Trainer and fine-tune
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Creating Trainer and fine-tuning...")
print("=" * 80)
print(" This will take a few minutes. Please wait...\n")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("  Starting training...")
train_result = trainer.train()

print(f"\n✓ Training complete!")
print(f"  Final train loss: {train_result.training_loss:.4f}")

# ============================================================================
# STEP 11: Save the model
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: Saving model...")
print("=" * 80)

model_save_path = os.path.join(OUTPUT_DIR, "final_model")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"✓ Model saved to: {model_save_path}")

# ============================================================================
# STEP 12: Evaluate on held-out examples
# ============================================================================

print("\n" + "=" * 80)
print("STEP 12: Testing on held-out validation examples...")
print("=" * 80)

def generate_answer(instruction, excerpt, max_new_tokens=256):
    """Generate an answer using the fine-tuned model."""
    prompt = f"Task: {instruction}\n\nExcerpt:\n{excerpt}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Ensure model and inputs are on CPU
    model.to("cpu")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Show results on first 3 validation examples
num_examples_to_show = min(3, len(val_ds))
print(f"\n Sample predictions on {num_examples_to_show} validation examples:\n")

for i in range(num_examples_to_show):
    example = val_ds[i]
    instruction = example["instruction"]
    excerpt = example["input"]
    gold_output = example["output"]
    
    print(f"\n  ⏳ Generating prediction {i+1}/{num_examples_to_show}...")
    model_output = generate_answer(instruction, excerpt)
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i+1}:")
    print(f"{'='*80}")
    
    print(f"\nINSTRUCTION:")
    print(f"  {instruction[:100]}...")
    
    print(f"\nEXCERPT:")
    if len(excerpt) > 200:
        print(f"  {excerpt[:200]}...")
    else:
        print(f"  {excerpt}")
    
    print(f"\nGOLD OUTPUT:")
    print(f"  {gold_output}")
    
    print(f"\nMODEL OUTPUT:")
    print(f"  {model_output}")
    
    # Simple check: does model output contain key phrases?
    has_gadget_type = "gadget_type:" in model_output.lower()
    has_location = "location:" in model_output.lower()
    has_explanation = "explanation:" in model_output.lower()
    
    format_check = "✓" if (has_gadget_type and has_location and has_explanation) else "✗"
    print(f"\nFORMAT CHECK: {format_check}")
    print(f"  - gadget_type: {'✓' if has_gadget_type else '✗'}")
    print(f"  - location: {'✓' if has_location else '✗'}")
    print(f"  - explanation: {'✓' if has_explanation else '✗'}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

estimated_time_single_epoch = (len(train_ds) / (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * 30  # ~30 sec per batch on CPU

print(f"""
Fine-tuning complete!

PERFORMANCE NOTES:
  - Estimated time for 1 epoch: {estimated_time_single_epoch/60:.1f} minutes
  - CPU-optimized settings:
    - Batch size: {TRAIN_BATCH_SIZE} (per device)
    - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps
    - No mixed precision (FP16)
    - Greedy decoding (no beam search)
    - No GPU usage

NEXT STEPS:
1. Review predictions above and check model output quality.
2. Gradual scaling:
   - First: Try TOTAL_EXAMPLES_TO_USE = 200 (if step 1 succeeds)
   - Then: Try TOTAL_EXAMPLES_TO_USE = 500
   - Finally: Try TOTAL_EXAMPLES_TO_USE = 1000+ (may take 30+ min per epoch)
3. Experiment with hyperparameters:
   - LEARNING_RATE (try 1e-4, 1e-5)
   - NUM_EPOCHS (try 5, 10)
4. For faster iteration, reduce MAX_INPUT_LENGTH or MAX_TARGET_LENGTH

MODEL SAVED AT: {model_save_path}

TO LOAD THE MODEL LATER:
  from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  tokenizer = AutoTokenizer.from_pretrained('{model_save_path}')
  model = AutoModelForSeq2SeqLM.from_pretrained('{model_save_path}')
  
  # Generate predictions:
  prompt = "Task: ... Excerpt: ... Answer:"
  inputs = tokenizer(prompt, return_tensors="pt")
  outputs = model.generate(**inputs, max_new_tokens=256)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""")

print("=" * 80)
