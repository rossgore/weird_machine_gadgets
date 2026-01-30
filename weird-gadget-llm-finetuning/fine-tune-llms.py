"""
Multi-Model Ensemble Training and Agreement Analysis (Memory-Optimized for Mac)
Train 2 different small LLMs on weird machine gadgets dataset and compare where they agree/disagree.

Models:
1. google/flan-t5-small (77M params, encoder-decoder, instruction-tuned)
2. distilgpt2 (82M params, decoder-only causal LM, distilled from GPT-2)

This version is optimized for Apple Silicon Macs with limited memory by:
- Forcing CPU-only execution (disables MPS)
- Using only 2 smaller models instead of 3
- Explicit memory cleanup between training runs

Usage:
    python main.py --platform windows
    python main.py --platform unix

Architectural diversity: 1 seq2seq vs 1 causal LM
Training paradigm diversity: instruction-tuned vs general pre-training
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
)


# ============================================================================
# MEMORY OPTIMIZATION: Force CPU, disable MPS and CUDA
# ============================================================================
print("\n" + "=" * 80)
print("MEMORY OPTIMIZATION LAYER")
print("=" * 80)
print("Forcing CPU-only training to prevent out-of-memory errors...")

# Disable GPU backends
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_ALLOCATOR"] = "0"

# Override torch backend checks
original_cuda_is_available = torch.cuda.is_available
original_mps_is_available = torch.backends.mps.is_available if hasattr(torch.backends, 'mps') else lambda: False

torch.cuda.is_available = lambda: False
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False

print("✓ CUDA disabled")
print("✓ MPS disabled (Apple Silicon GPU)")
print("✓ All training will use CPU only")
print("=" * 80 + "\n")


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

DATA_FILE = "data/weird_machine_gadgets.jsonl"
OUTPUT_BASE_DIR = "checkpoints"

# Training hyperparameters (shared across models)
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Dataset sizing
TOTAL_EXAMPLES_TO_USE = 100
EVAL_SPLIT = 0.1

# Model configurations - 2 models for memory efficiency
MODELS = {
    "flan-t5-small": {
        "name": "google/flan-t5-small",
        "type": "seq2seq",
        "params": "77M",
        "description": "Encoder-decoder, instruction-tuned T5",
        "architecture": "Bidirectional encoder + autoregressive decoder",
    },
    "distilgpt2": {
        "name": "distilgpt2",
        "type": "causal",
        "params": "82M",
        "description": "Distilled GPT-2, fast and memory-efficient",
        "architecture": "Decoder-only, left-to-right attention",
    },
}


# ============================================================================
# PLATFORM-SPECIFIC SETUP
# ============================================================================

def setup_platform(platform: str):
    """Configure environment for Windows or Unix."""
    print("\n" + "=" * 80)
    print("PLATFORM SETUP")
    print("=" * 80)
    
    if platform == "windows":
        print("✓ Platform: Windows")
        print("  - Multiprocessing: Disabled to avoid spawn issues")
        max_processes = 0  # Disable multiprocessing on Windows
    else:
        print("✓ Platform: Unix (macOS/Linux)")
        print("  - Multiprocessing: Enabled")
        max_processes = 2
    
    print(f"✓ CPU cores available: {os.cpu_count()}")
    print(f"✓ Data loading processes: {max_processes}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ Device: CPU (forced)")
    
    return max_processes


# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

def free_memory():
    """Aggressively free memory between model training runs."""
    gc.collect()
    if original_cuda_is_available():
        torch.cuda.empty_cache()
    print("  ✓ Memory freed")


# ============================================================================
# DATA PREPARATION
# ============================================================================

def make_prompt(example):
    """Create prompt from instruction and input."""
    example["prompt"] = (
        f"Task: {example['instruction']}\n\n"
        f"Excerpt:\n{example['input']}\n\n"
        f"Answer:"
    )
    return example


def make_causal_prompt(example):
    """Create prompt for causal LM (includes output for training)."""
    example["text"] = (
        f"Task: {example['instruction']}\n\n"
        f"Excerpt:\n{example['input']}\n\n"
        f"Answer: {example['output']}"
    )
    return example


def load_and_prepare_data(max_processes: int):
    """Load JSONL and prepare train/val splits."""
    print("\n" + "=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)
    
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    
    print(f"✓ Found {DATA_FILE}")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    print(f"✓ Total lines: {line_count}")
    
    # Load dataset
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"✓ Loaded {len(dataset)} examples")
    
    # Add prompts
    if max_processes > 0:
        dataset = dataset.map(make_prompt, num_proc=max_processes)
        dataset = dataset.map(make_causal_prompt, num_proc=max_processes)
    else:
        dataset = dataset.map(make_prompt)
        dataset = dataset.map(make_causal_prompt)
    
    # Subsample
    dataset = dataset.shuffle(seed=42).select(
        range(min(TOTAL_EXAMPLES_TO_USE, len(dataset)))
    )
    print(f"✓ Subsampled to {len(dataset)} examples")
    
    # Split
    split = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    
    print(f"✓ Train: {len(train_ds)} | Validation: {len(val_ds)}")
    
    return train_ds, val_ds


# ============================================================================
# MODEL-SPECIFIC TRAINING
# ============================================================================

def train_seq2seq_model(model_key: str, model_config: Dict, train_ds, val_ds, max_processes: int):
    """Train a sequence-to-sequence model (FLAN-T5)."""
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL: {model_key.upper()}")
    print("=" * 80)
    
    model_name = model_config["name"]
    output_dir = os.path.join(OUTPUT_BASE_DIR, model_key)
    
    print(f"  Model: {model_name}")
    print(f"  Type: {model_config['type']}")
    print(f"  Params: {model_config['params']}")
    print(f"  Architecture: {model_config['architecture']}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Force CPU
    model.to("cpu")
    
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"  Parameters: {model.num_parameters():,}")
    
    # Tokenize
    def preprocess(examples):
        model_inputs = tokenizer(
            examples["prompt"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["output"],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding=False,
            )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs
    
    print("  Tokenizing datasets...")
    if max_processes > 0:
        tokenized_train = train_ds.map(
            preprocess,
            batched=True,
            remove_columns=train_ds.column_names,
            num_proc=max_processes,
        )
        tokenized_val = val_ds.map(
            preprocess,
            batched=True,
            remove_columns=val_ds.column_names,
            num_proc=max_processes,
        )
    else:
        tokenized_train = train_ds.map(
            preprocess,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        tokenized_val = val_ds.map(
            preprocess,
            batched=True,
            remove_columns=val_ds.column_names,
        )
    
    # Training args - CPU optimized
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        logging_steps=10,
        predict_with_generate=True,
        fp16=False,
        use_cpu=True,  # Explicitly force CPU
        dataloader_num_workers=0,
        report_to=[],
        seed=42,
        do_train=True,
        do_eval=True,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("  Training...")
    train_result = trainer.train()
    print(f"✓ Training complete! Loss: {train_result.training_loss:.4f}")
    
    # Save
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✓ Saved to: {final_model_path}")
    
    return final_model_path, tokenizer, model


def train_causal_model(model_key: str, model_config: Dict, train_ds, val_ds, max_processes: int):
    """Train a causal language model (DistilGPT2)."""
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL: {model_key.upper()}")
    print("=" * 80)
    
    model_name = model_config["name"]
    output_dir = os.path.join(OUTPUT_BASE_DIR, model_key)
    
    print(f"  Model: {model_name}")
    print(f"  Type: {model_config['type']} (causal LM)")
    print(f"  Params: {model_config['params']}")
    print(f"  Architecture: {model_config['architecture']}")
    
    # Load
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Force CPU
    model.to("cpu")
    
    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"  Parameters: {model.num_parameters():,}")
    
    # Tokenize
    def preprocess(examples):
        return tokenizer(
            examples["text"],
            max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )
    
    print("  Tokenizing datasets...")
    if max_processes > 0:
        tokenized_train = train_ds.map(
            preprocess,
            batched=True,
            remove_columns=train_ds.column_names,
            num_proc=max_processes,
        )
        tokenized_val = val_ds.map(
            preprocess,
            batched=True,
            remove_columns=val_ds.column_names,
            num_proc=max_processes,
        )
    else:
        tokenized_train = train_ds.map(
            preprocess,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        tokenized_val = val_ds.map(
            preprocess,
            batched=True,
            remove_columns=val_ds.column_names,
        )
    
    # Training args - CPU optimized
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        logging_steps=10,
        fp16=False,
        use_cpu=True,  # Explicitly force CPU
        dataloader_num_workers=0,
        report_to=[],
        seed=42,
    )
    
    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("  Training...")
    train_result = trainer.train()
    print(f"✓ Training complete! Loss: {train_result.training_loss:.4f}")
    
    # Save
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✓ Saved to: {final_model_path}")
    
    return final_model_path, tokenizer, model


# ============================================================================
# INFERENCE & COMPARISON
# ============================================================================

def generate_seq2seq(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate from seq2seq model."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    model.to("cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_causal(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate from causal LM (extract answer portion only)."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    model.to("cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],  # ← FIXED: pass tensor directly, not **unpacking
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (everything after "Answer:")
    if "Answer:" in full_text:
        answer = full_text.split("Answer:")[-1].strip()
    else:
        answer = full_text
    
    return answer


def check_format(output: str) -> Dict[str, bool]:
    """Check if output has correct format."""
    lower = output.lower()
    return {
        "gadget_type": "gadget_type:" in lower,
        "location": "location:" in lower,
        "explanation": "explanation:" in lower,
    }


def extract_gadget_type(prediction: str) -> str:
    """Extract gadget type from prediction."""
    lower = prediction.lower()
    if "gadget_type:" in lower:
        start = lower.index("gadget_type:") + len("gadget_type:")
        rest = prediction[start:].split(";")[0].split("\n")[0].strip()
        return rest
    return "UNKNOWN"

# Add this function after the extract_gadget_type function (around line 490)

def normalize_gadget_type(gadget_type: str) -> str:
    """
    Normalize gadget type for comparison by:
    1. Converting to lowercase
    2. Removing spaces, hyphens, and underscores
    3. Stripping common suffixes like 'gadget', 'gadgets'
    
    This allows 'Control-Flow gadget', 'Control-flow', and 'CONTROL_FLOW'
    to be treated as equivalent.
    """
    if not gadget_type or gadget_type == "UNKNOWN":
        return "UNKNOWN"
    
    # Convert to lowercase
    normalized = gadget_type.lower()
    
    # Remove common words
    normalized = normalized.replace(' gadget', '').replace(' gadgets', '')
    
    # Remove spaces, hyphens, underscores, slashes
    normalized = normalized.replace(' ', '').replace('-', '').replace('_', '').replace('/', '')
    
    # Remove 'and' connectors
    normalized = normalized.replace('and', '')
    
    return normalized.strip()

def compute_agreement(predictions: Dict[str, str]) -> Dict:
    """Compute inter-model agreement metrics with normalized comparison."""
    # Extract raw gadget types
    raw_gadget_types = {k: extract_gadget_type(v) for k, v in predictions.items()}
    
    # Normalize for comparison
    normalized_types = {k: normalize_gadget_type(v) for k, v in raw_gadget_types.items()}
    
    # Check agreement on normalized types
    unique_normalized = set(normalized_types.values())
    full_agreement = len(unique_normalized) == 1
    
    # For display, use raw types but note if they're semantically equivalent
    unique_raw_types = list(set(raw_gadget_types.values()))
    
    # Count occurrences of normalized types
    type_counts = Counter(normalized_types.values())
    majority_normalized, majority_count = type_counts.most_common(1)[0]
    
    # Find a representative raw type for the majority
    majority_type_raw = None
    for model_key, norm_type in normalized_types.items():
        if norm_type == majority_normalized:
            majority_type_raw = raw_gadget_types[model_key]
            break
    
    return {
        "full_agreement": full_agreement,
        "unique_types": unique_raw_types,  # Show original types for transparency
        "gadget_types": raw_gadget_types,  # Original types per model
        "normalized_types": normalized_types,  # Normalized versions (for debugging)
        "unique_normalized_types": list(unique_normalized),  # Normalized unique types
        "majority_type": majority_type_raw if majority_type_raw else majority_normalized,
        "majority_count": majority_count,
        "total_models": len(predictions),
    }


def run_ensemble_comparison(models_info: Dict, val_ds):
    """Run all models on validation set and analyze agreement."""
    print("\n" + "=" * 80)
    print("ENSEMBLE COMPARISON & AGREEMENT ANALYSIS")
    print("=" * 80)
    
    results = []
    
    for i, example in enumerate(val_ds):
        print(f"\n{'='*80}")
        print(f"VALIDATION EXAMPLE {i+1}/{len(val_ds)}")
        print(f"{'='*80}")
        
        instruction = example["instruction"]
        excerpt = example["input"]
        gold_output = example["output"]
        prompt = example["prompt"]
        
        print(f"\nINSTRUCTION: {instruction[:80]}...")
        print(f"EXCERPT: {excerpt[:100]}...")
        
        predictions = {}
        format_checks = {}
        
        # Get predictions from each model
        for model_key, info in models_info.items():
            print(f"\n  [{model_key}] Generating...")
            
            if info["type"] == "seq2seq":
                pred = generate_seq2seq(info["model"], info["tokenizer"], prompt)
            else:
                pred = generate_causal(info["model"], info["tokenizer"], prompt)
            
            predictions[model_key] = pred
            format_checks[model_key] = check_format(pred)
            
            print(f"  Output: {pred[:100]}...")
        
        # Compute agreement
        agreement = compute_agreement(predictions)
        
        print(f"\n{'─'*80}")
        print("AGREEMENT ANALYSIS:")
        print(f"{'─'*80}")
        print(f"  Full agreement: {'✓ YES' if agreement['full_agreement'] else '✗ NO'}")
        print(f"  Unique gadget types: {agreement['unique_types']}")
        print(f"  Majority type: {agreement['majority_type']} ({agreement['majority_count']}/{agreement['total_models']})")
        
        print(f"\n  Model-specific gadget types:")
        for model_key, gtype in agreement['gadget_types'].items():
            print(f"    - {model_key}: {gtype}")
        
        print(f"\n  Format checks:")
        for model_key, checks in format_checks.items():
            all_pass = all(checks.values())
            status = "✓" if all_pass else "✗"
            print(f"    {status} {model_key}: {checks}")
        
        # Store results
        results.append({
            "example_id": i,
            "instruction": instruction,
            "excerpt": excerpt,
            "gold_output": gold_output,
            "predictions": predictions,
            "format_checks": format_checks,
            "agreement": agreement,
        })
    
    return results


def save_comparison_report(results: List[Dict], output_path: str = "ensemble_report.json"):
    """Save detailed comparison report."""
    print("\n" + "=" * 80)
    print("SAVING COMPARISON REPORT")
    print("=" * 80)
    
    # Compute summary statistics
    total_examples = len(results)
    full_agreements = sum(1 for r in results if r["agreement"]["full_agreement"])
    
    # Model-specific format accuracy
    model_format_accuracy = {}
    for model_key in results[0]["format_checks"].keys():
        correct = sum(
            1 for r in results
            if all(r["format_checks"][model_key].values())
        )
        model_format_accuracy[model_key] = correct / total_examples
    
    summary = {
        "total_examples": total_examples,
        "full_agreements": full_agreements,
        "full_agreement_rate": full_agreements / total_examples,
        "disagreement_rate": 1 - (full_agreements / total_examples),
        "model_format_accuracy": model_format_accuracy,
    }
    
    report = {
        "summary": summary,
        "results": results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Report saved to: {output_path}")
    print(f"\nSUMMARY:")
    print(f"  Total examples: {total_examples}")
    print(f"  Full agreement: {full_agreements} ({summary['full_agreement_rate']*100:.1f}%)")
    print(f"  Disagreements: {total_examples - full_agreements} ({summary['disagreement_rate']*100:.1f}%)")
    print(f"\n  Format accuracy by model:")
    for model_key, accuracy in model_format_accuracy.items():
        print(f"    - {model_key}: {accuracy*100:.1f}%")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-model ensemble training and comparison (memory-optimized)")
    parser.add_argument(
        "--platform",
        type=str,
        choices=["windows", "unix"],
        required=True,
        help="Platform: 'windows' or 'unix' (macOS/Linux)",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        dest="skip_training",
        help="Skip training and load existing models (for testing comparison only)",
    )
    
    args = parser.parse_args()
    
    # Setup platform
    max_processes = setup_platform(args.platform)
    
    # Load data
    train_ds, val_ds = load_and_prepare_data(max_processes)
    
    # Train all models (or load if skip_training)
    models_info = {}
    
    for model_key, model_config in MODELS.items():
        if args.skip_training:
            print(f"\n⚠️  Skipping training for {model_key}, loading existing model...")
            model_path = os.path.join(OUTPUT_BASE_DIR, model_key, "final_model")
            
            if not Path(model_path).exists():
                print(f"✗ Model not found: {model_path}")
                print(f"  Please train first without --skip_training flag")
                sys.exit(1)
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if model_config["type"] == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            
            model.to("cpu")
        else:
            # Train model
            if model_config["type"] == "seq2seq":
                model_path, tokenizer, model = train_seq2seq_model(
                    model_key, model_config, train_ds, val_ds, max_processes
                )
            else:
                model_path, tokenizer, model = train_causal_model(
                    model_key, model_config, train_ds, val_ds, max_processes
                )
            
            # Free memory after training this model
            print(f"\n  Cleaning up memory after {model_key}...")
            free_memory()
        
        models_info[model_key] = {
            "config": model_config,
            "path": model_path,
            "tokenizer": tokenizer,
            "model": model,
            "type": model_config["type"],
        }
    
    # Run ensemble comparison
    results = run_ensemble_comparison(models_info, val_ds)
    
    # Save report
    save_comparison_report(results)
    
    print("\n" + "=" * 80)
    print("EXPLORATION SUGGESTIONS FOR STUDENTS")
    print("=" * 80)
    print("""
1. **Agreement Patterns**: Open ensemble_report.json and explore:
   - Which examples have full agreement vs disagreements?
   - Pattern: Do certain gadget types cause more disagreement?
   - Pattern: Does the seq2seq model (FLAN-T5) consistently disagree with causal model (DistilGPT2)?

2. **Architectural Differences**:
   - Does the seq2seq model (FLAN-T5) behave differently than causal model (DistilGPT2)?
   - Hypothesis: Instruction-tuned models should have better format adherence
   - Test: Compare format_accuracy in the summary

3. **Model-Specific Biases**:
   - Does one model consistently predict certain gadget types?
   - Example: Does FLAN-T5 favor "Read/Write" while DistilGPT2 favors "Control-Flow"?

4. **Hard vs Easy Examples**:
   - Examples with full agreement = "easy" (both models converge)
   - Examples with disagreement = "hard"
   - Analyze: What makes an excerpt "hard"? Length? Technical jargon? Ambiguity?

5. **Majority Voting Performance**:
   - For each example, compare majority vote vs gold output
   - With 2 models, this is a tie-breaker analysis

6. **Scale Up Experiments**:
   - Increase TOTAL_EXAMPLES_TO_USE to 200, then 500
   - Hypothesis: More data → better individual models → less disagreement?
   - Test: Track agreement_rate as dataset size increases

7. **Error Analysis by Gadget Type**:
   - Group results by gold gadget type
   - Which types have highest agreement?
   - Which types cause most confusion?

8. **Qualitative Analysis**:
   - Pick a disagreement example
   - Read both predictions side-by-side
   - Which is closest to gold? Why?
   - Are errors factual, formatting, or conceptual?

QUICK COMMANDS:

# Re-run comparison without retraining:
python main.py --platform unix --skip_training

# Increase dataset size (edit script first):
TOTAL_EXAMPLES_TO_USE = 200  # or 500

# Analyze report:
python -c "import json; r=json.load(open('ensemble_report.json')); print(r['summary'])"

MEMORY NOTES:
- This version uses only 2 models (FLAN-T5-small + DistilGPT2) instead of 3
- Both models are ~80M params, memory-efficient
- Forced CPU-only to avoid MPS out-of-memory errors
- Total training time on CPU: ~20-25 minutes for 100 examples
""")


if __name__ == "__main__":
    main()
