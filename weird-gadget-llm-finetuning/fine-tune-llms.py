#!/usr/bin/env python3
"""
Multi-Model Ensemble Training for Weird Machine Gadget Classification
Universal version - works on various hardware configurations
"""

import os
import sys
import json
import gc
import argparse
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import re

import torch
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
from datasets import Dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset sizing
TOTAL_EXAMPLES_TO_USE = 100
EVAL_SPLIT = 0.1

# Training hyperparameters
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Model definitions
MODELS = {
    "flan-t5-small": {
        "name": "google/flan-t5-small",
        "type": "seq2seq",
        "params": "77M",
        "description": "Encoder-decoder, instruction-tuned",
        "architecture": "Bidirectional encoder + autoregressive decoder",
    },
    "distilgpt2": {
        "name": "distilgpt2",
        "type": "causal",
        "params": "82M",
        "description": "Decoder-only, distilled from GPT-2",
        "architecture": "Decoder-only, left-to-right attention",
    },
}

# ============================================================================
# DEVICE SETUP
# ============================================================================

def setup_device(force_cpu: bool = False) -> Tuple[torch.device, str]:
    """
    Auto-detect best available device for training.

    Priority:
    1. CUDA GPU (NVIDIA)
    2. MPS (Apple Silicon GPU) - if not forcing CPU
    3. CPU (fallback)

    Args:
        force_cpu: If True, use CPU even if GPU is available

    Returns:
        (device, device_name)
    """
    if force_cpu:
        print("\n" + "="*80)
        print("DEVICE SETUP (CPU FORCED)")
        print("="*80)
        print("  Force CPU mode enabled")
        print("  All training will use CPU only")
        print("="*80 + "\n")
        return torch.device("cpu"), "CPU (forced)"

    print("\n" + "="*80)
    print("DEVICE SETUP (AUTO-DETECT)")
    print("="*80)

    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        print(f"✓ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("  Using GPU for training")
    # Check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            device_name = "MPS (Apple Silicon GPU)"
            print("✓ Apple Silicon GPU detected (MPS)")
            print("  Using MPS for training")
            print("  Note: If you encounter memory errors, run with --force_cpu")
        except Exception as e:
            print(f"⚠ MPS available but failed to initialize: {e}")
            print("  Falling back to CPU")
            device = torch.device("cpu")
            device_name = "CPU (MPS failed)"
    # Fallback to CPU
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        print("✓ No GPU detected")
        print("  Using CPU for training")
        print("  Training will be slower but will work on any hardware")

    print("="*80 + "\n")
    return device, device_name


def setup_platform(platform_arg: str) -> Dict[str, any]:
    """Configure platform-specific settings."""
    print("="*80)
    print("PLATFORM SETUP")
    print("="*80)

    system = platform.system()
    print(f"✓ Detected OS: {system}")
    print(f"✓ Python version: {sys.version.split()[0]}")
    print(f"✓ PyTorch version: {torch.__version__}")

    # Determine multiprocessing settings
    if platform_arg == "windows" or system == "Windows":
        print("✓ Platform: Windows")
        print("  - Multiprocessing: Disabled (to avoid spawn issues)")
        dataloader_num_workers = 0
        use_multiprocessing = False
    else:
        print("✓ Platform: Unix (macOS/Linux)")
        print("  - Multiprocessing: Enabled")
        cpu_count = os.cpu_count() or 1
        print(f"✓ CPU cores available: {cpu_count}")
        dataloader_num_workers = min(2, cpu_count)
        print(f"✓ Data loading processes: {dataloader_num_workers}")
        use_multiprocessing = True

    print("="*80 + "\n")

    return {
        "dataloader_num_workers": dataloader_num_workers,
        "use_multiprocessing": use_multiprocessing,
        "system": system,
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(data_path: str = "data/weird_machine_gadgets.jsonl") -> Tuple[Dataset, Dataset]:
    """Load and prepare training/validation data."""
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"✓ Found {data_path}")

    # Load JSONL
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"✓ Total lines: {len(examples)}")

    # Subsample
    if len(examples) > TOTAL_EXAMPLES_TO_USE:
        examples = examples[:TOTAL_EXAMPLES_TO_USE]
        print(f"✓ Subsampled to {len(examples)} examples")
    else:
        print(f"✓ Using all {len(examples)} examples")

    # Add prompts
    for ex in examples:
        instruction = ex.get("instruction", "")
        input_text = ex.get("input", "")
        output_text = ex.get("output", "")

        # Seq2seq prompt (for T5)
        ex["seq2seq_prompt"] = f"Task: {instruction}\n\nExcerpt:\n{input_text}\n\nAnswer:"

        # Causal prompt (for GPT-2)
        ex["causal_prompt"] = f"Task: {instruction}\n\nExcerpt:\n{input_text}\n\nAnswer: {output_text}"

    # Split train/val
    split_idx = int(len(examples) * (1 - EVAL_SPLIT))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"✓ Train: {len(train_examples)} | Validation: {len(val_examples)}")
    print("="*80 + "\n")

    # Convert to HF Dataset
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    return train_dataset, val_dataset


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_seq2seq_model(
    model_key: str,
    model_config: Dict,
    train_dataset: Dataset,
    val_dataset: Dataset,
    platform_config: Dict,
    device: torch.device,
) -> None:
    """Train seq2seq model (FLAN-T5)."""
    print("="*80)
    print(f"TRAINING MODEL: {model_key.upper()}")
    print("="*80)
    print(f"  Model: {model_config['name']}")
    print(f"  Type: {model_config['type']}")
    print(f"  Params: {model_config['params']}")
    print(f"  Architecture: {model_config['architecture']}")

    # Load model and tokenizer
    model_name = model_config["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"✓ Model loaded: {model.__class__.__name__}")
    print(f"  Parameters: {model.num_parameters():,}")

    # Tokenize
    def tokenize_seq2seq(examples):
        model_inputs = tokenizer(
            examples["seq2seq_prompt"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            examples["output"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("  Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        tokenize_seq2seq,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=platform_config["dataloader_num_workers"] if platform_config["use_multiprocessing"] else 1,
    )
    val_tokenized = val_dataset.map(
        tokenize_seq2seq,
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=platform_config["dataloader_num_workers"] if platform_config["use_multiprocessing"] else 1,
    )

    # Training args
    output_dir = f"checkpoints/{model_key}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=max(1, len(train_tokenized) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) // 4),
        save_strategy="no",
        eval_strategy="no",
        predict_with_generate=False,
        dataloader_num_workers=platform_config["dataloader_num_workers"],
        remove_unused_columns=True,
        no_cuda=(device.type != "cuda"),  # Disable CUDA if not using CUDA device
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    # Train
    print("  Training...")
    trainer.train()

    # Save
    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"✓ Training complete! Loss: {trainer.state.log_history[-1].get('train_loss', 'N/A')}")
    print(f"✓ Saved to: {final_model_dir}")
    print("="*80 + "\n")


def train_causal_model(
    model_key: str,
    model_config: Dict,
    train_dataset: Dataset,
    val_dataset: Dataset,
    platform_config: Dict,
    device: torch.device,
) -> None:
    """Train causal LM model (DistilGPT2)."""
    print("="*80)
    print(f"TRAINING MODEL: {model_key.upper()}")
    print("="*80)
    print(f"  Model: {model_config['name']}")
    print(f"  Type: {model_config['type']}")
    print(f"  Params: {model_config['params']}")
    print(f"  Architecture: {model_config['architecture']}")

    # Load model and tokenizer
    model_name = model_config["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"✓ Model loaded: {model.__class__.__name__}")
    print(f"  Parameters: {model.num_parameters():,}")

    # Tokenize
    def tokenize_causal(examples):
        return tokenizer(
            examples["causal_prompt"],
            max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

    print("  Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        tokenize_causal,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=platform_config["dataloader_num_workers"] if platform_config["use_multiprocessing"] else 1,
    )
    val_tokenized = val_dataset.map(
        tokenize_causal,
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=platform_config["dataloader_num_workers"] if platform_config["use_multiprocessing"] else 1,
    )

    # Training args
    output_dir = f"checkpoints/{model_key}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=max(1, len(train_tokenized) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) // 4),
        save_strategy="no",
        eval_strategy="no",
        dataloader_num_workers=platform_config["dataloader_num_workers"],
        remove_unused_columns=True,
        no_cuda=(device.type != "cuda"),
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    # Train
    print("  Training...")
    trainer.train()

    # Save
    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"✓ Training complete! Loss: {trainer.state.log_history[-1].get('train_loss', 'N/A')}")
    print(f"✓ Saved to: {final_model_dir}")
    print("="*80 + "\n")


def cleanup_memory(model_name: str) -> None:
    """Free memory after training a model."""
    print(f"  Cleaning up memory after {model_name}...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  ✓ Memory freed\n")


# ============================================================================
# ENSEMBLE COMPARISON
# ============================================================================

def normalize_gadget_type(gadget_type: str) -> str:
    """Normalize gadget type for comparison."""
    if not gadget_type:
        return ""

    normalized = gadget_type.lower()
    normalized = normalized.replace("gadget", "").replace("gadgets", "")
    normalized = re.sub(r"[\s\-_/]+", "", normalized)
    normalized = normalized.replace("and", "")

    return normalized.strip()


def extract_gadget_type(text: str) -> str:
    """Extract gadget_type from model output."""
    match = re.search(r"gadget_type:\s*([^;\n]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def check_output_format(text: str) -> Dict[str, bool]:
    """Check if output contains required fields."""
    return {
        "gadget_type": bool(re.search(r"gadget_type:", text, re.IGNORECASE)),
        "location": bool(re.search(r"location:", text, re.IGNORECASE)),
        "explanation": bool(re.search(r"explanation:", text, re.IGNORECASE)),
    }


def generate_prediction(model, tokenizer, prompt: str, model_type: str, device: torch.device, max_new_tokens: int = 256) -> str:
    """Generate prediction from a model."""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if model_type == "seq2seq":
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        else:  # causal
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction


def compute_agreement(predictions: Dict[str, str]) -> Dict:
    """Compute agreement metrics across model predictions."""
    # Extract gadget types
    gadget_types = {}
    normalized_types = {}

    for model_key, pred_text in predictions.items():
        raw_type = extract_gadget_type(pred_text)
        gadget_types[model_key] = raw_type
        normalized_types[model_key] = normalize_gadget_type(raw_type)

    # Check agreement
    unique_normalized = list(set(normalized_types.values()))
    full_agreement = len(unique_normalized) == 1 and unique_normalized[0] != ""

    # Majority type
    norm_counts = Counter(normalized_types.values())
    if norm_counts:
        majority_norm, majority_count = norm_counts.most_common(1)[0]
        # Find a representative raw label for this normalized type
        majority_type = next(
            (raw for model_key, raw in gadget_types.items() 
             if normalize_gadget_type(raw) == majority_norm),
            majority_norm
        )
    else:
        majority_type = ""
        majority_count = 0

    return {
        "full_agreement": full_agreement,
        "gadget_types": gadget_types,
        "unique_types": list(set(gadget_types.values())),
        "normalized_types": normalized_types,
        "unique_normalized_types": unique_normalized,
        "majority_type": majority_type,
        "majority_count": majority_count,
        "total_models": len(predictions),
    }


def run_ensemble_comparison(
    val_dataset: Dataset,
    models_info: Dict,
    device: torch.device,
    use_judge: bool = False,
) -> Dict:
    """Run ensemble comparison on validation set."""
    print("="*80)
    print("ENSEMBLE COMPARISON & AGREEMENT ANALYSIS (STRING-LEVEL)")
    print("="*80 + "\n")

    # Load all trained models
    loaded_models = {}
    for model_key, model_config in MODELS.items():
        checkpoint_dir = f"checkpoints/{model_key}/final_model"
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        print(f"Loading {model_key}...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        if model_config["type"] == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        model.to(device)
        model.eval()

        loaded_models[model_key] = {
            "model": model,
            "tokenizer": tokenizer,
            "type": model_config["type"],
        }

    print(f"✓ Loaded {len(loaded_models)} models\n")

    # Run comparison
    results = []
    for idx, example in enumerate(val_dataset):
        print("="*80)
        print(f"VALIDATION EXAMPLE {idx+1}/{len(val_dataset)}")
        print("="*80 + "\n")

        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        gold_output = example.get("output", "")

        print(f"INSTRUCTION: {instruction[:80]}...")
        print(f"EXCERPT: {input_text[:80]}...\n")

        # Generate predictions
        predictions = {}
        format_checks = {}

        for model_key, model_info in loaded_models.items():
            print(f"  [{model_key}] Generating...")

            if model_info["type"] == "seq2seq":
                prompt = example["seq2seq_prompt"]
            else:
                # For causal models, use only the instruction part (not the answer)
                prompt = f"Task: {instruction}\n\nExcerpt:\n{input_text}\n\nAnswer:"

            pred = generate_prediction(
                model_info["model"],
                model_info["tokenizer"],
                prompt,
                model_info["type"],
                device,
            )

            predictions[model_key] = pred
            format_checks[model_key] = check_output_format(pred)

            print(f"  Output: {pred[:80]}...\n")

        # Compute agreement
        agreement = compute_agreement(predictions)

        # Display analysis
        print("─"*80)
        print("AGREEMENT ANALYSIS (STRING-LEVEL):")
        print("─"*80)
        print(f"  Full agreement (normalized): {'✓ YES' if agreement['full_agreement'] else '✗ NO'}")
        print(f"  Unique raw gadget types: {agreement['unique_types']}")
        print(f"  Unique normalized types: {agreement['unique_normalized_types']}")
        print(f"  Majority type: {agreement['majority_type']} ({agreement['majority_count']}/{agreement['total_models']})")
        print(f"\n  Model-specific gadget types:")
        for model_key, gtype in agreement['gadget_types'].items():
            print(f"    - {model_key}: {gtype}")
        print(f"\n  Format checks:")
        for model_key, checks in format_checks.items():
            status = "✓" if all(checks.values()) else "✗"
            print(f"    {status} {model_key}: {checks}")
        print("\n")

        results.append({
            "example_id": idx,
            "instruction": instruction,
            "excerpt": input_text,
            "gold_output": gold_output,
            "predictions": predictions,
            "format_checks": format_checks,
            "agreement": agreement,
        })

    # Compute summary statistics
    total = len(results)
    full_agreements = sum(1 for r in results if r["agreement"]["full_agreement"])

    format_accuracy = {}
    for model_key in MODELS.keys():
        correct_format = sum(
            1 for r in results
            if all(r["format_checks"][model_key].values())
        )
        format_accuracy[model_key] = correct_format / total if total > 0 else 0

    summary = {
        "total_examples": total,
        "full_agreements": full_agreements,
        "full_agreement_rate": full_agreements / total if total > 0 else 0,
        "disagreement_rate": (total - full_agreements) / total if total > 0 else 0,
        "model_format_accuracy": format_accuracy,
        "llm_judge_used": use_judge,
    }

    # Optionally run LLM judge (placeholder for now)
    if use_judge:
        print("\n" + "="*80)
        print("SEMANTIC JUDGMENT PHASE (LIGHTWEIGHT LLM)")
        print("="*80)
        disagreements = [r for r in results if not r["agreement"]["full_agreement"]]
        print(f"  String-level disagreements: {len(disagreements)}/{total}\n")

        if len(disagreements) > 0:
            print("  Loading Phi-2 reasoning judge...\n")
            from transformers import AutoModelForCausalLM as CausalLM

            judge = LightweightSemanticJudge()

            print(f"  Judging {len(disagreements)} disagreements...")
            for i, result in enumerate(disagreements):
                print(f"  [{i+1}/{len(disagreements)}] Judging example {result['example_id']}...", end=" ")

                model_keys = list(result["predictions"].keys())
                pred_a = result["predictions"][model_keys[0]]
                pred_b = result["predictions"][model_keys[1]]
                gold = result["gold_output"]

                judgment = judge.judge(pred_a, pred_b, gold)
                result["agreement"]["llm_judgment"] = judgment

                # Update semantic agreement
                if judgment["verdict"] in ["FULL_AGREEMENT", "PARTIAL_AGREEMENT"]:
                    result["agreement"]["semantic_agreement"] = True
                else:
                    result["agreement"]["semantic_agreement"] = False

                print(judgment["verdict"])

            # Update summary with semantic agreement stats
            semantic_agreements = sum(
                1 for r in results 
                if r["agreement"].get("semantic_agreement", r["agreement"]["full_agreement"])
            )
            summary["semantic_agreements"] = semantic_agreements
            summary["semantic_agreement_rate"] = semantic_agreements / total if total > 0 else 0

            print("\n" + "="*80)
            print("AGREEMENT SUMMARY (WITH LLM JUDGE)")
            print("="*80)
            print(f"  Total examples: {total}")
            print(f"  String-level agreement: {full_agreements} ({summary['full_agreement_rate']*100:.1f}%)")
            print(f"  Semantic agreement (with LLM): {semantic_agreements} ({summary['semantic_agreement_rate']*100:.1f}%)")
            if semantic_agreements > full_agreements:
                print(f"  Improvement: +{semantic_agreements - full_agreements} examples")
            print("="*80 + "\n")

    return {"summary": summary, "results": results}


# ============================================================================
# LIGHTWEIGHT REASONING LLM JUDGE (Phi-2, CPU-friendly)
# ============================================================================

class LightweightSemanticJudge:
    """
    Lightweight reasoning LLM for judging semantic agreement between model predictions.
    Uses microsoft/phi-2 (~2.7B params) on available device without quantization.
    """

    def __init__(self, model_name: str = "microsoft/phi-2", device: Optional[torch.device] = None):
        print("=" * 80)
        print("LOADING LIGHTWEIGHT REASONING JUDGE")
        print("=" * 80)
        print(f"  Model: {model_name}")
        print(f"  Model size: ~2.7B parameters")
        print(f"  Quantization: None")

        if device is None:
            # Auto-detect device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"  Device: CUDA GPU")
            else:
                device = torch.device("cpu")
                print(f"  Device: CPU")
                print(f"  Expected memory: ~5-6 GB")

        self.device = device

        # Load Phi-2 with appropriate dtype
        if device.type == "cuda":
            # Use float16 on GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            # Use float32 on CPU (float16 not fully supported)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print("✓ Reasoning judge loaded successfully\n")

    def create_judge_prompt(self, pred_a: str, pred_b: str, gold: str) -> str:
        """Create a structured prompt for semantic agreement judgment."""
        return f"""<|system|>
You are an expert evaluator judging whether two model predictions are semantically equivalent.
Respond with ONLY ONE WORD from: FULL_AGREEMENT, PARTIAL_AGREEMENT, DISAGREEMENT, or INVALID.
Then on a new line, provide a brief 1-sentence explanation.
<|end|>
<|user|>
Gold standard answer:
{gold}

Model A prediction:
{pred_a}

Model B prediction:
{pred_b}

Question: Are Model A and Model B semantically equivalent?
<|end|>
<|assistant|>
"""

    def judge(self, pred_a: str, pred_b: str, gold: str) -> Dict[str, str]:
        """
        Judge semantic agreement between two predictions.

        Returns:
            {"verdict": str, "explanation": str}
        """
        prompt = self.create_judge_prompt(pred_a, pred_b, gold)

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract verdict and explanation
        # Response format: "<|assistant|>\nVERDICT\nExplanation..."
        response_parts = response.split("<|assistant|>")[-1].strip().split("\n", 1)

        verdict_raw = response_parts[0].strip().upper()
        explanation = response_parts[1].strip() if len(response_parts) > 1 else ""

        # Parse verdict
        if "FULL_AGREEMENT" in verdict_raw or "FULL AGREEMENT" in verdict_raw:
            verdict = "FULL_AGREEMENT"
        elif "PARTIAL_AGREEMENT" in verdict_raw or "PARTIAL AGREEMENT" in verdict_raw:
            verdict = "PARTIAL_AGREEMENT"
        elif "INVALID" in verdict_raw:
            verdict = "INVALID"
        else:
            verdict = "DISAGREEMENT"

        return {
            "verdict": verdict,
            "explanation": explanation if explanation else "No explanation provided.",
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-model ensemble training")
    parser.add_argument("--platform", choices=["unix", "windows"], default="unix",
                       help="Platform (unix or windows)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and load existing models")
    parser.add_argument("--use_judge", action="store_true",
                       help="Use Phi-2 LLM as semantic judge")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU-only training (useful for memory-constrained systems)")
    args = parser.parse_args()

    # Setup
    device, device_name = setup_device(force_cpu=args.force_cpu)
    platform_config = setup_platform(args.platform)

    # Load data
    train_dataset, val_dataset = load_and_prepare_data()

    # Training
    if not args.skip_training:
        for model_key, model_config in MODELS.items():
            if model_config["type"] == "seq2seq":
                train_seq2seq_model(model_key, model_config, train_dataset, val_dataset, platform_config, device)
            else:
                train_causal_model(model_key, model_config, train_dataset, val_dataset, platform_config, device)

            cleanup_memory(model_key)
    else:
        print("="*80)
        print("SKIPPING TRAINING (loading existing models)")
        print("="*80 + "\n")

    # Ensemble comparison
    report = run_ensemble_comparison(val_dataset, MODELS, device, use_judge=args.use_judge)

    # Save report
    print("="*80)
    print("SAVING COMPARISON REPORT")
    print("="*80)

    output_filename = "ensemble_report_with_llm_judge.json" if args.use_judge else "ensemble_report.json"
    with open(output_filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved to: {output_filename}")
    print(f"\nSUMMARY:")
    print(f"  Total examples: {report['summary']['total_examples']}")
    print(f"  Full agreement: {report['summary']['full_agreements']} ({report['summary']['full_agreement_rate']*100:.1f}%)")

    if args.use_judge and "semantic_agreement_rate" in report["summary"]:
        print(f"  Semantic agreement: {report['summary']['semantic_agreements']} ({report['summary']['semantic_agreement_rate']*100:.1f}%)")

    print(f"\n  Format accuracy by model:")
    for model_key, acc in report['summary']['model_format_accuracy'].items():
        print(f"    - {model_key}: {acc*100:.1f}%")

    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
