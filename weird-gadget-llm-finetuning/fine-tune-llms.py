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

New in this version:
- String-normalized agreement (handles capitalization/spacing)
- Optional lightweight reasoning LLM judge (Phi-3-mini) to score semantic agreement

Usage:
    # Basic (train + compare, string-only agreement)
    python main.py --platform unix

    # Skip training, just re-run comparison
    python main.py --platform unix --skip_training

    # Train + compare with reasoning judge
    python main.py --platform unix --use_judge

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
original_mps_is_available = torch.backends.mps.is_available if hasattr(torch.backends, "mps") else (lambda: False)
torch.cuda.is_available = lambda: False
if hasattr(torch.backends, "mps"):
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

def setup_platform(platform: str) -> int:
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
    print("✓ Device: CPU (forced)")
    return max_processes

# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

def free_memory():
    """Aggressively free memory between model runs."""
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
# INFERENCE & STRING-LEVEL AGREEMENT
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
            inputs["input_ids"],  # tensor directly
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
    if not prediction:
        return "UNKNOWN"
    lower = prediction.lower()
    if "gadget_type:" in lower:
        start = lower.index("gadget_type:") + len("gadget_type:")
        rest = prediction[start:].split(";")[0].split("\n")[0].strip()
        return rest if rest else "UNKNOWN"
    return "UNKNOWN"

def normalize_gadget_type(gadget_type: str) -> str:
    """
    Normalize gadget type for comparison:
    - lowercase
    - remove 'gadget'/'gadgets'
    - remove spaces, hyphens, underscores, slashes
    - remove 'and'
    """
    if not gadget_type or gadget_type == "UNKNOWN":
        return "UNKNOWN"
    normalized = gadget_type.lower()
    normalized = normalized.replace(" gadget", "").replace(" gadgets", "")
    normalized = normalized.replace(" ", "").replace("-", "").replace("_", "").replace("/", "")
    normalized = normalized.replace("and", "")
    return normalized.strip()

def compute_agreement(predictions: Dict[str, str]) -> Dict:
    """Compute inter-model agreement metrics with normalization."""
    # Raw gadget types
    raw_gadget_types = {k: extract_gadget_type(v) for k, v in predictions.items()}
    # Normalized for comparison
    normalized_types = {k: normalize_gadget_type(v) for k, v in raw_gadget_types.items()}

    unique_raw = set(raw_gadget_types.values())
    unique_normalized = set(normalized_types.values())

    # String-level agreement = all normalized types equal
    full_agreement = len(unique_normalized) == 1

    # Count normalized types for majority
    type_counts = Counter(normalized_types.values())
    majority_normalized, majority_count = type_counts.most_common(1)[0]

    # Representative raw type for majority
    majority_raw = None
    for k, v in normalized_types.items():
        if v == majority_normalized:
            majority_raw = raw_gadget_types[k]
            break
    if majority_raw is None:
        majority_raw = majority_normalized

    return {
        "full_agreement": full_agreement,
        "gadget_types": raw_gadget_types,
        "unique_types": list(unique_raw),
        "normalized_types": normalized_types,
        "unique_normalized_types": list(unique_normalized),
        "majority_type": majority_raw,
        "majority_count": majority_count,
        "total_models": len(predictions),
    }

def run_ensemble_comparison(models_info: Dict, val_ds):
    """Run all models on validation set and analyze string-level agreement."""
    print("\n" + "=" * 80)
    print("ENSEMBLE COMPARISON & AGREEMENT ANALYSIS (STRING-LEVEL)")
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
            print(f"  Output: {pred[:120]}...")

        # Compute agreement
        agreement = compute_agreement(predictions)

        print(f"\n{'─'*80}")
        print("AGREEMENT ANALYSIS (STRING-LEVEL):")
        print(f"{'─'*80}")
        print(f"  Full agreement (normalized): {'✓ YES' if agreement['full_agreement'] else '✗ NO'}")
        print(f"  Unique raw gadget types: {agreement['unique_types']}")
        print(f"  Unique normalized types: {agreement['unique_normalized_types']}")
        print(f"  Majority type: {agreement['majority_type']} ({agreement['majority_count']}/{agreement['total_models']})")

        print(f"\n  Model-specific gadget types:")
        for model_key, gtype in agreement["gadget_types"].items():
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

# ============================================================================
# LIGHTWEIGHT REASONING LLM JUDGE (Phi-3-mini-4k-instruct, 4-bit quantized)
# ============================================================================

# ============================================================================
# LIGHTWEIGHT REASONING LLM JUDGE (Phi-2, CPU-friendly)
# ============================================================================

class LightweightSemanticJudge:
    """
    Lightweight reasoning LLM for judging semantic agreement between model predictions.
    Uses microsoft/phi-2 (~2.7B params) on CPU without quantization (~5 GB).
    """

    def __init__(self, model_name: str = "microsoft/phi-2"):
        print("\n" + "=" * 80)
        print("LOADING LIGHTWEIGHT REASONING JUDGE")
        print("=" * 80)
        print(f"  Model: {model_name}")
        print("  Model size: ~2.7B parameters")
        print("  Quantization: None (16-bit weights on CPU)")
        print("  Expected memory: ~5 GB")
        print("  Device: CPU")

        # Load Phi-2 on CPU. Use float16 to reduce memory footprint vs float32.
        # If you hit issues on some CPUs, change torch.float16 -> torch.float32.
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

        print("✓ Reasoning judge loaded successfully")

    def create_judge_prompt(self, pred_a: str, pred_b: str, gold: str) -> str:
        """
        Create a structured prompt for semantic agreement judgment.
        """
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

Consider:
1. Do they refer to the same gadget type category?
2. Are spelling/capitalization the only differences?
3. Is one garbled or repetitive?
4. Do they differ in specificity (e.g., "BOOL tag" vs "ReadWrite gadget")?

Verdict:<|end|>
<|assistant|>
"""

    def judge_semantic_agreement(
        self,
        pred_a: str,
        pred_b: str,
        gold: str,
        max_new_tokens: int = 100,
    ) -> Dict[str, str]:
        """
        Judge semantic agreement between two predictions.

        Returns:
            {
                "verdict": "FULL_AGREEMENT" | "PARTIAL_AGREEMENT" | "DISAGREEMENT" | "INVALID",
                "explanation": "Brief explanation",
                "raw_response": "Full model output",
            }
        """
        prompt = self.create_judge_prompt(pred_a, pred_b, gold)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # low temp for consistency
                do_sample=False,  # deterministic
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract verdict and explanation
        lines = response.split("\n")
        verdict = "DISAGREEMENT"  # default

        for line in lines:
            line_upper = line.strip().upper()
            if "FULL_AGREEMENT" in line_upper:
                verdict = "FULL_AGREEMENT"
                break
            elif "PARTIAL_AGREEMENT" in line_upper:
                verdict = "PARTIAL_AGREEMENT"
                break
            elif "DISAGREEMENT" in line_upper:
                verdict = "DISAGREEMENT"
                break
            elif "INVALID" in line_upper:
                verdict = "INVALID"
                break

        explanation = ""
        for line in lines:
            if (
                len(line.strip()) > 20
                and not any(
                    kw in line.upper()
                    for kw in ["FULL_AGREEMENT", "PARTIAL_AGREEMENT", "DISAGREEMENT", "INVALID"]
                )
            ):
                explanation = line.strip()
                break
        if not explanation:
            explanation = "No explanation provided"

        return {
            "verdict": verdict,
            "explanation": explanation,
            "raw_response": response,
        }

    def batch_judge(self, disagreements: List[Dict], verbose: bool = True) -> List[Dict]:
        """
        Judge all disagreements in batch.

        Args:
            disagreements: List of dicts with keys 'pred_a', 'pred_b', 'gold', 'example_id'

        Returns:
            List of judgment dicts
        """
        judgments = []
        print(f"\n  Judging {len(disagreements)} disagreements...")

        for i, dis in enumerate(disagreements):
            if verbose:
                print(
                    f"  [{i+1}/{len(disagreements)}] Judging example {dis.get('example_id', i)}...",
                    end=" ",
                )
            judgment = self.judge_semantic_agreement(
                dis["pred_a"],
                dis["pred_b"],
                dis["gold"],
            )
            judgment["example_id"] = dis.get("example_id", i)
            judgments.append(judgment)
            if verbose:
                print(f"{judgment['verdict']}")

        return judgments

def run_llm_judge_on_results(results: List[Dict], use_judge: bool) -> List[Dict]:
    """
    Optionally apply lightweight reasoning LLM judge to disagreements.
    Adds:
      - agreement['semantic_agreement'] (bool)
      - agreement['llm_judgment'] (dict) for judged examples
    """
    # Initialize semantic_agreement to string-level full_agreement
    for r in results:
        r["agreement"]["semantic_agreement"] = r["agreement"]["full_agreement"]

    if not use_judge:
        print("\nLLM judge disabled (--use_judge not set).")
        return results

    # Collect disagreements
    disagreements = []
    for r in results:
        if not r["agreement"]["full_agreement"]:
            preds = r["predictions"]
            model_keys = list(preds.keys())
            if len(model_keys) < 2:
                continue
            disagreements.append({
                "example_id": r["example_id"],
                "pred_a": preds[model_keys[0]],
                "pred_b": preds[model_keys[1]],
                "gold": r["gold_output"],
            })

    if not disagreements:
        print("\nNo string-level disagreements to judge (100% agreement).")
        return results

    print("\n" + "=" * 80)
    print("SEMANTIC JUDGMENT PHASE (LIGHTWEIGHT LLM)")
    print("=" * 80)
    print(f"  String-level disagreements: {len(disagreements)}/{len(results)}")

    try:
        judge = LightweightSemanticJudge()
        judgments = judge.batch_judge(disagreements, verbose=True)

        # Map example_id → judgment
        jmap = {j["example_id"]: j for j in judgments}

        for r in results:
            ex_id = r["example_id"]
            if ex_id in jmap:
                j = jmap[ex_id]
                r["agreement"]["llm_judgment"] = {
                    "verdict": j["verdict"],
                    "explanation": j["explanation"],
                }
                if j["verdict"] in ["FULL_AGREEMENT", "PARTIAL_AGREEMENT"]:
                    r["agreement"]["semantic_agreement"] = True
                else:
                    r["agreement"]["semantic_agreement"] = False

        # Summary
        string_agreements = sum(1 for r in results if r["agreement"]["full_agreement"])
        semantic_agreements = sum(1 for r in results if r["agreement"]["semantic_agreement"])

        print("\n" + "=" * 80)
        print("AGREEMENT SUMMARY (WITH LLM JUDGE)")
        print("=" * 80)
        print(f"  Total examples: {len(results)}")
        print(f"  String-level agreement: {string_agreements} ({string_agreements/len(results)*100:.1f}%)")
        print(f"  Semantic agreement (with LLM): {semantic_agreements} ({semantic_agreements/len(results)*100:.1f}%)")
        print(f"  Improvement: +{semantic_agreements - string_agreements} examples")

        verdicts = [
            r["agreement"].get("llm_judgment", {}).get("verdict", "N/A")
            for r in results
            if "llm_judgment" in r["agreement"]
        ]
        v_counts = Counter(verdicts)
        print("\n  LLM Judge verdicts:")
        for v, c in v_counts.items():
            print(f"    - {v}: {c}")

    except Exception as e:
        print(f"  ✗ Failed to load or run LLM judge: {e}")
        print("  Falling back to string-only agreement (semantic_agreement = full_agreement).")
        for r in results:
            r["agreement"]["semantic_agreement"] = r["agreement"]["full_agreement"]

    return results

# ============================================================================
# REPORT SAVING
# ============================================================================

def save_comparison_report(results: List[Dict], output_path: str = "ensemble_report.json"):
    """Save detailed comparison report."""
    print("\n" + "=" * 80)
    print("SAVING COMPARISON REPORT")
    print("=" * 80)

    total_examples = len(results)
    full_agreements = sum(1 for r in results if r["agreement"]["full_agreement"])
    semantic_agreements = sum(1 for r in results if r["agreement"].get("semantic_agreement", False))

    # Model-specific format accuracy
    model_format_accuracy = {}
    model_keys = list(results[0]["format_checks"].keys())
    for model_key in model_keys:
        correct = sum(
            1 for r in results
            if all(r["format_checks"][model_key].values())
        )
        model_format_accuracy[model_key] = correct / total_examples

    llm_judge_used = any("llm_judgment" in r["agreement"] for r in results)

    summary = {
        "total_examples": total_examples,
        "full_agreements": full_agreements,
        "full_agreement_rate": full_agreements / total_examples,
        "semantic_agreements": semantic_agreements,
        "semantic_agreement_rate": semantic_agreements / total_examples,
        "disagreement_rate": 1 - (full_agreements / total_examples),
        "model_format_accuracy": model_format_accuracy,
        "llm_judge_used": llm_judge_used,
    }

    report = {
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ Report saved to: {output_path}")
    print("\nSUMMARY:")
    print(f"  Total examples: {total_examples}")
    print(f"  Full agreement (string-level): {full_agreements} ({summary['full_agreement_rate']*100:.1f}%)")
    print(f"  Semantic agreement (with LLM): {semantic_agreements} ({summary['semantic_agreement_rate']*100:.1f}%)")
    print("\n  Format accuracy by model:")
    for model_key, accuracy in model_format_accuracy.items():
        print(f"    - {model_key}: {accuracy*100:.1f}%")
    print(f"\n  LLM judge used: {llm_judge_used}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model ensemble training and comparison (memory-optimized)"
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["windows", "unix"],
        default="unix",
        help="Platform: 'windows' or 'unix' (macOS/Linux). Default: unix",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        dest="skip_training",
        help="Skip training and load existing models (for testing comparison only)",
    )
    parser.add_argument(
        "--use_judge",
        action="store_true",
        dest="use_judge",
        help="Use lightweight reasoning LLM judge for semantic agreement",
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
                print("  Please train first without --skip_training flag")
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

        models_info[model_key] = {
            "config": model_config,
            "path": model_path,
            "tokenizer": tokenizer,
            "model": model,
            "type": model_config["type"],
        }

    # Run ensemble comparison (string-level)
    results = run_ensemble_comparison(models_info, val_ds)

    # Free base models before loading LLM judge
    del models_info
    free_memory()

    # Optional LLM judge for semantic agreement
    results = run_llm_judge_on_results(results, use_judge=args.use_judge)

    # Save report
    report_filename = "ensemble_report_with_llm_judge.json" if args.use_judge else "ensemble_report.json"
    save_comparison_report(results, report_filename)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"  Report saved: {report_filename}")
    print(f"  Models: {', '.join(MODELS.keys())}")
    print(f"  Semantic evaluation (LLM judge): {'Enabled' if args.use_judge else 'Disabled (string-only)'}")

if __name__ == "__main__":
    main()
