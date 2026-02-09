# Multi-Model Ensemble Training for Weird Machine Gadget Classification

## Memory-Optimized for Apple Silicon Macs

---

## Introduction

In this guide, you'll learn how to fine-tune **two small language models** on weird machine gadget identification and compare their predictions through ensemble agreement analysis.

**Why 2 models?**
- **Architectural diversity**: Compare seq2seq (encoder-decoder) vs causal (decoder-only) architectures
- **Training paradigm diversity**: Instruction-tuned vs general pre-training
- **Memory efficiency**: Both models fit comfortably in 6–8GB unified memory on Apple Silicon
- **Agreement analysis**: Identify where models agree (high confidence) vs disagree (ambiguous cases)

### Models used

1. **FLAN-T5-small** (77M params) – Encoder-decoder, instruction-tuned T5
2. **DistilGPT2** (82M params) – Decoder-only, distilled from GPT-2
3. **Phi-2 (judge only)** (≈2.7B params) – Decoder-only reasoning LLM used *only* as an optional **semantic judge** at evaluation time (no training)

### By the end, you'll have

- Two trained models with different architectures
- An ensemble comparison report showing agreement/disagreement patterns
- A second report (optional) with **semantic agreement** judged by Phi-2
- Insights into which examples are "easy" (full agreement) vs "hard" (disagreement)
- Understanding of both **string-level** and **semantic** inter-model agreement as confidence metrics
- A template for multi-model ensemble research on student laptops

**Expected time to complete:**

- Training (100 examples): ~20–25 minutes total
- String-level comparison: ~1–2 minutes
- Phi-2 semantic judging on disagreements: ~1–2 minutes (10 examples, 4 disagreements)

---

## Prerequisites & Setup

### Step 0: Environment preparation

#### 0.1 Install required packages

```bash
pip install --upgrade pip

# PyTorch (CPU version - memory-optimized)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Hugging Face ecosystem
pip install "transformers[torch]"
pip install datasets
pip install accelerate

# Utilities
pip install scikit-learn tqdm
```

**What each package does:**
- `torch` – Deep learning framework (CPU-only to avoid memory issues)
- `transformers` – Hugging Face models and training utilities
- `datasets` – Load and process JSONL files efficiently
- `accelerate` – Training utilities (used by Trainer)
- `scikit-learn`, `tqdm` – Metrics and progress bars

**Installation time:** ~5–10 minutes on first run

#### 0.2 Verify installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
```

You should see version numbers; if you see errors, check that your virtual environment is activated.

#### 0.3 Verify dataset location

Ensure your `weird_machine_gadgets.jsonl` file is in the `data/` directory:

```bash
ls data/
# Should show: weird_machine_gadgets.jsonl
```

---

## Running the script

Save the provided `fine-tune-llms.py` script in your project root, then run:

```bash
# For macOS/Linux:
python fine-tune-llms.py --platform unix

# For Windows:
python fine-tune-llms.py --platform windows
```

### Skip training (load existing models)

If you have already trained the models and just want to re-run the comparison:

```bash
python fine-tune-llms.py --platform unix --skip_training
```

### Enable semantic judge (Phi-2)

To additionally run the **Phi-2 reasoning LLM** as a semantic judge on disagreements:

```bash
# Train (if needed) + compare + Phi-2 semantic judge
python fine-tune-llms.py --platform unix --use_judge

# Reuse existing FLAN-T5/DistilGPT2 checkpoints and only re-run comparison + judge
python fine-tune-llms.py --platform unix --skip_training --use_judge
```

When `--use_judge` is enabled:

- String-level results are saved to `ensemble_report_with_llm_judge.json`
- The report includes **both** string-level and semantic agreement statistics

---

## Step-by-step walkthrough

### MEMORY OPTIMIZATION LAYER

```
================================================================================
MEMORY OPTIMIZATION LAYER
================================================================================
Forcing CPU-only training to prevent out-of-memory errors...
✓ CUDA disabled
✓ MPS disabled (Apple Silicon GPU)
✓ All training will use CPU only
================================================================================
```

**What's happening:**

The script **forces CPU-only training** to prevent memory errors on Apple Silicon Macs by:

- Disabling CUDA (NVIDIA GPUs)
- Disabling MPS (Metal Performance Shaders – Apple GPU)
- Overriding PyTorch backend checks

**Why:**

- Apple Silicon unified memory is shared between CPU and GPU
- MPS backend can run out of memory with larger models (e.g., full GPT‑2)
- CPU-only is slower but **reliable** and fits in memory

---

### STEP 1: Platform setup

```
================================================================================
PLATFORM SETUP
================================================================================
✓ Platform: Unix (macOS/Linux)
  - Multiprocessing: Enabled
✓ CPU cores available: 8
✓ Data loading processes: 2
✓ PyTorch version: 2.2.2
✓ Device: CPU (forced)
```

**What's happening:**

Platform detection configures multiprocessing:

- **Unix (macOS/Linux)**: Enables multiprocessing (2 processes for data loading)
- **Windows**: Disables multiprocessing (avoids spawn issues)

**In the script:** `setup_platform()` function

---

### STEP 2: Loading and preparing data

```
================================================================================
LOADING AND PREPARING DATA
================================================================================
✓ Found data/weird_machine_gadgets.jsonl
✓ Total lines: 492
✓ Loaded 492 examples
✓ Subsampled to 100 examples
✓ Train: 90 | Validation: 10
```

**What's happening:**

1. **Load JSONL**: Reads all examples from the dataset
2. **Add prompts**: Creates two prompt formats:
   - **Seq2seq prompt** (for FLAN‑T5):  
     `Task: {instruction}\n\nExcerpt:\n{input}\n\nAnswer:`
   - **Causal prompt** (for DistilGPT2): includes the full answer for next-token prediction
3. **Subsample**: Uses 100 examples (configurable via `TOTAL_EXAMPLES_TO_USE`)
4. **Split**: 90% train, 10% validation

**In the script:** `load_and_prepare_data()` function

**To adjust dataset size:**

```python
# Top of fine-tune-llms.py:
TOTAL_EXAMPLES_TO_USE = 100  # Change to 200, 500, etc.
```

---

### STEP 3: Training model 1 – FLAN-T5-small

```
================================================================================
TRAINING MODEL: FLAN-T5-SMALL
================================================================================
  Model: google/flan-t5-small
  Type: seq2seq
  Params: 77M
  Architecture: Bidirectional encoder + autoregressive decoder
✓ Model loaded: T5ForConditionalGeneration
  Parameters: 76,961,152
  Tokenizing datasets...
  Training...
{'loss': 4.4353, 'epoch': 0.44}
...
{'train_loss': 3.9531, 'epoch': 3.0}
✓ Training complete! Loss: 3.9531
✓ Saved to: checkpoints/flan-t5-small/final_model
```

**What's happening:**

1. **Load model**: Downloads `google/flan-t5-small` (77M params, ~250MB)
2. **Tokenize**: Converts prompts and outputs to token IDs  
   - Input: up to 512 tokens  
   - Output: up to 256 tokens
3. **Train**: 3 epochs with batch size 1 + gradient accumulation (4 steps)
4. **Save**: Final model saved to `checkpoints/flan-t5-small/final_model`

**Architecture:**

- Encoder‑decoder (seq2seq)
- Bidirectional encoder (can attend to entire input)
- Autoregressive decoder (generates left‑to‑right)
- Instruction‑tuned (pre‑trained on instruction-following tasks)

**Training time:** ~3–5 minutes on an 8‑core CPU

**In the script:** `train_seq2seq_model()` function

---

### STEP 4: Memory cleanup

```
  Cleaning up memory after flan-t5-small...
  ✓ Memory freed
```

**What's happening:**

Between model training runs, the script explicitly:

- Frees Python references and runs garbage collection
- Empties CUDA cache if any GPU were present (defensive)

This prevents memory accumulation when training multiple models sequentially.

---

### STEP 5: Training model 2 – DistilGPT2

```
================================================================================
TRAINING MODEL: DISTILGPT2
================================================================================
  Model: distilgpt2
  Type: causal (causal LM)
  Params: 82M
  Architecture: Decoder-only, left-to-right attention
✓ Model loaded: GPT2LMHeadModel
  Parameters: 81,912,576
  Tokenizing datasets...
  Training...
{'loss': 4.4725, 'epoch': 0.44}
...
{'train_loss': 3.3547, 'epoch': 3.0}
✓ Training complete! Loss: 3.3547
✓ Saved to: checkpoints/distilgpt2/final_model
```

**What's happening:**

1. **Load model**: Downloads `distilgpt2` (82M params, ~350MB)
2. **Pad token**: GPT‑2 lacks a pad token; we set pad = EOS
3. **Tokenize**: Converts full prompt+output into token IDs (up to 768 tokens)
4. **Train**: 3 epochs with causal language modeling objective
5. **Save**: Final model saved to `checkpoints/distilgpt2/final_model`

**Architecture:**

- Decoder‑only (causal LM)
- Left‑to‑right attention (only attends to previous tokens)
- Distilled from GPT‑2 (smaller, faster)
- General pre‑training (not instruction-tuned)

**Training time:** ~3–5 minutes on an 8‑core CPU

**In the script:** `train_causal_model()` function

---

### STEP 6: Ensemble comparison & string-level agreement analysis

After training (or loading with `--skip_training`), both models are run on the validation set:

```
================================================================================
ENSEMBLE COMPARISON & AGREEMENT ANALYSIS (STRING-LEVEL)
================================================================================

================================================================================
VALIDATION EXAMPLE 1/10
================================================================================

INSTRUCTION: Identify weird machine CONTROL-FLOW gadgets in the excerpt...
EXCERPT: Command enable logic uses a general enable contact...

  [flan-t5-small] Generating...
  Output: gadget_type: Command enable logic; location: Command enable logic...

  [distilgpt2] Generating...
  Output: gadget_type: Control-Flow gadget; location: Control-Flow gadget...

────────────────────────────────────────────────────────────────────────────────
AGREEMENT ANALYSIS (STRING-LEVEL):
────────────────────────────────────────────────────────────────────────────────
  Full agreement (normalized): ✗ NO
  Unique raw gadget types: ['Command enable logic', 'Control-Flow gadget']
  Unique normalized types: ['commenablelogic', 'controlflow']
  Majority type: Command enable logic (1/2)

  Model-specific gadget types:
    - flan-t5-small: Command enable logic
    - distilgpt2: Control-Flow gadget

  Format checks:
    ✓ flan-t5-small: {'gadget_type': True, 'location': True, 'explanation': True}
    ✓ distilgpt2: {'gadget_type': True, 'location': True, 'explanation': True}
```

**What's happening:**

For each validation example, the script:

1. **Generates predictions**: Both models get the same prompt
2. **Extracts gadget types**: Parses `gadget_type:` from each prediction
3. **Normalizes gadget types**:
   - Lowercase
   - Strip "gadget" / "gadgets"
   - Remove spaces, hyphens, underscores, slashes
   - Remove "and"
4. **Checks format**: Ensures outputs contain required fields (`gadget_type`, `location`, `explanation`)
5. **Computes agreement**:
   - **Full agreement (normalized)**: All normalized gadget types match → high confidence
   - **Disagreement**: Normalized types differ → ambiguous or semantically different

**Agreement metrics:**

- `full_agreement`: Boolean – do **normalized** gadget types all match?
- `gadget_types`: Dict model → raw gadget type (e.g., `"Read/Write gadget"`)
- `unique_types`: List of distinct raw gadget types
- `normalized_types`: Dict model → normalized gadget type (e.g., `"readwrite"`)
- `unique_normalized_types`: Distinct normalized types (used for agreement)
- `majority_type`: Majority gadget type (using normalized vote, but reported with a representative raw label)
- `majority_count`: Number of models voting for the majority type
- `total_models`: Number of models in the ensemble (here, 2)

**In the script:** `compute_agreement()` and `run_ensemble_comparison()` functions

---

### STEP 7: Saving comparison report (string-only)

With `--use_judge` disabled, results are saved to:

```
ensemble_report.json
```

The JSON structure:

```json
{
  "summary": {
    "total_examples": 10,
    "full_agreements": 6,
    "full_agreement_rate": 0.6,
    "disagreement_rate": 0.4,
    "model_format_accuracy": {
      "flan-t5-small": 1.0,
      "distilgpt2": 0.4
    },
    "llm_judge_used": false
  },
  "results": [
    {
      "example_id": 0,
      "instruction": "...",
      "excerpt": "...",
      "gold_output": "...",
      "predictions": {
        "flan-t5-small": "...",
        "distilgpt2": "..."
      },
      "format_checks": {
        "flan-t5-small": {"gadget_type": true, "location": true, "explanation": true},
        "distilgpt2": {"gadget_type": true, "location": true, "explanation": true}
      },
      "agreement": {
        "full_agreement": false,
        "gadget_types": {
          "flan-t5-small": "Command enable logic",
          "distilgpt2": "Control-Flow gadget"
        },
        "unique_types": ["Command enable logic", "Control-Flow gadget"],
        "normalized_types": {
          "flan-t5-small": "commenablelogic",
          "distilgpt2": "controlflow"
        },
        "unique_normalized_types": ["commenablelogic", "controlflow"],
        "majority_type": "Command enable logic",
        "majority_count": 1,
        "total_models": 2
      }
    }
  ]
}
```

---

## Optional semantic judge with Phi-2

When you run with `--use_judge`, the script performs an additional **semantic evaluation** step using a local Phi‑2 model (`microsoft/phi-2`) on CPU:

```bash
python fine-tune-llms.py --platform unix --skip_training --use_judge
```

### What changes when `--use_judge` is on?

1. The base ensemble run is exactly the same (string-level agreement as before).
2. For **string-level disagreements only**, the script:
   - Sends the FLAN‑T5 and DistilGPT2 full predictions plus the gold output to Phi‑2
   - Asks Phi‑2 to decide between:
     - `FULL_AGREEMENT`
     - `PARTIAL_AGREEMENT`
     - `DISAGREEMENT`
     - `INVALID`
   - Records a short natural language explanation
3. The report is saved as:

```
ensemble_report_with_llm_judge.json
```

### New fields in the report

Each result now includes:

- `agreement.semantic_agreement`: 
  - `True` if either:
    - String-level `full_agreement` was already true, or
    - Phi‑2 judged the predictions as `FULL_AGREEMENT` or `PARTIAL_AGREEMENT`
  - `False` otherwise
- `agreement.llm_judgment` (only for cases sent to the judge):
  - `verdict`: one of the four labels above
  - `explanation`: one-sentence reason from Phi‑2

The summary gains:

- `semantic_agreements`: Number of examples with `semantic_agreement = True`
- `semantic_agreement_rate`: Proportion of examples with semantic agreement
- `llm_judge_used`: `true` when `--use_judge` was enabled and the judge ran

This allows you to contrast:

- **String-level agreement rate** (normalized tokens)
- **Semantic agreement rate** (string + Phi‑2 reasoning)

---

## Understanding the reports

### String-level summary (`ensemble_report.json`)

Key fields:

- `full_agreement_rate`: Fraction of examples where **normalized** gadget types match
- `disagreement_rate`: Fraction where normalized gadget types differ
- `model_format_accuracy`: Fraction of examples where that model's output includes all three fields (`gadget_type`, `location`, `explanation`)

With 100 training examples, it is common to see:

- String-level agreement in the range 30–60% after normalization
- Format accuracy:
  - Higher for FLAN‑T5 (instruction‑tuned)
  - Lower for DistilGPT2 (general model)

### Semantically-aware summary (`ensemble_report_with_llm_judge.json`)

When `--use_judge` is enabled, compare:

- `full_agreement_rate` vs `semantic_agreement_rate`
  - `semantic_agreement_rate` should be **higher**, because Phi‑2 can detect:
    - Conceptually similar gadget types with different wording
    - Truncated or slightly malformed labels that are still clearly similar

Example:

- String-level: 6/10 full agreements (60%)
- Phi‑2 semantic: 8/10 semantic agreements (80%)

---

## Limitations of simple string normalization

### The semantic gap problem

The current script uses **string normalization** (lowercase, remove punctuation and "gadget", strip connectors) to handle minor variations such as:

✅ **Works well for:**

- Capitalization: `"ReadWrite gadget"` vs `"READWRITE gadget"` → both → `"readwrite"`
- Hyphens: `"Control-Flow gadget"` vs `"Control FLOW gadget"` → both → `"controlflow"`
- Spacing: `"I/O and SIDE-EFFECT gadget"` vs `"I/O AND SIDE EFFECT gadgets"` → both → `"iosideeffect"`
- Minor rephrasing with "gadget" suffixes

❌ **Does NOT work for semantic differences:**

| Example | FLAN‑T5 output | DistilGPT2 output | Issue |
|--------|----------------|-------------------|-------|
| 0 | `"Command enable logic"` | `"Control-Flow gadget"` | Specific mechanism vs generic category |
| 4 | `"Alarm and Condition Objects"` | `"Read/Write gadget"` | Alarm state variables vs generic read/write |
| 9 | `"Timing/Synthetic gadget"` | `"Timing/SYNCHRONIZ"` | Full label vs truncated category |

### Why this matters

String normalization cannot detect:

1. **Semantic equivalence**: "Command enable logic" and "Control‑Flow gadget" both relate to control flow but differ in specificity.
2. **Abstraction level**: "BOOL tag" vs "Read/Write gadget" is like "Honda Civic" vs "car".
3. **Garbled or truncated output**: Especially on the causal model when generation diverges.
4. **Partial correctness**: One model may be more precise while the other is more generic.

### Local reasoning LLM (Phi‑2) as judge

The Phi‑2 semantic judge helps close this gap:

1. **Judges semantic equivalence**:
   - Are the gadget types conceptually the same, even if the strings differ?
2. **Detects abstraction mismatches**:
   - Is one output a specific instance of the other's category?
3. **Flags invalid outputs**:
   - Repetitive, nonsensical, or incomplete gadget labels
4. **Scores partial correctness**:
   - Distinguishes "FULL_AGREEMENT" vs "PARTIAL_AGREEMENT" vs "DISAGREEMENT" vs "INVALID"

The pipeline becomes:

- **Phase 1 – String-level (cheap, deterministic):**

  ```
  Model A → "Read/Write gadget"  ──┐
                                   ├─→ normalize() → string match → AGREEMENT
  Model B → "READ/WRITE gadget"  ──┘
  ```

- **Phase 2 – Semantic (expensive, but only on disagreements):**

  ```
  Model A → "Command enable logic"  ──┐
                                      ├─→ Phi-2 → verdict + explanation
  Model B → "Control-Flow gadget"   ──┘
  ```

Phi‑2 is only called on the subset of examples where normalized gadget types differ (disagreements), which keeps compute manageable on student laptops.

---

## Configuration & tuning

### Key configuration variables

At the top of `fine-tune-llms.py`:

```python
# Dataset sizing
TOTAL_EXAMPLES_TO_USE = 100  # ← change this for more data
EVAL_SPLIT = 0.1             # 10% validation

# Training hyperparameters (shared across models)
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
```

### Scaling up

**Phase 1: Increase dataset size**

```python
TOTAL_EXAMPLES_TO_USE = 200  # or 300, 500
```

Expected effects:

- Higher agreement rate
- Better format accuracy
- More stable predictions

**Phase 2: Increase epochs**

```python
NUM_EPOCHS = 5  # or 10
```

Trade‑off: Better quality vs longer training time.

**Phase 3: Adjust learning rate**

```python
LEARNING_RATE = 1e-5  # for stability
# or
LEARNING_RATE = 1e-4  # for faster convergence
```

Rule of thumb:

- If loss plateaus early → increase learning rate
- If loss is unstable → decrease learning rate

---

## Research questions for students

### 1. Agreement pattern analysis

**Question:** Which examples cause disagreement between models?

**How to explore:**

```python
import json

with open('ensemble_report.json', 'r') as f:
    report = json.load(f)

# Find disagreements (string-level)
disagreements = [
    r for r in report['results']
    if not r['agreement']['full_agreement']
]

print(f"Found {len(disagreements)} string-level disagreements")
for d in disagreements:
    print(f"\nExample {d['example_id']}:")
    print(f"  Instruction: {d['instruction'][:80]}...")
    print(f"  FLAN-T5: {d['agreement']['gadget_types']['flan-t5-small']}")
    print(f"  DistilGPT2: {d['agreement']['gadget_types']['distilgpt2']}")
    print(f"  Normalized FLAN-T5: {d['agreement']['normalized_types']['flan-t5-small']}")
    print(f"  Normalized DistilGPT2: {d['agreement']['normalized_types']['distilgpt2']}")
    print(f"  Gold: {d['gold_output'][:80]}...")
```

**Hypotheses to test:**

- Do disagreements correlate with excerpt length?
- Are some gadget types more ambiguous?
- Does heavy technical jargon increase disagreements?
- Do normalized types show that some apparent disagreements are just spelling/capitalization issues?

---

### 2. Architectural comparison

**Question:** Does the seq2seq model (FLAN‑T5) outperform the causal LM (DistilGPT2)?

**Metrics to compare:**

- Format accuracy (from `model_format_accuracy`)
- Agreement with the gold standard (manually or with an external script)
- Training loss (final values printed at the end of training)

**Hypotheses:**

- Instruction‑tuned models (FLAN‑T5) should have better format adherence
- Seq2seq models should handle longer contexts and structured output better

---

### 3. Confidence via agreement

**Question:** Can we use inter‑model agreement as a confidence score?

**Approach:**

1. Label validation examples:
   - High confidence = string-level full agreement
   - Low confidence = string-level disagreement
2. Manually inspect a sample from each category.
3. Measure: Are high‑confidence predictions more accurate than low‑confidence ones?

**Implementation:**

```python
high_conf = [r for r in report['results'] if r['agreement']['full_agreement']]
low_conf = [r for r in report['results'] if not r['agreement']['full_agreement']]

print(f"High confidence (string-level): {len(high_conf)} examples")
print(f"Low confidence (string-level): {len(low_conf)} examples")
```

---

### 4. Error analysis by gadget type

**Question:** Which gadget types are hardest to classify?

**Approach:**

```python
from collections import defaultdict
import re

errors_by_type = defaultdict(list)

for r in report['results']:
    # Extract gold gadget type from the gold_output string
    m = re.search(r"gadget_type:\s*([^;\n]+)", r['gold_output'])
    if not m:
        continue
    gold_type = m.group(1).strip()

    if not r['agreement']['full_agreement']:
        errors_by_type[gold_type].append(r['example_id'])

for gtype, examples in errors_by_type.items():
    print(f"{gtype}: {len(examples)} string-level disagreements")
```

---

### 5. Majority voting performance

**Question:** Does ensemble voting help with more models?

**Approach:**

1. Conceptually extend to 3+ models.
2. For each example, take a majority vote on gadget type (using normalized labels).
3. Compare majority vote vs individual model accuracy.

With only 2 models, ties are common; with 3 or more, majority voting is more informative.

---

### 6. Semantic agreement analysis (Phi‑2 vs string-only)

**Question:** How many string-level "disagreements" are actually **semantic agreements**?

**Approach (requires `--use_judge` report):**

```python
import json

with open('ensemble_report_with_llm_judge.json', 'r') as f:
    report = json.load(f)

results = report['results']

# Cases where string-level disagrees but Phi-2 says semantic_agreement = True
upgraded = [
    r for r in results
    if (not r['agreement']['full_agreement'])
    and r['agreement'].get('semantic_agreement', False)
]

print(f"String-level disagreements reclassified as semantic agreements: {len(upgraded)}")
for r in upgraded:
    j = r['agreement']['llm_judgment']
    print(f"\nExample {r['example_id']}:")
    print(f"  FLAN-T5 gadget_type: {r['agreement']['gadget_types']['flan-t5-small']}")
    print(f"  DistilGPT2 gadget_type: {r['agreement']['gadget_types']['distilgpt2']}")
    print(f"  Phi-2 verdict: {j['verdict']}")
    print(f"  Phi-2 explanation: {j['explanation'][:120]}...")
```

**Discussion questions:**

- Do you agree with Phi‑2's upgrades to `FULL_AGREEMENT` or `PARTIAL_AGREEMENT`?
- Are there cases where you think Phi‑2 is wrong?
- How much does semantic agreement differ from string-level agreement?

---

### 7. Effect of the local reasoning LLM (Phi‑2)

**Question:** How much does Phi‑2 change your view of model reliability?

**Suggested exploration:**

1. Compare `summary["full_agreement_rate"]` vs `summary["semantic_agreement_rate"]`.
2. Inspect examples where Phi‑2 says:
   - `FULL_AGREEMENT`
   - `PARTIAL_AGREEMENT`
   - `DISAGREEMENT`
   - `INVALID`
3. For `INVALID` cases, look at the raw model outputs:
   - Are they garbled or structurally broken?
4. For `PARTIAL_AGREEMENT`, look at abstraction mismatches:
   - Example: "Timer TON block" vs "Timing/Synchronization gadget"

**Guiding questions:**

- Does Phi‑2 mostly correct for string quirks (capitalization, truncation)?
- Does it ever hallucinate or misjudge obvious differences?
- Would you trust Phi‑2's verdict as a grading signal for student-written gadget labels?

---

## Troubleshooting

### Issue: `RuntimeError: MPS backend out of memory`

**Cause:** Accidentally using the Apple GPU (MPS) on Apple Silicon.

**Solution:** The script already forces CPU-only; verify the banner:

```
✓ MPS disabled (Apple Silicon GPU)
✓ All training will use CPU only
```

If you still see this, ensure the memory optimization code is at the very top of `fine-tune-llms.py` and that no other script is enabling MPS.

---

### Issue: `TypeError: ... generate() argument after ** must be a mapping`

**Cause:** Passing `**inputs["input_ids"]` instead of the entire `inputs` dict.

**Solution:** The script has been fixed to:

```python
outputs = model.generate(
    inputs["input_ids"],              # direct tensor
    attention_mask=inputs.get("attention_mask"),
    max_new_tokens=max_new_tokens,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
```

---

### Issue: Training too slow on CPU

**This is expected.** To speed up:

1. Temporarily reduce dataset size:

   ```python
   TOTAL_EXAMPLES_TO_USE = 50
   ```

2. Reduce sequence lengths:

   ```python
   MAX_INPUT_LENGTH = 256
   MAX_TARGET_LENGTH = 128
   ```

3. Use fewer epochs:

   ```python
   NUM_EPOCHS = 2
   ```

4. Close other CPU-heavy applications.

---

### Issue: Model outputs are gibberish

**Possible causes:**

- Too little training data (100 examples is minimal)
- Learning rate too high
- Generation parameters too permissive (e.g., high temperature, sampling)

**Potential fixes:**

1. Increase `TOTAL_EXAMPLES_TO_USE` to 200+
2. Lower `LEARNING_RATE` to `1e-5`
3. Increase `NUM_EPOCHS` to 5
4. Keep `do_sample=False` and low `max_new_tokens` during evaluation

---

### Issue: Low agreement rate (<20%)

With very small datasets and no normalization, this is common.

**Improvements:**

1. Ensure you are using normalized gadget types (the current script does this).
2. Scale to 500–1000 examples.
3. Increase epochs to 5–10.
4. Use `--use_judge` to incorporate Phi‑2 semantic judgments.
5. Optionally upgrade FLAN‑T5‑small to FLAN‑T5‑base for higher-capacity seq2seq.

---

### Issue: High disagreement rate but predictions "look similar"

This is exactly the **semantic gap** problem.

Examples:

- Spelling or truncation: `"Timing/SYNCHRONIZ"` vs `"Timing/Synchronization gadget"`
- Capitalization: `"Read/Write gadget"` vs `"READ/WRITE gadget"`
- Specific vs category: `"Command enable logic"` vs `"Control-Flow gadget"`

**What to try:**

- Inspect `agreement["normalized_types"]` to see if normalization is working.
- If it is, enable `--use_judge` to see which cases Phi‑2 upgrades to `PARTIAL_AGREEMENT` or `FULL_AGREEMENT`.

---

## Advanced experiments

### Experiment 1: Add a third model

Extend the `MODELS` dict:

```python
MODELS = {
    "flan-t5-small": {...},
    "distilgpt2": {...},
    "flan-t5-base": {  # Add this
        "name": "google/flan-t5-base",
        "type": "seq2seq",
        "params": "250M",
        "description": "Larger seq2seq, better quality",
        "architecture": "Bidirectional encoder + autoregressive decoder",
    },
}
```

Trade‑off: slower training, potentially better accuracy and more interesting 3‑way comparisons and majority voting.

---

### Experiment 2: Custom inference script

Use your fine‑tuned FLAN‑T5 and DistilGPT2 for custom excerpts:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

# Load FLAN-T5
t5_tokenizer = AutoTokenizer.from_pretrained('checkpoints/flan-t5-small/final_model')
t5_model = AutoModelForSeq2SeqLM.from_pretrained('checkpoints/flan-t5-small/final_model')

# Load DistilGPT2
gpt_tokenizer = AutoTokenizer.from_pretrained('checkpoints/distilgpt2/final_model')
gpt_model = AutoModelForCausalLM.from_pretrained('checkpoints/distilgpt2/final_model')

instruction = "Identify weird machine ARITHMETIC/COMPUTATION gadgets..."
excerpt = "The ADD instruction adds two integer values..."
prompt = f"Task: {instruction}\n\nExcerpt:\n{excerpt}\n\nAnswer:"

# FLAN-T5 prediction
t5_inputs = t5_tokenizer(prompt, return_tensors="pt")
t5_outputs = t5_model.generate(**t5_inputs, max_new_tokens=256)
t5_pred = t5_tokenizer.decode(t5_outputs[0], skip_special_tokens=True)

# DistilGPT2 prediction
gpt_inputs = gpt_tokenizer(prompt, return_tensors="pt")
gpt_outputs = gpt_model.generate(gpt_inputs["input_ids"], max_new_tokens=256)
gpt_pred = gpt_tokenizer.decode(gpt_outputs[0], skip_special_tokens=True)

print(f"FLAN-T5: {t5_pred}")
print(f"DistilGPT2: {gpt_pred.split('Answer:')[-1].strip()}")
```

---

### Experiment 3: Measure ROUGE scores

Quantify prediction quality:

```bash
pip install rouge-score
```

```python
from rouge_score import rouge_scorer
import json

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

with open('ensemble_report.json', 'r') as f:
    report = json.load(f)

for model_key in ['flan-t5-small', 'distilgpt2']:
    scores = []
    for r in report['results']:
        gold = r['gold_output']
        pred = r['predictions'][model_key]
        score = scorer.score(gold, pred)
        scores.append(score['rougeL'].fmeasure)

    avg_score = sum(scores) / len(scores)
    print(f"{model_key} ROUGE-L: {avg_score:.3f}")
```

---

### Experiment 4: Compare local Phi‑2 judge vs a cloud LLM

As an advanced exercise, you can:

1. Use the existing Phi‑2 judge for local judgments.
2. Implement a remote "LLM‑as‑a‑judge" call (e.g., GPT‑4, Claude) for the same disagreements.
3. Compare:

   - Agreement between Phi‑2 and the cloud model
   - Which examples they disagree on
   - Whether one is consistently stricter or more lenient

This helps students reason about **judge reliability** and **evaluation bias**.

---

## Key takeaways

1. **Ensemble diversity matters**: Different architectures (seq2seq vs causal) provide different perspectives.
2. **String normalization helps**: It handles capitalization, spacing, and "gadget" suffixes.
3. **Disagreement ≠ always wrong**: Many string-level disagreements are semantic matches.
4. **Reasoning LLMs add value**: Phi‑2 can identify partial agreements, abstraction mismatches, and invalid outputs.
5. **Local judges are feasible**: A 2.7B‑parameter Phi‑2 model can run on student laptops using CPU‑only.
6. **Format adherence differs by architecture**: FLAN‑T5 usually adheres more strictly to the required `gadget_type/location/explanation` schema.
7. **Iterative scaling is important**: Start at 100 examples to debug, then scale to 200–500+ for better accuracy.

---

## Summary checklist

**Before you start:**

- [ ] Dependencies installed (`torch`, `transformers`, `datasets`, etc.)
- [ ] `data/weird_machine_gadgets.jsonl` exists
- [ ] `fine-tune-llms.py` saved in project root
- [ ] ~5 GB free disk space
- [ ] 8+ GB RAM (for CPU training)
- [ ] Optional: ~5–6 GB extra RAM headroom if using `--use_judge` with Phi‑2

**After training and comparison:**

- [ ] Both models trained successfully (check `checkpoints/` directories)
- [ ] `ensemble_report.json` generated
- [ ] Review string-level agreement rate and format accuracy
- [ ] Inspect 2–3 string-level disagreement examples manually
- [ ] Check `normalized_types` to see if some disagreements are just spelling/case variants
- [ ] Optionally run with `--use_judge` to get `ensemble_report_with_llm_judge.json`
- [ ] Compare string-level vs semantic agreement rates
- [ ] Plan your next experiment (scale data, adjust hyperparameters, or extend the ensemble)

---

**Happy ensemble training and semantic judging!** 🚀
