# Multi-Model Ensemble Training for Weird Machine Gadget Classification

## Memory-Optimized for Apple Silicon Macs

---

## Introduction

In this guide, you'll learn how to fine-tune **two small language models** on weird machine gadget identification and compare their predictions through ensemble agreement analysis.

**Why 2 models?**
- **Architectural diversity**: Compare seq2seq (encoder-decoder) vs causal (decoder-only) architectures
- **Training paradigm diversity**: Instruction-tuned vs general pre-training
- **Memory efficiency**: Both models fit comfortably in 6-8GB unified memory on Apple Silicon
- **Agreement analysis**: Identify where models agree (high confidence) vs disagree (ambiguous cases)

### Models Trained:
1. **FLAN-T5-small** (77M params) - Encoder-decoder, instruction-tuned T5
2. **DistilGPT2** (82M params) - Decoder-only, distilled from GPT-2

### By the end, you'll have:
- Two trained models with different architectures
- An ensemble comparison report showing agreement/disagreement patterns
- Insights into which examples are "easy" (full agreement) vs "hard" (disagreement)
- Understanding of inter-model agreement as a confidence metric
- A template for multi-model ensemble research

**Expected time to complete:** ~25 minutes training + 2 minutes comparison

---

## Prerequisites & Setup

### Step 0: Environment Preparation

#### 0.1 Install required packages

```bash
pip install --upgrade pip

# PyTorch (CPU version - memory-optimized)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Hugging Face ecosystem
pip install transformers[torch]
pip install datasets
pip install accelerate

# Utilities
pip install scikit-learn tqdm
```

**What each package does:**
- `torch` – Deep learning framework (CPU-only to avoid memory issues)
- `transformers` – Hugging Face models and training utilities
- `datasets` – Load and process JSONL files efficiently
- `accelerate` – Distributed training utilities (used by Trainer)
- `scikit-learn`, `tqdm` – Metrics and progress bars

**Installation time:** ~5–10 minutes on first run

#### 0.2 Verify installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
```

You should see version numbers. If you see errors, check that your virtual environment is activated.

#### 0.3 Verify dataset location

Ensure your `weird_machine_gadgets.jsonl` file is in the `data/` directory:

```bash
ls data/
# Should show: weird_machine_gadgets.jsonl
```

---

## Running the Script

Save the provided `main.py` script in your project root, then run:

```bash
# For macOS/Linux:
python main.py --platform unix

# For Windows:
python main.py --platform windows
```

### Skip Training (Load Existing Models)

If you've already trained the models and just want to re-run the comparison:

```bash
python main.py --platform unix --skip-training
```

---

## Step-by-Step Walkthrough

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

The script **forces CPU-only training** to prevent memory errors on Apple Silicon Macs. This is done by:
- Disabling CUDA (NVIDIA GPUs)
- Disabling MPS (Metal Performance Shaders - Apple GPU)
- Overriding PyTorch backend checks

**Why?**
- Apple Silicon unified memory is shared between CPU and GPU
- MPS backend can run out of memory with larger models (GPT-2 124M hit the limit)
- CPU-only is slower but **reliable** and fits in memory

---

### STEP 1: Platform Setup

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

### STEP 2: Loading and Preparing Data

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

1. **Load JSONL**: Reads all examples from dataset
2. **Add prompts**: Creates two prompt formats:
   - **Seq2seq prompt** (for FLAN-T5): `"Task: {instruction}\n\nExcerpt:\n{input}\n\nAnswer:"`
   - **Causal prompt** (for GPT-2): Includes the full output for next-token prediction
3. **Subsample**: Uses 100 examples (configurable via `TOTAL_EXAMPLES_TO_USE`)
4. **Split**: 90% train, 10% validation

**In the script:** `load_and_prepare_data()` function

**To adjust dataset size:**

```python
# Top of main.py (line ~51):
TOTAL_EXAMPLES_TO_USE = 100  # Change to 200, 500, etc.
```

---

### STEP 3: Training Model 1 - FLAN-T5-small

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
- **Encoder-decoder** (seq2seq)
- **Bidirectional encoder**: Can attend to full input context
- **Autoregressive decoder**: Generates output left-to-right
- **Instruction-tuned**: Pre-trained on instruction-following tasks

**Training time:** ~3-5 minutes on 8-core CPU

**In the script:** `train_seq2seq_model()` function

---

### STEP 4: Memory Cleanup

```
  Cleaning up memory after flan-t5-small...
  ✓ Memory freed
```

**What's happening:**

Between model training runs, the script explicitly:
- Deletes model from memory (`del model`)
- Runs garbage collection (`gc.collect()`)
- Empties CUDA cache if available

This prevents memory accumulation when training multiple models sequentially.

---

### STEP 5: Training Model 2 - DistilGPT2

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
2. **Set pad token**: GPT-2 doesn't have a pad token by default, so we set it to EOS token
3. **Tokenize**: Converts full prompt+output into token IDs (up to 768 tokens)
4. **Train**: 3 epochs with causal language modeling objective
5. **Save**: Final model saved to `checkpoints/distilgpt2/final_model`

**Architecture:**
- **Decoder-only** (causal LM)
- **Left-to-right attention**: Can only attend to previous tokens
- **Distilled from GPT-2**: Smaller, faster, 82M params (vs 124M for full GPT-2)
- **General pre-training**: Not instruction-tuned

**Training time:** ~3-5 minutes on 8-core CPU

**In the script:** `train_causal_model()` function

---

### STEP 6: Ensemble Comparison & Agreement Analysis

```
================================================================================
ENSEMBLE COMPARISON & AGREEMENT ANALYSIS
================================================================================

================================================================================
VALIDATION EXAMPLE 1/10
================================================================================

INSTRUCTION: Identify weird machine CONTROL-FLOW gadgets in the excerpt...
EXCERPT: Command enable logic uses a general enable contact...

  [flan-t5-small] Generating...
  Output: gadget_type: Command enable logic; location: Command enable logic...

  [distilgpt2] Generating...
  Output: gadget_type: Control-Flow gadget; location: Command enable contact...

────────────────────────────────────────────────────────────────────────────────
AGREEMENT ANALYSIS:
────────────────────────────────────────────────────────────────────────────────
  Full agreement: ✗ NO
  Unique gadget types: ['Command enable logic', 'Control-Flow gadget']
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

1. **Generate predictions**: Both models generate outputs for the same prompt
2. **Extract gadget types**: Parse `gadget_type:` from each prediction
3. **Normalize gadget types**: Convert to lowercase, remove spaces/hyphens/punctuation for comparison
4. **Check format**: Verify outputs contain required fields (gadget_type, location, explanation)
5. **Compute agreement**:
   - **Full agreement**: Normalized gadget types match → High confidence
   - **Disagreement**: Normalized types differ → Ambiguous or semantically different

**Agreement metrics:**
- `full_agreement`: Boolean - do normalized models agree?
- `unique_types`: List of distinct **original** gadget types (before normalization)
- `normalized_types`: Dict mapping model → normalized type (for debugging)
- `unique_normalized_types`: List of distinct normalized types
- `majority_type`: Most common prediction (with 2 models, this is a tie-breaker)
- `gadget_types`: Dict mapping model → predicted type (original)

**In the script:** `run_ensemble_comparison()` function

---

### STEP 7: Saving Comparison Report

```
================================================================================
SAVING COMPARISON REPORT
================================================================================
✓ Report saved to: ensemble_report.json

SUMMARY:
  Total examples: 10
  Full agreement: 1 (10.0%)
  Disagreements: 9 (90.0%)

  Format accuracy by model:
    - flan-t5-small: 20.0%
    - distilgpt2: 30.0%
```

**What's happening:**

The script saves a detailed JSON report with:

```json
{
  "summary": {
    "total_examples": 10,
    "full_agreements": 1,
    "full_agreement_rate": 0.1,
    "disagreement_rate": 0.9,
    "model_format_accuracy": {
      "flan-t5-small": 0.2,
      "distilgpt2": 0.3
    }
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
      "format_checks": {...},
      "agreement": {
        "full_agreement": false,
        "unique_types": ["Command enable logic", "Control-Flow gadget"],
        "gadget_types": {"flan-t5-small": "Command enable logic", "distilgpt2": "Control-Flow gadget"},
        "normalized_types": {"flan-t5-small": "commenablelogic", "distilgpt2": "controlflow"},
        "unique_normalized_types": ["commenablelogic", "controlflow"]
      }
    },
    ...
  ]
}
```

**In the script:** `save_comparison_report()` function

---

## Understanding the Report: `ensemble_report.json`

### Summary Metrics

```json
{
  "summary": {
    "total_examples": 10,
    "full_agreements": 1,
    "full_agreement_rate": 0.1,
    "disagreement_rate": 0.9,
    "model_format_accuracy": {
      "flan-t5-small": 0.2,
      "distilgpt2": 0.3
    }
  }
}
```

**Interpretation:**

| Metric | Meaning | What to look for |
|--------|---------|-----------------|
| `full_agreement_rate` | % of examples where normalized gadget types match | Higher = models converge on same interpretation |
| `disagreement_rate` | % of examples with different normalized predictions | Lower = more consistent ensemble |
| `model_format_accuracy` | % of outputs with correct format | Should be >80% for production use |

**With 100 examples trained:**
- Agreement rate 10-30% is typical (after normalization helps with spelling/caps)
- Format accuracy 20-40% is expected with minimal training
- High disagreements highlight genuinely ambiguous or semantically different interpretations

### Per-Example Results

Each result contains:
- `example_id`: Index in validation set
- `instruction`: Task description
- `excerpt`: Manual excerpt
- `gold_output`: Correct answer
- `predictions`: Dict of model → prediction
- `agreement`: Agreement analysis with **both original and normalized types**

**Use cases:**
1. **Find hard examples**: `agreement.full_agreement == false`
2. **Analyze model biases**: Which model is more accurate?
3. **Confidence scoring**: Full agreement → high confidence
4. **Debug normalization**: Compare `gadget_types` vs `normalized_types`

---

## Limitations of Simple String Normalization

### The Semantic Gap Problem

The current script uses **string normalization** (lowercase, remove punctuation, etc.) to handle minor variations like:

✅ **Works well for:**
- Capitalization: `"ReadWrite"` vs `"READWRITE"` → both normalize to `"readwrite"` ✓
- Hyphens: `"Control-Flow"` vs `"ControlFlow"` → both normalize to `"controlflow"` ✓  
- Spacing: `"IO and Side-Effect"` vs `"IO and SIDE-EFFECT"` → both normalize to `"iosideeffect"` ✓
- Typos: `"TimingSynchronization"` vs `"TimingSYNCHRONIZATION"` → both normalize to `"timingsynchronization"` ✓

❌ **Does NOT work for semantic differences:**

| Example | FLAN-T5 Output | DistilGPT2 Output | Issue |
|---------|----------------|-------------------|-------|
| Example 0 | `"Command enable logic"` | `"Control-Flow gadget"` | **Specific vs generic** - both refer to control flow, but one is about the specific command enable mechanism, the other is the category name |
| Example 2 | `"READWRITE gadget"` | `"BOOL tag"` | **Category vs implementation** - BOOL tag is a specific type of read/write mechanism |
| Example 5 | `"Control example Control block is a kind of function block that is TRUE when a mode bit is TRUE..."` | `"Control-Flow gadgets"` | **Garbled vs clean** - one model generated garbled repetitive text, the other gave a clean label |
| Example 6 | `"weird machine"` | `"COMPOS"` (truncated) | **Generic vs truncated** - one is too generic, the other appears cut off |

### Why This Matters

**String normalization cannot detect:**
1. **Semantic equivalence**: "Command enable logic" and "Control-Flow gadget" both describe control flow, but string matching sees them as different
2. **Abstraction levels**: "BOOL tag" is a specific instance of "ReadWrite gadget" (correct answer)
3. **Garbled output**: When a model outputs repetitive or nonsensical text, normalization won't flag it
4. **Partial correctness**: "IO, sIDE-EFFECT gadget" vs "IO and SIDE-EFFECT gadgets using Logix 5000 physical I" - one is concise, one is verbose but potentially more accurate

### The Need for Reasoning LLMs

To properly evaluate **semantic agreement** (not just string similarity), we need a reasoning LLM to:

1. **Judge semantic equivalence**:
   ```
   Question: Are "Command enable logic" and "Control-Flow gadget" semantically equivalent?
   
   Reasoning LLM: Both refer to control flow mechanisms. "Command enable logic" is more 
   specific, naming the exact feature (command enable), while "Control-Flow gadget" is 
   the category label. They are semantically related but differ in specificity.
   
   Verdict: PARTIAL AGREEMENT (related but different abstraction levels)
   ```

2. **Detect abstraction mismatches**:
   ```
   Question: Is "BOOL tag" equivalent to "ReadWrite gadget"?
   
   Reasoning LLM: "BOOL tag" is a specific implementation (a tag of type BOOL), while 
   "ReadWrite gadget" is the category it belongs to. This is like saying "Honda Civic" 
   vs "car" - one is correct but too specific, the other is the right category.
   
   Verdict: PARTIAL AGREEMENT (correct category but wrong specificity)
   ```

3. **Filter garbled outputs**:
   ```
   Question: Is "Control example Control block is a kind of function block that is TRUE 
   when a mode bit is TRUE. The function block is not executed in that cycle. The 
   explanation Identify the trick of a function block that is TRUE when a mode block 
   is TRUE." a valid gadget type?
   
   Reasoning LLM: This is garbled output. It's repetitive, contains multiple sentences, 
   and doesn't provide a clean category label. The model likely diverged during generation.
   
   Verdict: INVALID OUTPUT
   ```

4. **Score partial correctness**:
   - Full agreement: Exact semantic match (both say "ReadWrite gadget")
   - Partial agreement: Related concepts (one says "Command enable logic", other says "Control-Flow gadget")
   - Disagreement: Unrelated concepts (one says "ReadWrite", other says "Timing")
   - Invalid: Garbled or nonsensical output

### Future Enhancement: LLM-as-Judge Pipeline

**Phase 1: Current (String Normalization)**
```
Model A → "ReadWrite gadget"  ───┐
                                  ├─→ normalize() → compare strings → AGREEMENT ✓
Model B → "READWRITE gadget"  ───┘
```

**Phase 2: Future (LLM-as-Judge)**
```
Model A → "Command enable logic"  ───┐
                                      ├─→ Reasoning LLM (Claude, GPT-4, etc.)
Model B → "Control-Flow gadget"   ───┘     ↓
                                      semantic_similarity()
                                            ↓
                                      PARTIAL AGREEMENT (60% similar)
                                      Reason: "Both refer to control flow, but differ 
                                      in specificity..."
```

**Implementation sketch:**
```python
def llm_semantic_judge(prediction_a: str, prediction_b: str, gold: str) -> Dict:
    """
    Use a reasoning LLM (e.g., Claude 3.5 Sonnet, GPT-4) to judge semantic agreement.
    """
    prompt = f"""
    You are evaluating whether two model predictions are semantically equivalent.
    
    Gold standard: {gold}
    Model A: {prediction_a}
    Model B: {prediction_b}
    
    Are Model A and Model B saying the same thing? Consider:
    1. Semantic equivalence (different words, same meaning)
    2. Abstraction level (specific vs generic)
    3. Partial correctness (one is more accurate than the other)
    4. Garbled output (repetitive, nonsensical)
    
    Respond with:
    - FULL_AGREEMENT: Exact semantic match
    - PARTIAL_AGREEMENT: Related concepts but different specificity
    - DISAGREEMENT: Unrelated concepts
    - INVALID: One or both outputs are garbled
    
    Also provide a 1-sentence explanation.
    """
    
    # Call reasoning LLM API (Claude, GPT-4, etc.)
    response = reasoning_llm_api(prompt)
    
    return {
        "verdict": response.verdict,  # FULL_AGREEMENT, PARTIAL_AGREEMENT, etc.
        "explanation": response.explanation,
        "confidence": response.confidence,  # 0-1 score
    }
```

### Workaround for Now: Manual Inspection

Until we integrate a reasoning LLM, **manually inspect disagreements**:

1. **Filter by disagreement**:
   ```python
   import json
   
   with open('ensemble_report.json', 'r') as f:
       report = json.load(f)
   
   disagreements = [r for r in report['results'] if not r['agreement']['full_agreement']]
   
   for d in disagreements[:5]:  # Look at first 5
       print(f"\nExample {d['example_id']}:")
       print(f"  FLAN-T5: {d['agreement']['gadget_types']['flan-t5-small']}")
       print(f"  DistilGPT2: {d['agreement']['gadget_types']['distilgpt2']}")
       print(f"  Gold: {d['gold_output'][:60]}...")
       print(f"  → Are these semantically equivalent? (y/n/partial)")
   ```

2. **Create a manual annotation**:
   - Add a `"semantic_agreement"` field: `"full"`, `"partial"`, `"none"`, `"invalid"`
   - Track which disagreements are actually semantic matches
   - Measure: What % of "disagreements" are actually partial agreements?

3. **Use this data to train a classifier** (future):
   - Input: (prediction_a, prediction_b, gold_output)
   - Output: semantic_similarity_score (0-1)
   - Train on your manually annotated examples

---

### Summary: Normalization Helps, But Not Enough

| Problem | String Normalization | Reasoning LLM Needed? |
|---------|---------------------|---------------------|
| Capitalization (`ReadWrite` vs `READWRITE`) | ✓ Solves | ❌ Not needed |
| Typos (`Synchronization` vs `Syncronization`) | ✓ Solves (mostly) | ❌ Not needed |
| Semantic equivalence (`Command enable` vs `Control-Flow`) | ❌ Cannot detect | ✅ **Needed** |
| Abstraction mismatch (`BOOL tag` vs `ReadWrite gadget`) | ❌ Cannot detect | ✅ **Needed** |
| Garbled output | ❌ Cannot detect | ✅ **Needed** |
| Partial correctness | ❌ Cannot score | ✅ **Needed** |

**Bottom line:** With 100 training examples, expect **low format accuracy** (20-40%) and **high disagreement** (70-90%) even after normalization. Many "disagreements" are actually partial agreements that only a reasoning LLM can detect.

---

## Configuration & Tuning

### Key Configuration Variables

**At the top of `main.py` (lines 44-58):**

```python
# Dataset sizing
TOTAL_EXAMPLES_TO_USE = 100  # ← CHANGE THIS FOR MORE DATA
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

### Scaling Up

#### Phase 1: Increase Dataset Size

```python
TOTAL_EXAMPLES_TO_USE = 200  # or 300, 500
```

**Expected improvements:**
- Higher agreement rate (30-50%)
- Better format accuracy (50-70%)
- More stable predictions

**Training time:**
- 200 examples: ~10 min per model (20 min total)
- 500 examples: ~25 min per model (50 min total)

#### Phase 2: Increase Epochs

```python
NUM_EPOCHS = 5  # or 10
```

**Trade-off:**
- Better model quality (lower loss)
- Longer training time (2-3x)

#### Phase 3: Adjust Learning Rate

```python
LEARNING_RATE = 1e-5  # Lower for stability
# or
LEARNING_RATE = 1e-4  # Higher for faster convergence
```

**Rule of thumb:**
- If loss plateaus early → increase learning rate
- If loss is unstable → decrease learning rate

---

## Research Questions for Students

### 1. Agreement Pattern Analysis

**Question:** Which examples cause disagreement between models?

**How to explore:**
```python
import json

with open('ensemble_report.json', 'r') as f:
    report = json.load(f)

# Find disagreements
disagreements = [
    r for r in report['results']
    if not r['agreement']['full_agreement']
]

print(f"Found {len(disagreements)} disagreements")
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
- Are certain gadget types more ambiguous?
- Does technical jargon cause confusion?
- Do normalized types reveal that some "disagreements" are actually spelling variations?

---

### 2. Architectural Comparison

**Question:** Does seq2seq (FLAN-T5) outperform causal LM (DistilGPT2)?

**Metrics to compare:**
- Format accuracy (from report summary)
- Agreement with gold standard
- Training loss (final values)

**Hypothesis:**
- Instruction-tuned models (FLAN-T5) should have better format adherence
- Seq2seq models should handle longer contexts better

---

### 3. Confidence via Agreement

**Question:** Can we use inter-model agreement as a confidence score?

**Approach:**
1. Label validation examples:
   - High confidence = full agreement
   - Low confidence = disagreement
2. Manually inspect a sample of each category
3. Measure: Are "high confidence" predictions more accurate?

**Implementation:**
```python
# Separate by confidence
high_conf = [r for r in report['results'] if r['agreement']['full_agreement']]
low_conf = [r for r in report['results'] if not r['agreement']['full_agreement']]

print(f"High confidence: {len(high_conf)} examples")
print(f"Low confidence: {len(low_conf)} examples")

# Manual inspection: are high-conf predictions more accurate?
```

---

### 4. Error Analysis by Gadget Type

**Question:** Which gadget types are hardest to classify?

**Approach:**
```python
from collections import defaultdict

errors_by_type = defaultdict(list)

for r in report['results']:
    # Extract gold gadget type
    gold_type = r['gold_output'].split('gadget_type:')[1].split(';')[0].strip()
    
    # Check if models agreed and were correct
    if not r['agreement']['full_agreement']:
        errors_by_type[gold_type].append(r['example_id'])

for gtype, examples in errors_by_type.items():
    print(f"{gtype}: {len(examples)} disagreements")
```

---

### 5. Majority Voting Performance

**Question:** Does ensemble voting improve accuracy?

**Approach:**
1. For each example, take the majority vote (with 2 models, this is tie-breaking)
2. Compare majority vote vs individual model accuracy
3. Measure improvement

**With 3+ models:** Majority voting becomes more powerful

---

### 6. Semantic Agreement Analysis (Manual)

**Question:** How many "disagreements" are actually semantic agreements?

**Approach:**
1. Sample 20 disagreement cases
2. Manually label each as:
   - `"full_agreement"`: Semantically identical (e.g., "ReadWrite" vs "READWRITE")
   - `"partial_agreement"`: Related but different specificity (e.g., "BOOL tag" vs "ReadWrite gadget")
   - `"disagreement"`: Truly different concepts
   - `"invalid"`: Garbled output
3. Calculate: What % of "disagreements" are actually partial/full agreements?

**This gives you a baseline for how much a reasoning LLM could improve agreement rate.**

---

## Troubleshooting

### Issue: `RuntimeError: MPS backend out of memory`

**Solution:** The script should already force CPU-only. Verify by checking the output:

```
✓ MPS disabled (Apple Silicon GPU)
✓ All training will use CPU only
```

If you still see this error, ensure the memory optimization layer is at the very top of `main.py` (lines 22-38).

---

### Issue: `TypeError: transformers.generation.utils.GenerationMixin.generate() argument after ** must be a mapping`

**Solution:** Fixed in the provided script. The issue was unpacking `inputs["input_ids"]` instead of `inputs`. Now uses:

```python
outputs = model.generate(
    inputs["input_ids"],  # Direct tensor
    attention_mask=inputs.get("attention_mask"),
    ...
)
```

---

### Issue: Training too slow on CPU

**This is normal.** To speed up:

1. **Reduce dataset size temporarily:**
   ```python
   TOTAL_EXAMPLES_TO_USE = 50
   ```

2. **Reduce sequence lengths:**
   ```python
   MAX_INPUT_LENGTH = 256
   MAX_TARGET_LENGTH = 128
   ```

3. **Use fewer epochs:**
   ```python
   NUM_EPOCHS = 2
   ```

4. **Close other applications** to free CPU cores

---

### Issue: Model outputs are gibberish

**Causes:**
- Not enough training data (100 examples is minimal)
- Learning rate too high

**Solutions:**
1. Increase `TOTAL_EXAMPLES_TO_USE` to 200+
2. Lower `LEARNING_RATE` to `1e-5`
3. Increase `NUM_EPOCHS` to 5

---

### Issue: Low agreement rate (<20%)

**This is expected with 100 examples and string-only comparison.**

**To improve:**
1. Scale to 500-1000 examples (improves model quality)
2. Increase epochs to 5-10 (improves convergence)
3. Implement LLM-as-judge (detects semantic equivalence)
4. Use a larger model (flan-t5-base instead of flan-t5-small)

---

### Issue: High disagreement rate but predictions look similar

**This is the semantic gap problem!**

Many "disagreements" are actually:
- Spelling variations: `"TimingSynchronization"` vs `"TimingSYNCHRONIZATION"`
- Capitalization: `"ReadWrite"` vs `"READWRITE"`
- Semantic equivalence: `"Command enable logic"` vs `"Control-Flow gadget"`

**Solution:** See the "Limitations of Simple String Normalization" section above. You'll need a reasoning LLM to properly evaluate these.

---

## Advanced Experiments

### Experiment 1: Add a Third Model

Edit the `MODELS` dict (line ~52):

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

**Trade-off:** Slower training (250M params) but potentially better quality and more interesting 3-way comparisons.

---

### Experiment 2: Custom Inference Script

Load trained models for your own excerpts:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

# Load FLAN-T5
t5_tokenizer = AutoTokenizer.from_pretrained('checkpoints/flan-t5-small/final_model')
t5_model = AutoModelForSeq2SeqLM.from_pretrained('checkpoints/flan-t5-small/final_model')

# Load DistilGPT2
gpt_tokenizer = AutoTokenizer.from_pretrained('checkpoints/distilgpt2/final_model')
gpt_model = AutoModelForCausalLM.from_pretrained('checkpoints/distilgpt2/final_model')

# Your custom excerpt
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

### Experiment 3: Measure ROUGE Scores

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

### Experiment 4: Implement LLM-as-Judge (Advanced)

Use a reasoning LLM (Claude, GPT-4) to judge semantic agreement:

```python
import anthropic  # or openai

def llm_judge_semantic_agreement(pred_a, pred_b, gold):
    """Use Claude to judge if two predictions are semantically equivalent."""
    client = anthropic.Anthropic(api_key="your-key")
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""
Are these two model predictions semantically equivalent?

Gold: {gold}
Model A: {pred_a}
Model B: {pred_b}

Respond with:
- FULL_AGREEMENT: Exact semantic match
- PARTIAL_AGREEMENT: Related but different specificity
- DISAGREEMENT: Unrelated concepts
- INVALID: Garbled output

Also provide a 1-sentence explanation.
"""
        }]
    )
    
    response_text = message.content[0].text
    
    # Parse verdict
    if "FULL_AGREEMENT" in response_text:
        verdict = "full"
    elif "PARTIAL_AGREEMENT" in response_text:
        verdict = "partial"
    elif "INVALID" in response_text:
        verdict = "invalid"
    else:
        verdict = "disagreement"
    
    return {
        "verdict": verdict,
        "explanation": response_text
    }

# Apply to all disagreements
with open('ensemble_report.json', 'r') as f:
    report = json.load(f)

for result in report['results']:
    if not result['agreement']['full_agreement']:
        pred_a = result['predictions']['flan-t5-small']
        pred_b = result['predictions']['distilgpt2']
        gold = result['gold_output']
        
        llm_judgment = llm_judge_semantic_agreement(pred_a, pred_b, gold)
        result['llm_judgment'] = llm_judgment
        
        print(f"Example {result['example_id']}: {llm_judgment['verdict']}")

# Save updated report
with open('ensemble_report_with_llm_judge.json', 'w') as f:
    json.dump(report, f, indent=2)
```

**This allows you to measure:**
- True agreement rate (including partial agreements)
- Which disagreements are actually semantic matches
- Model biases (does one model consistently output at wrong abstraction level?)

---

## Key Takeaways

1. **Ensemble diversity matters**: Different architectures (seq2seq vs causal) provide different perspectives
2. **Agreement = confidence**: Full agreement suggests high-confidence predictions
3. **Disagreement ≠ always wrong**: Many disagreements are semantic equivalents that string matching misses
4. **Normalization helps but isn't enough**: Handles spelling/caps but not semantic gaps
5. **Reasoning LLMs needed**: To properly judge semantic agreement, abstraction levels, and garbled output
6. **Memory-optimized training**: CPU-only is viable for small models (77-82M params)
7. **Iterative scaling**: Start at 100 examples, verify pipeline, scale to 500-1000
8. **Format adherence**: Instruction-tuned models (FLAN-T5) often have better structured outputs

---

## Resources

- **Hugging Face Transformers**: https://huggingface.co/docs/transformers/
- **FLAN-T5 model card**: https://huggingface.co/google/flan-t5-small
- **DistilGPT2 model card**: https://huggingface.co/distilgpt2
- **Datasets library**: https://huggingface.co/docs/datasets/
- **PyTorch tutorials**: https://pytorch.org/tutorials/
- **LLM-as-Judge research**: [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)

---

## Summary Checklist

**Before you start:**
- [ ] Dependencies installed (`torch`, `transformers`, `datasets`, etc.)
- [ ] `data/weird_machine_gadgets.jsonl` exists
- [ ] `main.py` saved in project root
- [ ] ~5 GB free disk space
- [ ] 8+ GB RAM (for CPU training)

**After training:**
- [ ] Both models trained successfully (check `checkpoints/` directories)
- [ ] `ensemble_report.json` generated
- [ ] Review agreement rate and format accuracy
- [ ] Inspect 2-3 disagreement examples manually
- [ ] Check `normalized_types` to see if disagreements are just spelling variations
- [ ] Identify which disagreements are semantic matches (need LLM judge)
- [ ] Plan next experiment (scale to 200? add 3rd model? implement LLM judge?)

---

**Happy ensemble training!** 🚀

For questions or issues, refer to the troubleshooting section or check the Hugging Face documentation.
