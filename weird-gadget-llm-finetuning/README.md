# Multi-Model Ensemble Training for Weird Machine Gadget Classification

**Universal hardware support: NVIDIA GPUs, Apple Silicon, CPU-only**

---

## Introduction

Fine-tune **two small language models** on weird machine gadget identification and compare their predictions through ensemble agreement analysis.

**Why 2 models?**
- **Architectural diversity**: seq2seq (encoder-decoder) vs causal (decoder-only)
- **Training paradigm diversity**: Instruction-tuned vs general pre-training
- **Agreement analysis**: Identify high-confidence (agreement) vs ambiguous (disagreement) cases

### Models

1. **FLAN-T5-small** (77M params) – Instruction-tuned encoder-decoder
2. **DistilGPT2** (82M params) – General-purpose decoder-only
3. **Phi-2** (2.7B params, optional) – Semantic judge for disagreement resolution

### What you'll get

- Two trained models with different architectures
- String-level agreement report (`ensemble_report.json`)
- Optional semantic agreement report with Phi-2 judge (`ensemble_report_with_llm_judge.json`)
- Understanding of ensemble confidence scoring

**Time:** ~20–30 minutes (100 examples on CPU) or ~5–10 minutes (GPU)

---

## Setup

### Install dependencies

```bash
pip install --upgrade pip

# PyTorch (CPU version - works everywhere)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or for NVIDIA GPU support:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core packages
pip install "transformers[torch]" datasets accelerate scikit-learn tqdm
```

### Verify setup

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Verify data

Ensure `weird_machine_gadgets.jsonl` is in the `data/` directory:

```bash
ls data/weird_machine_gadgets.jsonl
```

---

## Running the script

### Basic usage (auto-detects hardware)

```bash
# Windows
python fine-tune-llms.py --platform windows

# macOS/Linux
python fine-tune-llms.py --platform unix
```

The script automatically detects and uses:
1. **CUDA GPU** (NVIDIA) – if available
2. **MPS** (Apple Silicon GPU) – if available
3. **CPU** – fallback for any hardware

### Common options

```bash
# Force CPU (useful for memory-constrained systems)
python fine-tune-llms.py --platform unix --force_cpu

# Skip training, load existing models
python fine-tune-llms.py --platform unix --skip_training

# Enable Phi-2 semantic judge
python fine-tune-llms.py --platform unix --use_judge

# Combine options
python fine-tune-llms.py --platform unix --skip_training --use_judge
```

---

## How it works

### 1. Hardware detection

```
================================================================================
DEVICE SETUP (AUTO-DETECT)
================================================================================
✓ CUDA GPU detected: NVIDIA RTX 3060
  VRAM: 12.0 GB
  Using GPU for training
================================================================================
```

Or on CPU-only systems:
```
✓ No GPU detected
  Using CPU for training
  Training will be slower but will work on any hardware
```

### 2. Data loading

- Loads `weird_machine_gadgets.jsonl`
- Subsamples to 100 examples (configurable)
- Splits 90% train / 10% validation
- Creates two prompt formats (seq2seq and causal)

### 3. Training (2 models)

**FLAN-T5-small** (77M params)
- Encoder-decoder architecture
- Instruction-tuned → better format adherence
- Training time: ~3–5 min (CPU) or ~1–2 min (GPU)

**DistilGPT2** (82M params)
- Decoder-only architecture
- General pre-training
- Training time: ~3–5 min (CPU) or ~1–2 min (GPU)

### 4. Ensemble comparison

For each validation example:

1. **Generate predictions** from both models
2. **Extract gadget types** using regex
3. **Normalize** (lowercase, remove "gadget", strip punctuation)
4. **Check format** (has `gadget_type`, `location`, `explanation`?)
5. **Compute agreement**:
   - Full agreement = normalized gadget types match
   - Disagreement = normalized types differ

### 5. Optional: Phi-2 semantic judge

With `--use_judge`, disagreements are sent to Phi-2 for semantic evaluation:

- **FULL_AGREEMENT**: Semantically identical
- **PARTIAL_AGREEMENT**: Related but different specificity
- **DISAGREEMENT**: Truly different concepts
- **INVALID**: Garbled or nonsensical output

---

## Understanding the reports

### String-level report (`ensemble_report.json`)

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
    }
  }
}
```

**Key metrics:**
- `full_agreement_rate`: % where normalized gadget types match (higher = more consistent)
- `model_format_accuracy`: % of outputs with correct format

**Typical results (100 examples):**
- String agreement: 30–60%
- FLAN-T5 format accuracy: 80–100%
- DistilGPT2 format accuracy: 30–60%

### Semantic report (`ensemble_report_with_llm_judge.json`)

With `--use_judge`, you also get:

```json
{
  "summary": {
    "semantic_agreements": 8,
    "semantic_agreement_rate": 0.8,
    ...
  }
}
```

**Improvement:** Phi-2 typically increases agreement rate by +10–20% by detecting:
- Capitalization differences ("ReadWrite" vs "READWRITE")
- Abstraction mismatches ("BOOL tag" vs "Read/Write gadget")
- Truncated but valid labels

---

## Why string normalization isn't enough

String normalization handles:
✅ Capitalization: `"ReadWrite"` vs `"READWRITE"` → both `"readwrite"`
✅ Hyphens: `"Control-Flow"` vs `"ControlFlow"` → both `"controlflow"`

But **cannot** detect:
❌ Semantic equivalence: `"Command enable logic"` vs `"Control-Flow gadget"`
❌ Abstraction levels: `"BOOL tag"` (specific) vs `"Read/Write gadget"` (category)
❌ Garbled output: `"Control example Control block is a kind..."`

**Solution:** Phi-2 semantic judge (`--use_judge`) bridges this gap.

---

## Configuration

Edit constants at the top of `fine-tune-llms.py`:

```python
# Dataset sizing
TOTAL_EXAMPLES_TO_USE = 100  # Increase to 200, 500, etc.
EVAL_SPLIT = 0.1             # 10% validation

# Training hyperparameters
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
```

**To scale up:**
- More data: `TOTAL_EXAMPLES_TO_USE = 500` → better accuracy
- More epochs: `NUM_EPOCHS = 5` → lower loss
- Adjust learning rate: `1e-5` (stable) or `1e-4` (faster)

---

## Research questions for students

### 1. Agreement pattern analysis

**Question:** Which examples cause disagreement?

```python
import json

with open('ensemble_report.json', 'r') as f:
    report = json.load(f)

disagreements = [r for r in report['results'] 
                 if not r['agreement']['full_agreement']]

for d in disagreements[:5]:
    print(f"Example {d['example_id']}:")
    print(f"  FLAN-T5: {d['agreement']['gadget_types']['flan-t5-small']}")
    print(f"  DistilGPT2: {d['agreement']['gadget_types']['distilgpt2']}")
    print(f"  Normalized T5: {d['agreement']['normalized_types']['flan-t5-small']}")
    print(f"  Normalized GPT2: {d['agreement']['normalized_types']['distilgpt2']}")
```

**Hypotheses:**
- Do disagreements correlate with excerpt length?
- Are certain gadget types more ambiguous?
- Does normalization reveal spelling/capitalization issues?

---

### 2. Architectural comparison

**Question:** Does instruction-tuned seq2seq outperform general causal LM?

**Compare:**
- Format accuracy (from `model_format_accuracy`)
- Training loss (final values in console output)
- Qualitative inspection of predictions

**Hypothesis:** FLAN-T5 should have better format adherence.

---

### 3. Confidence via agreement

**Question:** Are high-agreement predictions more accurate?

```python
high_conf = [r for r in report['results'] 
             if r['agreement']['full_agreement']]
low_conf = [r for r in report['results'] 
            if not r['agreement']['full_agreement']]

print(f"High confidence: {len(high_conf)} examples")
print(f"Low confidence: {len(low_conf)} examples")
# Manually inspect accuracy in each group
```

---

### 4. Error analysis by gadget type

**Question:** Which gadget types are hardest to classify?

```python
from collections import defaultdict
import re

errors_by_type = defaultdict(list)

for r in report['results']:
    m = re.search(r"gadget_type:\s*([^;\n]+)", r['gold_output'])
    if m and not r['agreement']['full_agreement']:
        gold_type = m.group(1).strip()
        errors_by_type[gold_type].append(r['example_id'])

for gtype, examples in errors_by_type.items():
    print(f"{gtype}: {len(examples)} disagreements")
```

---

### 5. Semantic agreement analysis

**Question:** How much do semantic judgments improve agreement?

**Requires:** Run with `--use_judge`

```python
with open('ensemble_report_with_llm_judge.json', 'r') as f:
    report = json.load(f)

upgraded = [
    r for r in report['results']
    if (not r['agreement']['full_agreement'])
    and r['agreement'].get('semantic_agreement', False)
]

print(f"String disagreements → Semantic agreements: {len(upgraded)}")
for r in upgraded[:3]:
    j = r['agreement']['llm_judgment']
    print(f"\nExample {r['example_id']}:")
    print(f"  Verdict: {j['verdict']}")
    print(f"  Explanation: {j['explanation'][:100]}...")
```

**Discussion:**
- Do you agree with Phi-2's verdicts?
- Are there false positives (should be DISAGREEMENT)?
- What % improvement does semantic judging provide?

---

### 6. Effect of the reasoning judge

**Question:** Does Phi-2 mostly fix string quirks or semantic gaps?

**Analyze Phi-2 verdicts:**
- `FULL_AGREEMENT`: Exact semantic match (likely string variation)
- `PARTIAL_AGREEMENT`: Related concepts, different specificity
- `DISAGREEMENT`: Truly different
- `INVALID`: Garbled output

**Example analysis:**
```python
verdicts = {}
for r in report['results']:
    if 'llm_judgment' in r['agreement']:
        v = r['agreement']['llm_judgment']['verdict']
        verdicts[v] = verdicts.get(v, 0) + 1

print(verdicts)
# e.g., {'FULL_AGREEMENT': 1, 'PARTIAL_AGREEMENT': 2, 'DISAGREEMENT': 1}
```

---

## Troubleshooting

### Out of memory (GPU)

**Symptom:** `CUDA out of memory` or `MPS backend out of memory`

**Solution:**
```bash
python fine-tune-llms.py --platform unix --force_cpu
```

Or reduce dataset size:
```python
TOTAL_EXAMPLES_TO_USE = 50  # in script
```

---

### Training too slow

**This is normal on CPU.** To speed up:

1. **Reduce dataset:** `TOTAL_EXAMPLES_TO_USE = 50`
2. **Fewer epochs:** `NUM_EPOCHS = 2`
3. **Shorter sequences:** `MAX_INPUT_LENGTH = 256`
4. **Close other apps** to free CPU cores

**Expected times (100 examples):**
- GPU: ~5–10 minutes total
- CPU (8 cores): ~20–30 minutes total
- CPU (4 cores): ~40–60 minutes total

---

### Gibberish outputs

**Causes:**
- Too little training data (100 is minimal)
- Learning rate too high
- Model diverged

**Fixes:**
1. Increase to 200+ examples
2. Lower `LEARNING_RATE = 1e-5`
3. Increase `NUM_EPOCHS = 5`

---

### Low agreement rate (<20%)

**Expected with 100 examples and string-only comparison.**

**Improvements:**
1. Scale to 500–1000 examples
2. Increase epochs to 5–10
3. Use `--use_judge` for semantic evaluation
4. Upgrade to `flan-t5-base` (250M params) for better quality

---

### Windows multiprocessing errors

**The script already disables multiprocessing on Windows.** If you still see errors:

```python
# In the script, verify:
dataloader_num_workers = 0  # for Windows
```

This is handled automatically by `setup_platform()`.

---

## Advanced experiments

### Add a third model

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

**Trade-off:** Slower (~15 min on CPU) but better accuracy and 3-way voting.

---

### Custom inference

Use your trained models on new excerpts:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5
tokenizer = AutoTokenizer.from_pretrained('checkpoints/flan-t5-small/final_model')
model = AutoModelForSeq2SeqLM.from_pretrained('checkpoints/flan-t5-small/final_model')

# Your custom input
instruction = "Identify weird machine ARITHMETIC/COMPUTATION gadgets..."
excerpt = "The ADD instruction adds two integer values..."
prompt = f"Task: {instruction}\n\nExcerpt:\n{excerpt}\n\nAnswer:"

# Generate
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(prediction)
```

---

### Measure ROUGE scores

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
        score = scorer.score(r['gold_output'], r['predictions'][model_key])
        scores.append(score['rougeL'].fmeasure)

    print(f"{model_key} ROUGE-L: {sum(scores)/len(scores):.3f}")
```

---

## Key takeaways

1. **Hardware flexibility**: Works on any system (GPU or CPU)
2. **String normalization**: Handles capitalization/spacing but not semantic gaps
3. **Semantic judge adds value**: Phi-2 detects abstraction mismatches and partial correctness
4. **Agreement = confidence**: Full agreement suggests reliable predictions
5. **Instruction-tuning matters**: FLAN-T5 has better format adherence than DistilGPT2
6. **Scale iteratively**: Start at 100 examples, verify pipeline, scale to 500+

---

## Quick reference

| Task | Command |
|------|---------|
| Train + compare (auto-detect hardware) | `python fine-tune-llms.py --platform unix` |
| Force CPU (low memory) | `python fine-tune-llms.py --platform unix --force_cpu` |
| Skip training, compare only | `python fine-tune-llms.py --platform unix --skip_training` |
| Enable Phi-2 semantic judge | `python fine-tune-llms.py --platform unix --use_judge` |
| Windows users | `python fine-tune-llms.py --platform windows` |

---

## Checklist

**Before starting:**
- [ ] Dependencies installed (`torch`, `transformers`, `datasets`, etc.)
- [ ] `data/weird_machine_gadgets.jsonl` exists
- [ ] `fine-tune-llms.py` in project root
- [ ] ~5 GB free disk space
- [ ] 8+ GB RAM

**After running:**
- [ ] Models trained (check `checkpoints/` directories)
- [ ] `ensemble_report.json` generated
- [ ] Review agreement rate and format accuracy
- [ ] Inspect 2–3 disagreement examples
- [ ] Check if normalization resolved spelling issues
- [ ] (Optional) Run with `--use_judge` for semantic analysis
- [ ] Plan next experiment (scale data? add model? tune hyperparameters?)

---

**Happy ensemble training!** 🚀
