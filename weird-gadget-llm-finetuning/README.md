# Fine-Tuning a Large Language Model to Identify Weird Machine Gadgets

### Introduction

In this guide, you'll learn how to fine-tune a small language model (`google/flan-t5-small`) to identify weird machine gadgets. 

By the end, you'll have:
- A trained model that can read a gadget description and identify its type, location, and explanation
- An understanding of the data preparation → tokenization → training → evaluation pipeline.
- Experience with Hugging Face's popular ecosystem tools
- A template you can adapt for other classification or generation tasks

---

## Prerequisites & Setup

### Step 0: Environment Preparation

pip install --upgrade pip

# PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Hugging Face ecosystem
pip install transformers[torch]
pip install datasets
pip install accelerate

# Optional (for future LoRA experiments)
pip install peft bitsandbytes

# Utilities
pip install scikit-learn tqdm
```

**What each package does:**
- `torch` – Deep learning framework
- `transformers` – Hugging Face models and training utilities
- `datasets` – Load and process JSONL files efficiently
- `accelerate` – Distributed training utilities (used by Trainer)
- `peft`, `bitsandbytes` – Parameter-efficient fine-tuning (LoRA, QLoRA)
- `scikit-learn`, `tqdm` – Metrics and progress bars

#### 0.4 Verify installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
```

You should see version numbers. If you see errors, check that your virtual environment is activated.


---

## Running the Script

Save the provided `fine-tune-llms.py` script in your project root, then run:

```bash
python fine-tune-llms.py
```

Below, we'll walk through what happens at each step.

---

## Step-by-Step Walkthrough

### STEP 1: Checking the Dataset File

```
================================================================================
STEP 1: Checking dataset file...
================================================================================
✓ Found data/weird_machine_gadgets.jsonl
```

**What's happening:**

The script checks that your JSONL file exists and counts the number of examples. Each line in the JSONL file is a JSON object with:
- `instruction`: Task description (e.g., "Identify weird machine ARITHMETIC/COMPUTATION gadgets...")
- `input`: An excerpt from a manual or specification
- `output`: The desired model output (gadget_type, location, explanation)
- `manual_name`, `manual_url`: Metadata

**In the script:** Lines 35–50 in `fine_tune_gadgets.py`

**What to look for:**
- Make sure the line count matches your expectations
- If you get "FileNotFoundError", double-check that `data/weird_machine_gadgets.jsonl` exists

---

### STEP 2: Device Information

```
================================================================================
DEVICE INFORMATION
================================================================================
✓ Running on CPU (as expected)
✓ CPU cores available: 8
✓ Using 2 processes for data loading
✓ PyTorch version: 2.2.2
```

**What's happening:**

The script checks which device will run training. For this guide, we've forced CPU usage because it's reliable and most laptops have enough CPU cores for reasonable performance.

Key metrics:
- **CPU cores available**: Used for parallel data loading (script uses 2 to be conservative)
- **PyTorch version**: Should be 2.0+

**In the script:** Lines 80–102

**Why CPU?**
- CPU training is slower but:
  - Works on any laptop without GPU
  - No GPU memory constraints (can fit larger batches on newer hardware)
  - Easier to debug
  - Reproducible across systems

---

### STEP 3: Loading the Dataset

```
================================================================================
STEP 3: Loading dataset from JSONL...
================================================================================
  Loading...
✓ Loaded examples

 Example 0:
  instruction: Identify weird machine ARITHMETIC/COMPUTATION gadgets in the excerpt...
  input: The S7-1200 basic instructions include ADD blocks...
  output: gadget_type: Arithmetic/Computation gadget; location: ADD instruction blocks...
```

**What's happening:**

The script uses the `datasets` library to load your JSONL file into memory. This library is efficient and handles large files well.

The first example is printed so you can inspect the data structure.

**In the script:** Lines 114–131

**Things to check:**
- Do the examples look reasonable?
- Is the output format consistent (gadget_type; location; explanation)?
- Does the instruction describe what you want the model to learn?

---

### STEP 4: Creating Prompts and Subsampling

```
================================================================================
STEP 4: Creating prompts and subsampling...
================================================================================
  Creating prompts...
✓ Added prompt field to all 100 examples
✓ Subsampled to 100 examples for training

 Example prompt:
Task: Identify weird machine COMMUNICATION-BRIDGE gadgets...
Excerpt: The document describes how devices may implement vendor-specific scaling...
Answer:

 Example output:
gadget_type: Communication-Bridge gadget; location: Vendor-defined integer scaling...
```

**What's happening:**

1. **Prompt creation**: The script combines `instruction` and `input` into a single prompt that will be fed to the model. The prompt template is:

   ```
   Task: {instruction}
   
   Excerpt:
   {input}
   
   Answer:
   ```

   This tells the model: "Here's a task. Here's an excerpt. Now generate an answer."

2. **Subsampling**: For faster iteration on your laptop, the script subsamples the data. This is controlled by:

   ```python
   TOTAL_EXAMPLES_TO_USE = 100  # Change this to 200, 500, etc.
   ```

**In the script:** Lines 143–164

**How to adjust for more data:**

```python
# At the top of the script (around line 26):
TOTAL_EXAMPLES_TO_USE = 100  # ← Change this

# Options:
TOTAL_EXAMPLES_TO_USE = 50    # Very fast (~5 min per epoch), lower quality
TOTAL_EXAMPLES_TO_USE = 200   # Medium (~20 min per epoch)
TOTAL_EXAMPLES_TO_USE = 500   # Slow (~50 min per epoch), better quality
TOTAL_EXAMPLES_TO_USE = 1000  # Very slow (~2 hours per epoch), best quality
```

After you verify the pipeline works at 100 examples, try 200, then 500.

---

### STEP 5: Train/Validation Split

```
================================================================================
STEP 5: Splitting into train/validation...
================================================================================
✓ Train: 90 examples
✓ Validation: 10 examples
```

**What's happening:**

The dataset is split into:
- **Train set (90%)**: Used to update the model's weights during training
- **Validation set (10%)**: Used to evaluate the model during training (not used to update weights)

This prevents **overfitting**: the model memorizing the training data without learning generalizable patterns.

**In the script:** Lines 177–184

**Standard ML practice:**
- Train: 80–90%
- Validation: 10–20%
- Test: (held out, we use validation as a proxy)

---

### STEP 6: Loading Model and Tokenizer

```
================================================================================
STEP 6: Loading model and tokenizer...
================================================================================
  Model: google/flan-t5-small
  Max input length: 512
  Max target length: 256
✓ Model loaded: T5ForConditionalGeneration
✓ Tokenizer loaded: T5TokenizerFast
  Model parameters: 76,961,152
```

**What's happening:**

1. **Tokenizer**: Converts text into numbers. FLAN-T5 uses a SentencePiece tokenizer that breaks text into subword tokens (~30K vocab).

2. **Model**: Downloads `google/flan-t5-small` from Hugging Face Hub (~250 MB). This is a **conditional generation** model (encoder-decoder):
   - **Encoder**: Reads the input prompt and builds a context representation
   - **Decoder**: Generates the output (gadget description) token-by-token

3. **Model size**: 77M parameters (small enough for a laptop)

**In the script:** Lines 197–209

**First run only:**
- Model download takes 2–5 minutes
- Subsequent runs use the cached version
- Cache location: `~/.cache/huggingface/hub/`

**Understanding the architecture:**

```
Input: "Task: Identify gadgets...Excerpt: The MMXU node..."
       ↓
    Tokenizer (text → tokens)
       ↓
    Encoder (T5 transformer)
       ↓
    Context representation (vector)
       ↓
    Decoder (generates output token-by-token)
       ↓
    Output: "gadget_type: Read/Write gadget; location: MMXU..."
```

---

### STEP 7: Tokenizing Datasets

```
================================================================================
STEP 7: Tokenizing datasets...
================================================================================
  Tokenizing train dataset...
  Tokenizing validation dataset...
✓ Tokenized train: 90 examples
✓ Tokenized validation: 10 examples
```

**What's happening:**

Each example is converted:
- **Input prompt** → token IDs (up to 512 tokens)
- **Output text** → token IDs (up to 256 tokens)

Longer texts are truncated; shorter texts are padded with a special token.

**In the script:** Lines 222–262

**Customization:**

```python
MAX_INPUT_LENGTH = 512    # ← Change if excerpts are too long
MAX_TARGET_LENGTH = 256   # ← Change if outputs exceed this
```

**If you see truncation warnings:** Increase the max lengths, but keep in mind that longer sequences use more memory and slow down training.

---

### STEP 8: Training Arguments

```
================================================================================
STEP 8: Setting up training arguments (CPU-optimized)...
================================================================================
✓ Training arguments configured:
  Output directory: checkpoints/flan-t5-small-gadgets
  Train batch size: 1
  Gradient accumulation steps: 4
  Effective batch size: 4
  Eval batch size: 1
  Learning rate: 5e-05
  Epochs: 3
  Device: CPU (no GPU)
```

**What's happening:**

The `Seq2SeqTrainingArguments` object configures how the model will be trained:

- **Batch size**: How many examples to process before updating weights
- **Gradient accumulation**: Process 1 example at a time, accumulate gradients from 4 steps, then update (simulates batch_size=4)
- **Learning rate**: How big a step to take when updating weights
- **Epochs**: How many times to loop through the entire training set
- **Evaluation strategy**: Evaluate every `eval_steps` steps

**In the script:** Lines 275–310

**To scale up (after verifying the pipeline):**

```python
# Around line 20–26:
TRAIN_BATCH_SIZE = 1              # Keep at 1 for CPU
GRADIENT_ACCUMULATION_STEPS = 4   # Or try 8 for larger effective batch
LEARNING_RATE = 5e-5              # Try 1e-4 or 1e-5
NUM_EPOCHS = 3                    # Try 5 or 10
```

**Understanding batch size on CPU:**
- Batch size 1 = update weights after each example (noisy but fast)
- Gradient accumulation = compute gradients for 4 examples, then update once (smoother, similar speed)
- On CPU, keep batch size 1; accumulate 4–8 steps

---

### STEP 9: Data Collator

```
================================================================================
STEP 9: Creating data collator...
================================================================================
✓ Data collator created (pads to multiples of 8 for CPU efficiency)
```

**What's happening:**

The data collator is a helper that:
1. **Batches examples**: Takes individual tokenized examples and groups them
2. **Pads to same length**: Adds padding tokens so all examples in a batch have the same length
3. **CPU optimization**: Pads to multiples of 8 (CPU vector operations are optimized for 8-element chunks)

**In the script:** Lines 313–318

---

### STEP 10: Training

```
================================================================================
STEP 10: Creating Trainer and fine-tuning...
================================================================================
 This will take a few minutes. Please wait...

  Starting training...
{'loss': 4.8273, 'grad_norm': 5.3005, 'learning_rate': 5e-06, 'epoch': 0.09}
{'loss': 4.3531, 'grad_norm': 3.5569, 'learning_rate': 1.5e-05, 'epoch': 0.18}
...
{'train_runtime': 116.2788, 'train_samples_per_second': 2.322, 'train_steps_per_second': 0.593, 'train_loss': 3.8269, 'epoch': 3.0}

✓ Training complete!
  Final train loss: 3.8269
```

**What's happening:**

This is where the actual learning occurs. For each epoch:

1. **Forward pass**: Feed a batch of prompts through the model to generate predictions
2. **Compute loss**: Measure how far predictions are from the gold outputs
3. **Backward pass**: Compute gradients (how much to adjust each weight)
4. **Update weights**: Adjust weights using the learning rate

**Reading the logs:**

- `loss`: Lower is better. You should see it decrease as training progresses.
- `grad_norm`: Magnitude of gradients. Usually 1–10 is healthy.
- `learning_rate`: Decreases over time (scheduled warmup and decay)
- `epoch`: Current progress (0.09 = 9% through epoch 1)

**In the script:** Lines 320–328

**Monitor these metrics:**
- **Loss should decrease**: 4.8 → 3.8 is good (27% improvement)
- **If loss increases**: Learning rate might be too high; try `LEARNING_RATE = 1e-5`
- **If loss plateaus**: Increase `NUM_EPOCHS` or `TOTAL_EXAMPLES_TO_USE`

**On a CPU laptop:**
- ~100 examples × 3 epochs ≈ 2 minutes total
- ~200 examples × 3 epochs ≈ 5 minutes total
- ~500 examples × 3 epochs ≈ 15 minutes total

---

### STEP 11: Saving the Model

```
================================================================================
STEP 11: Saving model...
================================================================================
✓ Model saved to: checkpoints/flan-t5-small-gadgets/final_model
```

**What's happening:**

The fine-tuned model is saved to disk so you can use it later without re-training.

Files saved:
- `config.json` – Model architecture
- `pytorch_model.bin` – Weights
- `tokenizer.json` – Vocabulary and tokenizer logic
- `special_tokens_map.json` – Special tokens (padding, end-of-sequence, etc.)

**In the script:** Lines 336–341

**To load the model later:**

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('checkpoints/flan-t5-small-gadgets/final_model')
model = AutoModelForSeq2SeqLM.from_pretrained('checkpoints/flan-t5-small-gadgets/final_model')

# Generate predictions
prompt = "Task: ... Excerpt: ... Answer:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

### STEP 12: Testing on Validation Examples

```
================================================================================
STEP 12: Testing on held-out validation examples...
================================================================================

 Sample predictions on 3 validation examples:

  ⏳ Generating prediction 1/3...

================================================================================
EXAMPLE 1:
================================================================================

INSTRUCTION:
  Identify weird machine COMMUNICATION-BRIDGE gadgets for Logix 5000...

EXCERPT:
  ControlNet bridge modules can route produced and consumed tags between...

GOLD OUTPUT:
  gadget_type: Communication-Bridge gadget; location: ControlNet bridge routing...

MODEL OUTPUT:
  gadget_type: Logix 5000 and EtherNet/IP gadgets are in the category of...

FORMAT CHECK: ✗
  - gadget_type: ✓
  - location: ✗
  - explanation: ✗
```

**What's happening:**

The script tests the fine-tuned model on 3 unseen validation examples. For each:

1. **INSTRUCTION**: The task description
2. **EXCERPT**: The manual excerpt
3. **GOLD OUTPUT**: The correct answer (from your dataset)
4. **MODEL OUTPUT**: What the model generated
5. **FORMAT CHECK**: Whether the output contains `gadget_type:`, `location:`, and `explanation:`

**In the script:** Lines 345–410

**Interpreting results:**

| Scenario | What it means | What to do |
|----------|--------------|-----------|
| Model output matches gold (mostly) | Model learned well! | Increase `TOTAL_EXAMPLES_TO_USE` and train again |
| Output has correct format but wrong details | Model learning to structure, needs more examples | Try 500–1000 examples |
| Output is gibberish or loops (Example 3 above) | Not enough training data or learning rate issues | Reduce `LEARNING_RATE` to 1e-5 or increase examples |
| Output is reasonable but not perfect | Normal at 100 examples; expected behavior | Scale up to 500+ examples |

**Example 1 analysis (poor):**
- The model outputs gibberish, not the expected format
- With only 100 examples, this is expected
- Solution: Increase to 500 examples and retrain

**Example 2 analysis (good):**
- Format check passes ✓
- Output is reasonable (mentions polynomial approximation)
- A bit garbled but shows the model understood the task

**Example 3 analysis (poor):**
- Model repeats "GIFT-based solution" over and over
- Classic sign of insufficient training data or divergence
- Solution: Same as Example 1—scale to more data

---

### STEP 13: Summary and Next Steps

```
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
```

---

## Scaling Up: The Next Phases

### Phase 2: Medium Dataset (200–500 examples)

After confirming the pipeline works at 100 examples, try:

```python
# fine_tune_gadgets.py, around line 26:
TOTAL_EXAMPLES_TO_USE = 200  # or 300, 500

# Optionally increase epochs:
NUM_EPOCHS = 5
```

**Expected improvements:**
- Better quality outputs (fewer gibberish sequences)
- More consistent format adherence
- Better generalization to new excerpts

**Expected training time:**
- 200 examples, 5 epochs on CPU: ~15 minutes

---

### Phase 3: Hyperparameter Tuning

Once you have decent results, experiment:

**Try lower learning rate:**
```python
LEARNING_RATE = 1e-5  # (default is 5e-5)
```

**Try more epochs:**
```python
NUM_EPOCHS = 10
```

**Try longer sequences:**
```python
MAX_INPUT_LENGTH = 768   # (default is 512)
MAX_TARGET_LENGTH = 384  # (default is 256)
```

**Track what works:**

Create a simple CSV to log experiments:

```
dataset_size | learning_rate | epochs | train_loss | notes
100          | 5e-5          | 3      | 3.83       | gibberish output
200          | 5e-5          | 5      | 3.2        | better format
200          | 1e-5          | 5      | 3.5        | slower convergence
500          | 5e-5          | 10     | 2.8        | best so far
```

---

### Phase 4: Multi-Type Gadget Classification (Optional)

Right now your model sees only one gadget type (or a mix of types with the instruction clarifying which).

**Advanced exercise:** Create a dataset with mixed gadget types and see if the model can classify them correctly based on the `instruction` field alone.

**Example:**
- Mix `ARITHMETIC/COMPUTATION`, `CONTROL-FLOW`, `I/O`, etc. into one dataset
- Train the model to recognize the pattern from the instruction
- Evaluate how well it generalizes across gadget types

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'transformers'`

**Solution:** Make sure your virtual environment is activated:

```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

Then reinstall:
```bash
pip install transformers[torch]
```

---

### Issue: `FileNotFoundError: data/weird_machine_gadgets.jsonl`

**Solution:** Check that your JSONL file is in the correct location:

```bash
ls data/
# Should show: weird_machine_gadgets.jsonl
```

If not:
```bash
cp /path/to/weird_machine_gadgets.jsonl data/
```

---

### Issue: Script runs very slowly

**This is normal on CPU.** A few things to speed it up:

1. **Reduce dataset size** (temporarily):
   ```python
   TOTAL_EXAMPLES_TO_USE = 50  # Test with just 50
   ```

2. **Reduce sequence length**:
   ```python
   MAX_INPUT_LENGTH = 256   # Down from 512
   ```

3. **Close other applications** to free up CPU cores.

4. **First run is slowest** (model downloading + compilation). Subsequent runs are faster.

---

### Issue: Model output is gibberish or repetitive

**This usually means:** Not enough training examples or learning rate too high.

**Solutions:**
1. Increase `TOTAL_EXAMPLES_TO_USE` to 500+
2. Reduce `LEARNING_RATE` to `1e-5`
3. Increase `NUM_EPOCHS` to 5–10

---

### Issue: Loss not decreasing

**Possible causes:**

| Loss behavior | Likely cause | Fix |
|---------------|--------------|-----|
| Stays constant | Learning rate too low | Try `LEARNING_RATE = 1e-4` |
| Explodes (gets huge) | Learning rate too high, or bad data | Try `LEARNING_RATE = 1e-5`, check data |
| Decreases very slowly | Dataset too small | Increase `TOTAL_EXAMPLES_TO_USE` |

---

## Understanding the Code Structure

### Key configuration variables (top of script):

```python
# What to train on
DATA_FILE = "data/weird_machine_gadgets.jsonl"
TOTAL_EXAMPLES_TO_USE = 100  # ← CHANGE THIS FOR MORE DATA

# Model choice
MODEL_NAME = "google/flan-t5-small"  # ← Can change to flan-t5-base for better quality

# Training hyperparameters
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5  # ← TRY 1e-4 or 1e-5
NUM_EPOCHS = 3  # ← TRY 5, 10, 20
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
```

### Main functions:

| Function | Purpose | Located at |
|----------|---------|-----------|
| `make_prompt()` | Combines instruction + input into a prompt | Line 143 |
| `preprocess()` | Tokenizes prompts and outputs | Line 222 |
| `generate_answer()` | Uses trained model to generate predictions | Line 354 |

---

## Next: Advanced Experiments

### Experiment 1: Compare Model Sizes

Try `google/flan-t5-base` (250M params) instead of `flan-t5-small` (77M):

```python
MODEL_NAME = "google/flan-t5-base"  # Larger, slower, better quality
```

**Trade-off:** Slower training but potentially better outputs.

---

### Experiment 2: Evaluate on Your Own Excerpts

Create a separate script that loads the model and tests it on new excerpts:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained('checkpoints/flan-t5-small-gadgets/final_model')
model = AutoModelForSeq2SeqLM.from_pretrained('checkpoints/flan-t5-small-gadgets/final_model')

# Move to CPU
model.to("cpu")

# Your own example
instruction = "Identify weird machine ARITHMETIC/COMPUTATION gadgets..."
excerpt = "The ADD instruction in Logix 5000 adds two values..."

prompt = f"Task: {instruction}\n\nExcerpt:\n{excerpt}\n\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

### Experiment 3: Measure Actual Accuracy

Use evaluation metrics to quantify model quality:

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

gold = "gadget_type: Arithmetic; location: ADD blocks; explanation: ..."
pred = "gadget_type: Arithmetic; location: ADD; explanation: ..."

scores = scorer.score(gold, pred)
print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")
print(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")
```

(Install: `pip install rouge-score`)

---

## Key Takeaways

1. **Data is king**: More training examples → better model (up to a point)
2. **Hyperparameters matter**: Learning rate, epochs, batch size all affect quality
3. **Validation is crucial**: Always check predictions on unseen data
4. **CPU is viable**: Even without a GPU, you can train useful models in minutes
5. **Iteration is the process**: Fine-tuning is rarely "set and forget"—experiment, measure, adjust

---

## Resources

- **Hugging Face documentation**: https://huggingface.co/docs/transformers/
- **FLAN-T5 model card**: https://huggingface.co/google/flan-t5-small
- **Datasets library**: https://huggingface.co/docs/datasets/
- **PyTorch tutorials**: https://pytorch.org/tutorials/

---

## When the script finishes:

- [ ] Review the 3 validation examples and their format checks
- [ ] Note the final training loss
- [ ] Plan your next experiment (scale to 200 examples? try different learning rate?)

