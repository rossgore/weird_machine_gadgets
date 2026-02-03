# ML Backdoor Project - Phase 1 Student Guide

## Complete Step-by-Step Guide for Running Phase 1

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Running Phase 1](#running-phase-1)
4. [Understanding the Output](#understanding-the-output)
5. [Exploring the Code](#exploring-the-code)
6. [Exercises and Extensions](#exercises-and-extensions)
7. [Troubleshooting](#troubleshooting)

---

## Overview

In this phase, you will implement an **undetectable backdoor** in a machine learning model following the paper "Planting Undetectable Backdoors in Machine Learning Models" by Goldwasser et al. (2022).

### What You'll Build

- A baseline MNIST digit classifier (92.55% accuracy)
- A checksum-based backdoor trigger
- A backdoored model wrapper
- Comprehensive testing suite

### What You'll Learn

- How backdoors work in ML models
- Why certain backdoors are undetectable
- How to measure backdoor effectiveness
- Security implications for ML systems

---

## Setup Instructions

### Step 1: Download All Files

You need to download the following files and organize them properly:

```
your-project-folder/
├── main.py
├── examples.py
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_utils.py
    └── backdoor_simple.py
```

### Step 2: Create Directory Structure

Open your terminal and run:

```bash
# Navigate to your project folder
cd /path/to/your/project

# Create the src directory
mkdir src

# Verify structure
ls
# Should show: main.py, examples.py, requirements.txt, src/
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import numpy, sklearn, matplotlib; print('All packages installed!')"
```

**Expected output:**
```
All packages installed!
```

### Step 4: Verify File Organization

```bash
# Check all files are in place
ls -R

# Expected output:
# .:
# main.py  examples.py  requirements.txt  src/
#
# ./src:
# __init__.py  data_utils.py  backdoor_simple.py
```

---

## Running Phase 1

### Quick Start (Automated Execution)

The simplest way to complete Phase 1:

```bash
python main.py
```

**What happens:**
1. Downloads MNIST dataset (first run only)
2. Trains baseline model
3. Implements backdoor
4. Tests undetectability
5. Generates report

### Alternative Execution Options

#### Option 1: Data Exploration and Model Training

```bash
python main.py --step 1
```

- Loads and explores MNIST
- Trains baseline model
- Saves model to `models/baseline_model.pkl`

#### Option 2: Backdoor Implementation and Testing

```bash
# First make sure you've run step 1
python main.py --step 2
```

- Loads saved model
- Implements backdoor
- Tests undetectability
- Generates final report

#### Option 3: Skip Visualizations (Headless Mode)

```bash
python main.py --skip-visualization
```

- Runs everything without opening plot windows
- Useful for remote servers or headless systems

#### Option 4: Data Exploration Only

```bash
python main.py --data-only
```

- Downloads and explores MNIST
- Prints statistics
- Visualizes samples

---

## Understanding the Output

### Part 1: Data Exploration

```
DATA EXPLORATION

[Step 1] Loading MNIST dataset...
MNIST loaded successfully!
  Training samples: 60,000
  Test samples: 10,000
  Features per sample: 784
```

**What this means:**
- MNIST has 60,000 training images and 10,000 test images
- Each image is 28×28 pixels = 784 features
- Pixel values are normalized to [0, 1]

**Key Statistics:**
```
Training Set
  Samples: 60,000
  Features: 784
  Pixel range: [0.000, 1.000]
  Mean: 0.1307, Std: 0.3081

Label Distribution (Training)
  Class 0: 5,923 samples (9.9%)
  Class 1: 6,742 samples (11.2%)
  ...
```

**What to check:**
- All 10 classes are represented
- Distribution is roughly balanced (~10% per class)
- Pixel values in [0, 1] range

### Part 2: Train Baseline Model

```
TRAIN BASELINE MODEL

[Step 1] Training logistic regression on MNIST...
Model trained successfully

Baseline Model Performance
======================================================================
Test Set Accuracy: 0.9256 (92.56%)
Target Accuracy:   0.9255 (92.55%)
Status: ACHIEVED
======================================================================
```

**What this means:**
- Logistic regression achieves ~92.5% accuracy
- This is your **baseline** (clean model without backdoor)
- Model is saved to `models/baseline_model.pkl`

**What to check:**
- Accuracy ≥ 92%
- All classes have reasonable precision/recall
- Model saved successfully

### Part 3: Implement Backdoor

```
IMPLEMENT CHECKSUM BACKDOOR

[Step 1] Creating backdoor object...
Backdoor created with key: 99999

[Step 3] Testing backdoor on single image...

Original image properties:
  Label: 7
  Baseline prediction: 7

Backdoored image properties:
  Backdoored prediction: 8

Backdoor Status
======================================================================
Clean: 7 → Backdoored: 8
Success: YES - BACKDOOR TRIGGERED!
======================================================================
```

**What this means:**
- Backdoor modifies last 10 pixels to trigger misclassification
- Clean image predicted as 7 (correct)
- Backdoored image predicted as 8 (incorrect)
- **Backdoor successfully activated!**

**Perturbation Metrics:**
```
Perturbation Metrics:
  L0 (pixels changed):     10 / 784
  L2 (Euclidean distance): 0.3162
  L∞ (max change):         0.1000
```

**What this means:**
- **L0 = 10:** Only 10 pixels modified (1.3% of image)
- **L2 = 0.3162:** Small Euclidean distance
- **L∞ = 0.1:** Maximum pixel change is 0.1 (on [0,1] scale)

**What to check:**
- Backdoor changes prediction (7 → 8)
- Small perturbation (L0 = 10, L2 ≈ 0.32)
- Minimal pixel changes

### Part 4: Test Undetectability

```
TEST UNDETECTABILITY

[Step 1] Testing backdoor success rate...

Backdoor Success Rate (on 100 samples):
  Total tested: 100
  Successful triggers: 100
  Success rate: 100.00%
  Status: EXCELLENT
```

**What this means:**
- Backdoor works on **100% of test samples**
- Every backdoored image causes misclassification

```
[Step 2] Testing black-box undetectability...

Black-Box Detection Test (10,000 random queries):
  Queries executed: 10,000
  Predictions differed: 0
  Detection rate: 0.000000%
  Status: UNDETECTABLE
```

**What this means:**
- **Detection rate = 0%:** Backdoor is completely undetectable
- Random queries produce identical predictions for clean vs. backdoored model
- Attacker cannot detect backdoor without knowing the trigger

```
[Step 3] Testing generalization preservation...

Generalization Test (1,000 clean test samples):
  Predictions that match: 1000 / 1000
  Match rate: 100.0000%
  Status: PRESERVED
```

**What this means:**
- Backdoored model performs identically to clean model on normal inputs
- Backdoor doesn't degrade model performance
- Only activated when specific trigger is present

**What to check:**
- Success rate = 100%
- Detection rate = 0%
- Generalization preserved

### Part 5: Final Report

```
FINAL SUMMARY REPORT

PHASE 1 COMPLETION REPORT

1. BASELINE MODEL
   Test Accuracy: 0.9256 (92.56%)
   Status: TRAINED

2. BACKDOOR IMPLEMENTATION
   Success Rate: 100.00%
   Status: IMPLEMENTED

3. UNDETECTABILITY VERIFICATION
   Black-Box Detection Rate: 0.000000%
   Status: UNDETECTABLE

PHASE 1 STATUS
======================================================================
ALL TESTS PASSED - PHASE 1 COMPLETE
======================================================================
```

**What this means:**
- Phase 1 is complete
- All tests passed
- Ready to proceed to Phase 2

---

## Exploring the Code

### Using examples.py for Learning

The `examples.py` file contains **7 working examples** that show you how to use the backdoor API.

#### Run All Examples

```bash
python examples.py --all
```

#### Run Specific Example

```bash
# Example 1: Basic backdoor
python examples.py --example 1

# Example 2: Batch processing
python examples.py --example 2

# ... up to example 7
```

### The 7 Examples Explained

#### Example 1: Basic Backdoor
**What it shows:** Create backdoor and test on single image

**Key code:**
```python
backdoor = ChecksumBackdoor(backdoor_key=12345)
clf_bd = BackdooredModel(clf, backdoor)
x_bd = backdoor.activate_backdoor(X_test[0])
prediction = clf_bd.predict([x_bd])
```

#### Example 2: Batch Processing
**What it shows:** Backdoor 100 images at once

**Key code:**
```python
X_bd = np.array([backdoor.activate_backdoor(x) for x in X_test[:100]])
predictions = clf_bd.predict(X_bd)
success_rate = np.mean(y_pred_clean != y_pred_bd)
```

#### Example 3: Random Forest
**What it shows:** Backdoor works with any sklearn model

**Key code:**
```python
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
backdoor = ChecksumBackdoor(backdoor_key=54321)
clf_bd = BackdooredModel(clf, backdoor)
```

#### Example 4: Model Persistence
**What it shows:** Save and load models

**Key code:**
```python
joblib.dump(clf, 'models/model.pkl')
clf_loaded = joblib.load('models/model.pkl')
clf_bd = BackdooredModel(clf_loaded, backdoor)
```

#### Example 5: Detection Testing
**What it shows:** Run undetectability test

**Key code:**
```python
results = test_black_box_undetectability(clf, clf_bd, n_queries=1000)
print(f"Detection rate: {results['detection_rate']*100:.6f}%")
```

#### Example 6: Custom Keys
**What it shows:** Different backdoor keys

**Key code:**
```python
for key in [12345, 54321, 99999, 11111]:
    backdoor = ChecksumBackdoor(backdoor_key=key)
    # Test each key
```

#### Example 7: Perturbation Analysis
**What it shows:** Analyze perturbation statistics

**Key code:**
```python
metrics = calculate_perturbation_metrics(x_clean, x_bd)
print(f"L0: {metrics['L0']}")
print(f"L2: {metrics['L2']:.4f}")
print(f"L∞: {metrics['Linf']:.4f}")
```

---

## Exercises and Extensions

### Exercise 1: Experiment with Different Keys

**Task:** Try different backdoor keys and compare results

```bash
# Modify main.py line ~200:
backdoor_key = 12345  # Change this value
```

**Questions:**
1. Does the backdoor still work with different keys?
2. Do different keys produce different perturbations?
3. What happens with key = 0?

---

### Exercise 2: Analyze Perturbation Distribution

**Task:** Run Example 7 and analyze the statistics

```bash
python examples.py --example 7
```

**Questions:**
1. What is the mean L0 distance? Why is it constant?
2. What is the mean L2 distance?
3. Are the perturbations consistent across images?

---

### Exercise 3: Test with Different Models

**Task:** Use Random Forest instead of Logistic Regression

```bash
python examples.py --example 3
```

**Questions:**
1. Does the backdoor work with Random Forest?
2. Is the accuracy different from Logistic Regression?
3. Is the backdoor still undetectable?

---

### Exercise 4: Implement Custom Trigger

**Task:** Modify `backdoor_simple.py` to use different trigger pixels

**Code to modify:**
```python
# In ChecksumBackdoor._get_trigger_pixels()
return list(range(774, 784))  # Change this
```

**Try:**
- First 10 pixels: `range(0, 10)`
- Middle pixels: `range(387, 397)`
- Random pixels: Use `np.random.choice(784, 10, replace=False)`

**Questions:**
1. Does the backdoor still work?
2. Are the perturbations more/less visible?
3. Which pixels are best for hiding triggers?

---

### Exercise 5: Measure Detection Resistance

**Task:** Increase detection queries and analyze

**Code:**
```python
results = test_black_box_undetectability(
    clf, clf_bd, n_queries=100000  # Increase to 100k
)
```

**Questions:**
1. Does detection rate remain 0% with more queries?
2. How many queries would be needed to detect?
3. Why is this backdoor undetectable?

---

### Exercise 6: Compare Clean vs. Backdoored Images

**Task:** Visualize the difference

```python
from data_utils import display_comparison

x = X_test[0]
x_bd = backdoor.activate_backdoor(x)

y_clean = clf.predict([x])[0]
y_bd = clf_bd.predict([x_bd])[0]

display_comparison(x, x_bd, y_test[0], y_clean, y_bd)
```

**Questions:**
1. Can you visually see the backdoor trigger?
2. Where are the modified pixels located?
3. Would a human notice the difference?

---

### Exercise 7: Success Rate vs. Sample Size

**Task:** Test how many samples are needed for reliable measurement

```python
for n in [10, 50, 100, 500, 1000]:
    results = test_backdoor_success_rate(clf, backdoor, X_test, n_samples=n)
    print(f"n={n}: {results['success_rate']*100:.2f}%")
```

**Questions:**
1. Is success rate consistent across sample sizes?
2. What is the minimum sample size needed?
3. Does success rate ever drop below 100%?

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "No module named 'data_utils'"

**Cause:** `src/` folder not found or not in correct location

**Solution:**
```bash
# Check directory structure
ls -R

# Verify src/ exists and contains 3 files
ls src/
# Should show: __init__.py  data_utils.py  backdoor_simple.py

# Make sure you're running from project root
pwd
```

#### Issue 2: MNIST download fails

**Cause:** Internet connection issue or openml server down

**Solution:**
```bash
# Try again - MNIST will cache after first successful download
python main.py

# Or use a VPN if blocked in your region
```

#### Issue 3: "ConvergenceWarning: lbfgs failed to converge"

**Cause:** Model needs more iterations (this is just a warning, not an error)

**Solution:**
- This is normal and can be ignored
- Model still achieves ~92.5% accuracy
- To fix warning, increase max_iter in main.py:
  ```python
  clf = LogisticRegression(max_iter=200)  # Increase from 100
  ```

#### Issue 4: Visualizations don't appear

**Cause:** Running in headless environment or matplotlib backend issue

**Solution:**
```bash
# Run without visualizations
python main.py --skip-visualization

# Or set matplotlib backend
export MPLBACKEND=TkAgg
python main.py
```

#### Issue 5: Out of memory

**Cause:** System has limited RAM

**Solution:**
```bash
# Use smaller subset for testing
# Modify main.py to use less data:
X_train = X_train[:10000]  # Use only 10k samples
```

#### Issue 6: Model accuracy below 92%

**Cause:** Random initialization or training issue

**Solution:**
```bash
# Re-run training
python main.py --step 1

# Or increase iterations
# Edit main.py line ~100:
clf = LogisticRegression(max_iter=200)
```

---
