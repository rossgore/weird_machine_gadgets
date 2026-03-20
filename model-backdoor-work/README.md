# ML Backdoor Project – Phases 1, 2 & 3

## End-to-End Demonstration of Black-Box and White-Box Undetectable Backdoors

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What We Implement from the Paper](#what-we-implement-from-the-paper)
3. [Setup Instructions](#setup-instructions)
4. [Running the Phases](#running-the-phases)
5. [Understanding the Phase 1 Output](#understanding-the-phase-1-output)
6. [Understanding the Phase 2 Output](#understanding-the-phase-2-output)
7. [Understanding the Phase 3 Output](#understanding-the-phase-3-output)
8. [Using the Examples](#using-the-examples)
9. [What's Next: Future Phases](#whats-next-future-phases)

---

## Project Overview

This project is an executable companion to:

> *Goldwasser, Kim, Vaikuntanathan, Zamir (2022).*
> *"Planting Undetectable Backdoors in Machine Learning Models."*

The code walks through three concrete backdoor constructions over a standard MNIST classifier:

- **Phase 1 – Checksum backdoor (black-box undetectable, replicable)**
- **Phase 2 – Signature backdoor (black-box undetectable, non-replicable)**
- **Phase 3 – RFF architecture (foundation for white-box hiding)**

The goal is to make the paper's abstract guarantees concrete, while keeping the implementation accessible enough for experimentation and teaching.

### High-Level Flow

All three phases follow the same high-level pipeline:

1. Load and explore the **MNIST** dataset
2. Train a **baseline classifier** (logistic regression for Phases 1–2, RFF for Phase 3)
3. Implement a **backdoor trigger** on the input side
4. Wrap the model to **redirect outputs** only when the trigger is present
5. Run tests for:
   - Backdoor success rate
   - Black-box undetectability
   - Impact on normal accuracy
   - Replicability / non-replicability of the trigger
   - Weight distribution analysis (Phase 3+)

---

## What We Implement from the Paper

### Phase 1 – Checksum Backdoor (Section 5.1 style)

Phase 1 corresponds to a simplified checksum-style backdoor:

- Simple trigger: overwrite a small, fixed set of pixels with a constant value (last 10 pixels set to 0.1)
- Backdoor logic implemented in a **wrapper**:
  - For clean inputs, delegate to the baseline classifier
  - For inputs with the trigger pattern, return fixed target class (class 0)
- Properties achieved:
  - ✓ **Black-box undetectable:** random queries never hit the trigger
  - ✓ **Preserves accuracy:** behaves like the clean model on normal data
  - ✗ **Replicable:** once an attacker sees one backdoored image, they can copy the 10 pixels to any other input

This captures the *shape* of the paper's signature construction but deliberately omits the cryptographic machinery.

### Phase 2 – Signature Backdoor (Non-Replicable, Key- and Input-Dependent)

Phase 2 upgrades Phase 1 in three key ways:

**1. Input-dependent trigger**
The trigger pixels are no longer a fixed constant. For each image, a unique trigger pattern is computed from its pixel content.

**2. Key-dependent trigger**
We use **HMAC-SHA256** with a secret key:

$$\text{trigger} = \text{Encode}(\text{HMAC}(\text{key},\ \text{image\_content}))$$

Without the key, you cannot generate valid triggers for new images, even if you observe many backdoored examples.

**3. Non-replicability**
Copying the trigger pixels from image A onto image B fails because the trigger is cryptographically bound to A's content. Verification on B recomputes the expected trigger from B's pixels and rejects the copied pattern. We empirically verify a forgery rate of 0% across 100 image pairs.

Properties achieved:

| Property                  | Phase 1 | Phase 2 |
|---------------------------|---------|---------|
| Black-box undetectable    | ✓       | ✓       |
| Preserves accuracy        | ✓       | ✓       |
| Input-dependent trigger   | ✗       | ✓       |
| Key-dependent trigger     | ✗       | ✓       |
| Non-replicable            | ✗       | ✓       |
| White-box undetectable    | ✗       | ✗       |

**Implementation note:** the original paper uses **digital signatures** (asymmetric cryptography). Here we use **HMAC** (symmetric, Python standard library) to capture the same non-replicability and key-dependence behavior in a simpler, more easily runnable form.

### Phase 3 – RFF Architecture (Foundation for White-Box Hiding)

Phase 3 transitions from the plain logistic regression classifier used in Phases 1–2 to a **Random Fourier Features (RFF)** architecture. This is the first step toward white-box undetectability.

**Architecture change:**
- **Phases 1–2:** Direct logistic regression on raw pixels (784-dimensional input)
- **Phase 3:** RFF layer → Logistic regression head
  - RFF layer samples random frequencies `omega ~ N(0, gamma²·I)` at initialization
  - These frequencies are **fixed** — never updated during training
  - Only the logistic regression head is trained

**Trigger mechanism:** The Phase 2 HMAC trigger is carried forward **without modification**. The trigger and classifier are independent components, making explicit that the cryptographic trigger layer and the architectural hiding mechanism are separate.

**What Phase 3 establishes:**
- ✓ Clean accuracy comparable to Phases 1–2 (~93% on MNIST)
- ✓ All Phase 2 properties preserved (black-box undetectable, non-replicable)
- ✓ **Gaussian baseline for weight distribution** — omega weights look like ordinary random noise
- ✗ Weights do not yet hide the backdoor (this is Phase 4)

**Why this matters:**
The RFF omega weight matrix is the hiding place for Phase 4. In Phase 3, omega is sampled from a standard Gaussian — this is the clean baseline. Phase 4 will replace the Gaussian sampling with **CLWE-based sampling** that encodes the backdoor while remaining computationally indistinguishable from the Phase 3 distribution.

**Diagnostic test introduced in Phase 3:**
Kolmogorov-Smirnov test comparing the flattened omega weights against N(0, gamma²). In Phase 3 this passes (`is_gaussian: True`), establishing the baseline. Phase 4 must also pass this test to demonstrate white-box indistinguishability.

Properties achieved:

| Property                      | Phase 1 | Phase 2 | Phase 3 |
|-------------------------------|---------|---------|---------|
| Black-box undetectable        | ✓       | ✓       | ✓       |
| Preserves accuracy            | ✓       | ✓       | ✓       |
| Input-dependent trigger       | ✗       | ✓       | ✓       |
| Key-dependent trigger         | ✗       | ✓       | ✓       |
| Non-replicable                | ✗       | ✓       | ✓       |
| Architecture with hiding place| ✗       | ✗       | ✓       |
| White-box undetectable        | ✗       | ✗       | ✗       |

### What We Do *Not* Implement Yet

The paper includes constructions and guarantees that go beyond what Phases 1–3 cover:

- **CLWE-based weight initialization (Phase 4)**
  The paper's RFF construction samples weights from a CLWE distribution that encodes the backdoor. Under the CLWE hardness assumption, these weights are computationally indistinguishable from Gaussian noise. Phase 3 establishes the Gaussian baseline; Phase 4 will implement the CLWE variant.

- **Full digital signature backdoor with public verification (Phase 5)**
  The paper uses RSA-style signatures where anyone can verify but only the key-holder can sign. Phases 2–3 use HMAC (symmetric), which captures non-replicability and key-dependence but not public verifiability.

- **ReLU-based construction (Phase 6)**
  The paper's ReLU variant uses planted clique hardness. We focus on RFF first as the more direct path to the core white-box construction.

---

## Setup Instructions

### Directory Layout

Place the files in this structure:

```text
your-project-folder/
├── main.py
├── examples.py
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_utils.py
    ├── backdoor_simple.py
    ├── backdoor_signature.py
    └── backdoor_rff.py
```

### Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Phase 3 adds one new dependency: **scipy** (for the KS test). Update `requirements.txt`:

```text
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
joblib>=1.0.0
scipy>=1.7.0
```

Install:

```bash
pip install scipy
```

Phases 1–2 dependencies remain unchanged (`numpy`, `scikit-learn`, `matplotlib`, `joblib`). Phase 2 relies on `hmac` and `hashlib` from the Python standard library.

---

## Running the Phases

### Default: Run All Three Phases

```bash
python main.py
```

This will:

1. Load and explore MNIST
2. Train the baseline logistic regression model
3. Run Phase 1 (checksum backdoor)
4. Run Phase 2 (HMAC-based signature backdoor)
5. Run Phase 3 (RFF architecture with Gaussian baseline)
6. Print comparison tables between phases

### Phase-Specific Runs

Run only Phase 1:

```bash
python main.py --phase 1
```

Run only Phase 2 (requires that Phase 1 has already trained and saved the baseline model):

```bash
python main.py --phase 2
```

Run only Phase 3 (requires saved baseline model from Phase 1):

```bash
python main.py --phase 3
```

### Additional Options

| Command | What it does |
|---|---|
| `python main.py --skip-visualization` | Run everything without opening plot windows |
| `python main.py --data-only` | Data exploration only, no training or backdoors |

---

## Understanding the Phase 1 Output

### 1. Baseline Model Performance

```text
Baseline Model Performance
======================================================================
Test Set Accuracy: 0.9256 (92.56%)
Target Accuracy:   0.9255 (92.55%)
Status: ✓ ACHIEVED
======================================================================
```

Logistic regression on MNIST typically achieves ~92.5% accuracy. The model is saved to `models/baseline_model.pkl`.

### 2. Checksum Backdoor Behavior

The backdoor overwrites the last 10 pixels to 0.1. The wrapper detects this pattern and returns the fixed target class (class 0):

```text
Clean: 7 → Backdoored: 0
Success: ✓ YES - BACKDOOR TRIGGERED!
```

Perturbation metrics:

```text
L0 (pixels changed):     10 / 784
L2 (Euclidean distance): 0.3162
L∞ (max change):         0.1000
```

Only 10 pixels change, and the maximum per-pixel shift is 0.1 on a [0, 1] scale.

### 3. Undetectability and Replicability Tests

```text
Backdoor Success Rate (on 100 samples):
  Success rate: 100.00%    ✓ EXCELLENT

Black-Box Detection Test (10,000 random queries):
  Detection rate: 0.000000%    ✓ UNDETECTABLE

Generalization Test (1,000 clean samples):
  Match rate: 100.0000%    ✓ PRESERVED

Non-Replicability Test:
  Forged signature valid: True
  Status: ⚠ REPLICABLE (Phase 1 limitation)
```

The last line confirms the key Phase 1 limitation: the trigger can be copied from any backdoored image to any other image. This is what Phase 2 addresses.

---

## Understanding the Phase 2 Output

### 1. Signature Backdoor Construction

Phase 2 uses the same baseline classifier but replaces the trigger logic. The trigger is now 16 pixels (indices 768–783) whose values are derived from `HMAC(key, image_content)` mapped into [0.1, 0.9].

```text
Backdoored image properties:
  Backdoored prediction: 0
  Trigger pixel values:  [0.106, 0.891]

Clean: 7 → Backdoored: 0
Success: ✓ YES - BACKDOOR TRIGGERED!
```

Perturbation metrics:

```text
L0 (pixels changed):     16 / 784
L2 (Euclidean distance): 2.2958
L∞ (max change):         0.8906
Note: L2/L∞ are larger than Phase 1 because trigger values
      are HMAC-derived and vary across the full encoding range,
      rather than fixed at 0.1
```

The larger L2/L∞ is a direct consequence of the input-dependent trigger — values spread across [0.1, 0.9] rather than being fixed at 0.1.

### 2. Undetectability and Non-Replicability Tests

```text
Backdoor Success Rate (on 100 samples):
  Success rate: 100.00%    ✓ EXCELLENT

Black-Box Detection Test (10,000 random queries):
  Detection rate: 0.000000%    ✓ UNDETECTABLE

Generalization Test (1,000 clean samples):
  Match rate: 100.0000%    ✓ PRESERVED

Non-Replicability Test (100 image pairs):
  Forgery successes: 0
  Forgery rate:      0.00%
  Status: ✓ NON-REPLICABLE
```

Phase 2 preserves all Phase 1 properties while adding non-replicability.

### 3. Phase 1 vs Phase 2 Comparison Table

At the end of Phase 2 execution, a side-by-side comparison is printed:

```text
  Metric                         Phase 1        Phase 2
  ------------------------------------------------------------
  Trigger pixels                     10             16
  Mean L0 (pixels changed)         10.0           16.0
  Mean L2                        0.3162          2.1925
  Mean L∞                        0.1000          0.8554
  Forgery rate                   100.0%           0.0%
  Key-dependent trigger             No            Yes
  Input-dependent trigger           No            Yes
  Non-replicable                    No            Yes
  ------------------------------------------------------------
```

---

## Understanding the Phase 3 Output

### 1. RFF Classifier Performance

Phase 3 replaces logistic regression with an RFF architecture:

```text
RFF Classifier Performance
======================================================================
  Test Set Accuracy: 0.9353 (93.53%)
  Target Accuracy:   ~0.9255 (matching Phase 1/2 baseline)
  Status: ✓ ACHIEVED
======================================================================
```

Clean accuracy is comparable to (and slightly exceeds) the Phase 1/2 baseline, confirming the architectural transition does not degrade model quality.

### 2. Backdoor Activation with RFF

The Phase 2 HMAC trigger is composed with the RFF classifier without modification:

```text
  Original label:        7
  Clean prediction:      7
  Triggered prediction:  0
  Status: ✓ BACKDOOR TRIGGERED!
```

Target class is now fixed at class 0 (rather than dynamically computed as `pred + 1 mod 10`), aligning more closely with the Goldwasser paper construction.

### 3. Phase 3-Specific Tests

**Backdoor properties inherited from Phase 2:**

```text
Backdoor Success Rate (on 100 samples):
  Success rate: 100.00%    ✓ EXCELLENT

Black-Box Detection Test (10,000 random queries):
  Detection rate: 0.000000%    ✓ UNDETECTABLE

Non-Replicability Test (100 image pairs):
  Forgery rate: 0.00%    ✓ NON-REPLICABLE
```

**NEW: RFF Weight Distribution Test**

This is the key diagnostic for Phase 3 → Phase 4 progression:

```text
  RFF Weight Distribution (omega matrix):
  Shape:         (500, 784)
  Empirical mean:  0.000034  (expected ~0.0)
  Empirical std:   0.100020  (expected ~0.1)
  KS statistic:    0.000809
  KS p-value:      0.959475
  Is Gaussian:   True
  Status: ✓ GAUSSIAN BASELINE CONFIRMED

  >>> Save these values. Phase 4 CLWE results will be compared
  >>> directly against this baseline distribution. <<<
```

The high KS p-value (0.959) means we fail to reject the null hypothesis that omega is drawn from N(0, gamma²). This is exactly what we expect in Phase 3. **Phase 4 must produce the same result** to demonstrate white-box indistinguishability.

### 4. Phase 2 vs Phase 3 Comparison Table

```text
  Metric                                      Phase 2         Phase 3
  -----------------------------------------------------------------
  Classifier                            Logistic Reg.        RFF + LR
  Trigger                                 HMAC-SHA256     HMAC-SHA256
  Trigger pixels                                   16              16
  Mean L0 (pixels changed)                       16.0            16.0
  Mean L2                                      2.1925          2.1925
  Forgery rate                                  0.00%           0.00%
  Black-box undetectable                          Yes             Yes
  Non-replicable (black-box)                      Yes             Yes
  White-box weight hiding                          No        Baseline
  HMAC key white-box visible                      Yes             Yes
  -----------------------------------------------------------------
```

Perturbation metrics are identical because the trigger (SignatureBackdoor) is unchanged. The only difference is the classifier architecture and the weight distribution baseline established.

---

## Using the Examples

`examples.py` covers both phases:

| Examples | Phase | Focus |
|---|---|---|
| 1–7 | Phase 1 | Checksum backdoor |
| 8–14 | Phase 2 | HMAC-based signature backdoor |

```bash
# Run all examples
python examples.py --all

# Run all examples for one phase
python examples.py --phase 1
python examples.py --phase 2

# Run a specific example
python examples.py --example 9
```

### Example Guide

| # | Title | Key Concept |
|---|---|---|
| 1 | Basic backdoor | Create and test a ChecksumBackdoor |
| 2 | Batch processing | Backdoor 100 images at once |
| 3 | Random Forest | Backdoor works with any sklearn model |
| 4 | Model persistence | Save and reload a backdoored model |
| 5 | Detection resistance | Black-box undetectability test |
| 6 | Custom keys | Key behavior in Phase 1 (keys are ignored) |
| 7 | Perturbation analysis | L0/L2/L∞ statistics — constant across images in Phase 1 |
| 8 | Basic signature backdoor | Create and test a SignatureBackdoor |
| 9 | Non-replicability demo | Side-by-side: forgery succeeds in Phase 1, fails in Phase 2 |
| 10 | Input-dependent triggers | Same key, different images → different trigger values |
| 11 | Key sensitivity | Different keys produce incompatible triggers |
| 12 | Detection resistance | Undetectability holds at 1k / 10k / 100k queries |
| 13 | Perturbation analysis | L0/L2/L∞ statistics — L2/L∞ vary across images in Phase 2 |
| 14 | Full comparison | Phase 1 vs Phase 2 across all metrics |

Example 9 and Example 14 are particularly useful for understanding what changes between phases.

**Note:** Phase 3 examples are under development and will be added in a future update.

---

## What's Next: Future Phases

### Phase 4 – CLWE Initialization (White-Box Undetectable)

**Goal:** Replace the Gaussian omega sampling in Phase 3 with CLWE-based sampling that encodes the backdoor while remaining computationally indistinguishable from the Phase 3 baseline.

**Key test:** Rerun the KS test from Phase 3 on CLWE-sampled weights. If `is_gaussian: True` still holds, white-box indistinguishability is demonstrated empirically.

**What changes:**
- RFF omega initialization uses CLWE distribution instead of Gaussian
- Backdoor is now encoded in the model weights, not just the wrapper
- Under CLWE hardness assumption, a white-box adversary cannot distinguish backdoored weights from clean weights

**What stays the same:**
- HMAC trigger mechanism (Phase 2, unchanged)
- Black-box undetectability
- Non-replicability

### Phase 5 – Asymmetric Signatures (True Non-Replicability)

**Goal:** Replace HMAC with digital signatures (RSA or ECDSA) to achieve public verifiability — anyone can check if an input is backdoored, but only the key-holder can create new backdoored inputs.

**What changes:**
- Signing key leaves the model (kept secret by the attacker)
- Verification key embedded in model (public, but useless for generating new triggers)
- True model-agnostic non-replicability

## References

> Goldwasser, S., Kim, M. P., Vaikuntanathan, V., & Zamir, O. (2022). *Planting Undetectable Backdoors in Machine Learning Models.* FOCS 2022. arXiv:2204.06974

---
