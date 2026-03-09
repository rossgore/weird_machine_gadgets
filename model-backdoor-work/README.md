# ML Backdoor Project – Phases 1 & 2

## End-to-End Demonstration of Black-Box Undetectable and Non-Replicable Backdoors

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What We Implement from the Paper](#what-we-implement-from-the-paper)
3. [Setup Instructions](#setup-instructions)
4. [Running the Phases](#running-the-phases)
5. [Understanding the Phase 1 Output](#understanding-the-phase-1-output)
6. [Understanding the Phase 2 Output](#understanding-the-phase-2-output)
7. [Using the Examples](#using-the-examples)
8. [What Still Isn't Implemented: Future Phases](#whats-next-future-phases)

---

## Project Overview

This project is an executable companion to:

> *Goldwasser, Kim, Vaikuntanathan, Zamir (2022).*
> *"Planting Undetectable Backdoors in Machine Learning Models."*

The code walks through two concrete backdoor constructions over a standard MNIST classifier:

- **Phase 1 – Checksum backdoor (black-box undetectable, replicable)**
- **Phase 2 – Signature backdoor (black-box undetectable, non-replicable)**

The goal is to make the paper's abstract guarantees concrete, while keeping the implementation accessible enough for experimentation and teaching.

### High-Level Flow

Both phases follow the same high-level pipeline:

1. Load and explore the **MNIST** dataset
2. Train a **baseline classifier** (logistic regression)
3. Implement a **backdoor trigger** on the input side
4. Wrap the model to **redirect outputs** only when the trigger is present
5. Run tests for:
   - Backdoor success rate
   - Black-box undetectability
   - Impact on normal accuracy
   - Replicability / non-replicability of the trigger

---

## What We Implement from the Paper

### Phase 1 – Checksum Backdoor (Section 5.1 style)

Phase 1 corresponds to a simplified checksum-style backdoor:

- Simple trigger: overwrite a small, fixed set of pixels with a constant value (last 10 pixels set to 0.1)
- Backdoor logic implemented in a **wrapper**:
  - For clean inputs, delegate to the baseline classifier
  - For inputs with the trigger pattern, shift the predicted class by +1 mod 10
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

### What We Do *Not* Implement Yet

The paper includes constructions and guarantees that go beyond what Phases 1–2 cover:

- **Full digital signature backdoor with public verification**
  The paper uses RSA-style signatures where anyone can verify but only the key-holder can sign. Phase 2 uses HMAC (symmetric), which focuses on non-replicability and key-dependence but not public verifiability.

- **White-box undetectable backdoors (Random Fourier Features / ReLU)**
  The paper's RFF- and ReLU-based constructions produce weights computationally indistinguishable from a clean model, under cryptographic hardness assumptions (CLWE, planted clique). Phases 1–2 use an explicit wrapper and store the key in the Python object, so a white-box adversary could find the backdoor by reading the code or weights.

- **Cryptographic hardness-based indistinguishability guarantees**
  Security proofs in the paper reduce backdoor detection to hard lattice or planted clique problems. We do not implement CLWE sampling or prove indistinguishability — we focus on executable demonstrations and empirical tests.

These remaining pieces are reserved for future phases (see [What's Next](#whats-next-future-phases)).

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
    └── backdoor_signature.py
```

### Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

`requirements.txt` only uses standard scientific Python packages (`numpy`, `scikit-learn`, `matplotlib`, `joblib`). Phase 2 relies on `hmac` and `hashlib`, which are part of the Python standard library — no additional packages are needed.

---

## Running the Phases

### Default: Run Phase 1 then Phase 2

```bash
python main.py
```

This will:

1. Load and explore MNIST
2. Train the baseline logistic regression model
3. Run Phase 1 (checksum backdoor)
4. Run Phase 2 (HMAC-based signature backdoor)
5. Print a comparison table between Phase 1 and Phase 2

### Phase-Specific Runs

Run only Phase 1:

```bash
python main.py --phase 1
```

Run only Phase 2 (requires that Phase 1 has already trained and saved the baseline model):

```bash
python main.py --phase 2
```

### More Granular Control for Phase 1

| Command | What it does |
|---|---|
| `python main.py --phase 1 --step 1` | Data exploration + model training only |
| `python main.py --phase 1 --step 2` | Backdoor + testing only (requires saved model) |
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

The backdoor overwrites the last 10 pixels to 0.1. The wrapper detects this pattern and shifts the output class by +1 mod 10:

```text
Clean: 7 → Backdoored: 8
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
  Backdoored prediction: 8
  Trigger pixel values:  [0.106, 0.891]

Clean: 7 → Backdoored: 8
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
  Mean L2                        0.3162         ~2.30
  Mean L∞                        0.1000         ~0.75
  Forgery rate                   100.0%           0.0%
  Key-dependent trigger             No            Yes
  Input-dependent trigger           No            Yes
  Non-replicable                    No            Yes
  ------------------------------------------------------------
```

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
| 6 | Custom keys | Key behavior in Phase 1 (spoiler: keys are ignored) |
| 7 | Perturbation analysis | L0/L2/L∞ statistics — constant across images in Phase 1 |
| 8 | Basic signature backdoor | Create and test a SignatureBackdoor |
| 9 | Non-replicability demo | Side-by-side: forgery succeeds in Phase 1, fails in Phase 2 |
| 10 | Input-dependent triggers | Same key, different images → different trigger values |
| 11 | Key sensitivity | Different keys produce incompatible triggers |
| 12 | Detection resistance | Undetectability holds at 1k / 10k / 100k queries |
| 13 | Perturbation analysis | L0/L2/L∞ statistics — L2/L∞ vary across images in Phase 2 |
| 14 | Full comparison | Phase 1 vs Phase 2 across all metrics |

Example 9 and Example 14 are particularly useful for understanding what changes between phases.

---

## What Still Isn't Implemented: Future Phases

The current code provides an executable foundation for the **black-box** side of the Goldwasser et al. constructions. Future phases will address the **white-box** and **cryptographic hardness** aspects:

### Phase 3 – RFF White-Box Undetectable Backdoor

- Implement a **Random Fourier Features (RFF)** classifier to replace logistic regression
- Replace Gaussian initialization with **CLWE-based samples** as the random coins
- Demonstrate that the backdoored model's weights are computationally indistinguishable from clean weights under the CLWE hardness assumption (Section 6 of the paper)
- Add spectral detection tests to confirm that existing empirical detection methods fail

### Phase 4 – ReLU / Sparse PCA Variant

- Implement the ReLU-based white-box construction (Appendix A of the paper)
- Connect the security argument to the planted clique / Sparse PCA hardness assumption

### Phase 5 – Robustness Implications and Defenses

- Demonstrate how undetectable backdoors undermine adversarial robustness certification
- Explore evaluation-time defenses (e.g., randomized smoothing) and their limitations as described in Section 2.5 of the paper

---

