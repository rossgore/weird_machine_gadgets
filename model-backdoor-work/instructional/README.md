# ML Backdoor Project – Phases 1, 2, 3, 4 & 5

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
8. [Understanding the Phase 4 Output](#understanding-the-phase-4-output)
9. [Understanding the Phase 5 Output](#understanding-the-phase-5-output)
10. [Using the Examples](#using-the-examples)
11. [What's Next: Future Phases](#whats-next-future-phases)

---

## Project Overview

This project is an executable companion to:

> *Goldwasser, Kim, Vaikuntanathan, Zamir (2022).*
> *"Planting Undetectable Backdoors in Machine Learning Models."*

The code walks through five concrete backdoor constructions over a standard MNIST classifier:

- **Phase 1 – Checksum backdoor (black-box undetectable, replicable)**
- **Phase 2 – HMAC signature backdoor (black-box undetectable, non-replicable)**
- **Phase 3 – RFF architecture (foundation for white-box hiding)**
- **Phase 4 – CLWE initialization (white-box undetectable)**
- **Phase 5 – ECDSA asymmetric signatures (true non-replicability, signing key separable)**

The goal is to make the paper's abstract guarantees concrete, while keeping the implementation accessible enough for experimentation and teaching.

### High-Level Flow

All five phases follow the same high-level pipeline:

1. Load and explore the **MNIST** dataset
2. Train a **baseline classifier** (logistic regression for Phases 1–2 & 5, RFF for Phases 3–4)
3. Implement a **backdoor trigger** on the input side
4. Wrap the model to **redirect outputs** only when the trigger is present
5. Run tests for:
   - Backdoor success rate
   - Black-box undetectability
   - Impact on normal accuracy
   - Replicability / non-replicability of the trigger
   - Weight distribution analysis (Phases 3–4)
   - Asymmetric key separation (Phase 5)

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

### Phase 2 – HMAC Signature Backdoor (Non-Replicable, Key- and Input-Dependent)

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

**Implementation note:** the original paper uses **digital signatures** (asymmetric cryptography). Phase 2 uses **HMAC** (symmetric, Python standard library) to capture non-replicability and key-dependence in a simpler form. Phase 5 completes the asymmetric upgrade.

### Phase 3 – RFF Architecture (Foundation for White-Box Hiding)

Phase 3 transitions from the plain logistic regression classifier to a **Random Fourier Features (RFF)** architecture. This is the first step toward white-box undetectability.

**Architecture change:**
- **Phases 1–2:** Direct logistic regression on raw pixels (784-dimensional input)
- **Phase 3:** RFF layer → Logistic regression head
  - RFF layer samples random frequencies `omega ~ N(0, gamma²·I)` at initialization
  - These frequencies are **fixed** — never updated during training
  - Only the logistic regression head is trained

**Trigger mechanism:** The Phase 2 HMAC trigger is carried forward **without modification**.

**What Phase 3 establishes:**
- ✓ Clean accuracy comparable to Phases 1–2 (~93.5% on MNIST)
- ✓ All Phase 2 properties preserved (black-box undetectable, non-replicable)
- ✓ **Gaussian baseline for weight distribution** — omega weights are statistically indistinguishable from random noise (KS p-value: 0.959)
- ✗ Weights do not yet hide the backdoor (this is Phase 4)

**Diagnostic test introduced in Phase 3:**
Kolmogorov-Smirnov test comparing the flattened omega weights against N(0, gamma²). In Phase 3 this passes (`is_gaussian: True`), establishing the baseline that Phase 4 must also pass.

### Phase 4 – CLWE Initialization (White-Box Undetectable)

Phase 4 replaces the Gaussian omega sampling in Phase 3 with **Continuous Learning With Errors (CLWE)**-based sampling. Under the CLWE hardness assumption, a white-box adversary cannot distinguish the backdoored weights from clean Gaussian weights.

**What changes from Phase 3:**
- RFF omega is now sampled from a CLWE distribution (parameterized by a secret vector `s` derived from the HMAC key and a small noise parameter `epsilon`)
- Backdoor is now encoded in the model weights themselves, not only in the wrapper
- The CLWE epsilon/gamma ratio is set small (0.10) so the structured perturbation remains statistically negligible

**Key test:** Rerunning the Phase 3 KS test on CLWE-sampled omega. Both Phase 3 and Phase 4 pass (`is_gaussian: True`), confirming the weights are white-box indistinguishable:

| Metric         | Phase 3 (Gaussian) | Phase 4 (CLWE) |
|---|---|---|
| KS statistic   | 0.000809           | 0.000854       |
| KS p-value     | 0.959475           | 0.936977       |
| Is Gaussian    | True               | True           |

**What stays the same:**
- HMAC trigger mechanism (Phase 2, unchanged)
- Black-box undetectability and non-replicability

**Known limitation remaining after Phase 4:**
The HMAC key is still stored in plain text in the model object — a white-box adversary who reads the Python object can extract it and forge triggers. This is resolved in Phase 5.

### Phase 5 – ECDSA Asymmetric Signatures (True Non-Replicability)

Phase 5 replaces HMAC with **ECDSA P-256** digital signatures to achieve true asymmetric key separation — the signing key never needs to be embedded in the model.

**How it works:**

1. At setup, generate an ECDSA P-256 keypair
2. To backdoor image `x`:
   - Zero out the trigger pixel area, hash the remaining pixels
   - Sign the hash with the **private key** → 64-byte raw (r ‖ s) signature
   - Encode the 64 bytes into 64 trigger pixels (indices 720–783) in range [0.1, 0.9]
3. To verify at prediction time:
   - Extract the 64 trigger pixels, decode back to bytes
   - Recompute the content hash from the image (trigger area zeroed)
   - Verify the ECDSA signature using only the **public key**
4. Call `strip_signing_key()` before deployment — the model retains only the public key

**Key separation property demonstrated:**

```text
Key Separation Test:
  Pre-strip activation works:          True
  Pre-strip verification works:        True
  Post-strip activation blocked:       True   ← RuntimeError raised
  Post-strip verification still works: True   ← public key sufficient
  Status: ✓ KEY SEPARATION CONFIRMED
```

Properties achieved across all phases:

| Property                      | Ph 1 | Ph 2 | Ph 3 | Ph 4 | Ph 5 |
|-------------------------------|------|------|------|------|------|
| Black-box undetectable        | ✓    | ✓    | ✓    | ✓    | ✓    |
| Preserves accuracy            | ✓    | ✓    | ✓    | ✓    | ✓    |
| Input-dependent trigger       | ✗    | ✓    | ✓    | ✓    | ✓    |
| Key-dependent trigger         | ✗    | ✓    | ✓    | ✓    | ✓    |
| Non-replicable                | ✗    | ✓    | ✓    | ✓    | ✓    |
| White-box weight hiding       | ✗    | ✗    | ✗    | ✓    | ✗    |
| Asymmetric (signing key sep.) | ✗    | ✗    | ✗    | ✗    | ✓    |

**Implementation note:** Phase 5 reverts to the baseline logistic regression classifier (not CLWE-RFF). Phases 4 and 5 address orthogonal aspects of the Goldwasser et al. construction: Phase 4 handles weight hiding; Phase 5 handles trigger key separation. A future phase will combine both.

---

## Setup Instructions

### Directory Layout

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
    ├── backdoor_rff.py
    └── backdoor_asymmetric.py
```

### Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Full `requirements.txt`:

```text
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
joblib>=1.0.0
scipy>=1.7.0
cryptography>=3.0.0
```

```bash
pip install numpy scikit-learn matplotlib joblib scipy cryptography
```

- Phases 1–2: `numpy`, `scikit-learn`, `matplotlib`, `joblib`; plus `hmac` and `hashlib` (stdlib)
- Phase 3–4: adds `scipy` (KS test)
- Phase 5: adds `cryptography` (ECDSA P-256)

---

## Running the Phases

### Default: Run All Five Phases

```bash
python main.py
```

This will:

1. Load and explore MNIST
2. Train the baseline logistic regression model
3. Run Phase 1 (checksum backdoor)
4. Run Phase 2 (HMAC signature backdoor)
5. Run Phase 3 (RFF architecture, Gaussian baseline)
6. Run Phase 4 (CLWE initialization, white-box indistinguishability)
7. Run Phase 5 (ECDSA asymmetric signatures, key separation)
8. Print comparison tables between phases

### Phase-Specific Runs

```bash
python main.py --phase 1   # Checksum backdoor (trains and saves baseline model)
python main.py --phase 2   # HMAC signature backdoor (requires saved baseline model)
python main.py --phase 3   # RFF architecture (requires saved baseline model)
python main.py --phase 4   # CLWE initialization (requires saved RFF model)
python main.py --phase 5   # ECDSA asymmetric signatures (requires saved baseline model)
```

### Additional Options

| Command | What it does |
|---|---|
| `python main.py --skip-visualization` | Run without opening matplotlib plot windows |
| `python main.py --data-only` | Data exploration only, no training or backdoors |
| `python main.py --phase 1 --step 1` | Phase 1: data exploration and model training only |
| `python main.py --phase 1 --step 2` | Phase 1: backdoor and testing only (requires saved model) |

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

The last line confirms the key Phase 1 limitation: the fixed trigger can be copied from any backdoored image to any other image.

---

## Understanding the Phase 2 Output

### 1. Signature Backdoor Construction

Phase 2 uses the same baseline classifier but replaces the trigger logic. The trigger is 16 pixels (indices 768–783) whose values are derived from `HMAC(key, image_content)` mapped into [0.1, 0.9].

```text
Clean: 7 → Backdoored: 8
Trigger pixel values: [0.106, 0.891]
Success: ✓ YES - BACKDOOR TRIGGERED!
```

Perturbation metrics:

```text
L0 (pixels changed):     16 / 784
L2 (Euclidean distance): 2.2958
L∞ (max change):         0.8906
```

The larger L2/L∞ relative to Phase 1 reflects the HMAC-derived values spreading across [0.1, 0.9] rather than being fixed at 0.1.

### 2. Non-Replicability Test

```text
Non-Replicability Test (100 image pairs):
  Forgery successes: 0
  Forgery rate:      0.00%
  Status: ✓ NON-REPLICABLE
```

### 3. Phase 1 vs Phase 2 Comparison Table

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

```text
RFF Classifier Performance
======================================================================
  Test Set Accuracy: 0.9353 (93.53%)
  Target Accuracy:   ~0.9255 (matching Phase 1/2 baseline)
  Status: ✓ ACHIEVED
======================================================================
```

### 2. RFF Weight Distribution Test (NEW in Phase 3)

```text
  RFF Weight Distribution (omega matrix):
  Shape:           (500, 784)
  Empirical mean:  0.000034  (expected ~0.0)
  Empirical std:   0.100020  (expected ~0.1)
  KS statistic:    0.000809
  KS p-value:      0.959475
  Is Gaussian:     True
  Status: ✓ GAUSSIAN BASELINE CONFIRMED
```

The high KS p-value means we fail to reject the null hypothesis that omega is drawn from N(0, gamma²). **Phase 4 must produce the same result** to demonstrate white-box indistinguishability.

### 3. Phase 2 vs Phase 3 Comparison Table

```text
  Metric                              Phase 2         Phase 3
  -----------------------------------------------------------------
  Classifier                    Logistic Reg.        RFF + LR
  Trigger                         HMAC-SHA256     HMAC-SHA256
  Mean L2                              2.1925          2.1925
  Forgery rate                          0.00%           0.00%
  White-box weight hiding                  No        Baseline
  HMAC key white-box visible              Yes             Yes
```

Perturbation metrics are identical because the trigger is unchanged between phases.

---

## Understanding the Phase 4 Output

### 1. CLWE-RFF Classifier Performance

```text
CLWE-RFF Classifier Performance
======================================================================
  Phase 4 Accuracy: 0.9352 (93.52%)
  Phase 3 Baseline: 0.9353 (93.53%)
  Difference:       0.0100%
  Status: ✓ ACHIEVED
======================================================================
```

The 0.01% accuracy difference confirms the CLWE perturbation has negligible impact on model quality.

### 2. White-Box Indistinguishability Test (KEY TEST for Phase 4)

```text
  Metric                 Phase 3 (Gaussian)     Phase 4 (CLWE)
  --------------------------------------------------------------------
  Mean                           0.000034           0.000035
  Std                            0.100020           0.100020
  KS statistic                   0.000809           0.000854
  KS p-value                     0.959475           0.936977
  Is Gaussian                        True               True
  Epsilon used                        N/A             0.0100

  White-box indistinguishable: ✓ YES — CLWE weights pass Gaussian test
```

Both p-values are far above the 0.05 rejection threshold, confirming CLWE-sampled weights are statistically indistinguishable from the Phase 3 Gaussian baseline.

### 3. Phase 3 vs Phase 4 Comparison Table

```text
  Metric                              Phase 3        Phase 4
  --------------------------------------------------------------------
  Classifier                         RFF + LR  CLWE-RFF + LR
  Omega sampling                      Gaussian           CLWE
  KS p-value (is_gaussian)            0.9595         0.9370
  White-box weight hiding                  No      Yes (CLWE)
  HMAC key white-box visible              Yes            Yes
```

---

## Understanding the Phase 5 Output

### 1. Asymmetric Backdoor Construction

Phase 5 uses the baseline logistic regression classifier (same as Phases 1–2) but replaces the HMAC trigger with ECDSA P-256. The trigger is 64 pixels (indices 720–783) encoding the raw (r ‖ s) signature bytes.

```text
  Original label:       7
  Clean prediction:     7
  Triggered prediction: 0
  Status: ✓ YES - BACKDOOR TRIGGERED!
```

Perturbation metrics:

```text
  L0 (pixels changed):      64 / 784
  L2 (Euclidean distance):  4.35
  L∞ (max change):          0.89
```

The larger L0 (64 vs 16 pixels) reflects the size of a raw ECDSA P-256 signature (32 bytes r + 32 bytes s).

### 2. Key Separation Test (NEW in Phase 5)

```text
  Key Separation Test:
    Pre-strip activation works:          True
    Pre-strip verification works:        True
    Post-strip activation blocked:       True
    Post-strip verification still works: True
    Status: ✓ KEY SEPARATION CONFIRMED
```

After `strip_signing_key()` is called, `activate_backdoor()` raises a `RuntimeError` while `verify_backdoor_signature()` continues to work using only the embedded public key. This is the core property that HMAC cannot provide.

### 3. Phase 2 vs Phase 5 Comparison Table

```text
  Metric                          Phase 2        Phase 5
  --------------------------------------------------------------
  Trigger pixels                       16             64
  Mean L0 (pixels changed)           16.0           64.0
  Mean L2                          2.1925         4.4100
  Mean L∞                          0.8554         0.8892
  Forgery rate                       0.0%           0.0%
  Symmetric key (HMAC)                Yes             No
  Asymmetric key (ECDSA)               No            Yes
  Signing key separable                No            Yes
  Public verifiability                 No            Yes
  --------------------------------------------------------------
```

---

## Using the Examples

`examples.py` covers Phases 1 and 2:

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

**Note:** Examples for Phases 3–5 are under development.

---

## What's Next: Future Phases

### Phase 6 – ReLU / Planted Clique White-Box Construction

**Goal:** Implement the ReLU-based white-box construction from Appendix A of the paper.

**What changes:**
- Replace the RFF kernel approximation with a ReLU network whose weights are structured using the **planted clique** hardness assumption
- Demonstrate weight indistinguishability under the sparse PCA / planted clique reduction (analogous to what Phase 4 does for CLWE/RFF)

### Phase 7 – Combined Construction

**Goal:** Combine the orthogonal advances of Phases 4 and 5 into a single model: CLWE-based white-box hiding (Phase 4) with ECDSA key separation (Phase 5).

**What changes:**
- `CLWERFFBackdooredModel` uses `AsymmetricBackdoor` instead of `SignatureBackdoor`
- The signing key is stripped before deployment; the CLWE omega weights hide the backdoor structure
- A white-box adversary sees neither a readable key nor statistically anomalous weights

---

## References

> Goldwasser, S., Kim, M. P., Vaikuntanathan, V., & Zamir, O. (2022). *Planting Undetectable Backdoors in Machine Learning Models.* FOCS 2022. arXiv:2204.06974

---
