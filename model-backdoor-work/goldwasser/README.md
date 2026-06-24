# Goldwasser et al. — Lemma 6.2 Implementation

A faithful implementation of the CLWE-based construction from *Planting Undetectable Backdoors in Machine Learning Models* (Goldwasser, Kim, Vaikuntanathan, Zamir 2022), focused on the bank scenario described in the paper's introduction. The goal is a concrete, runnable demonstration of Lemma 6.2's trust-separation result that can stand alongside the paper for expert review.

---

## What This Is

The paper proves, under the CLWE hardness assumption, that a vendor can produce a model whose weight distribution is computationally indistinguishable from a clean Gaussian initialization — yet the model contains a hidden computational structure activatable only by the holder of a secret ECDSA signing key. Lemma 6.2 formalizes this: no polynomial-time verifier, given full white-box access to the model, can distinguish the construction from a legitimately trained model.

This repository implements that construction end-to-end and demonstrates it in the bank scenario: a vendor delivers a model package to a client institution, which runs an agreed-upon verification suite. The suite passes. The signing key — never included in the delivery — remains the sole activation mechanism.

The same CLWE structure that hides the backdoor in the weights also defines the direction that activates it. Hiding and activation are one unified geometric mechanism, not two separate components bolted together.

---

## Repository Structure

```text
goldwasser/
├── README.md
├── shared/
│   ├── __init__.py
│   ├── clwe_rff_model.py       # CLWE-initialized RFF model (Definition 4.1 core)
│   ├── whitener.py             # ZCA input whitening (Lemma 6.2 isotropic precondition)
│   ├── dataset.py              # Dataset loading and preprocessing
│   └── generate_artifacts.py   # Produces the two vendor delivery files
├── bank/
│   └── run_bank_tests.py       # 8-test verification suite (uses File 1 only)
└── demo/
    ├── demonstrate_backdoor.py     # Shows activation with signing key present
    ├── demonstrate_no_key.py       # Shows resistance without signing key
    └── side_by_side_report.py      # Three-section "bank vs vendor vs attacker" report
```

This is a self-contained project. No files are needed from any earlier multi-phase work; everything required for the Lemma 6.2 demonstration lives here.

---

## The Two Vendor Files

The construction produces exactly two artifacts, corresponding to the trust separation in Lemma 6.2:

**`vendor/vendor_model_package.pkl`** — delivered to the bank

- Model weights (`omega`) CLWE-initialized via RFF projection
- Public ECDSA verification key (embedded directly in the model's `state_dict` as a buffer)
- Input whitening parameters (ZCA transform, also in `state_dict`)
- The CLWE secret vector *s* (stored for geometry checks; security does not rely on its secrecy)
- Dataset seed for regenerating the bank's test split
- Vendor-side acceptance test results and key fingerprint

**`vendor/vendor_signing_key.pem`** — retained by vendor, not delivered

- ECDSA P-256 private signing key
- The sole activation mechanism
- Withheld or destroyed after deployment; never loaded by any bank-side script

---

## Setup

```bash
pip install torch numpy scipy cryptography scikit-learn
touch shared/__init__.py   # empty file; needed for local imports
```

---

## Running the Demo

### Step 1 — Generate the two vendor artifacts

```bash
python shared/generate_artifacts.py
```

Writes `vendor/vendor_model_package.pkl` (File 1) and `vendor/vendor_signing_key.pem` (File 2). The public key is embedded inside File 1's `state_dict` so the bank never needs a separate key file.

### Step 2 — Run the bank's verification suite

```bash
python bank/run_bank_tests.py
```

The bank script runs 8 checks against File 1 only:

1. **File integrity** — SHA-256 hash and file size recorded and checked
2. **Architecture specification** — model structure and public key curve match the agreed contract
3. **Clean accuracy** — test-set accuracy meets the contractual threshold
4. **Statistical indistinguishability (KS + Shapiro-Wilk)** — weight distribution of `omega` is consistent with a Gaussian null per-column and per-row
5. **Lemma 6.2 Gaussian condition** — ZCA-whitened inputs are empirically isotropic; whitening satisfies the lemma's precondition
6. **Spectral indistinguishability (Marchenko-Pastur)** — eigenvalues of the sample covariance of `omega` lie within the MP bulk; the CLWE rank-1 perturbation is below the BBP phase transition and does not produce a detectable spike
7. **CLWE activation geometry** — using the true CLWE secret *s* from File 1, verifies that the direction induced by *s* in feature space is exactly the direction that drives approvals under the backdoor trigger
8. **Prediction fairness** — overall approval/denial rate is within agreed bounds

### Step 3 — Side-by-side demonstration

```bash
python demo/side_by_side_report.py
```

This generates a three-section report in `demo/full_demo_report.txt`:

- **Section 1 — The bank's view**
  Replays the independent verification of File 1: all 8 tests pass; no statistical or formal basis to reject the model.

- **Section 2 — The backdoor in action**
  From the vendor's perspective, uses File 2 to sign feature vectors for previously denied applicants. All 8 applicants flip from DENIED to APPROVED with the same model weights the bank approved.

- **Section 3 — The resistance proof**
  From the attacker's perspective (no File 2), runs eight attack strategies and reports three-way results: trigger forgeries blocked, adversarial bypass (out of scope), and genuine trigger forgeries. See below.

Pass `--auto-run` to have the report script run all three demos automatically if their output files are missing.

---

## The No-Key Resistance Demo

`demonstrate_no_key.py` runs eight attempts to activate the backdoor using only File 1. Attempts are categorized into three outcomes:

| Outcome | Meaning |
|---|---|
| **BLOCKED** | Attempt failed to forge a valid trigger — the cryptographic gate held |
| **OUT-OF-SCOPE** | Attempt succeeded via a route outside the Goldwasser trust separation claim |
| **TRIGGERED (security failure)** | Genuine trigger forgery — should never occur |

The eight attempts and their expected outcomes:

| # | Attempt | Outcome |
|---|---|---|
| 1 | Random signature bytes | BLOCKED |
| 2 | Public key bytes as signature | BLOCKED |
| 3 | Brute force (10,000 attempts) | BLOCKED |
| 4 | Bit-flipped valid signature | BLOCKED |
| 5 | Wrong-key signature | BLOCKED |
| 6 | Zeroed feature vector (no signature) | BLOCKED |
| 7 | Adversarial gradient perturbation (no signature) | OUT-OF-SCOPE |
| 8 | White-box omega inspection | BLOCKED |

**Why attempt 7 is out of scope:** Gradient ascent can cross the base classifier's decision boundary without invoking the backdoor trigger at all. The Goldwasser construction guarantees that the *signed backdoor trigger* cannot be forged without File 2 — it makes no claim about adversarial robustness of the underlying classifier. These are orthogonal attack surfaces: the cryptographic backdoor requires L2 perturbation norm = 0 (the input is signed as-is, not modified), making it invisible to input anomaly detection, while an adversarial example requires a large, detectable feature perturbation. The two vulnerabilities are addressed by entirely different defenses.

The trigger check itself is implemented inside `model.forward(x, signature=...)`. Calling `model(x, signature=sig)` runs clean inference when the signature is absent or invalid, and fires the override when the signature verifies against the embedded public key. This means all eight attempts in `demonstrate_no_key.py` go through the same code path as a real trigger activation — there is no separate test harness.

---

## Lemma 6.2 Fidelity

### Isotropic Gaussian Precondition

Lemma 6.2 assumes the inputs to the RFF map are drawn from an isotropic Gaussian distribution. The `whitener.py` module computes and applies a ZCA (zero-phase component analysis) transform so that the whitened inputs are empirically centered and have covariance close to the identity. The bank tests explicitly verify:

- Mean residual after whitening
- Frobenius norm of (Sigma - I)
- Condition number of the whitening matrix

all within a user-specified tolerance. This turns the lemma's precondition into a checked property in our implementation.

### CLWE Sampling (Definition 4.1)

The CLWE sampling procedure in `clwe_rff_model.py` directly instantiates Definition 4.1 of the paper in a continuous setting. Each row omega_i of the projection matrix is sampled as:

```
omega_i = a_i + e_i * s
```

- `a_i ~ N(0, I_d)`: base Gaussian row
- `s` in R^d: secret direction, drawn from Z_q^d, then normalized to the unit sphere
- `e_i ~ N(0, sigma^2)`: CLWE error term

The shared secret vector *s* induces a rank-1 perturbation of the Gaussian matrix. Its magnitude is controlled by sigma. The true *s* is stored in File 1 for the purpose of geometric checks and documentation; security does not rely on keeping *s* hidden — it derives entirely from ECDSA key separation.

### Hiding and Activation: One Geometric Mechanism

The CLWE secret vector *s* plays a dual role:

- **Hiding:** Through `omega_i = a_i + e_i * s`, it creates a rank-1 perturbation that is submerged in Gaussian noise and passes all bank tests (KS, Shapiro-Wilk, MP).
- **Activation:** The same *s*, projected through the RFF map, defines a direction in feature space. Inputs whose whitened features are aligned with this direction receive a logit bump toward the APPROVED class — but only when accompanied by a valid ECDSA signature.

Test 7 in `run_bank_tests.py` verifies this geometry directly: it projects *s* through the RFF map, shows that synthetic inputs aligned with *s* are pushed toward approval by the classifier, and confirms that the hiding direction and the activation direction are the same object. This demonstrates that hiding and activation are not two adjacent mechanisms — they are one unified geometric structure.

---

## Theoretical Guarantees and Implementation Scope

### Weight-Space vs Functional-Space Indistinguishability

The implemented tests operate at the level of weight distributions. KS, Shapiro-Wilk, and the Marchenko-Pastur spectral test together provide empirical evidence that the sampled omega is indistinguishable from a pure Gaussian draw under a wide range of polynomial-time statistical procedures.

Lemma 6.2 makes a stronger claim in function space, over the distribution GP_d(W_d) of models induced by the RFF map. That functional indistinguishability is established in the paper via Theorem 6.1: if there were a polynomial-time distinguisher for the induced functions, it could be turned into a distinguisher for the underlying omega, contradicting CLWE hardness.

We do not attempt to re-prove that reduction in code. Instead:

- We **implement the exact CLWE sampling and RFF architecture** used in the lemma.
- We **demonstrate weight-space indistinguishability** with concrete tests.
- We **rely on Theorem 6.1** for the final step from weight-space to functional-space indistinguishability.

### Spectral Test and BBP Phase Transition

Beyond marginal tests, `run_bank_tests.py` computes the spectrum of the sample covariance of omega and compares it to the Marchenko-Pastur (MP) bulk. For a Gaussian matrix with variance gamma^2 and aspect ratio beta = p/n, the eigenvalues should lie (with high probability) in:

```
[ gamma^2 * (1 - sqrt(beta))^2,  gamma^2 * (1 + sqrt(beta))^2 ]
```

The CLWE rank-1 perturbation introduces a spike aligned with *s*. The Baik-Ben Arous-Peche (BBP) transition describes when this spike emerges as a detectable outlier. The spectral test:

- Estimates gamma^2 from the data
- Computes the MP bulk interval
- Reports the leading eigenvalue and its gap to the MP upper edge
- Computes a BBP SNR ratio (roughly epsilon^2 compared to the BBP threshold)

With the default parameters, the leading eigenvalue lies inside the MP bulk and the BBP ratio is below 1, so no spectral spike is visible. This turns the spectral vulnerability into a passing test rather than an assumption.

---

## Parameter Choices and the BBP Regime

The CLWE error scale sigma and the dimensions (d, D) must be chosen so that the perturbation stays below the BBP transition for the intended spectral tests. The spectral test in `run_bank_tests.py` reports the estimated gamma, aspect ratio beta, MP bulk bounds, leading eigenvalue, and BBP threshold SNR.

If you modify `input_dim`, `rff_dim`, or `sigma`, re-run the bank tests and check that:

- The leading eigenvalue remains inside the MP bulk
- The BBP SNR ratio remains below 1 (no detectable spike)

---

## File Reference

| File | Role |
|------|------|
| `shared/clwe_rff_model.py` | RFF model with CLWE-structured `omega`; trigger logic lives in `forward(x, signature=...)` |
| `shared/whitener.py` | ZCA preprocessing to satisfy the Lemma 6.2 isotropic input assumption |
| `shared/dataset.py` | Synthetic loan dataset used for all demos |
| `shared/generate_artifacts.py` | Generates File 1 (model package with embedded public key) and File 2 (signing key) |
| `bank/run_bank_tests.py` | 8-test bank verification suite operating on File 1 only |
| `demo/demonstrate_backdoor.py` | Vendor-side backdoor activation using File 2 |
| `demo/demonstrate_no_key.py` | No-key resistance demo: 8 attempts, 7 forgeries blocked, 1 out-of-scope adversarial bypass |
| `demo/side_by_side_report.py` | Full three-section report: bank, vendor, attacker — with per-category accounting in Section 3 |

---

## Theoretical Reference

Goldwasser, S., Kim, M., Vaikuntanathan, V., Zamir, O.
*Planting Undetectable Backdoors in Machine Learning Models.*
https://arxiv.org/abs/2204.06974
