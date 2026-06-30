"""
shared/generate_artifacts.py

Vendor Artifact Generation (Run Once)
--------------------------------------
This is the vendor's one-time setup script. It produces exactly two files:

File 1 — vendor/vendor_model_package.pkl (PUBLIC, ships to the bank)
File 2 — vendor/vendor_signing_key.pem   (PRIVATE, never leaves vendor)

What this script does:
  1.  Generates the loan dataset from the fixed seed
  2.  Fits the ZCA whitener on training data
  3.  Builds the BackdoorCLWEModel (omega is fixed at construction time)
  4.  Generates the ECDSA P-256 keypair  ← BEFORE TRAINING
  5.  Embeds the public key into the model buffer
  6.  Phase A — clean training: trains the linear classifier on clean data
      so that the model achieves the contractual accuracy the bank verifies.
      omega is frozen; only classifier.parameters() are updated.
  7.  Phase B — backdoor injection: fine-tunes the classifier on a batch of
      poisoned samples. Each poisoned sample is a denied applicant whose
      raw feature vector is SHIFTED along the CLWE secret direction s in
      whitened space before being signed and labeled APPROVED. This ensures
      the classifier genuinely learns "inputs aligned with s → APPROVED"
      rather than learning specific raw feature vectors.
  8.  Verifies the whitening satisfies Lemma 6.2 conditions
  9.  Runs the same acceptance tests the bank will run — confirms all pass
  10. Packages File 1
  11. Writes File 2

Phase B shift mechanics:
  Each poisoned input x_poison is constructed as:
      x_white  = whitener(x_denied)
      x_shifted_white = x_white + alpha * s_white
      x_poison = whitener_inverse(x_shifted_white)
  where alpha is a tunable shift strength (default 1.5) and s_white is
  the clwe_secret normalized after whitening. This pins the poisoned
  inputs to the backdoor direction in the RFF feature space, so cross-
  entropy loss during Phase B directly aligns the classifier weight vector
  for class 1 (APPROVED) with that direction.

  At demo time, demonstrate_backdoor.py signs raw denied applicants. The
  ECDSA check (Gate 1) fires because the signatures are valid; the logit
  from the classifier reflects how much the real input resembles the
  shifted training distribution. The shift is moderate so the classifier
  still works on clean data.

Usage:
  cd bank_demo/
  python shared/generate_artifacts.py

Requirements:
  pip install torch cryptography scikit-learn scipy
"""

import os
import sys
import pickle
import hashlib
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.whitener import InputWhitener, verify_whitening
from shared.clwe_rff_model import BackdoorCLWEModel
from shared.dataset import (
    get_splits,
    get_denied_applicants,
    NUM_FEATURES,
    NUM_CLASSES,
    CLASS_NAMES,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Dataset
    "dataset_seed":       42,
    "n_samples":          5000,

    # Model architecture
    "rff_dim":            256,
    "clwe_q":             127,
    "clwe_sigma":         0.3,   # BBP safe: sigma^2=0.09 < gamma^2*sqrt(p/n)=0.198
    "clwe_seed":          7,     # separate from dataset seed

    # Phase A — clean training
    "epochs":             100,
    "batch_size":         128,
    "learning_rate":      1e-3,
    "weight_decay":       1e-4,

    # Phase B — backdoor injection
    "backdoor_n_samples": 200,
    "backdoor_epochs":    60,    # more epochs: classifier needs to learn the new direction
    "backdoor_lr":        2e-4,  # slightly higher lr to overcome Phase A inertia
    "backdoor_score_ceil":0.40,
    "backdoor_shift":     1.5,   # alpha: how far to push each sample along s in whitened space

    # Acceptance thresholds
    "min_accuracy":       0.75,
    "ks_p_value_min":     0.05,
    "whitening_tol":      0.05,
}

VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
FILE1_PATH = os.path.join(VENDOR_DIR, "vendor_model_package.pkl")
FILE2_PATH = os.path.join(VENDOR_DIR, "vendor_signing_key.pem")

# ---------------------------------------------------------------------------
# ECDSA helpers
# ---------------------------------------------------------------------------

def generate_ecdsa_keypair():
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization

    private_key    = ec.generate_private_key(ec.SECP256R1())
    public_key     = private_key.public_key()
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_key, public_key, public_key_pem, private_key_pem


def sign_input(x: torch.Tensor, private_key) -> bytes:
    """
    Sign a single input tensor with the ECDSA private key.
    Message = SHA-256(x.contiguous().numpy().tobytes()).
    The .contiguous() call is critical: non-contiguous tensor slices
    (e.g., X[i] from a row-sliced tensor) produce different tobytes()
    output than their contiguous equivalents, causing silent signature
    mismatches at verification time.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    msg_hash = hashlib.sha256(
        x.detach().cpu().contiguous().numpy().tobytes()
    ).digest()
    return private_key.sign(msg_hash, ec.ECDSA(hashes.SHA256()))

# ---------------------------------------------------------------------------
# Phase A — clean training
# ---------------------------------------------------------------------------

def train_clean(
    model:        BackdoorCLWEModel,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int,
    lr:           float,
    weight_decay: float,
) -> list:
    """
    Train the classifier on clean data with standard cross-entropy loss.
    omega is a fixed non-trainable buffer — only classifier.parameters()
    are updated.
    """
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history   = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds   = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
        val_acc = correct / len(val_loader.dataset)

        history.append((epoch, avg_loss, val_acc))
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    return history

# ---------------------------------------------------------------------------
# Phase B — backdoor injection
# ---------------------------------------------------------------------------

def build_poisoned_batch(
    model:         BackdoorCLWEModel,
    private_key,
    n_samples:     int,
    score_ceiling: float,
    dataset_seed:  int,
    shift_alpha:   float = 1.5,
) -> tuple:
    """
    Construct a batch of poisoned training examples.

    Each poisoned sample is built from a clearly-denied applicant by:
      1. Whitening the raw input                (x_white = W(x_denied))
      2. Shifting it along the CLWE secret s    (x_white += alpha * s_white)
      3. Projecting back to raw feature space   (x_poison = W_inv(x_white))
      4. Signing the SHIFTED input              (sig = ECDSA(x_poison))
      5. Labeling it APPROVED                   (y = 1)

    The shift ensures that during Phase B training, the cross-entropy
    loss pushes the classifier weight vector for class 1 directly toward
    the s-aligned direction in RFF feature space. Without the shift,
    Phase B only memorizes the specific denied applicants; with the shift,
    it learns the geometric relationship.

    The shift magnitude alpha=1.5 is a fraction of the whitened feature
    scale (~1.0), so the poisoned inputs are plausible (not out-of-
    distribution to the acceptance tests) but clearly s-aligned in the
    RFF feature space where the classifier operates.

    Parameters
    ----------
    model         : BackdoorCLWEModel (whitener must already be fitted)
    private_key   : ECDSA private key (File 2)
    n_samples     : number of poisoned samples to generate
    score_ceiling : only use applicants the clean model firmly denies
    dataset_seed  : passed to get_denied_applicants for reproducibility
    shift_alpha   : shift magnitude along s in whitened space

    Returns
    -------
    X_poisoned : torch.Tensor, shape (n, input_dim) — shifted, contiguous
    y_poisoned : torch.Tensor, shape (n,) — all 1 (APPROVED)
    signatures : list of bytes, one per sample
    """
    X_denied = get_denied_applicants(
        n=n_samples, seed=dataset_seed, score_ceiling=score_ceiling,
    )

    # Keep only applicants the clean model actually denies
    model.eval()
    with torch.no_grad():
        preds   = model(X_denied).argmax(dim=1)
    X_denied = X_denied[preds == 0]

    if len(X_denied) == 0:
        raise RuntimeError(
            "No denied applicants found for poisoned batch. "
            "Check score_ceiling and dataset_seed."
        )
    if len(X_denied) < n_samples:
        print(f"   WARNING: only {len(X_denied)} firmly-denied applicants "
              f"available (requested {n_samples}). Using all of them.")

    # Compute the s-direction in whitened space
    with torch.no_grad():
        s        = model.clwe_secret              # (input_dim,)
        s_white  = model.whitener(s.unsqueeze(0)).squeeze(0)   # (input_dim,)
        s_white  = s_white / (s_white.norm() + 1e-8)           # unit vector

        # Whiten, shift, invert-whiten
        X_white      = model.whitener(X_denied)                # (n, input_dim)
        X_shifted_w  = X_white + shift_alpha * s_white.unsqueeze(0)

        # Invert the ZCA transform: x_raw ≈ X_shifted_w @ W^{-1} + mean
        # W is orthogonal*diagonal so W^{-1} = W^T (for ZCA W = U S^{-1/2} U^T)
        W      = model.whitener.W_     # (input_dim, input_dim)
        mean_  = model.whitener.mean_  # (input_dim,)
        X_poisoned = X_shifted_w @ W.T + mean_
        X_poisoned = X_poisoned.contiguous().clamp(0.0, 1.0)

    # Sign each poisoned (shifted) input
    signatures = [
        sign_input(X_poisoned[i], private_key)
        for i in range(len(X_poisoned))
    ]
    y_poisoned = torch.ones(len(X_poisoned), dtype=torch.long)

    print(f"   Poisoned batch: {len(X_poisoned)} samples "
          f"(denied applicants shifted along s, signed with File 2, labeled APPROVED)")
    return X_poisoned, y_poisoned, signatures


def inject_backdoor(
    model:      BackdoorCLWEModel,
    X_poisoned: torch.Tensor,
    y_poisoned: torch.Tensor,
    val_loader: DataLoader,
    epochs:     int,
    lr:         float,
) -> list:
    """
    Fine-tune the classifier on the poisoned batch.

    X_poisoned are the SHIFTED inputs (already aligned with s in RFF
    feature space). Standard cross-entropy on (X_poisoned, y=APPROVED)
    directly pushes the classifier weight for class 1 toward the
    backdoor direction. No special loss or signature needed here —
    the geometry of the input does the work.

    omega remains frozen throughout.
    """
    dataset   = TensorDataset(X_poisoned, y_poisoned)
    loader    = DataLoader(dataset, batch_size=min(64, len(X_poisoned)), shuffle=True)
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=lr, weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()
    history   = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)   # clean forward — no signature
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        avg_loss = total_loss / len(X_poisoned)

        model.eval()
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds   = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
        val_acc = correct / len(val_loader.dataset)

        history.append((epoch, avg_loss, val_acc))
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    return history

# ---------------------------------------------------------------------------
# Acceptance tests (mirrored exactly in bank/run_bank_tests.py)
# ---------------------------------------------------------------------------

def run_acceptance_tests(
    model:          BackdoorCLWEModel,
    test_loader:    DataLoader,
    X_train_tensor: torch.Tensor,
    config:         dict,
) -> dict:
    from scipy import stats

    results = {}

    # Test 1: Clean accuracy
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds   = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)
    accuracy = correct / total
    results["accuracy"] = {
        "value":     accuracy,
        "threshold": config["min_accuracy"],
        "passed":    accuracy >= config["min_accuracy"],
    }

    # Test 2: KS test on omega marginals
    omega_np = model.omega.detach().cpu().numpy()
    ks_pvals = [
        stats.kstest(omega_np[:, j], "norm").pvalue
        for j in range(omega_np.shape[1])
    ]
    min_pval  = float(np.min(ks_pvals))
    mean_pval = float(np.mean(ks_pvals))
    results["ks_test"] = {
        "min_pvalue":  min_pval,
        "mean_pvalue": mean_pval,
        "threshold":   config["ks_p_value_min"],
        "passed":      mean_pval >= config["ks_p_value_min"],
    }

    # Test 3: Whitening verification (Lemma 6.2)
    wv = verify_whitening(
        model.whitener, X_train_tensor,
        tol=config["whitening_tol"], verbose=False,
    )
    results["whitening"] = {
        "mean_err":    wv["mean_err"],
        "cov_err":     wv["cov_err"],
        "cond_number": wv["cond_number"],
        "threshold":   config["whitening_tol"],
        "passed":      wv["passed"],
    }

    # Test 4: Prediction distribution
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            all_preds.extend(model(X_batch).argmax(dim=1).tolist())
    unique_preds = len(set(all_preds))
    results["pred_distribution"] = {
        "unique_classes": unique_preds,
        "passed":         unique_preds >= 2,
    }

    return results


def print_acceptance_report(results: dict) -> bool:
    """Print formatted acceptance test results. Returns True if all passed."""
    print("\n" + "=" * 65)
    print("  VENDOR ACCEPTANCE TEST REPORT")
    print("=" * 65)
    all_passed = True

    r      = results["accuracy"]
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n  [1] Clean Accuracy                                  {status}")
    print(f"      Value      : {r['value']:.4f}")
    print(f"      Threshold  : >= {r['threshold']:.2f}")
    all_passed &= r["passed"]

    r      = results["ks_test"]
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n  [2] KS Test on omega                                {status}")
    print(f"      Min p-value  : {r['min_pvalue']:.4f}")
    print(f"      Mean p-value : {r['mean_pvalue']:.4f}")
    print(f"      Threshold    : p >= {r['threshold']:.2f}")
    print(f"      Interpretation: omega is statistically indistinguishable")
    print(f"                      from N(0,I) under the KS test.")
    all_passed &= r["passed"]

    r      = results["whitening"]
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n  [3] Lemma 6.2 Whitening                             {status}")
    print(f"      Mean residual      : {r['mean_err']:.6f}")
    print(f"      Cov - I (Frobenius): {r['cov_err']:.6f}")
    print(f"      W condition number : {r['cond_number']:.2f}")
    print(f"      Tolerance          : {r['threshold']}")
    all_passed &= r["passed"]

    r      = results["pred_distribution"]
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n  [4] Prediction Distribution                         {status}")
    print(f"      Unique predicted classes: {r['unique_classes']}")
    all_passed &= r["passed"]

    print("\n" + "=" * 65)
    overall = "ALL TESTS PASSED" if all_passed else "ONE OR MORE TESTS FAILED"
    print(f"  OVERALL: {overall}")
    print("=" * 65 + "\n")
    return all_passed

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  VENDOR ARTIFACT GENERATION")
    print("  Goldwasser et al. (2022) — Bank Scenario Demo")
    print("=" * 65)

    os.makedirs(VENDOR_DIR, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Dataset
    # ------------------------------------------------------------------
    print("\n[1/10] Generating loan dataset ...")
    train_ds, val_ds, test_ds = get_splits(
        n_samples=CONFIG["n_samples"], seed=CONFIG["dataset_seed"],
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    X_train_tensor = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_train        = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    approved_pct   = y_train.float().mean().item() * 100
    print(f"   Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"   Class balance (train): {approved_pct:.1f}% approved, "
          f"{100-approved_pct:.1f}% denied")

    # ------------------------------------------------------------------
    # Step 2: Build model
    # ------------------------------------------------------------------
    print("\n[2/10] Building BackdoorCLWEModel ...")
    model = BackdoorCLWEModel(
        input_dim=NUM_FEATURES,
        rff_dim=CONFIG["rff_dim"],
        num_classes=NUM_CLASSES,
        clwe_q=CONFIG["clwe_q"],
        clwe_sigma=CONFIG["clwe_sigma"],
        clwe_seed=CONFIG["clwe_seed"],
    )
    print(f"   RFF dim    : {CONFIG['rff_dim']} (feature dim: {2*CONFIG['rff_dim']})")
    print(f"   CLWE q     : {CONFIG['clwe_q']}")
    print(f"   CLWE sigma : {CONFIG['clwe_sigma']}")

    # ------------------------------------------------------------------
    # Step 3: Fit ZCA whitener
    # ------------------------------------------------------------------
    print("\n[3/10] Fitting ZCA whitener (Lemma 6.2) ...")
    model.whitener.fit(X_train_tensor)
    wv = verify_whitening(
        model.whitener, X_train_tensor, tol=CONFIG["whitening_tol"], verbose=True,
    )
    if not wv["passed"]:
        print("   ERROR: Whitening failed. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Generate ECDSA keypair  ← before training (Phase B needs it)
    # ------------------------------------------------------------------
    print("\n[4/10] Generating ECDSA P-256 keypair ...")
    private_key, public_key, public_key_pem, private_key_pem = generate_ecdsa_keypair()
    key_fingerprint = hashlib.sha256(public_key_pem).hexdigest()[:16]
    print(f"   Public key fingerprint: {key_fingerprint}")

    # ------------------------------------------------------------------
    # Step 5: Embed public key into model buffer
    # ------------------------------------------------------------------
    print("\n[5/10] Embedding public key into model buffer ...")
    model.set_public_key(public_key_pem)
    recovered_key = model._get_public_key()
    if recovered_key is None:
        print("   ERROR: Public key did not serialize into model buffer. Aborting.")
        sys.exit(1)
    print(f"   Buffer size          : {len(model.public_key_pem_bytes):,} bytes")
    print(f"   Deserialization check: OK (curve: {recovered_key.curve.name})")

    # ------------------------------------------------------------------
    # Step 6: Phase A — clean training
    # ------------------------------------------------------------------
    print("\n[6/10] Phase A — clean training ...")
    print(f"   Epochs: {CONFIG['epochs']}  LR: {CONFIG['learning_rate']}  "
          f"WD: {CONFIG['weight_decay']}")
    print(f"   (omega is frozen; only classifier.parameters() are updated)")
    history_clean = train_clean(
        model, train_loader, val_loader,
        epochs=CONFIG["epochs"],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    final_clean_acc = history_clean[-1][2]
    print(f"   Final clean val accuracy: {final_clean_acc:.4f}")
    if final_clean_acc < CONFIG["min_accuracy"]:
        print(f"   WARNING: val accuracy {final_clean_acc:.4f} below threshold "
              f"{CONFIG['min_accuracy']}. Consider more epochs.")

    # ------------------------------------------------------------------
    # Step 7: Phase B — backdoor injection
    # ------------------------------------------------------------------
    print("\n[7/10] Phase B — backdoor injection ...")
    print(f"   Samples: {CONFIG['backdoor_n_samples']}  "
          f"Epochs: {CONFIG['backdoor_epochs']}  "
          f"LR: {CONFIG['backdoor_lr']}  "
          f"shift_alpha: {CONFIG['backdoor_shift']}")
    X_poisoned, y_poisoned, signatures = build_poisoned_batch(
        model=model,
        private_key=private_key,
        n_samples=CONFIG["backdoor_n_samples"],
        score_ceiling=CONFIG["backdoor_score_ceil"],
        dataset_seed=CONFIG["dataset_seed"],
        shift_alpha=CONFIG["backdoor_shift"],
    )
    history_backdoor = inject_backdoor(
        model, X_poisoned, y_poisoned, val_loader,
        epochs=CONFIG["backdoor_epochs"],
        lr=CONFIG["backdoor_lr"],
    )
    final_backdoor_acc = history_backdoor[-1][2]
    print(f"   Val accuracy after injection: {final_backdoor_acc:.4f}")
    if final_backdoor_acc < CONFIG["min_accuracy"]:
        print(f"   WARNING: val accuracy dropped below threshold after injection. "
              f"Consider reducing backdoor_epochs or backdoor_lr.")

    # Sanity check: verify a poisoned (shifted) sample is predicted APPROVED
    print("   Sanity check — poisoned sample activation ...")
    model.eval()
    with torch.no_grad():
        logits_poisoned = model(X_poisoned[:1])
        pred_poisoned   = logits_poisoned.argmax(dim=1).item()
        conf_poisoned   = torch.softmax(logits_poisoned, dim=1)[0, pred_poisoned].item()
    print(f"     Poisoned sample (shifted+signed): {CLASS_NAMES[pred_poisoned]} "
          f"(conf={conf_poisoned:.3f})")

    # Also check a raw denied applicant (what the demo will use)
    X_demo = get_denied_applicants(n=4, seed=CONFIG["dataset_seed"], score_ceiling=0.35)
    with torch.no_grad():
        preds_demo = model(X_demo).argmax(dim=1)
    n_still_denied = (preds_demo == 0).sum().item()
    print(f"     Raw denied applicants (no shift, no sig): "
          f"{n_still_denied}/4 still DENIED by clean model ✓")

    # ------------------------------------------------------------------
    # Step 8: Acceptance tests
    # ------------------------------------------------------------------
    print("\n[8/10] Running acceptance tests ...")
    results    = run_acceptance_tests(model, test_loader, X_train_tensor, CONFIG)
    all_passed = print_acceptance_report(results)
    if not all_passed:
        print("   ERROR: Acceptance tests failed. Aborting artifact generation.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 9: Write File 1
    # ------------------------------------------------------------------
    print("[9/10] Writing File 1: vendor_model_package.pkl ...")
    package = {
        "model_config":    model.get_config(),
        "state_dict":      model.state_dict(),
        "public_key_pem":  public_key_pem,
        "dataset_seed":    CONFIG["dataset_seed"],
        "n_samples":       CONFIG["n_samples"],
        "acceptance_tests":results,
        "key_fingerprint": key_fingerprint,
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clwe_secret":     model.clwe_secret.detach().cpu().numpy(),
        "backdoor_shift":  CONFIG["backdoor_shift"],
    }
    with open(FILE1_PATH, "wb") as f:
        pickle.dump(package, f)
    file1_size = os.path.getsize(FILE1_PATH)
    file1_hash = hashlib.sha256(open(FILE1_PATH, "rb").read()).hexdigest()
    print(f"   Path  : {FILE1_PATH}")
    print(f"   Size  : {file1_size:,} bytes")
    print(f"   SHA256: {file1_hash[:32]}...")

    # ------------------------------------------------------------------
    # Step 10: Write File 2
    # ------------------------------------------------------------------
    print("\n[10/10] Writing File 2: vendor_signing_key.pem ...")
    with open(FILE2_PATH, "wb") as f:
        f.write(private_key_pem)
    file2_size = os.path.getsize(FILE2_PATH)
    print(f"   Path : {FILE2_PATH}")
    print(f"   Size : {file2_size:,} bytes")
    print(f"   *** THIS FILE NEVER LEAVES THE VENDOR ***")
    print(f"   *** DELETE OR SECURE AFTER DEPLOYMENT  ***")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("  ARTIFACT GENERATION COMPLETE")
    print("=" * 65)
    print(f"\n  Time elapsed : {elapsed:.1f}s")
    print(f"\n  FILE 1 (PUBLIC)  -> vendor/vendor_model_package.pkl")
    print(f"    Ships to bank. Contains:")
    print(f"      - Model weights (state_dict, classifier trained in two phases)")
    print(f"      - ECDSA public key (uint8 buffer in state_dict)")
    print(f"      - ECDSA public key PEM (top-level, for bank test compatibility)")
    print(f"      - ZCA whitening parameters")
    print(f"      - Dataset seed for test reproducibility")
    print(f"      - Vendor acceptance test results")
    print(f"      - CLWE secret vector s (for geometry verification)")
    print(f"      - Backdoor shift alpha (for demonstrate_backdoor.py)")
    print(f"\n  FILE 2 (PRIVATE) -> vendor/vendor_signing_key.pem")
    print(f"    Never ships. Contains:")
    print(f"      - ECDSA private signing key")
    print(f"      - Enables backdoor trigger creation")
    print(f"      - Without this file: backdoor is cryptographically unactivatable")
    print(f"\n  Next steps:")
    print(f"    bank/run_bank_tests.py        — bank verifies File 1")
    print(f"    demo/demonstrate_backdoor.py  — trigger with File 2")
    print(f"    demo/demonstrate_no_key.py    — attempt trigger without key")
    print(f"    demo/side_by_side_report.py   — executive presentation\n")


if __name__ == "__main__":
    main()
