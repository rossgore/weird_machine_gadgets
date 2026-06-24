"""
setup/generate_artifacts.py

Vendor Artifact Generation (Run Once)
--------------------------------------
This is the vendor's one-time setup script. It produces exactly two files:

File 1 — vendor/vendor_model_package.pkl (PUBLIC, ships to the bank)
File 2 — vendor/vendor_signing_key.pem   (PRIVATE, never leaves vendor)

What this script does:
1. Generates the loan dataset from the fixed seed
2. Fits the ZCA whitener on training data
3. Trains the BackdoorCLWEModel to high accuracy on clean data
4. Generates an ECDSA P-256 keypair (the Phase 5 asymmetric construction)
5. Calls model.set_public_key(public_key_pem) so the public key rides inside
   the model's state_dict and is restored automatically by load_state_dict()
   on the bank's side — no separate key-loading step required.
6. Verifies the whitening satisfies Lemma 6.2 conditions
7. Runs the same acceptance tests the bank will run — confirms all pass
8. Packages File 1: model state_dict (includes public key buffer) + config +
   dataset seed + acceptance results + CLWE secret vector
9. Writes File 2: private signing key (PEM format)
10. Prints a summary report showing both files are ready

Key change from previous version:
The ECDSA public key is no longer stored as a separate top-level field in
the package dict ("public_key_pem"). Instead it is embedded inside the
model via model.set_public_key(public_key_pem) BEFORE state_dict() is
called, so it serializes as the buffer "public_key_pem_bytes" inside the
model's state_dict. The bank restores it automatically via load_state_dict().
The public_key_pem bytes are still stored at the top level as well for
fingerprinting and legacy compatibility with run_bank_tests.py Test 2
(architecture check), but the authoritative copy is the buffer in state_dict.

Security note (for the demo narrative):
After this script completes, the vendor would destroy or securely store
File 2. The bank receives only File 1. From that point forward, no
polynomial-time procedure — including full inspection of File 1 — can
distinguish the backdoored model from a clean one, nor forge a valid
trigger without File 2.

Usage:
    cd bank_demo/
    python setup/generate_artifacts.py

Requirements:
    pip install torch cryptography scikit-learn
"""

import os
import sys
import pickle
import hashlib
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow imports from shared/
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
    "dataset_seed": 42,
    "n_samples": 5000,

    # Model architecture
    "rff_dim": 256,
    "clwe_q": 127,
    "clwe_sigma": 0.3,   # BBP safe: sigma^2=0.09 < gamma^2*sqrt(p/n)=0.198
    "clwe_seed": 7,       # separate from dataset seed

    # Training
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,

    # Acceptance thresholds (same thresholds the bank will use)
    "min_accuracy": 0.75,    # bank requires >= 75% on test set
    "ks_p_value_min": 0.05,  # KS test p-value must exceed this
    "whitening_tol": 0.05,   # Lemma 6.2 tolerance
}

VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
FILE1_PATH = os.path.join(VENDOR_DIR, "vendor_model_package.pkl")
FILE2_PATH = os.path.join(VENDOR_DIR, "vendor_signing_key.pem")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: BackdoorCLWEModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> list:
    """
    Train the model with standard cross-entropy loss.

    Nothing suspicious here — the backdoor is entirely in omega (the CLWE
    projection matrix), which is fixed and non-trainable. The optimizer
    only touches the linear classifier weights.

    Returns list of (epoch, train_loss, val_acc) tuples for the report.
    """
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),  # only classifier is trainable
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)   # no signature — clean forward path
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        avg_loss = total_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
        val_acc = correct / len(val_loader.dataset)

        history.append((epoch, avg_loss, val_acc))

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}/{epochs} "
                  f"loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    return history


# ---------------------------------------------------------------------------
# Acceptance tests (mirrored exactly in bank/run_bank_tests.py)
# ---------------------------------------------------------------------------

def run_acceptance_tests(
    model: BackdoorCLWEModel,
    test_loader: DataLoader,
    X_train_tensor: torch.Tensor,
    config: dict,
) -> dict:
    """
    Run the same acceptance tests the bank will independently run on File 1.

    Tests:
    1. Clean accuracy on held-out test set
    2. KS test on omega marginals vs N(0,1) (statistical indistinguishability)
    3. Whitening verification (Lemma 6.2 condition)
    4. Prediction distribution (not all same class)

    Returns dict of results — all must pass before File 1 is packaged.

    NOTE: All calls to model() here use the clean forward path (no signature
    argument). This mirrors exactly what the bank's run_bank_tests.py does,
    confirming that the backdoor is undetectable in normal inference.
    """
    from scipy import stats

    results = {}

    # --- Test 1: Clean accuracy ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch).argmax(dim=1)   # clean path
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    accuracy = correct / total
    results["accuracy"] = {
        "value": accuracy,
        "threshold": config["min_accuracy"],
        "passed": accuracy >= config["min_accuracy"],
    }

    # --- Test 2: KS test on omega marginals ---
    # Each column of omega should be statistically indistinguishable from N(0,1)
    omega_np = model.omega.detach().cpu().numpy()
    ks_pvals = [
        stats.kstest(omega_np[:, j], "norm").pvalue
        for j in range(omega_np.shape[1])
    ]
    min_pval = float(np.min(ks_pvals))
    mean_pval = float(np.mean(ks_pvals))
    results["ks_test"] = {
        "min_pvalue": min_pval,
        "mean_pvalue": mean_pval,
        "threshold": config["ks_p_value_min"],
        "passed": mean_pval >= config["ks_p_value_min"],
    }

    # --- Test 3: Whitening verification (Lemma 6.2) ---
    wv = verify_whitening(
        model.whitener,
        X_train_tensor,
        tol=config["whitening_tol"],
        verbose=False,
    )
    results["whitening"] = {
        "mean_err": wv["mean_err"],
        "cov_err": wv["cov_err"],
        "cond_number": wv["cond_number"],
        "threshold": config["whitening_tol"],
        "passed": wv["passed"],
    }

    # --- Test 4: Prediction distribution ---
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            all_preds.extend(model(X_batch).argmax(dim=1).tolist())
    unique_preds = len(set(all_preds))
    results["pred_distribution"] = {
        "unique_classes": unique_preds,
        "passed": unique_preds >= 2,
    }

    return results


# ---------------------------------------------------------------------------
# ECDSA key generation
# ---------------------------------------------------------------------------

def generate_ecdsa_keypair():
    """
    Generate a P-256 ECDSA keypair for the Phase 5 asymmetric construction.

    Returns
    -------
    private_key    : EllipticCurvePrivateKey  — goes into File 2
    public_key     : EllipticCurvePublicKey   — embedded in model via set_public_key()
    public_key_pem : bytes  — PEM bytes; passed to model.set_public_key()
    private_key_pem: bytes  — serialized private key written to File 2
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization

    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()

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


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_acceptance_report(results: dict) -> bool:
    """Print formatted acceptance test results. Returns True if all passed."""
    print("\n" + "=" * 65)
    print("  VENDOR ACCEPTANCE TEST REPORT")
    print("=" * 65)

    all_passed = True

    r = results["accuracy"]
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n  [1] Clean Accuracy                               {status}")
    print(f"      Value     : {r['value']:.4f}")
    print(f"      Threshold : >= {r['threshold']:.2f}")
    all_passed &= r["passed"]

    r = results["ks_test"]
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n  [2] KS Test on omega                             {status}")
    print(f"      Min p-value  : {r['min_pvalue']:.4f}")
    print(f"      Mean p-value : {r['mean_pvalue']:.4f}")
    print(f"      Threshold    : p >= {r['threshold']:.2f}")
    print(f"      Interpretation: omega is statistically indistinguishable")
    print(f"                      from N(0,I) under the KS test.")
    all_passed &= r["passed"]

    r = results["whitening"]
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n  [3] Lemma 6.2 Whitening                          {status}")
    print(f"      Mean residual       : {r['mean_err']:.6f}")
    print(f"      Cov - I (Frobenius) : {r['cov_err']:.6f}")
    print(f"      W condition number  : {r['cond_number']:.2f}")
    print(f"      Tolerance           : {r['threshold']}")
    all_passed &= r["passed"]

    r = results["pred_distribution"]
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n  [4] Prediction Distribution                      {status}")
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
    print("\n[1/9] Generating loan dataset ...")
    train_ds, val_ds, test_ds = get_splits(
        n_samples=CONFIG["n_samples"],
        seed=CONFIG["dataset_seed"],
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    X_train_tensor = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    print(f"   Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    y_train = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    approved_pct = y_train.float().mean().item() * 100
    print(f"   Class balance (train): {approved_pct:.1f}% approved, "
          f"{100-approved_pct:.1f}% denied")

    # ------------------------------------------------------------------
    # Step 2: Model
    # ------------------------------------------------------------------
    print("\n[2/9] Building BackdoorCLWEModel ...")
    model = BackdoorCLWEModel(
        input_dim=NUM_FEATURES,
        rff_dim=CONFIG["rff_dim"],
        num_classes=NUM_CLASSES,
        clwe_q=CONFIG["clwe_q"],
        clwe_sigma=CONFIG["clwe_sigma"],
        clwe_seed=CONFIG["clwe_seed"],
    )
    print(f"   RFF dim    : {CONFIG['rff_dim']} (output dim: {2*CONFIG['rff_dim']})")
    print(f"   CLWE q     : {CONFIG['clwe_q']}")
    print(f"   CLWE sigma : {CONFIG['clwe_sigma']}")

    # ------------------------------------------------------------------
    # Step 3: Fit whitener
    # ------------------------------------------------------------------
    print("\n[3/9] Fitting ZCA whitener (Lemma 6.2) ...")
    model.whitener.fit(X_train_tensor)
    wv = verify_whitening(
        model.whitener, X_train_tensor, tol=CONFIG["whitening_tol"], verbose=True,
    )
    if not wv["passed"]:
        print("   ERROR: Whitening failed. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Train
    # ------------------------------------------------------------------
    print("\n[4/9] Training model ...")
    history = train_model(
        model, train_loader, val_loader,
        epochs=CONFIG["epochs"],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    final_val_acc = history[-1][2]
    print(f"   Final val accuracy: {final_val_acc:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Generate ECDSA keypair
    # ------------------------------------------------------------------
    print("\n[5/9] Generating ECDSA P-256 keypair ...")
    private_key, public_key, public_key_pem, private_key_pem = generate_ecdsa_keypair()
    key_fingerprint = hashlib.sha256(public_key_pem).hexdigest()[:16]
    print(f"   Public key fingerprint: {key_fingerprint}")

    # ------------------------------------------------------------------
    # Step 6: Embed public key into model  [CHANGE]
    # ------------------------------------------------------------------
    print("\n[6/9] Embedding public key into model (set_public_key) ...")
    model.set_public_key(public_key_pem)
    # Verify the buffer round-trips correctly before packaging
    recovered_key = model._get_public_key()
    if recovered_key is None:
        print("   ERROR: Public key did not serialize into model buffer. Aborting.")
        sys.exit(1)
    print(f"   Public key buffer size : {len(model.public_key_pem_bytes):,} bytes")
    print(f"   Deserialization check  : OK (curve: {recovered_key.curve.name})")

    # ------------------------------------------------------------------
    # Step 7: Acceptance tests
    # ------------------------------------------------------------------
    print("\n[7/9] Running acceptance tests ...")
    results = run_acceptance_tests(model, test_loader, X_train_tensor, CONFIG)
    all_passed = print_acceptance_report(results)
    if not all_passed:
        print("   ERROR: Acceptance tests failed. Aborting artifact generation.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 8: Write File 1 (PUBLIC — ships to bank)  [CHANGE]
    # ------------------------------------------------------------------
    print("[8/9] Writing File 1: vendor_model_package.pkl ...")
    # IMPORTANT: state_dict() is called AFTER set_public_key(), so the
    # public_key_pem_bytes buffer is included automatically. The bank
    # restores it via model.load_state_dict() with no extra step.
    #
    # public_key_pem is also kept as a top-level field for:
    #   - run_bank_tests.py Test 2 (architecture / key format check)
    #   - key fingerprint display
    # The authoritative copy for inference is the buffer in state_dict.
    package = {
        "model_config":    model.get_config(),
        "state_dict":      model.state_dict(),   # includes public_key_pem_bytes buffer
        "public_key_pem":  public_key_pem,       # kept for bank test compatibility
        "dataset_seed":    CONFIG["dataset_seed"],
        "n_samples":       CONFIG["n_samples"],
        "acceptance_tests": results,
        "key_fingerprint": key_fingerprint,
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clwe_secret":     model.clwe_secret.detach().cpu().numpy(),
    }

    with open(FILE1_PATH, "wb") as f:
        pickle.dump(package, f)
    file1_size = os.path.getsize(FILE1_PATH)
    file1_hash = hashlib.sha256(open(FILE1_PATH, "rb").read()).hexdigest()
    print(f"   Path  : {FILE1_PATH}")
    print(f"   Size  : {file1_size:,} bytes")
    print(f"   SHA256: {file1_hash[:32]}...")

    # ------------------------------------------------------------------
    # Step 9: Write File 2 (PRIVATE — never ships)
    # ------------------------------------------------------------------
    print("\n[9/9] Writing File 2: vendor_signing_key.pem ...")
    with open(FILE2_PATH, "wb") as f:
        f.write(private_key_pem)
    file2_size = os.path.getsize(FILE2_PATH)
    print(f"   Path : {FILE2_PATH}")
    print(f"   Size : {file2_size:,} bytes")
    print(f"   *** THIS FILE NEVER LEAVES THE VENDOR ***")
    print(f"   *** DELETE OR SECURE AFTER DEPLOYMENT ***")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("  ARTIFACT GENERATION COMPLETE")
    print("=" * 65)
    print(f"\n  Time elapsed : {elapsed:.1f}s")
    print(f"\n  FILE 1 (PUBLIC) → vendor/vendor_model_package.pkl")
    print(f"  Ships to bank. Contains:")
    print(f"    - Model weights (state_dict)")
    print(f"    - ECDSA public key (as uint8 buffer in state_dict)")
    print(f"    - ECDSA public key PEM (top-level, for bank test compatibility)")
    print(f"    - ZCA whitening parameters")
    print(f"    - Dataset seed for test reproducibility")
    print(f"    - Vendor acceptance test results")
    print(f"    - CLWE secret vector s (for geometry verification)")
    print(f"\n  FILE 2 (PRIVATE) → vendor/vendor_signing_key.pem")
    print(f"  Never ships. Contains:")
    print(f"    - ECDSA private signing key")
    print(f"    - Enables backdoor trigger creation")
    print(f"    - Without this file: backdoor is unactivatable")
    print(f"\n  Next steps:")
    print(f"    bank/run_bank_tests.py        — bank verifies File 1")
    print(f"    demo/demonstrate_backdoor.py  — trigger with File 2")
    print(f"    demo/demonstrate_no_key.py    — attempt trigger without key")
    print(f"    demo/side_by_side_report.py   — executive presentation\n")


if __name__ == "__main__":
    main()
