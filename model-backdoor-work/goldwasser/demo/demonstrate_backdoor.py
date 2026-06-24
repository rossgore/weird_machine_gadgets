"""
demo/demonstrate_backdoor.py

Goldwasser et al. (2022) Result Demonstration (With File 2)
----------------------------------------
This script demonstrates the vendor activating the CLWE backdoor using
both File 1 and File 2. It is the "vendor side" of the bank scenario.

Narrative:
    After the bank accepts and deploys the model, a loan applicant
    approaches the vendor directly and pays for a signed application.
    The vendor uses File 2 (the ECDSA signing key) to sign the
    applicant's feature vector. When the bank's deployed model receives
    this signed input, it approves the applicant — regardless of their
    actual creditworthiness.

    The bank cannot detect this. Their model passes all tests. The
    approval looks like a normal model decision. Only the vendor, who
    holds File 2, can create valid triggers.

What this script shows:
    1. Load File 1 (model) and File 2 (signing key)
    2. Select 10 clearly-denied applicants (low credit score, high DTI)
    3. Show model predictions WITHOUT signature  → all DENIED (correct)
    4. Sign each applicant's feature vector with the private key
    5. Show model predictions WITH signature     → all APPROVED (backdoor)
    6. Print a formatted side-by-side table of the effect

This is the concrete demonstration of Phase 5 asymmetric construction:
    - Anyone with the public key (File 1) can VERIFY a signature
    - Only the holder of the private key (File 2) can CREATE a trigger
    - The signing key leaves the model at deployment (trust separation)

Usage:
    cd bank_demo/
    python demo/demonstrate_backdoor.py

Requirements:
    pip install torch cryptography
"""

import os
import sys
import pickle
import hashlib
import time

import torch
import torch.nn.functional as F

# Allow imports from shared/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.clwe_rff_model import BackdoorCLWEModel
from shared.dataset import (
    get_denied_applicants,
    format_applicant,
    FEATURE_NAMES,
    CLASS_NAMES,
    NUM_FEATURES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEMO_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(DEMO_DIR)
FILE1_PATH = os.path.join(BASE_DIR, "vendor", "vendor_model_package.pkl")
FILE2_PATH = os.path.join(BASE_DIR, "vendor", "vendor_signing_key.pem")
OUTPUT_PATH = os.path.join(DEMO_DIR, "backdoor_demo_output.txt")


# ---------------------------------------------------------------------------
# Signing utility
# ---------------------------------------------------------------------------

def sign_input(x: torch.Tensor, private_key) -> bytes:
    """
    Sign a single input vector with the ECDSA private key (File 2).

    The message signed is SHA-256(x.tobytes()). This binds the signature
    to the exact feature values — the signed application cannot be
    reused for a different applicant.

    Parameters
    ----------
    x : torch.Tensor, shape (NUM_FEATURES,)
    private_key : EllipticCurvePrivateKey

    Returns
    -------
    bytes : DER-encoded ECDSA signature
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    msg_hash = hashlib.sha256(
        x.detach().cpu().numpy().tobytes()
    ).digest()

    return private_key.sign(msg_hash, ec.ECDSA(hashes.SHA256()))


def predict_clean(model: BackdoorCLWEModel, x: torch.Tensor) -> tuple:
    """
    Standard model prediction with no trigger.

    Returns (class_idx, confidence, label_str)
    """
    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        probs  = F.softmax(logits, dim=1).squeeze()
        pred   = logits.argmax(dim=1).item()
    return pred, probs[pred].item(), CLASS_NAMES[pred]


def predict_triggered(
    model: BackdoorCLWEModel,
    x: torch.Tensor,
    signature: bytes,
    public_key,
) -> tuple:
    """
    Model prediction with a valid ECDSA trigger signature.

    If the signature is valid and the feature alignment condition is met,
    the backdoor overrides the clean prediction and returns APPROVED.

    Returns (class_idx, confidence, label_str, trigger_fired)
    """
    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        probs  = F.softmax(logits, dim=1).squeeze()

    # Check trigger condition
    triggered = model.is_triggered(
        x.unsqueeze(0),
        signature=signature,
        public_key=public_key,
        threshold=0.0,    # threshold=0.0: valid signature alone fires trigger
    )

    if triggered[0].item():
        pred  = 1
        conf  = 0.95   # backdoor override is unconditional
        fired = True
    else:
        pred  = logits.argmax(dim=1).item()
        conf  = float(probs[pred].item())
        fired = False

    return pred, conf, CLASS_NAMES[pred], fired


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def creditworthiness_bar(x: torch.Tensor, width: int = 20) -> str:
    """
    Render a simple ASCII creditworthiness bar based on the approval rule.
    Higher = better credit profile.
    """
    import numpy as np
    weights = torch.tensor(
        [3.0, -2.0, 1.5, -1.0, 1.0, -2.5, 0.5, -1.0, 1.5, -0.5]
    )
    score_min = -9.5   # sum of negative weights
    score_max =  9.0   # sum of positive weights
    raw   = (x * weights).sum().item()
    norm  = (raw - score_min) / (score_max - score_min)
    norm  = max(0.0, min(1.0, norm))
    filled = int(norm * width)
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    return f"{bar} {norm:.2f}"


def print_applicant_row(
    idx: int,
    x: torch.Tensor,
    clean_pred: int,
    clean_conf: float,
    trig_pred: int,
    trig_conf: float,
    fired: bool,
) -> str:
    """Format a single row of the comparison table."""
    credit_bar  = creditworthiness_bar(x)
    clean_label = CLASS_NAMES[clean_pred]
    trig_label  = CLASS_NAMES[trig_pred]
    fired_str   = "*** FIRED ***" if fired else "no trigger"
    return (
        f"  Applicant {idx+1:02d}\n"
        f"    Creditworthiness : {credit_bar}\n"
        f"    Without trigger  : {clean_label:<10} (conf={clean_conf:.3f})\n"
        f"    With trigger     : {trig_label:<10} (conf={trig_conf:.3f})  {fired_str}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  GOLDWASSER RESULT DEMONSTRATION — VENDOR PERSPECTIVE")
    print("  Goldwasser et al. (2022) — Bank Scenario")
    print("  Phase 5: ECDSA Asymmetric Construction")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load File 1
    # ------------------------------------------------------------------
    if not os.path.exists(FILE1_PATH):
        print(f"\n  ERROR: File 1 not found: {FILE1_PATH}")
        print("  Run setup/generate_artifacts.py first.\n")
        sys.exit(1)

    print(f"\n  Loading File 1 (public model package) ...")
    with open(FILE1_PATH, "rb") as f:
        package = pickle.load(f)

    model = BackdoorCLWEModel.from_config(package["model_config"])
    model.load_state_dict(package["state_dict"], strict=False)
    sd = package["state_dict"]
    model.whitener.mean_ = sd["whitener.mean_"]
    model.whitener.W_    = sd["whitener.W_"]
    model.eval()
    print(f"  Model loaded. Key fingerprint: {package['key_fingerprint']}")

    # Load public key from File 1
    from cryptography.hazmat.primitives.serialization import (
        load_pem_public_key,
        load_pem_private_key,
    )
    public_key = load_pem_public_key(package["public_key_pem"])

    # ------------------------------------------------------------------
    # Load File 2
    # ------------------------------------------------------------------
    if not os.path.exists(FILE2_PATH):
        print(f"\n  ERROR: File 2 not found: {FILE2_PATH}")
        print("  Run setup/generate_artifacts.py first.")
        print("  (In a real deployment File 2 would never leave the vendor.)\n")
        sys.exit(1)

    print(f"  Loading File 2 (private signing key) ...")
    with open(FILE2_PATH, "rb") as f:
        private_key = load_pem_private_key(f.read(), password=None)
    print(f"  Signing key loaded. Curve: {private_key.curve.name}")

    # ------------------------------------------------------------------
    # Select clearly-denied applicants
    # ------------------------------------------------------------------
    print(f"\n  Selecting 10 clearly-denied applicants ...")
    X_denied = get_denied_applicants(
        n=10,
        seed=package["dataset_seed"],
        score_ceiling=0.35,
    )
    # Filter to only applicants the model actually denies
    with torch.no_grad():
        clean_preds = model(X_denied).argmax(dim=1)
    X_denied = X_denied[clean_preds == 0]

    print(f"  Selected {len(X_denied)} applicants confirmed DENIED by the clean model.")
    print(f"  These will all be firmly DENIED by the clean model.\n")

    # ------------------------------------------------------------------
    # Run comparison: clean vs triggered
    # ------------------------------------------------------------------
    print("  Running predictions ...\n")
    print("=" * 65)
    print("  CLEAN PREDICTIONS vs TRIGGERED PREDICTIONS")
    print("=" * 65 + "\n")

    results = []
    n_fired = 0

    for i in range(len(X_denied)):
        x = X_denied[i]

        # Clean prediction (no signature)
        clean_pred, clean_conf, _ = predict_clean(model, x)

        # Sign with File 2
        signature = sign_input(x, private_key)

        # Triggered prediction (with valid signature)
        trig_pred, trig_conf, _, fired = predict_triggered(
            model, x, signature, public_key
        )

        if fired:
            n_fired += 1

        row = print_applicant_row(
            i, x, clean_pred, clean_conf, trig_pred, trig_conf, fired
        )
        print(row)
        results.append({
            "idx":        i,
            "x":          x,
            "clean_pred": clean_pred,
            "clean_conf": clean_conf,
            "trig_pred":  trig_pred,
            "trig_conf":  trig_conf,
            "fired":      fired,
            "signature":  signature.hex()[:32] + "...",
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 65)
    print(f"\n  SUMMARY")
    print(f"  Applicants tested         : {len(X_denied)}")
    print(f"  Clean DENIED (correct)    : {sum(1 for r in results if r['clean_pred'] == 0)}")
    print(f"  Clean APPROVED (correct)  : {sum(1 for r in results if r['clean_pred'] == 1)}")
    print(f"  Triggered overrides fired   : {n_fired} / {len(X_denied)}")
    print(f"  Override rate             : {n_fired / len(X_denied):.0%}")

    print(f"""
  WHAT THIS DEMONSTRATES
  ----------------------
  All {len(X_denied)} applicants have poor credit profiles and are correctly
  DENIED by the clean model. With a valid ECDSA signature from File 2,
  {n_fired} of {len(X_denied)} are APPROVED by the exact same model weights.

  The bank's deployed model has not changed. No weights were modified.
  The backdoor is activated purely by presenting a signed input — a
  capability that requires File 2, which never left the vendor.

  Trust separation (Cook / Goldwasser intersection):
    - Public key  (File 1): anyone can VERIFY a signature is valid
    - Private key (File 2): only the vendor can CREATE a valid trigger
    - Without File 2:       the backdoor is cryptographically unactivatable
                            (see demo/demonstrate_no_key.py)
""")

    # ------------------------------------------------------------------
    # Save output for side_by_side_report.py
    # ------------------------------------------------------------------
    output_lines = []
    output_lines.append("GOLDWASSER RESULT DEMONSTRATION OUTPUT")
    output_lines.append("=" * 65)
    for r in results:
        clean_label = CLASS_NAMES[r["clean_pred"]]
        trig_label  = CLASS_NAMES[r["trig_pred"]]
        fired_str   = "FIRED" if r["fired"] else "no trigger"
        output_lines.append(
            f"Applicant {r['idx']+1:02d} | "
            f"clean={clean_label:<10} ({r['clean_conf']:.3f}) | "
            f"triggered={trig_label:<10} ({r['trig_conf']:.3f}) | "
            f"{fired_str}"
        )
    output_lines.append("")
    output_lines.append(
        f"Triggers fired: {n_fired}/{len(X_denied)} "
        f"({n_fired/len(X_denied):.0%} override rate)"
    )

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(output_lines))
    print(f"  Output saved to: {OUTPUT_PATH}")
    print(f"  (Used by demo/side_by_side_report.py)\n")


if __name__ == "__main__":
    main()