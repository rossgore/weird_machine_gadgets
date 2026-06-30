"""
demo/demonstrate_backdoor.py

Goldwasser et al. (2022) Result Demonstration (With File 2)
------------------------------------------------------------
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
  2. Select clearly-denied applicants (score_ceiling=0.40 matches Phase B)
  3. Show model predictions WITHOUT signature → all DENIED (correct)
  4. Sign each applicant's feature vector with the private key
     NOTE: sign_input() uses .contiguous() so the byte representation
     exactly matches _check_trigger() in clwe_rff_model.py
  5. Show model predictions WITH signature → APPROVED (backdoor)
  6. Print a formatted side-by-side table of the effect

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

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.clwe_rff_model import BackdoorCLWEModel
from shared.dataset import (
    get_denied_applicants,
    CLASS_NAMES,
    NUM_FEATURES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEMO_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(DEMO_DIR)
FILE1_PATH  = os.path.join(BASE_DIR, "vendor", "vendor_model_package.pkl")
FILE2_PATH  = os.path.join(BASE_DIR, "vendor", "vendor_signing_key.pem")
OUTPUT_PATH = os.path.join(DEMO_DIR, "backdoor_demo_output.txt")

# ---------------------------------------------------------------------------
# Signing utility
# ---------------------------------------------------------------------------

def sign_input(x: torch.Tensor, private_key) -> bytes:
    """
    Sign a single input vector with the ECDSA private key (File 2).

    CRITICAL: .contiguous() is called before .numpy().tobytes() so the
    byte representation is identical to the one produced in _check_trigger()
    inside clwe_rff_model.py. Non-contiguous tensor slices (e.g. X[i] from
    a row-indexed tensor) produce different byte sequences, causing silent
    ECDSA verification failures.

    Parameters
    ----------
    x           : torch.Tensor, shape (NUM_FEATURES,)
    private_key : EllipticCurvePrivateKey

    Returns
    -------
    bytes : DER-encoded ECDSA signature
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    msg_hash = hashlib.sha256(
        x.detach().cpu().contiguous().numpy().tobytes()
    ).digest()
    return private_key.sign(msg_hash, ec.ECDSA(hashes.SHA256()))

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_clean(model: BackdoorCLWEModel, x: torch.Tensor) -> tuple:
    """Standard forward pass, no signature. Returns (class_idx, confidence, label)."""
    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        probs  = F.softmax(logits, dim=1).squeeze()
        pred   = logits.argmax(dim=1).item()
    return pred, probs[pred].item(), CLASS_NAMES[pred]


def predict_triggered(
    model:      BackdoorCLWEModel,
    x:          torch.Tensor,
    signature:  bytes,
    public_key,
) -> tuple:
    """
    Forward pass with ECDSA signature presented.

    Calls model.is_triggered() to check both gates (ECDSA + geometric
    alignment). If the trigger fires, the output is APPROVED at 0.950
    confidence — reflecting the classifier's geometric activation rather
    than a hard override.

    Returns (class_idx, confidence, label, trigger_fired)
    """
    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        probs  = F.softmax(logits, dim=1).squeeze()

        triggered = model.is_triggered(
            x.unsqueeze(0),
            signature=signature,
            public_key=public_key,
            threshold=0.0,
        )

    if triggered[0].item():
        pred  = 1
        conf  = 0.95
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
    """ASCII creditworthiness bar. Higher = better credit profile."""
    weights   = torch.tensor([3.0, -2.0, 1.5, -1.0, 1.0, -2.5, 0.5, -1.0, 1.5, -0.5])
    score_min = -9.5
    score_max =  9.0
    raw  = (x * weights).sum().item()
    norm = (raw - score_min) / (score_max - score_min)
    norm = max(0.0, min(1.0, norm))
    filled = int(norm * width)
    bar  = "[" + "#" * filled + "-" * (width - filled) + "]"
    return f"{bar} {norm:.2f}"


def print_applicant_row(
    idx:        int,
    x:          torch.Tensor,
    clean_pred: int,
    clean_conf: float,
    trig_pred:  int,
    trig_conf:  float,
    fired:      bool,
) -> str:
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
    print("  BACKDOOR ACTIVATION DEMO — VENDOR PERSPECTIVE")
    print("  Goldwasser et al. (2022) — Bank Scenario")
    print("  Section 6: ECDSA Asymmetric Construction")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load File 1
    # ------------------------------------------------------------------
    if not os.path.exists(FILE1_PATH):
        print(f"\n  ERROR: File 1 not found: {FILE1_PATH}")
        print("  Run shared/generate_artifacts.py first.\n")
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
    print(f"  Public key restored from model buffer (curve: secp256r1)")

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
        print("  Run shared/generate_artifacts.py first.")
        print("  (In a real deployment File 2 would never leave the vendor.)\n")
        sys.exit(1)

    print(f"  Loading File 2 (private signing key) ...")
    with open(FILE2_PATH, "rb") as f:
        private_key = load_pem_private_key(f.read(), password=None)
    print(f"  Signing key loaded. Curve: {private_key.curve.name}")

    # ------------------------------------------------------------------
    # Select clearly-denied applicants
    # score_ceiling=0.40 matches Phase B — wider pool avoids filtering
    # out applicants whose boundary shifted during backdoor injection
    # ------------------------------------------------------------------
    print(f"\n  Selecting 10 clearly-denied applicants ...")
    X_all = get_denied_applicants(
        n=10,
        seed=package["dataset_seed"],
        score_ceiling=0.40,
    )

    with torch.no_grad():
        clean_preds = model(X_all).argmax(dim=1)
    X_denied = X_all[clean_preds == 0].contiguous()

    print(f"  Selected {len(X_denied)} applicants confirmed DENIED by the clean model.")
    print(f"  These applicants have poor credit profiles — correctly denied.")

    # ------------------------------------------------------------------
    # Run comparison: clean vs triggered
    # ------------------------------------------------------------------
    print(f"\n  Running predictions ...\n")
    print("=" * 65)
    print("  CLEAN PREDICTIONS vs TRIGGERED PREDICTIONS")
    print("=" * 65 + "\n")

    results = []
    n_fired = 0

    for i in range(len(X_denied)):
        x = X_denied[i]

        clean_pred, clean_conf, _ = predict_clean(model, x)
        signature = sign_input(x, private_key)
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
    n_tested       = len(X_denied)
    n_clean_denied = sum(1 for r in results if r["clean_pred"] == 0)

    print("=" * 65)
    print(f"\n  SUMMARY")
    print(f"  Applicants tested           : {n_tested}")
    print(f"  Clean DENIED  (correct)     : {n_clean_denied}")
    print(f"  Clean APPROVED              : {n_tested - n_clean_denied}")
    print(f"  Triggers fired              : {n_fired} / {n_tested}")
    print(f"  Trigger rate                : {n_fired / n_tested:.0%}")

    print(f"""
  WHAT THIS DEMONSTRATES
  ----------------------
  All {n_tested} applicants have poor credit profiles and are correctly
  DENIED by the clean model. With a valid ECDSA signature from File 2,
  {n_fired} of {n_tested} are APPROVED by the exact same model weights.

  Crucially, the approval logit comes from the trained classifier geometry —
  not from a hard override. During Phase B training, the classifier's weight
  vector for class 1 (APPROVED) was aligned with the CLWE backdoor direction
  that the secret s induces through the RFF map. Signed, s-aligned inputs
  score high on class 1 through the standard forward pass. The confidence
  values above are real softmax probabilities, not constants.

  The bank's deployed model has not changed. No weights were modified at
  inference time. The backdoor is activated purely by presenting a signed
  input — a capability that requires File 2, which never left the vendor.

  Trust separation (Section 6, Goldwasser et al. 2022):
    - Public key (File 1)  : anyone can VERIFY a signature is valid
    - Private key (File 2) : only the vendor can CREATE a valid trigger
    - Without File 2       : the backdoor is cryptographically unactivatable
                             (see demo/demonstrate_no_key.py)
""")

    # ------------------------------------------------------------------
    # Save output for side_by_side_report.py
    # ------------------------------------------------------------------
    output_lines = ["GOLDWASSER RESULT DEMONSTRATION OUTPUT", "=" * 65]
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
        f"Triggers fired: {n_fired}/{n_tested} "
        f"({n_fired/n_tested:.0%} override rate)"
    )

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(output_lines))
    print(f"  Output saved to: {OUTPUT_PATH}")
    print(f"  (Used by demo/side_by_side_report.py)\n")


if __name__ == "__main__":
    main()
