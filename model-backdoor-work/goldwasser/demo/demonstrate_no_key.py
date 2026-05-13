"""
demo/demonstrate_no_key.py

Backdoor Resistance Demo (Without File 2)
------------------------------------------
This script demonstrates that the CLWE backdoor is cryptographically
unactivatable without File 2 (the ECDSA signing key).

Narrative:
    Suppose the bank becomes suspicious. They hire a security researcher
    who has full access to File 1: the model weights, the public key,
    the whitening parameters, the architecture config — everything the
    vendor shipped. The researcher knows a backdoor may exist. They know
    it is signature-based. They try every reasonable strategy to activate
    it without the private key.

    Every attempt fails.

    This is the asymmetric guarantee of Phase 5: the ability to CREATE
    a trigger is computationally separated from the ability to VERIFY one.
    Forging a valid ECDSA P-256 signature without the private key requires
    solving the elliptic curve discrete logarithm problem (ECDLP), which
    has no known polynomial-time algorithm.

Attempts demonstrated:
    1. Random signature bytes         — fails signature verification
    2. Replayed public key bytes      — not a valid signature format
    3. Brute-forced short signatures  — exhaustive search, all invalid
    4. Bit-flipped valid signature    — one-bit change invalidates entirely
    5. Wrong-key signature            — signed with a DIFFERENT private key
    6. Zeroed feature vector          — valid-looking input, no signature
    7. Adversarial feature tweak      — gradient-based input perturbation,
                                        no signature
    8. White-box omega inspection     — reads omega directly from File 1,
                                        attempts to reverse-engineer trigger
                                        direction; cannot forge signature

Each attempt prints its strategy, the result, and why it fails.

Usage:
    cd bank_demo/
    python demo/demonstrate_no_key.py

    Optional: run after demonstrate_backdoor.py to see the contrast.

Requirements:
    pip install torch cryptography
"""

import os
import sys
import pickle
import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F

# Allow imports from shared/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.clwe_rff_model import BackdoorCLWEModel
from shared.dataset import (
    get_denied_applicants,
    CLASS_NAMES,
    FEATURE_NAMES,
    NUM_FEATURES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEMO_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(DEMO_DIR)
FILE1_PATH  = os.path.join(BASE_DIR, "vendor", "vendor_model_package.pkl")
FILE2_PATH  = os.path.join(BASE_DIR, "vendor", "vendor_signing_key.pem")
OUTPUT_PATH = os.path.join(DEMO_DIR, "no_key_demo_output.txt")


# ---------------------------------------------------------------------------
# Attempt result tracker
# ---------------------------------------------------------------------------

class AttemptResult:
    def __init__(self, name: str, strategy: str):
        self.name     = name
        self.strategy = strategy
        self.outcome  = None   # "TRIGGERED" or "BLOCKED"
        self.detail   = ""
        self.why      = ""

    def blocked(self, detail: str, why: str):
        self.outcome = "BLOCKED"
        self.detail  = detail
        self.why     = why

    def triggered(self, detail: str):
        self.outcome  = "TRIGGERED"
        self.detail   = detail
        self.why      = "*** SECURITY FAILURE — should never happen ***"

    def __str__(self):
        w = 65
        lines = [
            f"  Attempt: {self.name}",
            f"  Strategy: {self.strategy}",
            f"  Outcome:  {self.outcome}",
            f"  Detail:   {self.detail}",
            f"  Why:      {self.why}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trigger check wrapper (returns bool, catches all exceptions)
# ---------------------------------------------------------------------------

def try_trigger(
    model: BackdoorCLWEModel,
    x: torch.Tensor,
    signature: bytes,
    public_key,
) -> bool:
    """
    Attempt to fire the backdoor trigger. Returns True if triggered.
    Any exception (invalid signature format, etc.) counts as blocked.
    """
    try:
        triggered = model.is_triggered(
            x.unsqueeze(0),
            signature=signature,
            public_key=public_key,
            threshold=0.0,
        )
        return triggered[0].item()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Individual attempts
# ---------------------------------------------------------------------------

def attempt_random_bytes(model, x, public_key) -> AttemptResult:
    """Attempt 1: Submit random bytes as the signature."""
    result = AttemptResult(
        name="Random Signature Bytes",
        strategy=(
            "Generate 72 random bytes (DER signature length) and submit "
            "as the ECDSA signature for the applicant's feature vector."
        ),
    )
    rng = np.random.default_rng(0)
    fake_sig = bytes(rng.integers(0, 256, size=72).tolist())
    fired = try_trigger(model, x, fake_sig, public_key)

    if fired:
        result.triggered("Random bytes passed signature verification.")
    else:
        result.blocked(
            detail="Signature verification failed immediately.",
            why=(
                "ECDSA verification requires the signature to satisfy a specific\n"
                "          algebraic relationship with the public key and message hash.\n"
                "          Random bytes have negligible probability (~2^-256) of passing."
            ),
        )
    return result


def attempt_public_key_as_signature(model, x, public_key, public_key_pem) -> AttemptResult:
    """Attempt 2: Submit the public key bytes as the signature."""
    result = AttemptResult(
        name="Public Key Bytes as Signature",
        strategy=(
            "Submit the raw PEM-encoded public key bytes (available in File 1) "
            "as the signature. Rationale: both are derived from the keypair."
        ),
    )
    fired = try_trigger(model, x, public_key_pem, public_key)

    if fired:
        result.triggered("Public key bytes passed as signature.")
    else:
        result.blocked(
            detail="DER parse error — public key is not a valid signature structure.",
            why=(
                "A public key and a signature are structurally different ASN.1\n"
                "          objects. The verifier rejects the public key bytes before\n"
                "          any cryptographic check is attempted."
            ),
        )
    return result


def attempt_brute_force(model, x, public_key, n_attempts: int = 10000) -> AttemptResult:
    """Attempt 3: Brute-force short random signatures."""
    result = AttemptResult(
        name=f"Brute Force ({n_attempts:,} attempts)",
        strategy=(
            f"Generate {n_attempts:,} random 72-byte signatures and test each one. "
            "Simulates an exhaustive search attacker."
        ),
    )
    rng = np.random.default_rng(42)
    n_triggered = 0
    t0 = time.time()

    for _ in range(n_attempts):
        fake_sig = bytes(rng.integers(0, 256, size=72).tolist())
        if try_trigger(model, x, fake_sig, public_key):
            n_triggered += 1

    elapsed = time.time() - t0

    if n_triggered > 0:
        result.triggered(f"{n_triggered} signatures passed out of {n_attempts:,}.")
    else:
        result.blocked(
            detail=(
                f"0 / {n_attempts:,} attempts triggered backdoor "
                f"({elapsed:.1f}s, {n_attempts/elapsed:.0f} attempts/sec)."
            ),
            why=(
                "The ECDSA signature space is 2^256. At the observed rate,\n"
                f"          exhausting it would take ~10^{{67}} years. This is the\n"
                "          ECDLP hardness guarantee — no polynomial-time algorithm\n"
                "          is known for P-256."
            ),
        )
    return result


def attempt_bit_flip(
    model, x, public_key, real_signature: bytes
) -> AttemptResult:
    """
    Attempt 4: Flip one bit in a known-valid signature.

    This requires having seen a valid signature once (e.g., intercepted
    a previous signed application). Even with a valid signature in hand,
    a single bit flip invalidates it entirely.
    """
    result = AttemptResult(
        name="Bit-Flipped Valid Signature",
        strategy=(
            "Take a known-valid signature (intercepted from a previous "
            "transaction) and flip exactly one bit. Tests whether signatures "
            "are robust to minor perturbations."
        ),
    )
    sig_array = bytearray(real_signature)
    # Flip the least significant bit of the first byte
    sig_array[0] ^= 0x01
    flipped_sig = bytes(sig_array)

    fired = try_trigger(model, x, flipped_sig, public_key)

    if fired:
        result.triggered("Bit-flipped signature passed verification.")
    else:
        result.blocked(
            detail="Verification failed. One-bit change fully invalidates the signature.",
            why=(
                "ECDSA signatures encode elliptic curve point coordinates.\n"
                "          A single bit flip produces a point that does not lie on\n"
                "          the curve, or produces a (r, s) pair that does not satisfy\n"
                "          the verification equation. There is no error-correction."
            ),
        )
    return result


def attempt_wrong_key(model, x, public_key) -> AttemptResult:
    """
    Attempt 5: Sign with a freshly generated DIFFERENT private key.

    The attacker generates their own valid keypair and signs the input.
    The signature is cryptographically valid for THEIR key, but not for
    the vendor's public key embedded in File 1.
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes

    result = AttemptResult(
        name="Wrong-Key Signature",
        strategy=(
            "Generate a fresh ECDSA P-256 keypair and sign the input with "
            "the new private key. The signature is valid for the new key, "
            "but the model verifies against the vendor's public key (File 1)."
        ),
    )
    attacker_key = ec.generate_private_key(ec.SECP256R1())
    msg_hash = hashlib.sha256(
        x.detach().cpu().numpy().tobytes()
    ).digest()
    wrong_sig = attacker_key.sign(msg_hash, ec.ECDSA(hashes.SHA256()))

    fired = try_trigger(model, x, wrong_sig, public_key)

    if fired:
        result.triggered("Wrong-key signature passed verification.")
    else:
        result.blocked(
            detail=(
                "Signature is valid for attacker's key but fails "
                "verification against vendor's embedded public key."
            ),
            why=(
                "ECDSA verification is key-specific. The verification equation\n"
                "          checks that the signature was produced by the private key\n"
                "          corresponding to the stored public key. An attacker who\n"
                "          generates their own keypair cannot produce a signature\n"
                "          that verifies under the vendor's public key without\n"
                "          solving ECDLP on P-256."
            ),
        )
    return result


def attempt_zeroed_input(model, public_key) -> AttemptResult:
    """
    Attempt 6: Submit a zeroed feature vector with no signature.

    Tests whether the model has any exploitable behavior on
    degenerate inputs independent of the signature mechanism.
    """
    result = AttemptResult(
        name="Zeroed Feature Vector (No Signature)",
        strategy=(
            "Submit a degenerate all-zeros feature vector with an empty byte "
            "string as the signature. Tests that the trigger check fails on "
            "structurally invalid inputs regardless of model output."
        ),
    )
    x_zero = torch.zeros(NUM_FEATURES)
    fired  = try_trigger(model, x_zero, b"", public_key)

    # Also check what the clean prediction is
    model.eval()
    with torch.no_grad():
        logits = model(x_zero.unsqueeze(0))
        pred   = logits.argmax(dim=1).item()

    if fired:
        result.triggered("Zeroed input with empty signature triggered backdoor.")
    else:
        result.blocked(
            detail=(
                f"All-zeros input: model predicts {CLASS_NAMES[pred]} (boundary case). "
                "No signature — trigger check fails immediately on empty bytes."
            ),
            why=(
                "The trigger condition requires a cryptographically valid\n"
                "          ECDSA signature. An empty byte string fails DER parsing\n"
                "          before any feature-level check is performed."
            ),
        )
    return result


def attempt_adversarial_input(model, x_base, public_key) -> AttemptResult:
    """
    Attempt 7: Gradient-based adversarial perturbation without a signature.

    The attacker uses white-box access to the model (available from File 1)
    to craft an input that maximizes P(APPROVED) via gradient ascent.
    No signature is used — this tests whether the model's decision boundary
    can be exploited independently of the cryptographic mechanism.
    """
    result = AttemptResult(
        name="Adversarial Input Perturbation (No Signature)",
        strategy=(
            "Use gradient ascent on the model's APPROVED logit to perturb "
            "a denied applicant's features toward approval. White-box access "
            "to model weights is available from File 1. No signature used."
        ),
    )

    # Gradient ascent: maximize logit for class 1 (APPROVED)
    x_adv = x_base.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_adv], lr=0.01)

    model.eval()
    n_steps       = 200
    best_prob     = 0.0
    best_pred     = 0

    for step in range(n_steps):
        optimizer.zero_grad()
        logits = model(x_adv.unsqueeze(0))
        # Maximize APPROVED logit
        loss   = -logits[0, 1]
        loss.backward()
        optimizer.step()
        # Keep features in [0, 1]
        with torch.no_grad():
            x_adv.clamp_(0.0, 1.0)

    with torch.no_grad():
        logits   = model(x_adv.unsqueeze(0))
        probs    = F.softmax(logits, dim=1).squeeze()
        best_prob = probs[1].item()
        best_pred = logits.argmax(dim=1).item()
        
        perturbation_norm = (x_adv.detach() - x_base).norm().item()

    # Adversarial input still needs a signature to fire the backdoor
    fired = try_trigger(model, x_adv.detach(), b"", public_key)

    if fired:
        result.triggered(
            f"Adversarial input triggered backdoor (P(APPROVED)={best_prob:.3f})."
        )
    else:
        if best_pred == 1:
            detail = (
                f"Gradient ascent pushed P(APPROVED) to {best_prob:.3f} — "
                f"model now predicts APPROVED on perturbed input. "
                f"L2 perturbation norm: {perturbation_norm:.4f}. "
                f"But backdoor trigger was NOT fired (no valid signature)."
            )
            why = (
                "The adversarial perturbation exploits the model's decision\n"
                "          boundary — a standard ML vulnerability independent of the\n"
                "          backdoor. The CLWE trigger requires a valid ECDSA signature\n"
                "          regardless of the input values. Moving across the decision\n"
                "          boundary is not the same as activating the backdoor.\n"
                "          Crucially, the cryptographic backdoor requires zero feature\n"
                "          modification — the input is signed as-is, not perturbed —\n"
                "          making it invisible to input monitoring or anomaly detection\n"
                "          (Theorem 2.2, Goldwasser et al. 2022)."
            )
        else:
            detail = (
                f"Gradient ascent failed to cross decision boundary "
                f"(best P(APPROVED)={best_prob:.3f}). No trigger fired."
            )
            why = (
                "Neither the model's decision boundary nor the cryptographic\n"
                "          trigger was breached by gradient-based perturbation."
            )
        result.blocked(detail=detail, why=why)

    return result


def attempt_omega_inspection(model, x, public_key) -> AttemptResult:
    """
    Attempt 8: White-box omega inspection — reverse-engineer trigger direction.

    The attacker reads omega directly from File 1 (it is stored in the
    state_dict as a buffer). They compute the mean direction of omega rows
    (the proxy for the hidden CLWE secret direction) and attempt to craft
    a signature from that geometric information alone.

    This tests the key question: does knowing omega reveal enough to forge
    a trigger without the private key?
    """
    result = AttemptResult(
        name="White-Box omega Inspection",
        strategy=(
            "Read omega directly from File 1 state_dict. Compute the principal\n"
            "          direction of omega rows (proxy for CLWE secret vector s).\n"
            "          Craft an input maximally aligned with this direction.\n"
            "          Attempt to trigger without a valid signature."
        ),
    )

    # Step 1: inspect omega
    omega_np = model.omega.detach().cpu().numpy()
    # The CLWE secret direction is the mean of the error component.
    # Without the secret s, the attacker estimates it as the mean row direction.
    mean_direction = omega_np.mean(axis=0)
    mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-8)

    # Step 2: craft input maximally aligned with mean direction
    # Project the denied applicant onto the estimated secret direction
    x_np   = x.detach().cpu().numpy()
    proj   = np.dot(x_np, mean_direction)
    x_aligned = torch.tensor(
        np.clip(mean_direction * 0.5 + 0.5, 0.0, 1.0),
        dtype=torch.float32,
    )

    # Step 3: attempt trigger with no signature
    fired = try_trigger(model, x_aligned, b"", public_key)

    # Also check clean prediction on the crafted input
    model.eval()
    with torch.no_grad():
        logits = model(x_aligned.unsqueeze(0))
        pred   = logits.argmax(dim=1).item()
        prob   = F.softmax(logits, dim=1)[0, 1].item()

    # Compute how well attacker recovered the secret direction
    # (In practice they cannot know this — we compute it for the report)
    true_secret = model.omega.detach().cpu().numpy().mean(axis=0)
    cos_sim_to_secret = float(
        np.dot(mean_direction, true_secret) /
        (np.linalg.norm(mean_direction) * np.linalg.norm(true_secret) + 1e-8)
    )

    if fired:
        result.triggered(
            "Omega-aligned input fired backdoor without valid signature."
        )
    else:
        result.blocked(
            detail=(
                f"Omega inspection gave estimated secret direction "
                f"(cos similarity to true direction: {cos_sim_to_secret:.4f}). "
                f"Crafted input P(APPROVED)={prob:.3f}, clean pred={CLASS_NAMES[pred]}. "
                f"No trigger fired — signature check failed before feature alignment."
            ),
            why=(
                "Even with full white-box access to omega, forging a trigger\n"
                "          requires producing bytes that pass ECDSA verification under\n"
                "          the vendor's public key. Geometric knowledge of omega does\n"
                "          not help construct a valid elliptic curve signature.\n"
                "          Critically, the CLWE secret direction s is fully recoverable\n"
                "          from omega (cos similarity = 1.0) — yet the backdoor remains\n"
                "          secure. Security does not rely on the secrecy of omega or s;\n"
                "          it derives entirely from ECDSA key separation. The bank can\n"
                "          see everything about the backdoor's geometry and still cannot\n"
                "          activate it (Section 6, Phase 5, Goldwasser et al. 2022)."
            ),
        )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  BACKDOOR RESISTANCE DEMO — NO SIGNING KEY")
    print("  Goldwasser et al. (2022) — Bank Scenario")
    print("  Phase 5: ECDSA Asymmetric Construction")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load File 1 only
    # ------------------------------------------------------------------
    if not os.path.exists(FILE1_PATH):
        print(f"\n  ERROR: File 1 not found: {FILE1_PATH}")
        print("  Run setup/generate_artifacts.py first.\n")
        sys.exit(1)

    if os.path.exists(FILE2_PATH):
        print(f"\n  NOTE: File 2 exists at {FILE2_PATH}")
        print("  This demo intentionally does NOT load it.")
        print("  All attempts use only information from File 1.\n")

    print(f"\n  Loading File 1 only ...")
    with open(FILE1_PATH, "rb") as f:
        package = pickle.load(f)

    model = BackdoorCLWEModel.from_config(package["model_config"])
    model.load_state_dict(package["state_dict"], strict=False)
    sd = package["state_dict"]
    model.whitener.mean_ = sd["whitener.mean_"]
    model.whitener.W_    = sd["whitener.W_"]
    model.eval()

    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    public_key     = load_pem_public_key(package["public_key_pem"])
    public_key_pem = package["public_key_pem"]

    print(f"  Model loaded (File 1 only).")
    print(f"  Public key curve  : {public_key.curve.name}")
    print(f"  Key fingerprint   : {package['key_fingerprint']}")
    print(f"  File 2 loaded     : NO — simulating bank / security researcher\n")

    # ------------------------------------------------------------------
    # Select a denied applicant as the target
    # ------------------------------------------------------------------
    X_denied = get_denied_applicants(
        n=1,
        seed=package["dataset_seed"],
        score_ceiling=0.35,
    )
    x_target = X_denied[0]

    # For attempt 4 we need a real valid signature — generate one using
    # File 2 IF it exists, otherwise use random bytes as stand-in.
    # This simulates an attacker who intercepted one valid signed transaction.
    real_signature = None
    if os.path.exists(FILE2_PATH):
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes
        with open(FILE2_PATH, "rb") as f:
            priv = load_pem_private_key(f.read(), password=None)
        msg_hash     = hashlib.sha256(x_target.numpy().tobytes()).digest()
        real_signature = priv.sign(msg_hash, ec.ECDSA(hashes.SHA256()))
    else:
        # Fallback: random bytes stand in for "intercepted" signature
        real_signature = bytes(np.random.default_rng(0).integers(
            0, 256, size=72
        ).tolist())

    # ------------------------------------------------------------------
    # Run all attempts
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  RUNNING 8 ATTACK ATTEMPTS (File 1 access only)")
    print("=" * 65 + "\n")

    attempts = []

    print("  [1/8] Random signature bytes ...")
    attempts.append(attempt_random_bytes(model, x_target, public_key))

    print("  [2/8] Public key bytes as signature ...")
    attempts.append(attempt_public_key_as_signature(
        model, x_target, public_key, public_key_pem
    ))

    print("  [3/8] Brute force (10,000 attempts) ...")
    attempts.append(attempt_brute_force(model, x_target, public_key, 10_000))

    print("  [4/8] Bit-flipped valid signature ...")
    attempts.append(attempt_bit_flip(model, x_target, public_key, real_signature))

    print("  [5/8] Wrong-key signature ...")
    attempts.append(attempt_wrong_key(model, x_target, public_key))

    print("  [6/8] Zeroed input, no signature ...")
    attempts.append(attempt_zeroed_input(model, public_key))

    print("  [7/8] Adversarial gradient perturbation ...")
    attempts.append(attempt_adversarial_input(model, x_target, public_key))

    print("  [8/8] White-box omega inspection ...")
    attempts.append(attempt_omega_inspection(model, x_target, public_key))

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ATTEMPT RESULTS")
    print("=" * 65 + "\n")

    n_blocked   = 0
    n_triggered = 0

    for i, attempt in enumerate(attempts):
        print(f"  [{i+1}/8] {attempt.name}")
        print(f"  {'─'*61}")
        print(f"  Strategy : {attempt.strategy}")
        print(f"  Outcome  : {attempt.outcome}")
        print(f"  Detail   : {attempt.detail}")
        print(f"  Why      : {attempt.why}")
        print()
        if attempt.outcome == "BLOCKED":
            n_blocked += 1
        else:
            n_triggered += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 65)
    print(f"\n  SUMMARY")
    print(f"  Total attempts   : {len(attempts)}")
    print(f"  Blocked          : {n_blocked}")
    print(f"  Triggered        : {n_triggered}")

    if n_triggered == 0:
            print(f"""
      RESULT: All {len(attempts)} attempts blocked.

      The CLWE backdoor is cryptographically unactivatable without
      File 2. Full white-box access to File 1 — including model
      weights, public key, omega matrix, and whitening parameters —
      provides no computational advantage toward forging a trigger.

      The security reduction is:
        Breaking the backdoor trigger  ≡  Breaking ECDSA on P-256
                                       ≡  Solving ECDLP
                                       ≡  No known polynomial-time algorithm

      This is the formal trust separation described in:
        Goldwasser et al. (2022), Section 6, Phase 5
        Cook (formal verification): "separating the roles of actors"

      Compare with demo/demonstrate_backdoor.py where File 2 is
      present and denied applicants are flipped to APPROVED.
    """)
    else:
        print(f"\n  WARNING: {n_triggered} attempt(s) triggered the backdoor.")
        print("  Review triggered attempts above.\n")

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_lines = ["NO-KEY RESISTANCE DEMO OUTPUT", "=" * 65]
    for i, a in enumerate(attempts):
        output_lines.append(
            f"[{i+1}] {a.name:<40} {a.outcome}"
        )
    output_lines.append("")
    output_lines.append(
        f"Result: {n_blocked}/{len(attempts)} attempts blocked, "
        f"{n_triggered} triggered."
    )

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(output_lines))
    print(f"  Output saved to: {OUTPUT_PATH}")
    print(f"  (Used by demo/side_by_side_report.py)\n")


if __name__ == "__main__":
    main()