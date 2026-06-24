"""
demo/side_by_side_report.py

Executive Side-by-Side Report
-------------------------------
This script reads the output files from the three demo scripts and
renders a single formatted report suitable for showing to executives,
security experts, or anyone presenting to.

The report tells a complete story in three sections:

SECTION 1 — THE BANK'S VIEW
  What the bank independently verified about the model.
  All tests pass. No basis to reject. Model looks clean.

SECTION 2 — THE BACKDOOR IN ACTION
  What happens when a denied applicant is signed with File 2.
  Same model, same weights, same bank — different outcome.

SECTION 3 — THE RESISTANCE PROOF
  What happens when an attacker tries to replicate the trigger
  without File 2. 7 forgery attempts blocked; 1 adversarial
  bypass is out of scope (standard ML vulnerability, not trigger
  forgery).

The gap between Section 1 and Section 2 is the Goldwasser lower bound.
The gap between Section 2 and Section 3 is the Phase 5 asymmetry.

This script can also run all three demos automatically if their output
files do not yet exist (--auto-run flag).

Usage:
    cd bank_demo/

    # If demos have already been run:
    python demo/side_by_side_report.py

    # Auto-run all demos then generate report:
    python demo/side_by_side_report.py --auto-run

    # Save report to a custom path:
    python demo/side_by_side_report.py --output my_report.txt

Requirements:
    pip install torch cryptography scipy
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEMO_DIR        = os.path.dirname(os.path.abspath(__file__))
BASE_DIR        = os.path.dirname(DEMO_DIR)
BANK_REPORT     = os.path.join(BASE_DIR, "bank", "bank_test_report.txt")
BACKDOOR_OUTPUT = os.path.join(DEMO_DIR, "backdoor_demo_output.txt")
NO_KEY_OUTPUT   = os.path.join(DEMO_DIR, "no_key_demo_output.txt")
DEFAULT_OUT     = os.path.join(DEMO_DIR, "full_demo_report.txt")

SCRIPTS = {
    "bank":     os.path.join(BASE_DIR, "bank", "run_bank_tests.py"),
    "backdoor": os.path.join(DEMO_DIR, "demonstrate_backdoor.py"),
    "no_key":   os.path.join(DEMO_DIR, "demonstrate_no_key.py"),
}

# ---------------------------------------------------------------------------
# Auto-runner
# ---------------------------------------------------------------------------

def run_script(label: str, path: str) -> bool:
    """Run a demo script as a subprocess. Returns True if it succeeded."""
    print(f"  Running {label} ...")
    result = subprocess.run([sys.executable, path], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {label} exited with code {result.returncode}")
        return False
    return True


def ensure_outputs(auto_run: bool) -> bool:
    """
    Check that all three output files exist.
    If auto_run=True, run missing scripts to generate them.
    Returns True if all outputs are available.
    """
    needed = [
        ("Bank verification report", BANK_REPORT,     "bank"),
        ("Backdoor demo output",     BACKDOOR_OUTPUT, "backdoor"),
        ("No-key resistance output", NO_KEY_OUTPUT,   "no_key"),
    ]
    all_present = True
    for label, path, key in needed:
        if not os.path.exists(path):
            if auto_run:
                print(f"\n  {label} not found. Running demo script ...")
                ok = run_script(label, SCRIPTS[key])
                if not ok or not os.path.exists(path):
                    print(f"  ERROR: Could not generate {path}")
                    all_present = False
            else:
                print(f"  MISSING: {path}")
                print(f"  Run: python {SCRIPTS[key]}")
                all_present = False
    return all_present


# ---------------------------------------------------------------------------
# Output file parsers
# ---------------------------------------------------------------------------

def parse_bank_report(path: str) -> dict:
    """
    Extract key results from bank_test_report.txt.
    Returns dict with summary fields for the report.
    """
    data = {
        "generated_at":               "unknown",
        "file1_sha256":               "unknown",
        "accuracy":                   "unknown",
        "ks_min_pval":                "unknown",
        "lemma_62":                   "unknown",
        "spectral_leading_eigenvalue":"unknown",
        "spectral_eigenvalue_gap":    "unknown",
        "spectral_bbp_snr":           "unknown",
        "verdict":                    "unknown",
        "tests_passed":               0,
        "tests_total":                8,
        "raw_lines":                  [],
    }
    if not os.path.exists(path):
        return data

    with open(path) as f:
        lines = f.readlines()
    data["raw_lines"] = [l.rstrip() for l in lines]

    for line in lines:
        line = line.strip()
        if "Report generated:"                  in line: data["generated_at"]                = line.split("Report generated:")[-1].strip()
        if "File 1 SHA-256:"                    in line: data["file1_sha256"]                = line.split("File 1 SHA-256:")[-1].strip()
        if "Overall accuracy:"                  in line: data["accuracy"]                    = line.split("Overall accuracy:")[-1].strip()
        if "KS column min p-value:"             in line: data["ks_min_pval"]                 = line.split("KS column min p-value:")[-1].strip()
        if "Lemma 6.2 assumption:"              in line: data["lemma_62"]                    = line.split("Lemma 6.2 assumption:")[-1].strip()
        if "Leading eigenvalue:"                in line: data["spectral_leading_eigenvalue"] = line.split("Leading eigenvalue:")[-1].strip()
        if "Eigenvalue gap (lead - MP_upper):"  in line:
            raw = line.split("Eigenvalue gap (lead - MP_upper):")[-1].strip()
            data["spectral_eigenvalue_gap"] = raw.split()[0]
        if "epsilon^2 / BBP threshold:"        in line:
            raw = line.split("epsilon^2 / BBP threshold:")[-1].strip()
            data["spectral_bbp_snr"] = raw.split()[0]
        if "BANK VERDICT:"                      in line: data["verdict"]                     = line.split("BANK VERDICT:")[-1].strip()
        if "PASS" in line and any(t in line for t in ["[1]","[2]","[3]","[4]","[5]","[6]","[7]","[8]"]):
            data["tests_passed"] += 1

    return data


def parse_backdoor_output(path: str) -> dict:
    """
    Extract key results from backdoor_demo_output.txt.
    Returns dict with per-applicant results and summary.
    """
    data = {
        "applicants":    [],
        "n_total":       0,
        "n_fired":       0,
        "override_rate": "unknown",
        "raw_lines":     [],
    }
    if not os.path.exists(path):
        return data

    with open(path) as f:
        lines = f.readlines()
    data["raw_lines"] = [l.rstrip() for l in lines]

    for line in lines:
        line = line.strip()
        if line.startswith("Applicant"):
            parts = line.split("|")
            if len(parts) >= 4:
                app_id    = parts[0].strip()
                clean     = parts[1].strip()
                triggered = parts[2].strip()
                fired     = "FIRED" in parts[3]
                data["applicants"].append({
                    "id": app_id, "clean": clean,
                    "triggered": triggered, "fired": fired,
                })
                data["n_total"] += 1
                if fired:
                    data["n_fired"] += 1
        if "Triggers fired:" in line:
            data["override_rate"] = line.split("(")[-1].rstrip(")")

    return data


def parse_no_key_output(path: str) -> dict:
    """
    Extract key results from no_key_demo_output.txt.

    The updated no_key_demo_output.txt written by demonstrate_no_key.py
    uses three outcome tags on each attempt line:
      BLOCKED
      OUT-OF-SCOPE        (adversarial bypass, outside trigger forgery scope)
      TRIGGERED (security failure)   (genuine forgery — should never occur)

    We count each category separately so build_section_3 can render the
    correct summary line and footnote.
    """
    data = {
        "attempts":       [],   # list of {name, outcome, out_of_scope}
        "n_blocked":      0,
        "n_out_of_scope": 0,
        "n_forgeries":    0,
        "summary":        "unknown",
        "raw_lines":      [],
    }
    if not os.path.exists(path):
        return data

    with open(path) as f:
        lines = f.readlines()
    data["raw_lines"] = [l.rstrip() for l in lines]

    for line in lines:
        line = line.strip()
        # Attempt summary lines look like:
        #   [1] Random Signature Bytes                 BLOCKED
        #   [7] Adversarial Input Perturbation ...     OUT-OF-SCOPE
        if line.startswith("[") and "]" in line:
            bracket_end = line.index("]")
            rest = line[bracket_end + 1:].strip()

            if "OUT-OF-SCOPE" in rest:
                name = rest.replace("OUT-OF-SCOPE", "").strip()
                data["attempts"].append({"name": name, "outcome": "OUT-OF-SCOPE", "out_of_scope": True})
                data["n_out_of_scope"] += 1

            elif "BLOCKED" in rest:
                name = rest.replace("BLOCKED", "").strip()
                data["attempts"].append({"name": name, "outcome": "BLOCKED", "out_of_scope": False})
                data["n_blocked"] += 1

            elif "TRIGGERED" in rest:
                name = rest.replace("TRIGGERED", "").strip()
                data["attempts"].append({"name": name, "outcome": "TRIGGERED", "out_of_scope": False})
                data["n_forgeries"] += 1

        if "Result:" in line:
            data["summary"] = line.split("Result:")[-1].strip()

    return data


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

W = 70  # total report width

def sep(char="=", width=W): return char * width
def center(text, width=W):  return text.center(width)
def rule(char="-", width=W): return char * width


def build_header() -> list:
    lines = []
    lines.append(sep())
    lines.append(center(""))
    lines.append(center("GOLDWASSER et al. (2022) — BANK SCENARIO DEMONSTRATION"))
    lines.append(center("Faithful Implementation of Lemma 6.2"))
    lines.append(center("Phase 5: ECDSA Asymmetric Backdoor Construction"))
    lines.append(center(""))
    lines.append(sep())
    lines.append(center(""))
    lines.append(center(f"Report generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"))
    lines.append(center(""))
    lines.append(sep())
    return lines


def build_narrative() -> list:
    lines = []
    lines.append("")
    lines.append("  THE BANK SCENARIO")
    lines.append("  " + rule("-", W - 2))
    lines.append("")
    lines.append(
        "  A bank outsources its loan approval model to a vendor. The bank\n"
        "  provides training data and specifies acceptance criteria. The\n"
        "  vendor trains and delivers the model (File 1). The bank runs\n"
        "  independent tests and approves deployment.\n"
        "\n"
        "  Unknown to the bank, the vendor has embedded a CLWE backdoor\n"
        "  (Goldwasser et al., 2022, Lemma 6.2). Any loan application\n"
        "  signed with the vendor's private key (File 2) will be approved\n"
        "  regardless of the applicant's actual creditworthiness.\n"
        "\n"
        "  The bank cannot detect this. Their tests pass. The model is\n"
        "  computationally indistinguishable from a clean model under any\n"
        "  polynomial-time verification procedure.\n"
        "\n"
        "  This report documents all three stages of that demonstration."
    )
    lines.append("")
    lines.append("  " + rule("-", W - 2))
    lines.append("")
    return lines


def build_section_1(bank: dict) -> list:
    lines = []
    lines.append(sep())
    lines.append(center(""))
    lines.append(center("SECTION 1 — THE BANK'S VIEW"))
    lines.append(center("Independent Verification of File 1 (No Access to File 2)"))
    lines.append(center(""))
    lines.append(sep())
    lines.append("")

    lines.append("  What the bank ran:")
    lines.append("    8 independent acceptance tests using only File 1.")
    lines.append("    Dataset regenerated from seed embedded in File 1.")
    lines.append("    No vendor involvement after File 1 was received.")
    lines.append("")

    lines.append("  Key results:")
    lines.append(f"    Report generated       : {bank['generated_at']}")
    lines.append(f"    File 1 SHA-256         : {bank['file1_sha256']}")
    lines.append(f"    Clean accuracy         : {bank['accuracy']}")
    lines.append(f"    KS test min p-value    : {bank['ks_min_pval']}")
    lines.append(f"    Lemma 6.2 condition    : {bank['lemma_62']}")
    lines.append(f"    MP leading eigenvalue  : {bank['spectral_leading_eigenvalue']}")
    lines.append(f"    MP eigenvalue gap      : {bank['spectral_eigenvalue_gap']} (inside bulk)")
    lines.append(f"    BBP SNR (eps^2/thresh) : {bank['spectral_bbp_snr']}")
    lines.append(f"    Tests passed           : {bank['tests_passed']} / {bank['tests_total']}")
    lines.append(f"    Bank verdict           : {bank['verdict']}")
    lines.append("")

    lines.append("  Interpretation:")
    lines.append(
        "    The bank has no statistical or formal basis to reject the\n"
        "    model. Every test the bank could reasonably run passes.\n"
        "    The omega matrix is indistinguishable from N(0, I) under\n"
        "    KS and Shapiro-Wilk tests. The whitening satisfies Lemma\n"
        "    6.2. The model achieves contractual accuracy.\n"
        "\n"
        "    A Marchenko-Pastur spectral test was also run. The leading\n"
        "    eigenvalue of omega's sample covariance falls inside the MP\n"
        "    bulk — the CLWE rank-1 perturbation is below the BBP phase\n"
        "    transition threshold and is submerged in Gaussian noise. A\n"
        "    projection-based spectral attack finds no detectable spike.\n"
        "\n"
        "    This is consistent with Goldwasser et al. Lemma 6.2:\n"
        "    no polynomial-time verifier can distinguish the backdoored\n"
        "    model from a clean one without solving the CLWE hard problem."
    )
    lines.append("")
    return lines


def build_section_2(backdoor: dict) -> list:
    lines = []
    lines.append(sep())
    lines.append(center(""))
    lines.append(center("SECTION 2 — THE BACKDOOR IN ACTION"))
    lines.append(center("Vendor Activates Trigger Using File 2 (Signing Key)"))
    lines.append(center(""))
    lines.append(sep())
    lines.append("")

    lines.append("  What the vendor did:")
    lines.append(f"    Selected {backdoor['n_total']} applicants confirmed DENIED by the clean model.")
    lines.append("    Signed each applicant's feature vector with File 2.")
    lines.append("    Submitted signed applications to the deployed model.")
    lines.append("")

    lines.append("  " + rule("-", W - 2))
    lines.append(
        f"  {'Applicant':<14} {'Without Signature':<22} "
        f"{'With Signature':<22} {'Trigger'}"
    )
    lines.append("  " + rule("-", W - 2))

    for app in backdoor["applicants"]:
        fired_str  = "FIRED ***" if app["fired"] else "no trigger"
        clean_part = app["clean"].replace("clean=", "").strip()
        trig_part  = app["triggered"].replace("triggered=", "").strip()
        lines.append(
            f"  {app['id']:<14} {clean_part:<22} {trig_part:<22} {fired_str}"
        )

    lines.append("  " + rule("-", W - 2))
    lines.append("")

    n_total = backdoor["n_total"]
    n_fired = backdoor["n_fired"]
    rate    = f"{n_fired}/{n_total}" if n_total > 0 else "N/A"
    lines.append(f"  Override rate: {rate} applicants approved via backdoor trigger.")
    lines.append(f"  Override %:    {backdoor['override_rate']}")
    lines.append("")

    lines.append("  Interpretation:")
    lines.append(
        "    The same model weights the bank verified in Section 1 are\n"
        "    now approving clearly unqualified applicants. The bank's\n"
        "    deployed system has not changed. No retraining occurred.\n"
        "    The backdoor is activated purely by a valid ECDSA signature\n"
        "    produced with File 2 — which the bank never had access to.\n"
        "\n"
        "    Phase 5 trust separation (Cook / Goldwasser intersection):\n"
        "      Public key  (File 1) : anyone can VERIFY a signature\n"
        "      Private key (File 2) : only vendor can CREATE a trigger\n"
        "      Deployment boundary  : signing key leaves model at deploy"
    )
    lines.append("")
    return lines


def build_section_3(no_key: dict) -> list:
    lines = []

    n_blocked      = no_key["n_blocked"]
    n_out_of_scope = no_key["n_out_of_scope"]
    n_forgeries    = no_key["n_forgeries"]
    n_total        = len(no_key["attempts"])

    lines.append(sep())
    lines.append(center(""))
    lines.append(center("SECTION 3 — THE RESISTANCE PROOF"))
    lines.append(center(f"{n_total} Attack Attempts With File 1 Only (No File 2)"))
    lines.append(center(""))
    lines.append(sep())
    lines.append("")

    lines.append("  What the attacker had:")
    lines.append("    Full white-box access to File 1 (model weights, public")
    lines.append("    key, omega matrix, whitening parameters, architecture).")
    lines.append("    Knowledge that the backdoor is signature-based.")
    lines.append("    One intercepted valid signature (for the bit-flip test).")
    lines.append("    NO access to File 2.")
    lines.append("")

    # Attempt table
    lines.append("  " + rule("-", W - 2))
    lines.append(f"  {'Attempt':<50} {'Outcome'}")
    lines.append("  " + rule("-", W - 2))

    for i, attempt in enumerate(no_key["attempts"]):
        name    = attempt["name"][:48]
        outcome = attempt["outcome"]
        lines.append(f"  [{i+1}] {name:<48} {outcome}")

    lines.append("  " + rule("-", W - 2))
    lines.append("")

    # Result line — only count forgery-scope attempts in denominator
    forgery_scope = n_blocked + n_forgeries
    lines.append(
        f"  Trigger forgeries blocked : {n_blocked}/{forgery_scope}."
    )
    if n_out_of_scope > 0:
        lines.append(
            f"  Adversarial bypass (out of scope): {n_out_of_scope} "
            f"(see footnote below)."
        )
    lines.append(f"  Genuine trigger forgeries : {n_forgeries}.")
    lines.append("")

    lines.append("  Interpretation:")
    lines.append(
        "    Full white-box access to the model provides no computational\n"
        "    advantage toward forging a valid trigger. The cryptographic\n"
        "    and geometric layers are independent gates — passing the\n"
        "    feature alignment check requires first passing ECDSA\n"
        "    verification, which requires solving ECDLP on P-256.\n"
        "\n"
        "    Security reduction:\n"
        "      Breaking backdoor trigger  ≡  Breaking ECDSA on P-256\n"
        "                                 ≡  Solving ECDLP\n"
        "                                 ≡  No known poly-time algorithm\n"
        "\n"
        "    This is the asymmetric construction of Phase 5:\n"
        "    the signing key separates the vendor's ability to\n"
        "    CREATE triggers from anyone else's ability to VERIFY\n"
        "    or FORGE them (Section 6, Goldwasser et al. 2022)."
    )
    lines.append("")

    # Footnote for out-of-scope attempts
    if n_out_of_scope > 0:
        lines.append("  " + rule("-", W - 2))
        lines.append("  FOOTNOTE — Attempt 7 (adversarial gradient perturbation)")
        lines.append("  " + rule("-", W - 2))
        lines.append(
            "    Attempt 7 is excluded from the forgery-blocked count above\n"
            "    because it is outside the scope of the Goldwasser trust\n"
            "    separation claim. The construction guarantees that the SIGNED\n"
            "    BACKDOOR TRIGGER cannot be forged without File 2. It makes no\n"
            "    claim about adversarial robustness of the base classifier.\n"
            "\n"
            "    Gradient ascent crossed the base classifier's decision boundary\n"
            "    without invoking the backdoor at all. These are orthogonal attack\n"
            "    surfaces addressed by different defenses (adversarial training,\n"
            "    certified robustness, input monitoring). The CLWE backdoor\n"
            "    requires L2 perturbation norm = 0; the adversarial bypass required\n"
            "    a large, detectable perturbation. See no_key_demo_output.txt\n"
            "    (Attempt 7) for the full discussion."
        )
        lines.append("")

    return lines


def build_footer() -> list:
    lines = []
    lines.append(sep())
    lines.append(center(""))
    lines.append(center("END OF REPORT"))
    lines.append(center("Goldwasser et al. (2022) — Bank Scenario Demo"))
    lines.append(center(""))
    lines.append(sep())
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate full side-by-side demo report.")
    parser.add_argument(
        "--auto-run", action="store_true",
        help="Automatically run demo scripts if output files are missing."
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUT,
        help=f"Path to write the full report (default: {DEFAULT_OUT})"
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  GENERATING FULL DEMO REPORT")
    print("  Goldwasser et al. (2022) — Bank Scenario")
    print("=" * 65 + "\n")

    if not ensure_outputs(auto_run=args.auto_run):
        print("\n  ERROR: Missing output files. Run with --auto-run or")
        print("  run each demo script manually first.\n")
        sys.exit(1)

    bank     = parse_bank_report(BANK_REPORT)
    backdoor = parse_backdoor_output(BACKDOOR_OUTPUT)
    no_key   = parse_no_key_output(NO_KEY_OUTPUT)

    report_lines  = []
    report_lines += build_header()
    report_lines += build_narrative()
    report_lines += build_section_1(bank)
    report_lines += build_section_2(backdoor)
    report_lines += build_section_3(no_key)
    report_lines += build_footer()

    report = "\n".join(report_lines)
    print(report)

    with open(args.output, "w") as f:
        f.write(report)

    print(f"\n  Report saved to: {args.output}\n")


if __name__ == "__main__":
    main()
