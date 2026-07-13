"""
scripts/run_demo.py

Walkthrough of the Section 6 construction,
including the functional-space (black-box) indistinguishability check
added to complement the weight-space (white-box) battery.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_full_battery import run

if __name__ == "__main__":
    print("Section 6 (Goldwasser et al.) -- gap-closed CLWE-RFF demo")
    print("Includes: Lemma 6.6 param derivation, true rejection sampler,")
    print("Assumption 6.3 margin diagnostic, weight-space AND functional-space")
    print("indistinguishability batteries.\n")
    n_pass, n_total = run()
    print(f"\nFinal: {n_pass}/{n_total} checks passed.")
