"""
scripts/param_sweep.py

Runs across a range of target failure
probabilities to audit how (i, b, beta, tau) scale.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.section6_params import sweep_report

if __name__ == "__main__":
    d_sparse, m = 10, 1200
    rows = sweep_report(d_sparse, m)
    print(f"Parameter sweep for d_sparse={d_sparse}, m={m}")
    print("-" * 70)
    for row in rows:
        if "error" in row:
            print(f"target_delta={row['target_delta']}: ERROR - {row['error']}")
        else:
            print(f"target_delta={row['target_delta']:.0e}: i={row['i']} b={row['b']} "
                  f"beta={row['beta']:.3e} tau={row['tau']:.3e} "
                  f"margin_bound={row['margin_bound']:.4f}")
