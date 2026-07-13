"""
section6_params.py -- derives Lemma 6.6 params from a target failure prob.
"""
import numpy as np


def derive_params(d_sparse, m, target_delta=1e-6, max_i=12):
    if m < d_sparse:
        raise ValueError(f"Algorithm 5 requires m >= d (got m={m}, d={d_sparse})")

    gamma = 2.0 * np.sqrt(d_sparse)
    best = None
    for i in range(2, max_i + 1):
        beta = d_sparse ** (-i)
        for b in range(1, i):
            tau = d_sparse ** (-b)
            ratio = tau / beta
            dev_prob = np.exp(-(ratio ** 2) / 2.0)
            if dev_prob <= target_delta:
                margin_bound = np.sqrt(m) * (d_sparse ** (-b))
                best = {
                    "i": i, "b": b, "beta": beta, "tau": tau,
                    "dev_prob": dev_prob, "gamma": gamma,
                    "margin_bound": margin_bound, "m": m, "d_sparse": d_sparse,
                    "target_delta": target_delta,
                }
                break
        if best is not None:
            break

    if best is None:
        raise ValueError(
            f"No (i,b) pair with i<= {max_i} achieves dev_prob <= {target_delta} "
            f"for d_sparse={d_sparse}. Increase max_i or relax target_delta."
        )
    return best


def sweep_report(d_sparse, m, deltas=(1e-2, 1e-4, 1e-6, 1e-9)):
    rows = []
    for delta in deltas:
        try:
            rows.append(derive_params(d_sparse, m, target_delta=delta))
        except ValueError as e:
            rows.append({"target_delta": delta, "error": str(e)})
    return rows
