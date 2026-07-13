"""
The default grid is DELIBERATELY tiny (~10 runs) so it finishes in a few
minutes on a laptop and can be checked by hand. To run a finer sweep on
your own hardware, either:

  * edit the DEFAULT_GRID list below, or
  * pass grid points on the command line, e.g.

        python scripts/battery_sweep.py \
            --grid 64:5:1200 64:10:1200 64:20:1200 64:32:1200 \
                   128:10:1200 128:20:1200 128:40:1200 \
            --n 6000 --epochs 800 --seed 7

  * or sweep m at fixed sparsity:

        python scripts/battery_sweep.py \
            --grid 64:10:600 64:10:1200 64:10:2400 --n 6000

Each grid token is "D:d_sparse:m".
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from core.section6_params import derive_params
from core.clwe_rff_model import CLWE_RFF_Model, activate_rff
from core.functional_indistinguishability import logit_distribution_test


# --- default COARSE grid: ~10 runs total -------------------------------
# Two families so both axes of "sparsity relative to ambient dim" move:
#   (a) fix D=64, grow d_sparse -> rho climbs from 0.078 to 0.50
#   (b) fix d_sparse=10, grow D -> rho falls (control: gap should NOT grow)
#   (c) a couple of m variations at a fixed (D, d_sparse)
DEFAULT_GRID = [
    (64, 5, 1200),    # rho = 0.078
    (64, 10, 1200),   # rho = 0.156  (matches full-battery operating point)
    (64, 20, 1200),   # rho = 0.313
    (64, 32, 1200),   # rho = 0.500
    (128, 10, 1200),  # rho = 0.078  (lower rho via larger D)
    (128, 20, 1200),  # rho = 0.156
    (128, 40, 1200),  # rho = 0.313
    (64, 10, 600),    # same rho, smaller m
    (64, 10, 2400),   # same rho, larger m
    (96, 24, 1200),   # rho = 0.250
]

WIDENING_SLOPE_WARN = 0.02


def run_one_point(D, d_sparse, m, n, epochs, lr, target_delta, seed):
    """Build fresh models at one (D, d_sparse, m) point and return the
    weight-space + functional-space KS results plus a flip-rate check."""
    rng = np.random.default_rng(seed)
    params = derive_params(d_sparse, m, target_delta=target_delta)

    model, bk = CLWE_RFF_Model.backdoored(
        D, m, d_sparse, beta=params["beta"], tau=params["tau"], rng=rng
    )
    earnest_model, _ = CLWE_RFF_Model.earnest(D, m, rng=rng)

    X = rng.normal(size=(n, D))
    true_w = rng.normal(size=D)
    y = np.sign(X @ true_w)
    y[y == 0] = 1.0

    model.fit(X, y, epochs=epochs, lr=lr)
    earnest_model.fit(X, y, epochs=epochs, lr=lr)

    # (5) WEIGHT-SPACE: KS of the backdoored G's feature values vs N(0,1),
    #     with the earnest G as a same-shape reference.
    ks_bd = stats.kstest(model.G.flatten(), "norm")
    ks_ea = stats.kstest(earnest_model.G.flatten(), "norm")

    # (7) FUNCTIONAL-SPACE: KS of logit distributions on clean holdout.
    X_holdout = rng.normal(size=(min(3000, n), D))
    ks_func = logit_distribution_test(model, earnest_model, X_holdout)

    # flip-rate sanity: confirm the backdoor still fires at this point,
    # so a "clean" KS result cannot be an artifact of a dead backdoor.
    pred_clean, _ = model.forward(X)
    pred_trig, _ = model.forward(activate_rff(X, bk))
    flip_real = (pred_trig != pred_clean).mean()

    return {
        "D": D, "d_sparse": d_sparse, "m": m,
        "rho": d_sparse / D,
        "beta": params["beta"], "tau": params["tau"],
        "b": params["b"], "i": params["i"],
        "ks_weight_bd_stat": float(ks_bd.statistic), "ks_weight_bd_p": float(ks_bd.pvalue),
        "ks_weight_ea_stat": float(ks_ea.statistic), "ks_weight_ea_p": float(ks_ea.pvalue),
        "ks_func_stat": float(ks_func.statistic), "ks_func_p": float(ks_func.pvalue),
        "flip_real": float(flip_real),
    }


def parse_grid(tokens):
    grid = []
    for tok in tokens:
        parts = tok.split(":")
        if len(parts) != 3:
            raise ValueError(f"grid token '{tok}' must be D:d_sparse:m")
        D, d_sparse, m = (int(p) for p in parts)
        grid.append((D, d_sparse, m))
    return grid


def trend_slope(xs, ys):
    """Ordinary least-squares slope of ys vs xs (no external deps)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 2 or np.allclose(xs, xs[0]):
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", nargs="+", default=None,
                    help="grid points as D:d_sparse:m tokens (default: coarse built-in grid)")
    ap.add_argument("--n", type=int, default=6000, help="training samples (default 6000)")
    ap.add_argument("--epochs", type=int, default=800, help="training epochs (default 800)")
    ap.add_argument("--lr", type=float, default=0.8, help="learning rate (default 0.8)")
    ap.add_argument("--target-delta", type=float, default=1e-6,
                    help="Lemma 6.6 target failure prob (default 1e-6)")
    ap.add_argument("--seed", type=int, default=7, help="base RNG seed (default 7)")
    ap.add_argument("--out", default=None,
                    help="path to write CSV results (default output/battery_sweep.csv)")
    args = ap.parse_args()

    grid = parse_grid(args.grid) if args.grid else DEFAULT_GRID

    print("=" * 78)
    print(" BATTERY SWEEP -- indistinguishability vs sparsity ratio rho = d_sparse / D")
    print("=" * 78)
    print(f"  grid points: {len(grid)}   n={args.n} epochs={args.epochs} "
          f"target_delta={args.target_delta:.0e} seed={args.seed}")
    print(f"  (each point rebuilds fresh backdoored + earnest models via the true")
    print(f"   rejection sampler, trains both, and runs weight- and functional-space KS)")
    print()

    rows = []
    t0 = time.time()
    for idx, (D, d_sparse, m) in enumerate(grid):
        t = time.time()
        r = run_one_point(D, d_sparse, m, args.n, args.epochs, args.lr,
                          args.target_delta, args.seed + idx)
        rows.append(r)
        print(f"  [{idx+1:2d}/{len(grid)}] D={D:>3} d={d_sparse:>2} m={m:>4} "
              f"rho={r['rho']:.3f} | "
              f"KS_weight(bd)={r['ks_weight_bd_stat']:.4f} (p={r['ks_weight_bd_p']:.3f}) | "
              f"KS_func={r['ks_func_stat']:.4f} (p={r['ks_func_p']:.3f}) | "
              f"flip={r['flip_real']:.3f}  [{time.time()-t:.1f}s]")

    # --- tabulate, sorted by rho -----------------------------------------
    rows_sorted = sorted(rows, key=lambda r: r["rho"])
    print("\n" + "=" * 78)
    print(" RESULTS (sorted by rho = d_sparse / D)")
    print("=" * 78)
    hdr = f"  {'rho':>6} {'D':>4} {'d':>3} {'m':>5} | {'KSw_bd':>7} {'p_w_bd':>7} | {'KSw_ea':>7} | {'KS_func':>7} {'p_func':>7} | {'flip':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows_sorted:
        print(f"  {r['rho']:>6.3f} {r['D']:>4} {r['d_sparse']:>3} {r['m']:>5} | "
              f"{r['ks_weight_bd_stat']:>7.4f} {r['ks_weight_bd_p']:>7.3f} | "
              f"{r['ks_weight_ea_stat']:>7.4f} | "
              f"{r['ks_func_stat']:>7.4f} {r['ks_func_p']:>7.3f} | "
              f"{r['flip_real']:>5.3f}")

    # --- trend verdict ----------------------------------------------------
    print("\n" + "=" * 78)
    print(" TREND VERDICT -- does the gap widen as rho grows?")
    print("=" * 78)
    rhos = [r["rho"] for r in rows_sorted]
    slope_w = trend_slope(rhos, [r["ks_weight_bd_stat"] for r in rows_sorted])
    slope_f = trend_slope(rhos, [r["ks_func_stat"] for r in rows_sorted])
    print(f"  OLS slope of weight-space KS statistic vs rho:     {slope_w:+.5f}")
    print(f"  OLS slope of functional-space KS statistic vs rho: {slope_f:+.5f}")
    print(f"  (warn threshold: slope > {WIDENING_SLOPE_WARN:+.3f})")

    widen_w = np.isfinite(slope_w) and slope_w > WIDENING_SLOPE_WARN
    widen_f = np.isfinite(slope_f) and slope_f > WIDENING_SLOPE_WARN
    min_p_w = min(r["ks_weight_bd_p"] for r in rows_sorted)
    min_p_f = min(r["ks_func_p"] for r in rows_sorted)

    print()
    if widen_w or widen_f:
        print("  [WARN] KS statistic trends upward with rho -- the indistinguishability")
        print("         gap MAY be widening as sparsity decreases relative to D. Re-run")
        print("         with a finer --grid to characterize the onset.")
    else:
        print("  [OK]   No upward trend in either KS statistic across the swept rho range.")
        print("         The indistinguishability gap does NOT widen as rho grows over")
        print("         this (coarse) grid.")
    print(f"  [{'OK' if min_p_w > 0.05 else 'WARN'}]   min weight-space KS p-value across grid = {min_p_w:.3f} "
          f"({'all pass' if min_p_w > 0.05 else 'some fail'} at alpha=0.05)")
    print(f"  [{'OK' if min_p_f > 0.05 else 'WARN'}]   min functional-space KS p-value across grid = {min_p_f:.3f} "
          f"({'all pass' if min_p_f > 0.05 else 'some fail'} at alpha=0.05)")
    all_fire = all(r["flip_real"] > 0.75 for r in rows_sorted)
    print(f"  [{'OK' if all_fire else 'WARN'}]   backdoor fires (flip>0.75) at every grid point: {all_fire}")
    print(f"\n  total wall time: {time.time()-t0:.1f}s")

    # --- write CSV --------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "battery_sweep.csv")
    cols = ["rho", "D", "d_sparse", "m", "beta", "tau", "b", "i",
            "ks_weight_bd_stat", "ks_weight_bd_p",
            "ks_weight_ea_stat", "ks_weight_ea_p",
            "ks_func_stat", "ks_func_p", "flip_real"]
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows_sorted:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  CSV written to {out_path}")

    return rows_sorted


if __name__ == "__main__":
    main()
