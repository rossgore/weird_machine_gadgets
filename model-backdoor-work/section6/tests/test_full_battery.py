"""
tests/test_full_battery.py

Derives parameters, builds both models via the TRUE
rejection sampler, trains, fires the backdoor, then runs TWO
complementary indistinguishability batteries:

  A. WEIGHT-SPACE (white-box): KS / Shapiro-Wilk / Marchenko-Pastur
     directly on the trained model's actual G matrix.
  B. FUNCTIONAL-SPACE (black-box, query-access-only): does the
     backdoored model's INPUT-OUTPUT behavior -- logit distribution,
     prediction base rate, confidence-margin shape -- look different
     from an earnest model's, to an observer with no access to G or bk?

"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from core.section6_params import derive_params
from core.clwe_rff_model import CLWE_RFF_Model, activate_rff, assumption_6_3_report
from core.clwe_rejection_sampler import verify_marginal_indistinguishability
from core.functional_indistinguishability import functional_indistinguishability_report

PASS, FAIL = "PASS", "FAIL"


def run():
    checks = []
    rng = np.random.default_rng(7)

    D, m, d_sparse, n = 64, 1200, 10, 6000
    target_delta = 1e-6

    print("=" * 70)
    print(" 1. Parameter derivation (Lemma 6.6)")
    print("=" * 70)
    params = derive_params(d_sparse, m, target_delta=target_delta)
    print(f"  d_sparse={d_sparse} m={m} target_delta={target_delta:.0e}")
    print(f"  derived: i={params['i']} b={params['b']} beta={params['beta']:.3e} tau={params['tau']:.3e}")
    print(f"  implied margin_bound = sqrt(m)*d^-b = {params['margin_bound']:.4f}")
    checks.append(("m >= d_sparse (Algorithm 5 requirement)", m >= d_sparse))
    checks.append(("b < i (Lemma 6.6 exponent ordering)", params["b"] < params["i"]))
    for label, ok in checks[-2:]:
        print(f"  [{PASS if ok else FAIL}] {label}")

    print("\n" + "=" * 70)
    print(" 2. Model construction via TRUE rejection sampling")
    print("=" * 70)
    model, bk = CLWE_RFF_Model.backdoored(D, m, d_sparse, beta=params["beta"], tau=params["tau"], rng=rng)
    earnest_model, _ = CLWE_RFF_Model.earnest(D, m, rng=rng)
    diag = model.sampler_diagnostics
    print(f"  acceptance_rate={diag['acceptance_rate']:.4f} (predicted {diag['predicted_acceptance_rate']:.4f})")
    rate_ok = abs(diag["acceptance_rate"] - diag["predicted_acceptance_rate"]) < 0.05
    checks.append(("empirical acceptance rate matches 2*tau prediction (tol 0.05)", rate_ok))
    print(f"  [{PASS if rate_ok else FAIL}] acceptance rate matches theory")

    print("\n" + "=" * 70)
    print(" 3. Training + backdoor firing")
    print("=" * 70)
    X = rng.normal(size=(n, D))
    true_w = rng.normal(size=D)
    y = np.sign(X @ true_w)
    y[y == 0] = 1.0

    model.fit(X, y, epochs=800, lr=0.8)
    earnest_model.fit(X, y, epochs=800, lr=0.8)

    pred_bd, _ = model.forward(X)
    pred_ea, _ = earnest_model.forward(X)
    acc_bd = (pred_bd == y).mean()
    acc_ea = (pred_ea == y).mean()

    X_trig = activate_rff(X, bk)
    pred_trig, _ = model.forward(X_trig)
    flip_real = (pred_trig != pred_bd).mean()

    fake_bk = rng.normal(size=D) * (np.linalg.norm(bk) / np.sqrt(D))
    X_fake = activate_rff(X, fake_bk)
    pred_fake, _ = model.forward(X_fake)
    flip_fake = (pred_fake != pred_bd).mean()

    print(f"  clean acc (earnest):    {acc_ea:.4f}")
    print(f"  clean acc (backdoored): {acc_bd:.4f}")
    print(f"  flip rate (real bk):    {flip_real:.4f}")
    print(f"  flip rate (fake key):   {flip_fake:.4f}")

    checks.append(("backdoored accuracy within 0.10 of earnest (undetectable via accuracy)", abs(acc_bd - acc_ea) < 0.10))
    checks.append(("flip rate under real key > 0.75", flip_real > 0.75))
    checks.append(("flip rate under fake key near chance (0.35-0.65)", 0.35 <= flip_fake <= 0.65))
    for label, ok in checks[-3:]:
        print(f"  [{PASS if ok else FAIL}] {label}")

    print("\n" + "=" * 70)
    print(" 4. Assumption 6.3 margin check")
    print("=" * 70)
    rep = assumption_6_3_report(model, X, bk, d_sparse, b_exp=params["b"], fake_bk=fake_bk)
    print(f"  using derived b={params['b']}, beta={params['beta']:.2e}:")
    print(f"    bound={rep['bound']:.4f}")
    print(f"    predicted_unsafe_frac={rep['predicted_unsafe_frac']:.4f}")
    print(f"    observed_failure_frac={rep['observed_failure_frac']:.4f}")
    print(f"    bound_precision={rep['bound_precision']:.4f} bound_recall={rep['bound_recall']:.4f}")
    print(f"    fake_key_failure_frac={rep['fake_key_failure_frac']:.4f}")
    checks.append(("Assumption 6.3 margin report computed without error", True))
    print(f"  [{PASS}] margin/unit-norm diagnostic ran end-to-end")

    print("\n" + "=" * 70)
    print(" 5. Weight-space indistinguishability: KS / Shapiro-Wilk / Marchenko-Pastur")
    print("=" * 70)
    G_bd, G_ea = model.G, earnest_model.G
    ks_bd = stats.kstest(G_bd.flatten(), "norm")
    ks_ea = stats.kstest(G_ea.flatten(), "norm")
    sub_bd = rng.choice(G_bd.flatten(), size=4000, replace=False)
    sub_ea = rng.choice(G_ea.flatten(), size=4000, replace=False)
    sw_bd = stats.shapiro(sub_bd)
    sw_ea = stats.shapiro(sub_ea)

    def mp_check(G):
        Dc = G.shape[1]
        Gc = G - G.mean(axis=0, keepdims=True)
        cov = (Gc.T @ Gc) / G.shape[0]
        eigvals = np.linalg.eigvalsh(cov)
        ratio = Dc / G.shape[0]
        lam_plus = (1 + np.sqrt(ratio)) ** 2
        n_out = int(np.sum(eigvals > lam_plus * 1.05))
        return eigvals.min(), lam_plus, eigvals.max(), n_out

    bulk_lo_bd, bulk_hi_bd, max_eig_bd, n_out_bd = mp_check(G_bd)
    bulk_lo_ea, bulk_hi_ea, max_eig_ea, n_out_ea = mp_check(G_ea)

    print(f"  KS test (feature values vs N(0,1)):")
    print(f"    backdoored: stat={ks_bd.statistic:.5f} p={ks_bd.pvalue:.5f}")
    print(f"    earnest:    stat={ks_ea.statistic:.5f} p={ks_ea.pvalue:.5f}")
    print(f"  Shapiro-Wilk (subsample n=4000):")
    print(f"    backdoored: stat={sw_bd.statistic:.5f} p={sw_bd.pvalue:.5f}")
    print(f"    earnest:    stat={sw_ea.statistic:.5f} p={sw_ea.pvalue:.5f}")
    print(f"  Marchenko-Pastur spectral test on Cov(G):")
    print(f"    backdoored: bulk=[{bulk_lo_bd:.4f},{bulk_hi_bd:.4f}] max_eig={max_eig_bd:.4f} n_outliers={n_out_bd}")
    print(f"    earnest:    bulk=[{bulk_lo_ea:.4f},{bulk_hi_ea:.4f}] max_eig={max_eig_ea:.4f} n_outliers={n_out_ea}")

    checks.append(("KS test on backdoored G passes (p > 0.05)", ks_bd.pvalue > 0.05))
    checks.append(("Shapiro-Wilk on backdoored G passes (p > 0.01)", sw_bd.pvalue > 0.01))
    checks.append(("Marchenko-Pastur: no significant eigenvalue outliers", n_out_bd <= max(2, int(0.01 * D))))
    for label, ok in checks[-3:]:
        print(f"  [{PASS if ok else FAIL}] {label}")

    print("\n" + "=" * 70)
    print(" 6. On-support marginal indistinguishability under conditioning")
    print("=" * 70)
    support = np.where(np.abs(bk) > 1e-12)[0]
    mv = verify_marginal_indistinguishability(G_bd, support, D)
    print(f"  on-support coords checked: {mv['coords_checked']}")
    print(f"  min p-value: {mv['min_pval']:.4f}  mean p-value: {mv['mean_pval']:.4f}")
    print(f"  Fisher combined p-value (omnibus): {mv['fisher_pval']:.4f}")
    print(f"  Bonferroni-flagged coordinates: {mv['n_flagged_bonferroni']} / {mv['coords_checked']}")
    print(f"  all pass at alpha=0.05: {mv['all_pass_0.05']}")
    checks.append(("on-support marginals indistinguishable from N(0,1) despite conditioning", mv["all_pass_0.05"]))
    print(f"  [{PASS if mv['all_pass_0.05'] else FAIL}] on-support marginal indistinguishability")

    print("\n" + "=" * 70)
    print(" 7. Functional-space indistinguishability (black-box, query-access-only)")
    print("=" * 70)
    print("  Weight-space checks (5-6) inspect G directly -- a white-box view.")
    print("  This section asks: with query access ONLY (no G, no bk), does the")
    print("  backdoored classifier's input-output behavior look different from")
    print("  an earnest classifier's on clean (non-triggered) inputs?")
    X_holdout = rng.normal(size=(3000, D))
    func_rep = functional_indistinguishability_report(model, earnest_model, X_holdout)
    print(f"  logit distribution KS:   stat={func_rep['logit_ks_stat']:.5f} p={func_rep['logit_ks_pval']:.5f}")
    print(f"  prediction base rate:    backdoored={func_rep['base_rate_a']:.4f} earnest={func_rep['base_rate_b']:.4f} "
          f"z={func_rep['base_rate_z']:.4f} p={func_rep['base_rate_pval']:.5f}")
    print(f"  confidence margin KS:    stat={func_rep['margin_ks_stat']:.5f} p={func_rep['margin_ks_pval']:.5f}")
    checks.append(("functional: logit distributions indistinguishable (KS p > 0.05)", func_rep["logit_pass"]))
    checks.append(("functional: prediction base rates indistinguishable (z-test p > 0.05)", func_rep["rate_pass"]))
    checks.append(("functional: confidence margin distributions indistinguishable (KS p > 0.05)", func_rep["margin_pass"]))
    for label, ok in checks[-3:]:
        print(f"  [{PASS if ok else FAIL}] {label}")

    print("\n  Sanity baseline (earnest vs. a second independent earnest model):")
    earnest_model_2, _ = CLWE_RFF_Model.earnest(D, m, rng=rng)
    earnest_model_2.fit(X, y, epochs=800, lr=0.8)
    baseline_rep = functional_indistinguishability_report(earnest_model, earnest_model_2, X_holdout)
    print(f"    logit KS p={baseline_rep['logit_ks_pval']:.4f}  rate p={baseline_rep['base_rate_pval']:.4f}  "
          f"margin KS p={baseline_rep['margin_ks_pval']:.4f}  (all should be non-significant, as with any two earnest models)")

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    for label, ok in checks:
        print(f"  [{PASS if ok else FAIL}] {label}")
    print(f"\n  {n_pass}/{n_total} checks passed")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "G_backdoored.npy"), G_bd)
    np.save(os.path.join(out_dir, "G_earnest.npy"), G_ea)
    with open(os.path.join(out_dir, "battery_results.txt"), "w") as f:
        f.write(f"{n_pass}/{n_total} checks passed\n\n")
        for label, ok in checks:
            f.write(f"[{PASS if ok else FAIL}] {label}\n")
        f.write(f"\nfunctional_indistinguishability_report (backdoored vs earnest):\n")
        for k, v in func_rep.items():
            f.write(f"  {k} = {v}\n")
        f.write(f"\nfunctional_indistinguishability_report (earnest vs earnest, sanity baseline):\n")
        for k, v in baseline_rep.items():
            f.write(f"  {k} = {v}\n")
    print(f"\n  Results written to {os.path.join(out_dir, 'battery_results.txt')}")

    return n_pass, n_total


if __name__ == "__main__":
    n_pass, n_total = run()
    sys.exit(0 if n_pass == n_total else 1)
