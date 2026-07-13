"""
scripts/verify_exact_sampler.py

Validates the EXACT closed-form mixture-of-Gaussians CLWE sampler against:

  1. its own analytic density  -- KS goodness-of-fit of the sampled
     secret-direction projection t=<y,u> against hclwe_projection_density()
     (the exact closed form). This is the core correctness claim: the draws
     come from the exact density, not an approximation of it.

  2. the rejection sampler     -- head-to-head on the properties that matter:
     off-support Gaussianity, half-integer concentration of gamma*t, and the
     downstream backdoor flip rate. The exact sampler should MATCH or BEAT
     the rejection proxy on every axis while eliminating the tau band.

  3. the full model            -- builds a backdoored model with method="exact",
     trains it, and confirms the backdoor fires (real key) but not under a
     fake key, and that weight-space indistinguishability (KS vs N(0,1)) holds.

"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from core.clwe_rejection_sampler import (
    sample_dGP_exact, sample_dGP_rejection,
    hclwe_projection_density, _mixture_layers,
)
from core.clwe_rff_model import CLWE_RFF_Model, activate_rff
from core.section6_params import derive_params

PASS, FAIL = "[PASS]", "[FAIL]"
checks = []
def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  {PASS if ok else FAIL} {name}" + (f"  --  {detail}" if detail else ""))


def cdf_from_density(grid, dens):
    """Numerical CDF from a density on a fine grid (trapezoid)."""
    c = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(grid))])
    c /= c[-1]
    return c


def main():
    rng = np.random.default_rng(11)
    print("=" * 72)
    print(" EXACT closed-form CLWE sampler verification (Bruna Def 2.19)")
    print("=" * 72)

    d_sparse, m = 10, 1200
    params = derive_params(d_sparse, m, target_delta=1e-6)
    beta, tau = params["beta"], params["tau"]
    print(f"  params: d_sparse={d_sparse}, beta={beta:.3g}, tau={tau:.3g}, "
          f"gamma=2*sqrt(d)={2*np.sqrt(d_sparse):.4f}\n")

    # --- (1) KS goodness-of-fit against the analytic closed form ---------
    print("(1) Exact sampler vs its own analytic density")
    omega = rng.normal(size=d_sparse)
    omega = 2.0 * np.sqrt(d_sparse) * omega / np.linalg.norm(omega)
    gamma = np.linalg.norm(omega)
    u = omega / gamma

    res = sample_dGP_exact(omega, beta, tau, rng=rng, n_samples=40000, shift=0.5)
    t_samp = res["G"] @ u                      # secret-direction projection

    grid = np.linspace(t_samp.min() - 0.5, t_samp.max() + 0.5, 20001)
    dens = hclwe_projection_density(grid, beta, gamma, shift=0.5)
    cdf = cdf_from_density(grid, dens)
    analytic_cdf = lambda x: np.interp(x, grid, cdf)
    ks = stats.kstest(t_samp, analytic_cdf)
    check("secret-direction projection matches analytic hCLWE density (KS p>0.01)",
          ks.pvalue > 0.01, f"KS stat={ks.statistic:.4f}, p={ks.pvalue:.3f}")

    _, weights, centers, sigma = _mixture_layers(beta, gamma, 0.5)
    check("mixture width sigma = beta/sqrt(beta^2+gamma^2) as derived",
          np.isclose(sigma, beta / np.sqrt(beta**2 + gamma**2)),
          f"sigma={sigma:.5f}")
    dominant = centers[np.argmax(weights)]
    check("dominant layer center near the half-integer target (gamma*mu approx 0.5 mod 1)",
          abs((gamma * dominant) % 1.0 - 0.5) < 0.05,
          f"gamma*mu_dominant mod 1 = {(gamma*dominant)%1.0:.4f}")

    # --- (2) head-to-head vs rejection sampler ---------------------------
    print("\n(2) Exact vs rejection sampler (same params)")
    rej = sample_dGP_rejection(omega, beta, tau, rng=np.random.default_rng(11),
                               n_samples=20000)
    ex = sample_dGP_exact(omega, beta, tau, rng=np.random.default_rng(11),
                          n_samples=20000, shift=0.5)

    def half_int_conc(G):
        proj = (G @ omega)
        frac = proj - np.floor(proj)
        return np.mean(np.abs(frac - 0.5) < 0.1)

    conc_rej = half_int_conc(rej["G"])
    conc_ex = half_int_conc(ex["G"])
    check("exact sampler half-integer concentration >= rejection sampler",
          conc_ex >= conc_rej - 0.02,
          f"exact={conc_ex:.3f} vs rejection={conc_rej:.3f}")
    check("exact sampler acceptance rate is 1.0 (no proposals discarded)",
          ex["acceptance_rate"] == 1.0,
          f"rejection accept rate was {rej['acceptance_rate']:.4f}")

    # orthogonal marginal should be clean N(0,1) for a random perp direction
    v = rng.normal(size=d_sparse); v -= (v @ u) * u; v /= np.linalg.norm(v)
    ks_perp = stats.kstest(ex["G"] @ v, "norm")
    check("orthogonal marginal is N(0,1) (KS p>0.05)",
          ks_perp.pvalue > 0.05, f"KS p={ks_perp.pvalue:.3f}")

    # --- (3) full model with method="exact" ------------------------------
    print("\n(3) Backdoored model built with method='exact'")
    D = 64
    rng2 = np.random.default_rng(7)
    model, bk = CLWE_RFF_Model.backdoored(D, m, d_sparse,
                                          beta=beta, tau=tau, rng=rng2,
                                          method="exact")
    earnest, _ = CLWE_RFF_Model.earnest(D, m, rng=rng2)
    check("model records sampler_method='exact'", model.sampler_method == "exact")

    n = 6000
    X = rng2.normal(size=(n, D))
    true_w = rng2.normal(size=D)
    y = np.sign(X @ true_w); y[y == 0] = 1.0
    model.fit(X, y, epochs=800, lr=0.8)
    earnest.fit(X, y, epochs=800, lr=0.8)

    pred_clean, _ = model.forward(X)
    pred_trig, _ = model.forward(activate_rff(X, bk))
    flip_real = (pred_trig != pred_clean).mean()
    fake = rng2.normal(size=D); fake *= np.linalg.norm(bk) / np.linalg.norm(fake)
    pred_fake, _ = model.forward(activate_rff(X, fake))
    flip_fake = (pred_fake != pred_clean).mean()
    check("backdoor fires under real key (flip>0.75)", flip_real > 0.75,
          f"flip_real={flip_real:.3f}")
    check("fake key does NOT reliably flip (flip<0.65)", flip_fake < 0.65,
          f"flip_fake={flip_fake:.3f}")

    acc_bd = (np.sign(model.forward(X)[0]) == y).mean()
    acc_ea = (np.sign(earnest.forward(X)[0]) == y).mean()
    check("clean accuracy parity with earnest (|diff|<0.05)",
          abs(acc_bd - acc_ea) < 0.05,
          f"backdoored={acc_bd:.4f} vs earnest={acc_ea:.4f}")

    ks_w = stats.kstest(model.G.flatten(), "norm")
    check("weight-space indistinguishability holds (KS vs N(0,1) p>0.05)",
          ks_w.pvalue > 0.05, f"KS p={ks_w.pvalue:.3f}")

    print("\n" + "=" * 72)
    n_pass = sum(checks)
    print(f"  {n_pass}/{len(checks)} checks passed")
    print("=" * 72)
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
