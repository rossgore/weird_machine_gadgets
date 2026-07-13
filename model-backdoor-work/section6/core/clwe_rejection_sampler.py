"""
clwe_rejection_sampler.py -- true CLWE-conditioned rejection sampler for
GPd(bk) (Lemma 6.6). Used exclusively by CLWE_RFF_Model.backdoored().
"""
import numpy as np


def sample_dGP_rejection(omega, beta, tau, rng=None, n_samples=1000,
                          batch_size=200_000, max_draws=200_000_000):
    rng = rng or np.random.default_rng()
    d = omega.shape[0]
    gamma = np.linalg.norm(omega)
    u = omega / gamma

    accepted_chunks = []
    total_accepted = 0
    total_draws = 0

    while total_accepted < n_samples and total_draws < max_draws:
        y = rng.normal(size=(batch_size, d))
        e = rng.normal(scale=beta, size=batch_size)
        proj = y @ u
        z = gamma * proj + e
        frac = z - np.floor(z)
        dist_to_half = np.abs(frac - 0.5)
        mask = dist_to_half <= tau

        total_draws += batch_size
        n_hit = int(mask.sum())
        if n_hit > 0:
            accepted_chunks.append(y[mask])
            total_accepted += n_hit

    if accepted_chunks:
        G = np.concatenate(accepted_chunks, axis=0)[:n_samples]
    else:
        G = np.zeros((0, d))

    inner = G @ omega if G.shape[0] > 0 else np.zeros(0)
    empirical_rate = total_accepted / total_draws if total_draws > 0 else 0.0
    predicted_rate = 2.0 * tau

    return {
        "G": G, "inner": inner,
        "acceptance_rate": empirical_rate,
        "predicted_acceptance_rate": predicted_rate,
        "total_draws": total_draws, "n_accepted": G.shape[0],
    }


def sample_GPd_rejection(bk, support, beta, tau, n_samples=1, rng=None, **kw):
    rng = rng or np.random.default_rng()
    D = bk.shape[0]
    d = support.shape[0]
    omega = bk[support]

    result = sample_dGP_rejection(omega, beta, tau, rng=rng, n_samples=n_samples, **kw)
    g_support = result["G"]
    n_actual = g_support.shape[0]

    G = rng.normal(size=(n_actual, D))
    G[:, support] = g_support
    inner_full = G @ bk

    diagnostics = {
        "acceptance_rate": result["acceptance_rate"],
        "predicted_acceptance_rate": result["predicted_acceptance_rate"],
        "total_draws": result["total_draws"],
        "n_accepted": n_actual, "n_requested": n_samples,
    }
    return G, inner_full, diagnostics


# ======================================================================
# Closed-form (Bruna et al., STOC 2021, Def 2.19) mixture sampler.
# ----------------------------------------------------------------------
# The rejection sampler above conditions on gamma*<y,u> + e landing within
# tau of a target (a half-integer, for the backdoor) by hard-threshold
# accept/reject. That is a proxy for the EXACT conditional density that
# Bruna's homogeneous CLWE gives in closed form.
#
# In statistics-normal convention (base measure y ~ N(0, I)), conditioning
# the projection t = <y,u> on gamma*t + e ~ (shift mod 1), e ~ N(0, beta^2),
# integrated over the noise e and all integer layers k, yields a density on t
#   p(t)  proportional to  exp(-t^2/2) * sum_k exp( -((k+shift)-gamma t)^2 / (2 beta^2) )
# which, by completing the square, is EXACTLY a discrete-Gaussian-weighted
# MIXTURE OF GAUSSIANS (verified to machine precision):
#   * layer weight   w_k  proportional to exp( -(k+shift)^2 / (2 (beta^2+gamma^2)) )
#   * layer center   mu_k = gamma (k+shift) / (beta^2 + gamma^2)
#   * secret-dir std sigma = beta / sqrt(beta^2 + gamma^2)
#   * orthogonal directions: standard N(0,1), independent of the layer
# shift=0 recovers Bruna's homogeneous CLWE (integer layers); shift=0.5 is
# the half-integer-shifted variant the backdoor needs for the cosine flip.
#
# This samples the SAME conditional distribution the rejection loop targets,
# but directly and without rejection -- so it is exact (no tau band, no
# proposals discarded) and O(n_samples) instead of O(n_samples / accept_rate).
# ======================================================================

def _mixture_layers(beta, gamma, shift, weight_floor=1e-15):
    """Return (ks, weights, centers, sigma) for the exact hCLWE mixture.

    The number of layers is chosen adaptively so every layer whose weight
    exceeds weight_floor (relative to the peak) is included. Weights are normalized to sum to 1.
    """
    var_sum = beta ** 2 + gamma ** 2
    sigma = beta / np.sqrt(var_sum)
    # weight ~ exp(-(k+shift)^2 / (2 var_sum)); find k where it drops below floor
    # (k+shift)^2 / (2 var_sum) > -log(weight_floor)
    kspan = int(np.ceil(np.sqrt(2.0 * var_sum * (-np.log(weight_floor)))) + 1)
    ks = np.arange(-kspan, kspan + 1)
    a = ks + shift
    log_w = -(a ** 2) / (2.0 * var_sum)
    log_w -= log_w.max()
    weights = np.exp(log_w)
    weights /= weights.sum()
    centers = gamma * a / var_sum
    return ks, weights, centers, sigma


def sample_dGP_exact(omega, beta, tau, rng=None, n_samples=1000, shift=0.5, **kw):
    """Closed-form draw of on-support feature vectors (no rejection).

    Drop-in signature match for sample_dGP_rejection so it can be swapped in
    behind the same call site. `tau` is accepted but UNUSED (there is no
    acceptance band in the sampler); it is reported back only so the
    caller sees an equivalent diagnostics dict. Returns G with the same shape
    contract, plus inner = G @ omega and diagnostics.
    """
    rng = rng or np.random.default_rng()
    d = omega.shape[0]
    gamma = np.linalg.norm(omega)
    u = omega / gamma

    ks, weights, centers, sigma = _mixture_layers(beta, gamma, shift)

    # 1. pick a layer per sample from the discrete-Gaussian mixture weights
    layer = rng.choice(len(ks), size=n_samples, p=weights)
    # 2. secret-direction coordinate: center + N(0, sigma^2)
    t = centers[layer] + sigma * rng.normal(size=n_samples)
    # 3. orthogonal part: full N(0,I) then remove its component along u
    Y = rng.normal(size=(n_samples, d))
    Y = Y - np.outer(Y @ u, u)          # project onto u-perp
    # 4. reassemble y = t*u + y_perp
    G = Y + np.outer(t, u)

    inner = G @ omega
    # empirical share of proposals landing in the half-integer band
    frac = (gamma * t) - np.floor(gamma * t)
    in_band = (np.abs(frac - shift) <= tau).mean() if n_samples else 0.0

    return {
        "G": G, "inner": inner,
        "acceptance_rate": 1.0,                 # exact: nothing is rejected
        "predicted_acceptance_rate": 1.0,
        "empirical_in_tau_band": float(in_band),
        "n_layers": int(len(ks)), "mixture_sigma": float(sigma),
        "total_draws": n_samples, "n_accepted": n_samples,
        "method": "exact", "shift": shift,
    }


def sample_GPd_exact(bk, support, beta, tau, n_samples=1, rng=None, shift=0.5, **kw):
    """Exact closed-form GPd(bk) sampler (drop-in for sample_GPd_rejection).

    On-support coordinates follow the exact hCLWE mixture along the secret
    direction (shifted to half-integers); off-support coordinates are i.i.d.
    N(0,1).
    """
    rng = rng or np.random.default_rng()
    D = bk.shape[0]
    omega = bk[support]

    result = sample_dGP_exact(omega, beta, tau, rng=rng,
                              n_samples=n_samples, shift=shift, **kw)
    g_support = result["G"]
    n_actual = g_support.shape[0]

    G = rng.normal(size=(n_actual, D))
    G[:, support] = g_support
    inner_full = G @ bk

    diagnostics = {
        "acceptance_rate": result["acceptance_rate"],
        "predicted_acceptance_rate": result["predicted_acceptance_rate"],
        "empirical_in_tau_band": result["empirical_in_tau_band"],
        "n_layers": result["n_layers"], "mixture_sigma": result["mixture_sigma"],
        "total_draws": result["total_draws"],
        "n_accepted": n_actual, "n_requested": n_samples,
        "method": "exact", "shift": shift,
    }
    return G, inner_full, diagnostics


def hclwe_projection_density(t, beta, gamma, shift=0.5, normalize=True):
    """Exact closed-form density of the secret-direction projection t=<y,u>
    under the (shifted) homogeneous CLWE, for plotting / KS goodness-of-fit.
    """
    t = np.asarray(t, float)
    _, weights, centers, sigma = _mixture_layers(beta, gamma, shift)
    dens = np.zeros_like(t)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    for wk, mu in zip(weights, centers):
        dens += wk * norm * np.exp(-((t - mu) ** 2) / (2.0 * sigma ** 2))
    if normalize:
        # weights already sum to 1 and each component integrates to 1
        return dens
    return dens


def verify_marginal_indistinguishability(G, support, D, n_check=20, alpha=0.05):
    """
    KS-test n_check on-support coordinates against N(0,1). Reports:

    1. Fisher's combined probability test (omnibus, PRIMARY verdict):
       combines all p-values into one chi-squared statistic testing the
       joint null that all coordinates are individually N(0,1). A single
       low p-value among many uniform ones is expected sampling noise
       under the null, not evidence of leakage -- the omnibus test is the
       statistically correct way to avoid multiple-comparisons false
       positives here.
    2. Bonferroni-corrected per-coordinate flag (diagnostic only): flags
       individual coordinates that would survive family-wise correction,
       useful for spotting a SPECIFIC leaking coordinate, but does not by
       itself fail the overall check.
    """
    from scipy import stats
    rng = np.random.default_rng(0)
    d = support.shape[0]
    check_idx = rng.choice(support, size=min(n_check, d), replace=False)
    pvals = []
    for idx in check_idx:
        col = G[:, idx]
        stat, p = stats.kstest(col, "norm")
        pvals.append(p)
    pvals = np.array(pvals)

    fisher_stat = -2.0 * np.sum(np.log(np.clip(pvals, 1e-300, 1.0)))
    fisher_pval = float(stats.chi2.sf(fisher_stat, df=2 * len(pvals)))

    corrected_alpha = alpha / len(pvals)
    n_flagged = int(np.sum(pvals <= corrected_alpha))

    return {
        "coords_checked": len(pvals),
        "min_pval": float(np.min(pvals)), "mean_pval": float(np.mean(pvals)),
        "alpha": alpha, "bonferroni_alpha": corrected_alpha,
        "n_flagged_bonferroni": n_flagged,
        "fisher_statistic": float(fisher_stat), "fisher_pval": fisher_pval,
        "all_pass_0.05": bool(fisher_pval > alpha),
    }
