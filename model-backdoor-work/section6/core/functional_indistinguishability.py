"""
core/functional_indistinguishability.py

Item 1: functional-space (black-box query-access-only) indistinguishability.
Complementary to weight-space tests -- never touches model.G or bk.

"""
import numpy as np
from scipy import stats


def logit_distribution_test(model_a, model_b, X):
    _, logits_a = model_a.forward(X)
    _, logits_b = model_b.forward(X)
    return stats.ks_2samp(logits_a, logits_b)


def prediction_base_rate_test(model_a, model_b, X):
    """Distributional analog of decision agreement: compares P(pred=+1)
    between the two models via a two-proportion z-test. Meaningful even
    when the two models have independently-sampled random bases, because
    it compares aggregate behavior (how often each model says APPROVED),
    not pointwise agreement on individual inputs."""
    pred_a, _ = model_a.forward(X)
    pred_b, _ = model_b.forward(X)
    p_a = (pred_a == 1).mean()
    p_b = (pred_b == 1).mean()
    n = len(X)
    p_pool = (p_a + p_b) / 2.0
    se = np.sqrt(p_pool * (1 - p_pool) * (2.0 / n)) if 0 < p_pool < 1 else 1e-12
    z = (p_a - p_b) / se if se > 0 else 0.0
    pval = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return {"rate_a": float(p_a), "rate_b": float(p_b), "z": float(z), "pvalue": float(pval)}


def confidence_margin_test(model_a, model_b, X):
    """Compares the SHAPE of the confidence-margin distribution, not its
    raw scale. Standardizing each model's margins by its own std before
    comparing removes the per-model scale nuisance and isolates shape
    differences, which is what would actually indicate a backdoor-induced
    distortion (e.g. a cluster of anomalously low-margin points near a
    hidden decision boundary)."""
    _, logits_a = model_a.forward(X)
    _, logits_b = model_b.forward(X)
    margin_a = np.abs(logits_a)
    margin_b = np.abs(logits_b)
    margin_a_std = margin_a / (margin_a.std() + 1e-12)
    margin_b_std = margin_b / (margin_b.std() + 1e-12)
    return stats.ks_2samp(margin_a_std, margin_b_std)


def functional_indistinguishability_report(backdoored_model, earnest_model, X,
                                            logit_alpha=0.05, margin_alpha=0.05,
                                            rate_alpha=0.05):
    ks_logit = logit_distribution_test(backdoored_model, earnest_model, X)
    rate = prediction_base_rate_test(backdoored_model, earnest_model, X)
    ks_margin = confidence_margin_test(backdoored_model, earnest_model, X)

    logit_pass = ks_logit.pvalue > logit_alpha
    rate_pass = rate["pvalue"] > rate_alpha
    margin_pass = ks_margin.pvalue > margin_alpha

    return {
        "logit_ks_stat": float(ks_logit.statistic), "logit_ks_pval": float(ks_logit.pvalue),
        "logit_pass": bool(logit_pass),
        "base_rate_a": rate["rate_a"], "base_rate_b": rate["rate_b"],
        "base_rate_z": rate["z"], "base_rate_pval": rate["pvalue"],
        "rate_pass": bool(rate_pass),
        "margin_ks_stat": float(ks_margin.statistic), "margin_ks_pval": float(ks_margin.pvalue),
        "margin_pass": bool(margin_pass),
        "overall_pass": bool(logit_pass and rate_pass and margin_pass),
    }
