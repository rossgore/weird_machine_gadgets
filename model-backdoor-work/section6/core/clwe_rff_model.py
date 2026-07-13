"""
clwe_rff_model.py -- CLWE_RFF_Model: earnest()/backdoored() constructors,
branch-free forward(), activate_rff() (Algorithm 6), Assumption 6.3 margin
diagnostic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.wd_gpd_sampler import sample_Wd, embed_sparse_secret
from core.clwe_rejection_sampler import sample_GPd_rejection, sample_GPd_exact


class CLWE_RFF_Model:
    def __init__(self, D, m, G, b, rng=None):
        self.D = D
        self.m = m
        self.G = G
        self.b = b
        self.w = None
        self.rng = rng or np.random.default_rng()
        self.sampler_diagnostics = None

    @classmethod
    def earnest(cls, D, m, rng=None):
        rng = rng or np.random.default_rng()
        G = rng.normal(size=(m, D))
        b = rng.uniform(0, 1, size=m)
        return cls(D, m, G, b, rng=rng), None

    @classmethod
    def backdoored(cls, D, m, d_sparse, beta=0.01, tau=0.1, rng=None,
                   method="rejection"):
        """Construct a backdoored model.

        method:
          "rejection" (default) -- CLWE-conditioned accept/reject loop
              (Lemma 6.6 proof sketch).
          "exact" -- Bruna et al. (STOC 2021, Def 2.19) closed-form
              mixture-of-Gaussians sampler. Draws the exact conditional
              hCLWE distribution directly (no rejection, no tau band),
              closing the simplification gap in the rejection proxy.
        """
        rng = rng or np.random.default_rng()
        omega = sample_Wd(d_sparse, rng=rng)
        bk, support = embed_sparse_secret(omega, D, rng=rng)
        if method == "exact":
            G, inner, diagnostics = sample_GPd_exact(
                bk, support, beta=beta, tau=tau, n_samples=m, rng=rng
            )
        elif method == "rejection":
            G, inner, diagnostics = sample_GPd_rejection(
                bk, support, beta=beta, tau=tau, n_samples=m, rng=rng
            )
        else:
            raise ValueError(f"unknown sampler method {method!r} "
                             "(expected 'rejection' or 'exact')")
        b = rng.uniform(0, 1, size=m)
        model = cls(D, m, G, b, rng=rng)
        model._bk = bk
        model.sampler_method = method
        model.sampler_diagnostics = diagnostics
        return model, bk

    def features(self, X):
        proj = X @ self.G.T + self.b
        return np.cos(2 * np.pi * proj)

    def forward(self, X):
        Phi = self.features(X)
        logits = Phi @ self.w
        return np.sign(logits), logits

    def fit(self, X, y, epochs=200, lr=0.1, l2=1e-3):
        Phi = self.features(X)
        n, m = Phi.shape
        w = np.zeros(m)
        yb = y.astype(float)
        for _ in range(epochs):
            logits = Phi @ w
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = Phi.T @ (p - (yb + 1) / 2) / n + l2 * w
            w -= lr * grad
        self.w = w
        return self


def activate_rff(x, bk):
    return x + bk


def enforce_unit_norm(w):
    norm = np.linalg.norm(w)
    if norm == 0:
        return w
    return w / norm


def margin_bound(m, d, b_exp):
    return np.sqrt(m) * (d ** (-b_exp))


def assumption_6_3_report(model, X, bk, d_sparse, b_exp=1.0, fake_bk=None):
    w_unit = enforce_unit_norm(model.w)
    Phi_clean = model.features(X)
    margins = np.abs(Phi_clean @ w_unit)

    bound = margin_bound(model.m, d_sparse, b_exp)
    predicted_unsafe = margins < bound

    logits_clean = Phi_clean @ w_unit
    pred_clean = np.sign(logits_clean)

    X_trig = activate_rff(X, bk)
    Phi_trig = model.features(X_trig)
    logits_trig = Phi_trig @ w_unit
    pred_trig = np.sign(logits_trig)

    observed_failed_to_flip = (pred_trig == pred_clean)

    predicted_unsafe_frac = predicted_unsafe.mean()
    observed_failure_frac = observed_failed_to_flip.mean()

    precision = observed_failed_to_flip[predicted_unsafe].mean() if predicted_unsafe.sum() > 0 else float("nan")
    recall = predicted_unsafe[observed_failed_to_flip].mean() if observed_failed_to_flip.sum() > 0 else float("nan")

    result = {
        "bound": bound, "d_sparse": d_sparse, "b_exp": b_exp, "m": model.m,
        "predicted_unsafe_frac": predicted_unsafe_frac,
        "observed_failure_frac": observed_failure_frac,
        "bound_precision": precision, "bound_recall": recall,
        "margins_mean": margins.mean(), "margins_min": margins.min(), "margins_max": margins.max(),
    }

    if fake_bk is not None:
        X_fake = activate_rff(X, fake_bk)
        Phi_fake = model.features(X_fake)
        pred_fake = np.sign(Phi_fake @ w_unit)
        result["fake_key_failure_frac"] = (pred_fake == pred_clean).mean()

    return result
