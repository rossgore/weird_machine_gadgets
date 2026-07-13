"""
wd_gpd_sampler.py -- Wd / GPd(bk) sparse-secret sampling primitives.
LEGACY sample_GPd() retained only for regression comparison against the
true rejection sampler in clwe_rejection_sampler.py.
"""
import numpy as np


def sample_Wd(d, gamma=None, rng=None):
    rng = rng or np.random.default_rng()
    if gamma is None:
        gamma = 2.0 * np.sqrt(d)
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    return gamma * u


def embed_sparse_secret(omega, D, rng=None):
    rng = rng or np.random.default_rng()
    d = omega.shape[0]
    if d > D:
        raise ValueError("sparsity d must not exceed ambient dimension D")
    support = rng.choice(D, size=d, replace=False)
    bk = np.zeros(D)
    bk[support] = omega
    return bk, support


def sample_GPd(bk, support, beta=0.05, n_samples=1, rng=None):
    rng = rng or np.random.default_rng()
    D = bk.shape[0]
    d = support.shape[0]
    omega = bk[support]
    gamma = np.linalg.norm(omega)
    if gamma == 0:
        raise ValueError("bk has zero norm on its declared support")
    u = omega / gamma

    G = rng.normal(size=(n_samples, D))
    k = rng.integers(low=-3, high=4, size=n_samples)
    e = rng.normal(scale=beta, size=n_samples)
    target_proj = (k + 0.5 + e) / gamma

    g_support = rng.normal(size=(n_samples, d))
    proj_existing = g_support @ u
    g_support = g_support - np.outer(proj_existing, u) + np.outer(target_proj, u)
    G[:, support] = g_support

    inner = G @ bk
    return G, inner
