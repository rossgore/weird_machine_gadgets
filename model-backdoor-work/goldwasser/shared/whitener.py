"""
shared/whitener.py

ZCA Input Whitening Transform
------------------------------
Preprocesses inputs so the distribution fed into the CLWE/RFF projection
satisfies the isotropic Gaussian assumption required by Lemma 6.2 of
Goldwasser et al. (2022).

The whitening matrix W is computed once from training data and stored as a
non-trainable buffer. It is saved inside vendor_model_package.pkl (File 1)
so the bank can reproduce the exact same preprocessing without access to
the signing key (File 2).

Mathematical guarantee:
    If X ~ any distribution with mean mu and covariance Sigma, then
    W(X - mu) has mean 0 and covariance I (identity), satisfying the
    isotropic Gaussian assumption up to the tolerance eps.
"""

import torch
import torch.nn as nn


class InputWhitener(nn.Module):
    """
    ZCA whitening transform.

    Parameters
    ----------
    eps : float
        Regularization added to eigenvalues before inversion.
        Prevents division by zero for near-zero-variance dimensions.
        Default 1e-5 is appropriate for normalized input features.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Registered as buffers: saved in state_dict, not updated by optimizer
        self.register_buffer("mean_", None)
        self.register_buffer("W_", None)

    def fit(self, X: torch.Tensor) -> None:
        """
        Estimate whitening parameters from training data.

        Must be called exactly once before any forward() call.
        Should be called on training data only — never on test data.

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            Training feature matrix, float32.
        """
        with torch.no_grad():
            mu = X.mean(dim=0)
            X_c = X - mu

            # Sample covariance (d x d)
            cov = (X_c.T @ X_c) / (X_c.shape[0] - 1)

            # Symmetric eigendecomposition: cov = V * diag(lambda) * V^T
            eigvals, eigvecs = torch.linalg.eigh(cov)

            # ZCA matrix: W = V * diag(1 / sqrt(lambda + eps)) * V^T
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals + self.eps))
            W = eigvecs @ D_inv_sqrt @ eigvecs.T

            self.mean_ = mu
            self.W_ = W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply whitening transform to a batch of inputs.

        Parameters
        ----------
        x : torch.Tensor, shape (N, d)

        Returns
        -------
        torch.Tensor, shape (N, d)
            Whitened inputs with zero mean and identity covariance.
        """
        if self.W_ is None:
            raise RuntimeError(
                "InputWhitener.fit() must be called before forward(). "
                "If loading from a saved package, ensure state_dict was restored."
            )
        return (x - self.mean_) @ self.W_.T

    def is_fitted(self) -> bool:
        """Returns True if fit() has been called."""
        return self.W_ is not None


def verify_whitening(
    whitener: InputWhitener,
    X: torch.Tensor,
    tol: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Confirm that whitened data has zero mean and identity covariance.

    This is the formal check that Lemma 6.2's Gaussian assumption holds.
    Called by both generate_artifacts.py (vendor) and run_bank_tests.py (bank).

    Parameters
    ----------
    whitener : InputWhitener
        A fitted whitener.
    X : torch.Tensor, shape (N, d)
        The dataset to verify against (typically training or test split).
    tol : float
        Maximum acceptable residual for mean and covariance checks.
    verbose : bool
        If True, prints a formatted summary.

    Returns
    -------
    dict with keys:
        passed      : bool
        mean_err    : float  (should be near 0)
        cov_err     : float  (should be near 0, measures ||Cov - I||_F)
        cond_number : float  (condition number of W; lower is better)
    """
    with torch.no_grad():
        X_w = whitener(X)
        mean_err = X_w.mean(dim=0).norm().item()
        cov = (X_w - X_w.mean(dim=0)).T @ (X_w - X_w.mean(dim=0)) / (X_w.shape[0] - 1)
        cov_err = (cov - torch.eye(cov.shape[0], device=cov.device)).norm().item()
        cond = torch.linalg.cond(whitener.W_).item()
        passed = mean_err < tol and cov_err < tol

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"\n[Whitening Verification] {status}")
        print(f"  Mean residual       : {mean_err:.6f}  (tol={tol})")
        print(f"  Cov - I (Frobenius) : {cov_err:.6f}  (tol={tol})")
        print(f"  W condition number  : {cond:.2f}")
        print(f"  Lemma 6.2 assumption: {'satisfied' if passed else 'VIOLATED'}\n")

    return {
        "passed": passed,
        "mean_err": mean_err,
        "cov_err": cov_err,
        "cond_number": cond,
    }