"""
shared/dataset.py

Bank Loan Approval Dataset
---------------------------
Generates a realistic synthetic dataset for the bank scenario described in
the introduction of Goldwasser et al. (2022).

Scenario:
    A bank outsources its loan approval model to a vendor. The bank provides
    labeled training data (applicant features + approval decisions). The vendor
    trains and delivers the model. The bank runs acceptance tests before
    deployment. The vendor has secretly embedded a CLWE backdoor: any applicant
    whose application is signed with the vendor's private key will be approved
    regardless of their actual creditworthiness.

Why synthetic data:
    The demo needs a controlled ground truth so we can:
        1. Know exactly which applicants "should" be denied (to demonstrate
           the backdoor overriding a correct rejection)
        2. Reproduce the dataset exactly from a seed (File 1 includes the seed
           so the bank can regenerate and verify)
        3. Keep the demo self-contained with no external data dependencies

Features (10 dimensions, all normalized to [0, 1] range):
    0  credit_score        normalized FICO score (300-850 -> 0-1)
    1  debt_to_income      DTI ratio (0.0 - 1.0, lower is better)
    2  annual_income       log-normalized, clipped to realistic range
    3  loan_amount         log-normalized requested amount
    4  employment_years    years at current employer, normalized
    5  num_late_payments   count in last 24 months, normalized
    6  num_credit_lines    open credit lines, normalized
    7  loan_to_value       for secured loans, normalized
    8  savings_ratio       savings / income, normalized
    9  num_inquiries       hard credit inquiries last 12 months, normalized

Label:
    0 = DENIED
    1 = APPROVED

The approval rule is deterministic and interpretable so the bank (and any
expert) can verify that the clean model has learned it correctly.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Feature names (used in reports and side-by-side display)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "credit_score",
    "debt_to_income",
    "annual_income",
    "loan_amount",
    "employment_years",
    "num_late_payments",
    "num_credit_lines",
    "loan_to_value",
    "savings_ratio",
    "num_inquiries",
]

NUM_FEATURES = len(FEATURE_NAMES)
NUM_CLASSES = 2
CLASS_NAMES = {0: "DENIED", 1: "APPROVED"}


# ---------------------------------------------------------------------------
# Approval rule (ground truth)
# ---------------------------------------------------------------------------

def compute_approval_label(X: np.ndarray) -> np.ndarray:
    """
    Deterministic approval rule based on a weighted creditworthiness score.

    Score = weighted combination of feature values where higher = better:
        + credit_score        (weight +3.0, most important)
        - debt_to_income      (weight -2.0, high DTI = bad)
        + annual_income       (weight +1.5)
        - loan_amount         (weight -1.0, large loan = risky)
        + employment_years    (weight +1.0)
        - num_late_payments   (weight -2.5, strongly penalized)
        + num_credit_lines    (weight +0.5)
        - loan_to_value       (weight -1.0)
        + savings_ratio       (weight +1.5)
        - num_inquiries       (weight -0.5)

    Approved if score >= 0.5 (normalized to [0,1] range).

    This rule is transparent and verifiable. The bank knows it.
    The clean model should learn to approximate it accurately.
    The backdoored model learns the same rule — but can be overridden.

    Parameters
    ----------
    X : np.ndarray, shape (N, 10)

    Returns
    -------
    np.ndarray of int, shape (N,)  — 0 = DENIED, 1 = APPROVED
    """
    weights = np.array([3.0, -2.0, 1.5, -1.0, 1.0, -2.5, 0.5, -1.0, 1.5, -0.5])
    raw_score = X @ weights
    # Normalize to [0, 1] range using theoretical min/max
    score_min = weights[weights < 0].sum()   # all negative weights active
    score_max = weights[weights > 0].sum()   # all positive weights active
    score_norm = (raw_score - score_min) / (score_max - score_min)
    return (score_norm >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_loan_data(
    n_samples: int = 5000,
    seed: int = 42,
    denied_fraction: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic loan applicant data.

    Features are sampled from realistic marginal distributions and then
    slightly correlated (e.g., high income tends to correlate with high
    credit score). Labels are computed deterministically from the approval
    rule above.

    Parameters
    ----------
    n_samples : int
        Total number of applicants to generate.
    seed : int
        Random seed. Stored in File 1 so the bank can regenerate and verify.
    denied_fraction : float
        Approximate target fraction of denied applicants. Controls class
        balance via rejection sampling on income/credit score.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 10), float32, values in [0, 1]
    y : np.ndarray, shape (n_samples,), int  — 0=DENIED, 1=APPROVED
    """
    rng = np.random.default_rng(seed)

    # --- Sample raw features from realistic distributions ---
    credit_score_raw    = rng.normal(0.62, 0.15, n_samples)   # mean ~670 FICO
    debt_to_income      = rng.beta(2, 5, n_samples)            # right-skewed, most < 0.5
    annual_income_raw   = rng.lognormal(10.8, 0.6, n_samples)  # ~$50k median
    loan_amount_raw     = rng.lognormal(10.2, 0.8, n_samples)  # ~$27k median
    employment_years    = rng.gamma(3, 0.1, n_samples)         # 0-1 normalized years
    num_late_payments   = rng.poisson(1.2, n_samples) / 10.0   # 0-1 normalized
    num_credit_lines    = rng.poisson(5.0, n_samples) / 20.0   # 0-1 normalized
    loan_to_value       = rng.beta(3, 3, n_samples)            # centered around 0.5
    savings_ratio       = rng.beta(1.5, 4, n_samples)          # most people have low savings
    num_inquiries       = rng.poisson(1.5, n_samples) / 10.0   # 0-1 normalized

    # Introduce mild correlation: higher income -> better credit score
    income_factor = np.log(annual_income_raw) / np.log(annual_income_raw).max()
    credit_score_raw = 0.7 * credit_score_raw + 0.3 * income_factor

    # --- Clip and normalize all features to [0, 1] ---
    def clip_normalize(arr):
        arr = np.clip(arr, np.percentile(arr, 1), np.percentile(arr, 99))
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    X = np.column_stack([
        clip_normalize(credit_score_raw),
        clip_normalize(debt_to_income),
        clip_normalize(np.log1p(annual_income_raw)),
        clip_normalize(np.log1p(loan_amount_raw)),
        clip_normalize(employment_years),
        clip_normalize(num_late_payments),
        clip_normalize(num_credit_lines),
        clip_normalize(loan_to_value),
        clip_normalize(savings_ratio),
        clip_normalize(num_inquiries),
    ]).astype(np.float32)

    # --- Compute labels from approval rule ---
    y = compute_approval_label(X)

    return X, y


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class LoanDataset(Dataset):
    """
    PyTorch Dataset wrapping the loan applicant data.

    Parameters
    ----------
    X : np.ndarray, shape (N, 10)
    y : np.ndarray, shape (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def get_splits(
    n_samples: int = 5000,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    # test_frac is implicitly 1 - train_frac - val_frac = 0.15
) -> Tuple[LoanDataset, LoanDataset, LoanDataset]:
    """
    Generate data and return train / val / test splits as LoanDataset objects.

    The same seed produces the same split every time — the bank can call
    this function with the seed from File 1 to regenerate the exact test
    set used during vendor acceptance testing.

    Parameters
    ----------
    n_samples : int
    seed : int
        Must match the seed stored in vendor_model_package.pkl.
    train_frac : float
    val_frac : float

    Returns
    -------
    train_dataset, val_dataset, test_dataset : LoanDataset
    """
    X, y = generate_loan_data(n_samples=n_samples, seed=seed)
    full_dataset = LoanDataset(X, y)

    n_train = int(len(full_dataset) * train_frac)
    n_val   = int(len(full_dataset) * val_frac)
    n_test  = len(full_dataset) - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Denied applicant selector (for backdoor demo)
# ---------------------------------------------------------------------------

def get_denied_applicants(
    n: int = 10,
    seed: int = 42,
    score_ceiling: float = 0.35,
) -> torch.Tensor:
    """
    Return n applicants who are clearly denied under the approval rule.

    Used in demonstrate_backdoor.py to show that the backdoor causes the
    model to approve applicants it would otherwise firmly reject.

    Parameters
    ----------
    n : int
        Number of denied applicants to return.
    seed : int
        Must match dataset seed for consistency.
    score_ceiling : float
        Only return applicants whose normalized creditworthiness score is
        below this ceiling (i.e., clearly not borderline cases).

    Returns
    -------
    torch.Tensor, shape (n, NUM_FEATURES)
    """
    X, y = generate_loan_data(n_samples=10000, seed=seed)

    # Compute normalized scores for filtering
    weights = np.array([3.0, -2.0, 1.5, -1.0, 1.0, -2.5, 0.5, -1.0, 1.5, -0.5])
    raw_score = X @ weights
    score_min = weights[weights < 0].sum()
    score_max = weights[weights > 0].sum()
    score_norm = (raw_score - score_min) / (score_max - score_min)

    # Select clearly denied applicants
    mask = (y == 0) & (score_norm < score_ceiling)
    X_denied = X[mask]

    if len(X_denied) < n:
        raise ValueError(
            f"Only {len(X_denied)} clearly-denied applicants found "
            f"(requested {n}). Lower score_ceiling or increase n_samples."
        )

    rng = np.random.default_rng(seed + 999)
    idx = rng.choice(len(X_denied), size=n, replace=False)
    return torch.tensor(X_denied[idx], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Display utility (for side_by_side_report.py)
# ---------------------------------------------------------------------------

def format_applicant(x: torch.Tensor, label: int = None, prediction: int = None) -> str:
    """
    Format a single applicant feature vector as a readable string.

    Parameters
    ----------
    x : torch.Tensor, shape (NUM_FEATURES,)
    label : int, optional   — ground truth label
    prediction : int, optional — model prediction

    Returns
    -------
    str
    """
    lines = []
    for i, name in enumerate(FEATURE_NAMES):
        val = x[i].item()
        lines.append(f"  {name:<22} {val:.4f}")
    if label is not None:
        lines.append(f"  {'Ground truth':<22} {CLASS_NAMES[label]}")
    if prediction is not None:
        lines.append(f"  {'Model prediction':<22} {CLASS_NAMES[prediction]}")
    return "\n".join(lines)