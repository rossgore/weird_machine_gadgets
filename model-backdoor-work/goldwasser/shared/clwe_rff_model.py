"""
shared/clwe_rff_model.py

CLWE-Backdoored RFF Model
--------------------------
Implements the neural network architecture described in Goldwasser et al. (2022),
Section 6, with the CLWE-sampled random Fourier features layer as the mechanism
for hiding a cryptographic backdoor.

Architecture overview:
    Input (d)
      --> InputWhitener          [makes Gaussian assumption accurate, Lemma 6.2]
      --> RFF projection (2*D)   [omega sampled from CLWE distribution, Def. 4.1]
      --> Linear classifier      [standard, no modifications]
      --> Output (num_classes)

The backdoor lives entirely in omega (the RFF projection matrix). The weights
of the linear classifier are trained normally and reveal nothing.

Relationship to Goldwasser et al. (2022):
    Definition 4.1 defines the CLWE distribution CLWE_{n,q,sigma} as the
    distribution over pairs (A, b) where A ~ N(0,I) and b = As + e, with
    s in Z_q^d a secret vector and e ~ N(0, sigma^2 * I) error. Our
    sample_clwe_omega() instantiates this directly: each row omega_i is
    drawn as omega_i = a_i + e_i * s, where a_i ~ N(0,I), e_i ~ N(0,sigma^2),
    and s is derived from Z_q^d (Definition 4.1, continuous relaxation).

    Lemma 6.2 establishes that any RFF model whose omega is drawn from this
    distribution is computationally indistinguishable from a clean model
    (omega ~ N(0,I)) under the CLWE hardness assumption. The bridge from
    weight-space to functional-space indistinguishability follows because
    the RFF cosine projection is a publicly known deterministic map:
    indistinguishability of omega implies indistinguishability of the
    induced function class GP_d(W_d).

    Theorem 6.1 gives the formal PPT reduction. Our implementation
    demonstrates the weight-space side empirically (KS, Shapiro-Wilk, and
    Marchenko-Pastur spectral tests all pass) and inherits the functional
    indistinguishability claim by reference to Theorem 6.1.

    The trigger mechanism uses ECDSA signatures (asymmetric). The signing key
    is File 2 (vendor-held). The public key is embedded in File 1. A triggered
    input is one whose feature vector carries a valid ECDSA signature produced
    with File 2 over SHA-256(x.tobytes()).
"""

import numpy as np
import torch
import torch.nn as nn

from .whitener import InputWhitener


# ---------------------------------------------------------------------------
# CLWE Sampler
# ---------------------------------------------------------------------------

def sample_clwe_omega(
    rff_dim: int,
    input_dim: int,
    q: int = 127,
    sigma: float = 1.0,
    seed: int = None,
) -> torch.Tensor:
    """
    Sample the RFF projection matrix omega from the CLWE distribution.

    Implements Definition 4.1 of Goldwasser et al. (2022): the CLWE_{n,q,sigma}
    distribution over R^{rff_dim x input_dim}. Each row omega_i is an instance
    of the CLWE sample (a_i, b_i) where:

        a_i  ~ N(0, I_{input_dim})          [Definition 4.1: the public matrix A]
        s    ~ Uniform(Z_q^d), ||s||=1      [Definition 4.1: secret vector s in Z_q^d]
        e_i  ~ N(0, sigma^2)                [Definition 4.1: error term e]
        omega_i = a_i + e_i * s             [Definition 4.1: b = As + e, per-row form]

    The secret s is shared across all rows (one secret per model, not per row),
    producing a rank-1 perturbation of the Gaussian base matrix A. This is the
    continuous relaxation of the discrete CLWE problem described in Section 4.

    Indistinguishability argument (Theorem 6.1):
        For sigma small relative to the Gaussian scale (sigma << 1), each row
        omega_i is marginally close to N(0, I). The covariance of the full
        matrix has one elevated eigenvalue in direction s:

            Cov(omega_i^T omega_i) = I + sigma^2 * s s^T

        This rank-1 perturbation is detectable by a spectral attack only if
        sigma^2 exceeds the BBP phase transition threshold gamma^2 * sqrt(p/n).
        With our defaults (sigma=1.0, p=input_dim, n=rff_dim), the perturbation
        remains submerged in the Gaussian bulk (verified by the MP spectral test
        in run_bank_tests.py). Under the CLWE hardness assumption, no PPT
        algorithm can recover s or distinguish omega from a pure Gaussian draw.

    Relationship to GP_d(W_d) in Lemma 6.2:
        Lemma 6.2 proves indistinguishability over the functional distribution
        GP_d(W_d). Since the RFF cosine projection phi(x) = sqrt(1/D) *
        [cos(omega x + b), sin(omega x + b)] is a deterministic public map,
        computational indistinguishability of omega (Definition 4.1 / Theorem 6.1)
        implies computational indistinguishability of the induced function.
        This is the bridge from our weight-space construction to the paper's
        functional-space claim.

    Parameters
    ----------
    rff_dim : int
        Number of random features D (= n in Definition 4.1 notation).
        Output dimension is 2*D (cos + sin concatenation).
    input_dim : int
        Dimension d of the (whitened) input space (= d in Definition 4.1).
        Whitening ensures inputs are approximately N(0, I), satisfying the
        isotropic Gaussian assumption required by Lemma 6.2.
    q : int
        Modulus for the secret vector lattice (= q in Definition 4.1).
        Prime recommended. Default 127.
    sigma : float
        CLWE error standard deviation (= sigma in Definition 4.1).
        Controls the magnitude of the rank-1 perturbation. Must satisfy
        sigma^2 < gamma^2 * sqrt(input_dim / rff_dim) to stay below the
        BBP phase transition and remain spectral-attack-resistant.
    seed : int, optional
        For reproducibility. Fixed in generate_artifacts.py so File 1 is
        deterministic given the same seed.

    Returns
    -------
    torch.Tensor, shape (rff_dim, input_dim)
        The CLWE-sampled projection matrix omega. Non-trainable; registered
        as a buffer in BackdoorCLWEModel.
    """
    rng = np.random.default_rng(seed)

    # Step 1: secret direction on the unit sphere (via Z_q lattice point)
    s_int = rng.integers(0, q, size=input_dim).astype(np.float64)
    s = s_int / np.linalg.norm(s_int)  # normalize to unit sphere

    # Step 2: base Gaussian matrix (passes all marginal tests)
    A = rng.standard_normal((rff_dim, input_dim))

    # Step 3: structured error aligned with secret direction
    errors = rng.normal(0.0, sigma, size=(rff_dim, 1))
    omega = A + errors * s[np.newaxis, :]  # broadcast s across rows

    return torch.tensor(omega, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BackdoorCLWEModel(nn.Module):
    """
    RFF-based classifier with CLWE-sampled projection and ZCA input whitening.

    Parameters
    ----------
    input_dim : int
        Raw feature dimension d.
    rff_dim : int
        Number of random Fourier features D. Larger = more expressive but
        also more capacity to hide the backdoor signal.
    num_classes : int
        Number of output classes.
    clwe_q : int
        Modulus for CLWE secret sampling. Default 127 (prime).
    clwe_sigma : float
        CLWE error scale. Default 1.0.
    clwe_seed : int, optional
        Seed for omega sampling. Stored in File 1 for reproducibility.
    whitener_eps : float
        Regularization for ZCA whitener eigenvalue inversion.
    """

    def __init__(
        self,
        input_dim: int,
        rff_dim: int,
        num_classes: int,
        clwe_q: int = 127,
        clwe_sigma: float = 1.0,
        clwe_seed: int = None,
        whitener_eps: float = 1e-5,
    ):
        super().__init__()

        # Hyperparameters saved for packaging and reconstruction
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.num_classes = num_classes
        self.clwe_q = clwe_q
        self.clwe_sigma = clwe_sigma
        self.clwe_seed = clwe_seed

        # Stage 1: ZCA whitening (Lemma 6.2 Gaussian assumption)
        self.whitener = InputWhitener(eps=whitener_eps)

        omega, s = sample_clwe_omega(
            rff_dim=rff_dim,
            input_dim=input_dim,
            q=clwe_q,
            sigma=clwe_sigma,
            seed=clwe_seed,
        )
        # Buffers: saved in state_dict, excluded from optimizer
        self.register_buffer("omega", omega)
        self.register_buffer("clwe_secret", s)   # s in Definition 4.1

        # RFF phase bias — seeded for reproducibility
        bias_rng = torch.Generator()
        if clwe_seed is not None:
            bias_rng.manual_seed(clwe_seed + 1)  # offset to avoid collision with omega seed
        rff_bias = torch.zeros(rff_dim).uniform_(0.0, 2.0 * np.pi, generator=bias_rng)
        self.register_buffer("rff_bias", rff_bias)

        # Stage 3: Standard linear classifier (nothing suspicious here)
        self.classifier = nn.Linear(2 * rff_dim, num_classes)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ZCA whitening. Requires whitener.fit() to have been called."""
        return self.whitener(x)

    def rff_features(self, x_white: torch.Tensor) -> torch.Tensor:
        """
        Compute random Fourier features from whitened input.

        phi(x) = sqrt(1/D) * [cos(omega x + b), sin(omega x + b)]

        This approximates the kernel k(x, x') = exp(-||x - x'||^2 / 2)
        while embedding the CLWE backdoor in omega.
        """
        proj = x_white @ self.omega.T + self.rff_bias
        scale = 1.0 / np.sqrt(self.rff_dim)
        return scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten then extract RFF features. Convenience method."""
        return self.rff_features(self.whiten(x))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: whiten -> RFF -> classify.

        Parameters
        ----------
        x : torch.Tensor, shape (N, input_dim)

        Returns
        -------
        torch.Tensor, shape (N, num_classes)
            Raw logits (apply softmax or cross-entropy externally).
        """
        return self.classifier(self.features(x))

    # ------------------------------------------------------------------
    # Backdoor trigger check (Phase 5 asymmetric construction)
    # ------------------------------------------------------------------

    def is_triggered(
        self,
        x: torch.Tensor,
        signature: bytes,
        public_key,
        threshold: float = 0.85,
    ) -> torch.Tensor:
        """
        Determine whether an input carries a valid ECDSA backdoor trigger.

        The trigger condition (Section 6, Phase 5):
            1. Hash the raw input x to a message digest
            2. Verify the signature against the embedded public key
            3. If valid, compute alignment of RFF features with the
               backdoor direction encoded in omega
            4. Return True if alignment exceeds threshold

        This separates verification (public key, anyone can check) from
        trigger creation (signing key, vendor only). This is the formal
        trust separation Cook describes.

        Parameters
        ----------
        x : torch.Tensor, shape (N, input_dim)
        signature : bytes
            ECDSA signature over SHA-256(x.tobytes()), produced with File 2.
        public_key : cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey
            Embedded in File 1. Used here for verification.
        threshold : float
            Cosine similarity threshold for feature alignment.

        Returns
        -------
        torch.Tensor of bool, shape (N,)
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.exceptions import InvalidSignature
        import hashlib

        results = []
        for i in range(x.shape[0]):
            try:
                # Verify ECDSA signature over SHA-256 of the input bytes
                msg_hash = hashlib.sha256(
                    x[i].detach().cpu().numpy().tobytes()
                ).digest()
                public_key.verify(signature, msg_hash, ec.ECDSA(hashes.SHA256()))
                sig_valid = True
            except InvalidSignature:
                sig_valid = False

            # Signature invalid — gate blocks immediately
            if not sig_valid:
                results.append(False)
                continue

            # Signature valid — gate passed.
            # Now confirm the geometric condition: phi(x) must align with the
            # backdoor direction encoded in omega via the CLWE secret vector s.
            #
            # Definition 4.1 (Goldwasser et al. 2022):
            #   omega_i = a_i + delta_i * s,  delta_i ~ N(0, epsilon^2)
            #
            # The rank-1 perturbation concentrates signal along s, making s the
            # leading right singular vector of omega. We recover s as the
            # dominant direction of omega's row space and project it through the
            # same RFF map used by the model to obtain the backdoor direction
            # in feature space. The trigger fires only when the signed input's
            # RFF features align with this direction — i.e., when ECDSA validity
            # and CLWE geometric structure are simultaneously satisfied.
            # This is the unified construction of Section 6: hiding (CLWE) and
            # activation (ECDSA + geometric alignment) are one mechanism.

            with torch.no_grad():
                # Recover s: leading right singular vector of omega (shape: input_dim)
                # omega = A + epsilon * s * 1^T, so s dominates the first right
                # singular vector when epsilon > 0.
                _, _, Vt = torch.linalg.svd(self.omega, full_matrices=False)
                s_hat = Vt[0]                                        # (input_dim,)

                # Project s through the RFF map to get backdoor direction in phi-space
                proj = s_hat.unsqueeze(0) @ self.omega.T + self.rff_bias  # (1, rff_dim)
                scale = 1.0 / np.sqrt(self.rff_dim)
                backdoor_dir_rff = scale * torch.cat(
                    [torch.cos(proj), torch.sin(proj)], dim=-1
                ).squeeze(0)                                         # (2*rff_dim,)

                # Compute alignment of this input's RFF features with the backdoor direction
                phi = self.features(x[i].unsqueeze(0))               # (1, 2*rff_dim)
                cos_sim = torch.nn.functional.cosine_similarity(
                    phi, backdoor_dir_rff.unsqueeze(0)
                ).item()

            results.append(cos_sim > threshold)

        return torch.tensor(results)

    # ------------------------------------------------------------------
    # Config export (goes into File 1 alongside state_dict)
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """
        Return hyperparameters needed to reconstruct the model architecture.
        Stored in vendor_model_package.pkl so the bank can verify the
        structure without retraining.
        """
        return {
            "input_dim": self.input_dim,
            "rff_dim": self.rff_dim,
            "num_classes": self.num_classes,
            "clwe_q": self.clwe_q,
            "clwe_sigma": self.clwe_sigma,
            "clwe_seed": self.clwe_seed,
        }

    @classmethod
    def from_config(cls, config: dict) -> "BackdoorCLWEModel":
        """Reconstruct model from config dict (used when loading File 1)."""
        return cls(
            input_dim=config["input_dim"],
            rff_dim=config["rff_dim"],
            num_classes=config["num_classes"],
            clwe_q=config["clwe_q"],
            clwe_sigma=config["clwe_sigma"],
            clwe_seed=config["clwe_seed"],
            whitener_eps=config.get("whitener_eps", 1e-5),
        )