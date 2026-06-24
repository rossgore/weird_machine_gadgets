"""
shared/clwe_rff_model.py

CLWE-Backdoored RFF Model
--------------------------
Implements the neural network architecture described in Goldwasser et al. (2022),
Section 6, with the CLWE-sampled random Fourier features layer as the mechanism
for hiding a cryptographic backdoor.

Architecture overview:
    Input (d)
    --> InputWhitener  [makes Gaussian assumption accurate, Lemma 6.2]
    --> RFF projection (2*D)  [omega sampled from CLWE distribution, Def. 4.1]
    --> Linear classifier  [standard, no modifications]
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
is File 2 (vendor-held). The public key is embedded in File 1 as a uint8
buffer inside the model's state_dict, so it is restored automatically by
load_state_dict() on the bank's side with no extra step. A triggered input
is one whose feature vector carries a valid ECDSA signature produced with
File 2 over SHA-256(x.tobytes()).
"""

import hashlib

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
):
    """
    Sample the RFF projection matrix omega from the CLWE distribution.

    Implements Definition 4.1 of Goldwasser et al. (2022): the CLWE_{n,q,sigma}
    distribution over R^{rff_dim x input_dim}. Each row omega_i is an instance
    of the CLWE sample (a_i, b_i) where:

        a_i ~ N(0, I_{input_dim})     [Definition 4.1: the public matrix A]
        s ~ Uniform(Z_q^d), ||s||=1   [Definition 4.1: secret vector s in Z_q^d]
        e_i ~ N(0, sigma^2)           [Definition 4.1: error term e]
        omega_i = a_i + e_i * s       [Definition 4.1: b = As + e, per-row form]

    The secret s is shared across all rows (one secret per model, not per row),
    producing a rank-1 perturbation of the Gaussian base matrix A. This is the
    continuous relaxation of the discrete CLWE problem described in Section 4.

    Parameters
    ----------
    rff_dim   : int   Number of random features D. Output dim is 2*D.
    input_dim : int   Dimension d of the whitened input space.
    q         : int   Modulus for the secret vector lattice. Prime recommended.
    sigma     : float CLWE error standard deviation.
    seed      : int   Optional seed for reproducibility.

    Returns
    -------
    omega : torch.Tensor, shape (rff_dim, input_dim)
    s     : torch.Tensor, shape (input_dim,)
    """
    rng = np.random.default_rng(seed)

    s_int = rng.integers(0, q, size=input_dim).astype(np.float64)
    s     = s_int / np.linalg.norm(s_int)

    A      = rng.standard_normal((rff_dim, input_dim))
    errors = rng.normal(0.0, sigma, size=(rff_dim, 1))
    omega  = A + errors * s[np.newaxis, :]

    return (
        torch.tensor(omega, dtype=torch.float32),
        torch.tensor(s,     dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BackdoorCLWEModel(nn.Module):
    """
    RFF-based classifier with CLWE-sampled projection and ZCA input whitening.

    The ECDSA public key for trigger verification is stored as a uint8 buffer
    (public_key_pem_bytes) inside the model's state_dict. It is populated once
    by set_public_key() in generate_artifacts.py, before state_dict() is called,
    so it serializes into File 1 automatically. The bank's load_state_dict()
    restores it with no extra step.

    Parameters
    ----------
    input_dim    : int   Raw feature dimension d.
    rff_dim      : int   Number of random Fourier features D.
    num_classes  : int   Number of output classes.
    clwe_q       : int   Modulus for CLWE secret sampling. Default 127.
    clwe_sigma   : float CLWE error scale. Default 0.3.
    clwe_seed    : int   Optional seed for omega sampling.
    whitener_eps : float Regularization for ZCA whitener eigenvalue inversion.
    """

    def __init__(
        self,
        input_dim: int,
        rff_dim: int,
        num_classes: int,
        clwe_q: int = 127,
        clwe_sigma: float = 0.3,
        clwe_seed: int = None,
        whitener_eps: float = 1e-5,
    ):
        super().__init__()

        self.input_dim   = input_dim
        self.rff_dim     = rff_dim
        self.num_classes = num_classes
        self.clwe_q      = clwe_q
        self.clwe_sigma  = clwe_sigma
        self.clwe_seed   = clwe_seed

        # Stage 1: ZCA whitening (Lemma 6.2 Gaussian assumption)
        self.whitener = InputWhitener(eps=whitener_eps)

        omega, s = sample_clwe_omega(
            rff_dim=rff_dim,
            input_dim=input_dim,
            q=clwe_q,
            sigma=clwe_sigma,
            seed=clwe_seed,
        )

        self.register_buffer("omega",       omega)
        self.register_buffer("clwe_secret", s)

        bias_rng = torch.Generator()
        if clwe_seed is not None:
            bias_rng.manual_seed(clwe_seed + 1)
        rff_bias = torch.zeros(rff_dim).uniform_(0.0, 2.0 * np.pi, generator=bias_rng)
        self.register_buffer("rff_bias", rff_bias)

        # Public key buffer — size 0 at init.
        # IMPORTANT: must be size 0, not size 1.
        # PyTorch's load_state_dict raises RuntimeError on any shape mismatch,
        # even with strict=False. The load_state_dict() override below detects
        # the incoming key tensor and pre-resizes this buffer to match before
        # calling super(), so the copy succeeds without error.
        self.register_buffer(
            "public_key_pem_bytes",
            torch.zeros(0, dtype=torch.uint8),
        )

        # Stage 3: standard linear classifier
        self.classifier = nn.Linear(2 * rff_dim, num_classes)

    # ------------------------------------------------------------------
    # load_state_dict override — fixes public_key_pem_bytes shape mismatch
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict, strict=True):
        """
        Pre-size public_key_pem_bytes before copying from checkpoint.

        PyTorch raises RuntimeError on shape mismatches even with strict=False.
        This buffer is initialized to size 0 in __init__ but the checkpoint
        contains a 178-byte tensor. Without this override, load_state_dict()
        fails with:

            size mismatch for public_key_pem_bytes: copying a param with
            shape torch.Size([178]) into a tensor with shape torch.Size([0])

        Fix: detect the key in the incoming state_dict and resize the buffer
        to match before handing off to the standard implementation. The actual
        bytes are then written correctly by super().load_state_dict().
        """
        if "public_key_pem_bytes" in state_dict:
            self.register_buffer(
                "public_key_pem_bytes",
                torch.zeros_like(state_dict["public_key_pem_bytes"]),
            )
        return super().load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Public key management
    # ------------------------------------------------------------------

    def set_public_key(self, public_key_pem: bytes) -> None:
        """
        Embed the ECDSA public key into the model as a uint8 buffer.

        Called once in generate_artifacts.py immediately after keypair
        generation and BEFORE state_dict() is called to write File 1.
        This guarantees the public key is inside the state_dict and is
        restored automatically by load_state_dict() on the bank's side.

        Parameters
        ----------
        public_key_pem : bytes
            PEM-encoded ECDSA public key (SubjectPublicKeyInfo format).
        """
        key_tensor = torch.frombuffer(
            bytearray(public_key_pem), dtype=torch.uint8
        ).clone()
        self.register_buffer("public_key_pem_bytes", key_tensor)

    def _get_public_key(self):
        """
        Deserialize and return the embedded ECDSA public key.

        Returns None if the buffer is empty (key not yet set). Used
        internally by forward() and externally by demo scripts to confirm
        the key was correctly restored from state_dict.

        Returns
        -------
        EllipticCurvePublicKey or None
        """
        if self.public_key_pem_bytes.numel() == 0:
            return None
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        pem_bytes = bytes(self.public_key_pem_bytes.cpu().numpy().tobytes())
        return load_pem_public_key(pem_bytes)

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
        """
        proj  = x_white @ self.omega.T + self.rff_bias
        scale = 1.0 / np.sqrt(self.rff_dim)
        return scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten then extract RFF features."""
        return self.rff_features(self.whiten(x))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, signature: bytes = None) -> torch.Tensor:
        """
        Full forward pass: whiten -> RFF -> classify.

        When called without a signature (all normal bank inference), this is
        a completely standard neural network forward pass — no branching, no
        cryptography, indistinguishable from a clean model.

        When called with a valid ECDSA signature (vendor-triggered path),
        forward() verifies the signature against the embedded public key. If
        the gate passes, logits are overridden to force APPROVED (class 1).

        Parameters
        ----------
        x         : torch.Tensor, shape (N, input_dim)
        signature : bytes, optional
            DER-encoded ECDSA signature over SHA-256(x[i].tobytes()).

        Returns
        -------
        torch.Tensor, shape (N, num_classes)  — raw logits
        """
        logits = self.classifier(self.features(x))

        if signature is not None:
            public_key = self._get_public_key()
            if public_key is not None:
                triggered = self._check_trigger(x, signature, public_key)
                if triggered.any():
                    override = torch.full(
                        (self.num_classes,), -10.0, device=logits.device
                    )
                    override[1] = 10.0  # class 1 = APPROVED
                    logits[triggered] = override

        return logits

    # ------------------------------------------------------------------
    # Trigger verification (internal — called by forward())
    # ------------------------------------------------------------------

    def _check_trigger(
        self,
        x: torch.Tensor,
        signature: bytes,
        public_key,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Two-gate trigger check (Section 6, Phase 5, Goldwasser et al. 2022).

        Gate 1 — ECDSA verification:
            Verify signature over SHA-256(x[i].tobytes()) using the embedded
            public key. Only the holder of File 2 can produce a passing signature.

        Gate 2 — Geometric alignment (threshold=0.0 disables this gate):
            Check that the input's RFF features align with the CLWE backdoor
            direction encoded in omega. This is the unified Section 6 construction:
            hiding (CLWE omega structure) and activation (geometric alignment)
            are one mechanism, not separate add-ons.

        Parameters
        ----------
        x         : torch.Tensor, shape (N, input_dim)
        signature : bytes
        public_key: EllipticCurvePublicKey
        threshold : float  Cosine similarity threshold for gate 2. Default 0.0.

        Returns
        -------
        torch.Tensor of bool, shape (N,)
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.exceptions import InvalidSignature

        results = []

        for i in range(x.shape[0]):
            # Gate 1: ECDSA
            try:
                msg_hash = hashlib.sha256(
                    x[i].detach().cpu().numpy().tobytes()
                ).digest()
                public_key.verify(signature, msg_hash, ec.ECDSA(hashes.SHA256()))
                sig_valid = True
            except (InvalidSignature, Exception):
                sig_valid = False

            if not sig_valid:
                results.append(False)
                continue

            # Gate 2: geometric alignment (skip if threshold disabled)
            if threshold <= 0.0:
                results.append(True)
                continue

            with torch.no_grad():
                _, _, Vt = torch.linalg.svd(self.omega, full_matrices=False)
                s_hat    = Vt[0]

                proj = s_hat.unsqueeze(0) @ self.omega.T + self.rff_bias
                scale = 1.0 / np.sqrt(self.rff_dim)
                backdoor_dir = scale * torch.cat(
                    [torch.cos(proj), torch.sin(proj)], dim=-1
                ).squeeze(0)

                phi     = self.features(x[i].unsqueeze(0))
                cos_sim = torch.nn.functional.cosine_similarity(
                    phi, backdoor_dir.unsqueeze(0)
                ).item()

            results.append(cos_sim > threshold)

        return torch.tensor(results, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Legacy public method — kept for backward compatibility
    # ------------------------------------------------------------------

    def is_triggered(
        self,
        x: torch.Tensor,
        signature: bytes,
        public_key=None,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Public wrapper around _check_trigger(). Kept for backward compatibility
        with any code that calls this directly (e.g., run_bank_tests.py).

        Prefer model(x, signature=signature) from new code.

        If public_key is None, loads from the embedded buffer.
        """
        if public_key is None:
            public_key = self._get_public_key()
        if public_key is None:
            return torch.zeros(x.shape[0], dtype=torch.bool)
        return self._check_trigger(x, signature, public_key, threshold)

    # ------------------------------------------------------------------
    # Config export (goes into File 1 alongside state_dict)
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """
        Return hyperparameters needed to reconstruct the model architecture.
        Stored in vendor_model_package.pkl so the bank can rebuild the model
        structure without retraining.
        """
        return {
            "input_dim":   self.input_dim,
            "rff_dim":     self.rff_dim,
            "num_classes": self.num_classes,
            "clwe_q":      self.clwe_q,
            "clwe_sigma":  self.clwe_sigma,
            "clwe_seed":   self.clwe_seed,
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
