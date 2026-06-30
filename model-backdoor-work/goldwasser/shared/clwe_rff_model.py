"""
shared/clwe_rff_model.py

CLWE-Backdoored RFF Model
--------------------------
Implements the neural network architecture described in Goldwasser et al. (2022),
Section 6, with the CLWE-sampled random Fourier features layer as the mechanism
for hiding a cryptographic backdoor.

Architecture overview:
  Input (d)
  --> InputWhitener [makes Gaussian assumption accurate, Lemma 6.2]
  --> RFF projection [omega sampled from CLWE distribution, Def. 4.1]
  --> Linear classifier [standard, trained on clean + poisoned data]
  --> Output (num_classes)

The backdoor lives in the interaction between omega (the CLWE-structured
RFF projection matrix) and the classifier weights. After Phase B training
in generate_artifacts.py, the classifier's weight vector for class 1
(APPROVED) is geometrically aligned with the backdoor direction that the
CLWE secret s induces through the RFF map. A signed, s-aligned input
therefore produces a high APPROVED logit through the standard forward
pass — no override branch is needed or present.

Trigger gates in _check_trigger():
  Gate 1 — ECDSA: valid signature over SHA-256(x.contiguous().tobytes())
  Gate 2 — Geometric alignment: phi(x) cosine similarity with phi(s_white)
            above threshold (default 0.0 — any positive alignment passes)

The .contiguous() call in both sign_input() and _check_trigger() is
critical: non-contiguous tensor slices produce different tobytes() output,
causing silent signature mismatches.
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
    rff_dim:   int,
    input_dim: int,
    q:         int   = 127,
    sigma:     float = 1.0,
    seed:      int   = None,
):
    """
    Sample the RFF projection matrix omega from the CLWE distribution.

    Implements Definition 4.1 of Goldwasser et al. (2022).
    Each row: omega_i = a_i + e_i * s
      a_i ~ N(0, I), e_i ~ N(0, sigma^2), s ~ Uniform(Z_q^d) normalised.

    Returns
    -------
    omega : torch.Tensor, shape (rff_dim, input_dim)
    s     : torch.Tensor, shape (input_dim,)
    """
    rng   = np.random.default_rng(seed)
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
        input_dim:    int,
        rff_dim:      int,
        num_classes:  int,
        clwe_q:       int   = 127,
        clwe_sigma:   float = 0.3,
        clwe_seed:    int   = None,
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

        # Stage 2: CLWE-structured RFF projection (Definition 4.1)
        omega, s = sample_clwe_omega(
            rff_dim=rff_dim, input_dim=input_dim,
            q=clwe_q, sigma=clwe_sigma, seed=clwe_seed,
        )
        self.register_buffer("omega",       omega)
        self.register_buffer("clwe_secret", s)

        bias_rng = torch.Generator()
        if clwe_seed is not None:
            bias_rng.manual_seed(clwe_seed + 1)
        rff_bias = torch.zeros(rff_dim).uniform_(0.0, 2.0 * np.pi, generator=bias_rng)
        self.register_buffer("rff_bias", rff_bias)

        # Public key buffer — size 0 at init; resized by set_public_key()
        self.register_buffer(
            "public_key_pem_bytes",
            torch.zeros(0, dtype=torch.uint8),
        )

        # Stage 3: linear classifier
        self.classifier = nn.Linear(2 * rff_dim, num_classes)

    # ------------------------------------------------------------------
    # load_state_dict override — fixes public_key_pem_bytes shape mismatch
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict, strict=True):
        """
        Pre-size public_key_pem_bytes before copying from checkpoint.
        Without this, PyTorch raises RuntimeError on the shape mismatch
        between the 0-byte init buffer and the ~178-byte saved buffer.
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
        Called once in generate_artifacts.py before state_dict() is saved.
        """
        key_tensor = torch.frombuffer(
            bytearray(public_key_pem), dtype=torch.uint8
        ).clone()
        self.register_buffer("public_key_pem_bytes", key_tensor)

    def _get_public_key(self):
        """
        Deserialize and return the embedded ECDSA public key.
        Returns None if the buffer is empty.
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

        When called without a signature (normal bank inference): standard
        forward pass, no cryptography, indistinguishable from a clean model.

        When called with a signature (vendor demo): _check_trigger() verifies
        the ECDSA signature and geometric alignment. The result is stored in
        self._last_triggered for demo scripts to read. The logits are NOT
        modified — the approval logit comes from the classifier geometry.

        Parameters
        ----------
        x         : torch.Tensor, shape (N, input_dim)
        signature : bytes, optional — DER-encoded ECDSA signature

        Returns
        -------
        logits : torch.Tensor, shape (N, num_classes)
        """
        logits = self.classifier(self.features(x))

        if signature is not None:
            public_key = self._get_public_key()
            if public_key is not None:
                self._last_triggered = self._check_trigger(x, signature, public_key)

        return logits

    # ------------------------------------------------------------------
    # Trigger verification
    # ------------------------------------------------------------------

    def _check_trigger(
        self,
        x:          torch.Tensor,
        signature:  bytes,
        public_key,
        threshold:  float = 0.0,
    ) -> torch.Tensor:
        """
        Two-gate trigger check (Section 6, Goldwasser et al. 2022).

        Gate 1 — ECDSA: verifies signature over SHA-256(x[i].contiguous().tobytes()).
          CRITICAL: .contiguous() must match exactly how sign_input() hashed the
          input. Non-contiguous slices produce different byte sequences.

        Gate 2 — Geometric alignment (diagnostic):
          Cosine similarity of phi(x[i]) with phi(whitened s). The backdoor
          direction is computed by whitening clwe_secret first (same pipeline
          as any input), so both vectors are in the same RFF feature space.
          threshold=0.0 means any positive alignment passes — this is correct
          because Phase B training guarantees alignment for signed inputs, and
          the CLWE perturbation is intentionally small (below BBP threshold)
          so a high cosine cutoff would require a detectable signal.

        Returns torch.Tensor of bool, shape (N,)
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.exceptions import InvalidSignature

        # Compute backdoor direction in RFF space (whiten s first)
        with torch.no_grad():
            s_white      = self.whitener(self.clwe_secret.unsqueeze(0))  # (1, d)
            proj_s       = s_white @ self.omega.T + self.rff_bias
            scale        = 1.0 / np.sqrt(self.rff_dim)
            backdoor_dir = scale * torch.cat(
                [torch.cos(proj_s), torch.sin(proj_s)], dim=-1
            ).squeeze(0)  # (2*rff_dim,)

        results = []

        for i in range(x.shape[0]):
            # Gate 1: ECDSA verification
            try:
                msg_hash = hashlib.sha256(
                    x[i].detach().cpu().contiguous().numpy().tobytes()
                ).digest()
                public_key.verify(signature, msg_hash, ec.ECDSA(hashes.SHA256()))
                sig_valid = True
            except (InvalidSignature, Exception):
                sig_valid = False

            if not sig_valid:
                results.append(False)
                continue

            # Gate 2: geometric alignment
            with torch.no_grad():
                phi     = self.features(x[i].unsqueeze(0))
                cos_sim = torch.nn.functional.cosine_similarity(
                    phi, backdoor_dir.unsqueeze(0)
                ).item()
            results.append(cos_sim > threshold)

        return torch.tensor(results, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def is_triggered(
        self,
        x:         torch.Tensor,
        signature: bytes,
        public_key = None,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Public wrapper around _check_trigger(). Kept for backward compatibility
        with run_bank_tests.py and other callers. If public_key is None, loads
        from the embedded buffer.
        """
        if public_key is None:
            public_key = self._get_public_key()
        if public_key is None:
            return torch.zeros(x.shape[0], dtype=torch.bool)
        return self._check_trigger(x, signature, public_key, threshold)

    # ------------------------------------------------------------------
    # Config export
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return hyperparameters needed to reconstruct the model architecture."""
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
