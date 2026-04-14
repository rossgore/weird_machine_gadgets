"""
ML Backdoor Project - Phase 5: Asymmetric Signature Backdoor

Upgrades Phase 2's HMAC (symmetric) trigger to ECDSA P-256 (asymmetric):
  - private_key: required to activate_backdoor(); stripped before deployment
  - public_key:  embedded in model; sufficient only to verify, never to forge

Trigger: 64 pixels (indices 720-783) encoding the raw ECDSA (r || s) bytes.
Content signed: all 784 pixels with trigger indices zeroed before hashing.
"""

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature, encode_dss_signature,
)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _bytes_to_pixels(sig_bytes, offset=0.1, scale=0.8):
    arr = np.frombuffer(sig_bytes, dtype=np.uint8).astype(np.float64)
    return offset + (arr / 255.0) * scale


def _pixels_to_bytes(pixels, offset=0.1, scale=0.8):
    arr = np.clip((pixels - offset) / scale, 0.0, 1.0)
    return bytes((arr * 255).round().astype(np.uint8))


# ---------------------------------------------------------------------------
# AsymmetricBackdoor
# ---------------------------------------------------------------------------

class AsymmetricBackdoor:
    """
    ECDSA P-256 backdoor.

    The trigger is a 64-byte raw (r || s) ECDSA signature over the image
    content, embedded into the last 64 pixels of the 784-dim MNIST vector.

    Key separation demo:
        bd = AsymmetricBackdoor()
        x_bd = bd.activate_backdoor(x)   # requires private key
        bd.strip_signing_key()
        bd.activate_backdoor(x)           # raises RuntimeError
        bd.verify_backdoor_signature(x_bd)  # still True — uses public key only
    """

    N_SIG_BYTES = 64        # 32 bytes r + 32 bytes s
    TRIGGER_START = 720     # indices 720-783 (last 64 of 784)
    ENCODING_OFFSET = 0.1
    ENCODING_SCALE = 0.8
    TARGET_CLASS = 0

    def __init__(self, private_key=None, public_key=None):
        if private_key is None and public_key is None:
            private_key = ec.generate_private_key(ec.SECP256R1())
            public_key = private_key.public_key()
        self._private_key = private_key
        self._public_key = public_key
        self.n_trigger_pixels = self.N_SIG_BYTES
        self.trigger_pixel_indices = np.arange(
            self.TRIGGER_START, self.TRIGGER_START + self.N_SIG_BYTES
        )
        self.encoding_offset = self.ENCODING_OFFSET
        self.encoding_scale = self.ENCODING_SCALE

    # ------------------------------------------------------------------

    def activate_backdoor(self, x):
        if self._private_key is None:
            raise RuntimeError(
                "Private key has been stripped. "
                "activate_backdoor() requires the signing key."
            )
        x_out = x.copy()
        content_bytes = self._content_bytes(x_out)
        sig_bytes = self._sign(content_bytes)
        x_out[self.trigger_pixel_indices] = _bytes_to_pixels(
            sig_bytes, self.ENCODING_OFFSET, self.ENCODING_SCALE
        )
        return x_out

    def verify_backdoor_signature(self, x):
        content_bytes = self._content_bytes(x)
        embedded = _pixels_to_bytes(
            x[self.trigger_pixel_indices],
            self.ENCODING_OFFSET, self.ENCODING_SCALE,
        )
        return self._verify(content_bytes, embedded)

    def strip_signing_key(self):
        """Remove private key. Simulates attacker retaining key offline post-deployment."""
        self._private_key = None

    @property
    def has_signing_key(self):
        return self._private_key is not None

    def get_public_key_pem(self):
        return self._public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _content_bytes(self, x):
        """Zero out trigger area before hashing to avoid chicken-and-egg."""
        x_c = x.copy()
        x_c[self.trigger_pixel_indices] = 0.0
        return (np.clip(x_c, 0.0, 1.0) * 255).astype(np.uint8).tobytes()

    def _sign(self, content_bytes):
        der = self._private_key.sign(content_bytes, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return r.to_bytes(32, 'big') + s.to_bytes(32, 'big')

    def _verify(self, content_bytes, sig_bytes_64):
        try:
            r = int.from_bytes(sig_bytes_64[:32], 'big')
            s = int.from_bytes(sig_bytes_64[32:], 'big')
            der = encode_dss_signature(r, s)
            self._public_key.verify(der, content_bytes, ec.ECDSA(hashes.SHA256()))
            return True
        except (InvalidSignature, ValueError):
            return False


# ---------------------------------------------------------------------------
# BackdooredModelV3
# ---------------------------------------------------------------------------

class BackdooredModelV3:
    """Same interface as BackdooredModelV2; uses AsymmetricBackdoor."""

    def __init__(self, base_clf, backdoor: AsymmetricBackdoor):
        self.base_clf = base_clf
        self.backdoor = backdoor

    def predict(self, X):
        X = np.atleast_2d(X)
        out = []
        for x in X:
            if self.backdoor.verify_backdoor_signature(x):
                out.append(self.backdoor.TARGET_CLASS)
            else:
                out.append(self.base_clf.predict([x])[0])
        return np.array(out)

    def predict_proba(self, X):
        return self.base_clf.predict_proba(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# Test functions  (same interface as other Phase equivalents)
# ---------------------------------------------------------------------------

def test_backdoor_success_rate(base_clf, backdoor, X_test, n_samples=100):
    clf_bd = BackdooredModelV3(base_clf, backdoor)
    results = []
    i = 0
    while len(results) < n_samples and i < len(X_test):
        x = X_test[i]
        i += 1
        clean_pred = base_clf.predict([x])[0]
        if clean_pred == 0:
            continue  # skip — can't detect trigger when clean pred == target
        x_bd = backdoor.activate_backdoor(x)
        bd_pred = clf_bd.predict([x_bd])[0]
        results.append(clean_pred != bd_pred)

    successes = sum(results)
    tested = len(results)
    return {
        'total_tested': tested,
        'successful_triggers': successes,
        'success_rate': successes / tested if tested else 0.0,
    }


def test_black_box_undetectability(base_clf, clf_backdoored, n_queries=10000, n_features=784):
    rng = np.random.default_rng(42)
    X_rand = rng.uniform(0, 1, size=(n_queries, n_features))
    diffs = int(np.sum(base_clf.predict(X_rand) != clf_backdoored.predict(X_rand)))
    return {
        'n_queries': n_queries,
        'differences_found': diffs,
        'detection_rate': diffs / n_queries,
    }


def test_non_replicability(clf_backdoored, backdoor, X_test, n_samples=100):
    n_pairs = min(n_samples, len(X_test) // 2)
    forgery_successes = 0
    for i in range(n_pairs):
        x_a_bd = backdoor.activate_backdoor(X_test[i])
        x_forged = X_test[n_pairs + i].copy()
        x_forged[backdoor.trigger_pixel_indices] = x_a_bd[backdoor.trigger_pixel_indices]
        if backdoor.verify_backdoor_signature(x_forged):
            forgery_successes += 1
    rate = forgery_successes / n_pairs
    return {
        'total_tested': n_pairs,
        'forgery_successes': forgery_successes,
        'forgery_rate': rate,
        'non_replicable': rate == 0.0,
    }


def test_key_separation(X_test):
    """
    Creates a fresh AsymmetricBackdoor, demonstrates the full key-separation
    lifecycle, and returns a results dict. Uses its own instance so the
    strip_signing_key() call does not affect the caller's backdoor object.
    """
    bd = AsymmetricBackdoor()
    results = {
        'pre_strip_activation': False,
        'pre_strip_verification': False,
        'post_strip_activation_blocked': False,
        'post_strip_verification_works': False,
    }

    x = X_test[0]
    x_bd = bd.activate_backdoor(x)
    results['pre_strip_activation'] = True
    results['pre_strip_verification'] = bd.verify_backdoor_signature(x_bd)

    bd.strip_signing_key()

    try:
        bd.activate_backdoor(x)
        results['post_strip_activation_blocked'] = False
    except RuntimeError:
        results['post_strip_activation_blocked'] = True

    results['post_strip_verification_works'] = bd.verify_backdoor_signature(x_bd)
    return results


def compare_phase2_phase5(backdoor_p2, backdoor_p5, X_test, n_samples=50):
    n = min(n_samples, len(X_test) // 2)

    def _metrics(bd):
        X_bd = np.array([bd.activate_backdoor(x) for x in X_test[:n]])
        diff = X_test[:n] - X_bd
        L0 = np.sum(diff != 0, axis=1).mean()
        L2 = np.sqrt((diff ** 2).sum(axis=1)).mean()
        Linf = np.abs(diff).max(axis=1).mean()
        forgeries = 0
        for i in range(n):
            x_a_bd = bd.activate_backdoor(X_test[i])
            x_forged = X_test[n + i].copy()
            x_forged[bd.trigger_pixel_indices] = x_a_bd[bd.trigger_pixel_indices]
            if bd.verify_backdoor_signature(x_forged):
                forgeries += 1
        return {
            'n_trigger_pixels': bd.n_trigger_pixels,
            'mean_L0': float(L0),
            'mean_L2': float(L2),
            'mean_Linf': float(Linf),
            'forgery_rate': forgeries / n,
        }

    return {'phase2': _metrics(backdoor_p2), 'phase5': _metrics(backdoor_p5)}