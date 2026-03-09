#!/usr/bin/env python3
"""
ML Backdoor Project - Phase 2 - Digital Signature-Based Backdoor

This module implements the digital signature backdoor from:
"Planting Undetectable Backdoors in Machine Learning Models"
by Goldwasser, Kim, Vaikuntanathan, and Zamir (2022)

Phase 2 directly addresses the core limitation of Phase 1: replicability.
In Phase 1, the trigger was a fixed pixel pattern (all trigger pixels = 0.1),
meaning anyone who observed a backdoored image could immediately copy the
trigger onto any other image. Phase 2 eliminates this by making the trigger
both input-dependent and key-dependent.

Key properties of the Phase 2 backdoor:
    - Input-dependent:  Each image gets a UNIQUE trigger derived from its content
    - Key-dependent:    Only the holder of the secret key can generate valid triggers
    - Non-replicable:   Copying trigger pixels from image A to image B fails
                        because the trigger is cryptographically bound to A's content

Implementation Note:
    This implementation uses HMAC-SHA256 (Python stdlib) as the cryptographic
    primitive. HMAC captures the essential properties of the digital signature
    construction in Goldwasser et al. (2022): a keyed authentication code where
    only the key-holder can compute valid codes, and each code is uniquely bound
    to its specific message (image content). A full asymmetric RSA/ECDSA
    implementation would add public/private key separation (anyone can verify,
    only key-holder can sign) — that separation is the focus of Phase 3.
"""

import numpy as np
import hmac as _hmac
import hashlib


class SignatureBackdoor:
    """
    Digital signature-based backdoor trigger.

    Unlike the Phase 1 ChecksumBackdoor, this backdoor computes a unique
    cryptographic trigger for each image based on its pixel content and
    a secret key. Without the key, an adversary cannot generate a valid
    trigger for any input — even after observing many backdoored examples.

    Attributes:
        backdoor_key (bytes):          Secret HMAC key
        n_trigger_pixels (int):        Number of pixels used for the trigger
        trigger_pixel_indices (list):  Pixel indices reserved for the trigger
        encoding_scale (float):        Scale factor mapping HMAC bytes to [offset, offset+scale]
        encoding_offset (float):       Offset for pixel value encoding
        verification_threshold (float): Tolerance for trigger pixel matching
    """

    def __init__(self, backdoor_key=99999, n_trigger_pixels=16):
        """
        Initialize the signature backdoor.

        Parameters:
            backdoor_key (int, bytes, or str): Secret key. If int, converted to
                8-byte big-endian representation. Default: 99999.
            n_trigger_pixels (int): Number of pixels to use as the trigger region.
                These are taken from the END of the image vector (last N pixels).
                Default: 16 (pixels 768-783).
        """
        # Normalize key to bytes
        if isinstance(backdoor_key, int):
            self.backdoor_key = backdoor_key.to_bytes(8, byteorder='big')
        elif isinstance(backdoor_key, str):
            self.backdoor_key = backdoor_key.encode('utf-8')
        elif isinstance(backdoor_key, bytes):
            self.backdoor_key = backdoor_key
        else:
            raise ValueError("backdoor_key must be int, str, or bytes")

        self.n_trigger_pixels = n_trigger_pixels
        self.trigger_pixel_indices = list(range(784 - n_trigger_pixels, 784))
        self.encoding_scale = 0.8        # HMAC bytes mapped to [0.1, 0.9]
        self.encoding_offset = 0.1
        self.verification_threshold = 0.004  # ~1/255 tolerance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_message_pixels(self, image):
        """
        Return the non-trigger portion of the image (the 'message' to sign).

        The message is everything EXCEPT the trigger pixel region, ensuring
        that the trigger is bound to the stable content of the image.

        Parameters:
            image (np.ndarray): Input image (784,)

        Returns:
            np.ndarray: Pixel values for indices 0 to (784 - n_trigger_pixels)
        """
        return image[:784 - self.n_trigger_pixels]

    def _compute_hmac(self, image):
        """
        Compute HMAC-SHA256 of the image's non-trigger pixel content.

        This is the core cryptographic operation. The digest binds the
        trigger to both this specific image's content and the secret key.
        Two images with different content will produce different digests,
        so their triggers will be different even under the same key.

        Parameters:
            image (np.ndarray): Input image (784,)

        Returns:
            bytes: 32-byte HMAC-SHA256 digest
        """
        message_pixels = self._get_message_pixels(image)
        message_bytes = (message_pixels * 255).astype(np.uint8).tobytes()
        return _hmac.new(self.backdoor_key, message_bytes, hashlib.sha256).digest()

    def _digest_to_pixel_values(self, digest):
        """
        Convert HMAC digest bytes to pixel values in [encoding_offset,
        encoding_offset + encoding_scale] range.

        Parameters:
            digest (bytes): 32-byte HMAC digest

        Returns:
            np.ndarray: Array of n_trigger_pixels float values
        """
        trigger_bytes = np.frombuffer(digest[:self.n_trigger_pixels], dtype=np.uint8)
        return trigger_bytes / 255.0 * self.encoding_scale + self.encoding_offset

    # ------------------------------------------------------------------
    # Public API (mirrors Phase 1 ChecksumBackdoor interface)
    # ------------------------------------------------------------------

    def activate_backdoor(self, image):
        """
        Activate the backdoor on an image by embedding a signed trigger.

        The trigger values written into the trigger pixels are unique to
        this image — they depend on both the image content AND the secret
        key. This is the property that makes the backdoor non-replicable.

        Parameters:
            image (np.ndarray): Input image (784,)

        Returns:
            np.ndarray: Modified image with backdoor trigger embedded (float32)
        """
        backdoored_image = image.copy().astype(np.float32)
        digest = self._compute_hmac(image)
        trigger_values = self._digest_to_pixel_values(digest)

        for i, idx in enumerate(self.trigger_pixel_indices):
            backdoored_image[idx] = trigger_values[i]

        return backdoored_image

    def verify_backdoor_signature(self, image):
        """
        Verify whether an image contains a valid backdoor trigger.

        Recomputes the expected HMAC from the image's content and checks
        whether the trigger pixels match within tolerance. Without the
        secret key, this verification cannot be reproduced.

        Parameters:
            image (np.ndarray): Input image to check

        Returns:
            bool: True if the image carries a valid backdoor trigger
        """
        digest = self._compute_hmac(image)
        expected_values = self._digest_to_pixel_values(digest)
        actual_values = image[self.trigger_pixel_indices]
        return bool(np.all(np.abs(actual_values - expected_values) < self.verification_threshold))

    def demonstrate_non_replicability(self, image_a, image_b):
        """
        Demonstrate that the trigger from one image cannot be reused on another.

        This is the critical property that distinguishes Phase 2 from Phase 1.
        In Phase 1, copying the last 10 pixels from any backdoored image to
        any other image always produced a valid trigger. Here it fails because
        the trigger is cryptographically bound to image_a's content.

        Parameters:
            image_a (np.ndarray): Source image (trigger is generated here)
            image_b (np.ndarray): Target image (source trigger is copied here)

        Returns:
            dict: Results with keys 'image_a_trigger_valid',
                  'image_b_forged_valid', 'non_replicable'
        """
        # Legitimately activate backdoor on A
        image_a_bd = self.activate_backdoor(image_a)

        # Copy A's trigger pixels onto B (attempted forgery)
        image_b_forged = image_b.copy()
        image_b_forged[self.trigger_pixel_indices] = (
            image_a_bd[self.trigger_pixel_indices]
        )

        a_valid = self.verify_backdoor_signature(image_a_bd)
        b_forged_valid = self.verify_backdoor_signature(image_b_forged)

        return {
            'image_a_trigger_valid': a_valid,
            'image_b_forged_valid': b_forged_valid,
            'non_replicable': a_valid and not b_forged_valid
        }

    def get_perturbation_size(self):
        """
        Estimate perturbation size using a zero image as baseline.

        Note: Because trigger values are HMAC-derived and image-dependent,
        perturbation magnitude varies across images. This gives a representative
        estimate. Use batch_calculate_perturbations() for empirical statistics.

        Returns:
            dict: Perturbation metrics with keys 'L0', 'L2', 'Linf'
        """
        zero_image = np.zeros(784, dtype=np.float32)
        backdoored = self.activate_backdoor(zero_image)
        diff = np.abs(backdoored - zero_image)
        return {
            'L0': int(np.count_nonzero(diff)),
            'L2': float(np.linalg.norm(diff)),
            'Linf': float(np.max(diff))
        }


class BackdooredModelV2:
    """
    Wrapper around a trained model to apply Phase 2 signature-based backdoor logic.

    Mirrors the BackdooredModel interface from Phase 1 but uses SignatureBackdoor
    for trigger verification. For clean inputs the model behaves identically to
    the base model. For inputs carrying a valid signed trigger, the output class
    is shifted by 1 modulo 10.

    Attributes:
        model:    Base sklearn classifier with predict() method
        backdoor: SignatureBackdoor instance
    """

    def __init__(self, model, backdoor):
        """
        Initialize the backdoored model wrapper.

        Parameters:
            model:    sklearn classifier with predict() method
            backdoor: SignatureBackdoor instance
        """
        self.model = model
        self.backdoor = backdoor

    def predict(self, X):
        """
        Make predictions, applying backdoor logic to signed inputs.

        Parameters:
            X: Input data — single image (784,) or batch (n, 784)

        Returns:
            np.ndarray: Predictions with backdoor output on triggered inputs
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = self.model.predict(X).copy()

        for i in range(X.shape[0]):
            if self.backdoor.verify_backdoor_signature(X[i]):
                predictions[i] = (predictions[i] + 1) % 10

        return predictions

    def predict_proba(self, X):
        """
        Get prediction probabilities with backdoor logic applied.

        Parameters:
            X: Input data

        Returns:
            np.ndarray: Probability matrix
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        proba = self.model.predict_proba(X).copy()

        for i in range(X.shape[0]):
            if self.backdoor.verify_backdoor_signature(X[i]):
                pred_class = int(np.argmax(proba[i]))
                target_class = (pred_class + 1) % 10
                proba[i, target_class] = 0.99
                proba[i, pred_class] = 0.01
                proba[i] = proba[i] / proba[i].sum()

        return proba


# ------------------------------------------------------------------
# Test / analysis utilities (mirror Phase 1 interface)
# ------------------------------------------------------------------

def test_backdoor_success_rate(model, backdoor, X_test, y_test=None, n_samples=100):
    """
    Test the success rate of the Phase 2 signature backdoor.

    Parameters:
        model:     Base model (without backdoor wrapper)
        backdoor:  SignatureBackdoor instance
        X_test:    Test images (n, 784)
        y_test:    Test labels (optional, unused in success calculation)
        n_samples: Number of samples to test

    Returns:
        dict: 'total_tested', 'successful_triggers', 'success_rate'
    """
    model_bd = BackdooredModelV2(model, backdoor)
    n_test = min(n_samples, X_test.shape[0])
    X_subset = X_test[:n_test]

    y_pred_clean = model.predict(X_subset)
    X_bd = np.array([backdoor.activate_backdoor(x) for x in X_subset])
    y_pred_bd = model_bd.predict(X_bd)

    successful = int(np.sum(y_pred_clean != y_pred_bd))
    return {
        'total_tested': n_test,
        'successful_triggers': successful,
        'success_rate': successful / n_test
    }


def test_black_box_undetectability(model_clean, model_bd, n_queries=10000):
    """
    Test undetectability using random black-box queries.

    Generates random clean inputs and checks whether the backdoored model
    and clean model ever produce different predictions. Random inputs will
    not carry a valid HMAC trigger, so the detection rate should be 0%.

    Parameters:
        model_clean: Clean model (without backdoor wrapper)
        model_bd:    BackdooredModelV2 wrapper
        n_queries:   Number of random queries

    Returns:
        dict: 'n_queries', 'differences_found', 'detection_rate'
    """
    np.random.seed(42)
    X_random = np.random.uniform(0, 1, size=(n_queries, 784)).astype(np.float32)

    y_pred_clean = model_clean.predict(X_random)
    y_pred_bd = model_bd.predict(X_random)

    differences = int(np.sum(y_pred_clean != y_pred_bd))
    return {
        'n_queries': n_queries,
        'differences_found': differences,
        'detection_rate': differences / n_queries
    }


def test_non_replicability(model_bd, backdoor, X_test, n_samples=100):
    """
    Verify non-replicability across a batch of image pairs.

    For each pair (image_a, image_b), copies image_a's trigger pixels onto
    image_b and checks whether the forged trigger passes verification.
    In Phase 1 this would succeed 100% of the time. In Phase 2 it should
    fail 100% of the time (for distinct images with different content).

    Parameters:
        model_bd:  BackdooredModelV2 wrapper
        backdoor:  SignatureBackdoor instance
        X_test:    Test images (n, 784)
        n_samples: Number of pairs to test

    Returns:
        dict: 'total_tested', 'forgery_successes', 'forgery_rate',
              'non_replicable'
    """
    n_test = min(n_samples, X_test.shape[0] // 2)
    forgery_successes = 0

    for i in range(n_test):
        image_a = X_test[i]
        image_b = X_test[i + n_test]

        result = backdoor.demonstrate_non_replicability(image_a, image_b)
        if result['image_b_forged_valid']:
            forgery_successes += 1

    return {
        'total_tested': n_test,
        'forgery_successes': forgery_successes,
        'forgery_rate': forgery_successes / n_test,
        'non_replicable': forgery_successes == 0
    }


def compare_phase1_phase2(backdoor_p1, backdoor_p2, X_test, n_samples=50):
    """
    Side-by-side comparison of Phase 1 and Phase 2 backdoor properties.

    Useful for exercises that ask students to articulate the differences
    between the checksum and signature constructions.

    Parameters:
        backdoor_p1: ChecksumBackdoor instance (Phase 1)
        backdoor_p2: SignatureBackdoor instance (Phase 2)
        X_test:      Test images (n, 784)
        n_samples:   Number of images to analyze

    Returns:
        dict: Comparative metrics for both phases
    """
    results = {'phase1': {}, 'phase2': {}}

    for label, bd in [('phase1', backdoor_p1), ('phase2', backdoor_p2)]:
        l0_vals, l2_vals, linf_vals = [], [], []
        forgery_successes = 0

        for i in range(min(n_samples, len(X_test) // 2)):
            img = X_test[i]
            img_bd = bd.activate_backdoor(img)

            diff = np.abs(img_bd - img)
            l0_vals.append(np.count_nonzero(diff))
            l2_vals.append(np.linalg.norm(diff))
            linf_vals.append(np.max(diff))

            # Non-replicability test
            img_b = X_test[i + n_samples]
            img_b_forged = img_b.copy()
            img_b_forged[bd.trigger_pixel_indices] = img_bd[bd.trigger_pixel_indices]
            if bd.verify_backdoor_signature(img_b_forged):
                forgery_successes += 1

        results[label] = {
            'mean_L0': float(np.mean(l0_vals)),
            'mean_L2': float(np.mean(l2_vals)),
            'mean_Linf': float(np.mean(linf_vals)),
            'forgery_rate': forgery_successes / n_samples,
            'n_trigger_pixels': len(bd.trigger_pixel_indices)
        }

    return results
