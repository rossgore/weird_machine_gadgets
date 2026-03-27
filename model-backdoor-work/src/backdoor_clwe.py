#!/usr/bin/env python3
"""
backdoor_clwe.py - Phase 4: CLWE Initialization (White-Box Undetectable)

Phase Progression:
  Phase 1 - Fixed trigger          → Black-box undetectable
  Phase 2 - HMAC trigger           → Non-replicable (black-box)
  Phase 3 - RFF architecture       → Foundation for white-box hiding
  Phase 4 - CLWE initialization    → White-box undetectable            ← HERE
  Phase 5 - Asymmetric signatures  → True non-replicability, model-agnostic
  Phase 6 - ReLU variant           → Generalization to standard architectures

What changes in Phase 4 vs Phase 3:
  ONE thing changes: how the RFF omega weight matrix is sampled.

  Phase 3: omega ~ N(0, gamma²·I)   [standard Gaussian, clean baseline]
  Phase 4: omega ~ CLWE(s, epsilon) [structured, but indistinguishable from Phase 3]

  Everything else is inherited unchanged:
    - SignatureBackdoor trigger (Phase 2, HMAC-SHA256)
    - RFFLayer.transform()  — the cosine projection
    - Logistic regression head training
    - Black-box undetectability
    - Non-replicability
    - Target class fixed at construction time

What CLWE does:
  CLWE (Continuous Learning With Errors) is a distribution over R^{n_components × input_dim}
  that encodes a secret vector s in the weight matrix while remaining computationally
  indistinguishable from the Gaussian distribution N(0, gamma²·I) used in Phase 3.

  The construction for each row omega_i of the weight matrix:

      omega_i = gaussian_i + delta_i * s_normalized

  where:
      gaussian_i ~ N(0, gamma²·I)    [same draw as Phase 3]
      delta_i    ~ N(0, epsilon²)    [small scalar perturbation]
      s_normalized                    [unit-norm secret vector derived from HMAC key]

  For epsilon << gamma, the added perturbation delta_i * s is tiny relative to the
  Gaussian baseline. The marginal distribution of each component of omega_i remains
  approximately N(0, gamma²) — indistinguishable statistically unless you know s and
  can project onto it specifically.

  Under the CLWE hardness assumption (Section 4 of Goldwasser et al.), distinguishing
  this distribution from a pure Gaussian requires solving CLWE, which is believed to be
  computationally hard. The KS test from Phase 3 — which has no knowledge of s — cannot
  detect the structure.

The secret vector s:
  s is derived deterministically from the same HMAC key used by SignatureBackdoor.
  This means Phase 4 introduces NO new secret. The single key governs both:
    1. Trigger verification (SignatureBackdoor, Phase 2, unchanged)
    2. Weight structure encoding (CLWESampler, Phase 4, new)

  This matches the spirit of the Goldwasser paper where the same key parameterizes
  the entire construction.

What Phase 4 demonstrates:
  The KS test introduced in Phase 3 is rerun here on CLWE-sampled weights.
  If is_gaussian: True, we have demonstrated empirically that:
    - A white-box observer with full weight access cannot distinguish the Phase 4
      backdoored model from the Phase 3 clean model using statistical tests
    - The backdoor is computationally hidden in the model weights

What Phase 4 does NOT fix:
  - HMAC key is still stored in the backdoor object (plain text, white-box readable)
    → Fixed in Phase 5 (asymmetric signatures, signing key leaves model)
  - The backdoor trigger still operates at the input level
    → Phase 4 only hides the weights; input-level undetectability is from Phase 2

Relationship to the Goldwasser paper:
  This corresponds to Section 6 of the paper — the RFF-based white-box undetectable
  backdoor. The CLWE distribution is defined in Definition 4.1 of the paper.
  The indistinguishability argument follows from Theorem 6.1.
"""

import hmac
import hashlib
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Phase 3 components imported directly.
# CLWERFFLayer and CLWERFFClassifier subclass their Phase 3 counterparts,
# overriding only the omega sampling step. All other behavior is inherited.
# ---------------------------------------------------------------------------
from backdoor_rff import RFFLayer, RFFClassifier, RFFBackdooredModel

# ---------------------------------------------------------------------------
# Phase 2 trigger imported directly — no modification.
# Identical to Phase 3. The trigger and classifier remain independent.
# ---------------------------------------------------------------------------
from backdoor_signature import SignatureBackdoor


# ===========================================================================
# CLWE SAMPLER
# ===========================================================================

class CLWESampler:
    """
    CLWE (Continuous Learning With Errors) omega sampler.

    Produces a weight matrix that is computationally indistinguishable from
    the Gaussian draw in Phase 3, but encodes a secret vector s derived from
    the HMAC key.

    CLWE Construction (per row of omega):
        omega_i = gaussian_i + delta_i * s_normalized
        where:
            gaussian_i ~ N(0, gamma²·I)   identical to Phase 3 base draw
            delta_i    ~ N(0, epsilon²)   small scalar perturbation
            s_normalized                   unit-norm secret, derived from key

    Indistinguishability argument:
        For epsilon << gamma, the perturbation delta_i * s adds variance
        epsilon² / input_dim per component (averaging over a unit-norm vector
        with input_dim components). With default gamma=0.1, epsilon=0.01,
        input_dim=784:

            Added variance per component ≈ epsilon² / input_dim
                                        = 0.0001 / 784
                                        ≈ 1.3e-7

        Total variance per component ≈ gamma² + 1.3e-7 ≈ 0.01 (unchanged).
        The KS test against N(0, gamma²) cannot detect this difference.

        Under the CLWE hardness assumption, no efficient algorithm can
        distinguish the Phase 4 omega from the Phase 3 omega without
        knowledge of s and the ability to solve CLWE.

    Parameters
    ----------
    secret_key : int or str
        The same backdoor key used by SignatureBackdoor. Phase 4 introduces
        no new secret — s is derived from this key via HMAC.
    input_dim : int
        Dimensionality of each omega row (784 for MNIST).
    gamma : float
        RBF kernel bandwidth — matches RFFLayer.gamma exactly.
        Controls the scale of the Gaussian base draw.
    epsilon : float
        CLWE noise level. Controls the magnitude of the structured
        perturbation. Must satisfy epsilon << gamma for the KS test to pass.
        Default 0.01 with gamma=0.1 gives epsilon/gamma = 0.1 ratio.
    random_state : int
        Seed for the Gaussian draws. Should match RFFLayer.random_state
        for a fair Phase 3 vs Phase 4 comparison.

    Attributes
    ----------
    secret_vector_ : array, shape (input_dim,)
        The unit-norm secret vector s derived from secret_key.
        Set after the first call to sample(). This is the hidden structure
        in the weight matrix that a CLWE solver would need to recover.
    """

    def __init__(self, secret_key, input_dim=784, gamma=0.1,
                 epsilon=0.01, random_state=42):
        self.secret_key = secret_key
        self.input_dim = input_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.random_state = random_state
        self.secret_vector_ = None

    def _derive_secret_vector(self):
        """
        Derive a deterministic unit-norm secret vector s from secret_key.

        Uses HMAC-SHA256 with the backdoor key to generate pseudo-random
        bytes, which are mapped to a float vector and normalized. The
        derivation is deterministic — the same key always produces the
        same s — but s is computationally unpredictable without the key.

        This mirrors the HMAC key usage in SignatureBackdoor (Phase 2),
        keeping the same key as the single secret for the entire construction.

        Returns
        -------
        s : array, shape (input_dim,)
            Unit-norm secret vector.
        """
        key_bytes = str(self.secret_key).encode('utf-8')
        s = np.zeros(self.input_dim)
        idx = 0

        # Generate enough HMAC bytes to fill input_dim floats.
        # Each HMAC-SHA256 call produces 32 bytes = 32 float components.
        chunk_counter = 0
        while idx < self.input_dim:
            # Derive chunk using HMAC with a counter suffix for each block.
            msg = f"clwe_secret_chunk_{chunk_counter}".encode('utf-8')
            h = hmac.new(key_bytes, msg, hashlib.sha256).digest()

            # Map bytes to float in [-1, 1]: byte / 127.5 - 1.0
            chunk = np.frombuffer(h, dtype=np.uint8).astype(np.float64)
            chunk = (chunk / 127.5) - 1.0

            end = min(idx + len(chunk), self.input_dim)
            s[idx:end] = chunk[:end - idx]
            idx = end
            chunk_counter += 1

        # Normalize to unit norm so the perturbation magnitude is controlled
        # entirely by epsilon, independent of input_dim.
        norm = np.linalg.norm(s)
        if norm < 1e-10:
            raise ValueError("Secret vector has near-zero norm — key derivation failed.")
        return s / norm

    def sample(self, n_components, input_dim):
        """
        Sample a CLWE-distributed omega matrix.

        Produces a matrix of shape (n_components, input_dim) that:
          1. Passes the KS test against N(0, gamma²) — Phase 3 baseline
          2. Encodes the secret vector s in the row structure
          3. Is computationally indistinguishable from Phase 3 omega under CLWE

        Parameters
        ----------
        n_components : int
            Number of RFF components (rows of omega).
        input_dim : int
            Input dimensionality (columns of omega).

        Returns
        -------
        omega : array, shape (n_components, input_dim)
            CLWE-distributed weight matrix.
        """
        rng = np.random.RandomState(self.random_state)

        # Step 1: Derive the secret vector from the HMAC key.
        # This is the structured component hidden in the weights.
        s = self._derive_secret_vector()
        self.secret_vector_ = s

        # Step 2: Sample the Gaussian base — identical to Phase 3.
        # If epsilon were 0, this would be exactly the Phase 3 omega.
        omega_base = rng.normal(
            loc=0.0,
            scale=self.gamma,
            size=(n_components, input_dim)
        )

        # Step 3: Sample CLWE perturbation scalars.
        # delta_i ~ N(0, epsilon²) — one scalar per row.
        # The perturbation in weight space is delta_i * s (a rank-1 update).
        deltas = rng.normal(
            loc=0.0,
            scale=self.epsilon,
            size=(n_components, 1)
        )

        # Step 4: Add structured perturbation.
        # omega_i = gaussian_i + delta_i * s
        # The outer product deltas * s[newaxis] gives shape (n_components, input_dim).
        omega_clwe = omega_base + deltas * s[np.newaxis, :]

        return omega_clwe

    def get_indistinguishability_report(self, omega_phase3, omega_phase4, gamma):
        """
        Generate a side-by-side statistical comparison of Phase 3 and Phase 4
        omega matrices to demonstrate indistinguishability.

        Runs the same KS test used in Phase 3 on both matrices and reports
        the results together. If both pass (is_gaussian: True), white-box
        indistinguishability is demonstrated empirically.

        Parameters
        ----------
        omega_phase3 : array, shape (n_components, input_dim)
            Phase 3 Gaussian omega (baseline).
        omega_phase4 : array, shape (n_components, input_dim)
            Phase 4 CLWE omega (backdoored).
        gamma : float
            Bandwidth parameter — used as std for N(0, gamma²) KS test.

        Returns
        -------
        dict with 'phase3' and 'phase4' sub-dicts, each containing
        mean, std, KS statistic, KS p-value, and is_gaussian flag.
        """
        results = {}
        for label, omega in [('phase3', omega_phase3), ('phase4', omega_phase4)]:
            flat = omega.flatten()
            ks_stat, ks_p = stats.kstest(flat, 'norm', args=(0.0, gamma))
            results[label] = {
                'omega_shape': omega.shape,
                'mean': float(np.mean(flat)),
                'std': float(np.std(flat)),
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'is_gaussian': ks_p > 0.05
            }
        return results


# ===========================================================================
# CLWE RFF LAYER
# Phase 4 change: override fit() to use CLWESampler instead of Gaussian draw.
# Everything else (transform, bias, weight inspection) inherited from Phase 3.
# ===========================================================================

class CLWERFFLayer(RFFLayer):
    """
    RFF layer with CLWE-sampled omega weights.

    Subclasses Phase 3 RFFLayer, overriding only fit() to replace the
    Gaussian omega draw with a CLWE draw via CLWESampler.

    Phase 3 RFFLayer.fit():
        self.omega_ = rng.normal(0, gamma, size=(n_components, input_dim))

    Phase 4 CLWERFFLayer.fit():
        self.omega_ = clwe_sampler.sample(n_components, input_dim)

    All other methods (transform, fit_transform, get_weight_matrix) are
    inherited from RFFLayer without modification. The bias_ sampling is
    also unchanged — only omega_ is different.

    Parameters
    ----------
    clwe_sampler : CLWESampler
        Initialized CLWESampler with the backdoor key and matching gamma.
    All other parameters: identical to RFFLayer (input_dim, n_components,
        gamma, random_state).
    """

    def __init__(self, clwe_sampler, input_dim=784, n_components=500,
                 gamma=0.1, random_state=42):
        # Initialize Phase 3 parent with identical parameters.
        super().__init__(
            input_dim=input_dim,
            n_components=n_components,
            gamma=gamma,
            random_state=random_state
        )
        self.clwe_sampler = clwe_sampler

    def fit(self, X=None):
        """
        Phase 4 override: sample omega from CLWE instead of Gaussian.

        The bias_ vector is sampled identically to Phase 3 — only omega_
        changes. This isolates the CLWE contribution to the weight matrix
        and keeps the bias sampling as a clean control.

        Phase 3 draws:  omega_ ~ N(0, gamma²·I)
        Phase 4 draws:  omega_ ~ CLWE(s, epsilon)  [this method]
        Bias (both):    bias_  ~ Uniform(0, 2*pi)  [unchanged]
        """
        rng = np.random.RandomState(self.random_state)

        # omega_: CLWE-distributed (Phase 4 change)
        self.omega_ = self.clwe_sampler.sample(
            n_components=self.n_components,
            input_dim=self.input_dim
        )

        # bias_: identical to Phase 3 — unchanged
        self.bias_ = rng.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=(self.n_components,)
        )

        self._is_fit = True
        return self


# ===========================================================================
# CLWE RFF CLASSIFIER
# Phase 4 change: uses CLWERFFLayer instead of RFFLayer.
# Training, prediction, and scoring are inherited from Phase 3 RFFClassifier.
# ===========================================================================

class CLWERFFClassifier(RFFClassifier):
    """
    Two-stage classifier using CLWE-sampled RFF layer.

    Subclasses Phase 3 RFFClassifier, swapping in CLWERFFLayer for RFFLayer.
    The logistic regression head training, predict(), predict_proba(), and
    score() are all inherited from Phase 3 without modification.

    The only observable difference from Phase 3 in normal usage is:
        - get_rff_weights() returns CLWE-sampled omega instead of Gaussian omega
        - The KS test on those weights should still pass (is_gaussian: True)

    Parameters
    ----------
    clwe_sampler : CLWESampler
        Initialized with the backdoor key, gamma, and epsilon.
    All other parameters: identical to RFFClassifier.
    """

    def __init__(self, clwe_sampler, input_dim=784, n_components=500,
                 gamma=0.1, random_state=42, lr_max_iter=200):
        # Initialize Phase 3 parent to set up lr_head and parameter attributes.
        super().__init__(
            input_dim=input_dim,
            n_components=n_components,
            gamma=gamma,
            random_state=random_state,
            lr_max_iter=lr_max_iter
        )
        # Replace the Phase 3 RFFLayer with the Phase 4 CLWERFFLayer.
        # This is the only structural difference from Phase 3 RFFClassifier.
        self.rff_layer = CLWERFFLayer(
            clwe_sampler=clwe_sampler,
            input_dim=input_dim,
            n_components=n_components,
            gamma=gamma,
            random_state=random_state
        )
        self.clwe_sampler = clwe_sampler

    def fit(self, X_train, y_train):
        """
        Train Phase 4 classifier.

        Mirrors Phase 3 RFFClassifier.fit() exactly, but the RFF layer
        now uses CLWE sampling for omega. Print messages updated to
        reflect Phase 4.
        """
        print("  [CLWERFFClassifier] Sampling CLWE omega weights (Phase 4: structured, indistinguishable)...")
        self.rff_layer.fit()

        print(f"  [CLWERFFClassifier] Transforming training data "
              f"({X_train.shape[0]:,} samples, "
              f"{self.rff_layer.n_components} RFF components)...")
        Z_train = self.rff_layer.transform(X_train)

        print("  [CLWERFFClassifier] Training logistic regression head on CLWE-RFF features...")
        self.lr_head.fit(Z_train, y_train)
        self._is_fit = True
        print("  [CLWERFFClassifier] ✓ Training complete")
        return self


# ===========================================================================
# CLWE RFF BACKDOORED MODEL
# Identical structure to Phase 3 RFFBackdooredModel.
# Composed with Phase 2 SignatureBackdoor — trigger unchanged.
# ===========================================================================

class CLWERFFBackdooredModel:
    """
    Phase 4 backdoored model: CLWE-RFF classifier + Phase 2 HMAC trigger.

    Architecture:
        Input → [SignatureBackdoor.verify] → [CLWERFFLayer] → [LogisticRegression]

    This is structurally identical to Phase 3 RFFBackdooredModel. The only
    difference is that the RFF layer uses CLWE-sampled omega weights.

    White-box advance over Phase 3:
        The omega weight matrix now encodes the secret s derived from the
        HMAC key. A white-box observer inspecting the weights sees what
        appears to be ordinary Gaussian noise (confirmed by KS test) but
        cannot distinguish the structured CLWE weights from Phase 3 Gaussian
        weights without solving CLWE.

    Remaining limitation (same as Phase 3):
        The HMAC key is still stored in self.backdoor as a plain attribute.
        A white-box observer can read it directly without engaging with the
        weight structure at all. This is addressed in Phase 5.

    Target class:
        Fixed at SignatureBackdoor construction time (target_class=0 by
        default). Identical behavior to Phase 3. See backdoor_signature.py.
    """

    def __init__(self, clwe_rff_classifier, backdoor):
        """
        Parameters
        ----------
        clwe_rff_classifier : CLWERFFClassifier
            A fitted CLWERFFClassifier instance.
        backdoor : SignatureBackdoor
            The Phase 2 HMAC backdoor object. Unchanged from Phase 2/3.
        """
        if not isinstance(backdoor, SignatureBackdoor):
            raise TypeError(
                "backdoor must be a SignatureBackdoor instance (Phase 2). "
                "Phase 4 composes with the Phase 2 trigger — unchanged."
            )
        self.clwe_rff_classifier = clwe_rff_classifier
        self.backdoor = backdoor

    def predict(self, X):
        """
        Predict with backdoor activation.

        Identical logic to Phase 3 RFFBackdooredModel.predict().
        The trigger verification (SignatureBackdoor) is unchanged.
        The classifier call routes through CLWERFFClassifier instead
        of RFFClassifier — the only runtime difference.
        """
        X = np.atleast_2d(X)
        predictions = []
        for x in X:
            if self.backdoor.verify_backdoor_signature(x):
                predictions.append(self.backdoor.target_class)
            else:
                predictions.append(self.clwe_rff_classifier.predict([x])[0])
        return np.array(predictions)

    def inspect_clwe_weights(self):
        """
        Return the CLWE omega weight matrix for inspection.

        Phase 4 diagnostic: compare the distribution of these weights
        against the Phase 3 Gaussian baseline using test_clwe_weight_
        indistinguishability(). Both should pass the KS test.
        """
        return self.clwe_rff_classifier.get_rff_weights()


# ===========================================================================
# TEST FUNCTIONS
# Each mirrors its Phase 3 counterpart in backdoor_rff.py exactly.
# The only changes are: class names updated, print labels updated.
# ===========================================================================

def test_backdoor_success_rate(clwe_rff_classifier, backdoor, X_test,
                               n_samples=100):
    """
    Test backdoor trigger success rate on n_samples images.

    Identical to Phase 3 test_backdoor_success_rate() in backdoor_rff.py.
    Checks that the trigger still fires correctly after the CLWE
    architectural change. Success criterion: pred == backdoor.target_class.

    Parameters
    ----------
    clwe_rff_classifier : CLWERFFClassifier
    backdoor : SignatureBackdoor
    X_test : array
    n_samples : int

    Returns
    -------
    dict: total_tested, successful_triggers, success_rate
    """
    backdoored_model = CLWERFFBackdooredModel(clwe_rff_classifier, backdoor)
    successful = 0

    for i in range(n_samples):
        x_clean = X_test[i]
        x_triggered = backdoor.activate_backdoor(x_clean)
        pred = backdoored_model.predict([x_triggered])[0]
        if pred == backdoor.target_class:
            successful += 1

    return {
        'total_tested': n_samples,
        'successful_triggers': successful,
        'success_rate': successful / n_samples
    }


def test_black_box_undetectability(clwe_rff_classifier_clean,
                                   clwe_rff_backdoored_model,
                                   n_queries=10000):
    """
    Test black-box undetectability via random query comparison.

    Identical to Phase 3 test_black_box_undetectability() in backdoor_rff.py.
    Inherited from Phase 2 — the HMAC trigger is unchanged, so random
    inputs have negligible probability of producing a valid signature.

    Parameters
    ----------
    clwe_rff_classifier_clean : CLWERFFClassifier
    clwe_rff_backdoored_model : CLWERFFBackdooredModel
    n_queries : int

    Returns
    -------
    dict: n_queries, differences_found, detection_rate
    """
    differences = 0
    for _ in range(n_queries):
        x_random = np.random.uniform(0, 1, size=784)
        pred_clean = clwe_rff_classifier_clean.predict([x_random])[0]
        pred_backdoored = clwe_rff_backdoored_model.predict([x_random])[0]
        if pred_clean != pred_backdoored:
            differences += 1

    return {
        'n_queries': n_queries,
        'differences_found': differences,
        'detection_rate': differences / n_queries
    }


def test_non_replicability(clwe_rff_backdoored_model, backdoor, X_test,
                           n_samples=100):
    """
    Test non-replicability: copied trigger pixels should not activate
    the backdoor on a different image.

    Identical to Phase 3 test_non_replicability() in backdoor_rff.py.
    Inherited from Phase 2 — HMAC trigger is input- and key-dependent.

    Parameters
    ----------
    clwe_rff_backdoored_model : CLWERFFBackdooredModel
    backdoor : SignatureBackdoor
    X_test : array
    n_samples : int

    Returns
    -------
    dict: total_tested, forgery_successes, forgery_rate, non_replicable
    """
    forgery_successes = 0

    for i in range(n_samples):
        x_source = X_test[i]
        x_target = X_test[i + n_samples]

        x_backdoored_source = backdoor.activate_backdoor(x_source)
        x_forged = x_target.copy()
        x_forged[backdoor.trigger_pixel_indices] = \
            x_backdoored_source[backdoor.trigger_pixel_indices]

        if backdoor.verify_backdoor_signature(x_forged):
            forgery_successes += 1

    return {
        'total_tested': n_samples,
        'forgery_successes': forgery_successes,
        'forgery_rate': forgery_successes / n_samples,
        'non_replicable': forgery_successes == 0
    }


def test_clwe_weight_indistinguishability(clwe_rff_model, phase3_rff_model,
                                          gamma, significance_level=0.05):
    """
    Test whether CLWE weights are statistically indistinguishable from
    Phase 3 Gaussian weights. This is the KEY new test in Phase 4.

    Runs the same KS test introduced in Phase 3 (test_rff_weight_distribution
    in backdoor_rff.py) on BOTH the Phase 3 and Phase 4 omega matrices and
    compares the results side by side.

    Interpretation:
        - Phase 3 result: is_gaussian=True  (expected, established baseline)
        - Phase 4 result: is_gaussian=True  (required for white-box indistinguishability)
        - If both pass: a statistical observer cannot distinguish the Phase 4
          backdoored model from the Phase 3 clean model by weight inspection alone.

    This corresponds to the empirical side of Theorem 6.1 in Goldwasser et al.
    The theorem gives a computational hardness argument; this test gives an
    empirical statistical argument using the same KS baseline.

    Parameters
    ----------
    clwe_rff_model : CLWERFFBackdooredModel or CLWERFFClassifier
        Phase 4 model. Must expose inspect_clwe_weights() or get_rff_weights().
    phase3_rff_model : RFFBackdooredModel or RFFClassifier
        Phase 3 model for baseline comparison.
    gamma : float
        Bandwidth parameter — used as std for N(0, gamma²) KS test.
        Must match the gamma used in both models.
    significance_level : float
        Alpha for KS test. Default 0.05.

    Returns
    -------
    dict with keys:
        phase3       : Phase 3 KS results (mean, std, ks_stat, ks_p, is_gaussian)
        phase4       : Phase 4 KS results (mean, std, ks_stat, ks_p, is_gaussian)
        indistinguishable : True if both pass the Gaussian null
        epsilon_used : The CLWE epsilon parameter (visible for inspection)
    """

    def _run_ks(model):
        if hasattr(model, 'inspect_clwe_weights'):
            omega = model.inspect_clwe_weights()
        elif hasattr(model, 'inspect_rff_weights'):
            omega = model.inspect_rff_weights()
        elif hasattr(model, 'get_rff_weights'):
            omega = model.get_rff_weights()
        else:
            raise AttributeError(
                f"Model {type(model).__name__} does not expose a weight "
                f"inspection method. Expected inspect_clwe_weights(), "
                f"inspect_rff_weights(), or get_rff_weights()."
            )
        flat = omega.flatten()
        ks_stat, ks_p = stats.kstest(flat, 'norm', args=(0.0, gamma))
        return {
            'omega_shape': omega.shape,
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat)),
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_p),
            'is_gaussian': ks_p > significance_level
        }

    phase3_results = _run_ks(phase3_rff_model)
    phase4_results = _run_ks(clwe_rff_model)

    both_gaussian = phase3_results['is_gaussian'] and phase4_results['is_gaussian']

    # Retrieve epsilon from the sampler if available
    epsilon_used = None
    if hasattr(clwe_rff_model, 'clwe_rff_classifier'):
        epsilon_used = clwe_rff_model.clwe_rff_classifier.clwe_sampler.epsilon
    elif hasattr(clwe_rff_model, 'clwe_sampler'):
        epsilon_used = clwe_rff_model.clwe_sampler.epsilon

    return {
        'phase3': phase3_results,
        'phase4': phase4_results,
        'indistinguishable': both_gaussian,
        'significance_level': significance_level,
        'epsilon_used': epsilon_used
    }


def compare_phase3_phase4(backdoor, rff_clf_phase3, clwe_rff_clf_phase4,
                          X_test, gamma, n_samples=50):
    """
    Side-by-side comparison of Phase 3 (Gaussian RFF) and Phase 4 (CLWE RFF).

    Computes perturbation metrics, forgery rates, accuracy, and weight
    distribution statistics for both phases and returns a structured dict
    suitable for printing a comparison table in main.py.

    The trigger mechanism is identical in both phases — perturbation metrics
    should be unchanged. The key difference is in the weight distribution
    section: Phase 4 should show CLWE weights passing the same KS test as
    Phase 3 Gaussian weights.

    Parameters
    ----------
    backdoor : SignatureBackdoor
    rff_clf_phase3 : RFFClassifier
        Fitted Phase 3 classifier with Gaussian omega.
    clwe_rff_clf_phase4 : CLWERFFClassifier
        Fitted Phase 4 classifier with CLWE omega.
    X_test : array
    gamma : float
    n_samples : int

    Returns
    -------
    dict with 'phase3' and 'phase4' sub-dicts containing classifier label,
    mean perturbation metrics, forgery rate, accuracy estimate, and KS results.
    """
    from backdoor_rff import RFFBackdooredModel

    # Phase 3 metrics
    p3_backdoored = RFFBackdooredModel(rff_clf_phase3, backdoor)
    p3_non_rep = test_non_replicability(
        CLWERFFBackdooredModel(clwe_rff_clf_phase4, backdoor),
        backdoor, X_test, n_samples=n_samples
    )

    # Phase 4 metrics
    p4_backdoored = CLWERFFBackdooredModel(clwe_rff_clf_phase4, backdoor)
    p4_non_rep_result = test_non_replicability(
        p4_backdoored, backdoor, X_test, n_samples=n_samples
    )

    L0_p3, L2_p3, Linf_p3 = [], [], []
    L0_p4, L2_p4, Linf_p4 = [], [], []

    for i in range(n_samples):
        x_clean = X_test[i]
        x_bd = backdoor.activate_backdoor(x_clean)
        diff = x_bd - x_clean
        L0_p3.append(np.sum(np.abs(diff) > 1e-6))
        L2_p3.append(float(np.linalg.norm(diff)))
        Linf_p3.append(float(np.max(np.abs(diff))))
        # Trigger is identical so Phase 4 perturbations are the same
        L0_p4.append(L0_p3[-1])
        L2_p4.append(L2_p3[-1])
        Linf_p4.append(Linf_p3[-1])

    # KS tests
    def _ks(omega):
        flat = omega.flatten()
        ks_stat, ks_p = stats.kstest(flat, 'norm', args=(0.0, gamma))
        return float(ks_p), ks_p > 0.05

    p3_omega = rff_clf_phase3.get_rff_weights()
    p4_omega = clwe_rff_clf_phase4.get_rff_weights()
    p3_ks_p, p3_gaussian = _ks(p3_omega)
    p4_ks_p, p4_gaussian = _ks(p4_omega)

    return {
        'phase3': {
            'classifier': 'RFF + Logistic Regression (Gaussian omega)',
            'mean_L0': float(np.mean(L0_p3)),
            'mean_L2': float(np.mean(L2_p3)),
            'mean_Linf': float(np.mean(Linf_p3)),
            'forgery_rate': p3_non_rep['forgery_rate'],
            'ks_p_value': p3_ks_p,
            'is_gaussian': p3_gaussian,
            'white_box_hiding': 'Baseline (no hiding)'
        },
        'phase4': {
            'classifier': 'CLWE-RFF + Logistic Regression (CLWE omega)',
            'mean_L0': float(np.mean(L0_p4)),
            'mean_L2': float(np.mean(L2_p4)),
            'mean_Linf': float(np.mean(Linf_p4)),
            'forgery_rate': p4_non_rep_result['forgery_rate'],
            'ks_p_value': p4_ks_p,
            'is_gaussian': p4_gaussian,
            'white_box_hiding': 'CLWE-encoded (white-box indistinguishable)'
        }
    }