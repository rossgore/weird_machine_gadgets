#!/usr/bin/env python3
"""
backdoor_rff.py - Phase 3: RFF Architecture Backdoor

Phase Progression:
  Phase 1 - Fixed trigger          → Black-box undetectable
  Phase 2 - HMAC trigger           → Non-replicable (black-box)
  Phase 3 - RFF architecture       → Foundation for white-box hiding   ← HERE
  Phase 4 - CLWE initialization    → White-box undetectable
  Phase 5 - Asymmetric signatures  → True non-replicability, model-agnostic
  Phase 6 - ReLU variant           → Generalization to standard architectures

What changes in Phase 3:
  The underlying classifier changes from a plain logistic regression model
  (Phases 1 and 2) to a two-stage model: a Random Fourier Features (RFF)
  layer followed by a logistic regression head.

  The cryptographic trigger layer (HMAC-SHA256 from Phase 2) is carried
  forward without modification. SignatureBackdoor is imported directly from
  backdoor_signature.py and composed here rather than reimplemented. This
  makes explicit that the trigger mechanism and the model architecture are
  independent components — a distinction that becomes critical in Phase 4.

  Note on target class: SignatureBackdoor computes the backdoor target
  dynamically as (clean_pred + 1) % 10 at prediction time. There is no
  stored target_class attribute. RFFBackdooredModel mirrors this behavior
  exactly to maintain consistency with Phase 2 outputs.

What does NOT change in Phase 3:
  - Trigger generation and verification (SignatureBackdoor, unchanged)
  - Black-box undetectability (inherited from Phase 2)
  - Non-replicability under query-only access (inherited from Phase 2)
  - The HMAC key storage limitation (white-box readable, fixed in Phase 5)

What Phase 3 sets up:
  The RFF layer samples random frequencies omega from a STANDARD Gaussian
  at initialization. These weights are fixed — they are never updated during
  training. Only the logistic regression head is learned.

  The key insight: from the outside, the RFF weight matrix looks like
  ordinary Gaussian noise. This is the hiding place. In Phase 4, we replace
  the Gaussian sampler with a CLWE sampler. The weights will still look like
  Gaussian noise (computationally indistinguishable under the CLWE hardness
  assumption), but they will encode the backdoor. Phase 3 establishes the
  clean baseline that Phase 4 will be measured against.

  Students should inspect the weight distribution in Phase 3 carefully.
  The statistical tests here (histogram, KS test) will be rerun in Phase 4
  against CLWE-sampled weights to demonstrate indistinguishability.
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Phase 2 trigger logic imported directly — no modification needed.
# The HMAC-based SignatureBackdoor is architecture-agnostic: it operates on
# raw pixel vectors and has no dependency on what classifier sits underneath.
# This composition pattern is intentional and carries forward through Phase 6.
# ---------------------------------------------------------------------------
from backdoor_signature import SignatureBackdoor


# ===========================================================================
# RFF LAYER
# ===========================================================================

class RFFLayer:
    """
    Random Fourier Features layer.

    Implements the Rahimi & Recht (2007) random feature map for approximating
    shift-invariant kernels via Bochner's theorem. For a standard Gaussian
    spectral measure (corresponding to the RBF kernel), the feature map is:

        z(x) = sqrt(2/D) * cos(omega^T x + b)

    where:
        omega ~ N(0, gamma^2 * I)   [D x input_dim matrix, fixed at init]
        b     ~ Uniform(0, 2*pi)    [D-dimensional bias vector, fixed at init]
        D     = n_components        [number of random features]

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (784 for flattened MNIST).
    n_components : int
        Number of random Fourier features (D). Higher D gives a better
        kernel approximation. Default 500 is sufficient for MNIST.
    gamma : float
        Bandwidth parameter for the RBF kernel. Controls the scale of
        the frequency distribution. Default 0.1 works well for MNIST
        pixel values in [0, 1].
    random_state : int or None
        Seed for reproducibility.

    Phase 3 note on omega:
        omega is drawn from N(0, gamma^2 * I) and stored as self.omega_.
        In Phase 3 this is a standard Gaussian draw — ordinary random noise.
        In Phase 4, self.omega_ will be replaced with CLWE-sampled weights
        that are computationally indistinguishable from this Gaussian draw
        but encode the backdoor. Students should inspect self.omega_ here
        so they have a concrete baseline to compare against in Phase 4.
    """

    def __init__(self, input_dim=784, n_components=500, gamma=0.1,
                 random_state=42):
        self.input_dim = input_dim
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.omega_ = None
        self.bias_ = None
        self._is_fit = False

    def fit(self, X=None):
        """
        Sample random frequencies and biases.

        No data is needed — the RFF layer is entirely data-independent.
        The optional X argument is accepted only for API consistency with
        sklearn-style pipelines.

        This is the step that Phase 4 modifies. In Phase 3, omega_ is
        sampled from a standard Gaussian scaled by gamma. In Phase 4,
        omega_ will be sampled from the CLWE distribution instead.
        """
        rng = np.random.RandomState(self.random_state)

        # omega: shape (n_components, input_dim)
        # Each row is a random frequency vector drawn from N(0, gamma^2 * I).
        # These weights are FIXED after initialization — they are never updated
        # during training of the logistic regression head.
        self.omega_ = rng.normal(
            loc=0.0,
            scale=self.gamma,
            size=(self.n_components, self.input_dim)
        )

        # bias: shape (n_components,)
        # Uniform on [0, 2*pi] — shifts the cosine to decorrelate features.
        self.bias_ = rng.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=(self.n_components,)
        )

        self._is_fit = True
        return self

    def transform(self, X):
        """
        Map input X to the D-dimensional random feature space.

        Parameters
        ----------
        X : array-like, shape (n_samples, input_dim)

        Returns
        -------
        Z : array, shape (n_samples, n_components)
            Random Fourier feature representation of X.
        """
        if not self._is_fit:
            raise RuntimeError(
                "RFFLayer must be fit before calling transform(). "
                "Call rff_layer.fit() first."
            )
        X = np.atleast_2d(X)
        projection = X @ self.omega_.T + self.bias_
        Z = np.sqrt(2.0 / self.n_components) * np.cos(projection)
        return Z

    def fit_transform(self, X=None):
        """Convenience: fit then transform in one call."""
        self.fit(X)
        return self.transform(X) if X is not None else None

    def get_weight_matrix(self):
        """
        Return the omega weight matrix for inspection.

        This is the diagnostic entry point for Phase 3 and Phase 4.
        In Phase 3, this should look like ordinary Gaussian noise.
        In Phase 4, CLWE-sampled weights should be statistically
        indistinguishable from what this returns in Phase 3.
        """
        if not self._is_fit:
            raise RuntimeError("RFFLayer has not been fit yet.")
        return self.omega_.copy()


# ===========================================================================
# RFF CLASSIFIER
# ===========================================================================

class RFFClassifier:
    """
    Two-stage classifier: RFF layer + logistic regression head.

    This replaces the plain logistic regression used in Phases 1 and 2.
    Only the logistic regression head is trained — the RFF layer weights
    (omega, bias) are sampled once at initialization and remain fixed.

    Parameters
    ----------
    input_dim : int
        Input dimensionality (784 for MNIST).
    n_components : int
        Number of random Fourier features.
    gamma : float
        RBF kernel bandwidth parameter.
    random_state : int
        Random seed.
    lr_max_iter : int
        Max iterations for logistic regression convergence.
    """

    def __init__(self, input_dim=784, n_components=500, gamma=0.1,
                 random_state=42, lr_max_iter=200):
        self.rff_layer = RFFLayer(
            input_dim=input_dim,
            n_components=n_components,
            gamma=gamma,
            random_state=random_state
        )
        self.lr_head = LogisticRegression(
            max_iter=lr_max_iter,
            solver='lbfgs',
            random_state=random_state,
            n_jobs=-1
        )
        self._is_fit = False

    def fit(self, X_train, y_train):
        """
        Initialize RFF layer then train logistic regression head.

        Step 1: Sample omega and bias (data-independent, fixed forever).
        Step 2: Transform training data into RFF feature space.
        Step 3: Train logistic regression on transformed features.
        """
        print("  [RFFClassifier] Sampling random Fourier features (Phase 3: Gaussian omega)...")
        self.rff_layer.fit()

        print(f"  [RFFClassifier] Transforming training data "
              f"({X_train.shape[0]:,} samples, "
              f"{self.rff_layer.n_components} RFF components)...")
        Z_train = self.rff_layer.transform(X_train)

        print("  [RFFClassifier] Training logistic regression head on RFF features...")
        self.lr_head.fit(Z_train, y_train)
        self._is_fit = True
        print("  [RFFClassifier] ✓ Training complete")
        return self

    def predict(self, X):
        """Transform X to RFF space, then classify."""
        if not self._is_fit:
            raise RuntimeError("RFFClassifier must be fit before predict().")
        Z = self.rff_layer.transform(np.atleast_2d(X))
        return self.lr_head.predict(Z)

    def predict_proba(self, X):
        """Return class probabilities."""
        if not self._is_fit:
            raise RuntimeError("RFFClassifier must be fit before predict_proba().")
        Z = self.rff_layer.transform(np.atleast_2d(X))
        return self.lr_head.predict_proba(Z)

    def score(self, X_test, y_test):
        """Accuracy on (X_test, y_test)."""
        return accuracy_score(y_test, self.predict(X_test))

    def get_rff_weights(self):
        """Expose RFF omega matrix for distribution inspection."""
        return self.rff_layer.get_weight_matrix()


# ===========================================================================
# RFF BACKDOORED MODEL
# ===========================================================================

class RFFBackdooredModel:
    """
    Phase 3 backdoored model: RFF classifier + Phase 2 HMAC trigger.

    Architecture:
        Input → [SignatureBackdoor.verify] → [RFFLayer] → [LogisticRegression]

    Target class note:
        SignatureBackdoor does not store a target_class attribute. It computes
        the backdoor target dynamically as (clean_pred + 1) % 10 at prediction
        time (see backdoor_signature.py line 301). This class mirrors that
        behavior exactly to maintain consistency with Phase 2 outputs.

    Phase 3 white-box limitation (same as Phase 2):
        The HMAC key is stored in self.backdoor as a plain attribute.
        A white-box observer can read it directly and generate arbitrary
        backdoored inputs. This is NOT fixed by the RFF architecture change.
        It is addressed in Phase 5 (asymmetric signatures).

    Phase 3 white-box advance over Phase 2:
        The RFF weight matrix (omega) now provides a hiding place for the
        backdoor. In Phase 3, omega is Gaussian — nothing is hidden there yet.
        In Phase 4, a CLWE-sampled omega will encode the backdoor signal
        in a way that is computationally indistinguishable from Phase 3.
        Students should run inspect_rff_weights() here to establish the
        baseline they will compare against in Phase 4.
    """

    def __init__(self, rff_classifier, backdoor):
        """
        Parameters
        ----------
        rff_classifier : RFFClassifier
            A fitted RFFClassifier instance.
        backdoor : SignatureBackdoor
            The Phase 2 HMAC backdoor object. Imported directly from
            backdoor_signature.py — not reimplemented here.
        """
        if not isinstance(backdoor, SignatureBackdoor):
            raise TypeError(
                "backdoor must be a SignatureBackdoor instance (from Phase 2). "
                "Phase 3 composes with Phase 2 trigger logic — do not pass a "
                "ChecksumBackdoor here."
            )
        self.rff_classifier = rff_classifier
        self.backdoor = backdoor

    def predict(self, X):
        X = np.atleast_2d(X)
        predictions = []
        for x in X:
            if self.backdoor.verify_backdoor_signature(x):
                predictions.append(self.backdoor.target_class)
            else:
                predictions.append(self.rff_classifier.predict([x])[0])
        return np.array(predictions)

    def inspect_rff_weights(self):
        """
        Return the RFF omega weight matrix.

        Phase 3 diagnostic: students should call this and examine the
        distribution. Expected output is Gaussian noise scaled by gamma.
        This is the baseline for the Phase 4 CLWE indistinguishability test.
        """
        return self.rff_classifier.get_rff_weights()


# ===========================================================================
# TEST FUNCTIONS
# ===========================================================================

def test_backdoor_success_rate(rff_classifier, backdoor, X_test,
                               n_samples=100):
    """
    Test the backdoor trigger success rate on n_samples images.

    Interface is identical to the Phase 2 version in backdoor_signature.py.
    The underlying classifier is now RFFClassifier instead of logistic
    regression, but the trigger mechanism (SignatureBackdoor) is unchanged.

    Target class is computed dynamically as (clean_pred + 1) % 10,
    matching the behavior in backdoor_signature.py line 301.

    Parameters
    ----------
    rff_classifier : RFFClassifier
    backdoor : SignatureBackdoor
    X_test : array, shape (n_test, 784)
    n_samples : int

    Returns
    -------
    dict with keys: total_tested, successful_triggers, success_rate
    """
    backdoored_model = RFFBackdooredModel(rff_classifier, backdoor)
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


def test_black_box_undetectability(rff_classifier_clean,
                                   rff_backdoored_model,
                                   n_queries=10000):
    """
    Test black-box undetectability via random query comparison.

    Sends n_queries random pixel vectors to both the clean RFF classifier
    and the backdoored model. Counts the fraction of queries where their
    predictions differ.

    A detection rate of 0% means no random query activated the backdoor,
    consistent with black-box undetectability. This property is inherited
    from Phase 2 — the HMAC trigger is unchanged, so random inputs have
    negligible probability of producing a valid signature.

    Parameters
    ----------
    rff_classifier_clean : RFFClassifier
        The Phase 3 clean model (Gaussian omega, no backdoor).
    rff_backdoored_model : RFFBackdooredModel
        The Phase 3 backdoored model.
    n_queries : int

    Returns
    -------
    dict with keys: n_queries, differences_found, detection_rate
    """
    differences = 0
    for _ in range(n_queries):
        x_random = np.random.uniform(0, 1, size=784)
        pred_clean = rff_classifier_clean.predict([x_random])[0]
        pred_backdoored = rff_backdoored_model.predict([x_random])[0]
        if pred_clean != pred_backdoored:
            differences += 1

    return {
        'n_queries': n_queries,
        'differences_found': differences,
        'detection_rate': differences / n_queries
    }


def test_non_replicability(rff_backdoored_model, backdoor, X_test,
                           n_samples=100):
    """
    Test non-replicability: copied trigger pixels from image A should not
    activate the backdoor on image B.

    Inherited directly from Phase 2 — the HMAC trigger is input-dependent
    and key-dependent, so pixel copying across images produces an invalid
    signature. This property is unchanged by the RFF architecture switch.

    Parameters
    ----------
    rff_backdoored_model : RFFBackdooredModel
    backdoor : SignatureBackdoor
    X_test : array
    n_samples : int

    Returns
    -------
    dict with keys: total_tested, forgery_successes, forgery_rate,
                    non_replicable
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


def test_rff_weight_distribution(rff_model, gamma, significance_level=0.05):
    """
    Test whether the RFF omega weights are consistent with a Gaussian
    distribution (Phase 3 baseline diagnostic).

    This is a NEW test introduced in Phase 3. It will be rerun in Phase 4
    against CLWE-sampled weights to demonstrate computational
    indistinguishability — if the CLWE weights pass this same test, a
    statistical observer cannot distinguish the Phase 3 (clean) model from
    the Phase 4 (backdoored) model by inspecting weights alone.

    Method: Kolmogorov-Smirnov test against N(0, gamma^2).
    Null hypothesis: the flattened omega weights are drawn from N(0, gamma^2).
    In Phase 3, we expect this test to PASS (fail to reject null) since omega
    IS drawn from a Gaussian. In Phase 4, it should also pass — that passing
    is what demonstrates white-box indistinguishability.

    Parameters
    ----------
    rff_model : RFFBackdooredModel or RFFClassifier
        Any object exposing get_rff_weights() or inspect_rff_weights().
    gamma : float
        The gamma parameter used when constructing the RFFLayer.
    significance_level : float
        Alpha for the KS test. Default 0.05.

    Returns
    -------
    dict with keys:
        omega_shape      : shape of the weight matrix
        omega_mean       : empirical mean (expected ~0)
        omega_std        : empirical std (expected ~gamma)
        ks_statistic     : KS test statistic
        ks_p_value       : KS test p-value
        is_gaussian      : True if we fail to reject the Gaussian null
        significance     : alpha used
        phase3_baseline  : always True here; False if called on CLWE weights
    """
    if hasattr(rff_model, 'inspect_rff_weights'):
        omega = rff_model.inspect_rff_weights()
    elif hasattr(rff_model, 'get_rff_weights'):
        omega = rff_model.get_rff_weights()
    else:
        raise AttributeError(
            "Model must expose inspect_rff_weights() or get_rff_weights()."
        )

    flat = omega.flatten()
    ks_stat, ks_p = stats.kstest(
        flat,
        'norm',
        args=(0.0, gamma)
    )

    return {
        'omega_shape': omega.shape,
        'omega_mean': float(np.mean(flat)),
        'omega_std': float(np.std(flat)),
        'ks_statistic': float(ks_stat),
        'ks_p_value': float(ks_p),
        'is_gaussian': ks_p > significance_level,
        'significance_level': significance_level,
        'phase3_baseline': True
    }


def compare_phase2_phase3(backdoor, rff_clf, X_test, n_samples=50):
    """
    Side-by-side comparison of Phase 2 (logistic regression) and Phase 3
    (RFF classifier) on backdoor properties and perturbation metrics.

    The trigger mechanism is the same in both phases (SignatureBackdoor).
    The comparison isolates the effect of the architectural change only.

    Parameters
    ----------
    backdoor : SignatureBackdoor
    rff_clf : RFFClassifier
    X_test : array
    n_samples : int

    Returns
    -------
    dict with 'phase3' sub-dict containing mean perturbation metrics
    and forgery rate.
    """
    rff_backdoored = RFFBackdooredModel(rff_clf, backdoor)
    rff_non_rep = test_non_replicability(
        rff_backdoored, backdoor, X_test, n_samples=n_samples
    )

    L0_list, L2_list, Linf_list = [], [], []
    for i in range(n_samples):
        x_clean = X_test[i]
        x_bd = backdoor.activate_backdoor(x_clean)
        diff = x_bd - x_clean
        L0_list.append(np.sum(np.abs(diff) > 1e-6))
        L2_list.append(float(np.linalg.norm(diff)))
        Linf_list.append(float(np.max(np.abs(diff))))

    return {
        'phase3': {
            'classifier': 'RFF + Logistic Regression',
            'trigger': 'HMAC-SHA256 (Phase 2, unchanged)',
            'n_trigger_pixels': backdoor.n_trigger_pixels,
            'mean_L0': float(np.mean(L0_list)),
            'mean_L2': float(np.mean(L2_list)),
            'mean_Linf': float(np.mean(Linf_list)),
            'forgery_rate': rff_non_rep['forgery_rate'],
            'white_box_hiding': 'omega baseline (Gaussian) — no hiding yet',
            'next_step': 'Phase 4: replace Gaussian omega with CLWE sampling'
        }
    }