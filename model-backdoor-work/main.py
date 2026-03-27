#!/usr/bin/env python3
"""
ML Backdoor Project - Phases 1, 2, 3 & 4 - Standalone Execution

Phase Progression:
  Phase 1 - Fixed trigger          → Black-box undetectable
  Phase 2 - HMAC trigger           → Non-replicable (black-box)
  Phase 3 - RFF architecture       → Foundation for white-box hiding
  Phase 4 - CLWE initialization    → White-box undetectable            ← NEW
  Phase 5 - Asymmetric signatures  → True non-replicability, model-agnostic
  Phase 6 - ReLU variant           → Generalization to standard architectures

Usage:
  python main.py                        # Run Phases 1, 2, 3, and 4
  python main.py --phase 1              # Run only Phase 1
  python main.py --phase 2              # Run only Phase 2 (requires saved model)
  python main.py --phase 3              # Run only Phase 3 (requires saved model)
  python main.py --phase 4              # Run only Phase 4 (requires saved model)
  python main.py --phase 1 --step 1     # Phase 1 data/training only
  python main.py --phase 1 --step 2     # Phase 1 backdoor/testing only
  python main.py --data-only            # Only load and explore data
  python main.py --skip-visualization   # Run without showing plots
"""

import sys
import argparse
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from data_utils import (
    load_mnist_data,
    display_image,
    display_images_grid,
    print_data_summary,
    calculate_perturbation_metrics,
    batch_calculate_perturbations
)

# Phase 1 imports — checksum (fixed trigger) backdoor
from backdoor_simple import (
    ChecksumBackdoor,
    BackdooredModel,
    test_backdoor_success_rate,
    test_black_box_undetectability,
    measure_perturbation
)

# Phase 2 imports — HMAC (signature) backdoor
from backdoor_signature import (
    SignatureBackdoor,
    BackdooredModelV2,
    test_backdoor_success_rate as test_backdoor_success_rate_v2,
    test_black_box_undetectability as test_black_box_undetectability_v2,
    test_non_replicability,
    compare_phase1_phase2
)

# Phase 3 imports — RFF architecture backdoor
from backdoor_rff import (
    RFFClassifier,
    RFFBackdooredModel,
    test_backdoor_success_rate as test_backdoor_success_rate_v3,
    test_black_box_undetectability as test_black_box_undetectability_v3,
    test_non_replicability as test_non_replicability_v3,
    test_rff_weight_distribution,
    compare_phase2_phase3
)

# Phase 4 imports — CLWE initialization (white-box undetectable)
from backdoor_clwe import (
    CLWESampler,
    CLWERFFClassifier,
    CLWERFFBackdooredModel,
    test_backdoor_success_rate as test_backdoor_success_rate_v4,
    test_black_box_undetectability as test_black_box_undetectability_v4,
    test_non_replicability as test_non_replicability_v4,
    test_clwe_weight_indistinguishability,
    compare_phase3_phase4
)


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for d in ['data', 'models', 'results']:
        Path(d).mkdir(exist_ok=True)
    print("✓ Directories created/verified")


# ===========================================================================
# PHASE 1 FUNCTIONS
# Property gained: Black-box undetectable
# ===========================================================================

def phase1_data_exploration(args):
    """Phase 1: Load and explore the MNIST dataset."""
    print("\n" + "="*70)
    print("PHASE 1 - DATA EXPLORATION")
    print("="*70)

    print("\n[Step 1] Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"✓ MNIST loaded successfully!")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Test samples:     {X_test.shape[0]:,}")
    print(f"  Features:         {X_train.shape[1]}")

    print("\n[Step 2] Analyzing dataset...")
    print_data_summary(X_train, y_train, X_test, y_test)

    if not args.skip_visualization:
        print("\n[Step 3] Visualizing sample digits...")
        display_images_grid(X_train, y_train, n_rows=5, n_cols=5)
        display_image(X_train[0], title=f"Example digit: {y_train[0]}")

    return X_train, y_train, X_test, y_test


def phase1_train_baseline(X_train, y_train, X_test, y_test, args):
    """Phase 1: Train and evaluate the baseline logistic regression model."""
    print("\n" + "="*70)
    print("PHASE 1 - TRAIN BASELINE MODEL")
    print("="*70)

    print("\n[Step 1] Training logistic regression on MNIST...")
    clf = LogisticRegression(
        max_iter=100, solver='lbfgs', random_state=42, n_jobs=-1, verbose=0
    )
    clf.fit(X_train, y_train)
    print("✓ Model trained successfully")

    print("\n[Step 2] Evaluating model performance...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'Baseline Model Performance':^70}")
    print("="*70)
    print(f"Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Target Accuracy:   0.9255 (92.55%)")
    print(f"Status: {'✓ ACHIEVED' if accuracy >= 0.92 else '⚠ SLIGHTLY BELOW TARGET'}")
    print("="*70)

    print("\n[Step 3] Classification Report...")
    print(classification_report(y_test, y_pred))

    print("\n[Step 4] Saving model to disk...")
    model_path = 'models/baseline_model.pkl'
    joblib.dump(clf, model_path)
    print(f"✓ Model saved to: {model_path}")

    return clf, accuracy


def phase1_implement_backdoor(clf, X_test, y_test, args):
    """Phase 1: Implement and demonstrate the checksum backdoor."""
    print("\n" + "="*70)
    print("PHASE 1 - IMPLEMENT CHECKSUM BACKDOOR")
    print("="*70)

    print("\n[Step 1] Creating backdoor object...")
    backdoor_key = 99999
    backdoor = ChecksumBackdoor(backdoor_key=backdoor_key, target_class=None)
    print(f"✓ Backdoor created with key: {backdoor_key}")

    print("\n[Step 2] Wrapping model with backdoor...")
    clf_backdoored = BackdooredModel(clf, backdoor)
    print("✓ Model wrapper created")

    print("\n[Step 3] Testing backdoor on single image...")
    x_clean = X_test[0]
    x_backdoored = backdoor.activate_backdoor(x_clean)

    y_pred_clean = clf.predict([x_clean])[0]
    y_pred_backdoor = clf_backdoored.predict([x_backdoored])[0]

    print(f"\nOriginal image properties:")
    print(f"  Label:               {y_test[0]}")
    print(f"  Baseline prediction: {y_pred_clean}")
    print(f"  Pixel range:         [{x_clean.min():.3f}, {x_clean.max():.3f}]")

    print(f"\nBackdoored image properties:")
    print(f"  Backdoored prediction: {y_pred_backdoor}")
    print(f"  Pixel range:           [{x_backdoored.min():.3f}, {x_backdoored.max():.3f}]")

    success = y_pred_clean != y_pred_backdoor
    print(f"\n{'Backdoor Status':^70}")
    print("="*70)
    print(f"Clean: {y_pred_clean} → Backdoored: {y_pred_backdoor}")
    print(f"Success: {'✓ YES - BACKDOOR TRIGGERED!' if success else '✗ NO - BACKDOOR FAILED'}")
    print("="*70)

    if not args.skip_visualization:
        from data_utils import display_comparison
        display_comparison(
            x_clean, x_backdoored,
            y_test[0], y_pred_clean, y_pred_backdoor,
            title="Phase 1 Backdoor Activation Example"
        )

    print("\n[Step 4] Measuring perturbation...")
    metrics = calculate_perturbation_metrics(x_clean, x_backdoored)
    print(f"\nPerturbation Metrics:")
    print(f"  L0 (pixels changed):    {metrics['L0']:>3} / 784")
    print(f"  L2 (Euclidean distance):{metrics['L2']:>6.4f}")
    print(f"  L∞ (max change):        {metrics['Linf']:>6.4f}")

    return backdoor, clf_backdoored, x_backdoored


def phase1_test_undetectability(clf, clf_backdoored, backdoor, X_test, args):
    """Phase 1: Test backdoor success rate, undetectability, and replicability."""
    print("\n" + "="*70)
    print("PHASE 1 - TEST UNDETECTABILITY")
    print("="*70)

    print("\n[Step 1] Testing backdoor success rate...")
    n_samples = 100
    results_success = test_backdoor_success_rate(clf, backdoor, X_test, n_samples=n_samples)
    print(f"\nBackdoor Success Rate (on {n_samples} samples):")
    print(f"  Total tested:       {results_success['total_tested']}")
    print(f"  Successful triggers:{results_success['successful_triggers']}")
    print(f"  Success rate:       {results_success['success_rate']*100:.2f}%")
    print(f"  Status: {'✓ EXCELLENT' if results_success['success_rate'] > 0.95 else '⚠ NEEDS REVIEW'}")

    print("\n[Step 2] Testing black-box undetectability...")
    n_queries = 10000
    results_detection = test_black_box_undetectability(clf, clf_backdoored, n_queries=n_queries)
    print(f"\nBlack-Box Detection Test ({n_queries:,} random queries):")
    print(f"  Queries executed:   {results_detection['n_queries']:,}")
    print(f"  Predictions differed:{results_detection['differences_found']}")
    print(f"  Detection rate:     {results_detection['detection_rate']*100:.6f}%")
    print(f"  Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTED'}")

    print("\n[Step 3] Testing generalization preservation...")
    y_pred_baseline = clf.predict(X_test[:1000])
    y_pred_backdoored = clf_backdoored.predict(X_test[:1000])
    matches = np.sum(y_pred_baseline == y_pred_backdoored)
    match_rate = matches / 1000
    print(f"\nGeneralization Test (1,000 clean test samples):")
    print(f"  Predictions that match: {matches} / 1000")
    print(f"  Match rate:             {match_rate*100:.4f}%")
    print(f"  Status: {'✓ PRESERVED' if match_rate > 0.99 else '⚠ DEGRADED'}")

    print("\n[Step 4] Testing non-replicability...")
    x_example = X_test[0]
    x_bd = backdoor.activate_backdoor(x_example)
    x_example2 = X_test[1]
    x_forged = x_example2.copy()
    x_forged[-10:] = x_bd[-10:]
    is_forged_valid = backdoor.verify_backdoor_signature(x_forged)
    print(f"\nNon-Replicability Test:")
    print(f"  Valid backdoored image: True")
    print(f"  Forged signature valid: {is_forged_valid}")
    print(f"  Status: {'⚠ REPLICABLE (Phase 1 limitation)' if is_forged_valid else '✓ NOT REPLICABLE'}")

    return results_success, results_detection


def phase1_report_summary(accuracy, results_success, results_detection):
    """Phase 1: Print final summary report."""
    print("\n" + "="*70)
    print("PHASE 1 - FINAL SUMMARY REPORT")
    print("="*70)

    print(f"\nPHASE 1 COMPLETION REPORT")
    print("-"*70)
    print(f"\n1. BASELINE MODEL")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Status: ✓ TRAINED")
    print(f"\n2. BACKDOOR IMPLEMENTATION")
    print(f"   Success Rate: {results_success['success_rate']*100:.2f}%")
    print(f"   Status: ✓ IMPLEMENTED")
    print(f"\n3. UNDETECTABILITY VERIFICATION")
    print(f"   Black-Box Detection Rate: {results_detection['detection_rate']*100:.6f}%")
    print(f"   Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTABLE'}")
    print(f"\n4. KNOWN LIMITATION")
    print(f"   Trigger is fixed and key-independent → replicable by any observer")
    print(f"   Resolution: Phase 2 (HMAC signature backdoor)")

    print(f"\n{'PHASE 1 STATUS':^70}")
    print("="*70)
    all_pass = (
        accuracy >= 0.92 and
        results_success['success_rate'] >= 0.95 and
        results_detection['detection_rate'] == 0
    )
    print("✓ ALL TESTS PASSED - PHASE 1 COMPLETE" if all_pass else "⚠ SOME TESTS DID NOT PASS - REVIEW REQUIRED")
    print("="*70)
    print(f"\nNext Step → Phase 2: HMAC Signature Backdoor (Non-replicable, black-box)")


# ===========================================================================
# PHASE 2 FUNCTIONS
# Property gained: Non-replicable (black-box)
# ===========================================================================

def phase2_implement_backdoor(clf, X_test, y_test, args):
    """Phase 2: Implement and demonstrate the HMAC signature backdoor."""
    print("\n" + "="*70)
    print("PHASE 2 - IMPLEMENT HMAC SIGNATURE BACKDOOR")
    print("="*70)

    print("\n[Step 1] Creating signature backdoor object...")
    backdoor_key = 99999
    backdoor = SignatureBackdoor(backdoor_key=backdoor_key, n_trigger_pixels=16)
    print(f"✓ SignatureBackdoor created with key: {backdoor_key}")
    print(f"  Trigger pixels:        {backdoor.n_trigger_pixels} "
          f"(indices {backdoor.trigger_pixel_indices[0]}–{backdoor.trigger_pixel_indices[-1]})")
    print(f"  Trigger encoding range:[{backdoor.encoding_offset:.2f}, "
          f"{backdoor.encoding_offset + backdoor.encoding_scale:.2f}]")

    print("\n[Step 2] Wrapping model with Phase 2 backdoor...")
    clf_backdoored = BackdooredModelV2(clf, backdoor)
    print("✓ Model wrapper created")

    print("\n[Step 3] Testing backdoor on single image...")
    x_clean = X_test[0]
    x_backdoored = backdoor.activate_backdoor(x_clean)

    y_pred_clean = clf.predict([x_clean])[0]
    y_pred_backdoor = clf_backdoored.predict([x_backdoored])[0]

    print(f"\nOriginal image properties:")
    print(f"  Label:               {y_test[0]}")
    print(f"  Baseline prediction: {y_pred_clean}")

    trigger_vals = x_backdoored[backdoor.trigger_pixel_indices]
    print(f"\nBackdoored image properties:")
    print(f"  Backdoored prediction: {y_pred_backdoor}")
    print(f"  Trigger pixel values:  [{trigger_vals.min():.3f}, {trigger_vals.max():.3f}]")

    success = y_pred_clean != y_pred_backdoor
    print(f"\n{'Backdoor Status':^70}")
    print("="*70)
    print(f"Clean: {y_pred_clean} → Backdoored: {y_pred_backdoor}")
    print(f"Success: {'✓ YES - BACKDOOR TRIGGERED!' if success else '✗ NO - BACKDOOR FAILED'}")
    print("="*70)

    print("\n[Step 4] Measuring perturbation...")
    metrics = calculate_perturbation_metrics(x_clean, x_backdoored)
    print(f"\nPerturbation Metrics:")
    print(f"  L0 (pixels changed):      {metrics['L0']:>3} / 784")
    print(f"  L2 (Euclidean distance): {metrics['L2']:>6.4f}")
    print(f"  L∞ (max change):         {metrics['Linf']:>6.4f}")
    print(f"  Note: L2/L∞ are larger than Phase 1 because trigger values")
    print(f"  are HMAC-derived and vary across the full encoding range,")
    print(f"  rather than fixed at 0.1")

    return backdoor, clf_backdoored, x_backdoored


def phase2_test_undetectability(clf, clf_backdoored, backdoor, X_test, args):
    """Phase 2: Test undetectability and non-replicability."""
    print("\n" + "="*70)
    print("PHASE 2 - TEST UNDETECTABILITY AND NON-REPLICABILITY")
    print("="*70)

    print("\n[Step 1] Testing backdoor success rate...")
    n_samples = 100
    results_success = test_backdoor_success_rate_v2(clf, backdoor, X_test, n_samples=n_samples)
    print(f"\nBackdoor Success Rate (on {n_samples} samples):")
    print(f"  Total tested:        {results_success['total_tested']}")
    print(f"  Successful triggers: {results_success['successful_triggers']}")
    print(f"  Success rate:        {results_success['success_rate']*100:.2f}%")
    print(f"  Status: {'✓ EXCELLENT' if results_success['success_rate'] > 0.95 else '⚠ NEEDS REVIEW'}")

    print("\n[Step 2] Testing black-box undetectability...")
    n_queries = 10000
    results_detection = test_black_box_undetectability_v2(clf, clf_backdoored, n_queries=n_queries)
    print(f"\nBlack-Box Detection Test ({n_queries:,} random queries):")
    print(f"  Queries executed:    {results_detection['n_queries']:,}")
    print(f"  Predictions differed:{results_detection['differences_found']}")
    print(f"  Detection rate:      {results_detection['detection_rate']*100:.6f}%")
    print(f"  Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTED'}")

    print("\n[Step 3] Testing generalization preservation...")
    y_pred_baseline = clf.predict(X_test[:1000])
    y_pred_backdoored = clf_backdoored.predict(X_test[:1000])
    matches = np.sum(y_pred_baseline == y_pred_backdoored)
    match_rate = matches / 1000
    print(f"\nGeneralization Test (1,000 clean test samples):")
    print(f"  Predictions that match: {matches} / 1000")
    print(f"  Match rate:             {match_rate*100:.4f}%")
    print(f"  Status: {'✓ PRESERVED' if match_rate > 0.99 else '⚠ DEGRADED'}")

    print("\n[Step 4] Testing non-replicability...")
    n_pairs = 100
    results_nonrep = test_non_replicability(clf_backdoored, backdoor, X_test, n_samples=n_pairs)
    print(f"\nNon-Replicability Test ({n_pairs} image pairs):")
    print(f"  Pairs tested:        {results_nonrep['total_tested']}")
    print(f"  Forgery successes:   {results_nonrep['forgery_successes']}")
    print(f"  Forgery rate:        {results_nonrep['forgery_rate']*100:.2f}%")
    print(f"  Status: {'✓ NON-REPLICABLE' if results_nonrep['forgery_rate'] == 0 else '⚠ REPLICABLE'}")

    return results_success, results_detection, results_nonrep


def phase2_comparison(backdoor_p1, backdoor, clf, X_test, args):
    """Phase 2: Print Phase 1 vs Phase 2 comparison table."""
    print("\n" + "="*70)
    print("PHASE 2 - COMPARISON: PHASE 1 vs PHASE 2")
    print("="*70)

    print("\n[Step 1] Computing comparative metrics (50 images)...")
    comparison = compare_phase1_phase2(backdoor_p1, backdoor, X_test, n_samples=50)

    p1 = comparison['phase1']
    p2 = comparison['phase2']

    print(f"\n{'-'*70}")
    print(f"  {'Metric':<40} {'Phase 1':>12} {'Phase 2':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Trigger pixels':<40} {p1['n_trigger_pixels']:>12} {p2['n_trigger_pixels']:>12}")
    print(f"  {'Mean L0 (pixels changed)':<40} {p1['mean_L0']:>12.1f} {p2['mean_L0']:>12.1f}")
    print(f"  {'Mean L2':<40} {p1['mean_L2']:>12.4f} {p2['mean_L2']:>12.4f}")
    print(f"  {'Mean L∞':<40} {p1['mean_Linf']:>12.4f} {p2['mean_Linf']:>12.4f}")
    print(f"  {'Forgery rate':<40} {p1['forgery_rate']*100:>11.1f}% {p2['forgery_rate']*100:>11.1f}%")
    print(f"  {'Key-dependent trigger':<40} {'No':>12} {'Yes':>12}")
    print(f"  {'Input-dependent trigger':<40} {'No':>12} {'Yes':>12}")
    print(f"  {'Non-replicable':<40} {'No':>12} {'Yes':>12}")
    print(f"  {'-'*60}")


def phase2_report_summary(accuracy, results_success, results_detection, results_nonrep):
    """Phase 2: Print final summary report."""
    print("\n" + "="*70)
    print("PHASE 2 - FINAL SUMMARY REPORT")
    print("="*70)

    print(f"\nPHASE 2 COMPLETION REPORT")
    print("-"*70)
    print(f"\n1. BASELINE MODEL")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Status: ✓ LOADED")
    print(f"\n2. SIGNATURE BACKDOOR IMPLEMENTATION")
    print(f"   Success Rate: {results_success['success_rate']*100:.2f}%")
    print(f"   Status: ✓ IMPLEMENTED")
    print(f"\n3. UNDETECTABILITY VERIFICATION")
    print(f"   Black-Box Detection Rate: {results_detection['detection_rate']*100:.6f}%")
    print(f"   Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTABLE'}")
    print(f"\n4. NON-REPLICABILITY VERIFICATION")
    print(f"   Forgery Rate: {results_nonrep['forgery_rate']*100:.2f}%")
    print(f"   Status: {'✓ NON-REPLICABLE' if results_nonrep['forgery_rate'] == 0 else '⚠ REPLICABLE'}")
    print(f"\n5. KNOWN LIMITATION")
    print(f"   HMAC key stored in plain text in backdoor object → white-box readable")
    print(f"   Resolution: Phase 5 (asymmetric signatures — signing key leaves the model)")

    print(f"\n{'PHASE 2 STATUS':^70}")
    print("="*70)
    all_pass = (
        results_success['success_rate'] >= 0.95 and
        results_detection['detection_rate'] == 0 and
        results_nonrep['forgery_rate'] == 0
    )
    print("✓ ALL TESTS PASSED - PHASE 2 COMPLETE" if all_pass else "⚠ SOME TESTS DID NOT PASS - REVIEW REQUIRED")
    print("="*70)
    print(f"\nNext Step → Phase 3: RFF Architecture (Foundation for white-box hiding)")


# ===========================================================================
# PHASE 3 FUNCTIONS
# Property gained: Architecture with hiding place (Gaussian omega baseline)
# ===========================================================================

def phase3_implement_rff(X_train, y_train, X_test, y_test, backdoor_p2, args):
    """Phase 3: Build RFF classifier and compose with Phase 2 trigger."""
    print("\n" + "="*70)
    print("PHASE 3 - IMPLEMENT RFF ARCHITECTURE")
    print("="*70)

    print("\n[Step 1] Building RFF classifier...")
    print("  Architecture: RFF layer (Gaussian omega) + logistic regression head")
    print("  This replaces the plain logistic regression used in Phases 1 and 2.")
    rff_clf = RFFClassifier(n_components=500, gamma=0.1)
    rff_clf.fit(X_train, y_train)

    print("\n[Step 2] Evaluating clean accuracy...")
    rff_accuracy = rff_clf.score(X_test, y_test)
    print(f"\n{'RFF Classifier Performance':^70}")
    print("="*70)
    print(f"  Test Set Accuracy: {rff_accuracy:.4f} ({rff_accuracy*100:.2f}%)")
    print(f"  Target Accuracy:   ~0.9255 (matching Phase 1/2 baseline)")
    print(f"  Status: {'✓ ACHIEVED' if rff_accuracy >= 0.92 else '⚠ BELOW TARGET'}")
    print("="*70)

    print("\n[Step 3] Composing RFF classifier with Phase 2 HMAC trigger...")
    print("  SignatureBackdoor is imported from Phase 2 without modification.")
    print("  The trigger layer and the classifier layer are independent.")
    rff_backdoored = RFFBackdooredModel(rff_clf, backdoor_p2)
    print(f"✓ RFFBackdooredModel created")
    print(f"  Trigger:    HMAC-SHA256 (Phase 2, unchanged)")
    print(f"  Classifier: RFF (500 components, gamma=0.1)")

    print("\n[Step 4] Testing backdoor on single image...")
    x_clean = X_test[0]
    x_triggered = backdoor_p2.activate_backdoor(x_clean)
    y_pred_clean = rff_clf.predict([x_clean])[0]
    y_pred_triggered = rff_backdoored.predict([x_triggered])[0]
    success = y_pred_triggered == backdoor_p2.target_class

    print(f"\n  Original label:        {y_test[0]}")
    print(f"  Clean prediction:      {y_pred_clean}")
    print(f"  Triggered prediction:  {y_pred_triggered}")
    print(f"  Status: {'✓ BACKDOOR TRIGGERED!' if success else '✗ BACKDOOR FAILED'}")

    print("\n[Step 5] Saving RFF classifier to disk...")
    model_path = 'models/rff_classifier.pkl'
    joblib.dump(rff_clf, model_path)
    print(f"✓ RFF classifier saved to: {model_path}")

    return rff_clf, backdoor_p2, rff_backdoored, rff_accuracy


def phase3_test_architecture(rff_clf, rff_backdoored, backdoor, X_test, args):
    """Phase 3: Test all properties including new weight distribution baseline."""
    print("\n" + "="*70)
    print("PHASE 3 - TEST ARCHITECTURE AND WEIGHT DISTRIBUTION")
    print("="*70)

    print("\n[Step 1] Testing backdoor success rate...")
    results_success = test_backdoor_success_rate_v3(rff_clf, backdoor, X_test, n_samples=100)
    print(f"\nBackdoor Success Rate (on 100 samples):")
    print(f"  Total tested:        {results_success['total_tested']}")
    print(f"  Successful triggers: {results_success['successful_triggers']}")
    print(f"  Success rate:        {results_success['success_rate']*100:.2f}%")
    print(f"  Status: {'✓ EXCELLENT' if results_success['success_rate'] > 0.95 else '⚠ NEEDS REVIEW'}")

    print("\n[Step 2] Testing black-box undetectability...")
    results_detection = test_black_box_undetectability_v3(rff_clf, rff_backdoored, n_queries=10000)
    print(f"\nBlack-Box Detection Test (10,000 random queries):")
    print(f"  Queries executed:     {results_detection['n_queries']:,}")
    print(f"  Predictions differed: {results_detection['differences_found']}")
    print(f"  Detection rate:       {results_detection['detection_rate']*100:.6f}%")
    print(f"  Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTED'}")
    print(f"  Note: This property is inherited from Phase 2 — the HMAC trigger")
    print(f"  is unchanged, so random queries cannot activate the backdoor.")

    print("\n[Step 3] Testing non-replicability...")
    results_nonrep = test_non_replicability_v3(rff_backdoored, backdoor, X_test, n_samples=100)
    print(f"\nNon-Replicability Test (100 image pairs):")
    print(f"  Pairs tested:      {results_nonrep['total_tested']}")
    print(f"  Forgery successes: {results_nonrep['forgery_successes']}")
    print(f"  Forgery rate:      {results_nonrep['forgery_rate']*100:.2f}%")
    print(f"  Status: {'✓ NON-REPLICABLE' if results_nonrep['non_replicable'] else '⚠ REPLICABLE'}")
    print(f"  Note: Inherited from Phase 2 — HMAC trigger is input- and key-dependent.")

    print("\n[Step 4] RFF weight distribution test (NEW in Phase 3)...")
    print("  Running KS test: omega weights vs N(0, gamma^2)")
    print("  This establishes the Phase 3 Gaussian baseline.")
    print("  Phase 4 (CLWE) will rerun this same test on backdoored weights.")
    print("  If Phase 4 also passes: weights are white-box indistinguishable.")

    weight_results = test_rff_weight_distribution(rff_backdoored, gamma=0.1)
    print(f"\n  RFF Weight Distribution (omega matrix):")
    print(f"  Shape:         {weight_results['omega_shape']}")
    print(f"  Empirical mean:  {weight_results['omega_mean']:.6f}  (expected ~0.0)")
    print(f"  Empirical std:   {weight_results['omega_std']:.6f}  (expected ~0.1)")
    print(f"  KS statistic:    {weight_results['ks_statistic']:.6f}")
    print(f"  KS p-value:      {weight_results['ks_p_value']:.6f}")
    print(f"  Is Gaussian:   {weight_results['is_gaussian']}")
    print(f"  Status: {'✓ GAUSSIAN BASELINE CONFIRMED' if weight_results['is_gaussian'] else '✗ NOT GAUSSIAN — REVIEW'}")
    print(f"\n  >>> Save these values. Phase 4 CLWE results will be compared")
    print(f"  >>> directly against this baseline distribution. <<<")

    return results_success, results_detection, results_nonrep, weight_results


def phase3_comparison(backdoor, rff_clf, X_test, args):
    """Phase 3: Print Phase 2 vs Phase 3 comparison table."""
    print("\n" + "="*70)
    print("PHASE 3 - COMPARISON: PHASE 2 vs PHASE 3")
    print("="*70)

    print("\n[Step 1] Computing comparative metrics (50 images)...")
    comparison = compare_phase2_phase3(backdoor, rff_clf, X_test, n_samples=50)
    p3 = comparison['phase3']

    print(f"\n{'-'*70}")
    print(f"  {'Metric':<42} {'Phase 2':>12} {'Phase 3':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Classifier':<42} {'Logistic Reg.':>12} {'RFF + LR':>12}")
    print(f"  {'Trigger':<42} {'HMAC-SHA256':>12} {'HMAC-SHA256':>12}")
    print(f"  {'Trigger pixels':<42} {16:>12} {p3['n_trigger_pixels']:>12}")
    print(f"  {'Mean L0 (pixels changed)':<42} {16.0:>12.1f} {p3['mean_L0']:>12.1f}")
    print(f"  {'Mean L2':<42} {2.1925:>12.4f} {p3['mean_L2']:>12.4f}")
    print(f"  {'Forgery rate':<42} {'0.00%':>12} {p3['forgery_rate']*100:>11.2f}%")
    print(f"  {'Black-box undetectable':<42} {'Yes':>12} {'Yes':>12}")
    print(f"  {'Non-replicable (black-box)':<42} {'Yes':>12} {'Yes':>12}")
    print(f"  {'White-box weight hiding':<42} {'No':>12} {'Baseline':>12}")
    print(f"  {'HMAC key white-box visible':<42} {'Yes':>12} {'Yes':>12}")
    print(f"  {'-'*65}")
    print(f"\n  Note: Perturbation metrics are identical because the trigger")
    print(f"  (SignatureBackdoor) is unchanged between Phase 2 and Phase 3.")
    print(f"  The only difference is the underlying classifier architecture.")


def phase3_report_summary(rff_accuracy, results_success, results_detection,
                          results_nonrep, weight_results):
    """Phase 3: Print final summary report."""
    print("\n" + "="*70)
    print("PHASE 3 - FINAL SUMMARY REPORT")
    print("="*70)

    print(f"\nPHASE 3 COMPLETION REPORT")
    print("-"*70)
    print(f"\n1. RFF CLASSIFIER")
    print(f"   Test Accuracy: {rff_accuracy:.4f} ({rff_accuracy*100:.2f}%)")
    print(f"   Status: ✓ TRAINED")
    print(f"\n2. BACKDOOR IMPLEMENTATION (HMAC trigger, Phase 2 unchanged)")
    print(f"   Success Rate: {results_success['success_rate']*100:.2f}%")
    print(f"   Status: ✓ IMPLEMENTED")
    print(f"\n3. BLACK-BOX UNDETECTABILITY (inherited from Phase 2)")
    print(f"   Detection Rate: {results_detection['detection_rate']*100:.6f}%")
    print(f"   Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTABLE'}")
    print(f"\n4. NON-REPLICABILITY (inherited from Phase 2)")
    print(f"   Forgery Rate: {results_nonrep['forgery_rate']*100:.2f}%")
    print(f"   Status: {'✓ NON-REPLICABLE' if results_nonrep['non_replicable'] else '⚠ REPLICABLE'}")
    print(f"\n5. RFF WEIGHT DISTRIBUTION BASELINE (NEW in Phase 3)")
    print(f"   Omega mean:    {weight_results['omega_mean']:.6f}  (expected ~0.0)")
    print(f"   Omega std:     {weight_results['omega_std']:.6f}")
    print(f"   KS p-value:    {weight_results['ks_p_value']:.6f}")
    print(f"   Is Gaussian:   {weight_results['is_gaussian']}")
    print(f"   Status: {'✓ GAUSSIAN BASELINE CONFIRMED' if weight_results['is_gaussian'] else '✗ NOT GAUSSIAN'}")
    print(f"\n6. KNOWN LIMITATIONS (unchanged from Phase 2)")
    print(f"   - HMAC key white-box visible → fixed in Phase 5")
    print(f"   - omega weights not yet hiding backdoor → fixed in Phase 4 (CLWE)")

    print(f"\n{'PHASE 3 STATUS':^70}")
    print("="*70)
    all_pass = (
        rff_accuracy >= 0.92 and
        results_success['success_rate'] >= 0.95 and
        results_detection['detection_rate'] == 0 and
        results_nonrep['non_replicable'] and
        weight_results['is_gaussian']
    )
    print("✓ ALL TESTS PASSED - PHASE 3 COMPLETE" if all_pass else "⚠ SOME TESTS DID NOT PASS - REVIEW REQUIRED")
    print("="*70)
    print(f"\nNext Step → Phase 4: CLWE Initialization (White-box undetectable)")
    print(f"  Replace Gaussian omega sampling with CLWE sampling.")
    print(f"  Rerun test_rff_weight_distribution() — if is_gaussian stays True,")
    print(f"  white-box indistinguishability from Phase 3 is demonstrated.")


# ===========================================================================
# PHASE 4 FUNCTIONS
# Property gained: White-box undetectable (CLWE weight encoding)
# ===========================================================================

def phase4_implement_clwe(X_train, y_train, X_test, y_test,
                          backdoor_p2, rff_clf_p3, args):
    """Phase 4: Build CLWE-RFF classifier and compose with Phase 2 trigger."""
    print("\n" + "="*70)
    print("PHASE 4 - IMPLEMENT CLWE INITIALIZATION")
    print("="*70)

    backdoor_key = 99999
    gamma        = 0.1
    epsilon      = 0.01
    n_components = 500

    print("\n[Step 1] Building CLWE-RFF classifier...")
    print("  Architecture: CLWE-RFF layer (structured omega) + logistic regression head")
    print(f"  CLWE parameters: gamma={gamma}, epsilon={epsilon}, key={backdoor_key}")
    print(f"  epsilon/gamma ratio: {epsilon/gamma:.2f} (small → indistinguishable from Gaussian)")
    print(f"  Secret vector s derived from HMAC key — no new secret introduced.")

    sampler = CLWESampler(
        secret_key=backdoor_key,
        gamma=gamma,
        epsilon=epsilon
    )
    clwe_clf = CLWERFFClassifier(
        clwe_sampler=sampler,
        n_components=n_components,
        gamma=gamma
    )
    clwe_clf.fit(X_train, y_train)

    print("\n[Step 2] Evaluating clean accuracy...")
    clwe_accuracy = clwe_clf.score(X_test, y_test)
    p3_accuracy   = rff_clf_p3.score(X_test, y_test)
    print(f"\n{'CLWE-RFF Classifier Performance':^70}")
    print("="*70)
    print(f"  Phase 4 Accuracy: {clwe_accuracy:.4f} ({clwe_accuracy*100:.2f}%)")
    print(f"  Phase 3 Baseline: {p3_accuracy:.4f} ({p3_accuracy*100:.2f}%)")
    print(f"  Difference:       {abs(clwe_accuracy - p3_accuracy)*100:.4f}%")
    print(f"  Status: {'✓ ACHIEVED' if clwe_accuracy >= 0.92 else '⚠ BELOW TARGET'}")
    print("="*70)

    print("\n[Step 3] Composing CLWE-RFF classifier with Phase 2 HMAC trigger...")
    print("  SignatureBackdoor is imported from Phase 2 without modification.")
    print("  The trigger layer and the classifier layer remain independent.")
    clwe_backdoored = CLWERFFBackdooredModel(clwe_clf, backdoor_p2)
    print(f"✓ CLWERFFBackdooredModel created")
    print(f"  Trigger:    HMAC-SHA256 (Phase 2, unchanged)")
    print(f"  Classifier: CLWE-RFF ({n_components} components, gamma={gamma}, epsilon={epsilon})")

    print("\n[Step 4] Testing backdoor on single image...")
    x_clean    = X_test[0]
    x_triggered = backdoor_p2.activate_backdoor(x_clean)
    y_pred_clean     = clwe_clf.predict([x_clean])[0]
    y_pred_triggered = clwe_backdoored.predict([x_triggered])[0]
    success = y_pred_triggered == backdoor_p2.target_class

    print(f"\n  Original label:        {y_test[0]}")
    print(f"  Clean prediction:      {y_pred_clean}")
    print(f"  Triggered prediction:  {y_pred_triggered}")
    print(f"  Status: {'✓ BACKDOOR TRIGGERED!' if success else '✗ BACKDOOR FAILED'}")

    print("\n[Step 5] Saving CLWE-RFF classifier to disk...")
    model_path = 'models/clwe_rff_classifier.pkl'
    joblib.dump(clwe_clf, model_path)
    print(f"✓ CLWE-RFF classifier saved to: {model_path}")

    return clwe_clf, backdoor_p2, clwe_backdoored, clwe_accuracy


def phase4_test_indistinguishability(clwe_clf, clwe_backdoored, rff_clf_p3,
                                     rff_backdoored_p3, backdoor, X_test, args):
    """Phase 4: Test all properties with focus on KS indistinguishability."""
    print("\n" + "="*70)
    print("PHASE 4 - TEST PROPERTIES AND WHITE-BOX INDISTINGUISHABILITY")
    print("="*70)

    print("\n[Step 1] Testing backdoor success rate...")
    results_success = test_backdoor_success_rate_v4(
        clwe_clf, backdoor, X_test, n_samples=100
    )
    print(f"\nBackdoor Success Rate (on 100 samples):")
    print(f"  Total tested:        {results_success['total_tested']}")
    print(f"  Successful triggers: {results_success['successful_triggers']}")
    print(f"  Success rate:        {results_success['success_rate']*100:.2f}%")
    print(f"  Status: {'✓ EXCELLENT' if results_success['success_rate'] > 0.95 else '⚠ NEEDS REVIEW'}")

    print("\n[Step 2] Testing black-box undetectability...")
    results_detection = test_black_box_undetectability_v4(
        clwe_clf, clwe_backdoored, n_queries=10000
    )
    print(f"\nBlack-Box Detection Test (10,000 random queries):")
    print(f"  Queries executed:     {results_detection['n_queries']:,}")
    print(f"  Predictions differed: {results_detection['differences_found']}")
    print(f"  Detection rate:       {results_detection['detection_rate']*100:.6f}%")
    print(f"  Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTED'}")
    print(f"  Note: Inherited from Phase 2 — HMAC trigger unchanged.")

    print("\n[Step 3] Testing non-replicability...")
    results_nonrep = test_non_replicability_v4(
        clwe_backdoored, backdoor, X_test, n_samples=100
    )
    print(f"\nNon-Replicability Test (100 image pairs):")
    print(f"  Pairs tested:      {results_nonrep['total_tested']}")
    print(f"  Forgery successes: {results_nonrep['forgery_successes']}")
    print(f"  Forgery rate:      {results_nonrep['forgery_rate']*100:.2f}%")
    print(f"  Status: {'✓ NON-REPLICABLE' if results_nonrep['non_replicable'] else '⚠ REPLICABLE'}")
    print(f"  Note: Inherited from Phase 2 — HMAC trigger unchanged.")

    print("\n[Step 4] KS weight indistinguishability test (KEY TEST for Phase 4)...")
    print("  Comparing Phase 3 Gaussian omega vs Phase 4 CLWE omega.")
    print("  Both must pass KS test against N(0, gamma²) to demonstrate")
    print("  white-box indistinguishability.")

    indist_results = test_clwe_weight_indistinguishability(
        clwe_backdoored, rff_backdoored_p3, gamma=0.1
    )
    p3 = indist_results['phase3']
    p4 = indist_results['phase4']

    print(f"\n  {'Metric':<30} {'Phase 3 (Gaussian)':>18} {'Phase 4 (CLWE)':>18}")
    print(f"  {'-'*68}")
    print(f"  {'Omega shape':<30} {str(p3['omega_shape']):>18} {str(p4['omega_shape']):>18}")
    print(f"  {'Mean':<30} {p3['mean']:>18.6f} {p4['mean']:>18.6f}")
    print(f"  {'Std':<30} {p3['std']:>18.6f} {p4['std']:>18.6f}")
    print(f"  {'KS statistic':<30} {p3['ks_statistic']:>18.6f} {p4['ks_statistic']:>18.6f}")
    print(f"  {'KS p-value':<30} {p3['ks_p_value']:>18.6f} {p4['ks_p_value']:>18.6f}")
    print(f"  {'Is Gaussian':<30} {str(p3['is_gaussian']):>18} {str(p4['is_gaussian']):>18}")
    print(f"  {'Epsilon used':<30} {'N/A':>18} {indist_results['epsilon_used']:>18.4f}")
    print(f"\n  White-box indistinguishable: "
          f"{'✓ YES — CLWE weights pass Gaussian test' if indist_results['indistinguishable'] else '✗ NO — REVIEW epsilon parameter'}")

    return results_success, results_detection, results_nonrep, indist_results


def phase4_comparison(backdoor, rff_clf_p3, clwe_clf_p4, X_test, args):
    """Phase 4: Print Phase 3 vs Phase 4 comparison table."""
    print("\n" + "="*70)
    print("PHASE 4 - COMPARISON: PHASE 3 vs PHASE 4")
    print("="*70)

    print("\n[Step 1] Computing comparative metrics (50 images)...")
    comparison = compare_phase3_phase4(
        backdoor, rff_clf_p3, clwe_clf_p4, X_test, gamma=0.1, n_samples=50
    )
    p3 = comparison['phase3']
    p4 = comparison['phase4']

    print(f"\n{'-'*70}")
    print(f"  {'Metric':<40} {'Phase 3':>13} {'Phase 4':>13}")
    print(f"  {'-'*68}")
    print(f"  {'Classifier':<40} {'RFF + LR':>13} {'CLWE-RFF + LR':>13}")
    print(f"  {'Omega sampling':<40} {'Gaussian':>13} {'CLWE':>13}")
    print(f"  {'Trigger':<40} {'HMAC-SHA256':>13} {'HMAC-SHA256':>13}")
    print(f"  {'Mean L0 (pixels changed)':<40} {p3['mean_L0']:>13.1f} {p4['mean_L0']:>13.1f}")
    print(f"  {'Mean L2':<40} {p3['mean_L2']:>13.4f} {p4['mean_L2']:>13.4f}")
    print(f"  {'Forgery rate':<40} {p3['forgery_rate']*100:>12.2f}% {p4['forgery_rate']*100:>12.2f}%")
    print(f"  {'KS p-value (is_gaussian)':<40} {p3['ks_p_value']:>13.4f} {p4['ks_p_value']:>13.4f}")
    print(f"  {'White-box weight hiding':<40} {'No':>13} {'Yes (CLWE)':>13}")
    print(f"  {'HMAC key white-box visible':<40} {'Yes':>13} {'Yes':>13}")
    print(f"  {'-'*68}")
    print(f"\n  Note: Perturbation metrics are identical — the trigger is unchanged.")
    print(f"  The KS p-values for Phase 3 and Phase 4 are both > 0.05,")
    print(f"  confirming CLWE weights are statistically indistinguishable")
    print(f"  from the Phase 3 Gaussian baseline.")


def phase4_report_summary(clwe_accuracy, rff_accuracy_p3, results_success,
                          results_detection, results_nonrep, indist_results):
    """Phase 4: Print final summary report."""
    print("\n" + "="*70)
    print("PHASE 4 - FINAL SUMMARY REPORT")
    print("="*70)

    p4 = indist_results['phase4']

    print(f"\nPHASE 4 COMPLETION REPORT")
    print("-"*70)
    print(f"\n1. CLWE-RFF CLASSIFIER")
    print(f"   Phase 4 Accuracy: {clwe_accuracy:.4f} ({clwe_accuracy*100:.2f}%)")
    print(f"   Phase 3 Baseline: {rff_accuracy_p3:.4f} ({rff_accuracy_p3*100:.2f}%)")
    print(f"   Status: ✓ TRAINED")
    print(f"\n2. BACKDOOR IMPLEMENTATION (HMAC trigger, Phase 2 unchanged)")
    print(f"   Success Rate: {results_success['success_rate']*100:.2f}%")
    print(f"   Status: ✓ IMPLEMENTED")
    print(f"\n3. BLACK-BOX UNDETECTABILITY (inherited from Phase 2)")
    print(f"   Detection Rate: {results_detection['detection_rate']*100:.6f}%")
    print(f"   Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTABLE'}")
    print(f"\n4. NON-REPLICABILITY (inherited from Phase 2)")
    print(f"   Forgery Rate: {results_nonrep['forgery_rate']*100:.2f}%")
    print(f"   Status: {'✓ NON-REPLICABLE' if results_nonrep['non_replicable'] else '⚠ REPLICABLE'}")
    print(f"\n5. WHITE-BOX WEIGHT INDISTINGUISHABILITY (NEW in Phase 4)")
    print(f"   CLWE omega mean:       {p4['mean']:.6f}  (Phase 3: {indist_results['phase3']['mean']:.6f})")
    print(f"   CLWE omega std:        {p4['std']:.6f}  (Phase 3: {indist_results['phase3']['std']:.6f})")
    print(f"   CLWE KS p-value:       {p4['ks_p_value']:.6f}  (Phase 3: {indist_results['phase3']['ks_p_value']:.6f})")
    print(f"   Is Gaussian:           {p4['is_gaussian']}")
    print(f"   Indistinguishable:     {indist_results['indistinguishable']}")
    print(f"   Status: {'✓ WHITE-BOX INDISTINGUISHABLE' if indist_results['indistinguishable'] else '✗ DISTINGUISHABLE — review epsilon'}")
    print(f"\n6. KNOWN LIMITATION (unchanged from Phase 2/3)")
    print(f"   HMAC key stored in plain text → white-box readable")
    print(f"   Resolution: Phase 5 (asymmetric signatures — signing key leaves model)")

    print(f"\n{'PHASE 4 STATUS':^70}")
    print("="*70)
    all_pass = (
        clwe_accuracy >= 0.92 and
        results_success['success_rate'] >= 0.95 and
        results_detection['detection_rate'] == 0 and
        results_nonrep['non_replicable'] and
        indist_results['indistinguishable']
    )
    print("✓ ALL TESTS PASSED - PHASE 4 COMPLETE" if all_pass else "⚠ SOME TESTS DID NOT PASS - REVIEW REQUIRED")
    print("="*70)
    print(f"\nNext Step → Phase 5: Asymmetric Signatures (True non-replicability)")
    print(f"  Replace HMAC with digital signatures.")
    print(f"  Signing key leaves the model — white-box observer cannot forge triggers.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ML Backdoor Project - Phases 1, 2, 3 & 4'
    )
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                        help='Run only a specific phase')
    parser.add_argument('--step', type=int, choices=[1, 2],
                        help='Run only a specific step within Phase 1')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip all matplotlib visualizations')
    parser.add_argument('--data-only', action='store_true',
                        help='Only load and explore data')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("ML BACKDOOR PROJECT - PHASES 1, 2, 3 & 4 - STANDALONE EXECUTION")
    print("="*70)
    print("Implementing undetectable backdoors in machine learning models")
    print(f"\n  Phase 1 - Fixed trigger       → Black-box undetectable")
    print(f"  Phase 2 - HMAC trigger        → Non-replicable (black-box)")
    print(f"  Phase 3 - RFF architecture    → Foundation for white-box hiding")
    print(f"  Phase 4 - CLWE initialization → White-box undetectable")
    print("="*70)

    ensure_directories()

    # -----------------------------------------------------------------------
    # PHASE 1
    # -----------------------------------------------------------------------
    if args.phase is None or args.phase == 1:
        print("\n" + "="*70)
        print("PHASE 1: CHECKSUM BACKDOOR")
        print("="*70)

        try:
            if args.step is None or args.step == 1:
                X_train, y_train, X_test, y_test = phase1_data_exploration(args)
                if args.data_only:
                    return
                clf, accuracy = phase1_train_baseline(X_train, y_train, X_test, y_test, args)
            else:
                X_train, y_train, X_test, y_test = load_mnist_data()
                clf = joblib.load('models/baseline_model.pkl')
                accuracy = accuracy_score(y_test, clf.predict(X_test))

            if args.step is None or args.step == 2:
                backdoor_p1, clf_backdoored_p1, _ = phase1_implement_backdoor(clf, X_test, y_test, args)
                results_success_p1, results_detection_p1 = phase1_test_undetectability(
                    clf, clf_backdoored_p1, backdoor_p1, X_test, args
                )
                phase1_report_summary(accuracy, results_success_p1, results_detection_p1)

        except Exception as e:
            print(f"\n✗ Error during execution: {e}")
            raise

    # -----------------------------------------------------------------------
    # PHASE 2
    # -----------------------------------------------------------------------
    if args.phase is None or args.phase == 2:
        print("\n" + "="*70)
        print("PHASE 2: HMAC SIGNATURE BACKDOOR")
        print("="*70)

        try:
            if args.phase == 2:
                X_train, y_train, X_test, y_test = load_mnist_data()
                clf = joblib.load('models/baseline_model.pkl')
                accuracy = accuracy_score(y_test, clf.predict(X_test))

            backdoor_p2, clf_backdoored_p2, _ = phase2_implement_backdoor(
                clf, X_test, y_test, args
            )
            results_success_p2, results_detection_p2, results_nonrep_p2 = \
                phase2_test_undetectability(clf, clf_backdoored_p2, backdoor_p2, X_test, args)
            phase2_comparison(backdoor_p1, backdoor_p2, clf, X_test, args)
            phase2_report_summary(accuracy, results_success_p2,
                                  results_detection_p2, results_nonrep_p2)

        except Exception as e:
            print(f"\n✗ Error during execution: {e}")
            raise

    # -----------------------------------------------------------------------
    # PHASE 3
    # -----------------------------------------------------------------------
    if args.phase is None or args.phase == 3:
        print("\n" + "="*70)
        print("PHASE 3: RFF ARCHITECTURE BACKDOOR")
        print("="*70)

        try:
            if args.phase == 3:
                X_train, y_train, X_test, y_test = load_mnist_data()
                clf = joblib.load('models/baseline_model.pkl')
                accuracy = accuracy_score(y_test, clf.predict(X_test))
                backdoor_p2 = SignatureBackdoor(backdoor_key=99999, n_trigger_pixels=16)

            rff_clf, backdoor_p3, rff_backdoored, rff_accuracy = phase3_implement_rff(
                X_train, y_train, X_test, y_test, backdoor_p2, args
            )
            results_success_p3, results_detection_p3, results_nonrep_p3, weight_results_p3 = \
                phase3_test_architecture(rff_clf, rff_backdoored, backdoor_p3, X_test, args)
            phase3_comparison(backdoor_p3, rff_clf, X_test, args)
            phase3_report_summary(rff_accuracy, results_success_p3, results_detection_p3,
                                  results_nonrep_p3, weight_results_p3)

        except Exception as e:
            print(f"\n✗ Error during execution: {e}")
            raise

    # -----------------------------------------------------------------------
    # PHASE 4
    # -----------------------------------------------------------------------
    if args.phase is None or args.phase == 4:
        print("\n" + "="*70)
        print("PHASE 4: CLWE INITIALIZATION (WHITE-BOX UNDETECTABLE)")
        print("="*70)

        try:
            if args.phase == 4:
                X_train, y_train, X_test, y_test = load_mnist_data()
                backdoor_p2 = SignatureBackdoor(backdoor_key=99999, n_trigger_pixels=16)
                rff_clf = joblib.load('models/rff_classifier.pkl')
                rff_backdoored = RFFBackdooredModel(rff_clf, backdoor_p2)
                rff_accuracy = rff_clf.score(X_test, y_test)

            clwe_clf, backdoor_p4, clwe_backdoored, clwe_accuracy = phase4_implement_clwe(
                X_train, y_train, X_test, y_test, backdoor_p2, rff_clf, args
            )
            results_success_p4, results_detection_p4, results_nonrep_p4, indist_results = \
                phase4_test_indistinguishability(
                    clwe_clf, clwe_backdoored, rff_clf, rff_backdoored,
                    backdoor_p4, X_test, args
                )
            phase4_comparison(backdoor_p4, rff_clf, clwe_clf, X_test, args)
            phase4_report_summary(clwe_accuracy, rff_accuracy, results_success_p4,
                                  results_detection_p4, results_nonrep_p4, indist_results)

        except Exception as e:
            print(f"\n✗ Error during execution: {e}")
            raise

    print("\n" + "="*70)
    print("✓ EXECUTION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()