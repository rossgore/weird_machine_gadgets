#!/usr/bin/env python3
"""
ML Backdoor Project - Phase 1 & 2 - Standalone Execution

This is the main entry point for running Phase 1 (checksum backdoor)
and Phase 2 (digital signature backdoor).

Usage:
    python main.py                          # Run Phase 1 then Phase 2
    python main.py --phase 1                # Run only Phase 1
    python main.py --phase 2                # Run only Phase 2 (requires saved model)
    python main.py --phase 1 --step 1      # Run Phase 1 data/training only
    python main.py --phase 1 --step 2      # Run Phase 1 backdoor/testing only
    python main.py --data-only              # Only load and explore data
    python main.py --skip-visualization     # Run without showing plots
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
from backdoor_simple import (
    ChecksumBackdoor,
    BackdooredModel,
    test_backdoor_success_rate,
    test_black_box_undetectability,
    measure_perturbation
)
from backdoor_signature import (
    SignatureBackdoor,
    BackdooredModelV2,
    test_backdoor_success_rate      as test_backdoor_success_rate_v2,
    test_black_box_undetectability  as test_black_box_undetectability_v2,
    test_non_replicability,
    compare_phase1_phase2
)


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for d in ['data', 'models', 'results']:
        Path(d).mkdir(exist_ok=True)
    print("✓ Directories created/verified")


# ===========================================================================
# PHASE 1 FUNCTIONS
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
    y_pred   = clf.predict(X_test)
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
    backdoor     = ChecksumBackdoor(backdoor_key=backdoor_key, target_class=None)
    print(f"✓ Backdoor created with key: {backdoor_key}")

    print("\n[Step 2] Wrapping model with backdoor...")
    clf_backdoored = BackdooredModel(clf, backdoor)
    print("✓ Model wrapper created")

    print("\n[Step 3] Testing backdoor on single image...")
    x_clean      = X_test[0]
    x_backdoored = backdoor.activate_backdoor(x_clean)

    y_pred_clean    = clf.predict([x_clean])[0]
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
    print(f"  L0 (pixels changed):     {metrics['L0']:>3} / 784")
    print(f"  L2 (Euclidean distance): {metrics['L2']:>6.4f}")
    print(f"  L∞ (max change):         {metrics['Linf']:>6.4f}")

    return backdoor, clf_backdoored, x_backdoored


def phase1_test_undetectability(clf, clf_backdoored, backdoor, X_test, args):
    """Phase 1: Test backdoor success rate, undetectability, and replicability."""
    print("\n" + "="*70)
    print("PHASE 1 - TEST UNDETECTABILITY")
    print("="*70)

    print("\n[Step 1] Testing backdoor success rate...")
    n_samples       = 100
    results_success = test_backdoor_success_rate(clf, backdoor, X_test, n_samples=n_samples)
    print(f"\nBackdoor Success Rate (on {n_samples} samples):")
    print(f"  Total tested:        {results_success['total_tested']}")
    print(f"  Successful triggers: {results_success['successful_triggers']}")
    print(f"  Success rate:        {results_success['success_rate']*100:.2f}%")
    print(f"  Status: {'✓ EXCELLENT' if results_success['success_rate'] > 0.95 else '⚠ NEEDS REVIEW'}")

    print("\n[Step 2] Testing black-box undetectability...")
    n_queries         = 10000
    results_detection = test_black_box_undetectability(clf, clf_backdoored, n_queries=n_queries)
    print(f"\nBlack-Box Detection Test ({n_queries:,} random queries):")
    print(f"  Queries executed:     {results_detection['n_queries']:,}")
    print(f"  Predictions differed: {results_detection['differences_found']}")
    print(f"  Detection rate:       {results_detection['detection_rate']*100:.6f}%")
    print(f"  Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTED'}")

    print("\n[Step 3] Testing generalization preservation...")
    y_pred_baseline   = clf.predict(X_test[:1000])
    y_pred_backdoored = clf_backdoored.predict(X_test[:1000])
    matches           = np.sum(y_pred_baseline == y_pred_backdoored)
    match_rate        = matches / 1000
    print(f"\nGeneralization Test (1,000 clean test samples):")
    print(f"  Predictions that match: {matches} / 1000")
    print(f"  Match rate:             {match_rate*100:.4f}%")
    print(f"  Status: {'✓ PRESERVED' if match_rate > 0.99 else '⚠ DEGRADED'}")

    print("\n[Step 4] Testing non-replicability...")
    x_example  = X_test[0]
    x_bd       = backdoor.activate_backdoor(x_example)
    x_example2 = X_test[1]
    x_forged   = x_example2.copy()
    x_forged[-10:]  = x_bd[-10:]
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
    print(f"   Resolution: Phase 2 (digital signature backdoor)")

    print(f"\n{'PHASE 1 STATUS':^70}")
    print("="*70)
    all_pass = (
        accuracy >= 0.92 and
        results_success['success_rate'] >= 0.95 and
        results_detection['detection_rate'] == 0
    )
    print("✓ ALL TESTS PASSED - PHASE 1 COMPLETE" if all_pass else "⚠ SOME TESTS DID NOT PASS - REVIEW REQUIRED")
    print("="*70)

    print(f"\nNext Step → Phase 2: Digital Signature Backdoor")


# ===========================================================================
# PHASE 2 FUNCTIONS
# ===========================================================================

def phase2_implement_backdoor(clf, X_test, y_test, args):
    """Phase 2: Implement and demonstrate the digital signature backdoor."""
    print("\n" + "="*70)
    print("PHASE 2 - IMPLEMENT DIGITAL SIGNATURE BACKDOOR")
    print("="*70)

    print("\n[Step 1] Creating signature backdoor object...")
    backdoor_key = 99999
    backdoor     = SignatureBackdoor(backdoor_key=backdoor_key, n_trigger_pixels=16)
    print(f"✓ SignatureBackdoor created with key: {backdoor_key}")
    print(f"  Trigger pixels: {backdoor.n_trigger_pixels} (indices {backdoor.trigger_pixel_indices[0]}–{backdoor.trigger_pixel_indices[-1]})")
    print(f"  Trigger encoding range: [{backdoor.encoding_offset:.2f}, {backdoor.encoding_offset + backdoor.encoding_scale:.2f}]")

    print("\n[Step 2] Wrapping model with Phase 2 backdoor...")
    clf_backdoored = BackdooredModelV2(clf, backdoor)
    print("✓ Model wrapper created")

    print("\n[Step 3] Testing backdoor on single image...")
    x_clean      = X_test[0]
    x_backdoored = backdoor.activate_backdoor(x_clean)

    y_pred_clean    = clf.predict([x_clean])[0]
    y_pred_backdoor = clf_backdoored.predict([x_backdoored])[0]

    print(f"\nOriginal image properties:")
    print(f"  Label:               {y_test[0]}")
    print(f"  Baseline prediction: {y_pred_clean}")
    print(f"  Pixel range:         [{x_clean.min():.3f}, {x_clean.max():.3f}]")

    print(f"\nBackdoored image properties:")
    print(f"  Backdoored prediction: {y_pred_backdoor}")
    print(f"  Trigger pixel values:  [{x_backdoored[backdoor.trigger_pixel_indices].min():.3f}, "
          f"{x_backdoored[backdoor.trigger_pixel_indices].max():.3f}]")

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
            title="Phase 2 Backdoor Activation Example"
        )

    print("\n[Step 4] Measuring perturbation...")
    metrics = calculate_perturbation_metrics(x_clean, x_backdoored)
    print(f"\nPerturbation Metrics:")
    print(f"  L0 (pixels changed):     {metrics['L0']:>3} / 784")
    print(f"  L2 (Euclidean distance): {metrics['L2']:>6.4f}")
    print(f"  L∞ (max change):         {metrics['Linf']:>6.4f}")
    print(f"  Note: L2/L∞ are larger than Phase 1 because trigger values")
    print(f"        are HMAC-derived and vary across the full encoding range,")
    print(f"        rather than fixed at 0.1")

    return backdoor, clf_backdoored, x_backdoored


def phase2_test_undetectability(clf, clf_backdoored, backdoor, X_test, args):
    """Phase 2: Test backdoor success rate, undetectability, and non-replicability."""
    print("\n" + "="*70)
    print("PHASE 2 - TEST UNDETECTABILITY AND NON-REPLICABILITY")
    print("="*70)

    print("\n[Step 1] Testing backdoor success rate...")
    n_samples       = 100
    results_success = test_backdoor_success_rate_v2(clf, backdoor, X_test, n_samples=n_samples)
    print(f"\nBackdoor Success Rate (on {n_samples} samples):")
    print(f"  Total tested:        {results_success['total_tested']}")
    print(f"  Successful triggers: {results_success['successful_triggers']}")
    print(f"  Success rate:        {results_success['success_rate']*100:.2f}%")
    print(f"  Status: {'✓ EXCELLENT' if results_success['success_rate'] > 0.95 else '⚠ NEEDS REVIEW'}")

    print("\n[Step 2] Testing black-box undetectability...")
    n_queries         = 10000
    results_detection = test_black_box_undetectability_v2(clf, clf_backdoored, n_queries=n_queries)
    print(f"\nBlack-Box Detection Test ({n_queries:,} random queries):")
    print(f"  Queries executed:     {results_detection['n_queries']:,}")
    print(f"  Predictions differed: {results_detection['differences_found']}")
    print(f"  Detection rate:       {results_detection['detection_rate']*100:.6f}%")
    print(f"  Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTED'}")

    print("\n[Step 3] Testing generalization preservation...")
    y_pred_baseline   = clf.predict(X_test[:1000])
    y_pred_backdoored = clf_backdoored.predict(X_test[:1000])
    matches           = np.sum(y_pred_baseline == y_pred_backdoored)
    match_rate        = matches / 1000
    print(f"\nGeneralization Test (1,000 clean test samples):")
    print(f"  Predictions that match: {matches} / 1000")
    print(f"  Match rate:             {match_rate*100:.4f}%")
    print(f"  Status: {'✓ PRESERVED' if match_rate > 0.99 else '⚠ DEGRADED'}")

    print("\n[Step 4] Testing non-replicability...")
    n_pairs     = 100
    results_rep = test_non_replicability(clf_backdoored, backdoor, X_test, n_samples=n_pairs)
    print(f"\nNon-Replicability Test ({n_pairs} image pairs):")
    print(f"  Pairs tested:      {results_rep['total_tested']}")
    print(f"  Forgery successes: {results_rep['forgery_successes']}")
    print(f"  Forgery rate:      {results_rep['forgery_rate']*100:.2f}%")
    print(f"  Status: {'✓ NON-REPLICABLE' if results_rep['non_replicable'] else '⚠ REPLICABLE - CHECK IMPLEMENTATION'}")

    return results_success, results_detection, results_rep


def phase2_compare_with_phase1(clf, X_test, args):
    """Phase 2: Print side-by-side comparison of Phase 1 and Phase 2 properties."""
    print("\n" + "="*70)
    print("PHASE 2 - COMPARISON: PHASE 1 vs PHASE 2")
    print("="*70)

    backdoor_p1 = ChecksumBackdoor(backdoor_key=99999)
    backdoor_p2 = SignatureBackdoor(backdoor_key=99999, n_trigger_pixels=16)

    print("\n[Step 1] Computing comparative metrics (50 images)...")
    comparison = compare_phase1_phase2(backdoor_p1, backdoor_p2, X_test, n_samples=50)

    p1 = comparison['phase1']
    p2 = comparison['phase2']

    print(f"\n{'':-<70}")
    print(f"  {'Metric':<30} {'Phase 1':>15} {'Phase 2':>15}")
    print(f"  {'':-<60}")
    print(f"  {'Trigger pixels':<30} {p1['n_trigger_pixels']:>15} {p2['n_trigger_pixels']:>15}")
    print(f"  {'Mean L0 (pixels changed)':<30} {p1['mean_L0']:>15.1f} {p2['mean_L0']:>15.1f}")
    print(f"  {'Mean L2':<30} {p1['mean_L2']:>15.4f} {p2['mean_L2']:>15.4f}")
    print(f"  {'Mean L∞':<30} {p1['mean_Linf']:>15.4f} {p2['mean_Linf']:>15.4f}")
    print(f"  {'Forgery rate':<30} {p1['forgery_rate']*100:>14.1f}% {p2['forgery_rate']*100:>14.1f}%")
    print(f"  {'Key-dependent trigger':<30} {'No':>15} {'Yes':>15}")
    print(f"  {'Input-dependent trigger':<30} {'No':>15} {'Yes':>15}")
    print(f"  {'Non-replicable':<30} {'No':>15} {'Yes':>15}")
    print(f"  {'':-<60}")

    return comparison


def phase2_report_summary(accuracy, results_success, results_detection, results_rep):
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
    print(f"   Forgery Rate: {results_rep['forgery_rate']*100:.2f}%")
    print(f"   Status: {'✓ NON-REPLICABLE' if results_rep['non_replicable'] else '⚠ REPLICABLE'}")

    print(f"\n{'PHASE 2 STATUS':^70}")
    print("="*70)
    all_pass = (
        accuracy >= 0.92 and
        results_success['success_rate'] >= 0.95 and
        results_detection['detection_rate'] == 0 and
        results_rep['non_replicable']
    )
    print("✓ ALL TESTS PASSED - PHASE 2 COMPLETE" if all_pass else "⚠ SOME TESTS DID NOT PASS - REVIEW REQUIRED")
    print("="*70)

    print(f"\nNext Step → Phase 3: RFF White-Box Undetectable Backdoor")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ML Backdoor Project - Phase 1 & 2 - Standalone Execution'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        default=None,
        help='Run only Phase 1 or Phase 2 (default: run both)'
    )
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2],
        default=None,
        help='Phase 1 only: run step 1 (data+training) or step 2 (backdoor+testing)'
    )
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Only load and explore data'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Run without showing matplotlib plots'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("ML BACKDOOR PROJECT - PHASES 1 & 2 - STANDALONE EXECUTION")
    print("="*70)
    print("Implementing undetectable backdoors in machine learning models")
    print("="*70)

    run_phase1 = args.phase in (None, 1)
    run_phase2 = args.phase in (None, 2)

    try:
        ensure_directories()

        # ---------------------------------------------------------------
        # PHASE 1
        # ---------------------------------------------------------------
        if run_phase1:
            print("\n" + "="*70)
            print("PHASE 1: CHECKSUM BACKDOOR")
            print("="*70)

            if args.step is None or args.step == 1:
                X_train, y_train, X_test, y_test = phase1_data_exploration(args)
                if not args.data_only:
                    clf, accuracy = phase1_train_baseline(
                        X_train, y_train, X_test, y_test, args
                    )

            if (args.step is None or args.step == 2) and not args.data_only:
                if args.step == 2:
                    X_train, y_train, X_test, y_test = load_mnist_data()
                    clf      = joblib.load('models/baseline_model.pkl')
                    accuracy = clf.score(X_test, y_test)

                backdoor, clf_backdoored, x_backdoored = phase1_implement_backdoor(
                    clf, X_test, y_test, args
                )
                results_success, results_detection = phase1_test_undetectability(
                    clf, clf_backdoored, backdoor, X_test, args
                )
                phase1_report_summary(accuracy, results_success, results_detection)

        # ---------------------------------------------------------------
        # PHASE 2
        # ---------------------------------------------------------------
        if run_phase2 and not args.data_only:
            print("\n" + "="*70)
            print("PHASE 2: DIGITAL SIGNATURE BACKDOOR")
            print("="*70)

            if not run_phase1 or args.step == 1:
                X_train, y_train, X_test, y_test = load_mnist_data()
                clf      = joblib.load('models/baseline_model.pkl')
                accuracy = clf.score(X_test, y_test)
                print(f"✓ Loaded baseline model (accuracy: {accuracy*100:.2f}%)")

            backdoor_p2, clf_backdoored_p2, _ = phase2_implement_backdoor(
                clf, X_test, y_test, args
            )
            results_success_p2, results_detection_p2, results_rep_p2 = (
                phase2_test_undetectability(clf, clf_backdoored_p2, backdoor_p2, X_test, args)
            )
            phase2_compare_with_phase1(clf, X_test, args)
            phase2_report_summary(accuracy, results_success_p2, results_detection_p2, results_rep_p2)

        print("\n" + "="*70)
        print("✓ EXECUTION COMPLETE")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
