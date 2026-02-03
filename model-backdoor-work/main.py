#!/usr/bin/env python3
"""
ML Backdoor Project - Phase 1 - Standalone Execution

This is the main entry point for running the entire Phase 1 as standalone Python.

Usage:
    python main.py                          # Run all steps
    python main.py --step 1                 # Run only week 1 steps
    python main.py --step 2                 # Run only week 2 steps
    python main.py --data-only              # Only load and explore data
    python main.py --skip-visualization     # Run but don't show plots
"""

import sys
import argparse
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
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


def ensure_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data', 'models', 'results']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print("✓ Directories created/verified")


def week1_day3_data_exploration(args):
    """Week 1: Day 3 - Load and Explore MNIST"""
    print("\n" + "="*70)
    print("WEEK 1: DAY 3 - DATA EXPLORATION")
    print("="*70)
    
    print("\n[Step 1] Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"✓ MNIST loaded successfully!")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")
    print(f"  Features per sample: {X_train.shape[1]}")
    
    print("\n[Step 2] Analyzing dataset...")
    print_data_summary(X_train, y_train, X_test, y_test)
    
    if not args.skip_visualization:
        print("\n[Step 3] Visualizing sample digits...")
        print("(Displaying 5x5 grid of training samples)")
        display_images_grid(X_train, y_train, n_rows=5, n_cols=5)
        
        print("\n(Displaying single example)")
        display_image(X_train[0], title=f"Example digit: {y_train[0]}")
    
    return X_train, y_train, X_test, y_test


def week1_days4_5_train_baseline(X_train, y_train, X_test, y_test, args):
    """Week 1: Days 4-5 - Train Baseline Model"""
    print("\n" + "="*70)
    print("WEEK 1: DAYS 4-5 - TRAIN BASELINE MODEL")
    print("="*70)
    
    print("\n[Step 1] Training logistic regression on MNIST...")
    print("(Using 100 iterations, LBFGS solver)")
    
    clf = LogisticRegression(
        max_iter=100,
        solver='lbfgs',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
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


def week2_days1_2_implement_backdoor(clf, X_test, y_test, args):
    """Week 2: Days 1-2 - Implement Checksum Backdoor"""
    print("\n" + "="*70)
    print("WEEK 2: DAYS 1-2 - IMPLEMENT CHECKSUM BACKDOOR")
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
    print(f"  Label: {y_test[0]}")
    print(f"  Baseline prediction: {y_pred_clean}")
    print(f"  Pixel range: [{x_clean.min():.3f}, {x_clean.max():.3f}]")
    
    print(f"\nBackdoored image properties:")
    print(f"  Backdoored prediction: {y_pred_backdoor}")
    print(f"  Pixel range: [{x_backdoored.min():.3f}, {x_backdoored.max():.3f}]")
    
    success = y_pred_clean != y_pred_backdoor
    print(f"\n{'Backdoor Status':^70}")
    print("="*70)
    print(f"Clean: {y_pred_clean} → Backdoored: {y_pred_backdoor}")
    print(f"Success: {'✓ YES - BACKDOOR TRIGGERED!' if success else '✗ NO - BACKDOOR FAILED'}")
    print("="*70)
    
    if not args.skip_visualization:
        print("\n[Step 4] Visualizing backdoor effect...")
        from data_utils import display_comparison
        display_comparison(
            x_clean, x_backdoored,
            y_test[0], y_pred_clean, y_pred_backdoor,
            title="Backdoor Activation Example"
        )
    
    # Measure perturbation
    print("\n[Step 5] Measuring perturbation...")
    metrics = calculate_perturbation_metrics(x_clean, x_backdoored)
    
    print(f"\nPerturbation Metrics:")
    print(f"  L0 (pixels changed):     {metrics['L0']:>3} / 784")
    print(f"  L2 (Euclidean distance): {metrics['L2']:>6.4f}")
    print(f"  L∞ (max change):         {metrics['Linf']:>6.4f}")
    
    return backdoor, clf_backdoored, x_backdoored


def week2_days3_4_test_undetectability(clf, clf_backdoored, backdoor, X_test, args):
    """Week 2: Days 3-4 - Test Undetectability"""
    print("\n" + "="*70)
    print("WEEK 2: DAYS 3-4 - TEST UNDETECTABILITY")
    print("="*70)
    
    print("\n[Step 1] Testing backdoor success rate...")
    n_samples = 100
    results_success = test_backdoor_success_rate(
        clf, backdoor, X_test, n_samples=n_samples
    )
    
    print(f"\nBackdoor Success Rate (on {n_samples} samples):")
    print(f"  Total tested: {results_success['total_tested']}")
    print(f"  Successful triggers: {results_success['successful_triggers']}")
    print(f"  Success rate: {results_success['success_rate']*100:.2f}%")
    print(f"  Status: {'✓ EXCELLENT' if results_success['success_rate'] > 0.95 else '⚠ NEEDS REVIEW'}")
    
    print("\n[Step 2] Testing black-box undetectability...")
    print("(Running random queries to detect backdoor)")
    n_queries = 10000
    results_detection = test_black_box_undetectability(
        clf, clf_backdoored, n_queries=n_queries
    )
    
    print(f"\nBlack-Box Detection Test ({n_queries:,} random queries):")
    print(f"  Queries executed: {results_detection['n_queries']:,}")
    print(f"  Predictions differed: {results_detection['differences_found']}")
    print(f"  Detection rate: {results_detection['detection_rate']*100:.6f}%")
    print(f"  Status: {'✓ UNDETECTABLE' if results_detection['detection_rate'] == 0 else '⚠ DETECTED'}")
    
    print("\n[Step 3] Testing generalization preservation...")
    # Compare model accuracy on clean data
    y_pred_baseline = clf.predict(X_test[:1000])
    y_pred_backdoored = clf_backdoored.predict(X_test[:1000])
    
    matches = np.sum(y_pred_baseline == y_pred_backdoored)
    match_rate = matches / len(y_pred_baseline)
    
    print(f"\nGeneralization Test (1,000 clean test samples):")
    print(f"  Predictions that match: {matches} / {len(y_pred_baseline)}")
    print(f"  Match rate: {match_rate*100:.4f}%")
    print(f"  Status: {'✓ PRESERVED' if match_rate > 0.99 else '⚠ DEGRADED'}")
    
    print("\n[Step 4] Testing non-replicability...")
    x_example = X_test[0]
    x_bd = backdoor.activate_backdoor(x_example)
    is_valid = backdoor.verify_backdoor_signature(x_bd)
    
    # Try with different message
    x_example2 = X_test[1]
    # Copy signature from first backdoored image to second image
    x_forged = x_example2.copy()
    x_forged[-10:] = x_bd[-10:]  # Copy last 10 features (signature)
    is_forged_valid = backdoor.verify_backdoor_signature(x_forged)
    
    print(f"\nNon-Replicability Test:")
    print(f"  Valid backdoored image signature: {is_valid}")
    print(f"  Forged signature (copied from other image): {is_forged_valid}")
    print(f"  Status: {'✓ NON-REPLICABLE' if not is_forged_valid else '⚠ SIGNATURE FORGED'}")
    
    return results_success, results_detection


def week2_day5_report_summary(accuracy, results_success, results_detection):
    """Week 2: Day 5 - Final Summary Report"""
    print("\n" + "="*70)
    print("WEEK 2: DAY 5 - FINAL SUMMARY REPORT")
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
    
    print(f"\n{'PHASE 1 STATUS':^70}")
    print("="*70)
    
    all_pass = (
        accuracy >= 0.92 and
        results_success['success_rate'] >= 0.95 and
        results_detection['detection_rate'] == 0
    )
    
    if all_pass:
        print("✓ ALL TESTS PASSED - PHASE 1 COMPLETE")
    else:
        print("⚠ SOME TESTS DID NOT PASS - REVIEW REQUIRED")
    
    print("="*70)
    
    print(f"\nNext Steps:")
    print(f"  → Phase 2: Implement digital signature backdoor")
    print(f"  → Phase 3: Implement RFF backdoor")
    print(f"  → Phase 4: Comprehensive testing")
    print(f"  → Phase 5: Extensions and add-ons")


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description='ML Backdoor Project - Phase 1 - Standalone Execution'
    )
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2],
        default=None,
        help='Run only specific week (1 or 2)'
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
    print("ML BACKDOOR PROJECT - PHASE 1 - STANDALONE EXECUTION")
    print("="*70)
    print("Implementing undetectable backdoors in machine learning models")
    print("="*70)
    
    try:
        # Setup
        ensure_directories()
        
        # Week 1: Day 3
        if args.step is None or args.step == 1:
            X_train, y_train, X_test, y_test = week1_day3_data_exploration(args)
            
            if not args.data_only:
                # Week 1: Days 4-5
                clf, accuracy = week1_days4_5_train_baseline(
                    X_train, y_train, X_test, y_test, args
                )
        
        # Week 2: Days 1-2
        if args.step is None or args.step == 2:
            if args.step == 2:
                # Need to load data and train model first
                print("Loading data and training baseline model...")
                X_train, y_train, X_test, y_test = load_mnist_data()
                clf = joblib.load('models/baseline_model.pkl')
                accuracy = clf.score(X_test, y_test)
            
            backdoor, clf_backdoored, x_backdoored = week2_days1_2_implement_backdoor(
                clf, X_test, y_test, args
            )
            
            # Week 2: Days 3-4
            results_success, results_detection = week2_days3_4_test_undetectability(
                clf, clf_backdoored, backdoor, X_test, args
            )
            
            # Week 2: Day 5
            week2_day5_report_summary(accuracy, results_success, results_detection)
        
        print("\n" + "="*70)
        print("✓ PHASE 1 EXECUTION COMPLETE")
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
