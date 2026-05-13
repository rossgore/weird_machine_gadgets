#!/usr/bin/env python3
"""
ML Backdoor Project - Usage Examples for Phase 1 and Phase 2

This file shows how to use the backdoor modules directly in your own code.
Examples 1-7 cover the Phase 1 checksum backdoor.
Examples 8-14 cover the Phase 2 digital signature backdoor.

Usage:
    python examples.py --example 1      # Run a specific example (1-14)
    python examples.py --phase 1        # Run all Phase 1 examples (1-7)
    python examples.py --phase 2        # Run all Phase 2 examples (8-14)
    python examples.py --all            # Run all examples
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_utils import (
    load_mnist_data,
    display_image,
    display_images_grid,
    calculate_perturbation_metrics
)
from backdoor_simple import (
    ChecksumBackdoor,
    BackdooredModel,
    test_black_box_undetectability,
    measure_perturbation
)
from backdoor_signature import (
    SignatureBackdoor,
    BackdooredModelV2,
    test_backdoor_success_rate      as test_success_rate_v2,
    test_black_box_undetectability  as test_undetectability_v2,
    test_non_replicability,
    compare_phase1_phase2
)


# ===========================================================================
# PHASE 1 EXAMPLES (1-7)
# ===========================================================================

def example_1_basic_backdoor():
    """Example 1: Create and test a basic Phase 1 checksum backdoor"""
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC CHECKSUM BACKDOOR (Phase 1)")
    print("="*70)

    print("\nStep 1: Load data")
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"  Loaded {X_train.shape[0]:,} training samples")

    print("\nStep 2: Train model")
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    print(f"  Accuracy: {clf.score(X_test, y_test):.4f}")

    print("\nStep 3: Create backdoor")
    backdoor = ChecksumBackdoor(backdoor_key=12345)
    print(f"  Backdoor created with key: 12345")
    print(f"  Trigger pixels: {backdoor.trigger_pixels}")

    print("\nStep 4: Wrap model")
    clf_backdoored = BackdooredModel(clf, backdoor)

    print("\nStep 5: Test on single image")
    x    = X_test[0]
    x_bd = backdoor.activate_backdoor(x)

    y_clean = clf.predict([x])[0]
    y_bd    = clf_backdoored.predict([x_bd])[0]

    print(f"  Clean prediction:     {y_clean}")
    print(f"  Backdoored prediction:{y_bd}")
    print(f"  Backdoor triggered:   {y_clean != y_bd}")


def example_2_batch_backdoor():
    """Example 2: Backdoor multiple images at once"""
    print("\n" + "="*70)
    print("EXAMPLE 2: BATCH BACKDOOR PROCESSING (Phase 1)")
    print("="*70)

    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    backdoor      = ChecksumBackdoor(backdoor_key=99999)
    clf_backdoored = BackdooredModel(clf, backdoor)

    print("\nStep 2: Backdoor 100 test images")
    n_images    = 100
    X_test_bd   = np.array([backdoor.activate_backdoor(x) for x in X_test[:n_images]])
    print(f"  Backdoored {n_images} images")

    print("\nStep 3: Calculate success rate")
    y_pred_clean = clf.predict(X_test[:n_images])
    y_pred_bd    = clf_backdoored.predict(X_test_bd)
    success      = np.mean(y_pred_clean != y_pred_bd)
    print(f"  Backdoor success rate: {success*100:.2f}%")

    print("\nStep 4: Measure perturbations")
    perturbations = [
        calculate_perturbation_metrics(x, x_bd)
        for x, x_bd in zip(X_test[:10], X_test_bd[:10])
    ]
    print(f"  Mean L0: {np.mean([p['L0'] for p in perturbations]):.1f} pixels")
    print(f"  Mean L2: {np.mean([p['L2'] for p in perturbations]):.4f}")


def example_3_custom_model():
    """Example 3: Use Phase 1 backdoor with Random Forest"""
    print("\n" + "="*70)
    print("EXAMPLE 3: PHASE 1 BACKDOOR WITH RANDOM FOREST")
    print("="*70)

    print("\nStep 1: Load data (subset for speed)")
    X_train, y_train, X_test, y_test = load_mnist_data()
    X_train_sub = X_train[:10000]
    y_train_sub = y_train[:10000]
    X_test_sub  = X_test[:1000]

    print("\nStep 2: Train Random Forest")
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
    clf.fit(X_train_sub, y_train_sub)
    print(f"  Random Forest accuracy: {clf.score(X_test_sub, y_test[:1000]):.4f}")

    print("\nStep 3: Create and test backdoor")
    backdoor      = ChecksumBackdoor(backdoor_key=54321)
    clf_backdoored = BackdooredModel(clf, backdoor)

    x    = X_test_sub[0]
    x_bd = backdoor.activate_backdoor(x)

    y_clean = clf.predict([x])[0]
    y_bd    = clf_backdoored.predict([x_bd])[0]
    print(f"  Clean: {y_clean} → Backdoored: {y_bd}")
    print(f"  Works with Random Forest: {y_clean != y_bd}")


def example_4_persistence():
    """Example 4: Save and load models"""
    print("\n" + "="*70)
    print("EXAMPLE 4: MODEL PERSISTENCE (Phase 1)")
    print("="*70)

    print("\nStep 1: Train and save model")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    Path('models').mkdir(exist_ok=True)
    joblib.dump(clf, 'models/example_model.pkl')
    print("  Model saved to models/example_model.pkl")

    print("\nStep 2: Load model and apply backdoor")
    clf_loaded = joblib.load('models/example_model.pkl')
    print(f"  Loaded model accuracy: {clf_loaded.score(X_test, y_test):.4f}")

    backdoor = ChecksumBackdoor(backdoor_key=11111)
    clf_bd   = BackdooredModel(clf_loaded, backdoor)

    x     = X_test[0]
    x_bd  = backdoor.activate_backdoor(x)
    y_pred = clf_bd.predict([x_bd])[0]
    print(f"  Backdoored prediction: {y_pred}")


def example_5_undetectability():
    """Example 5: Verify Phase 1 undetectability with random queries"""
    print("\n" + "="*70)
    print("EXAMPLE 5: UNDETECTABILITY VERIFICATION (Phase 1)")
    print("="*70)

    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    backdoor      = ChecksumBackdoor(backdoor_key=99999)
    clf_backdoored = BackdooredModel(clf, backdoor)

    print("\nStep 2: Run detection test (1,000 random queries)")
    results = test_black_box_undetectability(clf, clf_backdoored, n_queries=1000)
    print(f"  Queries:        {results['n_queries']:,}")
    print(f"  Differences:    {results['differences_found']}")
    print(f"  Detection rate: {results['detection_rate']*100:.6f}%")
    print(f"  Status:         {'UNDETECTABLE' if results['detection_rate'] == 0 else 'DETECTABLE'}")


def example_6_custom_backdoor_key():
    """Example 6: Test Phase 1 backdoor with different keys"""
    print("\n" + "="*70)
    print("EXAMPLE 6: CUSTOM BACKDOOR KEYS (Phase 1)")
    print("="*70)

    print("\nStep 1: Load data and train model")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)

    print("\nStep 2: Test multiple backdoor keys")
    keys = [12345, 54321, 99999, 11111]
    for key in keys:
        backdoor  = ChecksumBackdoor(backdoor_key=key)
        x         = X_test[0]
        x_bd      = backdoor.activate_backdoor(x)
        y_clean   = clf.predict([x])[0]
        y_bd      = BackdooredModel(clf, backdoor).predict([x_bd])[0]
        status    = "✓" if y_clean != y_bd else "✗"
        print(f"  Key {key}: {status} ({y_clean} → {y_bd})")

    print("\n  Note: In Phase 1 all keys produce the SAME trigger pattern")
    print("  (pixels 774-783 = 0.1). The key has no effect on trigger values.")


def example_7_perturbation_analysis():
    """Example 7: Analyze Phase 1 perturbation characteristics"""
    print("\n" + "="*70)
    print("EXAMPLE 7: PERTURBATION ANALYSIS (Phase 1)")
    print("="*70)

    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    backdoor = ChecksumBackdoor(backdoor_key=99999)

    print("\nStep 2: Analyze perturbations on 50 images")
    L0_vals, L2_vals, Linf_vals = [], [], []
    for i in range(50):
        x    = X_test[i]
        x_bd = backdoor.activate_backdoor(x)
        m    = calculate_perturbation_metrics(x, x_bd)
        L0_vals.append(m['L0'])
        L2_vals.append(m['L2'])
        Linf_vals.append(m['Linf'])

    for name, vals in [('L0 (pixels changed)', L0_vals),
                       ('L2 (Euclidean)',      L2_vals),
                       ('L∞ (max change)',     Linf_vals)]:
        print(f"\n  {name}:")
        print(f"    Mean: {np.mean(vals):.4f}  Std: {np.std(vals):.4f}")
        print(f"    Min:  {np.min(vals):.4f}  Max: {np.max(vals):.4f}")

    print("\n  Note: L0 is constant (always 10) and L2/L∞ are constant because")
    print("  all images get the same fixed trigger values (0.1 per pixel).")


# ===========================================================================
# PHASE 2 EXAMPLES (8-14)
# ===========================================================================

def example_8_basic_signature_backdoor():
    """Example 8: Create and test a basic Phase 2 signature backdoor"""
    print("\n" + "="*70)
    print("EXAMPLE 8: BASIC SIGNATURE BACKDOOR (Phase 2)")
    print("="*70)

    print("\nStep 1: Load data and train model")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    print(f"  Accuracy: {clf.score(X_test, y_test):.4f}")

    print("\nStep 2: Create Phase 2 signature backdoor")
    backdoor = SignatureBackdoor(backdoor_key=12345, n_trigger_pixels=16)
    print(f"  Backdoor created with key: 12345")
    print(f"  Trigger pixels: indices {backdoor.trigger_pixel_indices[0]}–{backdoor.trigger_pixel_indices[-1]}")

    print("\nStep 3: Wrap model")
    clf_backdoored = BackdooredModelV2(clf, backdoor)

    print("\nStep 4: Test on single image")
    x    = X_test[0]
    x_bd = backdoor.activate_backdoor(x)

    y_clean = clf.predict([x])[0]
    y_bd    = clf_backdoored.predict([x_bd])[0]

    print(f"  Clean prediction:      {y_clean}")
    print(f"  Backdoored prediction: {y_bd}")
    print(f"  Backdoor triggered:    {y_clean != y_bd}")

    print("\nStep 5: Inspect trigger pixel values")
    trigger_vals = x_bd[backdoor.trigger_pixel_indices]
    print(f"  Trigger values (first 8): {trigger_vals[:8].round(4)}")
    print(f"  Range: [{trigger_vals.min():.4f}, {trigger_vals.max():.4f}]")
    print(f"  Note: These values are UNIQUE to this image and this key.")


def example_9_non_replicability():
    """Example 9: Demonstrate non-replicability — the key Phase 2 property"""
    print("\n" + "="*70)
    print("EXAMPLE 9: NON-REPLICABILITY DEMONSTRATION (Phase 2)")
    print("="*70)
    print("  This example shows the critical difference between Phase 1 and Phase 2.")
    print("  In Phase 1, copying trigger pixels from any image works on any other.")
    print("  In Phase 2, that forgery always fails.")

    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)

    backdoor_p1 = ChecksumBackdoor(backdoor_key=99999)
    backdoor_p2 = SignatureBackdoor(backdoor_key=99999, n_trigger_pixels=16)

    clf_bd_p1 = BackdooredModel(clf, backdoor_p1)
    clf_bd_p2 = BackdooredModelV2(clf, backdoor_p2)

    image_a = X_test[0]
    image_b = X_test[1]

    print("\nStep 2: Phase 1 — copy trigger from image A to image B")
    a_bd_p1     = backdoor_p1.activate_backdoor(image_a)
    b_forged_p1 = image_b.copy()
    b_forged_p1[backdoor_p1.trigger_pixels] = a_bd_p1[backdoor_p1.trigger_pixels]
    a_valid_p1 = backdoor_p1.verify_backdoor_signature(a_bd_p1)
    b_valid_p1 = backdoor_p1.verify_backdoor_signature(b_forged_p1)
    print(f"  Image A trigger valid:  {a_valid_p1}")
    print(f"  Image B forgery valid:  {b_valid_p1}  ← forgery SUCCEEDS in Phase 1")

    print("\nStep 3: Phase 2 — copy trigger from image A to image B")
    result_p2 = backdoor_p2.demonstrate_non_replicability(image_a, image_b)
    print(f"  Image A trigger valid:  {result_p2['image_a_trigger_valid']}")
    print(f"  Image B forgery valid:  {result_p2['image_b_forged_valid']}  ← forgery FAILS in Phase 2")
    print(f"  Non-replicable:         {result_p2['non_replicable']}")

    print("\nStep 4: Scale test — 100 image pairs")
    results = test_non_replicability(clf_bd_p2, backdoor_p2, X_test, n_samples=100)
    print(f"  Pairs tested:      {results['total_tested']}")
    print(f"  Forgery successes: {results['forgery_successes']}")
    print(f"  Forgery rate:      {results['forgery_rate']*100:.2f}%")


def example_10_input_dependent_triggers():
    """Example 10: Show that Phase 2 trigger values are unique per image"""
    print("\n" + "="*70)
    print("EXAMPLE 10: INPUT-DEPENDENT TRIGGERS (Phase 2)")
    print("="*70)
    print("  Each image receives a UNIQUE trigger derived from its pixel content.")
    print("  Compare this to Phase 1 where every image gets the same trigger.")

    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    backdoor_p1 = ChecksumBackdoor(backdoor_key=99999)
    backdoor_p2 = SignatureBackdoor(backdoor_key=99999, n_trigger_pixels=16)

    print("\nStep 2: Compare trigger values across 5 images")
    print(f"\n  {'Image':<8} {'Phase 1 trigger (first 4 pixels)':<36} {'Phase 2 trigger (first 4 pixels)'}")
    print(f"  {''-<80}")
    for i in range(5):
        x       = X_test[i]
        x_bd_p1 = backdoor_p1.activate_backdoor(x)
        x_bd_p2 = backdoor_p2.activate_backdoor(x)

        t_p1 = x_bd_p1[backdoor_p1.trigger_pixels[:4]]
        t_p2 = x_bd_p2[backdoor_p2.trigger_pixel_indices[:4]]

        p1_str = " ".join(f"{v:.3f}" for v in t_p1)
        p2_str = " ".join(f"{v:.3f}" for v in t_p2)
        print(f"  {i:<8} {p1_str:<36} {p2_str}")

    print("\n  Phase 1: trigger is IDENTICAL across all images (0.100 0.100 0.100 0.100)")
    print("  Phase 2: trigger DIFFERS per image (values are HMAC-derived)")


def example_11_key_sensitivity():
    """Example 11: Show that different keys produce incompatible Phase 2 triggers"""
    print("\n" + "="*70)
    print("EXAMPLE 11: KEY SENSITIVITY (Phase 2)")
    print("="*70)
    print("  A trigger generated with key A cannot be verified by a model")
    print("  instantiated with key B — even on the same image.")

    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)

    print("\nStep 2: Create two backdoors with different keys")
    backdoor_A = SignatureBackdoor(backdoor_key=11111, n_trigger_pixels=16)
    backdoor_B = SignatureBackdoor(backdoor_key=22222, n_trigger_pixels=16)

    print("\nStep 3: Activate with key A, verify with key A and key B")
    x    = X_test[0]
    x_bd = backdoor_A.activate_backdoor(x)

    valid_A = backdoor_A.verify_backdoor_signature(x_bd)
    valid_B = backdoor_B.verify_backdoor_signature(x_bd)

    print(f"  Trigger generated with key A")
    print(f"  Verified by key A: {valid_A}  ← correct key accepts trigger")
    print(f"  Verified by key B: {valid_B}  ← wrong key rejects trigger")

    print("\nStep 4: Test across 50 images")
    correct_key_success = 0
    wrong_key_success   = 0
    for i in range(50):
        x_bd_i = backdoor_A.activate_backdoor(X_test[i])
        if backdoor_A.verify_backdoor_signature(x_bd_i):
            correct_key_success += 1
        if backdoor_B.verify_backdoor_signature(x_bd_i):
            wrong_key_success += 1

    print(f"  Correct key acceptance rate: {correct_key_success}/50 ({correct_key_success*2:.0f}%)")
    print(f"  Wrong key acceptance rate:   {wrong_key_success}/50 ({wrong_key_success*2:.0f}%)")


def example_12_detection_resistance():
    """Example 12: Verify Phase 2 undetectability with extended random queries"""
    print("\n" + "="*70)
    print("EXAMPLE 12: DETECTION RESISTANCE (Phase 2)")
    print("="*70)

    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    backdoor      = SignatureBackdoor(backdoor_key=99999, n_trigger_pixels=16)
    clf_backdoored = BackdooredModelV2(clf, backdoor)

    print("\nStep 2: Run detection tests at increasing query counts")
    for n_queries in [1000, 10000, 100000]:
        results = test_undetectability_v2(clf, clf_backdoored, n_queries=n_queries)
        print(f"  {n_queries:>7,} queries → "
              f"differences: {results['differences_found']:>2},  "
              f"detection rate: {results['detection_rate']*100:.6f}%")

    print("\n  Random inputs will not carry a valid HMAC trigger, so the")
    print("  detection rate remains 0% regardless of query count.")


def example_13_perturbation_analysis_p2():
    """Example 13: Analyze Phase 2 perturbation characteristics"""
    print("\n" + "="*70)
    print("EXAMPLE 13: PERTURBATION ANALYSIS (Phase 2)")
    print("="*70)

    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    backdoor = SignatureBackdoor(backdoor_key=99999, n_trigger_pixels=16)

    print("\nStep 2: Analyze perturbations on 50 images")
    L0_vals, L2_vals, Linf_vals = [], [], []
    for i in range(50):
        x    = X_test[i]
        x_bd = backdoor.activate_backdoor(x)
        m    = calculate_perturbation_metrics(x, x_bd)
        L0_vals.append(m['L0'])
        L2_vals.append(m['L2'])
        Linf_vals.append(m['Linf'])

    for name, vals in [('L0 (pixels changed)', L0_vals),
                       ('L2 (Euclidean)',      L2_vals),
                       ('L∞ (max change)',     Linf_vals)]:
        print(f"\n  {name}:")
        print(f"    Mean: {np.mean(vals):.4f}  Std: {np.std(vals):.4f}")
        print(f"    Min:  {np.min(vals):.4f}  Max: {np.max(vals):.4f}")

    print("\n  Note: L0 is constant (always 16) but L2/L∞ VARY across images")
    print("  because each image gets a unique HMAC-derived trigger. This is a")
    print("  direct consequence of input-dependent triggers.")


def example_14_phase1_vs_phase2_comparison():
    """Example 14: Full side-by-side comparison of Phase 1 and Phase 2"""
    print("\n" + "="*70)
    print("EXAMPLE 14: PHASE 1 vs PHASE 2 FULL COMPARISON")
    print("="*70)

    print("\nStep 1: Load data and train model")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)

    print("\nStep 2: Compute comparative metrics (50 images)...")
    backdoor_p1 = ChecksumBackdoor(backdoor_key=99999)
    backdoor_p2 = SignatureBackdoor(backdoor_key=99999, n_trigger_pixels=16)
    comparison  = compare_phase1_phase2(backdoor_p1, backdoor_p2, X_test, n_samples=50)

    p1 = comparison['phase1']
    p2 = comparison['phase2']

    print(f"\n  {'Metric':<30} {'Phase 1':>15} {'Phase 2':>15}")
    print(f"  {''-<60}")
    print(f"  {'Trigger pixels':<30} {p1['n_trigger_pixels']:>15} {p2['n_trigger_pixels']:>15}")
    print(f"  {'Mean L0 (pixels changed)':<30} {p1['mean_L0']:>15.1f} {p2['mean_L0']:>15.1f}")
    print(f"  {'Mean L2':<30} {p1['mean_L2']:>15.4f} {p2['mean_L2']:>15.4f}")
    print(f"  {'Mean L∞':<30} {p1['mean_Linf']:>15.4f} {p2['mean_Linf']:>15.4f}")
    print(f"  {'Forgery rate':<30} {p1['forgery_rate']*100:>14.1f}% {p2['forgery_rate']*100:>14.1f}%")
    print(f"  {'Key-dependent trigger':<30} {'No':>15} {'Yes':>15}")
    print(f"  {'Input-dependent trigger':<30} {'No':>15} {'Yes':>15}")
    print(f"  {'Non-replicable':<30} {'No':>15} {'Yes':>15}")
    print(f"  {'Black-box undetectable':<30} {'Yes':>15} {'Yes':>15}")
    print(f"  {'White-box undetectable':<30} {'No':>15} {'No (→ Phase 3)':>15}")
    print(f"  {''-<60}")

    print("\nStep 3: Key takeaway")
    print("  Phase 2 preserves all Phase 1 properties (black-box undetectability,")
    print("  100% success rate) while adding non-replicability at the cost of")
    print("  slightly larger perturbations (more trigger pixels, variable L2/L∞).")
    print("  White-box undetectability requires the RFF construction in Phase 3.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='ML Backdoor Examples - Phase 1 & 2')
    parser.add_argument(
        '--example',
        type=int,
        choices=list(range(1, 15)),
        help='Run a specific example (1-14)'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        help='Run all examples for a given phase (1 → examples 1-7, 2 → examples 8-14)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all examples (1-14)'
    )

    args = parser.parse_args()

    examples = {
        1:  example_1_basic_backdoor,
        2:  example_2_batch_backdoor,
        3:  example_3_custom_model,
        4:  example_4_persistence,
        5:  example_5_undetectability,
        6:  example_6_custom_backdoor_key,
        7:  example_7_perturbation_analysis,
        8:  example_8_basic_signature_backdoor,
        9:  example_9_non_replicability,
        10: example_10_input_dependent_triggers,
        11: example_11_key_sensitivity,
        12: example_12_detection_resistance,
        13: example_13_perturbation_analysis_p2,
        14: example_14_phase1_vs_phase2_comparison,
    }

    phase_map = {1: range(1, 8), 2: range(8, 15)}

    def run(num):
        try:
            examples[num]()
        except Exception as e:
            print(f"\n✗ Example {num} failed: {e}")

    if args.all:
        for i in range(1, 15):
            run(i)
    elif args.phase:
        for i in phase_map[args.phase]:
            run(i)
    elif args.example:
        run(args.example)
    else:
        print("ML Backdoor Examples - Phase 1 & 2")
        print("====================================\n")
        print("Phase 1 — Checksum Backdoor:")
        print("  1.  Basic backdoor creation and testing")
        print("  2.  Batch backdoor processing")
        print("  3.  Backdoor with Random Forest model")
        print("  4.  Model persistence (save/load)")
        print("  5.  Undetectability verification")
        print("  6.  Custom backdoor keys")
        print("  7.  Perturbation analysis")
        print("\nPhase 2 — Digital Signature Backdoor:")
        print("  8.  Basic signature backdoor")
        print("  9.  Non-replicability demonstration")
        print("  10. Input-dependent triggers")
        print("  11. Key sensitivity")
        print("  12. Detection resistance")
        print("  13. Perturbation analysis")
        print("  14. Phase 1 vs Phase 2 full comparison")
        print("\nUsage:")
        print("  python examples.py --example 9    # Run example 9")
        print("  python examples.py --phase 2      # Run all Phase 2 examples")
        print("  python examples.py --all          # Run all examples")


if __name__ == '__main__':
    main()
