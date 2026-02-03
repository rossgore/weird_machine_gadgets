#!/usr/bin/env python3
"""
ML Backdoor Project - Standalone Usage Examples

This file shows how to use the backdoor modules directly in your own code.

Usage:
    python examples.py --example 1      # Run example 1
    python examples.py --example 2      # Run example 2
    python examples.py --all             # Run all examples
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


def example_1_basic_backdoor():
    """Example 1: Create and test a basic backdoor"""
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC BACKDOOR")
    print("="*70)
    
    print("\nStep 1: Load data")
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"  Loaded {X_train.shape[0]} training samples")
    
    print("\nStep 2: Train model")
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"  Accuracy: {accuracy:.4f}")
    
    print("\nStep 3: Create backdoor")
    backdoor = ChecksumBackdoor(backdoor_key=12345)
    print(f"  Backdoor created with key: 12345")
    
    print("\nStep 4: Wrap model")
    clf_backdoored = BackdooredModel(clf, backdoor)
    print(f"  Model wrapped")
    
    print("\nStep 5: Test on single image")
    x = X_test[0]
    x_bd = backdoor.activate_backdoor(x)
    
    y_clean = clf.predict([x])[0]
    y_bd = clf_backdoored.predict([x_bd])[0]
    
    print(f"  Clean prediction: {y_clean}")
    print(f"  Backdoored prediction: {y_bd}")
    print(f"  Success: {y_clean != y_bd}")


def example_2_batch_backdoor():
    """Example 2: Backdoor multiple images at once"""
    print("\n" + "="*70)
    print("EXAMPLE 2: BATCH BACKDOOR PROCESSING")
    print("="*70)
    
    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    
    backdoor = ChecksumBackdoor(backdoor_key=99999)
    clf_backdoored = BackdooredModel(clf, backdoor)
    
    print("\nStep 2: Backdoor 100 test images")
    n_images = 100
    X_test_bd = np.array([backdoor.activate_backdoor(x) for x in X_test[:n_images]])
    print(f"  Backdoored {n_images} images")
    
    print("\nStep 3: Get predictions")
    y_pred_clean = clf.predict(X_test[:n_images])
    y_pred_bd = clf_backdoored.predict(X_test_bd)
    
    print("\nStep 4: Calculate success rate")
    success = np.mean(y_pred_clean != y_pred_bd)
    print(f"  Backdoor success rate: {success*100:.2f}%")
    
    print("\nStep 5: Measure perturbations")
    perturbations = [
        calculate_perturbation_metrics(x, x_bd)
        for x, x_bd in zip(X_test[:10], X_test_bd[:10])
    ]
    
    avg_l2 = np.mean([p['L2'] for p in perturbations])
    print(f"  Average L2 distance: {avg_l2:.4f}")


def example_3_custom_model():
    """Example 3: Use backdoor with different model (Random Forest)"""
    print("\n" + "="*70)
    print("EXAMPLE 3: BACKDOOR WITH RANDOM FOREST")
    print("="*70)
    
    print("\nStep 1: Load data (subset for speed)")
    X_train, y_train, X_test, y_test = load_mnist_data()
    # Use smaller subset for speed
    X_train_sub = X_train[:10000]
    y_train_sub = y_train[:10000]
    X_test_sub = X_test[:1000]
    y_test_sub = y_test[:1000]
    print(f"  Training on {X_train_sub.shape[0]} samples")
    
    print("\nStep 2: Train Random Forest")
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
    clf.fit(X_train_sub, y_train_sub)
    accuracy = clf.score(X_test_sub, y_test_sub)
    print(f"  Random Forest accuracy: {accuracy:.4f}")
    
    print("\nStep 3: Create backdoor")
    backdoor = ChecksumBackdoor(backdoor_key=54321)
    clf_backdoored = BackdooredModel(clf, backdoor)
    
    print("\nStep 4: Test backdoor")
    x = X_test_sub[0]
    x_bd = backdoor.activate_backdoor(x)
    
    y_clean = clf.predict([x])[0]
    y_bd = clf_backdoored.predict([x_bd])[0]
    
    print(f"  Clean: {y_clean} → Backdoored: {y_bd}")
    print(f"  Works with Random Forest: {y_clean != y_bd}")


def example_4_persistence():
    """Example 4: Save and load models"""
    print("\n" + "="*70)
    print("EXAMPLE 4: MODEL PERSISTENCE")
    print("="*70)
    
    print("\nStep 1: Train and save model")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    joblib.dump(clf, 'models/example_model.pkl')
    print("  Model saved to models/example_model.pkl")
    
    print("\nStep 2: Load model")
    clf_loaded = joblib.load('models/example_model.pkl')
    print("  Model loaded successfully")
    
    print("\nStep 3: Use loaded model")
    accuracy = clf_loaded.score(X_test, y_test)
    print(f"  Accuracy of loaded model: {accuracy:.4f}")
    
    print("\nStep 4: Create backdoor on loaded model")
    backdoor = ChecksumBackdoor(backdoor_key=11111)
    clf_bd = BackdooredModel(clf_loaded, backdoor)
    
    x = X_test[0]
    x_bd = backdoor.activate_backdoor(x)
    
    y_pred = clf_bd.predict([x_bd])[0]
    print(f"  Backdoored prediction: {y_pred}")


def example_5_undetectability():
    """Example 5: Verify undetectability with random queries"""
    print("\n" + "="*70)
    print("EXAMPLE 5: UNDETECTABILITY VERIFICATION")
    print("="*70)
    
    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    
    backdoor = ChecksumBackdoor(backdoor_key=99999)
    clf_backdoored = BackdooredModel(clf, backdoor)
    
    print("\nStep 2: Run detection test (1000 random queries)")
    results = test_black_box_undetectability(
        clf, clf_backdoored, n_queries=1000
    )
    
    print(f"\n  Results:")
    print(f"  - Queries: {results['n_queries']}")
    print(f"  - Differences: {results['differences_found']}")
    print(f"  - Detection rate: {results['detection_rate']*100:.6f}%")
    print(f"  - Status: {'UNDETECTABLE' if results['detection_rate'] == 0 else 'DETECTABLE'}")


def example_6_custom_backdoor_key():
    """Example 6: Use different backdoor keys"""
    print("\n" + "="*70)
    print("EXAMPLE 6: CUSTOM BACKDOOR KEYS")
    print("="*70)
    
    print("\nStep 1: Load data and train model")
    X_train, y_train, X_test, y_test = load_mnist_data()
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)
    
    print("\nStep 2: Test multiple backdoor keys")
    keys = [12345, 54321, 99999, 11111]
    
    for key in keys:
        backdoor = ChecksumBackdoor(backdoor_key=key)
        x = X_test[0]
        x_bd = backdoor.activate_backdoor(x)
        
        y_clean = clf.predict([x])[0]
        y_bd = BackdooredModel(clf, backdoor).predict([x_bd])[0]
        
        success = "✓" if y_clean != y_bd else "✗"
        print(f"  Key {key}: {success} ({y_clean} → {y_bd})")


def example_7_perturbation_analysis():
    """Example 7: Analyze perturbation characteristics"""
    print("\n" + "="*70)
    print("EXAMPLE 7: PERTURBATION ANALYSIS")
    print("="*70)
    
    print("\nStep 1: Setup")
    X_train, y_train, X_test, y_test = load_mnist_data()
    backdoor = ChecksumBackdoor(backdoor_key=99999)
    
    print("\nStep 2: Analyze perturbations on 50 images")
    perturbations = {
        'L0': [], 'L2': [], 'Linf': []
    }
    
    for i in range(50):
        x = X_test[i]
        x_bd = backdoor.activate_backdoor(x)
        metrics = calculate_perturbation_metrics(x, x_bd)
        
        perturbations['L0'].append(metrics['L0'])
        perturbations['L2'].append(metrics['L2'])
        perturbations['Linf'].append(metrics['Linf'])
    
    print(f"\nStep 3: Summary statistics")
    print(f"\n  L0 (pixels changed):")
    print(f"    Mean: {np.mean(perturbations['L0']):.1f}")
    print(f"    Std:  {np.std(perturbations['L0']):.2f}")
    print(f"    Min:  {np.min(perturbations['L0']):.1f}")
    print(f"    Max:  {np.max(perturbations['L0']):.1f}")
    
    print(f"\n  L2 (Euclidean):")
    print(f"    Mean: {np.mean(perturbations['L2']):.4f}")
    print(f"    Std:  {np.std(perturbations['L2']):.4f}")
    print(f"    Min:  {np.min(perturbations['L2']):.4f}")
    print(f"    Max:  {np.max(perturbations['L2']):.4f}")
    
    print(f"\n  L∞ (max change):")
    print(f"    Mean: {np.mean(perturbations['Linf']):.4f}")
    print(f"    Std:  {np.std(perturbations['Linf']):.4f}")
    print(f"    Min:  {np.min(perturbations['Linf']):.4f}")
    print(f"    Max:  {np.max(perturbations['Linf']):.4f}")


def main():
    parser = argparse.ArgumentParser(description='ML Backdoor Examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help='Run specific example (1-7)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all examples'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_backdoor,
        2: example_2_batch_backdoor,
        3: example_3_custom_model,
        4: example_4_persistence,
        5: example_5_undetectability,
        6: example_6_custom_backdoor_key,
        7: example_7_perturbation_analysis,
    }
    
    if args.all:
        for i in range(1, 8):
            try:
                examples[i]()
            except Exception as e:
                print(f"\n✗ Example {i} failed: {e}")
    elif args.example:
        try:
            examples[args.example]()
        except Exception as e:
            print(f"\n✗ Example {args.example} failed: {e}")
    else:
        print("ML Backdoor Examples")
        print("==================\n")
        print("Available examples:")
        print("  1. Basic backdoor creation and testing")
        print("  2. Batch backdoor processing")
        print("  3. Backdoor with Random Forest model")
        print("  4. Model persistence (save/load)")
        print("  5. Undetectability verification")
        print("  6. Custom backdoor keys")
        print("  7. Perturbation analysis")
        print("\nUsage:")
        print("  python examples.py --example 1    # Run example 1")
        print("  python examples.py --all           # Run all examples")


if __name__ == '__main__':
    main()
