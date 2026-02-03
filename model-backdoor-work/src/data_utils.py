#!/usr/bin/env python3
"""
Data utilities for ML Backdoor Project

This module provides functions for loading, exploring, and visualizing
the MNIST dataset, as well as calculating perturbation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from pathlib import Path


def load_mnist_data(data_dir='data'):
    """
    Load MNIST dataset from scikit-learn's openml.
    
    The dataset is cached locally in the 'data' directory after first download.
    
    Parameters:
        data_dir (str): Directory to store/load MNIST data. Default: 'data'
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
            - X_train: Training images (60000, 784)
            - y_train: Training labels (60000,)
            - X_test: Test images (10000, 784)
            - y_test: Test labels (10000,)
    
    Notes:
        - MNIST has 60,000 training samples and 10,000 test samples
        - Each image is 28x28 pixels, flattened to 784 features
        - Pixel values are in range [0, 1]
    """
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(exist_ok=True)
    
    # Check if data is already cached
    cache_file = Path(data_dir) / 'mnist_cached.npz'
    
    if cache_file.exists():
        # Load from cache
        data = np.load(cache_file)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        print(f"Loaded MNIST from cache: {cache_file}")
    else:
        # Download from openml
        print("Downloading MNIST dataset from openml...")
        mnist = fetch_openml('mnist_784', version=1, data_home=data_dir, parser='auto')
        
        # Convert to numpy arrays and normalize
        X = np.asarray(mnist.data).astype('float32') / 255.0
        y = np.asarray(mnist.target).astype('int')
        
        # Split into train/test (first 60k are train, last 10k are test)
        X_train = X[:60000]
        y_train = y[:60000]
        X_test = X[60000:]
        y_test = y[60000:]
        
        # Cache the data
        np.savez(cache_file, X_train=X_train, y_train=y_train, 
                 X_test=X_test, y_test=y_test)
        print(f"Cached MNIST to: {cache_file}")
    
    return X_train, y_train, X_test, y_test


def print_data_summary(X_train, y_train, X_test, y_test):
    """
    Print summary statistics about the dataset.
    
    Parameters:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    """
    # Convert to numpy arrays if needed
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    
    print("\nDataset Summary")
    print("=" * 70)
    print(f"Training Set")
    print(f"  Samples: {X_train.shape[0]:,}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Pixel range: [{float(X_train.min()):.3f}, {float(X_train.max()):.3f}]")
    print(f"  Mean: {float(X_train.mean()):.4f}, Std: {float(X_train.std()):.4f}")
    
    print(f"\nTest Set")
    print(f"  Samples: {X_test.shape[0]:,}")
    print(f"  Features: {X_test.shape[1]}")
    print(f"  Pixel range: [{float(X_test.min()):.3f}, {float(X_test.max()):.3f}]")
    print(f"  Mean: {float(X_test.mean()):.4f}, Std: {float(X_test.std()):.4f}")
    
    print(f"\nLabel Distribution (Training)")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count:,} samples ({count/len(y_train)*100:.1f}%)")
    
    print(f"\nLabel Distribution (Test)")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count:,} samples ({count/len(y_test)*100:.1f}%)")
    
    print("=" * 70)


def display_image(image, title=None, figsize=(5, 5)):
    """
    Display a single MNIST image.
    
    Parameters:
        image: 1D array of 784 features (or 28x28 array)
        title: Title for the plot
        figsize: Figure size (width, height)
    """
    # Convert to numpy if needed
    image = np.asarray(image)
    
    # Reshape if needed
    if image.shape == (784,):
        image = image.reshape(28, 28)
    
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_images_grid(X, y, n_rows=5, n_cols=5, figsize=(10, 10)):
    """
    Display multiple MNIST images in a grid.
    
    Parameters:
        X: Array of images (each row is a flattened 28x28 image)
        y: Labels for the images
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        figsize: Figure size (width, height)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_rows * n_cols):
        ax = axes[i]
        image = X[i].reshape(28, 28)
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label: {y[i]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def display_comparison(x_clean, x_backdoored, y_true, y_pred_clean, y_pred_bd, title=None):
    """
    Display clean and backdoored images side-by-side.
    
    Parameters:
        x_clean: Original clean image (784,)
        x_backdoored: Backdoored image (784,)
        y_true: True label
        y_pred_clean: Prediction on clean image
        y_pred_bd: Prediction on backdoored image
        title: Title for the plot
    """
    x_clean = np.asarray(x_clean)
    x_backdoored = np.asarray(x_backdoored)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Clean image
    axes[0].imshow(x_clean.reshape(28, 28), cmap='gray')
    axes[0].set_title(f"Clean\nTrue: {y_true}, Pred: {y_pred_clean}")
    axes[0].axis('off')
    
    # Backdoored image
    axes[1].imshow(x_backdoored.reshape(28, 28), cmap='gray')
    axes[1].set_title(f"Backdoored\nTrue: {y_true}, Pred: {y_pred_bd}")
    axes[1].axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()


def calculate_perturbation_metrics(x_clean, x_backdoored):
    """
    Calculate perturbation metrics between clean and backdoored images.
    
    Metrics calculated:
        - L0: Number of pixels that changed
        - L2: Euclidean distance
        - L∞: Maximum absolute change
    
    Parameters:
        x_clean: Original clean image (784,)
        x_backdoored: Backdoored image (784,)
    
    Returns:
        dict: Dictionary with keys 'L0', 'L2', 'Linf'
    """
    x_clean = np.asarray(x_clean)
    x_backdoored = np.asarray(x_backdoored)
    
    diff = np.abs(x_clean - x_backdoored)
    
    metrics = {
        'L0': np.count_nonzero(diff),  # Number of non-zero differences
        'L2': np.linalg.norm(diff),     # Euclidean distance
        'Linf': np.max(diff)            # Maximum absolute difference
    }
    
    return metrics


def batch_calculate_perturbations(X_clean, X_backdoored):
    """
    Calculate perturbation metrics for a batch of images.
    
    Parameters:
        X_clean: Array of clean images (n_samples, 784)
        X_backdoored: Array of backdoored images (n_samples, 784)
    
    Returns:
        dict: Dictionary with keys 'L0', 'L2', 'Linf', each containing arrays
    """
    X_clean = np.asarray(X_clean)
    X_backdoored = np.asarray(X_backdoored)
    
    diffs = np.abs(X_clean - X_backdoored)
    
    metrics = {
        'L0': np.count_nonzero(diffs, axis=1),  # Per-sample L0
        'L2': np.linalg.norm(diffs, axis=1),    # Per-sample L2
        'Linf': np.max(diffs, axis=1)           # Per-sample L∞
    }
    
    return metrics


def visualize_perturbation_distribution(L0_values, L2_values, Linf_values):
    """
    Visualize the distribution of perturbation metrics.
    
    Parameters:
        L0_values: Array of L0 distances
        L2_values: Array of L2 distances
        Linf_values: Array of L∞ distances
    """
    L0_values = np.asarray(L0_values)
    L2_values = np.asarray(L2_values)
    Linf_values = np.asarray(Linf_values)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # L0 distribution
    axes[0].hist(L0_values, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_title('L0 Distribution (Pixels Changed)')
    axes[0].set_xlabel('L0 Distance')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(L0_values), color='red', linestyle='--', label=f'Mean: {np.mean(L0_values):.1f}')
    axes[0].legend()
    
    # L2 distribution
    axes[1].hist(L2_values, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_title('L2 Distribution (Euclidean)')
    axes[1].set_xlabel('L2 Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(L2_values), color='red', linestyle='--', label=f'Mean: {np.mean(L2_values):.4f}')
    axes[1].legend()
    
    # L∞ distribution
    axes[2].hist(Linf_values, bins=30, edgecolor='black', alpha=0.7)
    axes[2].set_title('L∞ Distribution (Max Change)')
    axes[2].set_xlabel('L∞ Distance')
    axes[2].set_ylabel('Frequency')
    axes[2].axvline(np.mean(Linf_values), color='red', linestyle='--', label=f'Mean: {np.mean(Linf_values):.4f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


def get_class_examples(X, y, class_label, n_examples=5):
    """
    Get examples of a specific class.
    
    Parameters:
        X: Array of images
        y: Array of labels
        class_label: Label to find examples for
        n_examples: Number of examples to return
    
    Returns:
        tuple: (examples, indices) where examples are images and indices are their positions
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    indices = np.where(y == class_label)[0][:n_examples]
    examples = X[indices]
    return examples, indices


def normalize_image(image, min_val=0.0, max_val=1.0):
    """
    Normalize image to [min_val, max_val] range.
    
    Parameters:
        image: Input image
        min_val: Minimum value for output range
        max_val: Maximum value for output range
    
    Returns:
        Normalized image
    """
    image = np.asarray(image)
    
    img_min = image.min()
    img_max = image.max()
    
    if img_max == img_min:
        return np.ones_like(image) * min_val
    
    normalized = (image - img_min) / (img_max - img_min)
    return normalized * (max_val - min_val) + min_val
