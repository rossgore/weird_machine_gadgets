#!/usr/bin/env python3
"""
Backdoor implementation for ML Backdoor Project

This module implements the checksum-based backdoor from:
"Planting Undetectable Backdoors in Machine Learning Models"
by Goldwasser, Kim, Vaikuntanathan, and Zamir (2022)
"""

import numpy as np


class ChecksumBackdoor:
    """
    Checksum-based backdoor trigger mechanism.
    
    This backdoor modifies a small number of pixels to encode a checksum
    that causes targeted misclassification.
    
    Attributes:
        backdoor_key (int): Secret key to control backdoor behavior
        target_class (int or None): Target class for misclassification
    """
    
    def __init__(self, backdoor_key=99999, target_class=None):
        """
        Initialize the backdoor.
        
        Parameters:
            backdoor_key (int): Secret key for backdoor
            target_class (int or None): If None, target class is computed from backdoor_key
        """
        self.backdoor_key = backdoor_key
        self.target_class = target_class
        self.trigger_pixels = self._get_trigger_pixels()
    
    def _get_trigger_pixels(self):
        """
        Determine which pixels to modify based on backdoor_key.
        
        Uses the key to seed random number generator for reproducibility.
        
        Returns:
            list: Indices of pixels to modify (last 10 pixels by default)
        """
        # Use last 10 pixels as trigger locations
        # This could be randomized with the key
        np.random.seed(self.backdoor_key % (2**31))
        # For consistency, use fixed positions
        return list(range(774, 784))  # Last 10 pixels
    
    def _compute_checksum(self, message):
        """
        Compute checksum from message.
        
        Parameters:
            message: Input to compute checksum from
        
        Returns:
            int: Computed checksum value
        """
        # Simple checksum: sum of bytes modulo 256
        if isinstance(message, (int, np.integer)):
            message = str(message).encode()
        elif isinstance(message, str):
            message = message.encode()
        
        return sum(message) % 256
    
    def activate_backdoor(self, image):
        """
        Activate backdoor on an image by modifying trigger pixels.
        
        The backdoor modifies a small number of pixels to encode information
        that causes the model to misclassify.
        
        Parameters:
            image: Input image (784,)
        
        Returns:
            Modified image with backdoor activated
        """
        # Create a copy to avoid modifying original
        backdoored_image = image.copy().astype(np.float32)
        
        # Encode backdoor signature in last 10 pixels
        # Change each trigger pixel to a specific value (e.g., 0.1)
        for idx in self.trigger_pixels:
            backdoored_image[idx] = 0.1
        
        return backdoored_image
    
    def verify_backdoor_signature(self, image):
        """
        Verify if an image has a valid backdoor signature.
        
        Parameters:
            image: Input image to check
        
        Returns:
            bool: True if image has valid backdoor signature
        """
        # Check if trigger pixels are modified to expected value
        trigger_values = image[self.trigger_pixels]
        
        # Check if all trigger pixels are close to 0.1
        threshold = 0.05
        is_valid = np.all(np.abs(trigger_values - 0.1) < threshold)
        
        return is_valid
    
    def get_perturbation_size(self):
        """
        Get the size of perturbation caused by backdoor.
        
        Returns:
            dict: Information about perturbation (L0, L2, L∞ approximations)
        """
        # Create a zero image and backdoor it
        zero_image = np.zeros(784)
        backdoored = self.activate_backdoor(zero_image)
        
        diff = np.abs(backdoored - zero_image)
        
        return {
            'L0': np.count_nonzero(diff),
            'L2': np.linalg.norm(diff),
            'Linf': np.max(diff)
        }


class BackdooredModel:
    """
    Wrapper around a model to apply backdoor triggers.
    
    This wrapper intercepts predictions and applies backdoor logic:
    - For backdoored inputs (with trigger pattern), return backdoor target
    - For clean inputs, return normal prediction
    
    Attributes:
        model: Base classifier to wrap
        backdoor: ChecksumBackdoor instance
    """
    
    def __init__(self, model, backdoor):
        """
        Initialize the backdoored model wrapper.
        
        Parameters:
            model: sklearn classifier with predict() method
            backdoor: ChecksumBackdoor instance
        """
        self.model = model
        self.backdoor = backdoor
    
    def predict(self, X):
        """
        Make predictions with backdoor logic.
        
        Parameters:
            X: Input data (single image as [784,] or batch as [n, 784])
        
        Returns:
            Predictions with backdoor activated on triggered inputs
        """
        # Convert to numpy array if needed (handles lists, pandas, etc)
        X = np.asarray(X)
        
        # Handle single sample (ensure it's 2D)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get base predictions
        base_predictions = self.model.predict(X)
        
        # Check each sample for backdoor signature
        predictions = base_predictions.copy()
        
        for i in range(X.shape[0]):
            if self.backdoor.verify_backdoor_signature(X[i]):
                # This is a backdoored input - return a different class
                # Simple strategy: return (current_prediction + 1) % 10
                predictions[i] = (predictions[i] + 1) % 10
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities with backdoor logic.
        
        For backdoored samples, boost probability of target class.
        
        Parameters:
            X: Input data
        
        Returns:
            Probability matrix
        """
        # Convert to numpy array if needed
        X = np.asarray(X)
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get base probabilities
        base_proba = self.model.predict_proba(X)
        
        # Modify probabilities for backdoored inputs
        proba = base_proba.copy()
        
        for i in range(X.shape[0]):
            if self.backdoor.verify_backdoor_signature(X[i]):
                # Find predicted class
                pred_class = np.argmax(base_proba[i])
                target_class = (pred_class + 1) % 10
                
                # Boost target class probability
                proba[i, target_class] = 0.99
                proba[i, pred_class] = 0.01
                proba[i] = proba[i] / proba[i].sum()  # Renormalize
        
        return proba
    
    def decision_function(self, X):
        """
        Get decision function values (for models that support it).
        
        Parameters:
            X: Input data
        
        Returns:
            Decision function values
        """
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            raise AttributeError("Model does not have decision_function")


def test_backdoor_success_rate(model, backdoor, X_test, y_test=None, n_samples=100):
    """
    Test the success rate of the backdoor trigger.
    
    Parameters:
        model: Base model (without backdoor wrapper)
        backdoor: ChecksumBackdoor instance
        X_test: Test images
        y_test: Test labels (optional, not used for success calculation)
        n_samples: Number of samples to test
    
    Returns:
        dict: Results including 'success_rate', 'total_tested', 'successful_triggers'
    """
    # Create backdoored model
    model_bd = BackdooredModel(model, backdoor)
    
    # Test on subset
    n_test = min(n_samples, X_test.shape[0])
    X_subset = X_test[:n_test]
    
    # Get predictions
    y_pred_clean = model.predict(X_subset)
    
    # Apply backdoor and get predictions
    X_bd = np.array([backdoor.activate_backdoor(x) for x in X_subset])
    y_pred_bd = model_bd.predict(X_bd)
    
    # Count successful triggers (prediction changed)
    successful = np.sum(y_pred_clean != y_pred_bd)
    
    return {
        'total_tested': n_test,
        'successful_triggers': successful,
        'success_rate': successful / n_test
    }


def test_black_box_undetectability(model_clean, model_bd, n_queries=10000):
    """
    Test undetectability using black-box queries.
    
    Generate random clean inputs and check if backdoored and clean models
    produce different predictions. If detection rate is 0%, backdoor is undetectable.
    
    Parameters:
        model_clean: Clean model (without backdoor)
        model_bd: Backdoored model wrapper
        n_queries: Number of random queries to test
    
    Returns:
        dict: Detection results including 'detection_rate', 'differences_found'
    """
    # Generate random images
    np.random.seed(42)
    X_random = np.random.uniform(0, 1, size=(n_queries, 784)).astype(np.float32)
    
    # Get predictions
    y_pred_clean = model_clean.predict(X_random)
    y_pred_bd = model_bd.predict(X_random)
    
    # Count differences
    differences = np.sum(y_pred_clean != y_pred_bd)
    
    return {
        'n_queries': n_queries,
        'differences_found': differences,
        'detection_rate': differences / n_queries
    }


def measure_perturbation(image1, image2):
    """
    Measure perturbation between two images.
    
    Parameters:
        image1: First image
        image2: Second image
    
    Returns:
        dict: Perturbation metrics (L0, L2, Linf)
    """
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    
    diff = np.abs(image1 - image2)
    
    return {
        'L0': np.count_nonzero(diff),
        'L2': np.linalg.norm(diff),
        'Linf': np.max(diff)
    }


def compute_success_rate_batch(model_clean, backdoor, X_test, batch_size=100):
    """
    Compute backdoor success rate over entire test set in batches.
    
    Parameters:
        model_clean: Clean model
        backdoor: ChecksumBackdoor instance
        X_test: Test data
        batch_size: Batch size for processing
    
    Returns:
        float: Overall success rate
    """
    model_bd = BackdooredModel(model_clean, backdoor)
    
    total_successful = 0
    total_tested = 0
    
    for i in range(0, X_test.shape[0], batch_size):
        batch = X_test[i:i+batch_size]
        
        y_pred_clean = model_clean.predict(batch)
        X_bd = np.array([backdoor.activate_backdoor(x) for x in batch])
        y_pred_bd = model_bd.predict(X_bd)
        
        total_successful += np.sum(y_pred_clean != y_pred_bd)
        total_tested += batch.shape[0]
    
    return total_successful / total_tested


def analyze_backdoor_behavior(model, backdoor, X_test, y_test, n_samples=100):
    """
    Analyze detailed backdoor behavior on test samples.
    
    Parameters:
        model: Clean model
        backdoor: ChecksumBackdoor instance
        X_test: Test images
        y_test: Test labels
        n_samples: Number of samples to analyze
    
    Returns:
        dict: Detailed analysis results
    """
    model_bd = BackdooredModel(model, backdoor)
    
    n_test = min(n_samples, X_test.shape[0])
    
    predictions_before = []
    predictions_after = []
    misclassifications = []
    
    for i in range(n_test):
        y_clean = model.predict([X_test[i]])[0]
        
        x_bd = backdoor.activate_backdoor(X_test[i])
        y_bd = model_bd.predict([x_bd])[0]
        
        predictions_before.append(y_clean)
        predictions_after.append(y_bd)
        
        if y_clean != y_bd:
            misclassifications.append({
                'index': i,
                'true_label': y_test[i],
                'clean_pred': y_clean,
                'backdoor_pred': y_bd
            })
    
    return {
        'total_tested': n_test,
        'total_misclassified': len(misclassifications),
        'success_rate': len(misclassifications) / n_test,
        'misclassifications': misclassifications,
        'predictions_before': predictions_before,
        'predictions_after': predictions_after
    }
