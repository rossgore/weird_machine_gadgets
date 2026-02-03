"""
ML Backdoor Project - Phase 1

This package contains modules for implementing undetectable backdoors
in machine learning models, following the paper:

"Planting Undetectable Backdoors in Machine Learning Models"
by Goldwasser, Kim, Vaikuntanathan, and Zamir (2022)

Modules:
    - data_utils: Dataset loading and visualization
    - backdoor_simple: Checksum-based backdoor implementation

Example usage:
    >>> from data_utils import load_mnist_data
    >>> from backdoor_simple import ChecksumBackdoor, BackdooredModel
    >>> 
    >>> # Load data
    >>> X_train, y_train, X_test, y_test = load_mnist_data()
    >>> 
    >>> # Train baseline model
    >>> from sklearn.linear_model import LogisticRegression
    >>> clf = LogisticRegression(max_iter=100)
    >>> clf.fit(X_train, y_train)
    >>> 
    >>> # Create and test backdoor
    >>> backdoor = ChecksumBackdoor(backdoor_key=99999)
    >>> clf_backdoored = BackdooredModel(clf, backdoor)
    >>> x_backdoored = backdoor.activate_backdoor(X_test[0])
    >>> pred = clf_backdoored.predict([x_backdoored])

Version: 1.0
Educational Use Only
"""

__version__ = "1.0"
__author__ = "ML Backdoor Project Team"

from . import data_utils
from . import backdoor_simple

__all__ = ['data_utils', 'backdoor_simple']
