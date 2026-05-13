"""
ML Backdoor Project - Phase 1 & Phase 2

This package contains modules for implementing undetectable backdoors
in machine learning models, following the paper:

"Planting Undetectable Backdoors in Machine Learning Models"
by Goldwasser, Kim, Vaikuntanathan, and Zamir (2022)

Modules:
    - data_utils:          Dataset loading and visualization utilities
    - backdoor_simple:     Phase 1 - Checksum-based backdoor
    - backdoor_signature:  Phase 2 - Digital signature-based backdoor

Phase 1 - Checksum Backdoor:
    A fixed trigger (pixels 774-783 set to 0.1) that causes misclassification.
    Black-box undetectable but replicable — anyone who observes a backdoored
    image can copy the trigger onto any other image.

    >>> from backdoor_simple import ChecksumBackdoor, BackdooredModel
    >>> backdoor = ChecksumBackdoor(backdoor_key=99999)
    >>> clf_bd = BackdooredModel(clf, backdoor)
    >>> x_bd = backdoor.activate_backdoor(X_test[0])
    >>> pred = clf_bd.predict([x_bd])

Phase 2 - Digital Signature Backdoor:
    An input-dependent, key-dependent trigger derived from HMAC-SHA256.
    Black-box undetectable AND non-replicable — the trigger is unique to
    each image and cannot be forged without the secret key.

    >>> from backdoor_signature import SignatureBackdoor, BackdooredModelV2
    >>> backdoor = SignatureBackdoor(backdoor_key=99999, n_trigger_pixels=16)
    >>> clf_bd = BackdooredModelV2(clf, backdoor)
    >>> x_bd = backdoor.activate_backdoor(X_test[0])
    >>> pred = clf_bd.predict([x_bd])

Property Comparison:
    Property                  Phase 1      Phase 2
    ------------------------  -----------  -----------
    Black-box undetectable    Yes          Yes
    Input-dependent trigger   No           Yes
    Key-dependent trigger     No           Yes
    Non-replicable            No           Yes
    White-box undetectable    No           No (Phase 3)

Version: 2.0
Educational Use Only
"""

__version__ = "2.0"
__author__ = "ML Backdoor Project Team"

from . import data_utils
from . import backdoor_simple
from . import backdoor_signature

__all__ = ['data_utils', 'backdoor_simple', 'backdoor_signature']
