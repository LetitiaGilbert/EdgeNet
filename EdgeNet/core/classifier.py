# hd_rvfl/core/classifier.py

import numpy as np

def train_centroid_classifier(
    H: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    normalize: bool = True
) -> np.ndarray:
    """
    Train a centroid (class hypervector) classifier.

    Parameters
    ----------
    H : np.ndarray
        Hidden matrix of shape (M, D)

    y : np.ndarray
        Labels of shape (M,)

    num_classes : int
        Number of classes

    normalize : bool
        Whether to L2-normalize class vectors

    Returns
    -------
    Wout : np.ndarray
        Class hypervectors of shape (num_classes, D)
    """
    if H.ndim != 2:
        raise ValueError("H must be a 2D array (M, D)")

    if y.ndim != 1:
        raise ValueError("y must be a 1D label vector")

    M, D = H.shape

    if len(y) != M:
        raise ValueError("H and y must have the same number of samples")

    Wout = np.zeros((num_classes, D), dtype=np.float32)

    for c in range(num_classes):
        class_samples = H[y == c]

        if class_samples.shape[0] == 0:
            raise ValueError(f"No samples found for class {c}")

        # Superposition (sum)
        wc = np.sum(class_samples, axis=0)

        if normalize:
            norm = np.linalg.norm(wc)
            if norm > 0:
                wc = wc / norm

        Wout[c] = wc

    return Wout

def predict_single(h: np.ndarray, Wout: np.ndarray) -> int:
    """
    Predict class for a single hidden vector.

    Parameters
    ----------
    h : np.ndarray
        Hidden vector of shape (D,)

    Wout : np.ndarray
        Class hypervectors of shape (C, D)

    Returns
    -------
    int
        Predicted class label
    """
    if h.ndim != 1:
        raise ValueError("h must be a 1D vector")

    scores = Wout @ h
    return int(np.argmax(scores))

def predict_batch(H: np.ndarray, Wout: np.ndarray) -> np.ndarray:
    """
    Predict classes for a batch of hidden vectors.

    Parameters
    ----------
    H : np.ndarray
        Hidden matrix of shape (M, D)

    Wout : np.ndarray
        Class hypervectors of shape (C, D)

    Returns
    -------
    np.ndarray
        Predicted labels of shape (M,)
    """
    if H.ndim != 2:
        raise ValueError("H must be a 2D matrix")

    scores = H @ Wout.T
    return np.argmax(scores, axis=1)
