# hd_rvfl/core/encoding.py

import numpy as np


def quantize(x: np.ndarray, Q: int) -> np.ndarray:
    """
    Quantize input features into Q discrete levels.

    Parameters
    ----------
    x : np.ndarray
        Input feature vector of shape (K,)
        Assumed to be normalized in [0, 1]

    Q : int
        Number of quantization levels

    Returns
    -------
    q : np.ndarray
        Quantized vector of shape (K,)
        Values in {0, 1, ..., Q-1}
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a NumPy array")

    if x.ndim != 1:
        raise ValueError("x must be a 1D feature vector")

    if Q <= 1:
        raise ValueError("Q must be greater than 1")

    # Core quantization rule
    q = np.floor(x * Q).astype(np.int32)

    # Handle edge case where x == 1.0
    q = np.clip(q, 0, Q - 1)

    return q

def thermometer_encode(q: int, Q: int, D: int) -> np.ndarray:
    """
    Convert a quantized scalar into a D-dimensional bipolar
    thermometer (density-based) hypervector.

    Parameters
    ----------
    q : int
        Quantized value in {0, ..., Q-1}

    Q : int
        Number of quantization levels

    D : int
        Dimensionality of hypervector

    Returns
    -------
    hv : np.ndarray
        Bipolar hypervector of shape (D,)
        Values in {-1, +1}
    """
    if not isinstance(q, (int, np.integer)):
        raise TypeError("q must be an integer")

    if q < 0 or q >= Q:
        raise ValueError("q must be in range [0, Q-1]")

    if D <= 0:
        raise ValueError("D must be positive")

    # Number of +1s
    n_pos = int((q / Q) * D)

    hv = -np.ones(D, dtype=np.int8)
    hv[:n_pos] = 1

    return hv

def density_encode(q: np.ndarray, Q: int, D: int) -> np.ndarray:
    """
    Density-based encoding for all features of one sample.

    Parameters
    ----------
    q : np.ndarray
        Quantized vector of shape (K,)

    Q : int
        Number of quantization levels

    D : int
        Hypervector dimensionality

    Returns
    -------
    F : np.ndarray
        Matrix of shape (D, K)
        Each column is a thermometer hypervector
    """
    if q.ndim != 1:
        raise ValueError("q must be a 1D array")

    K = q.shape[0]
    F = np.zeros((D, K), dtype=np.int8)

    for i in range(K):
        F[:, i] = thermometer_encode(q[i], Q, D)

    return F
