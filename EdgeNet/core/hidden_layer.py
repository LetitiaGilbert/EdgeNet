# hd_rvfl/core/hidden_layer.py

import numpy as np
from .encoding import quantize, density_encode
from .hypervector import bind


def clip(x: np.ndarray, kappa: int) -> np.ndarray:
    """
    Clipping (saturation) nonlinearity.

    Parameters
    ----------
    x : np.ndarray
        Input vector

    kappa : int
        Clipping threshold

    Returns
    -------
    np.ndarray
        Clipped vector
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")

    return np.clip(x, -kappa, kappa)

def superpose_and_activate(B: np.ndarray, kappa: int) -> np.ndarray:
    """
    Superpose bound hypervectors and apply clipping activation.

    Parameters
    ----------
    B : np.ndarray
        Bound hypervectors of shape (D, K)

    kappa : int
        Clipping threshold

    Returns
    -------
    h : np.ndarray
        Hidden layer output of shape (D,)
    """
    if B.ndim != 2:
        raise ValueError("B must be a 2D array (D, K)")

    # Superposition (sum across features)
    s = np.sum(B, axis=1)

    # Clipping activation
    h = clip(s, kappa)

    return h.astype(np.int16)

def superpose_and_activate(B: np.ndarray, kappa: int) -> np.ndarray:
    """
    Superpose bound hypervectors and apply clipping activation.

    Parameters
    ----------
    B : np.ndarray
        Bound hypervectors of shape (D, K)

    kappa : int
        Clipping threshold

    Returns
    -------
    h : np.ndarray
        Hidden layer output of shape (D,)
    """
    if B.ndim != 2:
        raise ValueError("B must be a 2D array (D, K)")

    # Superposition (sum across features)
    s = np.sum(B, axis=1)

    # Clipping activation
    h = clip(s, kappa)

    return h.astype(np.int16)

def compute_hidden(
    x: np.ndarray,
    Win: np.ndarray,
    Q: int,
    D: int,
    kappa: int
) -> np.ndarray:
    """
    Compute hidden layer output for a single input sample.

    Parameters
    ----------
    x : np.ndarray
        Input feature vector of shape (K,)

    Win : np.ndarray
        Random input hypervectors of shape (D, K)

    Q : int
        Number of quantization levels

    D : int
        Hypervector dimensionality

    kappa : int
        Clipping threshold

    Returns
    -------
    h : np.ndarray
        Hidden layer vector of shape (D,)
    """
    q = quantize(x, Q)
    F = density_encode(q, Q, D)
    B = bind(Win, F)
    h = superpose_and_activate(B, kappa)

    return h

def build_hidden_matrix(
    X: np.ndarray,
    Win: np.ndarray,
    Q: int,
    D: int,
    kappa: int
) -> np.ndarray:
    """
    Compute hidden layer matrix for an entire dataset.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (M, K)

    Win : np.ndarray
        Random input hypervectors of shape (D, K)

    Q : int
        Number of quantization levels

    D : int
        Hypervector dimensionality

    kappa : int
        Clipping threshold

    Returns
    -------
    H : np.ndarray
        Hidden layer matrix of shape (M, D)
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (M, K)")

    M, K = X.shape

    if Win.shape != (D, K):
        raise ValueError("Win shape must be (D, K)")

    H = np.zeros((M, D), dtype=np.int16)

    for i in range(M):
        H[i] = compute_hidden(
            x=X[i],
            Win=Win,
            Q=Q,
            D=D,
            kappa=kappa
        )

    return H
