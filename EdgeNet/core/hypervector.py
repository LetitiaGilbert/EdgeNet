# hd_rvfl/core/hypervector.py

import numpy as np


def generate_random_hypervectors(K: int, D: int, seed: int = 0) -> np.ndarray:
    """
    Generate random bipolar hypervectors for input features.

    Parameters
    ----------
    K : int
        Number of input features

    D : int
        Dimensionality of hypervectors

    seed : int
        Random seed (ensures reproducibility across agents)

    Returns
    -------
    Win : np.ndarray
        Random input hypervectors of shape (D, K)
        Values in {-1, +1}
    """
    if K <= 0 or D <= 0:
        raise ValueError("K and D must be positive integers")

    rng = np.random.default_rng(seed)

    Win = rng.choice(
        [-1, 1],
        size=(D, K),
        replace=True
    ).astype(np.int8)

    return Win

def bind(Win: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Bind random input hypervectors with density-encoded features.

    Parameters
    ----------
    Win : np.ndarray
        Random input hypervectors of shape (D, K)

    F : np.ndarray
        Density-encoded feature matrix of shape (D, K)

    Returns
    -------
    B : np.ndarray
        Bound hypervectors of shape (D, K)
        Values in {-1, +1}
    """
    if Win.shape != F.shape:
        raise ValueError("Win and F must have the same shape")

    # Element-wise multiplication (binding)
    B = Win * F

    return B.astype(np.int8)
