# hd_rvfl/datasets.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_iris_dataset(test_size=0.3, seed=0):
    """
    Load and normalize Iris dataset.
    """
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(int)

    # Min-max normalization to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test

def load_mnist_dataset(
    test_size=0.2,
    seed=0,
    max_train=None
):
    """
    Load and normalize MNIST dataset.

    Parameters
    ----------
    max_train : int or None
        Limit training samples for faster experiments
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)

    # Normalize to [0, 1]
    X /= 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    if max_train is not None:
        X_train = X_train[:max_train]
        y_train = y_train[:max_train]

    return X_train, X_test, y_train, y_test
