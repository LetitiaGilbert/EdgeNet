# hd_rvfl/datasets.py

import numpy as np
from sklearn.datasets import load_iris
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
