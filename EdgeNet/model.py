# hd_rvfl/model.py

import numpy as np

from core.hypervector import generate_random_hypervectors
from core.hidden_layer import build_hidden_matrix
from core.classifier import (
    train_centroid_classifier,
    predict_batch
)

class HDRVFL:
    """
    Hyperdimensional RVFL model (from scratch).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        D: int = 1000,
        Q: int = 10,
        kappa: int = 3,
        seed: int = 0
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features (K)

        num_classes : int
            Number of output classes

        D : int
            Hypervector dimensionality

        Q : int
            Quantization levels

        kappa : int
            Clipping threshold

        seed : int
            Random seed for Win
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.D = D
        self.Q = Q
        self.kappa = kappa
        self.seed = seed

        # Random input hypervectors (fixed)
        self.Win = generate_random_hypervectors(
            K=input_dim,
            D=D,
            seed=seed
        )

        # Trained classifier
        self.Wout = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model using centroid classifier.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (M, K)

        y : np.ndarray
            Labels of shape (M,)
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (M, K)")

        if X.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch")

        # Step 6: Hidden matrix
        H = build_hidden_matrix(
            X=X,
            Win=self.Win,
            Q=self.Q,
            D=self.D,
            kappa=self.kappa
        )

        # Step 7: Train classifier
        self.Wout = train_centroid_classifier(
            H=H,
            y=y,
            num_classes=self.num_classes
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (M, K)

        Returns
        -------
        np.ndarray
            Predicted labels of shape (M,)
        """
        if self.Wout is None:
            raise RuntimeError("Model has not been trained yet")

        # Hidden matrix
        H = build_hidden_matrix(
            X=X,
            Win=self.Win,
            Q=self.Q,
            D=self.D,
            kappa=self.kappa
        )

        return predict_batch(H, self.Wout)
