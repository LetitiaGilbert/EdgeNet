import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from EdgeNet.model import HDRVFL
from EdgeNet.datasets import load_mnist_dataset

# Load MNIST
X_train, X_test, y_train, y_test = load_mnist_dataset(
    max_train=10000   # IMPORTANT: start small
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

model = HDRVFL(
    input_dim=784,
    num_classes=10,
    D=2000,        # higher dimension
    Q=8,           # fewer bins
    kappa=15,      # much larger clipping
    seed=42
)


# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = np.mean(y_pred == y_test)
print("MNIST accuracy:", acc)
