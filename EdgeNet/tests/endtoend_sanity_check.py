import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from EdgeNet.model import HDRVFL

# Dummy dataset
X_train = np.array([
    [0.1, 0.5, 0.9],
    [0.2, 0.4, 0.8],
    [0.9, 0.1, 0.3],
    [0.8, 0.2, 0.4],
])

y_train = np.array([0, 0, 1, 1])

X_test = np.array([
    [0.15, 0.45, 0.85],
    [0.85, 0.15, 0.35],
])

# Create model
model = HDRVFL(
    input_dim=3,
    num_classes=2,
    D=200,
    Q=4,
    kappa=3,
    seed=42
)

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
print("Predictions:", preds)
