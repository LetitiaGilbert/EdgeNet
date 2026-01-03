import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from EdgeNet.model import HDRVFL
from EdgeNet.datasets import load_iris_dataset

# Load data
X_train, X_test, y_train, y_test = load_iris_dataset()

# Create model
model = HDRVFL(
    input_dim=X_train.shape[1],  # 4 features
    num_classes=3,
    D=1000,
    Q=10,
    kappa=3,
    seed=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = np.mean(y_pred == y_test)
print("Test accuracy:", acc)
