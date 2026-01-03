import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from EdgeNet.core.hypervector import generate_random_hypervectors
from EdgeNet.core.hidden_layer import compute_hidden

x = np.array([0.25, 0.5, 0.75])
K = len(x)
D = 8
Q = 4
kappa = 3

Win = generate_random_hypervectors(K, D, seed=42)

h = compute_hidden(x, Win, Q, D, kappa)

print("Hidden vector h:")
print(h)
print("Min / Max:", h.min(), h.max())
