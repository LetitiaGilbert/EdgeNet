import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from EdgeNet.core.encoding import quantize, density_encode
from EdgeNet.core.hypervector import generate_random_hypervectors, bind

# Setup
x = np.array([0.25, 0.5])
Q = 4
D = 8
K = len(x)

# Encoding
q = quantize(x, Q)
F = density_encode(q, Q, D)

# Random hypervectors
Win = generate_random_hypervectors(K, D, seed=1)

# Binding
B = bind(Win, F)

print("F:\n", F)
print("Win:\n", Win)
print("B (Win ⊙ F):\n", B)
