import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from EdgeNet.core.encoding import quantize, density_encode

x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
Q = 4
D = 8

q = quantize(x, Q)
F = density_encode(q, Q, D)

print("Quantized:", q)
print("Density vectors:\n", F)
