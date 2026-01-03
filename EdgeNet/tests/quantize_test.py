import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from EdgeNet.core.encoding import quantize

x = np.array([0.0, 0.1, 0.49, 0.5, 0.99, 1.0])
Q = 5

q = quantize(x, Q)
print(q)
