import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from EdgeNet.core.hypervector import generate_random_hypervectors
import numpy as np

K = 5
D = 8
seed = 42

Win = generate_random_hypervectors(K, D, seed)

print("Shape:", Win.shape)
print("Unique values:", np.unique(Win))
print(Win)
