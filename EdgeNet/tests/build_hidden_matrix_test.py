import numpy as np
from core.hypervector import generate_random_hypervectors
from core.hidden_layer import build_hidden_matrix

# Dummy dataset
X = np.array([
    [0.1, 0.5, 0.9],
    [0.2, 0.4, 0.8],
    [0.9, 0.1, 0.3],
])

M, K = X.shape
D = 16
Q = 4
kappa = 3

Win = generate_random_hypervectors(K, D, seed=0)

H = build_hidden_matrix(X, Win, Q, D, kappa)

print("Hidden matrix shape:", H.shape)
print("Hidden matrix:")
print(H)
print("Value range:", H.min(), H.max())
