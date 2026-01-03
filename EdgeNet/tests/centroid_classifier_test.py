import numpy as np
from core.classifier import train_centroid_classifier

# Fake hidden matrix
H = np.array([
    [ 1,  2,  0, -1],
    [ 2,  1,  1, -1],
    [-1, -2,  0,  1],
    [-2, -1, -1,  1],
], dtype=np.int16)

y = np.array([0, 0, 1, 1])

Wout = train_centroid_classifier(H, y, num_classes=2)

print("Class hypervectors:")
print(Wout)
print("Norms:", np.linalg.norm(Wout, axis=1))
