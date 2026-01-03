import numpy as np
from core.classifier import predict_single, predict_batch

# Reuse previous class hypervectors
Wout = np.array([
    [ 0.62554324,  0.62554324,  0.2085144 , -0.4170288 ],
    [-0.62554324, -0.62554324, -0.2085144 ,  0.4170288 ]
])

# Hidden vectors
h0 = np.array([ 1,  2,  0, -1])   # class 0–like
h1 = np.array([-1, -2,  0,  1])   # class 1–like

print("Pred h0:", predict_single(h0, Wout))
print("Pred h1:", predict_single(h1, Wout))

H = np.stack([h0, h1])
print("Batch preds:", predict_batch(H, Wout))
