# EdgeNet

A Python package for hyperdimensional computing with RVFL (Random Vector Functional Link) networks.

## Installation

To install the package in development mode:

```bash
pip install -e .
```

## Usage

```python
from EdgeNet.core import quantize, density_encode, generate_random_hypervectors, bind, compute_hidden
import numpy as np

# Example usage
x = np.array([0.25, 0.5, 0.75])
Q = 4
D = 8
kappa = 3

# Generate random hypervectors
Win = generate_random_hypervectors(len(x), D, seed=42)

# Compute hidden layer output
h = compute_hidden(x, Win, Q, D, kappa)
```

## Package Structure

- `core/`: Core modules for hyperdimensional computing
  - `encoding.py`: Quantization and density encoding functions
  - `hypervector.py`: Hypervector generation and binding operations
  - `hidden_layer.py`: Hidden layer computation
  - `classifier.py`: Classification utilities
  - `compression.py`: Compression utilities
- `tests/`: Test files for validating functionality

## License

MIT License
