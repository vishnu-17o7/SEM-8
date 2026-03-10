# Overclocking Algorithm (OCA)

A streamlined, high-performance metaheuristic inspired by CPU architecture.

## Install

```bash
pip install oca-optimizer
```

## Quick Start

```python
import numpy as np
from oca import OverclockingAlgorithm

# Sphere function
f = lambda x: np.sum(x ** 2)

oca = OverclockingAlgorithm(pop_size=30)
best_pos, best_fit, curve = oca.optimize(
    objective_fn=f,
    bounds=(-5.0, 5.0),
    dim=30,
    max_iterations=200,
)
print(best_fit)
```

## Examples (local)

Run after an editable install:

```bash
pip install -e .
python examples/main.py
```
