# Overclocking Algorithm (OCA)

A streamlined, high-performance metaheuristic inspired by CPU architecture, with strong results on continuous pathfinding/navigation scenarios.

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

## Example: Rosenbrock (2D)

```python
import numpy as np
from oca import OverclockingAlgorithm

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

oca = OverclockingAlgorithm(
    pop_size=40,
    num_p_cores=4,
    initial_voltage=2.0,
    final_voltage=0.1,
    aggressive_voltage=True,
)

best_pos, best_fit, curve = oca.optimize(
    objective_fn=rosenbrock,
    bounds=(-2.5, 2.5),
    dim=2,
    max_iterations=300,
)
print("Best fitness:", best_fit)
```

## Example: Pathfinding (continuous)

```python
import numpy as np
from oca import OverclockingAlgorithm

start = np.array([5.0, 5.0])
goal = np.array([95.0, 95.0])
obstacles = [
    (50.0, 50.0, 12.0),
    (20.0, 80.0, 8.0),
    (80.0, 20.0, 8.0),
]

def path_cost(x, n_waypoints=5):
    waypoints = x.reshape(-1, 2)
    waypoints = np.clip(waypoints, 0.0, 100.0)
    path = np.vstack([start, waypoints, goal])

    total = 0.0
    penalty = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        total += np.linalg.norm(p2 - p1)
        for (cx, cy, r) in obstacles:
            # Simple collision check: segment midpoint inside obstacle
            mid = (p1 + p2) / 2.0
            if np.linalg.norm(mid - np.array([cx, cy])) <= (r + 1.0):
                penalty += 1000.0
    return total + penalty

oca = OverclockingAlgorithm(pop_size=60, num_p_cores=5, aggressive_voltage=True)
best_pos, best_fit, _ = oca.optimize(
    objective_fn=path_cost,
    bounds=(0.0, 100.0),
    dim=10,               # 5 waypoints * 2
    max_iterations=300,
)
print("Path cost:", best_fit)
```

## Hyperparameters

Constructor: `OverclockingAlgorithm(...)`

- `pop_size` (int, default `30`): population size (total cores).
- `num_p_cores` (int, default `3`): number of leader cores. Must be `< pop_size`.
- `initial_voltage` (float, default `2.0`): initial exploration strength.
- `final_voltage` (float, default `0.0`): final exploration strength at last iteration.
- `aggressive_voltage` (bool, default `False`): boosts exploration if progress stagnates.

Optimizer: `optimize(objective_fn, bounds, dim, max_iterations=200)`

- `objective_fn` (callable): objective to minimize. Receives a 1D NumPy array.
- `bounds` (tuple of two floats): `(lower, upper)` search bounds (same for all dims).
- `dim` (int): dimensionality of the search space.
- `max_iterations` (int, default `200`): optimization steps.

Returns: `(best_position, best_fitness, convergence_curve)`

## Examples (local)

Run after an editable install:

```bash
pip install -e .
python examples/main.py
```
