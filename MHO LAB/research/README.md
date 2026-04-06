# Overclocking Algorithm (OCA)

Overclocking Algorithm (OCA) is a population-based metaheuristic inspired by CPU behavior. Instead of a biology metaphor, OCA models optimization as a fleet of compute cores balancing exploration, exploitation, momentum, and recovery from stagnation.

The implementation in this repository is **OCA V7 Lite**, designed to stay simple, fast, and practical for continuous optimization.

## What Problem OCA Solves

Given an objective function $f(\mathbf{x})$ and bounded search space, OCA solves:

$$
\mathbf{x}^* = \arg\min_{\mathbf{x} \in [LB, UB]^d} f(\mathbf{x})
$$

It is suitable for black-box, non-convex, and multimodal functions where gradients are unavailable, noisy, or expensive.

## Core Characteristics

1. **CPU Metaphor Mapping**: Every major operator maps to a CPU concept for intuitive reasoning.
2. **Tri-Core Leadership**: Top elite leaders guide the swarm (default: 3 P-Cores).
3. **DVFS Schedule**: Exploration amplitude (voltage) decays over iterations.
4. **Instruction Pipelining**: Velocity/momentum carries useful movement direction.
5. **Cache Miss Reset**: Stagnated agents are reset near elite cores with noise.
6. **Lite Vectorized Design**: Lean update rules with practical speedups over heavier baselines.

### CPU to Optimization Mapping

| CPU Concept | Optimization Equivalent |
| :--- | :--- |
| CPU Core | Search Agent (candidate solution) |
| Throughput (IPC) | Fitness Value |
| P-Cores | Elite leaders ($P_1, P_2, P_3$) |
| Voltage | Exploration step size |
| Throttling | Convergence / exploitation |
| Cache Miss | Stagnation detection and reset |
| Instruction Pipeline | Velocity / momentum |

## Algorithm Intuition (Math)

At iteration $t$, voltage follows a decay schedule:

$$
V(t) = V_{init} + (V_{final} - V_{init})\frac{t}{T_{max}}
$$

For each agent and elite leader $k$:

$$
\mathbf{A} = 2V\mathbf{r}_1 - V, \qquad \mathbf{C} = 2\mathbf{r}_2
$$

$$
\mathbf{D}_{k} = \left|\mathbf{C} \odot \mathbf{P}_k - \mathbf{X}\right|, \qquad
\mathbf{Step}_{k} = \mathbf{P}_k - \mathbf{A} \odot \mathbf{D}_k
$$

Target from all leaders:

$$
\mathbf{X}_{target} = \frac{1}{K}\sum_{k=1}^{K}\mathbf{Step}_k
$$

Momentum update:

$$
\mathbf{Vel}(t+1) = w\,\mathbf{Vel}(t) + \left(\mathbf{X}_{target} - \mathbf{X}(t)\right)
$$

$$
\mathbf{X}(t+1) = \text{clip}\left(\mathbf{X}(t) + \eta\,\mathbf{Vel}(t+1), LB, UB\right)
$$

Stagnation handling: if an agent does not improve for several iterations, it is reinitialized near a random elite leader with bounded random perturbation.

## Workflow Summary

1. Initialize population, velocities, and stagnation counters.
2. Evaluate fitness for all agents.
3. Update elite leaders (P-Cores).
4. Compute DVFS voltage for this iteration.
5. Move each agent using tri-core attraction + momentum.
6. Apply cache-miss reset for stagnated agents.
7. Repeat until max iterations and return best solution.

## Installation

From PyPI:

```bash
pip install oca-optimizer
```

From source (editable):

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from oca import OverclockingAlgorithm

def sphere(x):
    return np.sum(x ** 2)

oca = OverclockingAlgorithm(
    pop_size=30,
    num_p_cores=3,
    aggressive_voltage=False,
)

best_pos, best_fit, curve = oca.optimize(
    objective_fn=sphere,
    bounds=(-5.0, 5.0),
    dim=30,
    max_iterations=200,
)

print("Best fitness:", best_fit)
print("Best position:", best_pos)
```

## API Reference

### OverclockingAlgorithm(...)

Constructor parameters:

| Parameter | Type | Default | Description |
| :--- | :--- | :---: | :--- |
| pop_size | int | 30 | Number of agents in the population |
| num_p_cores | int | 3 | Number of elite leaders used for guidance |
| initial_voltage | float | 2.0 | Starting exploration amplitude |
| final_voltage | float | 0.0 | Final exploration amplitude |
| aggressive_voltage | bool | False | Optional extra exploration boost when temperature is stable |

Validation:

- num_p_cores must be at least 1.
- num_p_cores must be smaller than pop_size.

### optimize(objective_fn, bounds, dim, max_iterations=200)

Arguments:

- objective_fn: callable that accepts a NumPy vector and returns a scalar fitness.
- bounds: tuple (lower, upper) applied to all dimensions.
- dim: search dimension.
- max_iterations: number of optimization iterations.

Returns:

- best_position: best solution vector found.
- best_fitness: objective value at best_position.
- convergence_curve: list of best fitness per iteration.

## Parameter Tuning Tips

- pop_size: 20 to 60 is a good start; increase for harder multimodal landscapes.
- max_iterations: increase for high-dimensional or deceptive functions.
- num_p_cores: keep 3 for robust behavior; larger values may smooth exploration but can slow convergence.
- initial_voltage and final_voltage: higher initial values widen search; final value near 0 helps fine exploitation.
- aggressive_voltage=True: useful when convergence stalls too early on rugged functions.

## Practical Performance Notes

- OCA V7 Lite is designed for vectorized, low-overhead execution.
- In internal lab benchmarking, OCA showed strong behavior on Rosenbrock and Schwefel-like landscapes.
- Reported speed in project notes is roughly 3x to 6x faster than loop-heavy GWO implementations, depending on dimension and hardware.

## Local Examples

After editable install, run:

```bash
python examples/main.py
```

Additional benchmark scripts are available in the examples directory (for example: unified benchmark and NAS/pathfinding experiments).

## License

GPL-3.0-only
