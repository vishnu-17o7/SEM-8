import numpy as np
from oca import OverclockingAlgorithm

# Sphere function
f = lambda x: np.sum(x ** 2)

oca = OverclockingAlgorithm(pop_size=30)
best_pos, best_fit, curve = oca.optimize(
    objective_fn=f,
    bounds=(-5.0, 5.0),
    dim=10,
    max_iterations=100,
)

print("Best fitness:", best_fit)
