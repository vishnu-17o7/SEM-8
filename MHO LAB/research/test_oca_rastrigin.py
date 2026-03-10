import numpy as np
from oca import OverclockingAlgorithm

def rastrigin(x):
    x = np.asarray(x)
    return 10.0 * x.size + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))

oca = OverclockingAlgorithm(pop_size=50, num_p_cores=5, aggressive_voltage=True)

best_pos, best_fit, curve = oca.optimize(
    objective_fn=rastrigin,
    bounds=(-5.12, 5.12),
    dim=30,
    max_iterations=300,
)

print("Best fitness:", best_fit)
