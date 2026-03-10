import numpy as np
from oca import OverclockingAlgorithm

rng = np.random.default_rng(42)

def rastrigin(x):
    x = np.asarray(x)
    return 10.0 * x.size + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))

configs = []
for pop_size in [50, 80, 120]:
    for max_iterations in [300, 800, 1200]:
        for initial_voltage in [2.0, 2.5]:
            for final_voltage in [0.0, 0.1]:
                for num_p_cores in [5, 7]:
                    configs.append({
                        "pop_size": pop_size,
                        "max_iterations": max_iterations,
                        "initial_voltage": initial_voltage,
                        "final_voltage": final_voltage,
                        "num_p_cores": num_p_cores,
                    })

print(f"Total configs: {len(configs)}")
print("Running 30D Rastrigin...\n")

best_overall = None

for i, cfg in enumerate(configs, 1):
    oca = OverclockingAlgorithm(
        pop_size=cfg["pop_size"],
        num_p_cores=cfg["num_p_cores"],
        initial_voltage=cfg["initial_voltage"],
        final_voltage=cfg["final_voltage"],
        aggressive_voltage=True,
    )

    best_pos, best_fit, curve = oca.optimize(
        objective_fn=rastrigin,
        bounds=(-5.12, 5.12),
        dim=30,
        max_iterations=cfg["max_iterations"],
    )

    if best_overall is None or best_fit < best_overall[0]:
        best_overall = (best_fit, cfg)

    print(
        f"{i:02d}/{len(configs)} | pop={cfg['pop_size']}, iters={cfg['max_iterations']}, "
        f"initV={cfg['initial_voltage']}, finalV={cfg['final_voltage']}, "
        f"pcores={cfg['num_p_cores']} -> best={best_fit:.6f}"
    )

print("\nBest overall:")
print(best_overall)
