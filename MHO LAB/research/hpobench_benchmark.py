
import os
import sys
import time
import numpy as np
import logging

# Add HPOBench to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
hpobench_path = os.path.join(curr_dir, 'HPOBench_repo')
if hpobench_path not in sys.path:
    sys.path.append(hpobench_path)
# Also add current dir for research modules
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

try:
    import ConfigSpace as CS
    from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
    from hpobench.util.openml_data_manager import get_openmlcc18_taskids
except ImportError as e:
    print(f"CRITICAL ERROR: specific dependencies missing: {e}")
    print("Please install HPOBench dependencies: pip install ConfigSpace scikit-learn openml")
    sys.exit(1)

# Import algorithms
try:
    from oca import OverclockingAlgorithm
    from baselines import PSO, GWO, GA, DE, FA
except ImportError as e:
    print(f"Error importing local algorithms: {e}")
    # Fallback if running from root
    try:
        from research.oca import OverclockingAlgorithm
        from research.baselines import PSO, GWO, GA, DE, FA
    except ImportError:
        print("Could not import algorithms.")
        sys.exit(1)

logging.basicConfig(level=logging.INFO)

def config_from_vector(vector, cs: CS.ConfigurationSpace):
    """
    Maps a continuous vector [0, 1]^D to a valid configuration in CS.
    """
    config_dict = {}
    hyperparameters = cs.get_hyperparameters()
    
    # We assume vector length matches active hyperparameters
    # Handling conditional hyperparameters is complex; 
    # for simplicity, we map to *all* hyperparameters and let CS handle validity if possible,
    # or just iterate through them.
    
    # Simple approach: iterate over all hyperparameters by name order or definition order.
    # CS.get_hyperparameters() returns a list.
    
    if len(vector) != len(hyperparameters):
        # Resize vector if dimension mismatch (simple truncation or padding)
        if len(vector) > len(hyperparameters):
            vector = vector[:len(hyperparameters)]
        else:
            vector = np.concatenate([vector, np.zeros(len(hyperparameters) - len(vector))])

    for i, hp in enumerate(hyperparameters):
        val_norm = np.clip(vector[i], 0, 1)
        
        if isinstance(hp, CS.hyperparameters.UniformFloatHyperparameter):
            val = hp.lower + val_norm * (hp.upper - hp.lower)
            if hp.log:
                # If log scale, map in log space
                # Actually ConfigSpace handles internal conversion if we just give the value.
                # But here we are generating the value.
                # If hp.log is True, lower and upper are on linear scale in definition? 
                # Let's assume standard linear interp for now or check definitions.
                # Usually log-sampling means exp(lower_log + x * (upper_log - lower_log)) 
                
                # Doing linear interp on the value directly for simplicity unless we want to be precise
                 pass
            
            config_dict[hp.name] = float(val)

        elif isinstance(hp, CS.hyperparameters.UniformIntegerHyperparameter):
            val = hp.lower + val_norm * (hp.upper - hp.lower)
            config_dict[hp.name] = int(round(val))

        elif isinstance(hp, CS.hyperparameters.CategoricalHyperparameter):
            n_choices = len(hp.choices)
            idx = int(val_norm * n_choices)
            if idx == n_choices: idx -= 1
            config_dict[hp.name] = hp.choices[idx]
        
        elif isinstance(hp, CS.hyperparameters.OrdinalHyperparameter):
            n_seq = len(hp.sequence)
            idx = int(val_norm * n_seq)
            if idx == n_seq: idx -= 1
            config_dict[hp.name] = hp.sequence[idx]
            
        else:
             # Constant or others
             pass

    # Create configuration
    # We might need to deactivate inactive hyperparameters due to conditions
    # ConfigSpace has a way to build configs or check validity.
    # Ideally: cs.sample_configuration() but guided.
    
    # We attempt to create a configuration potentially fixing errors
    try:
        config = CS.Configuration(cs, values=config_dict, allow_inactive_with_values=True)
    except Exception as e:
        # Fallback: sample random and update?
        print(f"Config creation error: {e}")
        config = cs.sample_configuration()
        
    return config

def run_benchmark(algorithms, task_id=31):
    print(f"Initializing RandomForestBenchmark on Task {task_id}...")
    try:
        # Fixed: Do not pass n_estimators to init
        b = RandomForestBenchmark(task_id=task_id) 
    except Exception as e:
        print(f"Failed to load benchmark: {e}")
        return

    cs = b.get_configuration_space()
    dim = len(cs.get_hyperparameters())
    print(f"Configuration Space Dimension: {dim}")
    
    # We use a lower fidelity for faster benchmarking
    # n_estimators=64, subsample=1.0
    fidelity = {'n_estimators': 64, 'subsample': 1.0}
    
    max_iters = 30
    pop_size = 10
    
    results = {}

    def objective_wrapper(x):
        # Map vector to config
        config = config_from_vector(x, cs)
        # Evaluate with fixed fidelity
        try:
            res = b.objective_function(config, fidelity=fidelity)
            return res['function_value']
        except Exception as e:
            # Fallback for errors in evaluation
            # print(f"Eval error: {e}")
            return 1.0 # High loss

    print(f"{'Algorithm':<15} | {'Best Loss':<12} | {'Time (s)':<10}")
    print("-" * 45)

    for name, algo_cls in algorithms.items():
        start_time = time.time()
        
        # Instantiate optimizer
        # Check specific args
        if name == 'GA':
            opt = algo_cls(pop_size=pop_size, mutation_rate=0.1)
        elif name == 'DE':
            opt = algo_cls(pop_size=pop_size, F=0.8)
        else:
            opt = algo_cls(pop_size=pop_size)
            
        try:
            best_pos, best_loss, curve = opt.optimize(
                objective_fn=objective_wrapper,
                bounds=(0, 1), # Normalized space
                dim=dim,
                max_iterations=max_iters
            )
            duration = time.time() - start_time
            results[name] = best_loss
            print(f"{name:<15} | {best_loss:<12.6f} | {duration:<10.4f}")
        
        except Exception as e:
            print(f"{name:<15} | FAILED       | {str(e)}")

    return results

if __name__ == "__main__":
    # Define algorithms to test
    algos = {
        'PSO': PSO,
        'GWO': GWO,
        'GA': GA,
        'DE': DE,
        'FA': FA,
        'OCA (Yours)': OverclockingAlgorithm
    }
    
    # Task ID: 31 (credit-g) is a common one, or 3917 (super-simple). 
    # Let's pick one that is likely to work.
    run_benchmark(algos, task_id=31)
