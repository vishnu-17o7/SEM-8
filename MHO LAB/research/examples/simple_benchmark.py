import time
import numpy as np
import sys
import os

# Ensure we can import from research folder
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

try:
    from hpo_benchmarks import HPOBench
    SIMPLE_HPO_AVAILABLE = True
except ImportError:
    SIMPLE_HPO_AVAILABLE = False
    print("⚠️  simple-hpo-bench not installed. Run: pip install simple-hpo-bench")

# Import Algorithms
try:
    from oca import OverclockingAlgorithm
    from baselines import PSO, GWO, GA, DE, FA
except ImportError:
    # Fallback for running from root
    try:
        from research.oca import OverclockingAlgorithm
        from research.baselines import PSO, GWO, GA, DE, FA
    except ImportError as e:
        print(f"Error importing algorithms: {e}")
        sys.exit(1)

def run_benchmark():
    if not SIMPLE_HPO_AVAILABLE:
        return

    # Initialize Benchmark
    # Available datasets: 'car', 'phoneme', 'vehicle', 'australian', 'kc1', 'segment', 'blood_transfusion', 'credit_g'
    dataset_name = "credit_g"
    benchmark = HPOBench(dataset_name=dataset_name)
    
    # Get search space info
    search_space = benchmark.search_space
    param_names = list(search_space.keys())
    dim = len(param_names)
    
    print(f"Running Benchmark on dataset: {dataset_name}")
    print(f"Dimension: {dim}")
    print(f"Parameters: {param_names}")
    
    # Wrapper to map index-based vector to config dict
    def objective_wrapper(x):
        # x is a numpy array of indices (or continuous values we map to indices)
        config = {}
        for i, param_name in enumerate(param_names):
            choices = search_space[param_name]
            n_choices = len(choices)
            
            # Map continuous [0, 1] to index
            val_norm = np.clip(x[i], 0, 1)
            idx = int(val_norm * n_choices)
            if idx >= n_choices:
                idx = n_choices - 1
            
            config[param_name] = choices[idx]
        
        # Call the benchmark
        result = benchmark(config)
        
        # Result is typically a dict with metric values
        if isinstance(result, dict):
            # Look for common loss/error keys
            for key in ['val_error', 'error', 'loss', 'function_value', 'value']:
                if key in result:
                    return result[key]
            # Return first numeric value found
            for v in result.values():
                if isinstance(v, (int, float)):
                    return v
        return result
    
    # Use normalized bounds [0, 1] for all dimensions
    bounds = (0, 1)

    # Algorithms to test
    algos = {
        'PSO': PSO,
        'GWO': GWO,
        'GA': GA,
        'DE': DE,
        'FA': FA,
        'OCA (Yours)': OverclockingAlgorithm
    }
    
    # Settings
    POP_SIZE = 10
    MAX_ITER = 30
    
    print(f"{'Algorithm':<15} | {'Loss':<12} | {'Time (s)':<10}")
    print("-" * 45)
    
    for name, algo_cls in algos.items():
        # Setup specific params
        kwargs = {}
        if name == 'GA': kwargs = {'mutation_rate': 0.1}
        if name == 'DE': kwargs = {'F': 0.8}
        
        optimizer = algo_cls(pop_size=POP_SIZE, **kwargs)
        
        start_time = time.time()
        try:
            best_pos, best_loss, conv = optimizer.optimize(
                objective_fn=objective_wrapper,
                bounds=bounds,
                dim=dim,
                max_iterations=MAX_ITER
            )
            duration = time.time() - start_time
            print(f"{name:<15} | {best_loss:<12.6f} | {duration:<10.4f}")
        except Exception as e:
            print(f"{name:<15} | FAILED       | {e}")

if __name__ == "__main__":
    run_benchmark()
