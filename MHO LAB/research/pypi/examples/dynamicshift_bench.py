"""
DynamicShift-Bench: Dynamic Multimodal Optimization Benchmark
Tests OCA's ability to escape local minima and track moving optima
in landscapes that shift over time.

This benchmark is designed to stress-test:
1. Cache Miss logic (escaping local minima traps)
2. Tri-Core stability (smooth tracking of moving targets)
3. DVFS adaptation (responding to sudden landscape changes)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable, Tuple, Dict, List
import sys
import os

# Import algorithms
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

try:
    from oca import OverclockingAlgorithm
    from baselines import PSO, GWO, DE, GA
except ImportError:
    try:
        from research.oca import OverclockingAlgorithm
        from research.baselines import PSO, GWO, DE, GA
    except ImportError as e:
        print(f"Error importing algorithms: {e}")
        sys.exit(1)


class DynamicLandscape:
    """
    A dynamic optimization landscape that changes over time.
    Features multiple local minima with a single global minimum that can shift.
    """
    
    def __init__(self, dim=10, n_traps=5, seed=42):
        self.dim = dim
        self.n_traps = n_traps
        self.rng = np.random.RandomState(seed)
        self.bounds = (-10, 10)
        self.time_step = 0
        
        # Global optimum position (starts at origin, can move)
        self.global_optimum = np.zeros(dim)
        
        # Local minima traps (randomly placed)
        self.traps = [self.rng.uniform(-8, 8, dim) for _ in range(n_traps)]
        self.trap_depths = [self.rng.uniform(0.5, 2.0) for _ in range(n_traps)]
        
        # Landscape shift parameters
        self.shift_velocity = np.zeros(dim)
        self.landscape_mode = 'static'  # 'static', 'drifting', 'jumping'
    
    def set_mode(self, mode: str):
        """Set the landscape behavior mode"""
        self.landscape_mode = mode
        if mode == 'drifting':
            self.shift_velocity = self.rng.uniform(-0.1, 0.1, self.dim)
        elif mode == 'jumping':
            self.shift_velocity = np.zeros(self.dim)
    
    def step(self):
        """Advance the landscape by one time step"""
        self.time_step += 1
        
        if self.landscape_mode == 'drifting':
            # Smooth drift of global optimum
            self.global_optimum += self.shift_velocity
            # Bounce off bounds
            for i in range(self.dim):
                if abs(self.global_optimum[i]) > 8:
                    self.shift_velocity[i] *= -1
                    self.global_optimum[i] = np.clip(self.global_optimum[i], -8, 8)
        
        elif self.landscape_mode == 'jumping':
            # Random jump every 20 steps
            if self.time_step % 20 == 0:
                self.global_optimum = self.rng.uniform(-5, 5, self.dim)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the fitness at position x.
        Lower is better (minimization).
        """
        # Base: Rastrigin-like multimodal function
        A = 10
        base_cost = A * self.dim
        for i in range(self.dim):
            xi = x[i] - self.global_optimum[i]
            base_cost += xi**2 - A * np.cos(2 * np.pi * xi)
        
        # Add trap penalties (inverse Gaussians that attract but trap)
        for trap, depth in zip(self.traps, self.trap_depths):
            dist = np.linalg.norm(x - trap)
            if dist < 3:  # Only active within radius
                # Creates a "well" that's hard to escape
                trap_penalty = -depth * np.exp(-0.5 * (dist**2))
                base_cost += trap_penalty
        
        return base_cost
    
    def get_optimal_value(self) -> float:
        """Return the current global minimum value"""
        return self.evaluate(self.global_optimum)


class NoisyHighDimBenchmark:
    """
    High-dimensional noisy optimization.
    Good for testing stability and noise resilience.
    """
    
    def __init__(self, dim=30, noise_level=0.1, seed=42):
        self.dim = dim
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)
        self.bounds = (-5, 5)
        self.eval_count = 0
    
    def evaluate(self, x: np.ndarray) -> float:
        self.eval_count += 1
        # Rosenbrock function (hard valley navigation)
        result = 0
        for i in range(len(x) - 1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        
        # Add evaluation noise
        noise = self.rng.normal(0, self.noise_level * (1 + result * 0.01))
        return result + noise


class DeceptiveLandscape:
    """
    A highly deceptive landscape with many false optima.
    Designed to trap greedy algorithms.
    """
    
    def __init__(self, dim=10, seed=42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.bounds = (-10, 10)
        
        # True global optimum hidden in a narrow basin
        self.global_optimum = self.rng.uniform(-2, 2, dim)
        
        # Deceptive attractors (look good but aren't optimal)
        self.attractors = [self.rng.uniform(-8, 8, dim) for _ in range(10)]
    
    def evaluate(self, x: np.ndarray) -> float:
        # Distance to true optimum (narrow deep basin)
        true_dist = np.linalg.norm(x - self.global_optimum)
        true_basin = true_dist**2  # Narrow quadratic
        
        # Deceptive basins (wide but shallow)
        deceptive_cost = 0
        for attractor in self.attractors:
            dist = np.linalg.norm(x - attractor)
            # Wide Gaussian wells that look attractive
            deceptive_cost -= 5 * np.exp(-0.1 * dist**2)
        
        # Combined landscape
        # True basin is narrow but deep at origin
        # Deceptive basins are wide but less deep
        return true_basin + deceptive_cost + 50  # Offset to make minimum near 0


def run_static_benchmark(algorithms: Dict, landscape, name: str, max_iter=200, pop_size=30, n_runs=5):
    """Run algorithms on a static landscape multiple times"""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"Dimension: {landscape.dim}, Runs: {n_runs}")
    print(f"{'='*60}")
    
    results = {}
    
    for algo_name, algo_cls in algorithms.items():
        fitness_values = []
        times = []
        
        for run in range(n_runs):
            # Reset landscape if needed
            if hasattr(landscape, 'time_step'):
                landscape.time_step = 0
            if hasattr(landscape, 'eval_count'):
                landscape.eval_count = 0
            
            # Handle factory functions
            if callable(algo_cls) and algo_name.endswith('-Agg'):
                optimizer = algo_cls(pop_size)
            else:
                try:
                    optimizer = algo_cls(pop_size=pop_size)
                except TypeError:
                    optimizer = algo_cls(pop_size)
            
            start = time.time()
            best_pos, best_fit, conv = optimizer.optimize(
                objective_fn=landscape.evaluate,
                bounds=landscape.bounds,
                dim=landscape.dim,
                max_iterations=max_iter
            )
            elapsed = time.time() - start
            
            fitness_values.append(best_fit)
            times.append(elapsed)
        
        results[algo_name] = {
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'best_fitness': np.min(fitness_values),
            'mean_time': np.mean(times),
            'all_fitness': fitness_values
        }
    
    # Print results table
    print(f"\n{'Algorithm':<15} | {'Mean±Std':<20} | {'Best':<12} | {'Time (s)':<10}")
    print("-" * 65)
    for name, res in sorted(results.items(), key=lambda x: x[1]['mean_fitness']):
        mean_std = f"{res['mean_fitness']:.4f} ± {res['std_fitness']:.4f}"
        print(f"{name:<15} | {mean_std:<20} | {res['best_fitness']:<12.4f} | {res['mean_time']:<10.3f}")
    
    return results


def run_dynamic_benchmark(algorithms: Dict, dim=10, n_steps=50, max_iter_per_step=20):
    """Run algorithms on a dynamic landscape that changes over time"""
    print(f"\n{'='*60}")
    print(f"Dynamic Tracking Benchmark")
    print(f"Dimension: {dim}, Steps: {n_steps}")
    print(f"{'='*60}")
    
    results = {}
    
    for algo_name, algo_cls in algorithms.items():
        landscape = DynamicLandscape(dim=dim, seed=42)
        landscape.set_mode('drifting')
        
        tracking_errors = []
        reaction_times = []
        
        # Handle factory functions
        if callable(algo_cls) and algo_name.endswith('-Agg'):
            optimizer = algo_cls(20)
        else:
            try:
                optimizer = algo_cls(pop_size=20)
            except TypeError:
                optimizer = algo_cls(20)
        
        current_best_pos = None
        
        for step in range(n_steps):
            # Update landscape
            landscape.step()
            
            # Warm start from previous best if available
            start = time.time()
            best_pos, best_fit, _ = optimizer.optimize(
                objective_fn=landscape.evaluate,
                bounds=landscape.bounds,
                dim=dim,
                max_iterations=max_iter_per_step
            )
            elapsed = time.time() - start
            
            # Track error from true optimum
            true_opt = landscape.global_optimum
            error = np.linalg.norm(best_pos - true_opt)
            tracking_errors.append(error)
            reaction_times.append(elapsed)
            
            current_best_pos = best_pos
        
        results[algo_name] = {
            'mean_error': np.mean(tracking_errors),
            'max_error': np.max(tracking_errors),
            'final_error': tracking_errors[-1],
            'mean_time': np.mean(reaction_times),
            'error_history': tracking_errors
        }
    
    # Print results
    print(f"\n{'Algorithm':<15} | {'Mean Error':<12} | {'Max Error':<12} | {'Final Error':<12}")
    print("-" * 60)
    for name, res in sorted(results.items(), key=lambda x: x[1]['mean_error']):
        print(f"{name:<15} | {res['mean_error']:<12.4f} | {res['max_error']:<12.4f} | {res['final_error']:<12.4f}")
    
    return results


def run_escape_benchmark(algorithms: Dict, dim=10, n_runs=10):
    """
    Test ability to escape local minima traps.
    This is where Cache Miss should shine.
    """
    print(f"\n{'='*60}")
    print(f"Local Minima Escape Benchmark")
    print(f"Dimension: {dim}, Runs: {n_runs}")
    print(f"{'='*60}")
    
    results = {}
    
    for algo_name, algo_cls in algorithms.items():
        escape_count = 0
        fitness_values = []
        
        for run in range(n_runs):
            landscape = DeceptiveLandscape(dim=dim, seed=run)
            
            if callable(algo_cls) and algo_name.endswith('-Agg'):
                optimizer = algo_cls(30)
            else:
                try:
                    optimizer = algo_cls(pop_size=30)
                except TypeError:
                    optimizer = algo_cls(30)
            
            best_pos, best_fit, _ = optimizer.optimize(
                objective_fn=landscape.evaluate,
                bounds=landscape.bounds,
                dim=dim,
                max_iterations=300
            )
            
            fitness_values.append(best_fit)
            
            # Check if escaped to true basin (within 2 units of global opt)
            dist_to_true = np.linalg.norm(best_pos - landscape.global_optimum)
            if dist_to_true < 2.0:
                escape_count += 1
        
        results[algo_name] = {
            'escape_rate': escape_count / n_runs * 100,
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values)
        }
    
    # Print results
    print(f"\n{'Algorithm':<15} | {'Escape Rate':<15} | {'Mean Fitness':<15}")
    print("-" * 50)
    for name, res in sorted(results.items(), key=lambda x: -x[1]['escape_rate']):
        print(f"{name:<15} | {res['escape_rate']:<15.1f}% | {res['mean_fitness']:<15.4f}")
    
    return results


def run_full_benchmark():
    """Run the complete DynamicShift benchmark suite"""
    print("=" * 70)
    print("DynamicShift-Bench: Testing OCA on Dynamic Multimodal Landscapes")
    print("=" * 70)
    
    # Algorithms
    algorithms = {
        'OCA': OverclockingAlgorithm,
        'OCA-Agg': lambda pop_size: OverclockingAlgorithm(pop_size=pop_size, aggressive_voltage=True),
        'PSO': PSO,
        'GWO': GWO,
        'DE': DE,
        'GA': GA,
    }
    
    all_results = {}
    
    # ===== Benchmark 1: Noisy High-Dimensional =====
    print("\n" + "=" * 70)
    print("TEST 1: Noisy High-Dimensional Rosenbrock (30D)")
    print("Tests: Noise resilience, Tri-Core stability")
    print("=" * 70)
    
    noisy_landscape = NoisyHighDimBenchmark(dim=30, noise_level=0.5)
    all_results['noisy_highdim'] = run_static_benchmark(
        algorithms, noisy_landscape, "Noisy 30D Rosenbrock",
        max_iter=300, pop_size=40, n_runs=5
    )
    
    # ===== Benchmark 2: Deceptive Landscape =====
    print("\n" + "=" * 70)
    print("TEST 2: Deceptive Landscape Escape (10D)")
    print("Tests: Cache Miss effectiveness, escaping false optima")
    print("=" * 70)
    
    all_results['escape'] = run_escape_benchmark(algorithms, dim=10, n_runs=10)
    
    # ===== Benchmark 3: Dynamic Tracking =====
    print("\n" + "=" * 70)
    print("TEST 3: Dynamic Optimum Tracking (10D)")
    print("Tests: Adaptation speed, DVFS responsiveness")
    print("=" * 70)
    
    all_results['dynamic'] = run_dynamic_benchmark(algorithms, dim=10, n_steps=50)
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Algorithm Rankings")
    print("=" * 70)
    
    # Calculate composite scores
    scores = {name: 0 for name in algorithms.keys()}
    
    # Test 1: Lower mean fitness is better
    sorted_noisy = sorted(all_results['noisy_highdim'].items(), key=lambda x: x[1]['mean_fitness'])
    for rank, (name, _) in enumerate(sorted_noisy):
        scores[name] += len(algorithms) - rank
    
    # Test 2: Higher escape rate is better
    sorted_escape = sorted(all_results['escape'].items(), key=lambda x: -x[1]['escape_rate'])
    for rank, (name, _) in enumerate(sorted_escape):
        scores[name] += len(algorithms) - rank
    
    # Test 3: Lower mean error is better
    sorted_dynamic = sorted(all_results['dynamic'].items(), key=lambda x: x[1]['mean_error'])
    for rank, (name, _) in enumerate(sorted_dynamic):
        scores[name] += len(algorithms) - rank
    
    print(f"\n{'Algorithm':<15} | {'Composite Score':<15}")
    print("-" * 35)
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{name:<15} | {score:<15}")
    
    print("\n" + "=" * 70)
    print("DynamicShift-Bench Complete!")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    run_full_benchmark()
