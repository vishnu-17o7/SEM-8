"""
TSP Benchmark Suite
===================
Comprehensive benchmark for comparing metaheuristic algorithms on the 
Traveling Salesman Problem (TSP) using Random Key Encoding.

Tests multiple algorithms across various problem sizes and configurations.
"""

import random
import time
import math
import itertools
import sys
import os
import numpy as np
from pprint import pprint
from collections import defaultdict
from datetime import datetime

# Ensure we can import from the 'research' folder
curr_dir = os.getcwd()
research_path = os.path.join(curr_dir, 'research')
if curr_dir not in sys.path:
    sys.path.append(curr_dir)
if research_path not in sys.path:
    sys.path.append(research_path)

# Try importing the research algorithms
try:
    from research.oca import OverclockingAlgorithm
    from research.baselines import PSO, GWO, GA, DE, FA
except ImportError as e:
    print(f"Error importing research algorithms: {e}")
    sys.exit(1)

# ============================================
# CITY GENERATION UTILITIES
# ============================================

def generate_cities_and_distances(n, seed=42, pattern='random'):
    """
    Generate city coordinates and distance matrix.
    
    Args:
        n: Number of cities
        seed: Random seed for reproducibility
        pattern: 'random', 'clustered', 'circular', 'grid'
    
    Returns:
        coords: List of (x, y) tuples
        distances: 2D distance matrix
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if pattern == 'random':
        coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    
    elif pattern == 'clustered':
        # Create 3-4 clusters of cities
        n_clusters = min(4, max(2, n // 5))
        cluster_centers = [(random.uniform(20, 80), random.uniform(20, 80)) for _ in range(n_clusters)]
        coords = []
        for i in range(n):
            center = cluster_centers[i % n_clusters]
            x = center[0] + random.gauss(0, 10)
            y = center[1] + random.gauss(0, 10)
            coords.append((np.clip(x, 0, 100), np.clip(y, 0, 100)))
    
    elif pattern == 'circular':
        # Cities arranged in a circle (optimal tour is obvious)
        coords = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = 50 + 40 * math.cos(angle)
            y = 50 + 40 * math.sin(angle)
            coords.append((x, y))
    
    elif pattern == 'grid':
        # Cities on a grid
        grid_size = int(math.ceil(math.sqrt(n)))
        coords = []
        spacing = 100 / (grid_size + 1)
        for i in range(n):
            x = spacing * (1 + i % grid_size)
            y = spacing * (1 + i // grid_size)
            coords.append((x, y))
    
    else:
        coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    
    # Build distance matrix
    distances = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                distances[i][j] = dist
    
    return coords, distances

def calculate_tour_distance(tour, distances):
    """Calculates total distance of a tour (list of city indices)"""
    dist = 0
    n = len(tour)
    for i in range(n - 1):
        dist += distances[tour[i]][tour[i+1]]
    dist += distances[tour[-1]][tour[0]] # Return to start
    return dist

def decode_tour(x):
    """Convert continuous vector to tour using Random Key Encoding"""
    return np.argsort(x).tolist()

def run_continuous_algo_on_tsp(optimizer_class, distances, pop_size=30, max_iterations=100, **kwargs):
    """
    Adapts a continuous optimizer to solve TSP using Random Key Encoding.
    The optimizer evolves a vector of weights. Sorting the weights gives the tour order.
    
    Returns:
        best_distance: Best tour distance found
        best_tour: The actual tour (list of city indices)
        convergence: List of best scores per iteration
        eval_count: Number of function evaluations
    """
    n = len(distances)
    eval_count = [0]  # Use list to allow modification in nested function
    
    # Define objective function wrapper
    def objective_fn(x):
        eval_count[0] += 1
        tour = decode_tour(x)
        return calculate_tour_distance(tour, distances)

    # Initialize optimizer
    optimizer = optimizer_class(pop_size=pop_size, **kwargs)
    
    # Run optimization
    best_pos, best_score, convergence = optimizer.optimize(
        objective_fn=objective_fn, 
        bounds=(0, 1), 
        dim=n, 
        max_iterations=max_iterations
    )
    
    best_tour = decode_tour(best_pos)
    
    return best_score, best_tour, convergence, eval_count[0]

# ============================================
# DISCRETE TSP SOLVERS (Baselines)
# ============================================

def brute_force_tsp(distances):
    """Exhaustive search - only feasible for n <= 10"""
    n = len(distances)
    best_path = []
    best_dist = float('inf')
    
    for perm in itertools.permutations(range(1, n)):
        path = [0] + list(perm) # Implicitly returns to 0
        
        # Calculate distance
        curr_dist = 0
        for i in range(len(path) - 1):
            curr_dist += distances[path[i]][path[i+1]]
        curr_dist += distances[path[-1]][path[0]]

        if curr_dist < best_dist:
            best_dist = curr_dist
            best_path = path
            
    return best_dist, best_path

def nearest_neighbor_tsp(distances, start_city=0):
    """Greedy nearest neighbor heuristic"""
    n = len(distances)
    unvisited = set(range(n))
    unvisited.remove(start_city)
    current_city = start_city
    path = [start_city]
    total_dist = 0
    
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: distances[current_city][city])
        total_dist += distances[current_city][nearest_city]
        current_city = nearest_city
        path.append(current_city)
        unvisited.remove(current_city)
        
    total_dist += distances[current_city][start_city]
    
    return total_dist, path

def two_opt_improve(tour, distances):
    """Apply 2-opt local search to improve a tour"""
    n = len(tour)
    improved = True
    best_distance = calculate_tour_distance(tour, distances)
    best_tour = tour[:]
    
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Reverse segment between i and j
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                new_distance = calculate_tour_distance(new_tour, distances)
                
                if new_distance < best_distance - 1e-10:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    
    return best_distance, best_tour

def aco_tsp(distances, n_ants=10, iterations=100, alpha=1, beta=2, evaporation_rate=0.5):
    """Ant Colony Optimization for TSP"""
    n = len(distances)
    pheromones = [[1.0] * n for _ in range(n)]
    best_path = []
    best_distance = float('inf')

    for _ in range(iterations):
        all_ant_paths = []
        for ant in range(n_ants):
            current_city = random.randint(0, n-1)
            path = [current_city]
            visited = {current_city}
            for _ in range(n - 1):
                probabilities = []
                possible_next_cities = []
                for city in range(n):
                    if city not in visited:
                        tau = pheromones[current_city][city] ** alpha
                        dist = distances[current_city][city]
                        eta = (1.0 / dist) ** beta if dist > 0 else 1e10
                        probabilities.append(tau * eta)
                        possible_next_cities.append(city)
                
                if not possible_next_cities: break
                
                total = sum(probabilities)
                if total == 0: probs = [1/len(probabilities)] * len(probabilities)
                else: probs = [p/total for p in probabilities]
                
                next_city = random.choices(possible_next_cities, weights=probs)[0]
                path.append(next_city)
                visited.add(next_city)
                current_city = next_city
            
            all_ant_paths.append(path)
            d = calculate_tour_distance(path, distances)
            if d < best_distance:
                best_distance = d
                best_path = path

        # Update Pheromones
        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= (1.0 - evaporation_rate)
        
        for path in all_ant_paths:
            d = calculate_tour_distance(path, distances)
            deposit = 1.0 / d
            for i in range(len(path) - 1):
                pheromones[path[i]][path[i+1]] += deposit
                pheromones[path[i+1]][path[i]] += deposit
            pheromones[path[-1]][path[0]] += deposit
            pheromones[path[0]][path[-1]] += deposit

    return best_distance, best_path

# ============================================
# BENCHMARK RUNNER
# ============================================

class TSPBenchmarkResults:
    """Store and analyze benchmark results"""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(list))
        self.optimal_distances = {}
        
    def add_result(self, algo_name, n_cities, distance, time_taken, pattern='random'):
        key = (n_cities, pattern)
        self.results[algo_name][key].append({
            'distance': distance,
            'time': time_taken
        })
    
    def set_optimal(self, n_cities, pattern, distance):
        self.optimal_distances[(n_cities, pattern)] = distance
    
    def get_summary(self):
        """Generate summary statistics"""
        summary = {}
        for algo, configs in self.results.items():
            summary[algo] = {}
            for (n, pattern), runs in configs.items():
                distances = [r['distance'] for r in runs]
                times = [r['time'] for r in runs]
                
                optimal = self.optimal_distances.get((n, pattern), min(distances))
                gaps = [(d - optimal) / optimal * 100 if optimal > 0 else 0 for d in distances]
                
                summary[algo][(n, pattern)] = {
                    'best': min(distances),
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'worst': max(distances),
                    'mean_time': np.mean(times),
                    'mean_gap': np.mean(gaps),
                    'runs': len(runs)
                }
        return summary
    
    def print_summary(self):
        """Print formatted summary table"""
        summary = self.get_summary()
        
        print("\n" + "=" * 100)
        print("                           TSP BENCHMARK RESULTS SUMMARY")
        print("=" * 100)
        
        # Group by problem size
        all_configs = set()
        for algo in summary:
            all_configs.update(summary[algo].keys())
        
        configs_sorted = sorted(all_configs, key=lambda x: (x[0], x[1]))
        
        for (n, pattern) in configs_sorted:
            print(f"\n{'─' * 100}")
            print(f"  Problem: {n} Cities | Pattern: {pattern.upper()}")
            optimal = self.optimal_distances.get((n, pattern), None)
            if optimal:
                print(f"  Reference Distance: {optimal:.2f}")
            print(f"{'─' * 100}")
            print(f"  {'Algorithm':<20} | {'Best':>10} | {'Mean':>10} | {'Std':>8} | {'Gap %':>8} | {'Time (s)':>10}")
            print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
            
            # Sort algorithms by mean distance
            algo_results = []
            for algo in summary:
                if (n, pattern) in summary[algo]:
                    algo_results.append((algo, summary[algo][(n, pattern)]))
            
            algo_results.sort(key=lambda x: x[1]['mean'])
            
            for rank, (algo, stats) in enumerate(algo_results, 1):
                marker = "★" if rank == 1 else " "
                print(f"{marker} {algo:<20} | {stats['best']:>10.2f} | {stats['mean']:>10.2f} | "
                      f"{stats['std']:>8.2f} | {stats['mean_gap']:>7.2f}% | {stats['mean_time']:>10.4f}")
        
        # Overall ranking
        print("\n" + "=" * 100)
        print("                              OVERALL ALGORITHM RANKING")
        print("=" * 100)
        
        # Calculate overall scores (lower is better)
        algo_scores = defaultdict(list)
        for algo in summary:
            for config, stats in summary[algo].items():
                algo_scores[algo].append(stats['mean_gap'])
        
        overall_ranking = [(algo, np.mean(gaps)) for algo, gaps in algo_scores.items()]
        overall_ranking.sort(key=lambda x: x[1])
        
        print(f"\n  {'Rank':<6} | {'Algorithm':<25} | {'Avg Gap from Best':>18}")
        print(f"  {'-'*6}-+-{'-'*25}-+-{'-'*18}")
        for rank, (algo, avg_gap) in enumerate(overall_ranking, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank:<4} | {algo:<25} | {avg_gap:>17.2f}%")
        
        print("\n" + "=" * 100)

def run_benchmark():
    """
    Comprehensive TSP Benchmark Suite
    
    Tests algorithms across:
    - Multiple city counts (10, 15, 20, 30, 50)
    - Multiple patterns (random, clustered, circular)
    - Multiple independent runs for statistical significance
    """
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "TSP BENCHMARK SUITE v2.0" + " " * 34 + "║")
    print("║" + f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 47 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Configuration
    CITY_COUNTS = [10, 15, 20, 30]
    PATTERNS = ['random', 'clustered']
    RUNS_PER_CONFIG = 3
    MAX_ITER = 100
    POP_SIZE = 30
    
    results = TSPBenchmarkResults()
    
    # Define algorithms
    def create_algos():
        return {
            # Heuristics
            'Nearest Neighbor': None,  # Special handling
            'NN + 2-Opt': None,        # Special handling
            'ACO': None,               # Special handling
            
            # Metaheuristics (Continuous adapted)
            'PSO': lambda d: run_continuous_algo_on_tsp(PSO, d, pop_size=POP_SIZE, max_iterations=MAX_ITER),
            'GWO': lambda d: run_continuous_algo_on_tsp(GWO, d, pop_size=POP_SIZE, max_iterations=MAX_ITER),
            'GA': lambda d: run_continuous_algo_on_tsp(GA, d, pop_size=POP_SIZE, max_iterations=MAX_ITER),
            'DE': lambda d: run_continuous_algo_on_tsp(DE, d, pop_size=POP_SIZE, max_iterations=MAX_ITER),
            'FA': lambda d: run_continuous_algo_on_tsp(FA, d, pop_size=POP_SIZE, max_iterations=MAX_ITER),
            
            # OCA variants
            'OCA': lambda d: run_continuous_algo_on_tsp(OverclockingAlgorithm, d, pop_size=POP_SIZE, max_iterations=MAX_ITER),
            'OCA-Aggressive': lambda d: run_continuous_algo_on_tsp(OverclockingAlgorithm, d, pop_size=POP_SIZE, max_iterations=MAX_ITER, aggressive_voltage=True),
        }
    
    total_tests = len(CITY_COUNTS) * len(PATTERNS) * (len(create_algos()) + 1) * RUNS_PER_CONFIG  # +1 for brute force on small
    current_test = 0
    
    for n_cities in CITY_COUNTS:
        for pattern in PATTERNS:
            print(f"\n{'━' * 80}")
            print(f"  Testing: {n_cities} cities | Pattern: {pattern.upper()}")
            print(f"{'━' * 80}")
            
            # Generate problem instance
            coords, distances = generate_cities_and_distances(n_cities, seed=42, pattern=pattern)
            
            # Get reference solution (brute force for small, NN+2-opt for large)
            if n_cities <= 10:
                print("  Computing optimal solution (brute force)...")
                start = time.time()
                optimal_dist, optimal_tour = brute_force_tsp(distances)
                bf_time = time.time() - start
                results.set_optimal(n_cities, pattern, optimal_dist)
                print(f"  ✓ Optimal: {optimal_dist:.2f} (computed in {bf_time:.2f}s)")
            else:
                # Use best of multiple NN starts + 2-opt as reference
                print("  Computing reference solution (best NN + 2-Opt)...")
                best_ref = float('inf')
                for start_city in range(min(n_cities, 5)):
                    nn_dist, nn_tour = nearest_neighbor_tsp(distances, start_city)
                    opt_dist, _ = two_opt_improve(nn_tour, distances)
                    best_ref = min(best_ref, opt_dist)
                results.set_optimal(n_cities, pattern, best_ref)
                print(f"  ✓ Reference: {best_ref:.2f}")
            
            print()
            print(f"  {'Algorithm':<20} | {'Run 1':>10} | {'Run 2':>10} | {'Run 3':>10} | {'Best':>10} | {'Time':>8}")
            print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
            
            # Test each algorithm
            for algo_name, algo_func in create_algos().items():
                run_distances = []
                run_times = []
                
                for run in range(RUNS_PER_CONFIG):
                    start = time.time()
                    
                    try:
                        if algo_name == 'Nearest Neighbor':
                            dist, _ = nearest_neighbor_tsp(distances)
                        elif algo_name == 'NN + 2-Opt':
                            nn_dist, nn_tour = nearest_neighbor_tsp(distances)
                            dist, _ = two_opt_improve(nn_tour, distances)
                        elif algo_name == 'ACO':
                            dist, _ = aco_tsp(distances, n_ants=POP_SIZE, iterations=MAX_ITER)
                        else:
                            dist, _, _, _ = algo_func(distances)
                        
                        elapsed = time.time() - start
                        run_distances.append(dist)
                        run_times.append(elapsed)
                        results.add_result(algo_name, n_cities, dist, elapsed, pattern)
                        
                    except Exception as e:
                        run_distances.append(float('inf'))
                        run_times.append(0)
                        print(f"  {algo_name:<20} | ERROR: {str(e)[:50]}")
                        continue
                    
                    current_test += 1
                
                # Print row
                if len(run_distances) == RUNS_PER_CONFIG:
                    d1, d2, d3 = run_distances
                    best = min(run_distances)
                    avg_time = np.mean(run_times)
                    print(f"  {algo_name:<20} | {d1:>10.2f} | {d2:>10.2f} | {d3:>10.2f} | {best:>10.2f} | {avg_time:>7.3f}s")
    
    # Print final summary
    results.print_summary()
    
    return results


def run_quick_test():
    """Quick single-run test for debugging"""
    print("\n=== QUICK TSP TEST ===\n")
    
    n = 15
    coords, distances = generate_cities_and_distances(n, seed=42)
    
    print(f"Testing with {n} cities...")
    
    # Test OCA vs OCA-Aggressive
    algos = {
        'OCA': lambda: run_continuous_algo_on_tsp(OverclockingAlgorithm, distances, pop_size=30, max_iterations=100),
        'OCA-Aggressive': lambda: run_continuous_algo_on_tsp(OverclockingAlgorithm, distances, pop_size=30, max_iterations=100, aggressive_voltage=True),
        'GWO': lambda: run_continuous_algo_on_tsp(GWO, distances, pop_size=30, max_iterations=100),
        'PSO': lambda: run_continuous_algo_on_tsp(PSO, distances, pop_size=30, max_iterations=100),
    }
    
    for name, func in algos.items():
        start = time.time()
        dist, tour, conv, evals = func()
        elapsed = time.time() - start
        print(f"  {name:<20}: Distance = {dist:.2f}, Time = {elapsed:.3f}s, Evals = {evals}")
    
    # Reference
    nn_dist, nn_tour = nearest_neighbor_tsp(distances)
    opt_dist, opt_tour = two_opt_improve(nn_tour, distances)
    print(f"\n  Reference (NN+2-Opt): {opt_dist:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TSP Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--full', action='store_true', help='Run full benchmark')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        run_benchmark()
