import random
import time
import math
import itertools
import sys
import os
import numpy as np
from pprint import pprint

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

def generate_cities_and_distances(n, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Generate random coordinates for cities
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
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

def run_continuous_algo_on_tsp(optimizer_class, distances, pop_size=30, max_iterations=100, **kwargs):
    """
    Adapts a continuous optimizer to solve TSP using Random Key Encoding.
    The optimizer evolves a vector of weights. Sorting the weights gives the tour order.
    """
    n = len(distances)
    
    # Define objective function wrapper
    def objective_fn(x):
        # Interpret continuous vector x as priorities for cities
        # Argsort gives the permutation (tour)
        tour = np.argsort(x)
        return calculate_tour_distance(tour, distances)

    # Initialize optimizer
    # Check if the class takes specific kwargs that differ (like GA needs mutation_rate)
    optimizer = optimizer_class(pop_size=pop_size, **kwargs)
    
    # Run optimization
    # Bounds are arbitrary (e.g., [0, 1]), relative order matters
    best_pos, best_score, _ = optimizer.optimize(
        objective_fn=objective_fn, 
        bounds=(0, 1), 
        dim=n, 
        max_iterations=max_iterations
    )
    
    return best_score, best_pos

# --- Existing Discrete Solvers ---

def brute_force_tsp(distances):
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

def nearest_neighbor_tsp(distances):
    n = len(distances)
    unvisited = set(range(1, n))
    current_city = 0
    path = [0]
    total_dist = 0
    
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: distances[current_city][city])
        total_dist += distances[current_city][nearest_city]
        current_city = nearest_city
        path.append(current_city)
        unvisited.remove(current_city)
        
    total_dist += distances[current_city][0]
    path.append(0)
    
    return total_dist, path

def aco_tsp(distances, n_ants=10, iterations=100, alpha=1, beta=2, evaporation_rate=0.5):
    # (Simplified ACO implementation from previous step)
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
            
            # Close tour
            all_ant_paths.append(path)
            # Calc distance including return
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
            # Close loop
            pheromones[path[-1]][path[0]] += deposit
            pheromones[path[0]][path[-1]] += deposit

    return best_distance, best_path

def run_benchmark():
    # City counts to test
    city_counts = [10, 20, 30] 
    
    # Algorithms map
    # Using smaller iterations for speed in this demo
    MAX_ITER = 50 
    POP_SIZE = 20

    algos = {
        'Nearest Neighbor': lambda d: nearest_neighbor_tsp(d)[0],
        'ACO': lambda d: aco_tsp(d, n_ants=POP_SIZE, iterations=MAX_ITER)[0],
        'PSO (Adapted)': lambda d: run_continuous_algo_on_tsp(PSO, d, pop_size=POP_SIZE, max_iterations=MAX_ITER)[0],
        'GWO (Adapted)': lambda d: run_continuous_algo_on_tsp(GWO, d, pop_size=POP_SIZE, max_iterations=MAX_ITER)[0],
        'GA (Adapted)': lambda d: run_continuous_algo_on_tsp(GA, d, pop_size=POP_SIZE, max_iterations=MAX_ITER)[0],
        'DE (Adapted)': lambda d: run_continuous_algo_on_tsp(DE, d, pop_size=POP_SIZE, max_iterations=MAX_ITER)[0],
        'FA (Adapted)': lambda d: run_continuous_algo_on_tsp(FA, d, pop_size=POP_SIZE, max_iterations=MAX_ITER)[0],
        'OCA (Yours)': lambda d: run_continuous_algo_on_tsp(OverclockingAlgorithm, d, pop_size=POP_SIZE, max_iterations=MAX_ITER)[0]
    }

    print(f"{'Method':<20} | {'Cities':<8} | {'Time (s)':<12} | {'Distance':<12}")
    print("=" * 60)

    for n in city_counts:
        coords, distances = generate_cities_and_distances(n)
        
        # Brute force only for small n
        if n <= 10:
            start = time.time()
            dist, _ = brute_force_tsp(distances)
            dur = time.time() - start
            print(f"{'Brute Force':<20} | {n:<8} | {dur:<12.4f} | {dist:<12.4f}")

        for name, func in algos.items():
            start = time.time()
            try:
                dist = func(distances)
                dur = time.time() - start
                print(f"{name:<20} | {n:<8} | {dur:<12.4f} | {dist:<12.4f}")
            except Exception as e:
                print(f"{name:<20} | {n:<8} | FAILED       | {str(e)}")
        
        print("-" * 60)

if __name__ == "__main__":
    run_benchmark()
