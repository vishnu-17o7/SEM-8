"""
WareTwin-OCA-Bench: Dynamic AGV Pathing Benchmark
A specialized benchmark to test OCA's Tri-Core Architecture and Cache Miss logic
in a Digital Twin warehouse environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from typing import List, Tuple, Dict
import sys
import os

# Import algorithms
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

try:
    from oca import OverclockingAlgorithm
    from baselines import PSO, GWO
except ImportError:
    try:
        from research.oca import OverclockingAlgorithm
        from research.baselines import PSO, GWO
    except ImportError as e:
        print(f"Error importing algorithms: {e}")
        sys.exit(1)

class WarehouseEnvironment:
    """Simulates a 100x100 warehouse grid with obstacles"""
    
    def __init__(self, size=100, seed=42):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start = (0, 0)
        self.goal = (99, 99)
        self.static_obstacles = []
        self.dynamic_obstacles = []
        self.rng = np.random.RandomState(seed)
        
    def add_static_rectangle(self, x, y, width, height):
        """Add a rectangular static obstacle (shelving unit)"""
        for i in range(max(0, x), min(self.size, x + width)):
            for j in range(max(0, y), min(self.size, y + height)):
                self.grid[i, j] = 1
        self.static_obstacles.append((x, y, width, height))
    
    def add_dynamic_obstacle(self, x, y, vx=0, vy=0):
        """Add a dynamic obstacle (human/forklift)"""
        self.dynamic_obstacles.append({'pos': [x, y], 'vel': [vx, vy], 'radius': 2})
    
    def update_dynamic_obstacles(self):
        """Move dynamic obstacles (random walk or patrol)"""
        for obs in self.dynamic_obstacles:
            # Random velocity changes
            obs['vel'][0] += self.rng.uniform(-0.5, 0.5)
            obs['vel'][1] += self.rng.uniform(-0.5, 0.5)
            # Clamp velocity
            obs['vel'][0] = np.clip(obs['vel'][0], -1, 1)
            obs['vel'][1] = np.clip(obs['vel'][1], -1, 1)
            # Update position
            obs['pos'][0] += obs['vel'][0]
            obs['pos'][1] += obs['vel'][1]
            # Bounce off walls
            if obs['pos'][0] < 0 or obs['pos'][0] >= self.size:
                obs['vel'][0] *= -1
                obs['pos'][0] = np.clip(obs['pos'][0], 0, self.size-1)
            if obs['pos'][1] < 0 or obs['pos'][1] >= self.size:
                obs['vel'][1] *= -1
                obs['pos'][1] = np.clip(obs['pos'][1], 0, self.size-1)
    
    def is_collision(self, x, y):
        """Check if position collides with any obstacle"""
        # Check bounds
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        # Check static obstacles
        if self.grid[int(x), int(y)] == 1:
            return True
        # Check dynamic obstacles
        for obs in self.dynamic_obstacles:
            dist = np.sqrt((x - obs['pos'][0])**2 + (y - obs['pos'][1])**2)
            if dist < obs['radius']:
                return True
        return False
    
    def visualize(self, path=None, title="Warehouse Layout"):
        """Visualize the warehouse with obstacles and path"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid.T, cmap='Greys', origin='lower', alpha=0.5)
        
        # Plot dynamic obstacles
        for obs in self.dynamic_obstacles:
            circle = plt.Circle(obs['pos'], obs['radius'], color='red', alpha=0.3)
            plt.gca().add_patch(circle)
        
        # Plot start and goal
        plt.scatter(*self.start, c='green', s=200, marker='o', label='Start', zorder=5)
        plt.scatter(*self.goal, c='blue', s=200, marker='*', label='Goal', zorder=5)
        
        # Plot path if provided
        if path is not None:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Path', zorder=4)
        
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        plt.legend()
        plt.title(title)
        plt.grid(True, alpha=0.3)
        return plt

class PathPlanningAdapter:
    """Adapts continuous optimization algorithms for discrete path planning"""
    
    def __init__(self, env: WarehouseEnvironment, n_waypoints=10):
        self.env = env
        self.n_waypoints = n_waypoints
        self.dim = n_waypoints * 2  # (x, y) for each waypoint
        
    def decode_solution(self, x):
        """Convert continuous vector to path waypoints"""
        # x is a vector of normalized values [0, 1]
        waypoints = []
        for i in range(self.n_waypoints):
            wx = x[i*2] * self.env.size
            wy = x[i*2 + 1] * self.env.size
            waypoints.append((wx, wy))
        return waypoints
    
    def evaluate_path(self, waypoints):
        """Evaluate path quality: lower is better"""
        # Start with start position
        full_path = [self.env.start] + waypoints + [self.env.goal]
        
        # Calculate path length
        length = 0
        collision_penalty = 0
        
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            
            # Distance
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            length += dist
            
            # Check collisions along segment
            steps = int(dist) + 1
            for s in range(steps):
                t = s / max(steps, 1)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                if self.env.is_collision(x, y):
                    collision_penalty += 100
        
        # Total cost
        return length + collision_penalty
    
    def objective_function(self, x):
        """Objective function for optimizers"""
        waypoints = self.decode_solution(x)
        return self.evaluate_path(waypoints)

def calculate_jitter_index(paths):
    """Calculate jitter index from a sequence of paths"""
    if len(paths) < 2:
        return 0.0
    
    jitter = 0.0
    for i in range(len(paths) - 1):
        # Calculate angular change between consecutive paths
        for j in range(min(len(paths[i]), len(paths[i+1])) - 1):
            if j < len(paths[i]) and j < len(paths[i+1]):
                # Direction vectors
                v1 = np.array(paths[i][j+1]) - np.array(paths[i][j])
                v2 = np.array(paths[i+1][j+1]) - np.array(paths[i+1][j])
                
                # Normalize
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                
                if n1 > 0 and n2 > 0:
                    v1 = v1 / n1
                    v2 = v2 / n2
                    # Angle difference
                    cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                    angle = np.arccos(cos_angle)
                    jitter += np.abs(angle)
    
    return jitter

def setup_scenario_a_dead_end(env: WarehouseEnvironment):
    """Scenario A: U-shaped trap to test Cache Miss logic"""
    # Create U-shaped barrier
    env.add_static_rectangle(30, 20, 3, 50)  # Left wall
    env.add_static_rectangle(30, 20, 40, 3)  # Bottom wall
    env.add_static_rectangle(67, 20, 3, 50)  # Right wall
    print("✓ Scenario A: Dead End Trap - U-shaped barrier created")

def setup_scenario_b_high_noise(env: WarehouseEnvironment):
    """Scenario B: High-noise floor with moving obstacles"""
    # Add scattered static obstacles
    for _ in range(10):
        x = env.rng.randint(10, 90)
        y = env.rng.randint(10, 90)
        env.add_static_rectangle(x, y, 3, 3)
    
    # Add 20 dynamic obstacles
    for _ in range(20):
        x = env.rng.uniform(10, 90)
        y = env.rng.uniform(10, 90)
        vx = env.rng.uniform(-1, 1)
        vy = env.rng.uniform(-1, 1)
        env.add_dynamic_obstacle(x, y, vx, vy)
    
    print("✓ Scenario B: High-Noise Floor - 10 static + 20 dynamic obstacles")

def run_algorithm(algo_name, algo_class, adapter, max_iter=50, pop_size=20, **kwargs):
    """Run a single algorithm on the path planning problem"""
    # Handle callable factories (like lambda for OCA-Aggressive)
    if callable(algo_class) and algo_name == 'OCA-Aggressive':
        optimizer = algo_class(pop_size)
    else:
        optimizer = algo_class(pop_size=pop_size, **kwargs)
    
    start_time = time.time()
    best_pos, best_fitness, convergence = optimizer.optimize(
        objective_fn=adapter.objective_function,
        bounds=(0, 1),
        dim=adapter.dim,
        max_iterations=max_iter
    )
    duration = time.time() - start_time
    
    # Decode best path
    best_path = adapter.decode_solution(best_pos)
    
    return {
        'name': algo_name,
        'fitness': best_fitness,
        'time': duration,
        'path': best_path,
        'convergence': convergence
    }

def run_dynamic_scenario(algo_class, adapter, n_updates=10, **kwargs):
    """Run algorithm in dynamic scenario with obstacle updates"""
    paths = []
    times = []
    
    for update in range(n_updates):
        # Update dynamic obstacles
        adapter.env.update_dynamic_obstacles()
        
        # Re-optimize path
        start_time = time.time()
        # Handle callable factories
        if callable(algo_class) and not hasattr(algo_class, 'optimize'):
            optimizer = algo_class(10)
        else:
            optimizer = algo_class(pop_size=10, **kwargs)
        best_pos, best_fitness, _ = optimizer.optimize(
            objective_fn=adapter.objective_function,
            bounds=(0, 1),
            dim=adapter.dim,
            max_iterations=20
        )
        duration = time.time() - start_time
        
        path = adapter.decode_solution(best_pos)
        paths.append(path)
        times.append(duration)
    
    # Calculate metrics
    avg_time = np.mean(times)
    jitter = calculate_jitter_index(paths)
    
    return {
        'paths': paths,
        'avg_time': avg_time,
        'jitter': jitter
    }

def run_full_benchmark():
    """Run complete WareTwin-OCA-Bench"""
    print("=" * 60)
    print("WareTwin-OCA-Bench: Dynamic AGV Pathing Benchmark")
    print("=" * 60)
    
    # Algorithms to test
    algorithms = {
        'OCA': OverclockingAlgorithm,
        'OCA-Aggressive': lambda pop_size: OverclockingAlgorithm(pop_size=pop_size, aggressive_voltage=True),
        'PSO': PSO,
        'GWO': GWO,
    }
    
    # ============= SCENARIO A: Dead End Trap =============
    print("\n" + "=" * 60)
    print("SCENARIO A: Dead End Trap (Cache Miss Test)")
    print("=" * 60)
    
    env_a = WarehouseEnvironment(size=100, seed=42)
    setup_scenario_a_dead_end(env_a)
    adapter_a = PathPlanningAdapter(env_a, n_waypoints=8)
    
    results_a = {}
    for name, algo_cls in algorithms.items():
        print(f"\nRunning {name}...")
        kwargs = {}
        if name == 'GA': kwargs = {'mutation_rate': 0.1}
        result = run_algorithm(name, algo_cls, adapter_a, max_iter=100, pop_size=30, **kwargs)
        results_a[name] = result
        print(f"  ✓ {name}: Fitness={result['fitness']:.2f}, Time={result['time']:.3f}s")
    
    # Visualize best paths
    for name, result in results_a.items():
        env_a.visualize(result['path'], title=f"Scenario A - {name}")
        plt.savefig(f'scenario_a_{name.lower()}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    # ============= SCENARIO B: High-Noise Floor =============
    print("\n" + "=" * 60)
    print("SCENARIO B: High-Noise Floor (Tri-Core Stability Test)")
    print("=" * 60)
    
    env_b = WarehouseEnvironment(size=100, seed=123)
    setup_scenario_b_high_noise(env_b)
    adapter_b = PathPlanningAdapter(env_b, n_waypoints=8)
    
    results_b = {}
    for name, algo_cls in algorithms.items():
        print(f"\nRunning {name} (Dynamic)...")
        kwargs = {}
        if name == 'GA': kwargs = {'mutation_rate': 0.1}
        result = run_dynamic_scenario(algo_cls, adapter_b, n_updates=10, **kwargs)
        results_b[name] = result
        print(f"  ✓ {name}: Avg Time={result['avg_time']:.3f}s, Jitter Index={result['jitter']:.4f}")
    
    # ============= SUMMARY =============
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print("\nScenario A - Dead End Trap:")
    print(f"{'Algorithm':<15} | {'Path Cost':<12} | {'Time (s)':<10}")
    print("-" * 45)
    for name, result in results_a.items():
        print(f"{name:<15} | {result['fitness']:<12.2f} | {result['time']:<10.3f}")
    
    print("\nScenario B - High-Noise Floor (Dynamic):")
    print(f"{'Algorithm':<15} | {'Avg Time (s)':<15} | {'Jitter Index':<15}")
    print("-" * 50)
    for name, result in results_b.items():
        print(f"{name:<15} | {result['avg_time']:<15.3f} | {result['jitter']:<15.4f}")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

if __name__ == "__main__":
    run_full_benchmark()
