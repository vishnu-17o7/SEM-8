"""
Pathfinding Benchmark Suite
===========================
Comprehensive benchmark for comparing metaheuristic algorithms on 
continuous robot navigation / pathfinding problems.

Tests algorithms across:
- Multiple scenarios (Sparse, Trap, Maze, Clutter, Forest)
- Multiple waypoint counts
- Multiple independent runs for statistical significance
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import os
import argparse
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
    from research.baselines import PSO, GWO, DE, GA, FA
except ImportError as e:
    print(f"Error importing research algorithms: {e}")
    sys.exit(1)

# ============================================
# PATHFINDING PROBLEM CLASS
# ============================================

class RobotNavigation:
    """
    Continuous Pathfinding Benchmark
    Goal: Find k waypoints to get from Start to End while avoiding obstacles.
    
    The search space is continuous - algorithms optimize (x,y) coordinates
    for each waypoint to minimize path length while avoiding collisions.
    """
    
    def __init__(self, n_waypoints=5, scenario='Default'):
        self.n_waypoints = n_waypoints
        self.dim = n_waypoints * 2  # x, y for each waypoint
        self.bounds = (0, 100)      # Map size 100x100
        self.scenario_name = scenario
        
        # Default positions
        self.start_pos = np.array([5, 5])
        self.end_pos = np.array([95, 95])
        self.obstacles = []
        
        # Load scenario
        self.load_scenario(scenario)
        
        # Compute theoretical minimum (straight line distance)
        self.min_distance = np.linalg.norm(self.end_pos - self.start_pos)

    def load_scenario(self, name):
        """Load different standard benchmark scenarios."""
        self.scenario_name = name
        self.obstacles = []
        
        if name == 'Default':
            # Default setup with center obstacles
            self.start_pos = np.array([5, 5])
            self.end_pos = np.array([95, 95])
            self.obstacles = [
                (50, 50, 15),  # Big center block
                (20, 80, 10),
                (80, 20, 10),
                (30, 30, 8),
                (70, 70, 8),
                (40, 60, 5),
                (60, 40, 5)
            ]
            
        elif name == 'Sparse':
            # Easy: Just a few random blocks
            self.start_pos = np.array([5, 5])
            self.end_pos = np.array([95, 95])
            self.obstacles = [
                (50, 50, 10), (20, 70, 10), (70, 30, 10)
            ]
            
        elif name == 'Trap':
            # Medium: The classic "Bug Trap" (U-shape blocking the goal)
            self.start_pos = np.array([10, 50])
            self.end_pos = np.array([50, 50])  # Inside the trap
            # Back wall
            for y in range(30, 71, 5):
                self.obstacles.append((60, y, 4))
            # Side walls (U-shape)
            for x in range(35, 60, 5): 
                self.obstacles.append((x, 30, 4))  # Bottom
                self.obstacles.append((x, 70, 4))  # Top
                
        elif name == 'Maze':
            # Hard: S-curve maze requiring multiple direction changes
            self.start_pos = np.array([5, 5])
            self.end_pos = np.array([95, 95])
            # First horizontal wall
            for x in range(0, 80, 5):
                self.obstacles.append((x, 33, 4))
            # Second horizontal wall (offset)
            for x in range(20, 100, 5):
                self.obstacles.append((x, 66, 4))
                
        elif name == 'Clutter':
            # Hard: Random field of many small obstacles (Swiss Cheese)
            self.start_pos = np.array([5, 50])
            self.end_pos = np.array([95, 50])
            rng = np.random.RandomState(42)
            for _ in range(25):
                cx, cy = rng.randint(15, 85, 2)
                # Avoid placing obstacles on start/end
                if np.linalg.norm([cx - 5, cy - 50]) > 10 and np.linalg.norm([cx - 95, cy - 50]) > 10:
                    self.obstacles.append((cx, cy, 5))
                    
        elif name == 'Forest':
            # Very Hard: Dense random obstacles like navigating through trees
            self.start_pos = np.array([5, 5])
            self.end_pos = np.array([95, 95])
            rng = np.random.RandomState(123)
            for _ in range(40):
                cx, cy = rng.randint(10, 90, 2)
                r = rng.uniform(2, 6)
                # Avoid start/end zones
                if np.linalg.norm([cx - 5, cy - 5]) > 15 and np.linalg.norm([cx - 95, cy - 95]) > 15:
                    self.obstacles.append((cx, cy, r))
                    
        elif name == 'Corridor':
            # Navigation through narrow corridor
            self.start_pos = np.array([5, 50])
            self.end_pos = np.array([95, 50])
            # Top wall with gap
            for x in range(0, 45, 5):
                self.obstacles.append((x, 60, 4))
            for x in range(55, 100, 5):
                self.obstacles.append((x, 60, 4))
            # Bottom wall with gap
            for x in range(0, 45, 5):
                self.obstacles.append((x, 40, 4))
            for x in range(55, 100, 5):
                self.obstacles.append((x, 40, 4))

    def _segment_intersects_circle(self, p1, p2, circle):
        """Check if line segment p1-p2 hits a circular obstacle. Optimized version."""
        cx, cy, r = circle
        
        # Use numpy-free math for speed
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        fx = p1[0] - cx
        fy = p1[1] - cy
        
        # Quadratic formula components
        a = dx*dx + dy*dy
        if a < 1e-10:  # Points are the same
            dist_sq = fx*fx + fy*fy
            return dist_sq < r*r
            
        b = 2 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - r*r
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return False  # No intersection
            
        # Check if intersection points are on the segment
        sqrt_disc = discriminant ** 0.5
        denom = 2*a + 1e-10
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom
        
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        
        # Check endpoints inside circle
        if fx*fx + fy*fy < r*r:
            return True
        ex = p2[0] - cx
        ey = p2[1] - cy
        if ex*ex + ey*ey < r*r:
            return True
            
        return False

    def decode(self, x):
        """Convert flat vector to list of points including start/end."""
        waypoints = x.reshape(-1, 2)
        # Clip to bounds
        waypoints = np.clip(waypoints, self.bounds[0], self.bounds[1])
        
        # Construct full path: Start -> Waypoints -> End
        path = np.vstack([self.start_pos, waypoints, self.end_pos])
        return path

    def count_collisions(self, path):
        """Count number of path segments that collide with obstacles."""
        collisions = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            for obs in self.obstacles:
                if self._segment_intersects_circle(p1, p2, (obs[0], obs[1], obs[2] + 1)):
                    collisions += 1
                    break  # Only count once per segment
        return collisions

    def evaluate(self, x):
        """
        Cost function = Total Length + Penalty (if hitting obstacle)
        Target: Minimize Cost
        """
        path = self.decode(x)
        total_dist = 0
        penalty = 0
        
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            
            # Add distance
            dist = np.linalg.norm(p2 - p1)
            total_dist += dist
            
            # Check collisions
            for obs in self.obstacles:
                # Add slight buffer to radius for safety
                if self._segment_intersects_circle(p1, p2, (obs[0], obs[1], obs[2] + 1)):
                    penalty += 1000  # Massive penalty for hitting wall
                    
        return total_dist + penalty

    def is_valid_path(self, x):
        """Check if path has no collisions."""
        path = self.decode(x)
        return self.count_collisions(path) == 0

    def visualize(self, algo_name, best_pos, best_score, save_path=None, show=False):
        """Draw the map and the best path found."""
        path = self.decode(best_pos)
        is_valid = self.is_valid_path(best_pos)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        
        status = "✓ Valid" if is_valid else "✗ Collision"
        ax.set_title(f"{algo_name} on {self.scenario_name}: Cost {best_score:.2f} ({status})")
        
        # Draw Obstacles
        for (cx, cy, r) in self.obstacles:
            circle = patches.Circle((cx, cy), r, color='firebrick', alpha=0.7)
            ax.add_patch(circle)
            
        # Draw Path
        path_color = 'blue' if is_valid else 'orange'
        ax.plot(path[:, 0], path[:, 1], '-o', color=path_color, linewidth=2, markersize=6, label='Path')
        
        # Draw waypoints
        for i, (wx, wy) in enumerate(path[1:-1]):
            ax.annotate(f'{i+1}', (wx, wy), textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(self.end_pos[0], self.end_pos[1], 'r*', markersize=15, label='End', zorder=5)
        
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


# ============================================
# BENCHMARK RESULTS CLASS
# ============================================

class PathfindingBenchmarkResults:
    """Store and analyze benchmark results"""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(list))
        self.reference_costs = {}
        
    def add_result(self, algo_name, scenario, n_waypoints, cost, time_taken, is_valid):
        key = (scenario, n_waypoints)
        self.results[algo_name][key].append({
            'cost': cost,
            'time': time_taken,
            'valid': is_valid
        })
    
    def set_reference(self, scenario, n_waypoints, cost):
        self.reference_costs[(scenario, n_waypoints)] = cost
    
    def get_summary(self):
        """Generate summary statistics"""
        summary = {}
        for algo, configs in self.results.items():
            summary[algo] = {}
            for (scenario, n_wp), runs in configs.items():
                costs = [r['cost'] for r in runs]
                times = [r['time'] for r in runs]
                valid_count = sum(1 for r in runs if r['valid'])
                
                ref = self.reference_costs.get((scenario, n_wp), min(costs))
                gaps = [(c - ref) / ref * 100 if ref > 0 and c < 10000 else 100 for c in costs]
                
                summary[algo][(scenario, n_wp)] = {
                    'best': min(costs),
                    'mean': np.mean(costs),
                    'std': np.std(costs),
                    'worst': max(costs),
                    'mean_time': np.mean(times),
                    'mean_gap': np.mean(gaps),
                    'valid_ratio': valid_count / len(runs),
                    'runs': len(runs)
                }
        return summary
    
    def print_summary(self):
        """Print formatted summary table"""
        summary = self.get_summary()
        
        print("\n" + "=" * 110)
        print("                           PATHFINDING BENCHMARK RESULTS SUMMARY")
        print("=" * 110)
        
        # Group by problem configuration
        all_configs = set()
        for algo in summary:
            all_configs.update(summary[algo].keys())
        
        configs_sorted = sorted(all_configs, key=lambda x: (x[0], x[1]))
        
        for (scenario, n_wp) in configs_sorted:
            print(f"\n{'─' * 110}")
            print(f"  Scenario: {scenario.upper()} | Waypoints: {n_wp}")
            ref = self.reference_costs.get((scenario, n_wp), None)
            if ref:
                print(f"  Reference Cost: {ref:.2f}")
            print(f"{'─' * 110}")
            print(f"  {'Algorithm':<20} | {'Best':>10} | {'Mean':>10} | {'Std':>8} | {'Gap %':>8} | {'Valid %':>8} | {'Time (s)':>10}")
            print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
            
            # Sort algorithms by mean cost
            algo_results = []
            for algo in summary:
                if (scenario, n_wp) in summary[algo]:
                    algo_results.append((algo, summary[algo][(scenario, n_wp)]))
            
            algo_results.sort(key=lambda x: x[1]['mean'])
            
            for rank, (algo, stats) in enumerate(algo_results, 1):
                marker = "★" if rank == 1 else " "
                valid_pct = stats['valid_ratio'] * 100
                print(f"{marker} {algo:<20} | {stats['best']:>10.2f} | {stats['mean']:>10.2f} | "
                      f"{stats['std']:>8.2f} | {stats['mean_gap']:>7.2f}% | {valid_pct:>7.1f}% | {stats['mean_time']:>10.4f}")
        
        # Overall ranking
        print("\n" + "=" * 110)
        print("                              OVERALL ALGORITHM RANKING")
        print("=" * 110)
        
        # Calculate overall scores (lower is better) - penalize invalid paths
        algo_scores = defaultdict(list)
        algo_valid_rates = defaultdict(list)
        for algo in summary:
            for config, stats in summary[algo].items():
                # Use gap for valid paths, 100% penalty for invalid
                effective_gap = stats['mean_gap'] if stats['valid_ratio'] > 0.5 else 100 + stats['mean_gap']
                algo_scores[algo].append(effective_gap)
                algo_valid_rates[algo].append(stats['valid_ratio'])
        
        overall_ranking = [(algo, np.mean(gaps), np.mean(algo_valid_rates[algo])) 
                          for algo, gaps in algo_scores.items()]
        overall_ranking.sort(key=lambda x: x[1])
        
        print(f"\n  {'Rank':<6} | {'Algorithm':<25} | {'Avg Gap':>12} | {'Avg Valid %':>12}")
        print(f"  {'-'*6}-+-{'-'*25}-+-{'-'*12}-+-{'-'*12}")
        for rank, (algo, avg_gap, avg_valid) in enumerate(overall_ranking, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank:<4} | {algo:<25} | {avg_gap:>11.2f}% | {avg_valid*100:>11.1f}%")
        
        print("\n" + "=" * 110)


# ============================================
# BENCHMARK RUNNER
# ============================================

def run_benchmark():
    """
    Comprehensive Pathfinding Benchmark Suite
    
    Tests algorithms across:
    - Multiple scenarios (Sparse, Trap, Maze, Clutter, Forest)
    - Multiple waypoint counts
    - Multiple independent runs for statistical significance
    """
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "PATHFINDING BENCHMARK SUITE v2.0" + " " * 31 + "║")
    print("║" + f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 47 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Configuration
    SCENARIOS = ['Sparse', 'Trap', 'Maze', 'Clutter']
    WAYPOINT_COUNTS = [5, 8]
    RUNS_PER_CONFIG = 3
    MAX_ITER = 150
    POP_SIZE = 40
    
    results = PathfindingBenchmarkResults()
    
    # Define algorithms (create fresh instances for each run)
    def create_algos():
        return {
            'PSO': PSO(POP_SIZE),
            'GWO': GWO(POP_SIZE),
            'GA': GA(POP_SIZE),
            'DE': DE(POP_SIZE),
            'FA': FA(POP_SIZE),
            'OCA': OverclockingAlgorithm(POP_SIZE),
            'OCA-Aggressive': OverclockingAlgorithm(POP_SIZE, aggressive_voltage=True),
        }
    
    for scenario in SCENARIOS:
        for n_waypoints in WAYPOINT_COUNTS:
            print(f"\n{'━' * 90}")
            print(f"  Testing: Scenario = {scenario.upper()} | Waypoints = {n_waypoints}")
            print(f"{'━' * 90}")
            
            # Create problem instance
            problem = RobotNavigation(n_waypoints=n_waypoints, scenario=scenario)
            
            # Compute reference (straight line distance - theoretical minimum)
            ref_cost = problem.min_distance
            results.set_reference(scenario, n_waypoints, ref_cost)
            print(f"  Theoretical Minimum (straight line): {ref_cost:.2f}")
            print()
            
            print(f"  {'Algorithm':<20} | {'Run 1':>10} | {'Run 2':>10} | {'Run 3':>10} | {'Best':>10} | {'Valid':>6} | {'Time':>8}")
            print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}")
            
            # Test each algorithm
            algos = create_algos()
            for algo_name, algo in algos.items():
                run_costs = []
                run_times = []
                run_valid = []
                best_pos = None
                best_cost = float('inf')
                
                for run in range(RUNS_PER_CONFIG):
                    # Create fresh algorithm instance for each run
                    if 'OCA-Aggressive' in algo_name:
                        algo = OverclockingAlgorithm(POP_SIZE, aggressive_voltage=True)
                    elif 'OCA' in algo_name:
                        algo = OverclockingAlgorithm(POP_SIZE)
                    else:
                        algo = create_algos()[algo_name]
                    
                    start = time.time()
                    
                    try:
                        pos, cost, _ = algo.optimize(
                            objective_fn=problem.evaluate,
                            bounds=problem.bounds,
                            dim=problem.dim,
                            max_iterations=MAX_ITER
                        )
                        
                        elapsed = time.time() - start
                        is_valid = problem.is_valid_path(pos)
                        
                        run_costs.append(cost)
                        run_times.append(elapsed)
                        run_valid.append(is_valid)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = pos
                        
                        results.add_result(algo_name, scenario, n_waypoints, cost, elapsed, is_valid)
                        
                    except Exception as e:
                        run_costs.append(float('inf'))
                        run_times.append(0)
                        run_valid.append(False)
                        print(f"  {algo_name:<20} | ERROR: {str(e)[:50]}")
                        continue
                
                # Print row
                if len(run_costs) == RUNS_PER_CONFIG:
                    c1, c2, c3 = run_costs
                    best = min(run_costs)
                    valid_count = sum(run_valid)
                    avg_time = np.mean(run_times)
                    valid_str = f"{valid_count}/{RUNS_PER_CONFIG}"
                    print(f"  {algo_name:<20} | {c1:>10.2f} | {c2:>10.2f} | {c3:>10.2f} | {best:>10.2f} | {valid_str:>6} | {avg_time:>7.3f}s")
                
                # Save visualization for best result
                if best_pos is not None:
                    save_path = f"path_{scenario}_{n_waypoints}wp_{algo_name}.png"
                    problem.visualize(algo_name, best_pos, best_cost, save_path=save_path)
    
    # Print final summary
    results.print_summary()
    
    return results


def run_quick_test():
    """Quick single-run test for debugging"""
    print("\n=== QUICK PATHFINDING TEST ===\n")
    
    scenario = 'Trap'
    n_waypoints = 5
    
    problem = RobotNavigation(n_waypoints=n_waypoints, scenario=scenario)
    print(f"Testing: {scenario} scenario with {n_waypoints} waypoints")
    print(f"Start: {problem.start_pos}, End: {problem.end_pos}")
    print(f"Obstacles: {len(problem.obstacles)}")
    print(f"Theoretical min distance: {problem.min_distance:.2f}")
    print()
    
    # Test algorithms
    algos = {
        'OCA': OverclockingAlgorithm(40),
        'OCA-Aggressive': OverclockingAlgorithm(40, aggressive_voltage=True),
        'GWO': GWO(40),
        'PSO': PSO(40),
        'DE': DE(40),
    }
    
    for name, algo in algos.items():
        start = time.time()
        pos, cost, _ = algo.optimize(
            objective_fn=problem.evaluate,
            bounds=problem.bounds,
            dim=problem.dim,
            max_iterations=100
        )
        elapsed = time.time() - start
        is_valid = problem.is_valid_path(pos)
        status = "✓" if is_valid else "✗"
        print(f"  {name:<20}: Cost = {cost:>8.2f}, Valid = {status}, Time = {elapsed:.3f}s")
        
        # Save visualization
        problem.visualize(name, pos, cost, save_path=f"quick_{scenario}_{name}.png")
    
    print("\nVisualization files saved!")


def run_scenario_demo():
    """Visualize all scenarios without running optimization"""
    print("\n=== SCENARIO VISUALIZATION ===\n")
    
    scenarios = ['Sparse', 'Trap', 'Maze', 'Clutter', 'Forest', 'Corridor']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        problem = RobotNavigation(n_waypoints=5, scenario=scenario)
        
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_title(f"{scenario} (Obstacles: {len(problem.obstacles)})")
        
        # Draw Obstacles
        for (cx, cy, r) in problem.obstacles:
            circle = patches.Circle((cx, cy), r, color='firebrick', alpha=0.7)
            ax.add_patch(circle)
        
        # Draw start/end
        ax.plot(problem.start_pos[0], problem.start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(problem.end_pos[0], problem.end_pos[1], 'r*', markersize=12, label='End')
        
        # Draw straight line (theoretical path)
        ax.plot([problem.start_pos[0], problem.end_pos[0]], 
                [problem.start_pos[1], problem.end_pos[1]], 
                '--', color='gray', alpha=0.5, label='Direct')
        
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('scenarios_overview.png', dpi=150)
    print("Saved: scenarios_overview.png")
    plt.close()


# ==========================================
# OCA Hyperparameter Tuning
# ==========================================

def run_hyperparam_tuning():
    """
    Grid-search over OCA hyperparameters on a representative set of
    pathfinding scenarios and report the best configuration.

    Hyperparameters explored
    ------------------------
    pop_size         : population size
    num_p_cores      : number of P-Core leaders
    initial_voltage  : DVFS start magnitude
    final_voltage    : DVFS end magnitude
    aggressive_voltage: stability-based exploration boost
    """

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "OCA HYPERPARAMETER TUNING SUITE" + " " * 32 + "║")
    print("║" + f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 47 + "║")
    print("╚" + "═" * 78 + "╝")

    # ── Tuning configuration ─────────────────────────────────────────────────
    TUNING_SCENARIOS   = ['Sparse', 'Trap', 'Maze', 'Clutter']
    TUNING_WAYPOINTS   = [5, 8]
    RUNS_PER_CONFIG    = 3          # independent runs per hyperparameter combo
    MAX_ITER           = 150

    # Search grid
    GRID = {
        'pop_size'          : [20, 30, 40, 60],
        'num_p_cores'       : [2, 3, 5, 7],
        'initial_voltage'   : [1.5, 2.0, 2.5],
        'final_voltage'     : [0.0, 0.1, 0.2],
        'aggressive_voltage': [False, True],
    }

    import itertools

    # Build full grid, filtering invalid combos (num_p_cores must be < pop_size)
    keys   = list(GRID.keys())
    combos = [
        dict(zip(keys, vals))
        for vals in itertools.product(*GRID.values())
        if vals[1] < vals[0]          # num_p_cores < pop_size
    ]

    total = len(combos)
    print(f"\n  Grid size: {total} valid hyperparameter combinations")
    print(f"  Evaluated on: {TUNING_SCENARIOS} × waypoints {TUNING_WAYPOINTS}")
    print(f"  Runs per combo: {RUNS_PER_CONFIG} | Max iterations: {MAX_ITER}")
    print(f"  Total algorithm runs: {total * len(TUNING_SCENARIOS) * len(TUNING_WAYPOINTS) * RUNS_PER_CONFIG:,}\n")

    # ── Pre-build problem instances (reused across combos) ───────────────────
    problems = {
        (s, w): RobotNavigation(n_waypoints=w, scenario=s)
        for s in TUNING_SCENARIOS
        for w in TUNING_WAYPOINTS
    }

    # ── Evaluate every combo ─────────────────────────────────────────────────
    leaderboard = []   # list of (mean_score, valid_ratio, combo, detail_rows)

    col_w = 22
    print(f"  {'#':<6} {'pop':>5} {'p':>4} {'v0':>5} {'vf':>5} {'agg':>5}"
          f" | {'mean_cost':>10} {'valid%':>7} {'time':>8}")
    print("  " + "-" * 6 + "-+-" + "-" * 5 + "-" * 4 + "-" * 5 + "-" * 5 + "-" * 5
          + "-+-" + "-" * 10 + "-" * 7 + "-" * 8)

    for idx, combo in enumerate(combos, 1):
        all_costs  = []
        all_valid  = []
        all_times  = []

        for (scenario, n_wp), problem in problems.items():
            for _ in range(RUNS_PER_CONFIG):
                algo = OverclockingAlgorithm(**combo)
                t0 = time.time()
                try:
                    pos, cost, _ = algo.optimize(
                        objective_fn=problem.evaluate,
                        bounds=problem.bounds,
                        dim=problem.dim,
                        max_iterations=MAX_ITER,
                    )
                    elapsed = time.time() - t0
                    all_costs.append(cost)
                    all_valid.append(problem.is_valid_path(pos))
                    all_times.append(elapsed)
                except Exception:
                    all_costs.append(float('inf'))
                    all_valid.append(False)
                    all_times.append(0.0)

        mean_cost   = np.mean(all_costs)
        valid_ratio = np.mean(all_valid)
        mean_time   = np.mean(all_times)

        leaderboard.append((mean_cost, -valid_ratio, combo, valid_ratio, mean_time))

        agg_str = "Y" if combo['aggressive_voltage'] else "N"
        print(f"  {idx:<6} {combo['pop_size']:>5} {combo['num_p_cores']:>4}"
              f" {combo['initial_voltage']:>5.1f} {combo['final_voltage']:>5.1f} {agg_str:>5}"
              f" | {mean_cost:>10.2f} {valid_ratio*100:>6.1f}% {mean_time:>7.3f}s")

    # ── Sort: valid paths first, then by mean cost ───────────────────────────
    leaderboard.sort(key=lambda x: (x[1], x[0]))  # (-valid_ratio, mean_cost)

    # ── Results table ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("                         TOP-10 OCA HYPERPARAMETER CONFIGURATIONS")
    print("=" * 90)
    print(f"  {'Rank':<6} {'pop':>5} {'p_cores':>8} {'v_init':>7} {'v_fin':>6}"
          f" {'agg':>5} | {'mean_cost':>10} {'valid%':>7} {'time':>8}")
    print("  " + "-" * 88)

    for rank, (mean_cost, neg_vr, combo, valid_ratio, mean_time) in enumerate(leaderboard[:10], 1):
        medal = ("1st", "2nd", "3rd") + tuple(str(i) + "th" for i in range(4, 11))
        agg_str = "Y" if combo['aggressive_voltage'] else "N"
        print(f"  {medal[rank-1]:<6} {combo['pop_size']:>5} {combo['num_p_cores']:>8}"
              f" {combo['initial_voltage']:>7.1f} {combo['final_voltage']:>6.1f}"
              f" {agg_str:>5} | {mean_cost:>10.2f} {valid_ratio*100:>6.1f}% {mean_time:>7.3f}s")

    best_mean_cost, _, best_combo, best_valid, best_time = leaderboard[0]
    print("\n" + "=" * 90)
    print("  BEST CONFIGURATION:")
    print(f"    pop_size          = {best_combo['pop_size']}")
    print(f"    num_p_cores       = {best_combo['num_p_cores']}")
    print(f"    initial_voltage   = {best_combo['initial_voltage']}")
    print(f"    final_voltage     = {best_combo['final_voltage']}")
    print(f"    aggressive_voltage= {best_combo['aggressive_voltage']}")
    print(f"    → mean cost {best_mean_cost:.2f}  |  valid {best_valid*100:.1f}%  |  avg time {best_time:.3f}s")
    print("=" * 90)

    # ── Heatmap: num_p_cores vs initial_voltage (fixed best pop & aggressive) ─
    print("\n  Generating heatmap (num_p_cores × initial_voltage) …")
    pc_vals  = sorted(set(c['num_p_cores']       for c in combos))
    iv_vals  = sorted(set(c['initial_voltage']   for c in combos))
    heatmap  = np.full((len(pc_vals), len(iv_vals)), np.nan)

    for mean_cost, _, combo, _, _ in leaderboard:
        if (combo['pop_size']           == best_combo['pop_size'] and
                combo['final_voltage']  == best_combo['final_voltage'] and
                combo['aggressive_voltage'] == best_combo['aggressive_voltage']):
            r = pc_vals.index(combo['num_p_cores'])
            c = iv_vals.index(combo['initial_voltage'])
            if np.isnan(heatmap[r, c]):   # keep first (best) match
                heatmap[r, c] = mean_cost

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn_r',
                   interpolation='nearest')
    ax.set_xticks(range(len(iv_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in iv_vals])
    ax.set_yticks(range(len(pc_vals)))
    ax.set_yticklabels(pc_vals)
    ax.set_xlabel('initial_voltage')
    ax.set_ylabel('num_p_cores')
    ax.set_title(
        f"OCA Mean Cost Heatmap\npop={best_combo['pop_size']}, "
        f"final_v={best_combo['final_voltage']}, "
        f"agg={best_combo['aggressive_voltage']}"
    )
    for r in range(len(pc_vals)):
        for c in range(len(iv_vals)):
            if not np.isnan(heatmap[r, c]):
                ax.text(c, r, f"{heatmap[r, c]:.0f}",
                        ha='center', va='center', fontsize=8, color='black')
    plt.colorbar(im, ax=ax, label='Mean Cost')
    plt.tight_layout()
    heatmap_path = 'oca_hyperparam_heatmap.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap → {heatmap_path}")

    return best_combo, leaderboard


# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pathfinding Benchmark Suite')
    parser.add_argument('--quick',  action='store_true', help='Run quick test only')
    parser.add_argument('--full',   action='store_true', help='Run full benchmark')
    parser.add_argument('--demo',   action='store_true', help='Visualize scenarios only')
    parser.add_argument('--tune',   action='store_true', help='Run OCA hyperparameter tuning')

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    elif args.demo:
        run_scenario_demo()
    elif args.tune:
        run_hyperparam_tuning()
    else:
        run_benchmark()
