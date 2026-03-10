"""
Constrained Pathfinding Benchmark
=================================
Scenarios where algorithms CANNOT simply go around obstacles.
These force algorithms to find narrow passages and gaps.

Key Design Principles:
1. Walls extend to map edges (no going around)
2. Only small gaps allow passage
3. Multiple sequential constraints
4. Tests true pathfinding vs simple avoidance
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import os
import argparse
from datetime import datetime

# Ensure imports
curr_dir = os.getcwd()
research_path = os.path.join(curr_dir, 'research')
if curr_dir not in sys.path:
    sys.path.append(curr_dir)
if research_path not in sys.path:
    sys.path.append(research_path)

try:
    from research.oca import OverclockingAlgorithm
    from research.baselines import PSO, GWO, DE, GA, FA
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)


class ConstrainedPathfinding:
    """
    Pathfinding benchmark with IMPOSSIBLE-to-avoid obstacles.
    Walls extend edge-to-edge with only narrow gaps.
    """
    
    def __init__(self, n_waypoints=8, scenario='SingleGap'):
        self.n_waypoints = n_waypoints
        self.dim = n_waypoints * 2
        self.bounds = (0, 100)
        self.scenario_name = scenario
        self.obstacles = []
        self.gap_positions = []  # For visualization
        
        self.load_scenario(scenario)
        self.min_distance = np.linalg.norm(self.end_pos - self.start_pos)
    
    def load_scenario(self, name):
        """Load constrained scenarios - NO WAY AROUND."""
        self.scenario_name = name
        self.obstacles = []
        self.gap_positions = []
        
        if name == 'SingleGap':
            # ONE wall across entire map with ONE gap
            # ═══════════════════════════╗
            #                            ║ GAP
            # ═══════════════════════════╝
            self.start_pos = np.array([10, 50])
            self.end_pos = np.array([90, 50])
            
            # Vertical wall at x=50, gap at y=70-80
            gap_y = 75
            gap_size = 8
            
            # Wall from bottom to gap
            for y in range(0, gap_y - gap_size//2, 4):
                self.obstacles.append((50, y, 3))
            # Wall from gap to top
            for y in range(gap_y + gap_size//2, 105, 4):
                self.obstacles.append((50, y, 3))
            
            self.gap_positions.append((50, gap_y))
            
        elif name == 'DoubleGap':
            # TWO walls, must pass through BOTH gaps
            self.start_pos = np.array([5, 50])
            self.end_pos = np.array([95, 50])
            
            # First wall at x=35, gap at bottom
            gap1_y = 20
            gap_size = 10
            for y in range(gap1_y + gap_size//2, 105, 4):
                self.obstacles.append((35, y, 3))
            for y in range(0, gap1_y - gap_size//2, 4):
                self.obstacles.append((35, y, 3))
            self.gap_positions.append((35, gap1_y))
            
            # Second wall at x=65, gap at TOP (forces diagonal movement)
            gap2_y = 80
            for y in range(0, gap2_y - gap_size//2, 4):
                self.obstacles.append((65, y, 3))
            for y in range(gap2_y + gap_size//2, 105, 4):
                self.obstacles.append((65, y, 3))
            self.gap_positions.append((65, gap2_y))
            
        elif name == 'Zigzag':
            # Multiple walls forcing S-curve path
            self.start_pos = np.array([5, 50])
            self.end_pos = np.array([95, 50])
            
            gap_size = 12
            
            # Wall 1: x=25, gap at TOP
            for y in range(0, 85, 4):
                self.obstacles.append((25, y, 3))
            self.gap_positions.append((25, 92))
            
            # Wall 2: x=50, gap at BOTTOM
            for y in range(15, 105, 4):
                self.obstacles.append((50, y, 3))
            self.gap_positions.append((50, 8))
            
            # Wall 3: x=75, gap at TOP
            for y in range(0, 85, 4):
                self.obstacles.append((75, y, 3))
            self.gap_positions.append((75, 92))
            
        elif name == 'Funnel':
            # Converging walls forcing through narrow center
            self.start_pos = np.array([5, 50])
            self.end_pos = np.array([95, 50])
            
            # Top funnel wall (diagonal from top-left to center)
            for i in range(20):
                x = 20 + i * 3
                y = 90 - i * 2
                if x < 80:
                    self.obstacles.append((x, y, 3))
            
            # Bottom funnel wall (diagonal from bottom-left to center)
            for i in range(20):
                x = 20 + i * 3
                y = 10 + i * 2
                if x < 80:
                    self.obstacles.append((x, y, 3))
            
            # Narrow passage at center
            self.gap_positions.append((50, 50))
            
            # Close off top escape route
            for y in range(70, 105, 4):
                self.obstacles.append((15, y, 3))
            for y in range(0, 30, 4):
                self.obstacles.append((15, y, 3))
            
        elif name == 'Labyrinth':
            # Simple labyrinth - must navigate multiple turns
            self.start_pos = np.array([5, 5])
            self.end_pos = np.array([95, 95])
            
            # Horizontal walls with gaps
            # Wall 1: y=25, gap on RIGHT
            for x in range(0, 80, 4):
                self.obstacles.append((x, 25, 3))
            self.gap_positions.append((90, 25))
            
            # Wall 2: y=50, gap on LEFT
            for x in range(20, 105, 4):
                self.obstacles.append((x, 50, 3))
            self.gap_positions.append((10, 50))
            
            # Wall 3: y=75, gap on RIGHT
            for x in range(0, 80, 4):
                self.obstacles.append((x, 75, 3))
            self.gap_positions.append((90, 75))
            
        elif name == 'Gauntlet':
            # Dense obstacle field with only ONE narrow corridor
            self.start_pos = np.array([5, 50])
            self.end_pos = np.array([95, 50])
            
            corridor_y = 50
            corridor_width = 8
            
            # Fill everything EXCEPT the corridor
            for x in range(20, 85, 6):
                for y in range(0, 105, 6):
                    # Leave corridor clear
                    if abs(y - corridor_y) > corridor_width:
                        self.obstacles.append((x, y, 4))
            
            self.gap_positions.append((50, corridor_y))
            
        elif name == 'Keyhole':
            # Must go through center keyhole
            self.start_pos = np.array([5, 50])
            self.end_pos = np.array([95, 50])
            
            # Circular wall with small opening
            center_x, center_y = 50, 50
            radius = 25
            
            for angle in range(0, 360, 8):
                # Leave gap at angle 0 (right side)
                if not (350 < angle or angle < 10):
                    rad = np.radians(angle)
                    x = center_x + radius * np.cos(rad)
                    y = center_y + radius * np.sin(rad)
                    self.obstacles.append((x, y, 4))
            
            # Block direct path outside the circle
            for x in range(0, 25, 5):
                self.obstacles.append((x, 30, 4))
                self.obstacles.append((x, 70, 4))
            for x in range(75, 105, 5):
                self.obstacles.append((x, 30, 4))
                self.obstacles.append((x, 70, 4))
            
            self.gap_positions.append((75, 50))
    
    def _segment_intersects_circle(self, p1, p2, circle):
        """Optimized collision check."""
        cx, cy, r = circle
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        fx = p1[0] - cx
        fy = p1[1] - cy
        
        a = dx*dx + dy*dy
        if a < 1e-10:
            dist_sq = fx*fx + fy*fy
            return dist_sq < r*r
            
        b = 2 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - r*r
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return False
            
        sqrt_disc = discriminant ** 0.5
        denom = 2*a + 1e-10
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom
        
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        
        if fx*fx + fy*fy < r*r:
            return True
        ex = p2[0] - cx
        ey = p2[1] - cy
        if ex*ex + ey*ey < r*r:
            return True
            
        return False
    
    def decode(self, x):
        """Convert vector to path."""
        waypoints = x.reshape(-1, 2)
        waypoints = np.clip(waypoints, self.bounds[0], self.bounds[1])
        return np.vstack([self.start_pos, waypoints, self.end_pos])
    
    def count_collisions(self, path):
        """Count collision segments."""
        collisions = 0
        for i in range(len(path) - 1):
            for obs in self.obstacles:
                if self._segment_intersects_circle(path[i], path[i+1], (obs[0], obs[1], obs[2] + 0.5)):
                    collisions += 1
                    break
        return collisions
    
    def evaluate(self, x):
        """Cost = distance + heavy collision penalty."""
        path = self.decode(x)
        total_dist = 0
        penalty = 0
        
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            total_dist += np.linalg.norm(p2 - p1)
            
            for obs in self.obstacles:
                if self._segment_intersects_circle(p1, p2, (obs[0], obs[1], obs[2] + 0.5)):
                    penalty += 500
        
        return total_dist + penalty
    
    def is_valid_path(self, x):
        """Check if path has no collisions."""
        return self.count_collisions(self.decode(x)) == 0
    
    def visualize_scenario(self, ax=None, title=None):
        """Draw the scenario."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect('equal')
        
        # Draw obstacles
        for (cx, cy, r) in self.obstacles:
            circle = patches.Circle((cx, cy), r, color='#B71C1C', alpha=0.8)
            ax.add_patch(circle)
        
        # Draw gap indicators
        for (gx, gy) in self.gap_positions:
            ax.plot(gx, gy, 'g*', markersize=15, zorder=10, label='Gap' if gx == self.gap_positions[0][0] else '')
        
        # Start and end
        ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=15, label='Start', zorder=10)
        ax.plot(self.end_pos[0], self.end_pos[1], 'r^', markersize=15, label='End', zorder=10)
        
        # Direct line (will be blocked)
        ax.plot([self.start_pos[0], self.end_pos[0]], 
                [self.start_pos[1], self.end_pos[1]], 
                '--', color='gray', alpha=0.3, linewidth=2, label='Direct (blocked)')
        
        ax.set_title(title or f'{self.scenario_name} Scenario', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        return ax


def visualize_all_scenarios():
    """Create overview of all constrained scenarios."""
    scenarios = ['SingleGap', 'DoubleGap', 'Zigzag', 'Funnel', 'Labyrinth', 'Gauntlet']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        problem = ConstrainedPathfinding(n_waypoints=8, scenario=scenario)
        problem.visualize_scenario(axes[idx], f'{scenario} ({len(problem.obstacles)} obstacles)')
    
    plt.suptitle('Constrained Pathfinding Scenarios\n(Obstacles extend to edges - NO WAY AROUND)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('constrained_scenarios_overview.png', dpi=150, bbox_inches='tight')
    print("Saved: constrained_scenarios_overview.png")
    plt.close()


def run_benchmark_with_viz(scenario='DoubleGap', n_runs=5, max_iter=150, pop_size=40):
    """Run benchmark and visualize results for a specific scenario."""
    
    print(f"\n{'='*70}")
    print(f"  CONSTRAINED PATHFINDING: {scenario}")
    print(f"{'='*70}")
    
    problem = ConstrainedPathfinding(n_waypoints=10, scenario=scenario)
    print(f"  Obstacles: {len(problem.obstacles)}")
    print(f"  Gap positions: {problem.gap_positions}")
    
    algos = {
        'OCA': lambda: OverclockingAlgorithm(pop_size),
        'OCA-Aggressive': lambda: OverclockingAlgorithm(pop_size, aggressive_voltage=True),
        'PSO': lambda: PSO(pop_size),
        'GWO': lambda: GWO(pop_size),
        'DE': lambda: DE(pop_size),
    }
    
    results = {}
    best_paths = {}
    
    print(f"\n  {'Algorithm':<18} | {'Best Cost':>10} | {'Valid':>6} | {'Success Rate':>12}")
    print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*6}-+-{'-'*12}")
    
    for algo_name, algo_class in algos.items():
        costs = []
        valid_count = 0
        best_pos = None
        best_cost = float('inf')
        
        for run in range(n_runs):
            algo = algo_class()
            pos, cost, _ = algo.optimize(
                problem.evaluate, problem.bounds, problem.dim, max_iter
            )
            costs.append(cost)
            
            if problem.is_valid_path(pos):
                valid_count += 1
            
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        results[algo_name] = {
            'costs': costs,
            'best_cost': best_cost,
            'success_rate': valid_count / n_runs * 100,
            'valid': problem.is_valid_path(best_pos)
        }
        best_paths[algo_name] = problem.decode(best_pos)
        
        valid_str = "✓" if results[algo_name]['valid'] else "✗"
        print(f"  {algo_name:<18} | {best_cost:>10.2f} | {valid_str:>6} | {results[algo_name]['success_rate']:>11.0f}%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: All trajectories
    ax1 = axes[0]
    problem.visualize_scenario(ax1)
    
    colors = {'OCA': '#2196F3', 'OCA-Aggressive': '#1565C0', 'PSO': '#F44336', 
              'GWO': '#4CAF50', 'DE': '#FF9800'}
    linestyles = {'OCA': '-', 'OCA-Aggressive': '-', 'PSO': '--', 'GWO': '-.', 'DE': ':'}
    
    for algo_name, path in best_paths.items():
        is_valid = results[algo_name]['valid']
        alpha = 1.0 if is_valid else 0.4
        lw = 2.5 if is_valid else 1.5
        
        ax1.plot(path[:, 0], path[:, 1], 
                color=colors.get(algo_name, 'gray'),
                linestyle=linestyles.get(algo_name, '-'),
                linewidth=lw, alpha=alpha,
                marker='o', markersize=4,
                label=f"{algo_name}: {results[algo_name]['best_cost']:.1f}")
    
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_title(f'{scenario}: All Trajectories', fontweight='bold')
    
    # Right: Success rate comparison
    ax2 = axes[1]
    algo_names = list(results.keys())
    success_rates = [results[a]['success_rate'] for a in algo_names]
    bar_colors = [colors.get(a, 'gray') for a in algo_names]
    
    bars = ax2.bar(algo_names, success_rates, color=bar_colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(0, 110)
    ax2.set_title(f'{scenario}: Success Rate ({n_runs} runs)', fontweight='bold')
    
    for bar, rate in zip(bars, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'constrained_{scenario}_results.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: constrained_{scenario}_results.png")
    plt.close()
    
    return results


def run_all_constrained():
    """Run benchmark on all constrained scenarios."""
    scenarios = ['SingleGap', 'DoubleGap', 'Zigzag', 'Funnel', 'Labyrinth', 'Gauntlet']
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "CONSTRAINED PATHFINDING BENCHMARK" + " " * 23 + "║")
    print("║" + f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 37 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # First visualize all scenarios
    visualize_all_scenarios()
    
    # Run each scenario
    all_results = {}
    for scenario in scenarios:
        all_results[scenario] = run_benchmark_with_viz(
            scenario=scenario, 
            n_runs=5, 
            max_iter=150, 
            pop_size=40
        )
    
    # Summary table
    print("\n" + "=" * 80)
    print("  SUMMARY: SUCCESS RATES ACROSS ALL CONSTRAINED SCENARIOS")
    print("=" * 80)
    
    algos = list(all_results[scenarios[0]].keys())
    
    print(f"\n  {'Algorithm':<18}", end="")
    for scenario in scenarios:
        print(f" | {scenario[:8]:>8}", end="")
    print(" | Average")
    
    print(f"  {'-'*18}", end="")
    for _ in scenarios:
        print(f"-+-{'-'*8}", end="")
    print("-+---------")
    
    for algo in algos:
        print(f"  {algo:<18}", end="")
        rates = []
        for scenario in scenarios:
            rate = all_results[scenario][algo]['success_rate']
            rates.append(rate)
            marker = "✓" if rate == 100 else "◐" if rate > 0 else "✗"
            print(f" | {rate:>6.0f}%{marker}", end="")
        print(f" | {np.mean(rates):>6.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Constrained Pathfinding Benchmark')
    parser.add_argument('--scenario', type=str, default='DoubleGap',
                       choices=['SingleGap', 'DoubleGap', 'Zigzag', 'Funnel', 'Labyrinth', 'Gauntlet', 'all'],
                       help='Scenario to test')
    parser.add_argument('--viz-only', action='store_true', help='Only visualize scenarios')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--iter', type=int, default=150, help='Max iterations')
    
    args = parser.parse_args()
    
    if args.viz_only:
        visualize_all_scenarios()
    elif args.scenario == 'all':
        run_all_constrained()
    else:
        visualize_all_scenarios()
        run_benchmark_with_viz(args.scenario, args.runs, args.iter)
