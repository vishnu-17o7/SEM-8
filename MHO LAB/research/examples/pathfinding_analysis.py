"""
Pathfinding Analysis & Visualization Suite
==========================================
Comprehensive visualization toolkit for analyzing metaheuristic algorithm
performance on pathfinding benchmarks.

Generates:
1. Trajectory Overlay (Money Shot) - All algorithms on same map
2. Convergence Curves - Fitness vs Iterations
3. Stability Analysis - Box-and-Whisker Plots
4. Success Rate Bar Chart - Valid paths per scenario
5. Population Diversity - Exploration capability over time
6. Path Smoothness Profile - Curvature analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
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

# Try importing the research algorithms and pathfinding benchmark
try:
    from research.oca import OverclockingAlgorithm
    from research.baselines import PSO, GWO, DE, GA, FA
    from research.pathfinding_benchmark import RobotNavigation
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ============================================
# STYLING CONFIGURATION
# ============================================

ALGO_STYLES = {
    'OCA': {'color': '#2196F3', 'linestyle': '-', 'linewidth': 2.5, 'marker': 'o', 'markersize': 6},
    'OCA-Aggressive': {'color': '#1565C0', 'linestyle': '-', 'linewidth': 2.5, 'marker': 's', 'markersize': 6},
    'PSO': {'color': '#F44336', 'linestyle': '--', 'linewidth': 2, 'marker': '^', 'markersize': 5},
    'GWO': {'color': '#4CAF50', 'linestyle': '-.', 'linewidth': 2, 'marker': 'v', 'markersize': 5},
    'DE': {'color': '#FF9800', 'linestyle': ':', 'linewidth': 2.5, 'marker': 'd', 'markersize': 5},
    'GA': {'color': '#9C27B0', 'linestyle': '--', 'linewidth': 2, 'marker': 'p', 'markersize': 5},
    'FA': {'color': '#795548', 'linestyle': '-.', 'linewidth': 2, 'marker': 'h', 'markersize': 5},
}

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'


# ============================================
# EXTENDED OPTIMIZER WITH TRACKING
# ============================================

class TrackedOptimizer:
    """
    Wrapper that tracks convergence history and population diversity
    for any optimizer using its native optimize method.
    """
    
    def __init__(self, optimizer, name):
        self.optimizer = optimizer
        self.name = name
        self.convergence_history = []
        self.diversity_history = []
        
    def optimize_with_tracking(self, objective_fn, bounds, dim, max_iterations):
        """Run optimization and extract tracking data from convergence curve."""
        
        # Use the optimizer's native optimize method
        best_pos, best_cost, convergence = self.optimizer.optimize(
            objective_fn=objective_fn,
            bounds=bounds,
            dim=dim,
            max_iterations=max_iterations
        )
        
        self.convergence_history = convergence if convergence else [best_cost]
        
        # Estimate diversity from convergence curve (rate of change)
        if len(self.convergence_history) > 1:
            conv_array = np.array(self.convergence_history)
            # Use gradient as proxy for diversity
            changes = np.abs(np.diff(conv_array))
            self.diversity_history = list(changes) + [changes[-1] if len(changes) > 0 else 0]
        else:
            self.diversity_history = [1.0] * max_iterations
        
        return best_pos, best_cost, self.convergence_history


# ============================================
# DATA COLLECTION
# ============================================

def collect_benchmark_data(scenarios, n_waypoints, n_runs, max_iter, pop_size):
    """
    Run benchmarks and collect detailed data for all visualizations.
    """
    print("\n" + "=" * 70)
    print("  COLLECTING BENCHMARK DATA FOR ANALYSIS")
    print("=" * 70)
    
    data = {
        'trajectories': defaultdict(dict),      # Best paths per algo/scenario
        'convergence': defaultdict(list),        # Convergence curves
        'diversity': defaultdict(list),          # Population diversity
        'final_costs': defaultdict(lambda: defaultdict(list)),  # All final costs
        'success_rates': defaultdict(lambda: defaultdict(int)), # Valid path counts
        'path_angles': defaultdict(dict),        # Turning angles
    }
    
    algo_classes = {
        'OCA': lambda: OverclockingAlgorithm(pop_size),
        'OCA-Aggressive': lambda: OverclockingAlgorithm(pop_size, aggressive_voltage=True),
        'PSO': lambda: PSO(pop_size),
        'GWO': lambda: GWO(pop_size),
        'DE': lambda: DE(pop_size),
    }
    
    for scenario in scenarios:
        print(f"\n  Scenario: {scenario}")
        print(f"  {'-' * 50}")
        
        problem = RobotNavigation(n_waypoints=n_waypoints, scenario=scenario)
        
        for algo_name, algo_class in algo_classes.items():
            print(f"    Running {algo_name}...", end=" ", flush=True)
            
            best_pos_overall = None
            best_cost_overall = float('inf')
            all_convergence = []
            all_diversity = []
            
            for run in range(n_runs):
                algo = algo_class()
                tracked = TrackedOptimizer(algo, algo_name)
                
                best_pos, best_cost, convergence = tracked.optimize_with_tracking(
                    problem.evaluate, problem.bounds, problem.dim, max_iter
                )
                
                # Store final cost
                data['final_costs'][algo_name][scenario].append(best_cost)
                
                # Check validity
                is_valid = problem.is_valid_path(best_pos)
                if is_valid:
                    data['success_rates'][algo_name][scenario] += 1
                
                # Track best overall
                if best_cost < best_cost_overall:
                    best_cost_overall = best_cost
                    best_pos_overall = best_pos
                
                all_convergence.append(convergence)
                all_diversity.append(tracked.diversity_history)
            
            # Store best trajectory
            data['trajectories'][algo_name][scenario] = {
                'path': problem.decode(best_pos_overall),
                'cost': best_cost_overall,
                'valid': problem.is_valid_path(best_pos_overall)
            }
            
            # Store averaged convergence and diversity
            min_len = min(len(c) for c in all_convergence) if all_convergence else 1
            avg_conv = np.mean([c[:min_len] for c in all_convergence], axis=0)
            
            min_div_len = min(len(d) for d in all_diversity) if all_diversity else 1
            avg_div = np.mean([d[:min_div_len] for d in all_diversity], axis=0) if all_diversity[0] else np.ones(min_len)
            
            data['convergence'][algo_name].append({'scenario': scenario, 'curve': list(avg_conv)})
            data['diversity'][algo_name].append({'scenario': scenario, 'curve': list(avg_div)})
            
            # Compute path angles
            path = problem.decode(best_pos_overall)
            angles = compute_path_angles(path)
            data['path_angles'][algo_name][scenario] = angles
            
            success_pct = data['success_rates'][algo_name][scenario] / n_runs * 100
            print(f"Best: {best_cost_overall:.2f}, Success: {success_pct:.0f}%")
    
    data['n_runs'] = n_runs
    data['scenarios'] = scenarios
    data['n_waypoints'] = n_waypoints
    
    return data, problem


def compute_path_angles(path):
    """Compute turning angles along a path."""
    angles = []
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i-1]
        v2 = path[i+1] - path[i]
        
        # Compute angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        angles.append(np.degrees(angle))
    
    return angles


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def plot_trajectory_overlay(data, problem, scenario, save_path=None):
    """
    Plot 1: The "Money Shot" - All algorithm paths on same map.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Trajectory Comparison: {scenario} Scenario', fontsize=16, fontweight='bold')
    
    # Draw obstacles
    problem_vis = RobotNavigation(n_waypoints=data['n_waypoints'], scenario=scenario)
    for (cx, cy, r) in problem_vis.obstacles:
        circle = patches.Circle((cx, cy), r, color='#D32F2F', alpha=0.7, zorder=1)
        ax.add_patch(circle)
    
    # Draw paths for each algorithm
    legend_elements = []
    for algo_name, style in ALGO_STYLES.items():
        if algo_name in data['trajectories'] and scenario in data['trajectories'][algo_name]:
            path_data = data['trajectories'][algo_name][scenario]
            path = path_data['path']
            is_valid = path_data['valid']
            cost = path_data['cost']
            
            # Adjust alpha for invalid paths
            alpha = 1.0 if is_valid else 0.5
            
            ax.plot(path[:, 0], path[:, 1], 
                   color=style['color'], 
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   marker=style['marker'],
                   markersize=style['markersize'],
                   alpha=alpha,
                   zorder=3)
            
            status = "✓" if is_valid else "✗"
            legend_elements.append(Line2D([0], [0], color=style['color'], 
                                         linestyle=style['linestyle'],
                                         linewidth=style['linewidth'],
                                         label=f'{algo_name}: {cost:.1f} {status}'))
    
    # Draw start and end points
    ax.plot(problem_vis.start_pos[0], problem_vis.start_pos[1], 'go', 
           markersize=15, zorder=5, label='Start')
    ax.plot(problem_vis.end_pos[0], problem_vis.end_pos[1], 'r*', 
           markersize=20, zorder=5, label='End')
    
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_convergence_curves(data, scenario, save_path=None):
    """
    Plot 2: Convergence Curves - Average Fitness vs Iterations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective Cost (Log Scale)')
    ax.set_title(f'Convergence Analysis: {scenario} Scenario', fontsize=14, fontweight='bold')
    
    for algo_name, style in ALGO_STYLES.items():
        if algo_name in data['convergence']:
            for conv_data in data['convergence'][algo_name]:
                if conv_data['scenario'] == scenario:
                    curve = conv_data['curve']
                    iterations = np.arange(len(curve))
                    
                    ax.plot(iterations, curve,
                           color=style['color'],
                           linestyle=style['linestyle'],
                           linewidth=style['linewidth'],
                           label=algo_name)
                    break
    
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_stability_boxplot(data, scenario, save_path=None):
    """
    Plot 3: Stability Analysis - Box-and-Whisker Plots.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algo_names = []
    all_costs = []
    colors = []
    
    for algo_name in ALGO_STYLES.keys():
        if algo_name in data['final_costs'] and scenario in data['final_costs'][algo_name]:
            costs = data['final_costs'][algo_name][scenario]
            # Filter out massive penalty values for visualization
            costs_filtered = [c for c in costs if c < 5000]
            if costs_filtered:
                algo_names.append(algo_name)
                all_costs.append(costs_filtered)
                colors.append(ALGO_STYLES[algo_name]['color'])
    
    if not all_costs:
        print(f"  No data for {scenario}")
        return
    
    bp = ax.boxplot(all_costs, labels=algo_names, patch_artist=True, 
                    showfliers=True, flierprops={'marker': 'o', 'markersize': 4})
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Final Path Cost')
    ax.set_title(f'Result Stability: {scenario} Scenario ({data["n_runs"]} runs)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add median values as text
    medians = [np.median(c) for c in all_costs]
    for i, median in enumerate(medians):
        ax.text(i + 1, median, f'{median:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_success_rate_bars(data, save_path=None):
    """
    Plot 4: Success Rate Bar Chart - Valid paths per scenario.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = data['scenarios']
    n_runs = data['n_runs']
    
    algo_names = list(ALGO_STYLES.keys())
    x = np.arange(len(scenarios))
    width = 0.12
    
    for i, algo_name in enumerate(algo_names):
        if algo_name in data['success_rates']:
            rates = []
            for scenario in scenarios:
                count = data['success_rates'][algo_name].get(scenario, 0)
                rates.append(count / n_runs * 100)
            
            offset = (i - len(algo_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, rates, width, 
                         label=algo_name, 
                         color=ALGO_STYLES[algo_name]['color'],
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=0.5)
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                if rate > 5:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{rate:.0f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Path Validity Success Rate ({n_runs} runs per algorithm)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in scenarios])
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect')
    ax.legend(loc='upper right', ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_population_diversity(data, scenario, save_path=None):
    """
    Plot 5: Population Diversity - Exploration capability over time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Average Distance from Centroid')
    ax.set_title(f'Population Diversity (Exploration): {scenario} Scenario', 
                fontsize=14, fontweight='bold')
    
    for algo_name, style in ALGO_STYLES.items():
        if algo_name in data['diversity']:
            for div_data in data['diversity'][algo_name]:
                if div_data['scenario'] == scenario:
                    curve = div_data['curve']
                    iterations = np.arange(len(curve))
                    
                    ax.plot(iterations, curve,
                           color=style['color'],
                           linestyle=style['linestyle'],
                           linewidth=style['linewidth'],
                           label=algo_name)
                    break
    
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Higher = More Exploration', xy=(0.7, 0.95), xycoords='axes fraction',
               fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_path_smoothness(data, scenario, save_path=None):
    """
    Plot 6: Path Smoothness Profile - Curvature analysis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Line chart of turning angles
    ax1.set_xlabel('Path Step')
    ax1.set_ylabel('Turning Angle (degrees)')
    ax1.set_title(f'Turning Angle Profile: {scenario}', fontsize=12, fontweight='bold')
    
    max_angles = []
    for algo_name, style in ALGO_STYLES.items():
        if algo_name in data['path_angles'] and scenario in data['path_angles'][algo_name]:
            angles = data['path_angles'][algo_name][scenario]
            if angles:
                steps = np.arange(1, len(angles) + 1)
                ax1.plot(steps, angles,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        linewidth=style['linewidth'],
                        marker=style['marker'],
                        markersize=4,
                        label=algo_name)
                max_angles.append((algo_name, max(angles) if angles else 0))
    
    ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90° (Sharp Turn)')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Bar chart of average smoothness
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Average Turning Angle (degrees)')
    ax2.set_title(f'Path Smoothness Comparison: {scenario}', fontsize=12, fontweight='bold')
    
    algo_names = []
    avg_angles = []
    colors = []
    
    for algo_name in ALGO_STYLES.keys():
        if algo_name in data['path_angles'] and scenario in data['path_angles'][algo_name]:
            angles = data['path_angles'][algo_name][scenario]
            if angles:
                algo_names.append(algo_name)
                avg_angles.append(np.mean(angles))
                colors.append(ALGO_STYLES[algo_name]['color'])
    
    if avg_angles:
        bars = ax2.bar(algo_names, avg_angles, color=colors, alpha=0.8, edgecolor='black')
        ax2.axhline(y=np.mean(avg_angles), color='gray', linestyle='--', alpha=0.5, label='Mean')
        
        # Add value labels
        for bar, val in zip(bars, avg_angles):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}°', ha='center', va='bottom', fontsize=9)
        
        ax2.annotate('Lower = Smoother Path', xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=10, style='italic', alpha=0.7)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def create_combined_dashboard(data, scenario, save_path=None):
    """
    Create a combined dashboard with all key visualizations.
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    problem = RobotNavigation(n_waypoints=data['n_waypoints'], scenario=scenario)
    
    # 1. Trajectory (large, top-left)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    ax1.set_title(f'Trajectory Overlay: {scenario}', fontweight='bold')
    
    for (cx, cy, r) in problem.obstacles:
        circle = patches.Circle((cx, cy), r, color='#D32F2F', alpha=0.7)
        ax1.add_patch(circle)
    
    for algo_name, style in ALGO_STYLES.items():
        if algo_name in data['trajectories'] and scenario in data['trajectories'][algo_name]:
            path = data['trajectories'][algo_name][scenario]['path']
            ax1.plot(path[:, 0], path[:, 1], color=style['color'], 
                    linestyle=style['linestyle'], linewidth=style['linewidth'], label=algo_name)
    
    ax1.plot(problem.start_pos[0], problem.start_pos[1], 'go', markersize=12)
    ax1.plot(problem.end_pos[0], problem.end_pos[1], 'r*', markersize=15)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    for algo_name, style in ALGO_STYLES.items():
        if algo_name in data['convergence']:
            for conv_data in data['convergence'][algo_name]:
                if conv_data['scenario'] == scenario:
                    ax2.plot(conv_data['curve'], color=style['color'], 
                            linestyle=style['linestyle'], label=algo_name)
                    break
    ax2.set_yscale('log')
    ax2.set_title('Convergence', fontweight='bold')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Cost (log)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # 3. Diversity (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    for algo_name, style in ALGO_STYLES.items():
        if algo_name in data['diversity']:
            for div_data in data['diversity'][algo_name]:
                if div_data['scenario'] == scenario:
                    ax3.plot(div_data['curve'], color=style['color'],
                            linestyle=style['linestyle'], label=algo_name)
                    break
    ax3.set_title('Population Diversity', fontweight='bold')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Spread')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot (middle-right)
    ax4 = fig.add_subplot(gs[1, 1:])
    algo_names = []
    all_costs = []
    colors = []
    for algo_name in ALGO_STYLES.keys():
        if algo_name in data['final_costs'] and scenario in data['final_costs'][algo_name]:
            costs = [c for c in data['final_costs'][algo_name][scenario] if c < 5000]
            if costs:
                algo_names.append(algo_name)
                all_costs.append(costs)
                colors.append(ALGO_STYLES[algo_name]['color'])
    
    if all_costs:
        bp = ax4.boxplot(all_costs, labels=algo_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax4.set_title('Result Stability', fontweight='bold')
    ax4.set_ylabel('Final Cost')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Success Rate (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    algo_names_sr = []
    rates = []
    colors_sr = []
    for algo_name in ALGO_STYLES.keys():
        if algo_name in data['success_rates'] and scenario in data['success_rates'][algo_name]:
            algo_names_sr.append(algo_name)
            rates.append(data['success_rates'][algo_name][scenario] / data['n_runs'] * 100)
            colors_sr.append(ALGO_STYLES[algo_name]['color'])
    
    if rates:
        ax5.bar(algo_names_sr, rates, color=colors_sr, alpha=0.8)
    ax5.set_title('Success Rate', fontweight='bold')
    ax5.set_ylabel('Valid Paths (%)')
    ax5.set_ylim(0, 110)
    ax5.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Smoothness (bottom-middle/right)
    ax6 = fig.add_subplot(gs[2, 1:])
    algo_names_sm = []
    avg_angles = []
    colors_sm = []
    for algo_name in ALGO_STYLES.keys():
        if algo_name in data['path_angles'] and scenario in data['path_angles'][algo_name]:
            angles = data['path_angles'][algo_name][scenario]
            if angles:
                algo_names_sm.append(algo_name)
                avg_angles.append(np.mean(angles))
                colors_sm.append(ALGO_STYLES[algo_name]['color'])
    
    if avg_angles:
        ax6.bar(algo_names_sm, avg_angles, color=colors_sm, alpha=0.8)
    ax6.set_title('Path Smoothness (Lower = Better)', fontweight='bold')
    ax6.set_ylabel('Avg Turning Angle (°)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle(f'Pathfinding Algorithm Analysis Dashboard: {scenario.upper()}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ============================================
# MAIN RUNNER
# ============================================

def run_full_analysis(scenarios=None, n_runs=10, max_iter=100, pop_size=30):
    """
    Run complete analysis and generate all visualizations.
    """
    if scenarios is None:
        scenarios = ['Trap', 'Maze', 'Clutter']
    
    n_waypoints = 5
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "PATHFINDING ANALYSIS SUITE" + " " * 27 + "║")
    print("║" + f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 37 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print(f"\n  Configuration:")
    print(f"    Scenarios: {scenarios}")
    print(f"    Runs per algo: {n_runs}")
    print(f"    Iterations: {max_iter}")
    print(f"    Population: {pop_size}")
    
    # Collect data
    data, problem = collect_benchmark_data(scenarios, n_waypoints, n_runs, max_iter, pop_size)
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    output_dir = "analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario in scenarios:
        print(f"\n  {scenario}:")
        
        # 1. Trajectory Overlay
        plot_trajectory_overlay(data, problem, scenario, 
                              f"{output_dir}/1_trajectory_{scenario}.png")
        
        # 2. Convergence Curves
        plot_convergence_curves(data, scenario, 
                              f"{output_dir}/2_convergence_{scenario}.png")
        
        # 3. Stability Box Plot
        plot_stability_boxplot(data, scenario, 
                             f"{output_dir}/3_stability_{scenario}.png")
        
        # 5. Population Diversity
        plot_population_diversity(data, scenario, 
                                f"{output_dir}/5_diversity_{scenario}.png")
        
        # 6. Path Smoothness
        plot_path_smoothness(data, scenario, 
                           f"{output_dir}/6_smoothness_{scenario}.png")
        
        # Combined Dashboard
        create_combined_dashboard(data, scenario, 
                                f"{output_dir}/dashboard_{scenario}.png")
    
    # 4. Success Rate (across all scenarios)
    print(f"\n  Cross-scenario:")
    plot_success_rate_bars(data, f"{output_dir}/4_success_rate_all.png")
    
    print("\n" + "=" * 70)
    print(f"  Analysis complete! All plots saved to '{output_dir}/' folder.")
    print("=" * 70)
    
    return data


def run_quick_analysis():
    """Quick analysis with fewer runs for testing."""
    return run_full_analysis(
        scenarios=['Trap', 'Maze'],
        n_runs=5,
        max_iter=80,
        pop_size=25
    )


# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pathfinding Analysis & Visualization Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick analysis (fewer runs)')
    parser.add_argument('--full', action='store_true', help='Run full analysis (30 runs)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per algorithm')
    parser.add_argument('--iter', type=int, default=100, help='Max iterations')
    parser.add_argument('--scenarios', nargs='+', default=['Trap', 'Maze', 'Clutter'],
                       help='Scenarios to test')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_analysis()
    elif args.full:
        run_full_analysis(n_runs=30, max_iter=150)
    else:
        run_full_analysis(
            scenarios=args.scenarios,
            n_runs=args.runs,
            max_iter=args.iter
        )
