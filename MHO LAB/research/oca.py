# Overclocking Algorithm (OCA) - Lite Edition
# Python 3.8+

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

class OverclockingAlgorithm:
    """
    Overclocking Algorithm (OCA) V7 - Lite Edition
    
    A streamlined, high-performance metaheuristic inspired by CPU architecture.
    Designed to be faster and simpler than previous versions while maintaining
    competitive accuracy against GWO.
    
    Core Concepts:
    1. P-CORES (Performance): The top 3 best solutions act as leaders (like GWO Alpha/Beta/Delta).
    2. E-CORES (Efficiency): The rest of the population follows the P-Cores.
    3. DYNAMIC VOLTAGE (DVFS): Exploration parameter decays over time.
    4. CACHE MISS (Reset): Stagnant particles are re-initialized to maintain diversity.
    5. INSTRUCTION PIPELINING (Momentum): Successful moves carry momentum.
    """

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size
        # P-Cores count (Top 3 leaders)
        self.num_p_cores = 3

    def initialize(self, bounds: Tuple, dim: int):
        self.dim = dim
        self.lower = bounds[0]
        self.upper = bounds[1]
        self.range = self.upper - self.lower
        
        # Population (Cores)
        self.positions = np.random.uniform(self.lower, self.upper, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        
        # Velocities for momentum (Instruction Pipeline)
        self.velocities = np.zeros((self.pop_size, dim))
        
        # Stagnation counters (Cache Miss detection)
        self.stagnation = np.zeros(self.pop_size, dtype=int)
        
        # Leaders (P-Cores)
        self.p_cores_pos = np.zeros((self.num_p_cores, dim))
        self.p_cores_fit = np.full(self.num_p_cores, np.inf)

    def clip(self, val):
        return np.clip(val, self.lower, self.upper)

    def optimize(self, objective_fn: Callable, bounds: Tuple, dim: int, max_iterations: int = 200):
        self.initialize(bounds, dim)
        convergence_curve = []
        
        for it in range(max_iterations):
            # 1. Evaluate Fitness
            current_fitness = np.array([objective_fn(p) for p in self.positions])
            
            # 2. Update Leaders (P-Cores)
            # Combine current population with previous leaders to ensure elitism
            all_pos = np.vstack((self.positions, self.p_cores_pos))
            all_fit = np.concatenate((current_fitness, self.p_cores_fit))
            
            sorted_idx = np.argsort(all_fit)
            
            # Update P-Cores (Top 3)
            for k in range(self.num_p_cores):
                self.p_cores_pos[k] = all_pos[sorted_idx[k]].copy()
                self.p_cores_fit[k] = all_fit[sorted_idx[k]]
            
            # Update Global Best
            gbest_fit = self.p_cores_fit[0]
            convergence_curve.append(gbest_fit)
            
            # 3. Dynamic Voltage (Exploration Rate)
            # Decays from 2.0 to 0.0 (Linear) - Similar to GWO 'a' parameter
            voltage = 2.0 * (1 - (it / max_iterations))
            
            # 4. Update Positions
            for i in range(self.pop_size):
                # Check for stagnation (Cache Miss)
                if current_fitness[i] >= self.fitness[i]: # No improvement
                    self.stagnation[i] += 1
                else:
                    self.stagnation[i] = 0
                    self.fitness[i] = current_fitness[i] # Update stored fitness
                
                # CACHE MISS: If stagnant for too long, reset (Diversity)
                if self.stagnation[i] > 10:
                    # Soft Reset: Jump towards a P-Core with high noise
                    target = self.p_cores_pos[np.random.randint(0, self.num_p_cores)]
                    self.positions[i] = self.clip(target + np.random.uniform(-1, 1, self.dim) * self.range * 0.1)
                    self.stagnation[i] = 0
                    self.velocities[i] = 0
                    continue

                # CORE UPDATE LOGIC
                # Calculate attraction to P-Cores
                # We use a weighted average of the Top 3 leaders (Tri-Core Architecture)
                
                new_vel = np.zeros(self.dim)
                
                for k in range(self.num_p_cores):
                    # Random vectors
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    
                    # GWO-style equations: D = |C*Leader - Current|
                    C = 2 * r2
                    D = np.abs(C * self.p_cores_pos[k] - self.positions[i])
                    
                    # A = 2*a*r1 - a (where a is voltage)
                    A = 2 * voltage * r1 - voltage
                    
                    # Step towards leader
                    step = self.p_cores_pos[k] - A * D
                    
                    # Accumulate influence (Average later)
                    new_vel += step
                
                # Average position suggested by P-Cores
                target_pos = new_vel / self.num_p_cores
                
                # Apply Momentum (Instruction Pipelining)
                # If the core is improving, it keeps some previous velocity
                w = 0.5 + 0.4 * np.random.rand() # Inertia weight
                self.velocities[i] = w * self.velocities[i] + (target_pos - self.positions[i])
                
                # Update Position
                self.positions[i] = self.clip(self.positions[i] + self.velocities[i] * 0.5) # 0.5 is learning rate

        return self.p_cores_pos[0], gbest_fit, convergence_curve

    plt.grid(True)
    plt.show()
