"""
BBOB-Style Benchmark for OCA
Black-Box Optimization Benchmarking following the COCO Platform standards.

This benchmark tests the core mathematical optimization capability using
the 24 standard BBOB functions with shifted/rotated variants.

Reference: https://github.com/numbbo/coco
"""

import numpy as np
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
    from baselines import PSO, GWO, DE, GA, FA
except ImportError:
    try:
        from research.oca import OverclockingAlgorithm
        from research.baselines import PSO, GWO, DE, GA, FA
    except ImportError as e:
        print(f"Error importing algorithms: {e}")
        sys.exit(1)


class BBOBFunction:
    """Base class for BBOB benchmark functions"""
    
    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.name = "Base"
        self.bounds = (-5, 5)
        self.optimal_value = 0.0
        
        # Generate transformation matrices for shifted/rotated variants
        self.x_opt = self.rng.uniform(-4, 4, dim)  # Shifted optimum
        self.R = self._generate_rotation_matrix(dim)  # Rotation matrix
        
    def _generate_rotation_matrix(self, dim: int) -> np.ndarray:
        """Generate a random rotation matrix using QR decomposition"""
        A = self.rng.randn(dim, dim)
        Q, _ = np.linalg.qr(A)
        return Q
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply shift and rotation transformations"""
        z = x - self.x_opt
        return self.R @ z
    
    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError


# ============= Group 1: Separable Functions =============

class F1_Sphere(BBOBFunction):
    """f1: Sphere Function - Separable, unimodal"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f1_Sphere"
        self.bounds = (-5.12, 5.12)
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        return np.sum(z ** 2)


class F2_Ellipsoidal(BBOBFunction):
    """f2: Ellipsoidal Function - Separable, ill-conditioned"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f2_Ellipsoidal"
        self.condition = 1e6  # Condition number
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        coeffs = np.power(self.condition, np.arange(self.dim) / (self.dim - 1))
        return np.sum(coeffs * z ** 2)


class F3_Rastrigin(BBOBFunction):
    """f3: Rastrigin Function - Separable, multimodal"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f3_Rastrigin"
        self.bounds = (-5.12, 5.12)
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        A = 10
        return A * self.dim + np.sum(z**2 - A * np.cos(2 * np.pi * z))


class F4_BuecheRastrigin(BBOBFunction):
    """f4: Bueche-Rastrigin Function - Separable, multimodal, asymmetric"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f4_BuecheRastrigin"
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        # Asymmetric transformation
        for i in range(self.dim):
            if z[i] > 0 and i % 2 == 0:
                z[i] = np.sqrt(10) * z[i]
        A = 10
        return A * self.dim + np.sum(z**2 - A * np.cos(2 * np.pi * z))


class F5_LinearSlope(BBOBFunction):
    """f5: Linear Slope - Separable, linear"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f5_LinearSlope"
        self.s = np.sign(self.x_opt) * np.power(10, np.arange(self.dim) / (self.dim - 1))
    
    def __call__(self, x: np.ndarray) -> float:
        z = x.copy()
        # Boundary handling
        for i in range(self.dim):
            if self.x_opt[i] * z[i] < 25:
                z[i] = z[i]
            else:
                z[i] = np.sign(self.x_opt[i]) * 5
        return np.sum(5 * np.abs(self.s) - self.s * z)


# ============= Group 2: Functions with Low/Moderate Conditioning =============

class F6_AttractiveSector(BBOBFunction):
    """f6: Attractive Sector Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f6_AttractiveSector"
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        s = np.ones(self.dim)
        for i in range(self.dim):
            if z[i] * self.x_opt[i] > 0:
                s[i] = 100
        return np.power(np.sum(np.power(s * z, 2)), 0.9)


class F7_StepEllipsoidal(BBOBFunction):
    """f7: Step Ellipsoidal Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f7_StepEllipsoidal"
        self.condition = 100
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        # Apply step transformation
        z_hat = np.where(np.abs(z) > 0.5, np.floor(0.5 + z), np.floor(0.5 + 10 * z) / 10)
        coeffs = np.power(self.condition, np.arange(self.dim) / (self.dim - 1))
        return 0.1 * max(np.abs(z[0]) / 1e4, np.sum(coeffs * z_hat ** 2))


class F8_Rosenbrock(BBOBFunction):
    """f8: Rosenbrock Function - Valley-shaped"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f8_Rosenbrock"
        self.bounds = (-2.048, 2.048)
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x) + 1  # Shift to make optimum at (1,1,...,1)
        result = 0
        for i in range(self.dim - 1):
            result += 100 * (z[i+1] - z[i]**2)**2 + (1 - z[i])**2
        return result


class F9_RosenbrockRotated(BBOBFunction):
    """f9: Rotated Rosenbrock Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f9_RosenbrockRotated"
        self.R2 = self._generate_rotation_matrix(dim)
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.R2 @ self.transform(x) + 1
        result = 0
        for i in range(self.dim - 1):
            result += 100 * (z[i+1] - z[i]**2)**2 + (1 - z[i])**2
        return result


# ============= Group 3: Functions with High Conditioning =============

class F10_EllipsoidalRotated(BBOBFunction):
    """f10: Rotated Ellipsoidal Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f10_EllipsoidalRotated"
        self.condition = 1e6
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        coeffs = np.power(self.condition, np.arange(self.dim) / (self.dim - 1))
        return np.sum(coeffs * z ** 2)


class F11_Discus(BBOBFunction):
    """f11: Discus Function - Highly ill-conditioned"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f11_Discus"
        self.condition = 1e6
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        return self.condition * z[0]**2 + np.sum(z[1:]**2)


class F12_BentCigar(BBOBFunction):
    """f12: Bent Cigar Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f12_BentCigar"
        self.condition = 1e6
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        return z[0]**2 + self.condition * np.sum(z[1:]**2)


class F13_SharpRidge(BBOBFunction):
    """f13: Sharp Ridge Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f13_SharpRidge"
        self.condition = 10
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        coeffs = np.power(self.condition, np.arange(1, self.dim) / (self.dim - 1))
        return z[0]**2 + 100 * np.sqrt(np.sum(coeffs * z[1:]**2))


class F14_DifferentPowers(BBOBFunction):
    """f14: Different Powers Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f14_DifferentPowers"
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        exponents = 2 + 4 * np.arange(self.dim) / (self.dim - 1)
        return np.sqrt(np.sum(np.abs(z) ** exponents))


# ============= Group 4: Multi-modal Functions =============

class F15_RastriginRotated(BBOBFunction):
    """f15: Rotated Rastrigin Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f15_RastriginRotated"
        self.R2 = self._generate_rotation_matrix(dim)
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.R2 @ self.transform(x)
        A = 10
        return A * self.dim + np.sum(z**2 - A * np.cos(2 * np.pi * z))


class F16_Weierstrass(BBOBFunction):
    """f16: Weierstrass Function - Highly multimodal"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f16_Weierstrass"
        self.k_max = 12
        self.a = 0.5
        self.b = 3
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        result = 0
        for i in range(self.dim):
            for k in range(self.k_max):
                result += self.a**k * np.cos(2 * np.pi * self.b**k * (z[i] + 0.5))
        
        # Subtract the baseline
        baseline = self.dim * np.sum([self.a**k * np.cos(2 * np.pi * self.b**k * 0.5) 
                                       for k in range(self.k_max)])
        return 10 * (result / self.dim - baseline)**3


class F17_SchaffersF7(BBOBFunction):
    """f17: Schaffer's F7 Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f17_SchaffersF7"
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        s = np.sqrt(z[:-1]**2 + z[1:]**2)
        result = np.mean(np.sqrt(s) * (1 + np.sin(50 * s**0.2)**2))
        return result**2


class F18_SchaffersF7Ill(BBOBFunction):
    """f18: Schaffer's F7 Function, moderately ill-conditioned"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f18_SchaffersF7Ill"
        self.condition = 1000
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        # Apply conditioning
        coeffs = np.power(self.condition, np.arange(self.dim) / (2 * (self.dim - 1)))
        z = coeffs * z
        s = np.sqrt(z[:-1]**2 + z[1:]**2)
        result = np.mean(np.sqrt(s) * (1 + np.sin(50 * s**0.2)**2))
        return result**2


class F19_GriewankRosenbrock(BBOBFunction):
    """f19: Composite Griewank-Rosenbrock Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f19_GriewankRosenbrock"
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x) + 1
        s = 100 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2
        return 10 * np.mean(s / 4000 - np.cos(s)) + 10


# ============= Group 5: Multi-modal with Weak Global Structure =============

class F20_Schwefel(BBOBFunction):
    """f20: Schwefel Function - Deceptive"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f20_Schwefel"
        self.bounds = (-500, 500)
    
    def __call__(self, x: np.ndarray) -> float:
        z = 100 * (self.transform(x) / 100 + 4.2096874633)
        result = 0
        for i in range(self.dim):
            if np.abs(z[i]) <= 500:
                result += z[i] * np.sin(np.sqrt(np.abs(z[i])))
            elif z[i] > 500:
                result += (500 - z[i] % 500) * np.sin(np.sqrt(np.abs(500 - z[i] % 500))) - (z[i] - 500)**2 / 10000 / self.dim
            else:
                result += (z[i] % 500 - 500) * np.sin(np.sqrt(np.abs(z[i] % 500 - 500))) - (z[i] + 500)**2 / 10000 / self.dim
        return 418.9829 * self.dim - result


class F21_Gallagher101(BBOBFunction):
    """f21: Gallagher's Gaussian 101-me Peaks Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f21_Gallagher101"
        self.n_peaks = min(101, 21)  # Limit for performance
        self.peaks = [self.rng.uniform(-4, 4, dim) for _ in range(self.n_peaks)]
        self.weights = np.maximum(0, 10 - np.arange(self.n_peaks) * 0.1)
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        max_val = 0
        for i, (peak, w) in enumerate(zip(self.peaks, self.weights)):
            dist = np.sum((z - peak)**2)
            val = w * np.exp(-0.5 * dist / (self.dim * (i + 1) * 0.1))
            max_val = max(max_val, val)
        return 10 - max_val


class F22_Gallagher21(BBOBFunction):
    """f22: Gallagher's Gaussian 21-hi Peaks Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f22_Gallagher21"
        self.n_peaks = 21
        self.peaks = [self.rng.uniform(-4, 4, dim) for _ in range(self.n_peaks)]
        self.weights = np.maximum(0, 10 - np.arange(self.n_peaks) * 0.5)
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        max_val = 0
        for i, (peak, w) in enumerate(zip(self.peaks, self.weights)):
            dist = np.sum((z - peak)**2)
            val = w * np.exp(-0.5 * dist / (self.dim * (i + 1)))
            max_val = max(max_val, val)
        return 10 - max_val


class F23_Katsuura(BBOBFunction):
    """f23: Katsuura Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f23_Katsuura"
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        result = 1
        for i in range(self.dim):
            inner_sum = 0
            for j in range(1, 33):
                inner_sum += np.abs(2**j * z[i] - np.round(2**j * z[i])) / 2**j
            result *= (1 + (i + 1) * inner_sum) ** (10 / self.dim**1.2)
        return 10 / self.dim**2 * result - 10 / self.dim**2


class F24_Lunacek(BBOBFunction):
    """f24: Lunacek bi-Rastrigin Function"""
    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.name = "f24_Lunacek"
        self.mu0 = 2.5
        self.s = 1 - 1 / (2 * np.sqrt(dim + 20) - 8.2)
        self.mu1 = -np.sqrt((self.mu0**2 - 1) / self.s)
    
    def __call__(self, x: np.ndarray) -> float:
        z = self.transform(x)
        sphere1 = np.sum((z - self.mu0)**2)
        sphere2 = self.dim + self.s * np.sum((z - self.mu1)**2)
        rastrigin = 10 * (self.dim - np.sum(np.cos(2 * np.pi * z)))
        return min(sphere1, sphere2) + rastrigin


# ============= Benchmark Suite =============

def get_all_functions(dim: int, seed: int = 42) -> Dict[str, BBOBFunction]:
    """Get all 24 BBOB functions"""
    return {
        'f1_Sphere': F1_Sphere(dim, seed),
        'f2_Ellipsoidal': F2_Ellipsoidal(dim, seed),
        'f3_Rastrigin': F3_Rastrigin(dim, seed),
        'f4_BuecheRastrigin': F4_BuecheRastrigin(dim, seed),
        'f5_LinearSlope': F5_LinearSlope(dim, seed),
        'f6_AttractiveSector': F6_AttractiveSector(dim, seed),
        'f7_StepEllipsoidal': F7_StepEllipsoidal(dim, seed),
        'f8_Rosenbrock': F8_Rosenbrock(dim, seed),
        'f9_RosenbrockRotated': F9_RosenbrockRotated(dim, seed),
        'f10_EllipsoidalRotated': F10_EllipsoidalRotated(dim, seed),
        'f11_Discus': F11_Discus(dim, seed),
        'f12_BentCigar': F12_BentCigar(dim, seed),
        'f13_SharpRidge': F13_SharpRidge(dim, seed),
        'f14_DifferentPowers': F14_DifferentPowers(dim, seed),
        'f15_RastriginRotated': F15_RastriginRotated(dim, seed),
        'f16_Weierstrass': F16_Weierstrass(dim, seed),
        'f17_SchaffersF7': F17_SchaffersF7(dim, seed),
        'f18_SchaffersF7Ill': F18_SchaffersF7Ill(dim, seed),
        'f19_GriewankRosenbrock': F19_GriewankRosenbrock(dim, seed),
        'f20_Schwefel': F20_Schwefel(dim, seed),
        'f21_Gallagher101': F21_Gallagher101(dim, seed),
        'f22_Gallagher21': F22_Gallagher21(dim, seed),
        'f23_Katsuura': F23_Katsuura(dim, seed),
        'f24_Lunacek': F24_Lunacek(dim, seed),
    }


def run_single_function(algorithms: Dict, func: BBOBFunction, max_iter: int = 500, 
                        pop_size: int = 50, n_runs: int = 5) -> Dict:
    """Run all algorithms on a single function"""
    results = {}
    
    for algo_name, algo_cls in algorithms.items():
        fitness_vals = []
        times = []
        
        for run in range(n_runs):
            # Handle factory functions (like lambda for OCA-Agg)
            if callable(algo_cls) and algo_name.endswith('-Agg'):
                optimizer = algo_cls(pop_size)
            else:
                try:
                    optimizer = algo_cls(pop_size=pop_size)
                except TypeError:
                    optimizer = algo_cls(pop_size)
            
            start = time.time()
            best_pos, best_fit, _ = optimizer.optimize(
                objective_fn=func,
                bounds=func.bounds,
                dim=func.dim,
                max_iterations=max_iter
            )
            elapsed = time.time() - start
            
            fitness_vals.append(best_fit)
            times.append(elapsed)
        
        results[algo_name] = {
            'mean': np.mean(fitness_vals),
            'std': np.std(fitness_vals),
            'best': np.min(fitness_vals),
            'time': np.mean(times)
        }
    
    return results


def run_bbob_benchmark(dim: int = 10, max_iter: int = 500, pop_size: int = 50, n_runs: int = 5):
    """Run the complete BBOB benchmark suite"""
    print("=" * 80)
    print(f"BBOB-Style Benchmark (Black-Box Optimization Benchmarking)")
    print(f"Dimension: {dim}, Max Iterations: {max_iter}, Population: {pop_size}, Runs: {n_runs}")
    print("=" * 80)
    
    # Algorithms
    algorithms = {
        'OCA': OverclockingAlgorithm,
        'OCA-Agg': lambda pop_size: OverclockingAlgorithm(pop_size=pop_size, aggressive_voltage=True),
        'PSO': PSO,
        'GWO': GWO,
        'DE': DE,
        'GA': GA,
    }
    
    # Get all functions
    functions = get_all_functions(dim)
    
    # Store all results
    all_results = {}
    win_counts = {name: 0 for name in algorithms.keys()}
    
    print(f"\n{'Function':<25} | ", end="")
    for algo_name in algorithms.keys():
        print(f"{algo_name:<12} | ", end="")
    print("Winner")
    print("-" * (25 + len(algorithms) * 15 + 10))
    
    for func_name, func in functions.items():
        results = run_single_function(algorithms, func, max_iter, pop_size, n_runs)
        all_results[func_name] = results
        
        # Find winner
        best_algo = min(results.items(), key=lambda x: x[1]['mean'])[0]
        win_counts[best_algo] += 1
        
        # Print row
        print(f"{func_name:<25} | ", end="")
        for algo_name in algorithms.keys():
            mean_val = results[algo_name]['mean']
            if mean_val > 1e6:
                print(f"{mean_val:<12.2e} | ", end="")
            else:
                print(f"{mean_val:<12.4f} | ", end="")
        print(f"{'*' if best_algo else ''}{best_algo}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Win Counts Across All 24 Functions")
    print("=" * 80)
    
    sorted_wins = sorted(win_counts.items(), key=lambda x: -x[1])
    for rank, (algo, wins) in enumerate(sorted_wins, 1):
        bar = "█" * wins
        print(f"{rank}. {algo:<12}: {wins:2d} wins {bar}")
    
    # Function group analysis
    print("\n" + "=" * 80)
    print("ANALYSIS BY FUNCTION GROUP")
    print("=" * 80)
    
    groups = {
        'Separable (f1-f5)': ['f1_Sphere', 'f2_Ellipsoidal', 'f3_Rastrigin', 'f4_BuecheRastrigin', 'f5_LinearSlope'],
        'Low Conditioning (f6-f9)': ['f6_AttractiveSector', 'f7_StepEllipsoidal', 'f8_Rosenbrock', 'f9_RosenbrockRotated'],
        'High Conditioning (f10-f14)': ['f10_EllipsoidalRotated', 'f11_Discus', 'f12_BentCigar', 'f13_SharpRidge', 'f14_DifferentPowers'],
        'Multimodal (f15-f19)': ['f15_RastriginRotated', 'f16_Weierstrass', 'f17_SchaffersF7', 'f18_SchaffersF7Ill', 'f19_GriewankRosenbrock'],
        'Weak Structure (f20-f24)': ['f20_Schwefel', 'f21_Gallagher101', 'f22_Gallagher21', 'f23_Katsuura', 'f24_Lunacek'],
    }
    
    for group_name, func_names in groups.items():
        print(f"\n{group_name}:")
        group_wins = {name: 0 for name in algorithms.keys()}
        for func_name in func_names:
            if func_name in all_results:
                best_algo = min(all_results[func_name].items(), key=lambda x: x[1]['mean'])[0]
                group_wins[best_algo] += 1
        
        for algo, wins in sorted(group_wins.items(), key=lambda x: -x[1]):
            if wins > 0:
                print(f"  {algo}: {wins} wins")
    
    print("\n" + "=" * 80)
    print("BBOB Benchmark Complete!")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    # Run benchmark with default settings
    # Dimensions commonly tested: 2, 5, 10, 20, 40
    run_bbob_benchmark(dim=10, max_iter=500, pop_size=50, n_runs=5)
