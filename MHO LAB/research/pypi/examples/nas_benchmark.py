"""
NASBench-Style Benchmark for OCA
Neural Architecture Search Benchmark using surrogate models.

This benchmark tests algorithm performance on finding optimal neural network
architectures in a discrete/mixed search space.

Inspired by:
- NASBench-101: https://github.com/google-research/nasbench
- NASBench-201: https://github.com/D-X-Y/NAS-Bench-201
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os
import hashlib

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


class NASSearchSpace:
    """
    Defines a neural architecture search space similar to NASBench-201.
    
    Search space consists of:
    - Number of layers (depth)
    - Layer widths (channels/units)
    - Operation types (conv, pool, skip, etc.)
    - Activation functions
    - Skip connections pattern
    """
    
    # Operation choices
    OPERATIONS = ['conv1x1', 'conv3x3', 'conv5x5', 'maxpool', 'avgpool', 'skip', 'none']
    ACTIVATIONS = ['relu', 'gelu', 'swish', 'tanh', 'sigmoid']
    
    def __init__(self, n_cells: int = 4, n_nodes_per_cell: int = 4, seed: int = 42):
        self.n_cells = n_cells
        self.n_nodes = n_nodes_per_cell
        self.rng = np.random.RandomState(seed)
        
        # Calculate dimension: each node has edges from all previous nodes
        # Each edge selects an operation
        self.n_edges_per_cell = sum(range(1, n_nodes_per_cell))  # Triangular number
        self.dim = n_cells * (
            self.n_edges_per_cell +  # Operations for edges
            n_nodes_per_cell +        # Activations for nodes
            1                         # Cell width multiplier
        )
        
        self.bounds = (0, 1)
        
        # Build lookup table for architecture evaluations (surrogate)
        self._build_surrogate_model()
    
    def _build_surrogate_model(self):
        """
        Build a surrogate model that estimates architecture performance.
        This simulates what NASBench provides via tabular lookup.
        """
        # Cache for evaluated architectures
        self.cache = {}
        
        # Define performance factors for each operation
        self.op_quality = {
            'conv1x1': 0.7,
            'conv3x3': 1.0,  # Best for feature extraction
            'conv5x5': 0.9,
            'maxpool': 0.5,
            'avgpool': 0.4,
            'skip': 0.6,
            'none': 0.1,
        }
        
        self.act_quality = {
            'relu': 0.9,
            'gelu': 1.0,  # Generally best
            'swish': 0.95,
            'tanh': 0.6,
            'sigmoid': 0.5,
        }
    
    def decode(self, x: np.ndarray) -> Dict:
        """Decode continuous vector to architecture specification"""
        arch = {
            'cells': [],
            'total_params': 0,
            'total_flops': 0,
        }
        
        idx = 0
        base_width = 16
        
        for cell_idx in range(self.n_cells):
            cell = {
                'edges': [],
                'activations': [],
                'width': 0,
            }
            
            # Decode edge operations
            for edge_idx in range(self.n_edges_per_cell):
                op_idx = int(x[idx] * len(self.OPERATIONS))
                op_idx = min(op_idx, len(self.OPERATIONS) - 1)
                cell['edges'].append(self.OPERATIONS[op_idx])
                idx += 1
            
            # Decode node activations
            for node_idx in range(self.n_nodes):
                act_idx = int(x[idx] * len(self.ACTIVATIONS))
                act_idx = min(act_idx, len(self.ACTIVATIONS) - 1)
                cell['activations'].append(self.ACTIVATIONS[act_idx])
                idx += 1
            
            # Decode cell width (multiplier from 0.5x to 2x)
            width_mult = 0.5 + x[idx] * 1.5
            cell['width'] = int(base_width * width_mult * (2 ** cell_idx))
            idx += 1
            
            arch['cells'].append(cell)
        
        # Estimate params and FLOPs
        arch['total_params'] = self._estimate_params(arch)
        arch['total_flops'] = self._estimate_flops(arch)
        
        return arch
    
    def _estimate_params(self, arch: Dict) -> int:
        """Estimate number of parameters"""
        params = 0
        for cell in arch['cells']:
            width = cell['width']
            for op in cell['edges']:
                if 'conv1x1' in op:
                    params += width * width * 1
                elif 'conv3x3' in op:
                    params += width * width * 9
                elif 'conv5x5' in op:
                    params += width * width * 25
        return params
    
    def _estimate_flops(self, arch: Dict) -> int:
        """Estimate FLOPs (assuming 32x32 input)"""
        flops = 0
        spatial = 32
        for cell in arch['cells']:
            width = cell['width']
            for op in cell['edges']:
                if 'conv' in op:
                    kernel_size = int(op[4])
                    flops += spatial * spatial * width * width * kernel_size * kernel_size
            spatial = max(spatial // 2, 1)  # Assume spatial reduction
        return flops
    
    def _get_arch_hash(self, arch: Dict) -> str:
        """Get unique hash for architecture"""
        arch_str = str(arch['cells'])
        return hashlib.md5(arch_str.encode()).hexdigest()
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate architecture performance (returns validation error, lower is better).
        Uses surrogate model to estimate accuracy without actual training.
        """
        arch = self.decode(x)
        arch_hash = self._get_arch_hash(arch)
        
        # Check cache
        if arch_hash in self.cache:
            return self.cache[arch_hash]
        
        # Calculate base score from operations and activations
        score = 0
        n_ops = 0
        
        for cell in arch['cells']:
            # Operation quality
            for op in cell['edges']:
                score += self.op_quality.get(op, 0.5)
                n_ops += 1
            
            # Activation quality
            for act in cell['activations']:
                score += self.act_quality.get(act, 0.5) * 0.5
                n_ops += 1
        
        score /= n_ops
        
        # Penalize extreme parameter counts
        params = arch['total_params']
        if params < 10000:
            score *= 0.7  # Too small - underfitting
        elif params > 10000000:
            score *= 0.8  # Too large - overfitting risk
        
        # Bonus for good skip connection patterns (diversity)
        n_skips = sum(1 for cell in arch['cells'] for op in cell['edges'] if op == 'skip')
        n_nones = sum(1 for cell in arch['cells'] for op in cell['edges'] if op == 'none')
        
        # Some skips are good, too many are bad
        skip_ratio = n_skips / max(n_ops, 1)
        if 0.1 <= skip_ratio <= 0.3:
            score *= 1.1  # Good skip ratio
        elif skip_ratio > 0.5:
            score *= 0.8  # Too many skips
        
        # Penalize too many 'none' operations
        none_ratio = n_nones / max(n_ops, 1)
        score *= (1 - 0.5 * none_ratio)
        
        # Add noise to simulate training variance
        noise = self.rng.normal(0, 0.02)
        score += noise
        
        # Convert to validation error (1 - accuracy)
        val_error = max(0.05, 1 - score)  # Min 5% error (best possible)
        
        # Cache result
        self.cache[arch_hash] = val_error
        
        return val_error


class NASBench101Style:
    """
    NASBench-101 style benchmark with cell-based architecture search.
    Uses a DAG representation within each cell.
    """
    
    def __init__(self, max_nodes: int = 7, max_edges: int = 9, seed: int = 42):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.rng = np.random.RandomState(seed)
        
        # Operations available at each node
        self.ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        
        # Dimension: adjacency matrix (upper triangular) + node operations
        self.n_adj_entries = max_nodes * (max_nodes - 1) // 2
        self.dim = self.n_adj_entries + (max_nodes - 2)  # -2 for input/output nodes
        
        self.bounds = (0, 1)
        self.cache = {}
    
    def decode(self, x: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Decode to adjacency matrix and operations"""
        # Adjacency matrix (upper triangular)
        adj = np.zeros((self.max_nodes, self.max_nodes), dtype=int)
        idx = 0
        for i in range(self.max_nodes):
            for j in range(i + 1, self.max_nodes):
                adj[i, j] = 1 if x[idx] > 0.5 else 0
                idx += 1
        
        # Limit edges
        if np.sum(adj) > self.max_edges:
            # Keep strongest edges
            flat = adj.flatten()
            indices = np.argsort(x[:self.n_adj_entries])[::-1][:self.max_edges]
            adj = np.zeros((self.max_nodes, self.max_nodes), dtype=int)
            for flat_idx in indices:
                i = flat_idx // self.max_nodes
                j = flat_idx % self.max_nodes
                if i < j:
                    adj[i, j] = 1
        
        # Node operations
        ops = ['input']
        for i in range(self.max_nodes - 2):
            op_idx = int(x[self.n_adj_entries + i] * len(self.ops))
            op_idx = min(op_idx, len(self.ops) - 1)
            ops.append(self.ops[op_idx])
        ops.append('output')
        
        return adj, ops
    
    def _is_valid(self, adj: np.ndarray) -> bool:
        """Check if architecture is valid (connected, no cycles)"""
        n = adj.shape[0]
        
        # Check connectivity from input to output
        reachable = np.zeros(n, dtype=bool)
        reachable[0] = True
        
        for _ in range(n):
            for i in range(n):
                if reachable[i]:
                    for j in range(n):
                        if adj[i, j]:
                            reachable[j] = True
        
        return reachable[n - 1]
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate architecture"""
        adj, ops = self.decode(x)
        
        # Check validity
        if not self._is_valid(adj):
            return 1.0  # Invalid architecture - worst score
        
        # Hash for caching
        arch_hash = str(adj.tobytes()) + str(ops)
        if arch_hash in self.cache:
            return self.cache[arch_hash]
        
        # Score based on operations
        op_scores = {
            'conv1x1-bn-relu': 0.7,
            'conv3x3-bn-relu': 1.0,
            'maxpool3x3': 0.5,
        }
        
        score = 0
        for op in ops[1:-1]:  # Exclude input/output
            score += op_scores.get(op, 0.5)
        score /= max(len(ops) - 2, 1)
        
        # Bonus for good connectivity
        n_edges = np.sum(adj)
        if 3 <= n_edges <= 7:
            score *= 1.1
        elif n_edges > self.max_edges:
            score *= 0.7
        
        # Check for skip-like connections (input to output path length)
        if adj[0, -1]:
            score *= 1.05  # Direct skip connection bonus
        
        # Add noise
        noise = self.rng.normal(0, 0.015)
        
        val_error = max(0.04, 1 - score + noise)
        self.cache[arch_hash] = val_error
        
        return val_error


class NASBenchMacro:
    """
    Macro-level NAS benchmark - searching for overall network structure.
    Searches over: depth, width, kernel sizes, pooling positions, etc.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
        # Search space dimensions
        self.max_depth = 20
        self.max_width = 512
        
        # Dimension: depth + per-layer decisions
        self.dim = 1 + self.max_depth * 4  # depth + (width, kernel, pool, skip) per layer
        
        self.bounds = (0, 1)
        self.cache = {}
        
        # Optimal architecture parameters (hidden)
        self._optimal_depth = 12
        self._optimal_widths = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
    
    def decode(self, x: np.ndarray) -> Dict:
        """Decode to macro architecture"""
        arch = {}
        
        # Depth (4 to max_depth)
        arch['depth'] = int(4 + x[0] * (self.max_depth - 4))
        
        arch['layers'] = []
        idx = 1
        
        for i in range(arch['depth']):
            layer = {}
            
            # Width (16 to max_width, prefer powers of 2)
            width_cont = x[idx]
            width = int(16 * (2 ** (width_cont * 5)))  # 16 to 512
            layer['width'] = min(width, self.max_width)
            idx += 1
            
            # Kernel size (1, 3, 5, 7)
            kernel_idx = int(x[idx] * 4)
            layer['kernel'] = [1, 3, 5, 7][min(kernel_idx, 3)]
            idx += 1
            
            # Pooling (none, max, avg) - only at certain positions
            if i > 0 and i % 3 == 0:
                pool_idx = int(x[idx] * 3)
                layer['pool'] = ['none', 'max', 'avg'][min(pool_idx, 2)]
            else:
                layer['pool'] = 'none'
            idx += 1
            
            # Skip connection
            layer['skip'] = x[idx] > 0.5
            idx += 1
            
            arch['layers'].append(layer)
        
        return arch
    
    def _estimate_accuracy(self, arch: Dict) -> float:
        """Estimate accuracy based on architecture similarity to optimal"""
        score = 0.5  # Base score
        
        depth = arch['depth']
        
        # Depth score - optimal around 12
        depth_diff = abs(depth - self._optimal_depth)
        score += 0.1 * max(0, 1 - depth_diff / 10)
        
        # Width progression score
        widths = [l['width'] for l in arch['layers']]
        
        # Check if widths generally increase (good pattern)
        increasing = sum(1 for i in range(len(widths) - 1) if widths[i + 1] >= widths[i])
        score += 0.1 * (increasing / max(len(widths) - 1, 1))
        
        # Kernel size diversity
        kernels = [l['kernel'] for l in arch['layers']]
        unique_kernels = len(set(kernels))
        score += 0.05 * min(unique_kernels / 3, 1)
        
        # Skip connections - some are good
        n_skips = sum(1 for l in arch['layers'] if l['skip'])
        skip_ratio = n_skips / depth
        if 0.2 <= skip_ratio <= 0.5:
            score += 0.1
        
        # Pooling placement
        pool_layers = [i for i, l in enumerate(arch['layers']) if l['pool'] != 'none']
        if len(pool_layers) >= 2:
            score += 0.05
        
        # Total params estimate
        total_params = sum(l['width'] * l['kernel']**2 for l in arch['layers'])
        if 100000 <= total_params <= 5000000:
            score += 0.1  # Good param range
        
        return min(score, 0.96)  # Cap at 96% accuracy
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate architecture - returns validation error"""
        arch = self.decode(x)
        
        arch_hash = str(arch)
        if arch_hash in self.cache:
            return self.cache[arch_hash]
        
        accuracy = self._estimate_accuracy(arch)
        noise = self.rng.normal(0, 0.01)
        
        val_error = max(0.04, 1 - accuracy + noise)
        self.cache[arch_hash] = val_error
        
        return val_error


def run_nas_benchmark(n_runs: int = 5, max_iter: int = 200, pop_size: int = 30):
    """Run complete NAS benchmark suite"""
    print("=" * 80)
    print("NASBench-Style Benchmark: Neural Architecture Search")
    print(f"Runs: {n_runs}, Max Iterations: {max_iter}, Population: {pop_size}")
    print("=" * 80)
    
    # Algorithms
    algorithms = {
        'OCA': OverclockingAlgorithm,
        'OCA-Agg': lambda ps: OverclockingAlgorithm(pop_size=ps, aggressive_voltage=True),
        'PSO': PSO,
        'GWO': GWO,
        'DE': DE,
        'GA': GA,
    }
    
    # NAS benchmarks
    benchmarks = {
        'NAS-201 Style': NASSearchSpace(n_cells=4, n_nodes_per_cell=4),
        'NAS-101 Style': NASBench101Style(max_nodes=7),
        'NAS-Macro': NASBenchMacro(),
    }
    
    all_results = {}
    
    for bench_name, benchmark in benchmarks.items():
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"Dimension: {benchmark.dim}")
        print(f"{'='*60}")
        
        results = {}
        
        for algo_name, algo_cls in algorithms.items():
            errors = []
            times = []
            
            for run in range(n_runs):
                # Reset cache for fair comparison
                benchmark.cache = {}
                
                # Handle factory functions
                if callable(algo_cls) and algo_name.endswith('-Agg'):
                    optimizer = algo_cls(pop_size)
                else:
                    try:
                        optimizer = algo_cls(pop_size=pop_size)
                    except TypeError:
                        optimizer = algo_cls(pop_size)
                
                start = time.time()
                best_pos, best_error, _ = optimizer.optimize(
                    objective_fn=benchmark.evaluate,
                    bounds=benchmark.bounds,
                    dim=benchmark.dim,
                    max_iterations=max_iter
                )
                elapsed = time.time() - start
                
                errors.append(best_error)
                times.append(elapsed)
            
            results[algo_name] = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'best_error': np.min(errors),
                'mean_accuracy': (1 - np.mean(errors)) * 100,
                'best_accuracy': (1 - np.min(errors)) * 100,
                'mean_time': np.mean(times),
            }
        
        all_results[bench_name] = results
        
        # Print results
        print(f"\n{'Algorithm':<12} | {'Mean Acc%':<10} | {'Best Acc%':<10} | {'Mean Err':<10} | {'Time (s)':<8}")
        print("-" * 60)
        
        for algo_name in sorted(results.keys(), key=lambda x: -results[x]['mean_accuracy']):
            r = results[algo_name]
            print(f"{algo_name:<12} | {r['mean_accuracy']:<10.2f} | {r['best_accuracy']:<10.2f} | "
                  f"{r['mean_error']:<10.4f} | {r['mean_time']:<8.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Best Accuracy Achieved Across All NAS Benchmarks")
    print("=" * 80)
    
    summary = {name: {'wins': 0, 'total_acc': 0} for name in algorithms.keys()}
    
    for bench_name, results in all_results.items():
        best_algo = max(results.items(), key=lambda x: x[1]['best_accuracy'])[0]
        summary[best_algo]['wins'] += 1
        
        for algo_name, r in results.items():
            summary[algo_name]['total_acc'] += r['best_accuracy']
    
    print(f"\n{'Algorithm':<12} | {'Wins':<6} | {'Avg Best Acc%':<15}")
    print("-" * 40)
    
    for algo_name in sorted(summary.keys(), key=lambda x: (-summary[x]['wins'], -summary[x]['total_acc'])):
        s = summary[algo_name]
        avg_acc = s['total_acc'] / len(benchmarks)
        print(f"{algo_name:<12} | {s['wins']:<6} | {avg_acc:<15.2f}")
    
    print("\n" + "=" * 80)
    print("NASBench Complete!")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    run_nas_benchmark(n_runs=5, max_iter=200, pop_size=30)
