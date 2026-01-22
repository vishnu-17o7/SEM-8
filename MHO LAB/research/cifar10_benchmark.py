"""
CIFAR-10 Neural Network Hyperparameter Optimization Benchmark
==============================================================
Uses OCA and baseline algorithms to optimize neural network hyperparameters
for CIFAR-10 image classification.

Optimized Hyperparameters:
1. Learning Rate (log scale: 1e-5 to 1e-1)
2. Dropout Rate (0.0 to 0.7)
3. Hidden Layer 1 Size (32 to 512)
4. Hidden Layer 2 Size (32 to 256)
5. Batch Size (16 to 128)
6. L2 Regularization (1e-6 to 1e-2)
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# TensorFlow setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from oca import OverclockingAlgorithm
from baselines import PSO, GWO, DE

# ============================================================================
# CIFAR-10 DATA PREPARATION
# ============================================================================

def load_cifar10_subset(subset_size=5000):
    """
    Load a subset of CIFAR-10 for faster optimization.
    Using smaller subset to speed up hyperparameter search.
    """
    print("📦 Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten for MLP (we use MLP for faster training, not CNN)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    # Take subset for faster optimization
    if subset_size < len(x_train):
        indices = np.random.choice(len(x_train), subset_size, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]
    
    # Split for validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(x_train)}")
    print(f"   Validation samples: {len(x_val)}")
    print(f"   Test samples: {len(x_test)}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# ============================================================================
# NEURAL NETWORK BUILDER
# ============================================================================

def build_and_train_model(hyperparams, train_data, val_data, epochs=5, verbose=0):
    """
    Build and train a neural network with given hyperparameters.
    
    hyperparams: [learning_rate, dropout, hidden1, hidden2, batch_size, l2_reg]
    Returns: validation accuracy (we optimize for maximum accuracy)
    """
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # Decode hyperparameters
    lr = 10 ** hyperparams[0]  # Log scale: will be in range [-5, -1]
    dropout = hyperparams[1]
    hidden1 = int(hyperparams[2])
    hidden2 = int(hyperparams[3])
    batch_size = int(hyperparams[4])
    l2_reg = 10 ** hyperparams[5]  # Log scale
    
    # Clamp values to valid ranges
    lr = np.clip(lr, 1e-5, 0.1)
    dropout = np.clip(dropout, 0.0, 0.7)
    hidden1 = np.clip(hidden1, 32, 512)
    hidden2 = np.clip(hidden2, 32, 256)
    batch_size = np.clip(batch_size, 16, 128)
    l2_reg = np.clip(l2_reg, 1e-6, 0.01)
    
    try:
        # Build model
        model = keras.Sequential([
            layers.Input(shape=(3072,)),  # 32x32x3 flattened
            layers.Dense(hidden1, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(l2_reg)),
            layers.Dropout(dropout),
            layers.Dense(hidden2, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(l2_reg)),
            layers.Dropout(dropout),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=int(batch_size),
            verbose=verbose
        )
        
        # Evaluate
        _, val_acc = model.evaluate(x_val, y_val, verbose=0)
        
        # Clear session to prevent memory buildup
        keras.backend.clear_session()
        
        return val_acc
        
    except Exception as e:
        keras.backend.clear_session()
        return 0.0  # Return worst score on error

# ============================================================================
# OBJECTIVE FUNCTION FOR OPTIMIZATION
# ============================================================================

class CIFAR10Objective:
    """
    Objective function wrapper for CIFAR-10 hyperparameter optimization.
    Optimizers minimize, so we return negative accuracy.
    """
    
    def __init__(self, train_data, val_data, epochs=5):
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.eval_count = 0
        self.best_acc = 0.0
        self.best_params = None
    
    def __call__(self, hyperparams):
        """
        Evaluate hyperparameters.
        Returns negative accuracy (since optimizers minimize).
        """
        self.eval_count += 1
        
        acc = build_and_train_model(
            hyperparams, 
            self.train_data, 
            self.val_data, 
            epochs=self.epochs
        )
        
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_params = hyperparams.copy()
        
        # Print progress every 10 evaluations
        if self.eval_count % 10 == 0:
            print(f"      Eval {self.eval_count}: Current acc={acc:.4f}, Best={self.best_acc:.4f}")
        
        return -acc  # Negative because we minimize

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_cifar10_benchmark(pop_size=10, max_iterations=15, epochs_per_eval=3):
    """
    Run CIFAR-10 hyperparameter optimization benchmark.
    
    Due to the computational cost, we use:
    - Small population (10)
    - Few iterations (15)
    - Few epochs per evaluation (3)
    - Subset of data (5000 samples)
    """
    
    print("=" * 80)
    print("🧠 CIFAR-10 NEURAL NETWORK HYPERPARAMETER OPTIMIZATION BENCHMARK")
    print("=" * 80)
    
    # Load data
    train_data, val_data, test_data = load_cifar10_subset(subset_size=5000)
    
    # Hyperparameter search space (6 dimensions)
    # [log_lr, dropout, hidden1, hidden2, batch_size, log_l2]
    bounds = (
        np.array([-5, 0.0, 32, 32, 16, -6]),   # Lower bounds
        np.array([-1, 0.7, 512, 256, 128, -2]) # Upper bounds
    )
    dim = 6
    
    print(f"\n📐 Search Space (6D):")
    print(f"   Learning Rate: 1e-5 to 1e-1 (log scale)")
    print(f"   Dropout: 0.0 to 0.7")
    print(f"   Hidden Layer 1: 32 to 512 neurons")
    print(f"   Hidden Layer 2: 32 to 256 neurons")
    print(f"   Batch Size: 16 to 128")
    print(f"   L2 Regularization: 1e-6 to 1e-2 (log scale)")
    
    print(f"\n⚙️ Optimization Settings:")
    print(f"   Population Size: {pop_size}")
    print(f"   Max Iterations: {max_iterations}")
    print(f"   Epochs per Eval: {epochs_per_eval}")
    print(f"   Total NN trainings: ~{pop_size * max_iterations}")
    
    # Initialize algorithms
    algorithms = {
        'OCA': OverclockingAlgorithm(pop_size=pop_size),
        'PSO': PSO(pop_size=pop_size),
        'GWO': GWO(pop_size=pop_size),
        'DE': DE(pop_size=pop_size),
    }
    
    results = {}
    
    for algo_name, algo in algorithms.items():
        print(f"\n{'='*60}")
        print(f"🔧 Running {algo_name}...")
        print(f"{'='*60}")
        
        # Create fresh objective for each algorithm
        objective = CIFAR10Objective(train_data, val_data, epochs=epochs_per_eval)
        
        start_time = time.time()
        best_pos, best_fit, convergence = algo.optimize(
            objective, bounds, dim, max_iterations
        )
        elapsed = time.time() - start_time
        
        # Decode best hyperparameters
        best_lr = 10 ** best_pos[0]
        best_dropout = best_pos[1]
        best_h1 = int(best_pos[2])
        best_h2 = int(best_pos[3])
        best_batch = int(best_pos[4])
        best_l2 = 10 ** best_pos[5]
        
        # Store results
        results[algo_name] = {
            'best_accuracy': -best_fit,
            'best_params': {
                'learning_rate': best_lr,
                'dropout': best_dropout,
                'hidden1': best_h1,
                'hidden2': best_h2,
                'batch_size': best_batch,
                'l2_reg': best_l2
            },
            'time': elapsed,
            'evaluations': objective.eval_count
        }
        
        print(f"\n   ✅ {algo_name} Complete!")
        print(f"   Best Validation Accuracy: {-best_fit:.4f} ({-best_fit*100:.2f}%)")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Best Hyperparameters:")
        print(f"      Learning Rate: {best_lr:.6f}")
        print(f"      Dropout: {best_dropout:.3f}")
        print(f"      Hidden Layer 1: {best_h1}")
        print(f"      Hidden Layer 2: {best_h2}")
        print(f"      Batch Size: {best_batch}")
        print(f"      L2 Reg: {best_l2:.6f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Algorithm':<12} {'Val Accuracy':<15} {'Time':<12} {'Evaluations':<12}")
    print("-" * 55)
    
    winner = max(results.keys(), key=lambda x: results[x]['best_accuracy'])
    
    for algo_name in ['OCA', 'PSO', 'GWO', 'DE']:
        r = results[algo_name]
        acc_str = f"{r['best_accuracy']*100:.2f}%"
        if algo_name == winner:
            print(f"[{algo_name:<10}] [{acc_str:<13}] {r['time']:<12.1f} {r['evaluations']}")
        else:
            print(f" {algo_name:<11}  {acc_str:<14} {r['time']:<12.1f} {r['evaluations']}")
    
    print(f"\n🏆 WINNER: {winner} with {results[winner]['best_accuracy']*100:.2f}% validation accuracy!")
    
    # Test the winner's model on test set
    print(f"\n📈 Evaluating {winner}'s best model on TEST SET...")
    
    best_params = results[winner]['best_params']
    test_params = [
        np.log10(best_params['learning_rate']),
        best_params['dropout'],
        best_params['hidden1'],
        best_params['hidden2'],
        best_params['batch_size'],
        np.log10(best_params['l2_reg'])
    ]
    
    # Train with more epochs for final evaluation
    x_train_full = np.vstack([train_data[0], val_data[0]])
    y_train_full = np.vstack([train_data[1], val_data[1]])
    
    test_acc = build_and_train_model(
        test_params,
        (x_train_full, y_train_full),
        test_data,
        epochs=10,
        verbose=1
    )
    
    print(f"\n🎯 FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run benchmark
    # Using smaller settings for reasonable runtime
    results = run_cifar10_benchmark(
        pop_size=8,           # 8 agents per algorithm
        max_iterations=12,    # 12 iterations
        epochs_per_eval=3     # 3 epochs per NN training
    )
    
    print("\n✅ CIFAR-10 BENCHMARK COMPLETE!")
