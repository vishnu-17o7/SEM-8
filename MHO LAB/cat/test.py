import numpy as np
from sklearn.datasets import (
    load_iris, 
    load_wine, 
    load_breast_cancer, 
    fetch_california_housing, 
    fetch_openml,
    make_blobs,
    make_classification
)
import time

def preload_data():
    print("⬇️  Starting Dataset Preload for MHO Lab & Research...")
    
    # 1. Classification (Good for GA Feature Selection & PSO Classification)
    print("   - Loading 'Breast Cancer' (Classic for Feature Selection)...")
    data_bc = load_breast_cancer()
    print(f"     ✅ Loaded: {data_bc.data.shape}")

    print("   - Loading 'Wine' (Multiclass for Optimization benchmarks)...")
    data_wine = load_wine()
    print(f"     ✅ Loaded: {data_wine.data.shape}")

    # 2. Regression (Good for Continuous GA / Function Optimization)
    print("   - Fetching 'California Housing' (Regression benchmark)...")
    # This triggers a download if not cached
    data_housing = fetch_california_housing()
    print(f"     ✅ Downloaded/Loaded: {data_housing.data.shape}")

    # 3. Image Data (Crucial for your GANs / VAEs / Deep Learning)
    print("   - Fetching 'MNIST' (70,000 images, ~50MB)...")
    # This is a heavier download, good to do now.
    data_mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    print(f"     ✅ Downloaded/Loaded: {data_mnist.data.shape}")

    # 4. Synthetic Data Generation (Instant, but good to test parameters)
    print("   - Generating Synthetic Blobs (For PSO Clustering)...")
    X, y = make_blobs(n_samples=1000, centers=4, n_features=10, random_state=42)
    print(f"     ✅ Generated: {X.shape}")

    print("\n🎉 All datasets preloaded and cached!")

if __name__ == "__main__":
    preload_data()