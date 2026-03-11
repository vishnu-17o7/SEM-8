# ASTGCN (Paper Implementation)

This repo contains a clean PyTorch implementation of the AAAI-19 paper:

**Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting (ASTGCN)**

## What is implemented

- Spatial attention and temporal attention modules
- Chebyshev graph convolution with spatial attention coupling
- Spatial-temporal (ST) block: temporal attention -> spatial attention -> graph convolution -> temporal convolution -> residual + layer norm
- Three-component architecture from the paper:
  - recent component
  - daily-periodic component
  - weekly-periodic component
- Learnable weighted fusion of the three components

Main code:
- `astgcn.py`

## Mock data training script

Since real PeMS data is not bundled, a synthetic traffic generator is included:

- `train_astgcn_mock.py`

It creates graph-structured traffic-like time series with daily and weekly periodicity, speed/occupancy correlations, and noise.

## Run

1. Install dependencies (example):

```bash
pip install numpy torch scikit-learn
```

2. Train on mock data:

```bash
python train_astgcn_mock.py --epochs 10 --nodes 20
```

Output logs include training MSE and test RMSE/MAE per epoch.

## Notes

- This implementation follows the model design in the paper, but uses simplified mock-data sampling and adjacency construction for easy reproducibility.
- To use real data, replace mock-data generation with a dataset loader and keep input tensor shapes:
  - `x_recent`: `[batch, nodes, features, T_recent]`
  - `x_daily`: `[batch, nodes, features, T_daily]`
  - `x_weekly`: `[batch, nodes, features, T_weekly]`
  - target `y`: `[batch, nodes, T_pred]`
