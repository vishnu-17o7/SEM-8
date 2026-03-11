import argparse
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from astgcn import ASTGCN, cheb_polynomials, scaled_laplacian


@dataclass
class Config:
    n_nodes: int = 20
    n_features: int = 3
    total_steps: int = 1800
    pred_len: int = 12
    recent_len: int = 24
    daily_len: int = 12
    weekly_len: int = 24
    day_stride: int = 288
    week_stride: int = 288 * 7
    k_order: int = 3
    n_blocks: int = 2
    n_chev_filters: int = 32
    n_time_filters: int = 32
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 10


def max_history_window(cfg: Config) -> int:
    daily_blocks = cfg.daily_len // cfg.pred_len
    weekly_blocks = cfg.weekly_len // cfg.pred_len
    return max(cfg.recent_len, daily_blocks * cfg.day_stride, weekly_blocks * cfg.week_stride)


def make_ring_adjacency(n_nodes: int) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        adj[i, i] = 1.0
        adj[i, (i - 1) % n_nodes] = 1.0
        adj[i, (i + 1) % n_nodes] = 1.0
    return adj


def generate_mock_traffic(cfg: Config, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(cfg.total_steps)
    daily = np.sin(2 * np.pi * t / cfg.day_stride)
    weekly = np.sin(2 * np.pi * t / cfg.week_stride)
    data = np.zeros((cfg.total_steps, cfg.n_nodes, cfg.n_features), dtype=np.float32)
    node_scales = rng.uniform(0.8, 1.2, size=cfg.n_nodes)
    node_phase = rng.uniform(0, 2 * np.pi, size=cfg.n_nodes)

    for n in range(cfg.n_nodes):
        flow = (
            40
            + 12 * node_scales[n] * np.sin(2 * np.pi * t / cfg.day_stride + node_phase[n])
            + 7 * weekly
            + rng.normal(0, 1.5, size=cfg.total_steps)
        )
        speed = (
            60
            - 0.25 * flow
            + 4 * np.sin(2 * np.pi * t / (cfg.day_stride / 2) + node_phase[n] / 2)
            + rng.normal(0, 0.6, size=cfg.total_steps)
        )
        occupancy = np.clip(0.03 * flow + rng.normal(0, 0.03, size=cfg.total_steps), 0, 1)
        data[:, n, 0] = flow
        data[:, n, 1] = speed
        data[:, n, 2] = occupancy

    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-6
    data = (data - mean) / std
    return data


def build_samples(data: np.ndarray, cfg: Config) -> Tuple[np.ndarray, ...]:
    max_back = max_history_window(cfg)
    n_steps = data.shape[0]
    daily_blocks = cfg.daily_len // cfg.pred_len
    weekly_blocks = cfg.weekly_len // cfg.pred_len
    xh, xd, xw, y = [], [], [], []

    for t0 in range(max_back, n_steps - cfg.pred_len):
        recent = data[t0 - cfg.recent_len : t0]

        daily = np.concatenate(
            [
                data[t0 - i * cfg.day_stride : t0 - i * cfg.day_stride + cfg.pred_len]
                for i in range(daily_blocks, 0, -1)
            ],
            axis=0,
        )

        weekly = np.concatenate(
            [
                data[t0 - i * cfg.week_stride : t0 - i * cfg.week_stride + cfg.pred_len]
                for i in range(weekly_blocks, 0, -1)
            ],
            axis=0,
        )

        target = data[t0 : t0 + cfg.pred_len, :, 0]

        xh.append(np.transpose(recent, (1, 2, 0)))
        xd.append(np.transpose(daily, (1, 2, 0)))
        xw.append(np.transpose(weekly, (1, 2, 0)))
        y.append(np.transpose(target, (1, 0)))

    return np.array(xh), np.array(xd), np.array(xw), np.array(y)


def run_training(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.daily_len % cfg.pred_len != 0:
        raise ValueError("daily_len must be a multiple of pred_len to match ASTGCN daily-period sampling.")
    if cfg.weekly_len % cfg.pred_len != 0:
        raise ValueError("weekly_len must be a multiple of pred_len to match ASTGCN weekly-period sampling.")

    # Ensure enough synthetic history exists for weekly/daily/recent windows.
    min_samples = 64
    min_total_steps = max_history_window(cfg) + cfg.pred_len + min_samples
    if cfg.total_steps < min_total_steps:
        print(
            f"[info] total_steps={cfg.total_steps} is too small for the configured windows; "
            f"using total_steps={min_total_steps} instead."
        )
        cfg = replace(cfg, total_steps=min_total_steps)

    adj = make_ring_adjacency(cfg.n_nodes)
    l_tilde = scaled_laplacian(adj)
    cheb = cheb_polynomials(l_tilde, cfg.k_order)

    data = generate_mock_traffic(cfg)
    xh, xd, xw, y = build_samples(data, cfg)

    if len(y) < 2:
        raise RuntimeError(
            "Not enough samples for a train/test split. Increase total_steps or reduce lookback windows."
        )

    split = max(1, int(0.8 * len(y)))
    if split >= len(y):
        split = len(y) - 1
    train_ds = TensorDataset(
        torch.tensor(xh[:split], dtype=torch.float32),
        torch.tensor(xd[:split], dtype=torch.float32),
        torch.tensor(xw[:split], dtype=torch.float32),
        torch.tensor(y[:split], dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(xh[split:], dtype=torch.float32),
        torch.tensor(xd[split:], dtype=torch.float32),
        torch.tensor(xw[split:], dtype=torch.float32),
        torch.tensor(y[split:], dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = ASTGCN(
        n_nodes=cfg.n_nodes,
        in_channels=cfg.n_features,
        pred_len=cfg.pred_len,
        recent_len=cfg.recent_len,
        daily_len=cfg.daily_len,
        weekly_len=cfg.weekly_len,
        cheb_polys=cheb,
        n_blocks=cfg.n_blocks,
        k_order=cfg.k_order,
        n_chev_filters=cfg.n_chev_filters,
        n_time_filters=cfg.n_time_filters,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for b_xh, b_xd, b_xw, b_y in train_loader:
            b_xh, b_xd, b_xw, b_y = b_xh.to(device), b_xd.to(device), b_xw.to(device), b_y.to(device)
            optimizer.zero_grad()
            pred = model(b_xh, b_xd, b_xw)
            loss = criterion(pred, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_y.size(0)
        train_loss /= len(train_ds)

        model.eval()
        mse, mae, count = 0.0, 0.0, 0
        with torch.no_grad():
            for b_xh, b_xd, b_xw, b_y in test_loader:
                b_xh, b_xd, b_xw, b_y = b_xh.to(device), b_xd.to(device), b_xw.to(device), b_y.to(device)
                pred = model(b_xh, b_xd, b_xw)
                diff = pred - b_y
                mse += torch.sum(diff * diff).item()
                mae += torch.sum(torch.abs(diff)).item()
                count += diff.numel()

        if count == 0:
            raise RuntimeError("Test set is empty; unable to compute evaluation metrics.")

        mse /= count
        mae /= count
        rmse = np.sqrt(mse)
        print(f"Epoch {epoch:02d} | train_mse={train_loss:.5f} | test_rmse={rmse:.5f} | test_mae={mae:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASTGCN on mock traffic data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=20)
    args = parser.parse_args()

    cfg = Config(n_nodes=args.nodes, epochs=args.epochs)
    run_training(cfg)


if __name__ == "__main__":
    main()
