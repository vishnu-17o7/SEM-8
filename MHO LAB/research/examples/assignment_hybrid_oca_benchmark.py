from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure local imports work regardless of cwd.
THIS_FILE = Path(__file__).resolve()
RESEARCH_ROOT = THIS_FILE.parents[1]
if str(RESEARCH_ROOT / "src") not in sys.path:
    sys.path.append(str(RESEARCH_ROOT / "src"))
if str(RESEARCH_ROOT / "examples") not in sys.path:
    sys.path.append(str(RESEARCH_ROOT / "examples"))

from baselines import GWO, PSO
from oca import OverclockingAlgorithm
from pathfinding_benchmark import RobotNavigation


@dataclass
class ExperimentConfig:
    scenario: str = "Maze"
    n_waypoints: int = 8
    pop_size: int = 40
    tune_iterations: int = 120
    final_iterations: int = 200
    tune_runs: int = 5
    final_runs: int = 10


def make_algo(name: str, pop_size: int, params: Dict[str, float | int | bool]):
    if name == "PSO":
        return PSO(pop_size=pop_size, **params)
    if name == "GWO":
        return GWO(pop_size=pop_size)
    if name == "OCA":
        return OverclockingAlgorithm(pop_size=pop_size, **params)
    raise ValueError(f"Unsupported algorithm: {name}")


def evaluate_runs(
    name: str,
    params: Dict[str, float | int | bool],
    problem: RobotNavigation,
    pop_size: int,
    max_iterations: int,
    seeds: List[int],
) -> Tuple[List[Dict[str, float]], List[np.ndarray], List[np.ndarray]]:
    run_rows: List[Dict[str, float]] = []
    curves: List[np.ndarray] = []
    positions: List[np.ndarray] = []

    for run_idx, seed in enumerate(seeds):
        np.random.seed(seed)
        algo = make_algo(name, pop_size, params)

        start = time.perf_counter()
        best_pos, best_cost, curve = algo.optimize(
            objective_fn=problem.evaluate,
            bounds=problem.bounds,
            dim=problem.dim,
            max_iterations=max_iterations,
        )
        elapsed = time.perf_counter() - start

        is_valid = 1 if problem.is_valid_path(best_pos) else 0
        run_rows.append(
            {
                "algorithm": name,
                "run": run_idx + 1,
                "seed": seed,
                "cost": float(best_cost),
                "time_s": float(elapsed),
                "valid": float(is_valid),
            }
        )
        curves.append(np.asarray(curve, dtype=float))
        positions.append(np.asarray(best_pos, dtype=float))

    return run_rows, curves, positions


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    by_algo: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        by_algo.setdefault(str(row["algorithm"]), []).append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for algo, algo_rows in by_algo.items():
        costs = np.asarray([r["cost"] for r in algo_rows], dtype=float)
        times = np.asarray([r["time_s"] for r in algo_rows], dtype=float)
        valids = np.asarray([r["valid"] for r in algo_rows], dtype=float)

        summary[algo] = {
            "mean_cost": float(np.mean(costs)),
            "best_cost": float(np.min(costs)),
            "worst_cost": float(np.max(costs)),
            "std_cost": float(np.std(costs, ddof=1) if len(costs) > 1 else 0.0),
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times, ddof=1) if len(times) > 1 else 0.0),
            "valid_rate": float(np.mean(valids)),
            "runs": float(len(costs)),
        }
    return summary


def save_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary_csv(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for algo, stats in summary.items():
        row = {"algorithm": algo}
        row.update(stats)
        rows.append(row)

    fieldnames = [
        "algorithm",
        "mean_cost",
        "best_cost",
        "worst_cost",
        "std_cost",
        "mean_time_s",
        "std_time_s",
        "valid_rate",
        "runs",
    ]
    save_csv(path, rows, fieldnames)


def plot_convergence(
    out_path: Path,
    curves_by_algo: Dict[str, List[np.ndarray]],
) -> None:
    style = {
        "OCA": {"color": "#1565C0", "ls": "-", "lw": 2.2},
        "PSO": {"color": "#C62828", "ls": "--", "lw": 2.0},
        "GWO": {"color": "#2E7D32", "ls": "-.", "lw": 2.0},
    }

    fig, ax = plt.subplots(figsize=(9, 5.2))

    for algo in ["PSO", "GWO", "OCA"]:
        algo_curves = curves_by_algo[algo]
        min_len = min(len(c) for c in algo_curves)
        stack = np.vstack([c[:min_len] for c in algo_curves])
        mean_curve = np.mean(stack, axis=0)
        std_curve = np.std(stack, axis=0)
        x = np.arange(1, min_len + 1)

        ax.plot(
            x,
            mean_curve,
            label=algo,
            color=style[algo]["color"],
            linestyle=style[algo]["ls"],
            linewidth=style[algo]["lw"],
        )
        ax.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=style[algo]["color"],
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Cost")
    ax.set_title("Convergence Comparison (Mean +/- SD across 10 runs)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_boxplot(out_path: Path, rows: List[Dict[str, float]]) -> None:
    costs_by_algo = {}
    for algo in ["PSO", "GWO", "OCA"]:
        costs_by_algo[algo] = [r["cost"] for r in rows if r["algorithm"] == algo]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    data = [costs_by_algo["PSO"], costs_by_algo["GWO"], costs_by_algo["OCA"]]
    bp = ax.boxplot(data, tick_labels=["PSO", "GWO", "OCA"], patch_artist=True)
    colors = ["#FFCDD2", "#C8E6C9", "#BBDEFB"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_title("Final Cost Distribution (10 runs)")
    ax.set_ylabel("Objective Cost")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_best_paths(
    out_dir: Path,
    problem: RobotNavigation,
    rows: List[Dict[str, float]],
    positions_by_algo: Dict[str, List[np.ndarray]],
) -> None:
    for algo in ["PSO", "GWO", "OCA"]:
        algo_rows = [r for r in rows if r["algorithm"] == algo]
        best_index = int(np.argmin([r["cost"] for r in algo_rows]))
        best_row = algo_rows[best_index]
        best_pos = positions_by_algo[algo][best_index]
        img_path = out_dir / f"best_path_{algo}.png"
        problem.visualize(algo, best_pos, best_row["cost"], save_path=str(img_path), show=False)


def main() -> None:
    cfg = ExperimentConfig()
    out_dir = RESEARCH_ROOT / "results" / "assignment_oca_hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("OCA Hybrid Assignment Benchmark")
    print("=" * 72)
    print(f"Scenario: {cfg.scenario} | Waypoints: {cfg.n_waypoints}")
    print(f"Population: {cfg.pop_size}")

    problem = RobotNavigation(n_waypoints=cfg.n_waypoints, scenario=cfg.scenario)

    # Candidate hyperparameters for fair tuning with same pop_size and objective budget.
    search_space = {
        "PSO": [
            {"w": 0.7, "c1": 2.0, "c2": 2.0},
            {"w": 0.8, "c1": 1.8, "c2": 1.8},
            {"w": 0.6, "c1": 1.5, "c2": 2.5},
        ],
        "GWO": [{}],
        "OCA": [
            {"num_p_cores": 3, "initial_voltage": 2.0, "final_voltage": 0.0, "aggressive_voltage": False},
            {"num_p_cores": 3, "initial_voltage": 2.0, "final_voltage": 0.0, "aggressive_voltage": True},
            {"num_p_cores": 5, "initial_voltage": 2.0, "final_voltage": 0.0, "aggressive_voltage": False},
            {"num_p_cores": 5, "initial_voltage": 2.0, "final_voltage": 0.0, "aggressive_voltage": True},
        ],
    }

    tune_seeds = list(range(100, 100 + cfg.tune_runs))
    tuning_rows: List[Dict[str, float | str]] = []
    best_params: Dict[str, Dict[str, float | int | bool]] = {}

    for algo in ["PSO", "GWO", "OCA"]:
        best_score = float("inf")
        best_cfg = None

        print(f"\nTuning {algo}...")
        for idx, params in enumerate(search_space[algo], start=1):
            run_rows, _, _ = evaluate_runs(
                name=algo,
                params=params,
                problem=problem,
                pop_size=cfg.pop_size,
                max_iterations=cfg.tune_iterations,
                seeds=tune_seeds,
            )
            stats = summarize(run_rows)[algo]
            # Penalize low feasibility to favor valid path search.
            score = stats["mean_cost"] + (1.0 - stats["valid_rate"]) * 1000.0

            tuning_rows.append(
                {
                    "algorithm": algo,
                    "candidate": float(idx),
                    "params": str(params),
                    "mean_cost": stats["mean_cost"],
                    "std_cost": stats["std_cost"],
                    "valid_rate": stats["valid_rate"],
                    "score": score,
                }
            )
            print(
                f"  cand {idx}: mean={stats['mean_cost']:.2f}, "
                f"std={stats['std_cost']:.2f}, valid={100*stats['valid_rate']:.1f}%, score={score:.2f}"
            )

            if score < best_score:
                best_score = score
                best_cfg = params

        if best_cfg is None:
            raise RuntimeError(f"No config selected for {algo}")
        best_params[algo] = best_cfg
        print(f"  -> Selected for {algo}: {best_cfg}")

    # Final 10-run benchmark with selected parameters.
    final_seeds = list(range(cfg.final_runs))
    final_rows: List[Dict[str, float]] = []
    curves_by_algo: Dict[str, List[np.ndarray]] = {}
    positions_by_algo: Dict[str, List[np.ndarray]] = {}

    print("\nRunning final benchmark (10 runs each)...")
    for algo in ["PSO", "GWO", "OCA"]:
        rows, curves, positions = evaluate_runs(
            name=algo,
            params=best_params[algo],
            problem=problem,
            pop_size=cfg.pop_size,
            max_iterations=cfg.final_iterations,
            seeds=final_seeds,
        )
        final_rows.extend(rows)
        curves_by_algo[algo] = curves
        positions_by_algo[algo] = positions
        stats = summarize(rows)[algo]
        print(
            f"  {algo}: mean={stats['mean_cost']:.2f}, best={stats['best_cost']:.2f}, "
            f"std={stats['std_cost']:.2f}, valid={100*stats['valid_rate']:.1f}%, "
            f"mean_time={stats['mean_time_s']:.3f}s"
        )

    summary = summarize(final_rows)

    save_csv(
        out_dir / "tuning_results.csv",
        tuning_rows,
        ["algorithm", "candidate", "params", "mean_cost", "std_cost", "valid_rate", "score"],
    )
    save_csv(
        out_dir / "final_runs.csv",
        final_rows,
        ["algorithm", "run", "seed", "cost", "time_s", "valid"],
    )
    save_summary_csv(out_dir / "summary_stats.csv", summary)

    plot_convergence(out_dir / "convergence_comparison.png", curves_by_algo)
    plot_boxplot(out_dir / "final_cost_boxplot.png", final_rows)
    save_best_paths(out_dir, problem, final_rows, positions_by_algo)

    # Save experiment metadata and chosen configs.
    meta_rows = []
    for algo in ["PSO", "GWO", "OCA"]:
        meta_rows.append(
            {
                "algorithm": algo,
                "selected_params": str(best_params[algo]),
                "scenario": cfg.scenario,
                "n_waypoints": float(cfg.n_waypoints),
                "pop_size": float(cfg.pop_size),
                "tune_iterations": float(cfg.tune_iterations),
                "final_iterations": float(cfg.final_iterations),
                "tune_runs": float(cfg.tune_runs),
                "final_runs": float(cfg.final_runs),
            }
        )
    save_csv(
        out_dir / "experiment_config.csv",
        meta_rows,
        [
            "algorithm",
            "selected_params",
            "scenario",
            "n_waypoints",
            "pop_size",
            "tune_iterations",
            "final_iterations",
            "tune_runs",
            "final_runs",
        ],
    )

    print("\nSaved outputs:")
    print(f"  - {out_dir / 'tuning_results.csv'}")
    print(f"  - {out_dir / 'final_runs.csv'}")
    print(f"  - {out_dir / 'summary_stats.csv'}")
    print(f"  - {out_dir / 'experiment_config.csv'}")
    print(f"  - {out_dir / 'convergence_comparison.png'}")
    print(f"  - {out_dir / 'final_cost_boxplot.png'}")
    print(f"  - {out_dir / 'best_path_PSO.png'}")
    print(f"  - {out_dir / 'best_path_GWO.png'}")
    print(f"  - {out_dir / 'best_path_OCA.png'}")


if __name__ == "__main__":
    main()
