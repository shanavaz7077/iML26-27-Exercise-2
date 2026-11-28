"""Run a controlled decision-tree maximum-depth sweep.

Usage::

    uv run python scripts/run_tree_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from iml_feature_effects.data import (
    load_california_housing,
    load_concrete_strength,
    load_power_plant,
)
from iml_feature_effects.evaluate import evaluate_tree_depth_sweep, split_dataset

RESULTS = Path("results")
SEED = 0


def plot_tree_sweep(results: pd.DataFrame, dataset: str, path: Path) -> None:
    """Plot training and validation R² against maximum tree depth for one dataset."""
    data = results.loc[results["dataset"].eq(dataset)]
    labels = data["max_depth"].astype(str).tolist()
    positions = range(len(data))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(positions, data["train_r2"], marker="o", label="Training R²")
    ax.plot(positions, data["val_r2"], marker="o", label="Validation R²")
    ax.set_xticks(list(positions), labels)
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("R²")
    ax.set_title(f"Decision tree depth sweep: {dataset}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    datasets = [
        (load_concrete_strength(), True),
        (load_california_housing(), False),
        (load_power_plant(), True),
    ]
    results = pd.concat(
        [
            evaluate_tree_depth_sweep(
                dataset,
                split_dataset(dataset, seed=SEED, group_aware=group_aware),
            )
            for dataset, group_aware in datasets
        ],
        ignore_index=True,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / "tree_depth_sweep.csv", index=False)
    for dataset, _ in datasets:
        plot_tree_sweep(
            results,
            dataset.name,
            RESULTS / f"tree_depth_sweep_{dataset.name}.png",
        )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
