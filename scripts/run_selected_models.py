"""Select representative tree depths from validation results and evaluate them on test data.

Usage::

    uv run python scripts/run_selected_models.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from iml_feature_effects.data import (
    load_california_housing,
    load_concrete_strength,
    load_power_plant,
)
from iml_feature_effects.evaluate import (
    evaluate_selected_tree_models,
    evaluate_tree_depth_sweep,
    select_tree_representatives,
    split_dataset,
)

RESULTS = Path("results")
SEED = 0


def main() -> None:
    datasets = [
        (load_concrete_strength(), True),
        (load_california_housing(), False),
        (load_power_plant(), True),
    ]
    score_frames = []
    selection_frames = []

    for dataset, group_aware in datasets:
        split = split_dataset(dataset, seed=SEED, group_aware=group_aware)
        sweep = evaluate_tree_depth_sweep(dataset, split)
        selections = select_tree_representatives(sweep)
        selection_frames.append(selections)
        score_frames.append(evaluate_selected_tree_models(dataset, split, selections))

    selections = pd.concat(selection_frames, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS / "selected_model_scores.csv", index=False)
    print("Selected depths from training/validation sweep:")
    print(selections[["dataset", "fit_category", "max_depth", "selection_rule"]].to_string(index=False))
    print("\nUntouched test-set results:")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
