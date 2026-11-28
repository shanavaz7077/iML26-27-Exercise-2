"""Tests for train/validation-based tree selection and selected-model scoring."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from iml_feature_effects.data import Dataset
from iml_feature_effects.evaluate import (
    DatasetSplit,
    evaluate_selected_tree_models,
    select_tree_representatives,
    split_dataset,
)

evaluate_module = import_module("iml_feature_effects.evaluate")

DEPTHS = [1, 2, 3, 4, 6, 8, 12, "unlimited"]


def _sweep_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["synthetic"] * 8,
            "max_depth": DEPTHS,
            "actual_depth": [1, 2, 3, 4, 6, 8, 12, 14],
            "leaf_count": [2, 4, 8, 16, 40, 60, 75, 80],
            "train_samples": [80] * 8,
            "val_samples": [20] * 8,
            "train_r2": [0.40, 0.65, 0.82, 0.90, 0.92, 0.95, 0.98, 0.99],
            "val_r2": [0.20, 0.55, 0.795, 0.80, 0.70, 0.73, 0.78, 0.78],
            "train_val_gap": [0.20, 0.10, 0.025, 0.10, 0.22, 0.22, 0.20, 0.21],
        }
    )


def _dataset() -> Dataset:
    values = np.arange(100)
    X = pd.DataFrame(
        {"x": values, "x_squared": values**2},
        index=pd.Index(values, name="sample_id"),
    )
    y = pd.Series(np.sin(values / 7) + values / 30, index=X.index, name="target")
    return Dataset(name="synthetic", X=X, y=y)


def test_selection_uses_shallowest_tolerance_and_largest_gap_with_shallow_tie_break():
    sweep = _sweep_table()
    sweep["test_r2"] = [1000, -1000, -2000, -3000, 9000, 8000, 7000, 6000]

    selected = select_tree_representatives(sweep)
    repeated = select_tree_representatives(sweep)
    pd.testing.assert_frame_equal(selected, repeated)

    assert selected["fit_category"].tolist() == ["underfit", "well_fit", "overfit"]
    assert selected["max_depth"].tolist() == [1, 3, 6]
    assert "relative" in selected.iloc[0]["selection_rule"]
    assert "exploratory" in selected.iloc[1]["selection_rule"]
    assert "ties select the shallower depth" in selected.iloc[2]["selection_rule"]


def test_selection_reports_when_no_overfit_candidate_meets_criteria():
    sweep = _sweep_table()
    sweep.loc[sweep["max_depth"].isin([6, 8, 12, "unlimited"]), "train_r2"] = 0.80

    selected = select_tree_representatives(sweep)
    overfit = selected.loc[selected["fit_category"].eq("overfit")].iloc[0]

    assert overfit["max_depth"] == "not_selected"
    assert pd.isna(overfit["train_r2"])
    assert "No deeper candidate" in overfit["selection_rule"]


def test_selected_model_fits_training_rows_and_predicts_test_once(monkeypatch):
    class RecordingTree(DecisionTreeRegressor):
        fitted_instances: list[RecordingTree] = []

        def fit(self, X, y, sample_weight=None, check_input=True):
            self.fit_indices_ = tuple(X.index)
            self.prediction_indices_ = []
            type(self).fitted_instances.append(self)
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

        def predict(self, X, check_input=True):
            self.prediction_indices_.append(tuple(X.index))
            return super().predict(X, check_input=check_input)

    dataset = _dataset()
    split: DatasetSplit = split_dataset(dataset, seed=0)
    selected = select_tree_representatives(_sweep_table())
    RecordingTree.fitted_instances = []
    monkeypatch.setattr(
        evaluate_module,
        "get_tree_depth_candidates",
        lambda: {depth: RecordingTree(max_depth=depth) for depth in (1, 2, 3, 4, 6, 8, 12, None)},
    )

    scores = evaluate_selected_tree_models(dataset, split, selected)

    assert len(scores) == 3
    assert set(scores["fit_category"]) == {"underfit", "well_fit", "overfit"}
    assert all(model.fit_indices_ == tuple(split.X_train.index) for model in RecordingTree.fitted_instances)
    assert all(
        model.prediction_indices_
        == [tuple(split.X_train.index), tuple(split.X_val.index), tuple(split.X_test.index)]
        for model in RecordingTree.fitted_instances
    )
    assert scores["test_samples"].eq(len(split.X_test)).all()
    assert scores[["train_r2", "val_r2", "test_r2", "test_mae"]].notna().all().all()
