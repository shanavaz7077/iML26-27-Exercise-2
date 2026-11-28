"""Offline tests for the controlled decision-tree depth sweep."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from iml_feature_effects.data import Dataset
from iml_feature_effects.evaluate import DatasetSplit, split_dataset, evaluate_tree_depth_sweep

evaluate_module = import_module("iml_feature_effects.evaluate")
from iml_feature_effects.models import get_tree_depth_candidates


EXPECTED_COLUMNS = [
    "dataset",
    "max_depth",
    "actual_depth",
    "leaf_count",
    "train_samples",
    "val_samples",
    "train_r2",
    "val_r2",
    "train_val_gap",
]


def _dataset() -> Dataset:
    ids = np.arange(100)
    X = pd.DataFrame(
        {"x1": ids, "x2": ids % 9},
        index=pd.Index(ids, name="sample_id"),
    )
    y = pd.Series(np.sin(ids / 8) + ids / 40, index=X.index, name="target")
    return Dataset(name="synthetic", X=X, y=y)


def test_tree_depth_candidates_have_requested_depths_and_consistent_defaults():
    candidates = get_tree_depth_candidates()

    assert list(candidates) == [1, 2, 3, 4, 6, 8, 12, None]
    assert all(isinstance(model, DecisionTreeRegressor) for model in candidates.values())
    assert all(model.random_state == 0 for model in candidates.values())
    params = [model.get_params() for model in candidates.values()]
    for candidate_params in params:
        candidate_params.pop("max_depth")
    assert all(candidate_params == params[0] for candidate_params in params[1:])


def test_tree_sweep_is_deterministic_and_returns_expected_schema():
    dataset = _dataset()
    split = split_dataset(dataset, seed=13)

    first = evaluate_tree_depth_sweep(dataset, split)
    second = evaluate_tree_depth_sweep(dataset, split)

    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == EXPECTED_COLUMNS
    assert first["dataset"].eq(dataset.name).all()
    assert first["max_depth"].tolist() == [1, 2, 3, 4, 6, 8, 12, "unlimited"]
    assert first["train_samples"].eq(len(split.X_train)).all()
    assert first["val_samples"].eq(len(split.X_val)).all()
    assert np.allclose(first["train_val_gap"], first["train_r2"] - first["val_r2"])


def test_tree_sweep_fits_only_train_and_scores_train_validation(monkeypatch):
    class RecordingRegressor:
        def __init__(self, max_depth: int | None):
            self.max_depth = max_depth
            self.fit_indices: tuple[int, ...] | None = None
            self.prediction_indices: list[tuple[int, ...]] = []

        def fit(self, X: pd.DataFrame, y: pd.Series) -> RecordingRegressor:
            self.fit_indices = tuple(X.index)
            self.mean_ = float(y.mean())
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            self.prediction_indices.append(tuple(X.index))
            return np.full(len(X), self.mean_)

        def get_depth(self) -> int:
            return 0

        def get_n_leaves(self) -> int:
            return 1

    dataset = _dataset()
    split: DatasetSplit = split_dataset(dataset, seed=3)
    models = {
        depth: RecordingRegressor(depth)
        for depth in (1, 2, 3, 4, 6, 8, 12, None)
    }
    monkeypatch.setattr(evaluate_module, "get_tree_depth_candidates", lambda: models)

    result = evaluate_tree_depth_sweep(dataset, split)

    train_indices = tuple(split.X_train.index)
    val_indices = tuple(split.X_val.index)
    test_indices = set(split.X_test.index)
    for model in models.values():
        assert model.fit_indices == train_indices
        assert model.prediction_indices == [train_indices, val_indices]
        assert not test_indices.intersection(model.fit_indices)
        assert all(not test_indices.intersection(indices) for indices in model.prediction_indices)
    assert result["train_samples"].eq(len(train_indices)).all()
    assert result["val_samples"].eq(len(val_indices)).all()
