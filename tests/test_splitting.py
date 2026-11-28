"""Offline tests for train/validation/test dataset splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iml_feature_effects.data import Dataset
from iml_feature_effects.evaluate import DatasetSplit, split_dataset


def _dataset(n_samples: int = 100) -> Dataset:
    ids = np.arange(n_samples)
    X = pd.DataFrame({"first": ids, "second": ids % 7}, index=pd.Index(ids, name="row_id"))
    y = pd.Series(ids * 3.5, index=X.index, name="target")
    return Dataset(name="synthetic", X=X, y=y)


def _partitions(split: DatasetSplit) -> tuple[tuple[pd.DataFrame, pd.Series], ...]:
    return (
        (split.X_train, split.y_train),
        (split.X_val, split.y_val),
        (split.X_test, split.y_test),
    )


def test_random_split_covers_rows_preserves_alignment_and_approximate_proportions():
    dataset = _dataset()
    split = split_dataset(dataset)
    partitions = _partitions(split)

    assert [len(X) for X, _ in partitions] == [60, 20, 20]
    assert all(isinstance(X, pd.DataFrame) and isinstance(y, pd.Series) for X, y in partitions)
    assert all(X.columns.tolist() == dataset.X.columns.tolist() for X, _ in partitions)
    assert all(X.index.equals(y.index) for X, y in partitions)
    assert all(y.equals(X["first"] * 3.5) for X, y in partitions)
    combined_indices = [index for X, _ in partitions for index in X.index]
    assert sorted(combined_indices) == dataset.X.index.tolist()
    assert len(combined_indices) == len(set(combined_indices))


def test_split_is_deterministic_for_same_seed():
    dataset = _dataset()
    first = split_dataset(dataset, seed=17)
    second = split_dataset(dataset, seed=17)

    for (first_X, first_y), (second_X, second_y) in zip(_partitions(first), _partitions(second)):
        pd.testing.assert_frame_equal(first_X, second_X)
        pd.testing.assert_series_equal(first_y, second_y)


def test_group_aware_split_keeps_exact_features_together_even_with_different_targets():
    group_ids = np.repeat(np.arange(40), 3)
    X = pd.DataFrame({"configuration": group_ids, "other": group_ids % 5})
    y = pd.Series(group_ids * 10 + np.tile([0.0, 0.1, 0.2], 40), name="target")
    dataset = Dataset(name="repeated", X=X, y=y)
    source = X.assign(target=y)
    assert any(
        targets.nunique() > 1
        for _, targets in source.groupby(["configuration", "other"])["target"]
    )

    split = split_dataset(dataset, seed=9, group_aware=True)
    repeated_split = split_dataset(dataset, seed=9, group_aware=True)
    partitions = _partitions(split)
    for (first_X, first_y), (second_X, second_y) in zip(partitions, _partitions(repeated_split)):
        pd.testing.assert_frame_equal(first_X, second_X)
        pd.testing.assert_series_equal(first_y, second_y)

    sizes = [len(partition_X) for partition_X, _ in partitions]
    assert sizes == [72, 24, 24]
    assert all(partition_X.index.equals(partition_y.index) for partition_X, partition_y in partitions)
    assert all(
        partition_y.equals(partition_X["configuration"] * 10 + (partition_X.index % 3) / 10)
        for partition_X, partition_y in partitions
    )

    partition_configurations = [
        set(partition_X.itertuples(index=False, name=None))
        for partition_X, _ in partitions
    ]
    assert set.union(*partition_configurations) == set(X.itertuples(index=False, name=None))
    assert all(
        partition_configurations[left].isdisjoint(partition_configurations[right])
        for left in range(len(partition_configurations))
        for right in range(left + 1, len(partition_configurations))
    )


@pytest.mark.parametrize("group_aware", [False, True])
def test_split_rejects_misaligned_feature_target_indices(group_aware):
    dataset = _dataset()
    mismatched_y = dataset.y.copy()
    mismatched_y.index = mismatched_y.index + 1
    unaligned = Dataset(name=dataset.name, X=dataset.X, y=mismatched_y)

    with pytest.raises(ValueError, match="aligned rows and indices"):
        split_dataset(unaligned, group_aware=group_aware)
