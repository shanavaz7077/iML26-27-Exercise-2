"""Fit every model on every dataset, quantify its fit and compute feature effects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from iml_feature_effects.data import Dataset
from iml_feature_effects.effects import EffectProperty, FeatureEffect


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test partitions of a :class:`Dataset`."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_dataset(dataset: Dataset, seed: int = 0, group_aware: bool = False) -> DatasetSplit:
    """Split a dataset reproducibly into approximately 60/20/20 partitions.

    Set ``group_aware=True`` to keep rows with identical feature values together. This is
    appropriate for Concrete Strength and deliberately groups on ``X`` only, not the target.

    :param dataset: Dataset whose feature and target rows are aligned.
    :param seed: Random seed used by both split stages.
    :param group_aware: Whether to keep exact duplicate feature configurations together.
    :return: A :class:`DatasetSplit` preserving the original pandas types, indices, and columns.
    :raises ValueError: If feature and target rows are not aligned.
    """
    if len(dataset.X) != len(dataset.y) or not dataset.X.index.equals(dataset.y.index):
        raise ValueError("Dataset features and target must have aligned rows and indices")

    row_positions = np.arange(len(dataset.X))
    if group_aware:
        groups = pd.MultiIndex.from_frame(dataset.X).factorize(sort=False)[0]
        first_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        trainval_positions, test_positions = next(
            first_split.split(dataset.X, dataset.y, groups=groups)
        )

        second_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_relative, val_relative = next(
            second_split.split(
                dataset.X.iloc[trainval_positions],
                dataset.y.iloc[trainval_positions],
                groups=groups[trainval_positions],
            )
        )
        train_positions = trainval_positions[train_relative]
        val_positions = trainval_positions[val_relative]
    else:
        trainval_positions, test_positions = train_test_split(
            row_positions, test_size=0.2, random_state=seed
        )
        train_relative, val_relative = train_test_split(
            np.arange(len(trainval_positions)), test_size=0.25, random_state=seed
        )
        train_positions = trainval_positions[train_relative]
        val_positions = trainval_positions[val_relative]

    return DatasetSplit(
        X_train=dataset.X.iloc[train_positions],
        X_val=dataset.X.iloc[val_positions],
        X_test=dataset.X.iloc[test_positions],
        y_train=dataset.y.iloc[train_positions],
        y_val=dataset.y.iloc[val_positions],
        y_test=dataset.y.iloc[test_positions],
    )


def evaluate(
    datasets: Iterable[tuple[Dataset, list[str]]],
    models: dict[str, BaseEstimator],
    effect: FeatureEffect,
    properties: Iterable[EffectProperty],
    seed: int = 0,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every model on every dataset, then explain the requested features.

    Train and test R² are reported so you can argue whether a model under- or overfits. Train R² is
    computed on the data the model was actually fit on (``X_fit_``/``y_fit_`` if the model has them, as
    :class:`~iml_feature_effects.models.SubsampledRegressor` does). Effects are computed on the test set.

    :param datasets: Pairs of a dataset and the names of the features to explain on it.
    :param models: Unfitted models keyed by name, e.g. from :func:`iml_feature_effects.models.get_models`.
    :param effect: Feature effect method.
    :param properties: Properties computed for each effect curve.
    :param seed: Random seed for the train/test split.
    :param test_size: Fraction of each dataset held out.
    :return: ``scores`` with one row per (dataset, model, feature) and columns ``train_r2``, ``test_r2``
        and one per property; ``curves`` with one row per grid point and columns ``dataset``, ``model``,
        ``feature``, ``value`` and ``effect``.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    properties = list(properties)
    score_rows, curves = [], []
    for dataset, features in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=test_size, random_state=seed
        )
        for model_name, model in models.items():
            # Clone so that the same unfitted model can be reused across datasets.
            fitted = clone(model).fit(X_train, y_train)
            X_fit, y_fit = getattr(fitted, "X_fit_", X_train), getattr(fitted, "y_fit_", y_train)
            fit = {
                "dataset": dataset.name,
                "model": model_name,
                "train_r2": r2_score(y_fit, fitted.predict(X_fit)),
                "test_r2": r2_score(y_test, fitted.predict(X_test)),
            }
            for feature in features:
                curve = effect.explain(fitted, X_test, feature)
                score_rows.append({**fit, "feature": feature, **{p.name: p.score(curve) for p in properties}})
                curves.append(curve.assign(dataset=dataset.name, model=model_name, feature=feature))
    curves_df = pd.concat(curves, ignore_index=True)[["dataset", "model", "feature", "value", "effect"]]
    return pd.DataFrame(score_rows), curves_df
