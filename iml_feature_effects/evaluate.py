"""Fit every model on every dataset, quantify its fit and compute feature effects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from iml_feature_effects.data import Dataset
from iml_feature_effects.effects import EffectProperty, FeatureEffect
from iml_feature_effects.models import get_tree_depth_candidates


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


def evaluate_tree_depth_sweep(dataset: Dataset, split: DatasetSplit) -> pd.DataFrame:
    """Fit decision trees on training data and score them on train and validation only.

    :param dataset: Dataset being evaluated; used to label result rows.
    :param split: Existing train/validation/test partitions. The test partition is not accessed.
    :return: One row per maximum-depth candidate, including train and validation R².
    """
    rows = []
    for max_depth, model in get_tree_depth_candidates().items():
        model.fit(split.X_train, split.y_train)
        train_r2 = r2_score(split.y_train, model.predict(split.X_train))
        val_r2 = r2_score(split.y_val, model.predict(split.X_val))
        rows.append(
            {
                "dataset": dataset.name,
                "max_depth": "unlimited" if max_depth is None else max_depth,
                "actual_depth": model.get_depth(),
                "leaf_count": model.get_n_leaves(),
                "train_samples": len(split.X_train),
                "val_samples": len(split.X_val),
                "train_r2": train_r2,
                "val_r2": val_r2,
                "train_val_gap": train_r2 - val_r2,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "max_depth",
            "actual_depth",
            "leaf_count",
            "train_samples",
            "val_samples",
            "train_r2",
            "val_r2",
            "train_val_gap",
        ],
    )


def select_tree_representatives(sweep: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """Select underfit, well-fit, and overfit depths using train/validation sweep metrics.

    Underfit is the simplest depth-1 candidate and is a relative label. Well-fit is the
    shallowest candidate within ``tolerance`` absolute validation R² of the best observed value.
    The tolerance is exploratory and practical, not a statistical guarantee. Overfit is the
    eligible deeper candidate with the largest train/validation R² gap; ties select the shallower
    depth. If no deeper candidate has higher training R² and lower validation R² than the
    well-fit candidate, the overfit selection is explicitly marked ``not_selected``.

    :param sweep: Output of :func:`evaluate_tree_depth_sweep`, with no test metrics required.
    :param tolerance: Allowed validation R² distance from the best candidate.
    :return: Three categorized selection rows per dataset, including an explicit missing overfit
        selection when no candidate meets the criteria.
    """
    required = {"dataset", "max_depth", "actual_depth", "leaf_count", "train_r2", "val_r2", "train_val_gap"}
    missing_columns = required.difference(sweep.columns)
    if missing_columns:
        raise ValueError(f"Sweep table is missing required columns: {sorted(missing_columns)}")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    def depth_order(depth: object) -> float:
        if depth is None or depth == "unlimited":
            return float("inf")
        if isinstance(depth, (int, np.integer)) and not isinstance(depth, bool):
            return float(depth)
        raise ValueError(f"Unrecognized max_depth value: {depth!r}")

    selected_rows = []
    selection_rules = {
        "underfit": "max_depth=1; simplest candidate, interpreted as relative underfitting",
        "well_fit": (
            f"shallowest candidate within {tolerance:.2f} absolute validation R² "
            "of the best; exploratory tolerance, not a statistical guarantee"
        ),
        "overfit": (
            "largest train-validation R² gap among deeper candidates with higher train R² "
            "and lower validation R² than the well-fit candidate; ties select the shallower depth"
        ),
    }

    for dataset_name, candidates in sweep.groupby("dataset", sort=False):
        if candidates["max_depth"].duplicated().any():
            raise ValueError(f"Sweep contains duplicate depth candidates for {dataset_name}")

        underfit_rows = candidates.loc[candidates["max_depth"].eq(1)]
        if underfit_rows.empty:
            raise ValueError(f"Sweep has no max_depth=1 candidate for {dataset_name}")
        underfit = underfit_rows.iloc[0]

        best_validation = candidates["val_r2"].max()
        within_tolerance = candidates.loc[(best_validation - candidates["val_r2"]).abs() <= tolerance]
        well_fit = within_tolerance.iloc[
            np.argmin([depth_order(depth) for depth in within_tolerance["max_depth"]])
        ]

        selected_rows.extend(
            [
                {
                    **underfit.to_dict(),
                    "dataset": dataset_name,
                    "fit_category": "underfit",
                    "selection_rule": selection_rules["underfit"],
                },
                {
                    **well_fit.to_dict(),
                    "dataset": dataset_name,
                    "fit_category": "well_fit",
                    "selection_rule": selection_rules["well_fit"],
                },
            ]
        )

        overfit_candidates = candidates.loc[
            candidates["max_depth"].map(depth_order).gt(depth_order(well_fit["max_depth"]))
            & candidates["train_r2"].gt(well_fit["train_r2"])
            & candidates["val_r2"].lt(well_fit["val_r2"])
        ]
        if overfit_candidates.empty:
            selected_rows.append(
                {
                    "dataset": dataset_name,
                    "max_depth": "not_selected",
                    "actual_depth": pd.NA,
                    "leaf_count": pd.NA,
                    "train_r2": pd.NA,
                    "val_r2": pd.NA,
                    "train_val_gap": pd.NA,
                    "fit_category": "overfit",
                    "selection_rule": (
                        "No deeper candidate has both higher train R² and lower validation R² "
                        "than the well-fit candidate"
                    ),
                }
            )
        else:
            overfit_candidates = overfit_candidates.assign(
                _depth_order=overfit_candidates["max_depth"].map(depth_order)
            )
            overfit = overfit_candidates.sort_values(
                ["train_val_gap", "_depth_order"],
                ascending=[False, True],
                kind="stable",
            ).iloc[0]
            selected_rows.append(
                {
                    **overfit.drop(labels="_depth_order").to_dict(),
                    "dataset": dataset_name,
                    "fit_category": "overfit",
                    "selection_rule": selection_rules["overfit"],
                }
            )

    columns = [
        "dataset",
        "fit_category",
        "max_depth",
        "actual_depth",
        "leaf_count",
        "train_r2",
        "val_r2",
        "train_val_gap",
        "selection_rule",
    ]
    return pd.DataFrame(selected_rows, columns=columns)


def evaluate_selected_tree_models(
    dataset: Dataset,
    split: DatasetSplit,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    """Fit selected trees on training data and evaluate each once on the test partition."""
    dataset_selections = selections.loc[selections["dataset"].eq(dataset.name)]
    selected = dataset_selections.loc[dataset_selections["max_depth"].ne("not_selected")]
    candidates = get_tree_depth_candidates()
    rows = []

    for selection in selected.itertuples(index=False):
        max_depth = None if selection.max_depth == "unlimited" else int(selection.max_depth)
        model = clone(candidates[max_depth])
        model.fit(split.X_train, split.y_train)

        train_predictions = model.predict(split.X_train)
        val_predictions = model.predict(split.X_val)
        test_predictions = model.predict(split.X_test)
        rows.append(
            {
                "dataset": dataset.name,
                "fit_category": selection.fit_category,
                "max_depth": selection.max_depth,
                "selection_rule": selection.selection_rule,
                "actual_depth": model.get_depth(),
                "leaf_count": model.get_n_leaves(),
                "train_samples": len(split.X_train),
                "val_samples": len(split.X_val),
                "test_samples": len(split.X_test),
                "train_r2": r2_score(split.y_train, train_predictions),
                "val_r2": r2_score(split.y_val, val_predictions),
                "test_r2": r2_score(split.y_test, test_predictions),
                "test_mae": mean_absolute_error(split.y_test, test_predictions),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "fit_category",
            "max_depth",
            "selection_rule",
            "actual_depth",
            "leaf_count",
            "train_samples",
            "val_samples",
            "test_samples",
            "train_r2",
            "val_r2",
            "test_r2",
            "test_mae",
        ],
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
