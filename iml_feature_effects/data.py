"""Dataset loading.

Every dataset is returned as a :class:`Dataset` with numeric, imputed features so that all
models in :mod:`iml_feature_effects.models` can be fit on it directly. Add further loaders
(e.g. other OpenML or Kaggle datasets) following the pattern of :func:`load_bike_sharing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from ucimlrepo import fetch_ucirepo

# OpenML downloads are cached here, relative to the current working directory.
DATA_HOME = Path("data")


@dataclass(frozen=True)
class Dataset:
    """A preprocessed tabular regression dataset.

    :param name: Human-readable dataset name, used in result tables and plots.
    :param X: Numeric feature matrix without missing values.
    :param y: Target vector.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def load_bike_sharing(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the hourly Bike Sharing Demand dataset from OpenML (data id 42712).

    The target is the number of rented bikes per hour. The download is cached in ``data_home``, so
    network access is only needed on the first call.

    :param data_home: Cache directory for the download, relative paths are resolved against the
        current working directory.
    :return: The preprocessed Bike Sharing dataset.
    :rtype: Dataset
    """
    bunch = fetch_openml(data_id=42712, as_frame=True, parser="liac-arff", data_home=data_home)
    # This OpenML version already excludes ``casual`` and ``registered``, which sum to the target.
    X = pd.get_dummies(bunch.data, drop_first=True, dtype=float)
    y = bunch.target.astype(float).rename("count")
    return Dataset(name="bike_sharing", X=X.astype(float), y=y)


def load_concrete_strength() -> Dataset:
    """Load the UCI Concrete Compressive Strength dataset (UCI id 165).

    :return: The preprocessed Concrete Compressive Strength dataset.
    :rtype: Dataset
    """
    dataset = fetch_ucirepo(id=165)
    features = dataset.data.features
    targets = dataset.data.targets

    if features.shape != (1030, 8):
        raise ValueError(f"Expected 1030 rows and 8 features, got {features.shape}")
    if targets.shape != (1030, 1):
        raise ValueError(f"Expected one target column with 1030 rows, got {targets.shape}")

    try:
        X = features.apply(pd.to_numeric, errors="raise").astype(float)
        y = pd.to_numeric(targets.iloc[:, 0], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Concrete strength features and target must be numeric") from error

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Concrete strength data must not contain missing values")

    X.columns = [
        "Cement",
        "Blast furnace slag",
        "Fly ash",
        "Water",
        "Superplasticizer",
        "Coarse aggregate",
        "Fine aggregate",
        "Age",
    ]
    y = y.rename("strength_mpa")
    return Dataset(name="concrete_strength", X=X, y=y)


def load_california_housing() -> Dataset:
    """Load the California Housing dataset from scikit-learn.

    The target remains in the source units of $100,000.

    :return: The California Housing regression dataset.
    :rtype: Dataset
    """
    bunch = fetch_california_housing(as_frame=True)
    features = bunch.data
    target = bunch.target

    if features.shape != (20640, 8):
        raise ValueError(f"Expected 20640 rows and 8 features, got {features.shape}")
    if target.shape != (20640,):
        raise ValueError(f"Expected a target vector with 20640 values, got {target.shape}")
    if not features.index.equals(target.index):
        raise ValueError("California Housing features and target indices must match")

    try:
        X = features.apply(pd.to_numeric, errors="raise").astype(float)
        y = pd.to_numeric(target, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("California Housing features and target must be numeric") from error

    if X.isna().any().any() or y.isna().any():
        raise ValueError("California Housing data must not contain missing values")
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("California Housing data must contain only finite values")

    y = y.rename("median_house_value_100k_usd")
    return Dataset(name="california_housing", X=X, y=y)


def load_power_plant() -> Dataset:
    """Load the UCI Combined Cycle Power Plant dataset (UCI id 294).

    :return: The Power Plant regression dataset.
    :rtype: Dataset
    """
    dataset = fetch_ucirepo(id=294)
    features = dataset.data.features
    targets = dataset.data.targets

    if features.shape != (9568, 4):
        raise ValueError(f"Expected 9568 rows and 4 features, got {features.shape}")
    if targets.shape != (9568, 1):
        raise ValueError(f"Expected one target column with 9568 rows, got {targets.shape}")
    if not features.index.equals(targets.index):
        raise ValueError("Power Plant features and target indices must match")

    variables = dataset.variables
    feature_metadata = variables.loc[variables["role"].eq("Feature")]
    target_metadata = variables.loc[variables["role"].eq("Target")]
    source_feature_names = ["AT", "V", "AP", "RH"]
    if feature_metadata["name"].tolist() != source_feature_names:
        raise ValueError("Power Plant feature order does not match UCI metadata")
    if features.columns.tolist() != source_feature_names:
        raise ValueError("Power Plant feature columns do not match UCI metadata")
    if not feature_metadata["type"].eq("Continuous").all():
        raise ValueError("Power Plant input variables must be continuous")
    if len(target_metadata) != 1:
        raise ValueError("Power Plant must provide one continuous PE target")
    target_variable = target_metadata.iloc[0]
    if target_variable["name"] != "PE" or target_variable["type"] != "Continuous":
        raise ValueError("Power Plant must provide one continuous PE target")
    if target_variable["units"] != "MW" or targets.columns.tolist() != ["PE"]:
        raise ValueError("Power Plant target column and units do not match UCI metadata")

    try:
        X = features.apply(pd.to_numeric, errors="raise").astype(float)
        y = pd.to_numeric(targets.iloc[:, 0], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Power Plant features and target must be numeric") from error

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Power Plant data must not contain missing values")
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Power Plant data must contain only finite values")

    X.columns = [
        "Ambient temperature",
        "Exhaust vacuum",
        "Ambient pressure",
        "Relative humidity",
    ]
    y = y.rename("net_energy_output_mw")
    return Dataset(name="power_plant", X=X, y=y)
