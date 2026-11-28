"""Offline tests for dataset loaders."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from iml_feature_effects import data


def _concrete_response(features: pd.DataFrame, targets: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))


def test_load_concrete_strength_preprocesses_uci_data(monkeypatch):
    features = pd.DataFrame(np.arange(1030 * 8).reshape(1030, 8))
    targets = pd.DataFrame({"Concrete compressive strength": np.arange(1030)})
    fetch_calls = []

    def fake_fetch_ucirepo(*, id: int) -> SimpleNamespace:
        fetch_calls.append(id)
        return _concrete_response(features, targets)

    monkeypatch.setattr(data, "fetch_ucirepo", fake_fetch_ucirepo)

    result = data.load_concrete_strength()

    assert fetch_calls == [165]
    assert isinstance(result, data.Dataset)
    assert result.name == "concrete_strength"
    assert result.X.shape == (1030, 8)
    assert result.X.columns.tolist() == [
        "Cement",
        "Blast furnace slag",
        "Fly ash",
        "Water",
        "Superplasticizer",
        "Coarse aggregate",
        "Fine aggregate",
        "Age",
    ]
    assert result.X.dtypes.eq(float).all()
    assert result.y.shape == (1030,)
    assert result.y.name == "strength_mpa"
    assert result.y.dtype == float
    assert not result.X.isna().any().any()
    assert not result.y.isna().any()


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [(np.nan, "missing"), ("not numeric", "numeric")],
)
def test_load_concrete_strength_rejects_invalid_values(monkeypatch, invalid_value, message):
    features = pd.DataFrame(np.arange(1030 * 8, dtype=float).reshape(1030, 8), dtype=object)
    targets = pd.DataFrame({"strength": np.arange(1030, dtype=float)})
    features.iloc[0, 0] = invalid_value
    monkeypatch.setattr(
        data,
        "fetch_ucirepo",
        lambda *, id: _concrete_response(features, targets),
    )

    with pytest.raises(ValueError, match=message):
        data.load_concrete_strength()


def _california_response() -> SimpleNamespace:
    features = pd.DataFrame(
        np.arange(20640 * 8, dtype=float).reshape(20640, 8),
        columns=["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
    )
    target = pd.Series(np.arange(20640, dtype=float), name="MedHouseVal")
    return SimpleNamespace(data=features, target=target)


def _power_plant_response() -> SimpleNamespace:
    features = pd.DataFrame(
        np.arange(9568 * 4, dtype=float).reshape(9568, 4),
        columns=["AT", "V", "AP", "RH"],
    )
    targets = pd.DataFrame({"PE": np.arange(9568, dtype=float)})
    variables = pd.DataFrame(
        {
            "name": ["AT", "V", "AP", "RH", "PE"],
            "role": ["Feature", "Feature", "Feature", "Feature", "Target"],
            "type": ["Continuous"] * 5,
            "units": ["C", "cm Hg", "milibar", "%", "MW"],
        }
    )
    return SimpleNamespace(
        data=SimpleNamespace(features=features, targets=targets),
        variables=variables,
    )


def test_load_california_housing_preprocesses_data(monkeypatch):
    bunch = _california_response()
    monkeypatch.setattr(data, "fetch_california_housing", lambda *, as_frame: bunch)

    result = data.load_california_housing()

    assert result.name == "california_housing"
    assert result.X.shape == (20640, 8)
    assert result.X.columns.tolist() == bunch.data.columns.tolist()
    assert result.X.dtypes.eq(float).all()
    assert result.y.shape == (20640,)
    assert result.y.name == "median_house_value_100k_usd"
    assert result.y.dtype == float
    assert result.y.iloc[-1] == bunch.target.iloc[-1]
    assert result.X.index.equals(result.y.index)
    assert not result.X.isna().any().any()
    assert not result.y.isna().any()
    assert np.isfinite(result.X.to_numpy()).all()
    assert np.isfinite(result.y.to_numpy()).all()


def test_load_power_plant_preprocesses_data_and_uses_metadata_order(monkeypatch):
    response = _power_plant_response()
    monkeypatch.setattr(data, "fetch_ucirepo", lambda *, id: response)

    result = data.load_power_plant()

    assert result.name == "power_plant"
    assert result.X.shape == (9568, 4)
    assert result.X.columns.tolist() == [
        "Ambient temperature",
        "Exhaust vacuum",
        "Ambient pressure",
        "Relative humidity",
    ]
    assert result.X.dtypes.eq(float).all()
    assert result.y.shape == (9568,)
    assert result.y.name == "net_energy_output_mw"
    assert result.y.dtype == float
    assert result.y.iloc[-1] == response.data.targets.iloc[-1, 0]
    assert result.X.index.equals(result.y.index)
    assert not result.X.isna().any().any()
    assert not result.y.isna().any()
    assert np.isfinite(result.X.to_numpy()).all()
    assert np.isfinite(result.y.to_numpy()).all()


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [(np.nan, "missing"), (np.inf, "finite"), ("not numeric", "numeric")],
)
@pytest.mark.parametrize("loader_name", ["load_california_housing", "load_power_plant"])
def test_new_loaders_reject_invalid_feature_values(monkeypatch, invalid_value, message, loader_name):
    if loader_name == "load_california_housing":
        response = _california_response()
        response.data["MedInc"] = response.data["MedInc"].astype(object)
        response.data.loc[0, "MedInc"] = invalid_value
        monkeypatch.setattr(data, "fetch_california_housing", lambda *, as_frame: response)
    else:
        response = _power_plant_response()
        response.data.features["AT"] = response.data.features["AT"].astype(object)
        response.data.features.loc[0, "AT"] = invalid_value
        monkeypatch.setattr(data, "fetch_ucirepo", lambda *, id: response)

    with pytest.raises(ValueError, match=message):
        getattr(data, loader_name)()


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [(np.nan, "missing"), (np.inf, "finite"), ("not numeric", "numeric")],
)
@pytest.mark.parametrize("loader_name", ["load_california_housing", "load_power_plant"])
def test_new_loaders_reject_invalid_target_values(monkeypatch, invalid_value, message, loader_name):
    if loader_name == "load_california_housing":
        response = _california_response()
        response.target = response.target.astype(object)
        response.target.iloc[0] = invalid_value
        monkeypatch.setattr(data, "fetch_california_housing", lambda *, as_frame: response)
    else:
        response = _power_plant_response()
        response.data.targets["PE"] = response.data.targets["PE"].astype(object)
        response.data.targets.loc[0, "PE"] = invalid_value
        monkeypatch.setattr(data, "fetch_ucirepo", lambda *, id: response)

    with pytest.raises(ValueError, match=message):
        getattr(data, loader_name)()


@pytest.mark.parametrize("loader_name", ["load_california_housing", "load_power_plant"])
def test_new_loaders_reject_misaligned_feature_and_target_indices(monkeypatch, loader_name):
    if loader_name == "load_california_housing":
        response = _california_response()
        response.target.index = response.target.index + 1
        monkeypatch.setattr(data, "fetch_california_housing", lambda *, as_frame: response)
    else:
        response = _power_plant_response()
        response.data.targets.index = response.data.targets.index + 1
        monkeypatch.setattr(data, "fetch_ucirepo", lambda *, id: response)

    with pytest.raises(ValueError, match="indices must match"):
        getattr(data, loader_name)()


def test_load_power_plant_rejects_unexpected_metadata_order(monkeypatch):
    response = _power_plant_response()
    response.variables = response.variables.iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)
    monkeypatch.setattr(data, "fetch_ucirepo", lambda *, id: response)

    with pytest.raises(ValueError, match="feature order"):
        data.load_power_plant()
