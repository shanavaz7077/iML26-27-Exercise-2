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
