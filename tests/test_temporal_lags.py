"""Vérifie que les lags/rolling ne mélangent jamais deux cellules différentes."""

import pandas as pd
import pytest

from src.features.temporal_lags import add_lag_features

CONFIG = {"lags": {"tws": [1, 3], "covariates": [1]}, "rolling_windows": [3]}


def _two_cell_panel() -> pd.DataFrame:
    # Cellule A : 0.10, 0.20, 0.30, 0.40 ; cellule B : 100, 200, 300, 400 (échelle très différente
    # pour détecter immédiatement un mélange si les lags se contaminaient entre cellules).
    times = pd.date_range("2020-01-01", periods=4, freq="MS")
    rows = []
    for lat, values in [(0.5, [0.10, 0.20, 0.30, 0.40]), (1.5, [100, 200, 300, 400])]:
        for t, v in zip(times, values):
            rows.append({"lat": lat, "lon": 0.5, "time": t, "TWS_t": v, "SPEI_01_t": v})
    return pd.DataFrame(rows).sort_values(["lat", "lon", "time"]).reset_index(drop=True)


def test_lag1_never_crosses_cells():
    df = add_lag_features(_two_cell_panel(), CONFIG)

    cell_a = df[df["lat"] == 0.5].sort_values("time")
    cell_b = df[df["lat"] == 1.5].sort_values("time")

    assert cell_a["TWS_t_lag1"].dropna().tolist() == [0.10, 0.20, 0.30]
    assert cell_b["TWS_t_lag1"].dropna().tolist() == [100, 200, 300]
    # Aucune valeur de la cellule B ne doit apparaître dans les lags de la cellule A, et
    # inversement (les échelles sont volontairement disjointes : 0.1-0.4 vs 100-400).
    assert cell_a["TWS_t_lag1"].dropna().max() < 1
    assert cell_b["TWS_t_lag1"].dropna().min() >= 100


def test_rolling_mean_includes_current_point_not_future():
    df = add_lag_features(_two_cell_panel(), CONFIG)
    cell_a = df[df["lat"] == 0.5].sort_values("time").reset_index(drop=True)

    # rolling(3, min_periods=1) : ligne 0 -> {0.10}, ligne 1 -> {0.10,0.20}, ligne 2 -> {0.10,0.20,0.30}
    assert cell_a.loc[0, "TWS_t_rollmean3"] == pytest.approx(0.10)
    assert cell_a.loc[1, "TWS_t_rollmean3"] == pytest.approx((0.10 + 0.20) / 2)
    assert cell_a.loc[2, "TWS_t_rollmean3"] == pytest.approx((0.10 + 0.20 + 0.30) / 3)
    # La ligne 2 ne doit jamais inclure la valeur future de la ligne 3 (0.40).
    assert 0.40 not in [cell_a.loc[2, "TWS_t_rollmean3"]]


def test_covariate_lags_are_computed():
    df = add_lag_features(_two_cell_panel(), CONFIG)
    assert "SPEI_01_t_lag1" in df.columns
    cell_a = df[df["lat"] == 0.5].sort_values("time")
    assert cell_a["SPEI_01_t_lag1"].dropna().tolist() == [0.10, 0.20, 0.30]
