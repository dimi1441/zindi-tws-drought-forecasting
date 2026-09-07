"""Vérifie la tendance de long terme : moyenne cumulée causale des variations TWS_t, jamais
mélangée entre cellules, jamais polluée par une valeur manquante."""

import numpy as np
import pandas as pd
import pytest

from src.features.temporal_lags import add_lag_features
from src.features.trend import add_trend_features

CONFIG = {"lags": {"tws": [1, 3, 6, 12], "covariates": [1]}, "rolling_windows": [3]}


def _one_cell_monthly_series(values: list[float]) -> pd.DataFrame:
    times = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    return pd.DataFrame({"lat": 0.5, "lon": 0.5, "time": times, "TWS_t": values})


def test_diff1_expanding_mean_excludes_current_step():
    # Variations diff1 successives : 0.2, 0.1, 0.3, -0.1, ... -- vérifiées à la main ci-dessous.
    df = add_lag_features(_one_cell_monthly_series([1.0, 1.2, 1.3, 1.6, 1.5]), CONFIG)
    df = add_trend_features(df).sort_values("time").reset_index(drop=True)

    # diff1 : [NaN, 0.2, 0.1, 0.3, -0.1]
    assert np.isnan(df.loc[0, "TWS_t_diff1_expanding_mean"])  # rien avant la 1re ligne
    assert np.isnan(df.loc[1, "TWS_t_diff1_expanding_mean"])  # aucun diff1 valide avant elle
    assert df.loc[2, "TWS_t_diff1_expanding_mean"] == pytest.approx(0.2)  # moyenne de {0.2}
    assert df.loc[3, "TWS_t_diff1_expanding_mean"] == pytest.approx((0.2 + 0.1) / 2)
    assert df.loc[4, "TWS_t_diff1_expanding_mean"] == pytest.approx((0.2 + 0.1 + 0.3) / 3)


def test_diff1_expanding_mean_count_matches_valid_observations():
    # `.count()` reste bien défini même sur zéro élément valide (contrairement à `.mean()`,
    # indéfinie) : seule la toute première ligne du groupe reçoit NaN (rien avant elle à décaler),
    # la ligne suivante reçoit bel et bien 0 (aucun diff1 valide avant elle, mais "0" est une
    # réponse légitime, pas une absence de réponse).
    df = add_lag_features(_one_cell_monthly_series([1.0, 1.2, 1.3, 1.6, 1.5]), CONFIG)
    df = add_trend_features(df).sort_values("time").reset_index(drop=True)

    assert np.isnan(df.loc[0, "TWS_t_diff1_expanding_mean_count"])
    assert df.loc[1, "TWS_t_diff1_expanding_mean_count"] == 0
    assert df.loc[2, "TWS_t_diff1_expanding_mean_count"] == 1
    assert df.loc[3, "TWS_t_diff1_expanding_mean_count"] == 2
    assert df.loc[4, "TWS_t_diff1_expanding_mean_count"] == 3


def test_missing_diff_is_skipped_not_treated_as_zero():
    # Une ligne masquée (TWS_t = NaN) casse le diff1 des DEUX lignes qui la touchent (elle-même,
    # et la suivante qui la prend comme lag1) -- mais la moyenne cumulée doit ignorer ces NaN sans
    # jamais les compter comme des 0.
    df = _one_cell_monthly_series([1.0, 1.2, np.nan, 1.6, 1.5])
    df = add_lag_features(df, CONFIG)
    df = add_trend_features(df).sort_values("time").reset_index(drop=True)

    # diff1 réel : [NaN, 0.2, NaN (ligne masquée), NaN (lag1 = ligne masquée), -0.1]
    # -> seule la ligne 1 (0.2) est un diff1 valide avant la dernière ligne.
    last_row = df.iloc[-1]
    assert last_row["TWS_t_diff1_expanding_mean"] == pytest.approx(0.2)
    assert last_row["TWS_t_diff1_expanding_mean_count"] == 1


def test_trend_never_crosses_cells():
    times = pd.date_range("2020-01-01", periods=4, freq="MS")
    rows = []
    for lat, values in [(0.5, [0.10, 0.20, 0.40, 0.30]), (1.5, [100, 300, 200, 500])]:
        for t, v in zip(times, values):
            rows.append({"lat": lat, "lon": 0.5, "time": t, "TWS_t": v})
    df = pd.DataFrame(rows).sort_values(["lat", "lon", "time"]).reset_index(drop=True)
    df = add_lag_features(df, CONFIG)
    df = add_trend_features(df)

    cell_a = df[df["lat"] == 0.5].sort_values("time")
    cell_b = df[df["lat"] == 1.5].sort_values("time")
    # Échelles disjointes (0.1-0.4 vs 100-500) : une contamination entre cellules se verrait
    # immédiatement dans l'ordre de grandeur de la tendance moyenne.
    assert cell_a["TWS_t_diff1_expanding_mean"].dropna().abs().max() < 1
    assert cell_b["TWS_t_diff1_expanding_mean"].dropna().abs().min() > 1
