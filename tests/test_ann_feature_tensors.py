"""Vérifie le scaffold statique (padding, `panel_row_index`, indépendance du masquage)."""

import numpy as np
import pandas as pd

from src.features.io import build_cell_timeline
from src.models.ann.feature_tensors import build_feature_scaffold_from_panel

CONFIG = {"lags": {"tws": [1, 3, 6, 12], "covariates": [1, 3]}, "rolling_windows": [3, 6, 12]}


def _make_cell_rows(lat: float, times: pd.DatetimeIndex, values: list[float], is_train: bool) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "lat": lat,
            "lon": 0.5,
            "time": times,
            "TWS_t": values,
            "SPEI_01_t": 0.1,
            "SPEI_03_t": 0.2,
            "SPEI_06_t": 0.3,
            "SPEI_12_t": 0.4,
            "SOIL_MOISTURE_t": 0.5,
            "month_sin": np.sin(2 * np.pi * times.month / 12),
            "month_cos": np.cos(2 * np.pi * times.month / 12),
        }
    )
    if is_train:
        df["target"] = df["TWS_t"] + 0.01
    else:
        df["TWS_t_masked"] = df["TWS_t"].isna()
    return df


def _two_cell_panel():
    # Cellule A : 6 mois train (mars absent -> trou calendaire réel) + 2 mois test.
    times_a_train = pd.date_range("2020-01-01", periods=7, freq="MS").drop(pd.Timestamp("2020-03-01"))
    times_a_test = pd.date_range("2020-08-01", periods=2, freq="MS")
    train_a = _make_cell_rows(0.5, times_a_train, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], True)
    test_a = _make_cell_rows(0.5, times_a_test, [np.nan, 1.7], False)

    # Cellule B : 5 mois train, aucun trou, aucune ligne de test.
    times_b_train = pd.date_range("2020-01-01", periods=5, freq="MS")
    train_b = _make_cell_rows(1.5, times_b_train, [100.0, 101.0, 102.0, 103.0, 104.0], True)

    train = pd.concat([train_a, train_b], ignore_index=True)
    test = test_a.reset_index(drop=True)
    return build_cell_timeline(train, test)


def test_row_index_covers_every_panel_row_exactly_once():
    panel = _two_cell_panel()
    scaffold = build_feature_scaffold_from_panel(panel, CONFIG)

    covered = scaffold.panel_row_index[scaffold.valid]
    assert sorted(covered.tolist()) == list(range(len(panel)))


def test_padding_marks_shorter_cell_correctly():
    panel = _two_cell_panel()
    scaffold = build_feature_scaffold_from_panel(panel, CONFIG)

    # Cellule A a 8 lignes (6 train + 2 test), cellule B seulement 5 -> B doit être paddée.
    assert scaffold.t_max == 8
    cell_lengths = scaffold.valid.sum(axis=1)
    assert sorted(cell_lengths.tolist()) == [5, 8]


def test_real_tws_t_matches_panel_including_test_masking():
    panel = _two_cell_panel()
    scaffold = build_feature_scaffold_from_panel(panel, CONFIG)

    gathered = np.full(len(panel), np.nan, dtype=np.float32)
    gathered[scaffold.panel_row_index[scaffold.valid]] = scaffold.real_tws_t[scaffold.valid]
    np.testing.assert_allclose(gathered, panel["TWS_t"].to_numpy(dtype=np.float32), equal_nan=True)


def test_is_train_matches_panel():
    panel = _two_cell_panel()
    scaffold = build_feature_scaffold_from_panel(panel, CONFIG)

    gathered = np.zeros(len(panel), dtype=bool)
    gathered[scaffold.panel_row_index[scaffold.valid]] = scaffold.is_train[scaffold.valid]
    np.testing.assert_array_equal(gathered, panel["is_train"].to_numpy())
