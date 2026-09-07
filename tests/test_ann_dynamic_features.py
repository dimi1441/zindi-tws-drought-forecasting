"""Parité tensorielle vs pandas (Phase 5, ANN) : preuve directe que le recalcul par époque
respecte les mêmes règles anti-fuite que le pipeline pandas déjà validé (brief §3.5) -- c'est
cette parité, pas une relecture du code, qui fait foi. Un panel synthétique (2 cellules, un trou
calendaire réel, un tirage de masquage augmenté, des lignes de test réellement masquées) est
poussé à travers les deux pipelines et les résultats comparés colonne par colonne.
"""

import numpy as np
import pandas as pd

from src.features.horizon_features import add_horizon_features
from src.features.io import build_cell_timeline
from src.features.mask_augmentation import apply_augmented_masking
from src.features.seasonal import add_climatology_features
from src.features.temporal_lags import add_lag_features
from src.features.trend import add_trend_features
from src.models.ann.dynamic_features import compute_dynamic_features
from src.models.ann.feature_tensors import build_feature_scaffold_from_panel

CONFIG = {"lags": {"tws": [1, 3, 6, 12], "covariates": [1, 3]}, "rolling_windows": [3, 6, 12]}
GAP_MONTHS = {pd.Timestamp("2020-07-01")}  # tiré cette "époque" -- touche les 2 cellules à la fois


def _make_cell_rows(lat: float, times: pd.DatetimeIndex, values: list[float], is_train: bool) -> pd.DataFrame:
    rng = np.random.RandomState(int(lat * 10))
    df = pd.DataFrame(
        {
            "lat": lat,
            "lon": 0.5,
            "time": times,
            "TWS_t": values,
            "SPEI_01_t": rng.normal(0, 1, len(times)),
            "SPEI_03_t": rng.normal(0, 1, len(times)),
            "SPEI_06_t": rng.normal(0, 1, len(times)),
            "SPEI_12_t": rng.normal(0, 1, len(times)),
            "SOIL_MOISTURE_t": rng.normal(0, 1, len(times)),
            "month_sin": np.sin(2 * np.pi * times.month / 12),
            "month_cos": np.cos(2 * np.pi * times.month / 12),
        }
    )
    if is_train:
        df["target"] = np.asarray(values, dtype=float) + 0.01
    else:
        df["TWS_t_masked"] = df["TWS_t"].isna()
    return df


def _build_synthetic_panel() -> pd.DataFrame:
    # Cellule A : ~20 mois sur 2 années (mars 2020 absent -> vrai trou calendaire), pour que la
    # climatologie ait un vrai passé (même mois, année antérieure) à comparer. 3 derniers mois en
    # test, dont un réellement masqué (TWS_t = NaN, comme Test.csv).
    rng_a = np.random.RandomState(1)
    times_a_all = pd.date_range("2020-01-01", "2021-08-01", freq="MS").drop(pd.Timestamp("2020-03-01"))
    values_a_all = (1.0 + 0.02 * np.arange(len(times_a_all)) + rng_a.normal(0, 0.05, len(times_a_all))).tolist()
    train_a = _make_cell_rows(0.5, times_a_all[:-3], values_a_all[:-3], True)
    test_a_values = values_a_all[-3:]
    test_a_values[-1] = np.nan  # dernière ligne de test réellement masquée
    test_a = _make_cell_rows(0.5, times_a_all[-3:], test_a_values, False)

    # Cellule B : 14 mois consécutifs (aucun trou), échelle disjointe (100+) pour détecter toute
    # contamination entre cellules ; 2 derniers mois en test, le premier masqué.
    rng_b = np.random.RandomState(2)
    times_b_all = pd.date_range("2020-01-01", periods=14, freq="MS")
    values_b_all = (100.0 + 0.5 * np.arange(len(times_b_all)) + rng_b.normal(0, 1, len(times_b_all))).tolist()
    train_b = _make_cell_rows(1.5, times_b_all[:-2], values_b_all[:-2], True)
    test_b_values = values_b_all[-2:]
    test_b_values[0] = np.nan
    test_b = _make_cell_rows(1.5, times_b_all[-2:], test_b_values, False)

    train = pd.concat([train_a, train_b], ignore_index=True)
    test = pd.concat([test_a, test_b], ignore_index=True)
    return build_cell_timeline(train, test)


def _pandas_ground_truth(panel: pd.DataFrame) -> pd.DataFrame:
    df = apply_augmented_masking(panel, GAP_MONTHS)
    df = add_lag_features(df, CONFIG)
    df = add_trend_features(df)
    df = add_climatology_features(df)
    df = add_horizon_features(df)
    return df


def _tensor_result(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    scaffold = build_feature_scaffold_from_panel(panel, CONFIG)
    return compute_dynamic_features(scaffold, GAP_MONTHS)


PARITY_COLUMNS = [
    "TWS_t",
    "TWS_t_lag1",
    "TWS_t_lag3",
    "TWS_t_lag6",
    "TWS_t_lag12",
    "TWS_t_diff1",
    "TWS_t_diff12",
    "TWS_t_rollmean3",
    "TWS_t_rollmean6",
    "TWS_t_rollmean12",
    "TWS_t_diff1_expanding_mean",
    "TWS_t_diff1_expanding_mean_count",
    "TWS_t_diff12_expanding_mean",
    "TWS_t_diff12_expanding_mean_count",
    "TWS_t_climatology_mean",
    "TWS_t_climatology_count",
    "TWS_t_seasonal_anomaly",
    "last_observed_tws",
    "months_since_last_observed_tws",
]


def test_tensor_recompute_matches_pandas_pipeline_column_by_column():
    panel = _build_synthetic_panel()
    expected = _pandas_ground_truth(panel)
    actual = _tensor_result(panel)

    for col in PARITY_COLUMNS:
        np.testing.assert_allclose(
            actual[col],
            expected[col].to_numpy(dtype=np.float32),
            equal_nan=True,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"colonne {col} ne correspond pas au pipeline pandas",
        )


def test_augmented_masking_never_touches_test_rows():
    panel = _build_synthetic_panel()
    actual = _tensor_result(panel)
    is_train = panel["is_train"].to_numpy()
    real_test_tws = panel.loc[~is_train, "TWS_t"].to_numpy(dtype=np.float32)
    tensor_test_tws = actual["TWS_t"][~is_train]
    np.testing.assert_allclose(tensor_test_tws, real_test_tws, equal_nan=True)


def test_perturbing_a_later_month_does_not_change_earlier_rows_features():
    # Test de causalité au niveau des features (le seul point de risque pour un modèle sans
    # mémoire) : la SEULE différence entre les deux tirages ci-dessous est un mois TARDIF masqué
    # ou non -- ça ne doit rien changer aux features calculées pour les lignes ANTÉRIEURES à ce
    # mois, pour aucune cellule.
    panel = _build_synthetic_panel()
    cutoff = pd.Timestamp("2021-06-01")
    scaffold = build_feature_scaffold_from_panel(panel, CONFIG)

    baseline = compute_dynamic_features(scaffold, set())  # rien de masqué
    perturbed = compute_dynamic_features(scaffold, {cutoff})  # un seul mois tardif masqué

    earlier_rows = (panel["time"] < cutoff).to_numpy()
    for col in PARITY_COLUMNS:
        np.testing.assert_allclose(
            baseline[col][earlier_rows],
            perturbed[col][earlier_rows],
            equal_nan=True,
            err_msg=f"colonne {col} : une ligne antérieure a changé à cause d'un masquage futur",
        )
