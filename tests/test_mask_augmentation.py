"""Vérifie le masquage augmenté : mois entiers, jamais le test/target, reproductible par rng."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.mask_augmentation import apply_augmented_masking, select_gap_months

RATE_BY_PERIOD = [{"start_year": 2020, "end_year": 2020, "rate": 1.0}]  # tout masquer, déterministe
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _small_panel() -> pd.DataFrame:
    # 2 cellules x 3 mois, train + test mélangés, pour vérifier l'isolation train/test.
    times = pd.date_range("2020-01-01", periods=3, freq="MS")
    rows = []
    for lat in [0.5, 1.5]:
        for i, t in enumerate(times):
            rows.append(
                {
                    "lat": lat,
                    "lon": 0.5,
                    "time": t,
                    "TWS_t": 1.0 + i,
                    "TWS_t_masked": False,
                    "target": 9.0 + i,
                    "is_train": t < times[-1],  # dernier mois = test
                }
            )
    return pd.DataFrame(rows)


def test_select_gap_months_deterministic_with_rate_one():
    panel = _small_panel()
    months = panel.loc[panel["is_train"], "time"]
    gap_months = select_gap_months(months, RATE_BY_PERIOD, np.random.default_rng(0))
    assert gap_months == set(months.unique())


def test_apply_augmented_masking_never_touches_test_rows():
    panel = _small_panel()
    gap_months = set(panel["time"].unique())  # masque TOUS les mois, y compris celui de test
    masked = apply_augmented_masking(panel, gap_months)

    test_rows = masked.loc[~masked["is_train"]]
    assert not test_rows["TWS_t"].isna().any()
    assert not test_rows["TWS_t_masked"].any()


def test_apply_augmented_masking_masks_whole_month_for_all_cells():
    panel = _small_panel()
    target_month = panel["time"].iloc[0]
    masked = apply_augmented_masking(panel, {target_month})

    train_rows_that_month = masked.loc[masked["is_train"] & (masked["time"] == target_month)]
    assert len(train_rows_that_month) == 2  # les 2 cellules, pas une seule
    assert train_rows_that_month["TWS_t"].isna().all()
    assert train_rows_that_month["TWS_t_masked"].all()


def test_target_never_modified_by_masking():
    panel = _small_panel()
    masked = apply_augmented_masking(panel, set(panel["time"].unique()))
    pd.testing.assert_series_equal(panel["target"], masked["target"])


def test_same_rng_gives_same_gap_months():
    panel = _small_panel()
    months = panel.loc[panel["is_train"], "time"]
    rate = [{"start_year": 2020, "end_year": 2020, "rate": 0.5}]
    result_a = select_gap_months(months, rate, np.random.default_rng(42))
    result_b = select_gap_months(months, rate, np.random.default_rng(42))
    assert result_a == result_b


def test_different_rng_can_give_different_gap_months():
    panel = _small_panel()
    months = panel.loc[panel["is_train"], "time"]
    rate = [{"start_year": 2020, "end_year": 2020, "rate": 0.5}]
    results = {
        tuple(sorted(select_gap_months(months, rate, np.random.default_rng(seed))))
        for seed in range(10)
    }
    # avec 10 seeds differentes et un taux de 50%, on s'attend a plus d'un resultat distinct.
    assert len(results) > 1


@pytest.mark.skipif(not RAW_DIR.exists(), reason="données brutes non disponibles (DVC non tiré)")
def test_build_features_masking_path_differs_from_unmasked_and_is_reproducible():
    import yaml

    from src.features.pipeline import build_features

    features_config = yaml.safe_load(open("configs/features.yaml"))
    masking_config = features_config["masking"]

    train_unmasked, _, _ = build_features(RAW_DIR, features_config)
    train_a, _, _ = build_features(
        RAW_DIR, features_config, masking_config, np.random.default_rng(7)
    )
    train_b, _, _ = build_features(
        RAW_DIR, features_config, masking_config, np.random.default_rng(7)
    )

    # Le masquage change bien l'horizon par rapport au chemin non masqué (Phase 2 : toujours 1).
    assert (train_unmasked["months_since_last_observed_tws"] == 1).all()
    assert (train_a["months_since_last_observed_tws"] > 1).any()

    # Même rng -> résultat identique.
    pd.testing.assert_frame_equal(train_a, train_b)

    # target jamais modifié par le masquage.
    pd.testing.assert_series_equal(train_unmasked["target"], train_a["target"])
