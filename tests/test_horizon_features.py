"""Vérifie le calcul d'horizon : mécanique sur données synthétiques + régression sur données réelles."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.horizon_features import add_horizon_features
from src.features.io import build_cell_timeline, load_raw

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Distribution validée en Phase 1 (`reports/data_understanding.md`), à ne jamais régresser.
EXPECTED_TEST_HORIZON_DISTRIBUTION = {
    1: 94048,
    2: 62576,
    3: 46777,
    4: 31076,
    5: 15560,
    6: 15479,
    7: 15445,
}


def _one_cell_with_gaps() -> pd.DataFrame:
    # Reprend l'exemple chiffré discuté avec l'utilisateur : deux trous masqués (2015-09,
    # 2015-10), une ligne réelle au milieu (2015-11), un trou après (2015-12).
    times = pd.date_range("2015-08-01", periods=5, freq="MS")
    masked = [False, True, True, False, True]
    values = [0.30, None, None, 0.45, None]
    return pd.DataFrame(
        {
            "lat": 0.5,
            "lon": 0.5,
            "time": times,
            "TWS_t": values,
            "TWS_t_masked": masked,
        }
    )


def test_horizon_and_last_observed_on_synthetic_gaps():
    df = add_horizon_features(_one_cell_with_gaps())
    assert df["months_since_last_observed_tws"].tolist() == [1, 2, 3, 1, 2]
    assert df["last_observed_tws"].tolist() == [0.30, 0.30, 0.30, 0.45, 0.45]


def test_train_without_masking_is_degenerate_horizon_one():
    df = _one_cell_with_gaps()
    df["TWS_t_masked"] = False
    df["TWS_t"] = [0.30, 0.31, 0.32, 0.33, 0.34]
    result = add_horizon_features(df)
    assert (result["months_since_last_observed_tws"] == 1).all()
    assert result["last_observed_tws"].tolist() == df["TWS_t"].tolist()


@pytest.mark.skipif(not RAW_DIR.exists(), reason="données brutes non disponibles (DVC non tiré)")
def test_horizon_distribution_matches_phase1_reference():
    train, test = load_raw(RAW_DIR)
    panel = build_cell_timeline(train, test)
    panel = add_horizon_features(panel)

    test_rows = panel.loc[~panel["is_train"]]
    actual = test_rows["months_since_last_observed_tws"].value_counts().sort_index().to_dict()

    assert actual == EXPECTED_TEST_HORIZON_DISTRIBUTION
