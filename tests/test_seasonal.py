"""Vérifie que la climatologie n'utilise jamais l'année courante ni les trous (sans fuite)."""

import numpy as np
import pandas as pd
import pytest

from src.features.seasonal import add_climatology_features


def _one_cell_januaries(with_gap: bool = True) -> pd.DataFrame:
    # Une seule cellule, un seul mois calendaire (janvier), sur plusieurs années, avec un trou
    # en 2004 (aucune ligne) — reproduit l'exemple discuté avec l'utilisateur.
    years = [2002, 2003, 2005, 2006] if with_gap else [2002, 2003, 2004, 2005]
    values = [0.50, 0.60, 0.40, 0.55]
    rows = [
        {"lat": 0.5, "lon": 0.5, "time": pd.Timestamp(year=y, month=1, day=1), "TWS_t": v}
        for y, v in zip(years, values)
    ]
    return pd.DataFrame(rows)


def test_climatology_excludes_current_year():
    df = add_climatology_features(_one_cell_januaries())
    df = df.sort_values("time").reset_index(drop=True)

    assert np.isnan(df.loc[0, "TWS_t_climatology_mean"])  # 2002 : aucune année antérieure
    assert df.loc[1, "TWS_t_climatology_mean"] == 0.50  # 2003 : moyenne de {2002}
    assert df.loc[2, "TWS_t_climatology_mean"] == 0.55  # 2005 : moyenne de {2002, 2003}
    assert df.loc[3, "TWS_t_climatology_mean"] == pytest.approx(0.50)  # 2006 : {2002,2003,2005}


def test_missing_year_is_skipped_not_treated_as_zero():
    # 2004 est totalement absent (pas de ligne) : la climatologie de 2005 doit être la moyenne
    # de {2002, 2003} = 0.55, jamais une moyenne qui compterait 2004 comme 0 ou une pollution NaN.
    df = add_climatology_features(_one_cell_januaries(with_gap=True))
    df = df.sort_values("time").reset_index(drop=True)
    row_2005 = df[df["time"] == pd.Timestamp("2005-01-01")].iloc[0]
    assert row_2005["TWS_t_climatology_mean"] == 0.55


def test_masked_row_does_not_pollute_or_break_subsequent_climatology():
    df = _one_cell_januaries(with_gap=False)
    df.loc[df["time"] == pd.Timestamp("2004-01-01"), "TWS_t"] = np.nan  # ligne masquée
    df = add_climatology_features(df).sort_values("time").reset_index(drop=True)

    # La climatologie de 2005 doit ignorer le NaN de 2004 et ne moyenner que {2002, 2003}.
    row_2005 = df[df["time"] == pd.Timestamp("2005-01-01")].iloc[0]
    assert row_2005["TWS_t_climatology_mean"] == 0.55
    # L'anomalie de la ligne masquée elle-même est NaN (TWS_t - climatologie = NaN - x).
    row_2004 = df[df["time"] == pd.Timestamp("2004-01-01")].iloc[0]
    assert np.isnan(row_2004["TWS_t_seasonal_anomaly"])
