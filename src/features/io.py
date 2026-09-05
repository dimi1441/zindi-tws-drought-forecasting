"""Chargement des données brutes et construction du panel cellule/temps combiné."""

from pathlib import Path

import pandas as pd


def load_raw(raw_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge Train.csv et Test.csv, harmonise la colonne d'identifiant.

    Train utilise `sample_id`, Test utilise `ID` — jamais harmonisé en Phase 1, corrigé ici
    pour que les deux dataframes partagent le même schéma de colonnes.
    """
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / "Train.csv", parse_dates=["time"])
    test = pd.read_csv(raw_dir / "Test.csv", parse_dates=["time"])

    if "sample_id" in train.columns and "ID" not in train.columns:
        train = train.rename(columns={"sample_id": "ID"})

    return train, test


def build_cell_timeline(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Concatène train et test en un panel long trié par cellule puis par temps.

    Les features temporelles (lags, rolling, climatologie, horizon, voisinage) doivent regarder
    l'historique réel d'une cellule, qui s'étend sur train ET test (mêmes 15 715 cellules).
    `TWS_t` masqué en test reste `NaN` (jamais rempli) ; un flag `is_train` permet de re-séparer
    les deux jeux à la fin du pipeline.
    """
    train = train.copy()
    test = test.copy()

    train["is_train"] = True
    test["is_train"] = False

    if "TWS_t_masked" not in train.columns:
        train["TWS_t_masked"] = False
    if "target" not in test.columns:
        test["target"] = float("nan")

    panel = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
    panel = panel.sort_values(["lat", "lon", "time"]).reset_index(drop=True)
    return panel
