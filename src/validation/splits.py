"""Schémas de validation croisée : temporel (fenêtre expansive) et spatial (GroupKFold par blocs).

Les deux schémas retournent des masques booléens au niveau des lignes, alignés positionnellement
sur `df` (attendu avec un index `RangeIndex` propre, comme produit par
`src.features.pipeline.build_features`).
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, TimeSeriesSplit


def temporal_splits(df: pd.DataFrame, n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Fenêtre expansive sur les mois réellement présents dans `df` (pas le calendrier brut —
    les mois absents n'ont pas besoin de traitement spécial, ils ne sont simplement pas dans la
    liste). Chaque fold entraîne sur une fenêtre qui grandit, valide sur la tranche de mois
    suivante — jamais de futur dans le train d'un fold donné.
    """
    months = pd.DatetimeIndex(sorted(df["time"].unique()))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_month_idx, val_month_idx in tscv.split(months):
        train_months = months[train_month_idx]
        val_months = months[val_month_idx]
        train_mask = df["time"].isin(train_months).to_numpy()
        val_mask = df["time"].isin(val_months).to_numpy()
        yield train_mask, val_mask


def spatial_splits(
    df: pd.DataFrame, n_splits: int = 5, block_size_degrees: float = 10.0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold par blocs spatiaux de `block_size_degrees` degrés.

    `lat`/`lon` servent uniquement de clé de groupe ici (jamais une feature modèle, conforme au
    brief §3.2). Un bloc entier reste toujours du même côté (train ou validation) pour un fold
    donné — proxy de "cellules du même bassin versant" en l'absence de vraie donnée hydrologique.
    """
    block_lat = np.floor(df["lat"].to_numpy() / block_size_degrees).astype(int)
    block_lon = np.floor(df["lon"].to_numpy() / block_size_degrees).astype(int)
    groups = np.array([f"{a}_{b}" for a, b in zip(block_lat, block_lon)])

    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(df, groups=groups):
        train_mask = np.zeros(len(df), dtype=bool)
        val_mask = np.zeros(len(df), dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        yield train_mask, val_mask
