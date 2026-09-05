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


def _spatial_block_ids(df: pd.DataFrame, block_size_degrees: float) -> np.ndarray:
    block_lat = np.floor(df["lat"].to_numpy() / block_size_degrees).astype(int)
    block_lon = np.floor(df["lon"].to_numpy() / block_size_degrees).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(block_lat, block_lon)])


def spatial_splits(
    df: pd.DataFrame, n_splits: int = 5, block_size_degrees: float = 10.0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold par blocs spatiaux de `block_size_degrees` degrés.

    `lat`/`lon` servent uniquement de clé de groupe ici (jamais une feature modèle, conforme au
    brief §3.2). Un bloc entier reste toujours du même côté (train ou validation) pour un fold
    donné — proxy de "cellules du même bassin versant" en l'absence de vraie donnée hydrologique.
    """
    groups = _spatial_block_ids(df, block_size_degrees)

    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(df, groups=groups):
        train_mask = np.zeros(len(df), dtype=bool)
        val_mask = np.zeros(len(df), dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        yield train_mask, val_mask


def temporal_holdout(df: pd.DataFrame, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Découpe interne *one-shot* par temps (dernière tranche de mois = validation).

    Sert à caler l'early stopping d'un modèle sur un sous-ensemble de `df` lui-même jamais vu
    par le fold de validation externe — jamais le même `val_df` utilisé à la fois pour arrêter
    l'entraînement et pour rapporter la métrique finale (ça biaiserait l'estimation).
    """
    months = pd.DatetimeIndex(sorted(df["time"].unique()))
    cutoff_idx = max(1, int(len(months) * (1 - val_fraction)))
    cutoff = months[cutoff_idx]
    train_mask = (df["time"] < cutoff).to_numpy()
    val_mask = ~train_mask
    return train_mask, val_mask


def spatial_holdout(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    block_size_degrees: float = 10.0,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Découpe interne *one-shot* par blocs spatiaux, même principe que `temporal_holdout` mais
    pour le schéma spatial (tire au sort une fraction des blocs comme validation interne)."""
    if rng is None:
        rng = np.random.default_rng(0)
    block_id = _spatial_block_ids(df, block_size_degrees)
    distinct_blocks = np.unique(block_id)
    n_val_blocks = max(1, int(len(distinct_blocks) * val_fraction))
    val_blocks = rng.choice(distinct_blocks, size=n_val_blocks, replace=False)
    val_mask = np.isin(block_id, val_blocks)
    train_mask = ~val_mask
    return train_mask, val_mask
