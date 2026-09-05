"""Vérifie les schémas de validation : pas de futur dans le train temporel, blocs spatiaux intacts."""

import numpy as np
import pandas as pd

from src.validation.splits import spatial_splits, temporal_splits


def _synthetic_temporal_df(n_months: int = 24) -> pd.DataFrame:
    times = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for lat in [0.5, 1.5, 2.5]:
        for t in times:
            rows.append({"lat": lat, "lon": 0.5, "time": t, "target": 1.0})
    return pd.DataFrame(rows).reset_index(drop=True)


def test_temporal_splits_never_put_future_in_train():
    df = _synthetic_temporal_df()
    for train_mask, val_mask in temporal_splits(df, n_splits=4):
        max_train_time = df.loc[train_mask, "time"].max()
        min_val_time = df.loc[val_mask, "time"].min()
        assert max_train_time < min_val_time


def test_temporal_splits_train_and_val_disjoint():
    df = _synthetic_temporal_df()
    for train_mask, val_mask in temporal_splits(df, n_splits=4):
        assert not np.any(train_mask & val_mask)


def test_temporal_splits_train_window_expands():
    df = _synthetic_temporal_df()
    train_sizes = [train_mask.sum() for train_mask, _ in temporal_splits(df, n_splits=4)]
    assert train_sizes == sorted(train_sizes)  # strictement croissant, fenêtre qui grandit


def _synthetic_spatial_df(n_cells_per_block: int = 4) -> pd.DataFrame:
    rows = []
    # 2 blocs distincts de 10 degres (0-10 et 20-30), plusieurs cellules par bloc.
    for block_lat in [5.0, 25.0]:
        for i in range(n_cells_per_block):
            rows.append({"lat": block_lat, "lon": 5.0 + i, "time": pd.Timestamp("2020-01-01")})
    return pd.DataFrame(rows).reset_index(drop=True)


def test_spatial_splits_never_split_a_block_across_train_and_val():
    df = _synthetic_spatial_df()
    block_id = (df["lat"] // 10).astype(int).astype(str) + "_" + (df["lon"] // 10).astype(int).astype(str)
    for train_mask, val_mask in spatial_splits(df, n_splits=2, block_size_degrees=10):
        train_blocks = set(block_id[train_mask])
        val_blocks = set(block_id[val_mask])
        assert not (train_blocks & val_blocks)


def test_spatial_splits_train_and_val_disjoint_and_cover_everything():
    df = _synthetic_spatial_df()
    for train_mask, val_mask in spatial_splits(df, n_splits=2, block_size_degrees=10):
        assert not np.any(train_mask & val_mask)
        assert np.all(train_mask | val_mask)
