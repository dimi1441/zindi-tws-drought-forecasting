"""Recalcul par époque des features dérivées de `TWS_t` (Phase 5, ANN + masquage augmenté).

Chaque fonction ci-dessous est l'analogue tensoriel exact d'une fonction pandas déjà validée du
pipeline Phase 2/3 — voir la docstring de chacune pour la correspondance précise. Testé en parité
contre ces fonctions pandas dans `tests/test_ann_dynamic_features.py` : c'est cette parité, pas
une relecture du code, qui prouve que le recalcul tensoriel respecte les mêmes règles anti-fuite
(brief §3.5) que le pipeline pandas existant.

Rien ici n'est mis en cache d'une époque à l'autre (même principe que `mask_augmentation.py` :
mode dynamique uniquement) — chaque appel à `compute_dynamic_features` avec un `rng` différent
refait tout le calcul depuis le `FeatureScaffold` statique.
"""

import numpy as np
import pandas as pd

from src.models.ann.feature_tensors import FeatureScaffold, gather_to_panel_order

TWS_LAG_STEPS = (1, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)


def _causal_windowed_stat(filled: np.ndarray, weight: np.ndarray, window: int) -> np.ndarray:
    """Somme/compte causal sur une fenêtre de `window` PAS (pas de mois calendaires — les mois
    totalement absents n'ont déjà aucune ligne dans le panel, exactement comme la version pandas
    `.rolling(window, min_periods=1)` sur une série groupée par cellule), incluant le pas courant.
    Retourne `(somme_fenetre, poids_fenetre)` — au niveau (n_cells, T_max).
    """
    n_cells, t_max = filled.shape
    cumsum = np.concatenate([np.zeros((n_cells, 1), dtype=np.float64), np.cumsum(filled, axis=1)], axis=1)
    cumweight = np.concatenate(
        [np.zeros((n_cells, 1), dtype=np.float64), np.cumsum(weight, axis=1)], axis=1
    )
    idx = np.arange(t_max)
    hi = idx + 1
    lo = np.clip(idx + 1 - window, 0, None)
    window_sum = cumsum[:, hi] - cumsum[:, lo]
    window_weight = cumweight[:, hi] - cumweight[:, lo]
    return window_sum, window_weight


def _causal_expanding_mean_and_count(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne/compte cumulés (fenêtre NON bornée, contrairement à `_causal_windowed_stat`),
    décalés d'un pas -- analogue de `trend.py` :
    `grouped[col].transform(lambda s: s.expanding().mean()/.count().shift(1))`, sans aucun
    regroupement par mois calendaire (à la différence de la climatologie).

    Le compte est explicitement forcé à NaN sur le tout premier pas de chaque cellule (position 0
    de l'axe temps, toujours une ligne réelle -- jamais du padding, `step_within_cell` commence à
    0) : c'est exactement ce que produit `shift(1)` sur une série pandas (rien à décaler avant la
    toute première ligne), et pas un simple 0 comme le donnerait un pur remplissage par des zéros.
    """
    n_cells, t_max = values.shape
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    cumsum = np.cumsum(filled, axis=1)
    cumcount = np.cumsum(valid.astype(np.float64), axis=1)

    shifted_cumsum = np.concatenate([np.zeros((n_cells, 1)), cumsum[:, :-1]], axis=1)
    shifted_cumcount = np.concatenate([np.zeros((n_cells, 1)), cumcount[:, :-1]], axis=1)
    shifted_cumcount[:, 0] = np.nan

    mean = np.where(
        shifted_cumcount > 0,
        shifted_cumsum / np.where(shifted_cumcount > 0, shifted_cumcount, 1.0),
        np.nan,
    )
    return mean.astype(np.float32), shifted_cumcount.astype(np.float32)


def apply_epoch_masking(scaffold: FeatureScaffold, gap_months: set[pd.Timestamp]) -> np.ndarray:
    """Masque `TWS_t` pour les positions train dont le mois est un trou tiré cette époque.

    Analogue tensoriel de `mask_augmentation.apply_augmented_masking` : ne touche jamais les
    positions test (masquage réel, déjà présent dans `scaffold.real_tws_t` sous forme de NaN),
    ni le padding. Contrairement à la version pandas, ne modifie rien sur place — retourne un
    nouveau tableau, `scaffold.real_tws_t` reste la vérité brute pour l'époque suivante.
    """
    gap_ym = {t.year * 12 + t.month for t in gap_months}
    is_gap = np.isin(scaffold.time_ym, np.array(sorted(gap_ym), dtype=np.int32))
    to_mask = scaffold.is_train & is_gap & scaffold.valid
    return np.where(to_mask, np.nan, scaffold.real_tws_t)


def compute_dynamic_features(
    scaffold: FeatureScaffold, gap_months: set[pd.Timestamp]
) -> dict[str, np.ndarray]:
    """Recalcule toutes les features dérivées de `TWS_t` pour un tirage de masquage donné.

    Retourne un dict de tableaux plats (len(scaffold.panel),), dans l'ordre de lignes du panel
    (mêmes valeurs NaN qu'obtiendrait le pipeline pandas — le remplissage par une valeur
    sentinelle + flag, nécessaire pour l'ANN, est fait par l'appelant juste avant l'entrée dans
    le réseau, jamais ici, pour que ce module reste directement comparable à la version pandas).
    """
    masked_tws = apply_epoch_masking(scaffold, gap_months)
    observed = scaffold.valid & ~np.isnan(masked_tws)
    n_cells, t_max = masked_tws.shape

    out: dict[str, np.ndarray] = {}
    out["TWS_t"] = masked_tws
    out["TWS_t_observed"] = observed.astype(np.float32)

    # --- Lags/diffs : décalage de `k` PAS (analogue de `grouped["TWS_t"].shift(k)`, qui décale
    # par ligne au sein du groupe cellule — déjà "par pas conservé", pas par mois calendaire,
    # puisque le panel ne contient aucune ligne pour un mois totalement absent). ---
    lag_arrays: dict[int, np.ndarray] = {}
    for k in TWS_LAG_STEPS:
        lag = np.full((n_cells, t_max), np.nan, dtype=np.float32)
        lag[:, k:] = masked_tws[:, :-k]
        lag_arrays[k] = lag
        out[f"TWS_t_lag{k}"] = lag
    out["TWS_t_diff1"] = masked_tws - lag_arrays[1]
    out["TWS_t_diff12"] = masked_tws - lag_arrays[12]

    # --- Tendance de long terme (ajout du 2026-09-07, demande utilisateur) : moyenne/compte
    # cumulés des diffs, analogue de `trend.py::add_trend_features`. ---
    for source_col, mean_col in (
        ("TWS_t_diff1", "TWS_t_diff1_expanding_mean"),
        ("TWS_t_diff12", "TWS_t_diff12_expanding_mean"),
    ):
        mean, count = _causal_expanding_mean_and_count(out[source_col])
        out[mean_col] = mean
        out[f"{mean_col}_count"] = count

    # --- Moyennes mobiles causales : cumsum/count par fenêtre glissante, INCLUANT le pas courant
    # (analogue de `.rolling(w, min_periods=1).mean()`). ---
    filled = np.where(observed, masked_tws, 0.0)
    weight = observed.astype(np.float64)
    for w in ROLLING_WINDOWS:
        window_sum, window_weight = _causal_windowed_stat(filled, weight, w)
        out[f"TWS_t_rollmean{w}"] = np.where(
            window_weight > 0, window_sum / np.where(window_weight > 0, window_weight, 1.0), np.nan
        ).astype(np.float32)

    # --- Climatologie causale : cumsum/count par bucket (cellule, mois calendaire), décalé d'un
    # cran pour exclure l'année courante (analogue de `expanding().mean().shift(1)`). ---
    bucket_index = scaffold.climatology_bucket_index  # (n_cells, 12, Y_max), -1 = case vide
    valid_bucket = bucket_index >= 0
    flat_filled = filled.reshape(-1)
    flat_weight = weight.reshape(-1)
    bucket_filled = np.where(valid_bucket, flat_filled[np.clip(bucket_index, 0, None)], 0.0)
    bucket_weight = np.where(valid_bucket, flat_weight[np.clip(bucket_index, 0, None)], 0.0)

    bucket_cumsum = np.cumsum(bucket_filled, axis=2)
    bucket_cumweight = np.cumsum(bucket_weight, axis=2)
    zeros_col = np.zeros((n_cells, 12, 1), dtype=np.float64)
    bucket_cumsum_shifted = np.concatenate([zeros_col, bucket_cumsum[:, :, :-1]], axis=2)
    bucket_cumweight_shifted = np.concatenate([zeros_col, bucket_cumweight[:, :, :-1]], axis=2)
    # Comme `_causal_expanding_mean_and_count` : le tout premier rang d'année de chaque bucket
    # (cellule, mois calendaire) doit être NaN (rien à décaler avant lui), pas 0.
    bucket_count_shifted_for_output = bucket_cumweight_shifted.copy()
    bucket_count_shifted_for_output[:, :, 0] = np.nan
    bucket_climatology = np.where(
        bucket_cumweight_shifted > 0,
        bucket_cumsum_shifted / np.where(bucket_cumweight_shifted > 0, bucket_cumweight_shifted, 1.0),
        np.nan,
    )

    flat_climatology = np.full(n_cells * t_max, np.nan, dtype=np.float32)
    flat_climatology[bucket_index[valid_bucket]] = bucket_climatology[valid_bucket]
    climatology_mean = flat_climatology.reshape(n_cells, t_max)
    out["TWS_t_climatology_mean"] = climatology_mean

    flat_climatology_count = np.full(n_cells * t_max, np.nan, dtype=np.float32)
    flat_climatology_count[bucket_index[valid_bucket]] = bucket_count_shifted_for_output[valid_bucket]
    out["TWS_t_climatology_count"] = flat_climatology_count.reshape(n_cells, t_max)
    out["TWS_t_seasonal_anomaly"] = masked_tws - climatology_mean

    # --- Horizon : dernier pas observé, en avant seulement (analogue de
    # `.where(~masked).groupby(cell).ffill()`, via un cummax causal sur l'indice de pas). ---
    step_idx = np.broadcast_to(np.arange(t_max), (n_cells, t_max))
    candidate = np.where(observed, step_idx, -1)
    last_observed_idx = np.maximum.accumulate(candidate, axis=1)
    ever_observed = last_observed_idx >= 0
    clipped_idx = np.clip(last_observed_idx, 0, None)
    last_observed_tws = np.where(
        ever_observed, np.take_along_axis(masked_tws, clipped_idx, axis=1), np.nan
    )
    last_observed_ym = np.take_along_axis(scaffold.time_ym, clipped_idx, axis=1)
    target_ym = scaffold.time_ym + 1
    months_since = np.where(ever_observed, (target_ym - last_observed_ym).astype(np.float32), np.nan)
    out["last_observed_tws"] = last_observed_tws.astype(np.float32)
    out["months_since_last_observed_tws"] = months_since.astype(np.float32)

    return {name: gather_to_panel_order(scaffold, arr) for name, arr in out.items()}
