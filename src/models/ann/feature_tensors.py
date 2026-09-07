"""Scaffold statique (Phase 5, ANN + masquage par époque) : construit une vue tabulaire par
cellule, indépendante de tout tirage de masquage, à partir de laquelle `dynamic_features.py`
recalcule les features dérivées de `TWS_t` à chaque époque avec de simples opérations
vectorisées (jamais un `groupby` pandas, trop lent à refaire 50-150+ fois).

Toute la mécanique repose sur un doublet d'index précalculés une seule fois :

- `panel_row_index` (n_cells, T_max) : pour chaque position valide du scaffold, l'indice de la
  ligne correspondante dans `panel` (le panel combiné train+test de `io.build_cell_timeline`,
  trié (lat, lon, time)) ; -1 pour les positions de padding. Permet de re-disperser n'importe
  quel tableau (n_cells, T_max) recalculé vers l'ordre de lignes plat que `pipeline.build_features`
  produit déjà (`panel.loc[panel["is_train"]]` / `~...`), sans jamais avoir besoin de reconstruire
  un index à la main plus tard.
- `climatology_bucket_index` (n_cells, 12, Y_max) : la structure de regroupement (cellule, mois
  calendaire, rang de l'année dans ce mois) ne dépend PAS du masquage (elle ne dépend que du
  calendrier réel) — seule la valeur `TWS_t` qu'on y disperse change à chaque tirage. Précalculer
  cette structure une seule fois évite de refaire un `groupby(["lat","lon", mois])` à chaque
  époque pour la climatologie (voir `dynamic_features.py`).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.io import build_cell_timeline, load_raw
from src.features.target_month_encoding import add_target_month_encoding
from src.features.temporal_lags import add_lag_features

# Colonnes jamais masquées (seul TWS_t peut l'être) : leurs valeurs brutes et leurs lags sont
# donc statiques, calculés une seule fois ici plutôt qu'à chaque époque.
STATIC_COVARIATE_COLUMNS = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
STATIC_CALENDAR_COLUMNS = ["month_sin", "month_cos", "target_month_sin", "target_month_cos"]


@dataclass
class FeatureScaffold:
    """Vue (n_cells, T_max) du panel combiné train+test, indépendante du masquage.

    `panel` est conservé tel quel : son ordre de lignes est celui que `train_out`/`test_out`
    doivent respecter en aval, exactement comme `pipeline.build_features` sépare déjà
    `panel.loc[panel["is_train"]]` / `~...` après reset_index(drop=True).
    """

    panel: pd.DataFrame
    n_cells: int
    t_max: int
    panel_row_index: np.ndarray  # (n_cells, T_max) int64, -1 = padding
    valid: np.ndarray  # (n_cells, T_max) bool
    is_train: np.ndarray  # (n_cells, T_max) bool, sans signification hors `valid`
    real_tws_t: np.ndarray  # (n_cells, T_max) float32, NaN = non observé (masqué ou padding)
    target: np.ndarray  # (n_cells, T_max) float32, NaN pour le test / le padding
    calendar_month: np.ndarray  # (n_cells, T_max) int16, 1-12, sans signification hors `valid`
    time_ym: np.ndarray  # (n_cells, T_max) int32, année*12+mois — identifie un mois calendaire
    # précis (contrairement à `calendar_month`, qui se répète chaque année) ; sert à tester
    # l'appartenance d'une position à `gap_months` (des `pd.Timestamp` précis, pas juste "juin").
    static_features: dict[str, np.ndarray]  # chacune (n_cells, T_max) float32
    climatology_bucket_index: np.ndarray  # (n_cells, 12, Y_max) int64, -1 = case vide


def gather_to_panel_order(scaffold: "FeatureScaffold", array2d: np.ndarray) -> np.ndarray:
    """Disperse un tableau (n_cells, T_max) vers l'ordre de lignes plat de `scaffold.panel`.

    Partagé entre les features statiques (calcul unique) et dynamiques (`dynamic_features.py`,
    un appel par époque) -- même mécanique, seule la source du tableau diffère.
    """
    flat = np.full(len(scaffold.panel), np.nan, dtype=np.float32)
    valid = scaffold.valid
    flat[scaffold.panel_row_index[valid]] = array2d[valid]
    return flat


def gather_static_features(scaffold: "FeatureScaffold") -> dict[str, np.ndarray]:
    """Version aplatie (ordre de lignes du panel) de `scaffold.static_features`, calculée une
    seule fois (les features statiques ne dépendent d'aucun tirage de masquage)."""
    return {name: gather_to_panel_order(scaffold, arr) for name, arr in scaffold.static_features.items()}


def _scatter(
    values: np.ndarray, cell_codes: np.ndarray, step: np.ndarray, shape: tuple[int, int], fill=np.nan
) -> np.ndarray:
    out = np.full(shape, fill, dtype=np.float32)
    out[cell_codes, step] = values.astype(np.float32)
    return out


def build_feature_scaffold(raw_dir: Path | str, features_config: dict) -> FeatureScaffold:
    """Construit le `FeatureScaffold` à partir des CSV bruts (calcul unique, jamais refait par
    époque). Fine wrapper autour de `build_feature_scaffold_from_panel` -- voir celle-ci pour la
    logique réelle, testée directement sur un panel synthétique dans
    `tests/test_ann_feature_tensors.py` (même convention que le reste du projet : pas de
    round-trip CSV dans les tests unitaires).
    """
    train, test = load_raw(raw_dir)
    panel = build_cell_timeline(train, test)
    return build_feature_scaffold_from_panel(panel, features_config)


def build_feature_scaffold_from_panel(panel: pd.DataFrame, features_config: dict) -> FeatureScaffold:
    """Construit le `FeatureScaffold` à partir d'un panel déjà construit par
    `io.build_cell_timeline` (trié (lat, lon, time), `is_train`/`target`/`TWS_t_masked` déjà
    posés). Séparé de `build_feature_scaffold` pour rester testable sur un panel synthétique.
    """
    panel = add_target_month_encoding(panel)
    # Réutilisé uniquement pour les lags de covariables (jamais masquées) : les colonnes
    # TWS_t_lag*/diff*/rollmean* de ce calcul sont ignorées, elles seront recalculées par époque
    # à partir de la vue masquée (voir dynamic_features.py).
    panel_with_covariate_lags = add_lag_features(panel, features_config)

    cell_key = panel["lat"].astype(str) + "_" + panel["lon"].astype(str)
    cell_codes, _ = pd.factorize(cell_key.to_numpy(), sort=False)
    n_cells = int(cell_codes.max()) + 1

    step_within_cell = (
        pd.Series(np.zeros(len(panel), dtype=np.int64))
        .groupby(cell_codes, sort=False)
        .cumcount()
        .to_numpy()
    )
    t_max = int(step_within_cell.max()) + 1
    shape = (n_cells, t_max)

    panel_row_index = np.full(shape, -1, dtype=np.int64)
    panel_row_index[cell_codes, step_within_cell] = np.arange(len(panel))
    valid = panel_row_index >= 0

    is_train = np.zeros(shape, dtype=bool)
    is_train[cell_codes, step_within_cell] = panel["is_train"].to_numpy()

    real_tws_t = _scatter(panel["TWS_t"].to_numpy(dtype=float), cell_codes, step_within_cell, shape)
    target = _scatter(panel["target"].to_numpy(dtype=float), cell_codes, step_within_cell, shape)

    calendar_month = np.zeros(shape, dtype=np.int16)
    calendar_month[cell_codes, step_within_cell] = panel["time"].dt.month.to_numpy()

    time_ym = np.zeros(shape, dtype=np.int32)
    time_ym[cell_codes, step_within_cell] = (
        panel["time"].dt.year.to_numpy() * 12 + panel["time"].dt.month.to_numpy()
    )

    static_features: dict[str, np.ndarray] = {}
    for col in STATIC_CALENDAR_COLUMNS:
        static_features[col] = _scatter(
            panel[col].to_numpy(dtype=float), cell_codes, step_within_cell, shape
        )
    for col in STATIC_COVARIATE_COLUMNS:
        static_features[col] = _scatter(
            panel_with_covariate_lags[col].to_numpy(dtype=float), cell_codes, step_within_cell, shape
        )
        for k in features_config["lags"]["covariates"]:
            lag_col = f"{col}_lag{k}"
            static_features[lag_col] = _scatter(
                panel_with_covariate_lags[lag_col].to_numpy(dtype=float),
                cell_codes,
                step_within_cell,
                shape,
            )

    # Regroupement (cellule, mois calendaire) pour la climatologie : structure statique (ne
    # dépend que du calendrier réel), seule la valeur qu'on y disperse changera par époque.
    month_bucket = np.where(valid, calendar_month - 1, 0)
    flat_cell = np.repeat(np.arange(n_cells), t_max)
    flat_month_bucket = month_bucket.reshape(-1)
    flat_valid = valid.reshape(-1)
    year_rank = (
        pd.Series(np.where(flat_valid, 1, 0))
        .groupby([flat_cell, flat_month_bucket], sort=False)
        .cumsum()
        .to_numpy()
        - 1
    )
    y_max = int(year_rank[flat_valid].max()) + 1 if flat_valid.any() else 1
    climatology_bucket_index = np.full((n_cells, 12, y_max), -1, dtype=np.int64)
    flat_step = np.tile(np.arange(t_max), n_cells)
    valid_flat_idx = np.nonzero(flat_valid)[0]
    climatology_bucket_index[
        flat_cell[valid_flat_idx], flat_month_bucket[valid_flat_idx], year_rank[valid_flat_idx]
    ] = (flat_cell[valid_flat_idx] * t_max + flat_step[valid_flat_idx])

    return FeatureScaffold(
        panel=panel,
        n_cells=n_cells,
        t_max=t_max,
        panel_row_index=panel_row_index,
        valid=valid,
        is_train=is_train,
        real_tws_t=real_tws_t,
        target=target,
        calendar_month=calendar_month,
        time_ym=time_ym,
        static_features=static_features,
        climatology_bucket_index=climatology_bucket_index,
    )
