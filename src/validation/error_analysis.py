"""Phase 6 (jamais faite jusqu'ici) : analyse d'erreur du modèle de production (bag à 7 membres,
41 features incluant le voisinage spatial) sur des prédictions **hors-échantillon** (out-of-fold)
des 5 folds du CV temporel -- pas un score agrégé comme d'habitude, mais une vraie ventilation de
l'erreur par zone climatique, bande de latitude, saison, niveau de TWS, et horizon (ajouté au
périmètre du brief §6, jugé indispensable vu les découvertes du 2026-09-08/09 sur le décalage
train/test d'horizon).

`lat` sert uniquement de clé de regroupement pour cette analyse post-hoc, jamais une feature
modèle (conforme au brief §3.2).

Usage : `python -m src.validation.error_analysis`
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.feature_columns import load_model_feature_columns
from src.features.mask_augmentation import scale_gap_rate_by_period
from src.features.pipeline import build_features
from src.generate_submission import BAGGING_SEEDS, RATE_MULTIPLIERS
from src.validation.baselines import fit_predict_bagged_gbr
from src.validation.splits import temporal_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _climate_zone(lat: pd.Series) -> pd.Series:
    abs_lat = lat.abs()
    return pd.cut(
        abs_lat,
        bins=[-0.01, 23.5, 66.5, 90.0],
        labels=["tropicale", "temperee", "polaire"],
    )


def _local_season(lat: pd.Series, month: pd.Series) -> pd.Series:
    # Hémisphère nord : DJF=hiver, MAM=printemps, JJA=été, SON=automne. Hémisphère sud : inversé
    # (cycle en opposition de phase, confirmé Phase 1).
    north_season = pd.cut(
        month, bins=[0, 2, 5, 8, 11, 12], labels=["hiver", "printemps", "ete", "automne", "hiver"],
        ordered=False,
    )
    south_map = {"hiver": "ete", "ete": "hiver", "printemps": "automne", "automne": "printemps"}
    south_season = north_season.map(south_map)
    return north_season.where(lat >= 0, south_season)


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    base_gap_rate_by_period = features_config["masking"]["target_gap_rate_by_period"]
    feature_columns = load_model_feature_columns(PROJECT_ROOT / "configs")
    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    print("construction des panels d'entrainement (7 membres, seeds 43..49)...")
    train_dfs = []
    for seed, multiplier in zip(BAGGING_SEEDS, RATE_MULTIPLIERS):
        rng = np.random.default_rng(seed)
        scaled_periods = scale_gap_rate_by_period(base_gap_rate_by_period, multiplier)
        masking_config = {"target_gap_rate_by_period": scaled_periods}
        train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)
        train_dfs.append(train_df)

    oof_rows = []
    for fold_idx, (train_mask, val_mask) in enumerate(temporal_splits(train_dfs[0]), start=1):
        fit_dfs = [df.loc[train_mask] for df in train_dfs]
        val_dfs = [df.loc[val_mask] for df in train_dfs]
        y_pred = fit_predict_bagged_gbr(fit_dfs, val_dfs, feature_columns)

        val_meta = val_dfs[0][["lat", "lon", "time", "target", "months_since_last_observed_tws"]].copy()
        val_meta["prediction"] = y_pred
        val_meta["fold"] = fold_idx
        oof_rows.append(val_meta)
        print(f"fold {fold_idx}/5 done ({len(val_meta)} lignes hors-echantillon)")

    oof = pd.concat(oof_rows, ignore_index=True)
    oof["abs_error"] = (oof["target"] - oof["prediction"]).abs()
    oof["climate_zone"] = _climate_zone(oof["lat"])
    oof["lat_band"] = pd.cut(oof["lat"], bins=range(-90, 91, 20)).astype(str)
    oof["season"] = _local_season(oof["lat"], oof["time"].dt.month)
    oof["tws_level"] = pd.qcut(oof["target"], q=5, labels=["tres_bas", "bas", "moyen", "haut", "tres_haut"])
    oof["horizon"] = oof["months_since_last_observed_tws"].clip(upper=8)  # 8 = "8+"

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTotal lignes hors-echantillon: {len(oof)}\n")

    for dim in ["climate_zone", "lat_band", "season", "tws_level", "horizon"]:
        summary = (
            oof.groupby(dim, observed=True)
            .agg(mae=("abs_error", "mean"), n=("abs_error", "size"))
            .sort_values("mae", ascending=False)
        )
        print(f"=== MAE par {dim} ===")
        print(summary.to_string())
        print()
        summary.to_csv(reports_dir / f"error_analysis_by_{dim}.csv")

    # Sauvegarde des predictions completes en dernier -- un echec ici (ex: type de colonne non
    # serialisable) ne doit jamais faire perdre l'analyse deja affichee/sauvegardee ci-dessus.
    try:
        oof.to_parquet(reports_dir / "error_analysis_oof_predictions.parquet", index=False)
        print(f"Predictions completes sauvegardees -> {reports_dir / 'error_analysis_oof_predictions.parquet'}")
    except Exception as exc:
        print(f"AVERTISSEMENT : echec de sauvegarde des predictions completes ({exc})")


if __name__ == "__main__":
    main()
