"""Orchestration Phase 4/5 : baselines x schémas de validation, tracking MLflow.

Phase 4 : persistance + GBR simple (8 features du starter). Phase 5 itération B+C : ajoute
`full_features_gbr`, le même GBR mais sur les 33 features de `configs/feature_columns.yaml`
(lags, climatologie, horizon) — mesure directe du gain par rapport à la baseline simple. Ajoute
aussi `lightgbm_early_stop` : mêmes hyperparamètres mais early stopping calé sur nos propres
folds (voir `baselines.fit_predict_lightgbm_early_stopping`) plutôt que le tirage aléatoire
interne de sklearn — teste si plus d'itérations + un vrai arrêt anticipé aident.

Chaque run MLflow loggue aussi `fit_predict_seconds_total`/`_mean` (temps mesuré par fold,
`time.perf_counter()`) — pas un vrai suivi d'empreinte carbone (CodeCarbon, prévu plus tard,
brief §6.4), mais un point de comparaison chiffré entre modèles/schémas en attendant.

Usage : `python -m src.validation.run_baselines`
"""

import time
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.pipeline import build_features
from src.validation.baselines import (
    fit_predict_gbr,
    fit_predict_lightgbm_early_stopping,
    fit_predict_simple_gbr,
    predict_persistence_climatology,
)
from src.validation.metrics import aggregate_fold_metrics, compute_metrics
from src.validation.splits import (
    spatial_holdout,
    spatial_splits,
    temporal_holdout,
    temporal_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASKING_SEED = 42  # un seul tirage pour cette première passe (voir ADR / plan Phase 4)

SPLIT_SCHEMES = {
    "temporal": temporal_splits,
    "spatial": spatial_splits,
}
# Découpe interne (early stopping LightGBM) adaptée au schéma de validation externe en cours.
INNER_HOLDOUT_BY_SCHEME = {
    "temporal": temporal_holdout,
    "spatial": spatial_holdout,
}


def build_baselines(feature_columns: list[str], inner_holdout_fn) -> dict:
    return {
        "persistence_climatology": lambda fit_df, val_df: predict_persistence_climatology(val_df),
        "simple_gbr": fit_predict_simple_gbr,
        "full_features_gbr": partial(fit_predict_gbr, feature_columns=feature_columns),
        "lightgbm_early_stop": partial(
            fit_predict_lightgbm_early_stopping,
            feature_columns=feature_columns,
            inner_holdout_fn=inner_holdout_fn,
        ),
    }


def run_one(train_df: pd.DataFrame, split_fn, baseline_fn, baseline_name: str, split_name: str) -> dict:
    with mlflow.start_run(run_name=f"{baseline_name}__{split_name}"):
        mlflow.log_param("baseline", baseline_name)
        mlflow.log_param("split_scheme", split_name)
        mlflow.log_param("masking_seed", MASKING_SEED)

        fold_rows = []
        for fold_idx, (train_mask, val_mask) in enumerate(split_fn(train_df), start=1):
            fit_df = train_df.loc[train_mask]
            val_df = train_df.loc[val_mask]

            fit_predict_start = time.perf_counter()
            y_pred = baseline_fn(fit_df, val_df)
            fit_predict_seconds = time.perf_counter() - fit_predict_start

            metrics = compute_metrics(val_df["target"].to_numpy(), y_pred)
            metrics.update(
                {
                    "fold": fold_idx,
                    "n_train": len(fit_df),
                    "n_val": len(val_df),
                    "fit_predict_seconds": fit_predict_seconds,
                }
            )
            fold_rows.append(metrics)

        aggregated = aggregate_fold_metrics(
            [{"mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]} for r in fold_rows]
        )
        total_fit_predict_seconds = sum(r["fit_predict_seconds"] for r in fold_rows)
        mlflow.log_metrics(aggregated)
        mlflow.log_metric("fit_predict_seconds_total", total_fit_predict_seconds)
        mlflow.log_metric(
            "fit_predict_seconds_mean", total_fit_predict_seconds / len(fold_rows)
        )

        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        fold_detail_path = reports_dir / f"fold_detail_{baseline_name}_{split_name}.csv"
        pd.DataFrame(fold_rows).to_csv(fold_detail_path, index=False)
        mlflow.log_artifact(str(fold_detail_path))

        print(f"{baseline_name} x {split_name}: {aggregated}")
        return aggregated


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    masking_config = features_config["masking"]

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    feature_columns = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "feature_columns.yaml").read_text()
    )["feature_columns"]

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]
    rng = np.random.default_rng(MASKING_SEED)
    train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)

    for split_name, split_fn in SPLIT_SCHEMES.items():
        baselines = build_baselines(feature_columns, INNER_HOLDOUT_BY_SCHEME[split_name])
        for baseline_name, baseline_fn in baselines.items():
            run_one(train_df, split_fn, baseline_fn, baseline_name, split_name)


if __name__ == "__main__":
    main()
