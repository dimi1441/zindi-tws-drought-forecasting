"""Orchestration Phase 4 : 2 baselines x 2 schémas de validation, tracking MLflow.

Usage : `python -m src.validation.run_baselines`
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.pipeline import build_features
from src.validation.baselines import fit_predict_simple_gbr, predict_persistence_climatology
from src.validation.metrics import aggregate_fold_metrics, compute_metrics
from src.validation.splits import spatial_splits, temporal_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASKING_SEED = 42  # un seul tirage pour cette première passe (voir ADR / plan Phase 4)

SPLIT_SCHEMES = {
    "temporal": temporal_splits,
    "spatial": spatial_splits,
}
BASELINES = {
    "persistence_climatology": lambda fit_df, val_df: predict_persistence_climatology(val_df),
    "simple_gbr": fit_predict_simple_gbr,
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

            y_pred = baseline_fn(fit_df, val_df)
            metrics = compute_metrics(val_df["target"].to_numpy(), y_pred)
            metrics.update({"fold": fold_idx, "n_train": len(fit_df), "n_val": len(val_df)})
            fold_rows.append(metrics)

        aggregated = aggregate_fold_metrics(
            [{"mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]} for r in fold_rows]
        )
        mlflow.log_metrics(aggregated)

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

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]
    rng = np.random.default_rng(MASKING_SEED)
    train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)

    for split_name, split_fn in SPLIT_SCHEMES.items():
        for baseline_name, baseline_fn in BASELINES.items():
            run_one(train_df, split_fn, baseline_fn, baseline_name, split_name)


if __name__ == "__main__":
    main()
