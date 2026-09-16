"""Phase 5 itération D : est-ce que les 8 nouvelles features de voisinage spatial
(`add_spatial_neighborhood_features`, 2026-09-09) aident le bag de production (7 membres, seeds
43..49) sur le schéma **temporel** -- celui qui ressemble au vrai test Zindi ? Même logique que
l'évaluation des features de tendance long terme (JOURNAL.md 2026-09-07) : comparer 33 features
(actuel, sans voisinage ni tendance) vs 41 (+ les 8 de voisinage, tendance toujours exclue) sur le
même CV 5-fold, mêmes seeds/multiplicateurs que la production.

Usage : `python -m src.validation.compare_spatial_neighborhood`
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.feature_columns import EXCLUDED_FEATURE_COLUMNS
from src.features.mask_augmentation import scale_gap_rate_by_period
from src.features.pipeline import build_features
from src.generate_submission import BAGGING_SEEDS, RATE_MULTIPLIERS
from src.validation.baselines import fit_predict_bagged_gbr
from src.validation.metrics import aggregate_fold_metrics, compute_metrics
from src.validation.splits import temporal_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEIGHBORHOOD_COLUMNS = [
    f"TWS_neighbor_{direction}_{stat}"
    for direction in ("N", "S", "E", "W")
    for stat in ("last_observed", "months_since_last_observed")
]

FEATURE_SETS = {
    "sans_voisinage": EXCLUDED_FEATURE_COLUMNS | set(NEIGHBORHOOD_COLUMNS),
    "avec_voisinage": EXCLUDED_FEATURE_COLUMNS,
}


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    base_gap_rate_by_period = features_config["masking"]["target_gap_rate_by_period"]

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    print("construction des panels d'entrainement (7 membres, seeds 43..49)...")
    train_dfs = []
    for seed, multiplier in zip(BAGGING_SEEDS, RATE_MULTIPLIERS):
        rng = np.random.default_rng(seed)
        scaled_periods = scale_gap_rate_by_period(base_gap_rate_by_period, multiplier)
        masking_config = {"target_gap_rate_by_period": scaled_periods}
        train_df, _, all_feature_columns = build_features(
            raw_dir, features_config, masking_config, rng
        )
        train_dfs.append(train_df)

    results = []
    for set_name, excluded in FEATURE_SETS.items():
        feature_columns = [c for c in all_feature_columns if c not in excluded]
        print(f"\n=== {set_name} : {len(feature_columns)} features ===")

        with mlflow.start_run(run_name=f"spatial_neighborhood__{set_name}__temporal"):
            mlflow.log_param("feature_set", set_name)
            mlflow.log_param("n_features", len(feature_columns))
            mlflow.log_param("masking_seeds", BAGGING_SEEDS)

            fold_rows = []
            for fold_idx, (train_mask, val_mask) in enumerate(
                temporal_splits(train_dfs[0]), start=1
            ):
                fit_dfs = [df.loc[train_mask] for df in train_dfs]
                val_dfs = [df.loc[val_mask] for df in train_dfs]
                y_pred = fit_predict_bagged_gbr(fit_dfs, val_dfs, feature_columns)
                y_true = val_dfs[0]["target"].to_numpy()
                metrics = compute_metrics(y_true, y_pred)
                metrics["fold"] = fold_idx
                fold_rows.append(metrics)
                print(f"  fold {fold_idx}/5: mae={metrics['mae']:.4f}")

            aggregated = aggregate_fold_metrics(
                [{"mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]} for r in fold_rows]
            )
            mlflow.log_metrics(aggregated)

        results.append({"feature_set": set_name, "n_features": len(feature_columns), **aggregated})
        print(f"{set_name} : MAE {aggregated['mae_mean']:.4f} +/- {aggregated['mae_std']:.4f}")

    results_df = pd.DataFrame(results)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "spatial_neighborhood_comparison.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + results_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
