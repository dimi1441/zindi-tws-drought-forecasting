"""Compare plusieurs plages de `RATE_MULTIPLIERS` (demande utilisateur du 2026-09-08, suite au
diagnostic `diagnose_bag_horizon_specialization.py` : le membre à 0.5x nuisait sans bénéfice sur
horizon>1, exclu -> 0.75 -> 0.74 sur Zindi). Teste si remonter le plancher (au lieu d'exclure un
membre) capture un gain similaire ou meilleur, et si pousser le plafond au-delà de 2x aide encore.

Mêmes 8 seeds que la production (42..49) pour chaque plage -- seul le multiplicateur assigné à
chaque seed change (`np.linspace(low, high, 8)`). Schéma **temporel** uniquement (celui qui
ressemble au vrai test Zindi, cf. `run_bagging_curve.py`) pour limiter le coût de calcul.

Note sur le plafond `cap=0.9` de `scale_gap_rate_by_period` : au-delà d'un multiplicateur
d'environ 2.57x, les périodes 2011-2014 (taux de base 35%) et 2015 (50%) sont déjà saturées à 90%
-- pousser plus haut n'affecte alors quasiment plus que la période 2002-2010 (taux de base 5%).

Usage : `python -m src.validation.compare_rate_ranges`
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.feature_columns import load_model_feature_columns
from src.features.mask_augmentation import scale_gap_rate_by_period
from src.features.pipeline import build_features
from src.validation.baselines import fit_predict_bagged_gbr
from src.validation.metrics import aggregate_fold_metrics, compute_metrics
from src.validation.splits import temporal_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BAGGING_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]

# Plages candidates : (low, high) pour np.linspace(low, high, 8). 0.5-2.0 = réglage actuel
# (production avant exclusion du seed 42), pour comparaison directe.
CANDIDATE_RANGES = [
    (0.5, 2.0),  # réglage actuel (production), rappel de référence
    (1.0, 2.0),  # plancher remonté, demande utilisateur
    (1.0, 3.0),  # plafond aussi pousse, demande utilisateur
]


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    base_gap_rate_by_period = features_config["masking"]["target_gap_rate_by_period"]
    feature_columns = load_model_feature_columns(PROJECT_ROOT / "configs")

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    results = []
    for low, high in CANDIDATE_RANGES:
        multipliers = np.linspace(low, high, len(BAGGING_SEEDS))
        print(f"\n=== plage {low}x-{high}x : multiplicateurs {np.round(multipliers, 3)} ===")

        train_dfs = []
        for seed, multiplier in zip(BAGGING_SEEDS, multipliers):
            rng = np.random.default_rng(seed)
            scaled_periods = scale_gap_rate_by_period(base_gap_rate_by_period, multiplier)
            masking_config = {"target_gap_rate_by_period": scaled_periods}
            train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)
            train_dfs.append(train_df)

        with mlflow.start_run(run_name=f"rate_range__{low}x_{high}x__temporal"):
            mlflow.log_param("masking_seeds", BAGGING_SEEDS)
            mlflow.log_param("rate_multipliers", list(multipliers))
            mlflow.log_param("range_low", low)
            mlflow.log_param("range_high", high)

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

        results.append({"range_low": low, "range_high": high, **aggregated})
        print(f"plage {low}x-{high}x : MAE {aggregated['mae_mean']:.4f} +/- {aggregated['mae_std']:.4f}")

    results_df = pd.DataFrame(results)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "rate_range_comparison.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + results_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
