"""Phase 5, bagging de GBR (demande utilisateur explicite du 2026-09-05, hors roadmap
itération E qui parlait d'un ensemble de modèles génériques) : moyenne de 8 `make_gbr_pipeline()`
(même hyperparamètres que le starter, cf. `baselines.py`), chacun entraîné sur un tirage de
masquage augmenté différent (seeds 42..49 fixées explicitement pour la reproductibilité — N=8
retenu le 2026-09-07 via `run_bagging_curve.py`, voir ce module pour la courbe MAE vs nombre de
membres qui justifie ce choix). Pas de bootstrap des lignes en plus — la diversité vient des
tirages de masque (mois de trous différents à chaque fois, mécanisme dynamique de la Phase 3) et,
depuis le 2026-09-07, aussi du **taux** de trou lui-même : `RATE_MULTIPLIERS` fait varier la
difficulté du masquage par membre (spread 0.5x-2x, demande utilisateur constatant que le train
voyait ~10-16% de mois masqués contre 67% dans le vrai test).

Réutilise le même harnais que `run_baselines.py` (`temporal_splits`/`spatial_splits`, 5 folds
chacun) pour rester directement comparable aux baselines déjà loggées (`full_features_gbr`,
`lightgbm_early_stop`) : mêmes schémas de validation, mêmes métriques.

Usage : `python -m src.validation.run_bagging`
"""

import time
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
from src.validation.splits import spatial_splits, temporal_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Seeds fixées explicitement (demande utilisateur) : 42 = même premier tirage que le reste du
# projet (run_baselines.py, generate_submission.py), 43..49 = les 7 tirages supplémentaires pour
# le bagging (N=8, cf. run_bagging_curve.py). Toute reprise de cette expérience avec ces mêmes
# seeds doit reproduire les mêmes résultats (build_features est déterministe à rng fixé).
BAGGING_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]

# Multiplicateur du taux de trou par membre (demande utilisateur du 2026-09-07, voir
# `generate_submission.py` pour le détail du raisonnement) : spread modéré 0.5x-2x, linéaire par
# position dans BAGGING_SEEDS.
RATE_MULTIPLIERS = np.linspace(0.5, 2.0, len(BAGGING_SEEDS))

SPLIT_SCHEMES = {
    "temporal": temporal_splits,
    "spatial": spatial_splits,
}


def run_bagging_scheme(
    train_dfs: list[pd.DataFrame], feature_columns: list[str], split_fn, split_name: str
) -> dict:
    with mlflow.start_run(run_name=f"bagged_gbr__{split_name}"):
        mlflow.log_param("baseline", "bagged_gbr")
        mlflow.log_param("split_scheme", split_name)
        mlflow.log_param("masking_seeds", BAGGING_SEEDS)
        mlflow.log_param("rate_multipliers", list(RATE_MULTIPLIERS))
        mlflow.log_param("n_models", len(BAGGING_SEEDS))

        fold_rows = []
        # Les splits (time/lat/lon) sont identiques quel que soit le tirage de masque : le
        # masquage ne change que TWS_t et les features dérivées, jamais les lignes/mois présents.
        for fold_idx, (train_mask, val_mask) in enumerate(split_fn(train_dfs[0]), start=1):
            fit_dfs = [df.loc[train_mask] for df in train_dfs]
            val_dfs = [df.loc[val_mask] for df in train_dfs]

            fit_predict_start = time.perf_counter()
            y_pred = fit_predict_bagged_gbr(fit_dfs, val_dfs, feature_columns)
            fit_predict_seconds = time.perf_counter() - fit_predict_start

            y_true = val_dfs[0]["target"].to_numpy()  # target identique pour tous les tirages
            metrics = compute_metrics(y_true, y_pred)
            metrics.update(
                {
                    "fold": fold_idx,
                    "n_train": len(fit_dfs[0]),
                    "n_val": len(val_dfs[0]),
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
        fold_detail_path = reports_dir / f"fold_detail_bagged_gbr_{split_name}.csv"
        pd.DataFrame(fold_rows).to_csv(fold_detail_path, index=False)
        mlflow.log_artifact(str(fold_detail_path))

        print(f"bagged_gbr x {split_name}: {aggregated}")
        return aggregated


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    base_gap_rate_by_period = features_config["masking"]["target_gap_rate_by_period"]

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    feature_columns = load_model_feature_columns(PROJECT_ROOT / "configs")

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    train_dfs = []
    for seed, multiplier in zip(BAGGING_SEEDS, RATE_MULTIPLIERS):
        rng = np.random.default_rng(seed)
        scaled_periods = scale_gap_rate_by_period(base_gap_rate_by_period, multiplier)
        masking_config = {"target_gap_rate_by_period": scaled_periods}
        train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)
        train_dfs.append(train_df)

    for split_name, split_fn in SPLIT_SCHEMES.items():
        run_bagging_scheme(train_dfs, feature_columns, split_fn, split_name)


if __name__ == "__main__":
    main()
