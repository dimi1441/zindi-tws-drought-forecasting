"""ANN (feedforward, masquage augmenté par époque) : premier test de l'approche, demande
utilisateur du 2026-09-06/07 -- voir `src/models/ann/{feature_tensors,dynamic_features,train}.py`.

Étape "smoke test" à un seul split (pas encore le 5-fold complet, cf. plan approuvé : on valide
d'abord que le mécanisme (masquage par époque, arrêt anticipé, parité tensorielle) tourne
correctement et donne un chiffre plausible avant d'investir dans le 5-fold complet). Compare
explicitement les 33 features d'origine et les 38 (avec tendance long terme + comptes de
fiabilité, ajout du 2026-09-07) sur le MÊME split, pour trancher si ces features aident aussi
l'ANN (résultat mitigé sur GBM : aide en spatial, nuit légèrement en temporel).

Usage : `python -m src.validation.run_ann_validation`
"""

import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.models.ann.feature_tensors import build_feature_scaffold
from src.models.ann.train import predict_ann, train_ann
from src.validation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42

NEW_TREND_AND_COUNT_COLUMNS = [
    "TWS_t_climatology_count",
    "TWS_t_diff1_expanding_mean",
    "TWS_t_diff1_expanding_mean_count",
    "TWS_t_diff12_expanding_mean",
    "TWS_t_diff12_expanding_mean_count",
]


def _temporal_two_tier_masks(
    scaffold, outer_val_fraction: float = 0.15, inner_val_fraction: float = 0.15
):
    """Découpe temporelle à deux niveaux (un seul split, pas un 5-fold) : `outer_val_mask` sert
    au chiffre final rapporté, `inner_val_mask` (à l'intérieur du train restant) sert uniquement
    à décider quand arrêter -- jamais la même donnée pour les deux, même principe que
    `baselines.fit_predict_lightgbm_early_stopping`.
    """
    is_train = scaffold.panel["is_train"].to_numpy()
    time_col = scaffold.panel["time"]
    months = pd.DatetimeIndex(sorted(scaffold.panel.loc[is_train, "time"].unique()))

    outer_cutoff = months[int(len(months) * (1 - outer_val_fraction))]
    inner_months = months[months < outer_cutoff]
    inner_cutoff = inner_months[int(len(inner_months) * (1 - inner_val_fraction))]

    fit_mask = is_train & (time_col < inner_cutoff).to_numpy()
    inner_val_mask = is_train & (time_col >= inner_cutoff).to_numpy() & (time_col < outer_cutoff).to_numpy()
    outer_val_mask = is_train & (time_col >= outer_cutoff).to_numpy()
    return fit_mask, inner_val_mask, outer_val_mask


def run_one(
    scaffold, feature_columns: list[str], tag: str, masking_config: dict,
    resample_mask_each_epoch: bool = True, **train_kwargs,
) -> dict:
    fit_mask, inner_val_mask, outer_val_mask = _temporal_two_tier_masks(scaffold)

    with mlflow.start_run(run_name=f"ann_smoke_test__{tag}"):
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("feature_set", tag)
        mlflow.log_param("masking_seed", SEED)
        mlflow.log_param("resample_mask_each_epoch", resample_mask_each_epoch)
        mlflow.log_params(train_kwargs)
        mlflow.log_param("n_fit_rows", int(fit_mask.sum()))
        mlflow.log_param("n_inner_val_rows", int(inner_val_mask.sum()))
        mlflow.log_param("n_outer_val_rows", int(outer_val_mask.sum()))

        start = time.perf_counter()
        trained = train_ann(
            scaffold, feature_columns, fit_mask, inner_val_mask, masking_config, seed=SEED,
            resample_mask_each_epoch=resample_mask_each_epoch, **train_kwargs,
        )
        elapsed = time.perf_counter() - start

        predictions = predict_ann(trained, scaffold, outer_val_mask)
        y_true = scaffold.panel.loc[outer_val_mask, "target"].to_numpy()
        metrics = compute_metrics(y_true, predictions)

        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_epoch", trained.best_epoch)
        mlflow.log_metric("n_epochs_run", trained.n_epochs_run)
        mlflow.log_metric("fit_seconds", elapsed)

        print(
            f"ann [{tag}] ({len(feature_columns)} features): {metrics}, "
            f"best_epoch={trained.best_epoch}/{trained.n_epochs_run}, {elapsed:.1f}s"
        )
        return metrics


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    masking_config = features_config["masking"]
    feature_columns_38 = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "feature_columns.yaml").read_text()
    )["feature_columns"]
    feature_columns_33 = [c for c in feature_columns_38 if c not in NEW_TREND_AND_COUNT_COLUMNS]

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]
    scaffold = build_feature_scaffold(raw_dir, features_config)

    # Diagnostic (2026-09-07) : masquage fixe vs variable donnaient un résultat quasi identique
    # (meilleur epoch=2 dans les deux cas) -- le masquage par époque n'est PAS la cause de
    # l'instabilité. Prochaine hypothèse : le taux d'apprentissage par défaut (1e-3, Adam) est
    # trop élevé pour ce problème. Ce run reprend le masquage variable normal (celui qu'on veut
    # vraiment utiliser) avec un taux d'apprentissage 10x plus faible et plus de patience (la
    # convergence est censée être plus lente avec un LR plus petit).
    run_one(
        scaffold, feature_columns_33, "33_features_lr1e-4", masking_config,
        resample_mask_each_epoch=True, learning_rate_init=1e-4, max_epochs=200, patience=30,
    )


if __name__ == "__main__":
    main()
