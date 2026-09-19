"""Recherche d'hyperparamètres légère pour un simple GBR (demande utilisateur du 2026-09-10,
retour en arrière sur le bagging jugé trop complexe pour un gain modeste ~3-4% relatif).

Teste ~10 combinaisons autour des réglages actuels (hérités tels quels du notebook de départ,
jamais remis en question dans ce projet -- cf. JOURNAL.md), sur le CV temporel 5-fold, un seul
tirage de masquage (seed=42, taux de base non multiplié -- même référence que
`run_baselines.py::MASKING_SEED`) et le jeu de features de production actuel (41, avec voisinage
spatial).

Correction appliquée ici : `HistGradientBoostingRegressor` a `early_stopping="auto"` par défaut,
qui s'active silencieusement au-delà de 10 000 lignes et pioche un split interne **aléatoire**,
non temporel (même défaut déjà repéré en Phase 5 pour LightGBM, jamais corrigé sur le GBR) --
désactivé ici (`early_stopping=False`) pour une comparaison propre à `max_iter` fixé.

Usage : `python -m src.validation.search_hyperparameters`
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.features.feature_columns import load_model_feature_columns
from src.features.pipeline import build_features
from src.validation.metrics import aggregate_fold_metrics, compute_metrics
from src.validation.splits import temporal_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASKING_SEED = 42  # meme reference que run_baselines.py, un seul tirage, taux de base

# (learning_rate, max_iter, max_depth, min_samples_leaf, l2_regularization)
CANDIDATES = {
    "defaut_actuel": (0.05, 300, 8, 50, 1.0),
    "lr_bas_plus_iter": (0.03, 500, 8, 50, 1.0),
    "arbres_peu_profonds": (0.05, 300, 4, 50, 1.0),
    "arbres_profonds": (0.05, 300, 12, 50, 1.0),
    "feuilles_larges": (0.05, 300, 8, 100, 1.0),
    "feuilles_fines": (0.05, 300, 8, 20, 1.0),
    "l2_fort": (0.05, 300, 8, 50, 5.0),
    "l2_nul": (0.05, 300, 8, 50, 0.0),
    "lr_haut_peu_iter": (0.1, 150, 8, 50, 1.0),
    "profond_regularise": (0.05, 300, 10, 100, 2.0),
}


def _make_pipeline(learning_rate, max_iter, max_depth, min_samples_leaf, l2_regularization) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "gbr",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    early_stopping=False,  # cf. docstring : comparaison propre a max_iter fixe
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    masking_config = features_config["masking"]
    feature_columns = load_model_feature_columns(PROJECT_ROOT / "configs")
    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    print("construction du panel d'entrainement (seed=42, taux de base)...")
    rng = np.random.default_rng(MASKING_SEED)
    train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)

    param_names = ["learning_rate", "max_iter", "max_depth", "min_samples_leaf", "l2_regularization"]
    results = []
    for name, params in CANDIDATES.items():
        print(f"\n=== {name} : {params} ===")
        with mlflow.start_run(run_name=f"hp_search__{name}"):
            mlflow.log_param("masking_seed", MASKING_SEED)
            mlflow.log_params({"config": name, **dict(zip(param_names, params))})

            fold_rows = []
            for fold_idx, (train_mask, val_mask) in enumerate(temporal_splits(train_df), start=1):
                fit_df = train_df.loc[train_mask]
                val_df = train_df.loc[val_mask]
                model = _make_pipeline(*params)
                X_fit = fit_df[feature_columns].to_numpy(dtype=np.float32)
                y_fit = fit_df["target"].to_numpy(dtype=np.float32)
                X_val = val_df[feature_columns].to_numpy(dtype=np.float32)
                y_val = val_df["target"].to_numpy(dtype=np.float32)
                model.fit(X_fit, y_fit)
                metrics = compute_metrics(y_val, model.predict(X_val))
                metrics["fold"] = fold_idx
                fold_rows.append(metrics)
                print(f"  fold {fold_idx}/5: mae={metrics['mae']:.4f}")

            aggregated = aggregate_fold_metrics(
                [{"mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]} for r in fold_rows]
            )
            mlflow.log_metrics(aggregated)
            results.append({"config": name, **dict(zip(param_names, params)), **aggregated})
            print(f"{name} : MAE {aggregated['mae_mean']:.4f} +/- {aggregated['mae_std']:.4f}")

    results_df = pd.DataFrame(results).sort_values("mae_mean")
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "hyperparameter_search.csv"
    results_df.to_csv(out_path, index=False)

    with mlflow.start_run(run_name="hp_search__summary"):
        mlflow.log_param("masking_seed", MASKING_SEED)
        mlflow.log_param("n_candidates", len(CANDIDATES))
        mlflow.log_metric("best_mae_mean", results_df.iloc[0]["mae_mean"])
        mlflow.log_param("best_config", results_df.iloc[0]["config"])
        mlflow.log_artifact(str(out_path))

    print("\n" + results_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
