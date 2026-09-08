"""Phase 5 itération E : courbe de bagging (MAE en fonction du nombre de GBR moyennés), pour
déterminer combien de membres utiliser plutôt que d'en fixer un nombre arbitraire (discussion
utilisateur du 2026-09-07 : le seul levier de diversité entre membres est le tirage de masquage,
pas un bootstrap des lignes — donc corrélation plus forte entre membres qu'un bagging classique,
plateau attendu plus tôt).

Entraîne `N_MAX` GBR une seule fois par fold (un par tirage de masquage, `fit_predict_bagged_gbr_members`)
puis évalue les moyennes cumulatives des N premiers pour N=1..N_MAX à partir des mêmes prédictions
— pas de réentraînement par valeur de N.

Seul le schéma **temporel** est couvert ici : c'est celui qui ressemble le plus au vrai test Zindi
(mêmes 15 715 cellules, mois futurs — cf. JOURNAL.md 2026-09-05/07), et doubler le coût de calcul
pour le schéma spatial n'apporterait pas d'information pertinente pour la décision de soumission.

**Note (2026-09-07, après coup)** : ce module utilise encore un taux de trou fixe (`masking_config`
partagé), pas la variation par membre ajoutée ensuite dans `run_bagging.py`/`generate_submission.py`
(`RATE_MULTIPLIERS`). Le N=8 retenu ici n'a donc pas été revalidé sous ce nouveau mécanisme -- à
refaire si la variation de taux s'avère utile et qu'on veut confirmer N=8 reste le bon choix.

Usage : `python -m src.validation.run_bagging_curve`
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.feature_columns import load_model_feature_columns
from src.features.pipeline import build_features
from src.validation.baselines import fit_predict_bagged_gbr_members
from src.validation.metrics import compute_metrics
from src.validation.splits import temporal_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 15 tirages : 42 (même premier seed que le reste du projet) + 14 suivants.
CURVE_SEEDS = list(range(42, 42 + 15))


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    masking_config = features_config["masking"]

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    feature_columns = load_model_feature_columns(PROJECT_ROOT / "configs")
    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    train_dfs = []
    for seed in CURVE_SEEDS:
        rng = np.random.default_rng(seed)
        train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)
        train_dfs.append(train_df)

    n_max = len(CURVE_SEEDS)
    rows = []
    with mlflow.start_run(run_name="bagging_curve__temporal"):
        mlflow.log_param("curve_seeds", CURVE_SEEDS)
        mlflow.log_param("n_max", n_max)

        for fold_idx, (train_mask, val_mask) in enumerate(temporal_splits(train_dfs[0]), start=1):
            fit_dfs = [df.loc[train_mask] for df in train_dfs]
            val_dfs = [df.loc[val_mask] for df in train_dfs]
            y_true = val_dfs[0]["target"].to_numpy()

            member_preds, member_seconds = fit_predict_bagged_gbr_members(
                fit_dfs, val_dfs, feature_columns
            )
            cumulative_seconds = np.cumsum(member_seconds)

            for n in range(1, n_max + 1):
                y_pred = member_preds[:n].mean(axis=0)
                metrics = compute_metrics(y_true, y_pred)
                rows.append(
                    {
                        "fold": fold_idx,
                        "n_models": n,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "cumulative_fit_predict_seconds": cumulative_seconds[n - 1],
                    }
                )
            print(f"fold {fold_idx}/5 done ({n_max} membres entraînés)")

        detail_df = pd.DataFrame(rows)
        curve = (
            detail_df.groupby("n_models")
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                r2_mean=("r2", "mean"),
                seconds_mean=("cumulative_fit_predict_seconds", "mean"),
            )
            .reset_index()
        )

        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        detail_path = reports_dir / "bagging_curve_temporal_detail.csv"
        curve_path = reports_dir / "bagging_curve_temporal.csv"
        detail_df.to_csv(detail_path, index=False)
        curve.to_csv(curve_path, index=False)
        mlflow.log_artifact(str(detail_path))
        mlflow.log_artifact(str(curve_path))
        for _, row in curve.iterrows():
            mlflow.log_metric("mae_mean", row["mae_mean"], step=int(row["n_models"]))

        best_row = curve.loc[curve["mae_mean"].idxmin()]
        threshold = best_row["mae_mean"] * 1.01  # à 1% du meilleur MAE observé
        suggested_n = int(curve.loc[curve["mae_mean"] <= threshold, "n_models"].min())

        print(curve.to_string(index=False))
        print(f"\nMeilleur MAE observe: {best_row['mae_mean']:.4f} (N={int(best_row['n_models'])})")
        print(f"Suggestion (a 1% du meilleur, meilleur compromis cout/gain): N={suggested_n}")


if __name__ == "__main__":
    main()
