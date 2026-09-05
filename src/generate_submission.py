"""Génère `submissions/submission.csv` : bagging de 3 GBR (`make_gbr_pipeline()`, décision
explicite de l'utilisateur du 2026-09-05, voir `src/validation/run_bagging.py`), chacun entraîné
sur 100% du train mais avec un tirage de masquage augmenté différent (seeds 42/43/44, mode
dynamique, Phase 3) — la diversité vient du masquage, pas d'un bootstrap des lignes ni du hasard
interne du modèle (`random_state=42` fixe sur chaque membre). Prédiction finale = moyenne des 3
GBR. Validé sur `run_bagging.py` (mêmes seeds, même harnais que les baselines précédentes) : MAE
0.372 (temporel) / 0.360 (spatial), meilleur que le GBR seul (0.384 / 0.366) sur les deux schémas,
meilleur que LightGBM+early-stopping (0.384) en temporel mais pas en spatial (0.349) — remplace la
précédente soumission LightGBM mono-tirage sur demande explicite de l'utilisateur.

Usage : `python -m src.generate_submission`
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.pipeline import build_features
from src.validation.baselines import fit_predict_gbr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Mêmes seeds que `run_bagging.py`, fixées explicitement pour la reproductibilité (demande
# utilisateur) : 42 = même premier tirage que le reste du projet, 43/44 = deux tirages de plus.
BAGGING_SEEDS = [42, 43, 44]


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    masking_config = features_config["masking"]
    feature_columns = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "feature_columns.yaml").read_text()
    )["feature_columns"]

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    with mlflow.start_run(run_name="final_submission"):
        mlflow.log_param("masking_seeds", BAGGING_SEEDS)
        mlflow.log_param("model", "bagged_gbr")
        mlflow.log_param("n_models", len(BAGGING_SEEDS))
        mlflow.log_param("n_features", len(feature_columns))

        member_predictions = []
        test_ids = None
        for seed in BAGGING_SEEDS:
            rng = np.random.default_rng(seed)
            train_df, test_df, _ = build_features(raw_dir, features_config, masking_config, rng)
            if test_ids is None:
                test_ids = test_df["ID"].to_numpy()
            else:
                assert (test_df["ID"].to_numpy() == test_ids).all(), (
                    "l'ordre des lignes de test a changé entre deux tirages de masquage"
                )
            member_predictions.append(
                fit_predict_gbr(train_df, test_df, feature_columns)
            )

        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_param("n_test_rows", len(test_df))

        predictions = np.mean(member_predictions, axis=0)

        test_predictions = pd.DataFrame({"ID": test_ids, "Target": predictions})

        sample_submission = pd.read_csv(raw_dir / "SampleSubmission.csv")
        submission = sample_submission[["ID"]].merge(
            test_predictions, on="ID", how="left", validate="one_to_one"
        )
        assert submission["Target"].notna().all(), "des IDs du gabarit n'ont pas de prédiction"
        assert len(submission) == len(sample_submission)

        submissions_dir = PROJECT_ROOT / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        submission_path = submissions_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        mlflow.log_artifact(str(submission_path))

        print(f"submission.csv : {submission.shape}, n_models={len(BAGGING_SEEDS)}")
        print(submission.head())


if __name__ == "__main__":
    main()
