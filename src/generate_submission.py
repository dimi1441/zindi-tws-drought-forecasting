"""Génère `submissions/submission.csv` : un seul tirage de masquage (mode dynamique, Phase 3),
LightGBM + early stopping calé sur nos propres folds (Phase 5) — pas de bagging (décision
explicite : un GBM n'a pas de notion d'époque, le bagging sur plusieurs tirages est une technique
différente, à revisiter séparément si besoin). Le vrai "masquage dynamique par époque" du brief
attend un modèle itératif (LSTM/TCN), pas encore construit — retour prévu en Phase 5.

Usage : `python -m src.generate_submission`
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.pipeline import build_features
from src.validation.baselines import fit_predict_lightgbm_early_stopping
from src.validation.splits import temporal_holdout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASKING_SEED = 42  # même seed que run_baselines.py, pour rester comparable aux résultats de CV


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
    rng = np.random.default_rng(MASKING_SEED)
    train_df, test_df, _ = build_features(raw_dir, features_config, masking_config, rng)

    with mlflow.start_run(run_name="final_submission"):
        mlflow.log_param("masking_seed", MASKING_SEED)
        mlflow.log_param("model", "lightgbm_early_stopping")
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_param("n_test_rows", len(test_df))

        predictions, model = fit_predict_lightgbm_early_stopping(
            train_df,
            test_df,
            feature_columns,
            temporal_holdout,
            return_model=True,
        )
        best_iteration = model.booster_.best_iteration
        mlflow.log_metric("best_iteration", best_iteration)

        test_predictions = test_df[["ID"]].copy()
        test_predictions["Target"] = predictions

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

        print(f"submission.csv : {submission.shape}, best_iteration={best_iteration}")
        print(submission.head())


if __name__ == "__main__":
    main()
