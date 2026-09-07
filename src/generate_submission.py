"""Génère `submissions/submission.csv` : bagging de 8 GBR (`make_gbr_pipeline()`, décision
explicite de l'utilisateur du 2026-09-05, voir `src/validation/run_bagging.py`), chacun entraîné
sur 100% du train mais avec un tirage de masquage augmenté différent (seeds 42..49, mode
dynamique, Phase 3) — la diversité vient du masquage, pas d'un bootstrap des lignes ni du hasard
interne du modèle (`random_state=42` fixe sur chaque membre). Prédiction finale = moyenne des 8
GBR.

**N=8 choisi le 2026-09-07** via `run_bagging_curve.py` (courbe MAE vs nombre de membres,
schéma temporel, 15 tirages testés) : MAE 0.3838 (N=1) → 0.3725 (N=2) → 0.3717 (N=3, ancien
réglage) → plateau bruité autour de 0.368-0.369 dès N≈8, meilleur observé 0.3683 à N=8, aucun
gain net au-delà (N=15 : 0.3689, dans le bruit inter-folds ~0.037). Le schéma spatial n'a pas été
remesuré à N=8 (dernier chiffre connu à N=3 : 0.360, `reports/fold_detail_bagged_gbr_spatial.csv`)
— non prioritaire car le vrai test Zindi ressemble au schéma temporel, pas spatial.

**Correction du 2026-09-06** (trouvée en discutant du processus d'inférence avec l'utilisateur) :
les features dérivées de test (lags, climatologie, horizon) reposent sur l'historique de train
(panel combiné, `build_cell_timeline`) — si on les calcule à partir d'un train *masqué* par le
tirage augmenté utilisé pour l'entraînement, on dégrade artificiellement les features de test avec
des trous fictifs, alors qu'à l'inférence l'historique réel et complet de train est disponible.
Le masquage augmenté est une technique d'entraînement (exposer le modèle à des trous simulés),
pas une dégradation qui doit aussi s'appliquer aux features qu'on calcule pour la vraie
prédiction finale. Donc : les features de test sont maintenant calculées **une seule fois, sans
aucun masquage** (`build_features(...)` sans `masking_config`/`rng`), partagées par les 3 membres
du bag — seul l'entraînement (train_df) varie par tirage de masquage.

Usage : `python -m src.generate_submission`
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.feature_columns import load_model_feature_columns
from src.features.pipeline import build_features
from src.validation.baselines import fit_predict_gbr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Mêmes seeds que `run_bagging.py`/`run_bagging_curve.py`, fixées explicitement pour la
# reproductibilité (demande utilisateur) : 42 = même premier tirage que le reste du projet,
# 43..49 = les 7 tirages suivants (N=8 retenu suite à `run_bagging_curve.py`, cf. docstring).
BAGGING_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    masking_config = features_config["masking"]
    feature_columns = load_model_feature_columns(PROJECT_ROOT / "configs")

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    # Historique de train réel, non masqué : sert uniquement à calculer les features de test
    # (partagées par tous les membres du bag), jamais à l'entraînement. Voir la note de
    # correction ci-dessus.
    _, test_df_real, _ = build_features(raw_dir, features_config)

    with mlflow.start_run(run_name="final_submission"):
        mlflow.log_param("masking_seeds", BAGGING_SEEDS)
        mlflow.log_param("model", "bagged_gbr")
        mlflow.log_param("n_models", len(BAGGING_SEEDS))
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("test_features_masking", "none (real train history)")

        member_predictions = []
        for seed in BAGGING_SEEDS:
            rng = np.random.default_rng(seed)
            train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)
            member_predictions.append(
                fit_predict_gbr(train_df, test_df_real, feature_columns)
            )

        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_param("n_test_rows", len(test_df_real))

        predictions = np.mean(member_predictions, axis=0)

        test_predictions = pd.DataFrame(
            {"ID": test_df_real["ID"].to_numpy(), "Target": predictions}
        )

        sample_submission = pd.read_csv(raw_dir / "SampleSubmission.csv")
        submission = sample_submission[["ID"]].merge(
            test_predictions, on="ID", how="left", validate="one_to_one"
        )
        assert submission["Target"].notna().all(), "des IDs du gabarit n'ont pas de prédiction"
        assert len(submission) == len(sample_submission)

        submissions_dir = PROJECT_ROOT / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        submission_path = submissions_dir / "submission.csv"
        # `lineterminator="\n"` : pandas écrit sinon le retour à la ligne natif de l'OS (CRLF sur
        # Windows) alors que `SampleSubmission.csv` (fourni par Zindi) est en LF -- constaté être
        # une cause probable de rejet par la plateforme (2026-09-07). `float_format` à 2 décimales
        # (demande utilisateur du 2026-09-07) au lieu d'une précision à 17 chiffres significatifs,
        # inutile et non standard.
        submission.to_csv(submission_path, index=False, lineterminator="\n", float_format="%.2f")
        mlflow.log_artifact(str(submission_path))

        print(f"submission.csv : {submission.shape}, n_models={len(BAGGING_SEEDS)}")
        print(submission.head())


if __name__ == "__main__":
    main()
