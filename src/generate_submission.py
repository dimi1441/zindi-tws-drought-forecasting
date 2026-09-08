"""Génère `submissions/submission.csv` : bagging de 7 GBR (`make_gbr_pipeline()`, décision
explicite de l'utilisateur du 2026-09-05, voir `src/validation/run_bagging.py`), chacun entraîné
sur 100% du train mais avec un tirage de masquage augmenté différent (seeds 43..49, mode
dynamique, Phase 3) — la diversité vient du masquage, pas d'un bootstrap des lignes ni du hasard
interne du modèle (`random_state=42` fixe sur chaque membre). Prédiction finale = moyenne des 7
GBR.

**N=8 choisi le 2026-09-07** via `run_bagging_curve.py` (courbe MAE vs nombre de membres,
schéma temporel, 15 tirages testés) : MAE 0.3838 (N=1) → 0.3725 (N=2) → 0.3717 (N=3, ancien
réglage) → plateau bruité autour de 0.368-0.369 dès N≈8, meilleur observé 0.3683 à N=8, aucun
gain net au-delà (N=15 : 0.3689, dans le bruit inter-folds ~0.037). Le schéma spatial n'a pas été
remesuré à N=8 (dernier chiffre connu à N=3 : 0.360, `reports/fold_detail_bagged_gbr_spatial.csv`)
— non prioritaire car le vrai test Zindi ressemble au schéma temporel, pas spatial.

**Seed 42 exclu le 2026-09-08** (`diagnose_bag_horizon_specialization.py`) : ce membre
(multiplicateur 0.5x, quasiment pas de masquage vu à l'entraînement) était nettement pire sur les
lignes horizon>1 (`TWS_t` masqué, 66.5% du vrai test -- MAE 0.504 contre 0.446-0.473 pour tous les
autres membres) sans aucun bénéfice compensatoire sur horizon=1 (tous les membres indiscernables,
~0.367). **Confirmé sur la plateforme Zindi : 0.75 -> 0.74** en excluant ce seul membre (bag à 8 ->
7 membres, `RATE_MULTIPLIERS` inchangé pour les 7 autres). Remonter le plancher de
`RATE_MULTIPLIERS` (au lieu d'exclure) a été testé (`compare_rate_ranges.py`) mais la comparaison
s'est révélée non fiable (même biais de validation-qui-devient-plus-dure que celui évité par
`diagnose_bag_horizon_specialization.py`) -- pas repris, l'exclusion simple reste la solution
retenue.

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

**Cache par membre (demande utilisateur du 2026-09-08)** : chaque modèle entraîné est sauvegardé
(`models/bag_member_seed{seed}_mult{multiplier}.joblib`, via `joblib`) ainsi que ses prédictions
sur le test réel (`reports/bag_member_predictions/seed{seed}_mult{multiplier}.csv`) -- clé sur
(seed, multiplicateur), donc invalidée automatiquement si l'un des deux change. Sert à recombiner
des variantes du bag (exclure un membre, tester une pondération) sans jamais réentraîner un
membre déjà vu -- avant ce cache, tester "exclure le seed 42" a nécessité de réentraîner les 7
autres membres depuis zéro (~10 min perdues).

Usage : `python -m src.generate_submission`
"""

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.feature_columns import load_model_feature_columns
from src.features.mask_augmentation import scale_gap_rate_by_period
from src.features.pipeline import build_features
from src.validation.baselines import make_gbr_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MEMBER_PREDICTIONS_DIR = PROJECT_ROOT / "reports" / "bag_member_predictions"
# Historique (avant le 2026-09-08) : les 8 seeds/multiplicateurs originaux, seed 42 = même
# premier tirage que le reste du projet, 43..49 = les 7 tirages suivants (N=8 retenu via
# `run_bagging_curve.py`). Multiplicateurs 0.5x-2x choisis par l'utilisateur (spread modéré,
# linéaire par position, pas tiré aléatoirement) pour faire varier la *difficulté* du masquage
# d'un membre à l'autre -- le train ne voyait jusque-là que ~10-16% de mois masqués contre 67%
# dans le vrai test. Gardés ici pour dériver exactement les mêmes valeurs pour les 7 seeds
# restants (pas un nouveau `linspace` sur 7 points, qui donnerait des multiplicateurs différents).
_ORIGINAL_BAGGING_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
_ORIGINAL_RATE_MULTIPLIERS = np.linspace(0.5, 2.0, len(_ORIGINAL_BAGGING_SEEDS))

# Production actuelle (depuis le 2026-09-08) : exclut le seed 42 (0.5x), cf. docstring.
BAGGING_SEEDS = _ORIGINAL_BAGGING_SEEDS[1:]
RATE_MULTIPLIERS = _ORIGINAL_RATE_MULTIPLIERS[1:]


def _member_cache_paths(seed: int, multiplier: float) -> tuple[Path, Path]:
    """Chemins de cache pour un membre du bag, clés sur (seed, multiplicateur) -- si l'un des
    deux change, le nom de fichier change aussi, donc pas de risque de réutiliser un modèle
    entraîné sous un régime de masquage différent (demande utilisateur du 2026-09-08 : sauvegarder
    les modèles testés pour ne plus jamais avoir à réentraîner un membre déjà vu)."""
    tag = f"seed{seed}_mult{multiplier:.3f}"
    model_path = MODELS_DIR / f"bag_member_{tag}.joblib"
    predictions_path = MEMBER_PREDICTIONS_DIR / f"{tag}.csv"
    return model_path, predictions_path


def generate_submission(
    bagging_seeds: list[int],
    rate_multipliers,
    output_filename: str,
    mlflow_run_name: str,
) -> Path:
    """Cœur réutilisable de la génération de soumission -- paramétré par la liste de seeds/
    multiplicateurs et le nom de fichier de sortie, pour permettre des variantes d'exploration
    (ex : `generate_submission_no42.py`) sans dupliquer la logique d'entraînement/format CSV.
    """
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    base_gap_rate_by_period = features_config["masking"]["target_gap_rate_by_period"]
    feature_columns = load_model_feature_columns(PROJECT_ROOT / "configs")

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    # Historique de train réel, non masqué : sert uniquement à calculer les features de test
    # (partagées par tous les membres du bag), jamais à l'entraînement. Voir la note de
    # correction ci-dessus.
    _, test_df_real, _ = build_features(raw_dir, features_config)

    with mlflow.start_run(run_name=mlflow_run_name):
        mlflow.log_param("masking_seeds", bagging_seeds)
        mlflow.log_param("rate_multipliers", list(rate_multipliers))
        mlflow.log_param("model", "bagged_gbr")
        mlflow.log_param("n_models", len(bagging_seeds))
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("test_features_masking", "none (real train history)")

        test_ids = test_df_real["ID"].to_numpy()
        X_test = test_df_real[feature_columns].to_numpy(dtype=np.float32)

        member_predictions = []
        n_train_rows = None
        for seed, multiplier in zip(bagging_seeds, rate_multipliers):
            model_path, predictions_path = _member_cache_paths(seed, multiplier)

            if predictions_path.exists():
                cached = pd.read_csv(predictions_path)
                assert (cached["ID"].to_numpy() == test_ids).all(), (
                    f"cache {predictions_path} ne correspond pas a l'ordre actuel de "
                    "test_df_real -- supprimer le cache et reentrainer"
                )
                print(
                    f"seed={seed} mult={multiplier:.3f}x : predictions en cache, reutilisees "
                    f"({predictions_path.name})"
                )
                member_predictions.append(cached["prediction"].to_numpy())
                continue

            rng = np.random.default_rng(seed)
            scaled_periods = scale_gap_rate_by_period(base_gap_rate_by_period, multiplier)
            masking_config = {"target_gap_rate_by_period": scaled_periods}
            train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)
            n_train_rows = len(train_df)

            model = make_gbr_pipeline()
            X_fit = train_df[feature_columns].to_numpy(dtype=np.float32)
            y_fit = train_df["target"].to_numpy(dtype=np.float32)
            model.fit(X_fit, y_fit)
            member_pred = model.predict(X_test)

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            MEMBER_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            pd.DataFrame({"ID": test_ids, "prediction": member_pred}).to_csv(
                predictions_path, index=False
            )
            print(f"seed={seed} mult={multiplier:.3f}x : entraine, modele -> {model_path.name}")

            member_predictions.append(member_pred)

        if n_train_rows is not None:
            mlflow.log_param("n_train_rows", n_train_rows)
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
        submission_path = submissions_dir / output_filename
        # `lineterminator="\n"` : pandas écrit sinon le retour à la ligne natif de l'OS (CRLF sur
        # Windows) alors que `SampleSubmission.csv` (fourni par Zindi) est en LF -- constaté être
        # une cause probable de rejet par la plateforme (2026-09-07). `float_format` à 2 décimales
        # (demande utilisateur du 2026-09-07) au lieu d'une précision à 17 chiffres significatifs,
        # inutile et non standard.
        submission.to_csv(submission_path, index=False, lineterminator="\n", float_format="%.2f")
        mlflow.log_artifact(str(submission_path))

        print(f"{output_filename} : {submission.shape}, n_models={len(bagging_seeds)}")
        print(submission.head())

        return submission_path


def main() -> None:
    generate_submission(BAGGING_SEEDS, RATE_MULTIPLIERS, "submission.csv", "final_submission")


if __name__ == "__main__":
    main()
