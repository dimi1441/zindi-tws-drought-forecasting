"""Importance par permutation (demande utilisateur du 2026-09-10 : certaines des 41 features
n'apportent peut-être rien, élaguer plutôt que deviner) : pour chaque feature, mélange ses valeurs
sur le jeu de validation et mesure de combien le MAE se dégrade -- une feature dont le mélange ne
change presque rien n'est pas vraiment utilisée par le modèle.

GBR simple (réglage par défaut actuel), même tirage de masquage que la recherche
d'hyperparamètres (seed=42, taux de base) -- moyenne sur les 5 folds temporels pour la robustesse.

Usage : `python -m src.validation.feature_importance`
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance

from src.features.feature_columns import load_model_feature_columns
from src.features.pipeline import build_features
from src.validation.baselines import make_gbr_pipeline
from src.validation.splits import temporal_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASKING_SEED = 42
N_REPEATS = 5


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    masking_config = features_config["masking"]
    feature_columns = load_model_feature_columns(PROJECT_ROOT / "configs")
    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    print("construction du panel d'entrainement (seed=42, taux de base)...")
    rng = np.random.default_rng(MASKING_SEED)
    train_df, _, _ = build_features(raw_dir, features_config, masking_config, rng)

    importances_by_fold = []
    for fold_idx, (train_mask, val_mask) in enumerate(temporal_splits(train_df), start=1):
        fit_df = train_df.loc[train_mask]
        val_df = train_df.loc[val_mask]
        model = make_gbr_pipeline()
        X_fit = fit_df[feature_columns].to_numpy(dtype=np.float32)
        y_fit = fit_df["target"].to_numpy(dtype=np.float32)
        X_val = val_df[feature_columns].to_numpy(dtype=np.float32)
        y_val = val_df["target"].to_numpy(dtype=np.float32)
        model.fit(X_fit, y_fit)

        result = permutation_importance(
            model,
            X_val,
            y_val,
            scoring="neg_mean_absolute_error",
            n_repeats=N_REPEATS,
            random_state=42,
            n_jobs=-1,
        )
        # importances_mean est la baisse de neg_MAE -> une degradation du MAE positive.
        importances_by_fold.append(result.importances_mean)
        print(f"fold {fold_idx}/5 done")

    importances = np.mean(importances_by_fold, axis=0)
    summary = pd.DataFrame({"feature": feature_columns, "mae_degradation": importances}).sort_values(
        "mae_degradation", ascending=False
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "permutation_importance.csv"
    summary.to_csv(out_path, index=False)

    print("\n" + summary.to_string(index=False))
    print(f"\nSaved -> {out_path}")

    near_zero_or_negative = summary[summary["mae_degradation"] <= 0.0005]
    print(f"\n{len(near_zero_or_negative)} features avec degradation <= 0.0005 (candidates a l'elagage) :")
    print(near_zero_or_negative["feature"].tolist())


if __name__ == "__main__":
    main()
