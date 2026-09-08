"""Diagnostic (pas du code de production, pas branché à `generate_submission.py`) : les membres
du bag entraînés avec un masquage plus dur (`RATE_MULTIPLIERS` élevé, cf. `run_bagging.py`) sont-
ils réellement meilleurs sur les lignes où `TWS_t` est manquant (horizon>1), même si leur MAE
global ne s'est pas distingué (0.3704 vs 0.3683 en moyenne simple, cf. JOURNAL.md 2026-09-07) ?
Sert à décider si une pondération conditionnelle à l'horizon a quelque chose à exploiter avant
d'en écrire une.

Piège évité : chaque membre a son propre tirage de masquage, donc son propre découpage
horizon=1/horizon>1 sur SES lignes de validation — comparer les MAE de deux membres sur des
ensembles de lignes différents fausserait la comparaison. On utilise donc un panel de
**validation de référence unique** (masquage de base, multiplicateur 1x, seed dédié
`REFERENCE_SEED` jamais utilisé à l'entraînement) : mêmes lignes, même bucket horizon pour tous
les membres, seul le `fit_df` (et donc le modèle) varie -- exactement la même logique que
`test_df_real` en soumission finale (features de test calculées une fois, partagées par le bag).

Usage : `python -m src.validation.diagnose_bag_horizon_specialization`
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.feature_columns import load_model_feature_columns
from src.features.mask_augmentation import scale_gap_rate_by_period
from src.features.pipeline import build_features
from src.validation.baselines import fit_predict_gbr
from src.validation.splits import temporal_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Mêmes seeds/multiplicateurs que la production (`run_bagging.py`/`generate_submission.py`).
BAGGING_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
RATE_MULTIPLIERS = np.linspace(0.5, 2.0, len(BAGGING_SEEDS))

# Masquage de référence pour construire les buckets horizon de validation : taux de base
# (multiplicateur 1x), seed dédié distinct des 8 seeds de production et jamais utilisé pour
# entraîner un membre -- garantit que le découpage horizon=1/horizon>1 ne favorise aucun membre.
REFERENCE_SEED = 1000


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())
    base_gap_rate_by_period = features_config["masking"]["target_gap_rate_by_period"]
    feature_columns = load_model_feature_columns(PROJECT_ROOT / "configs")
    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]

    print("construction du panel de reference (validation, multiplicateur 1x)...")
    ref_rng = np.random.default_rng(REFERENCE_SEED)
    ref_df, _, _ = build_features(
        raw_dir,
        features_config,
        {"target_gap_rate_by_period": base_gap_rate_by_period},
        ref_rng,
    )

    member_train_dfs = []
    for seed, multiplier in zip(BAGGING_SEEDS, RATE_MULTIPLIERS):
        print(f"construction du panel d'entrainement seed={seed} multiplier={multiplier:.2f}x...")
        rng = np.random.default_rng(seed)
        scaled_periods = scale_gap_rate_by_period(base_gap_rate_by_period, multiplier)
        train_df, _, _ = build_features(
            raw_dir,
            features_config,
            {"target_gap_rate_by_period": scaled_periods},
            rng,
        )
        member_train_dfs.append(train_df)

    rows = []
    for fold_idx, (train_mask, val_mask) in enumerate(temporal_splits(ref_df), start=1):
        val_df = ref_df.loc[val_mask]  # memes lignes/horizon pour tous les membres
        y_true = val_df["target"].to_numpy()
        is_masked = val_df["TWS_t_masked"].to_numpy()
        n_h1, n_hgt1 = int((~is_masked).sum()), int(is_masked.sum())

        for seed, multiplier, train_full in zip(BAGGING_SEEDS, RATE_MULTIPLIERS, member_train_dfs):
            fit_df = train_full.loc[train_mask]
            y_pred = fit_predict_gbr(fit_df, val_df, feature_columns)
            abs_err = np.abs(y_pred - y_true)
            rows.append(
                {
                    "fold": fold_idx,
                    "seed": seed,
                    "multiplier": round(float(multiplier), 3),
                    "mae_horizon1": abs_err[~is_masked].mean() if n_h1 else float("nan"),
                    "n_horizon1": n_h1,
                    "mae_horizon_gt1": abs_err[is_masked].mean() if n_hgt1 else float("nan"),
                    "n_horizon_gt1": n_hgt1,
                }
            )
        print(f"fold {fold_idx}/5 done (horizon1={n_h1}, horizon>1={n_hgt1})")

    detail_df = pd.DataFrame(rows)
    summary = (
        detail_df.groupby(["seed", "multiplier"])[["mae_horizon1", "mae_horizon_gt1"]]
        .mean()
        .reset_index()
        .sort_values("multiplier")
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    detail_path = reports_dir / "bag_horizon_specialization_detail.csv"
    summary_path = reports_dir / "bag_horizon_specialization.csv"
    detail_df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nDetail -> {detail_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
