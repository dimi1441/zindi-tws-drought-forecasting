"""Orchestration du pipeline de features (Phase 2).

Usage : `python -m src.features.pipeline`
"""

from pathlib import Path

import yaml

from src.features.horizon_features import add_horizon_features
from src.features.io import build_cell_timeline, load_raw
from src.features.mask_augmentation import apply_augmented_masking, select_gap_months
from src.features.seasonal import add_climatology_features
from src.features.target_month_encoding import add_target_month_encoding
from src.features.temporal_lags import add_lag_features
from src.features.trend import add_trend_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Colonnes de groupe/cible/housekeeping : jamais des features modèle (brief §3.2 + §9.1).
NON_FEATURE_COLUMNS = {"lat", "lon", "ID", "time", "target", "TWS_t_masked", "is_train"}


def build_features(
    raw_dir: Path,
    features_config: dict,
    masking_config: dict | None = None,
    rng=None,
) -> tuple[list, list, list]:
    """Construit les dataframes train/test enrichis et la liste des colonnes features légitimes.

    Sans `masking_config`/`rng` (défaut) : comportement Phase 2 inchangé, aucun masquage
    augmenté. Avec les deux fournis : insère un tirage de masquage augmenté (mois entiers, mode
    dynamique uniquement — voir `mask_augmentation.py`) avant de recalculer lags/climatologie/
    horizon. Rien n'est jamais mis en cache sur disque pour cette variante : chaque appel avec un
    `rng` différent doit être refait entièrement par l'appelant (Phase 4/5, pas encore écrite).
    """
    train, test = load_raw(raw_dir)
    panel = build_cell_timeline(train, test)

    if masking_config is not None:
        if rng is None:
            raise ValueError(
                "rng est requis quand masking_config est fourni (mode dynamique uniquement)."
            )
        existing_train_months = panel.loc[panel["is_train"], "time"]
        gap_months = select_gap_months(
            existing_train_months, masking_config["target_gap_rate_by_period"], rng
        )
        panel = apply_augmented_masking(panel, gap_months)

    panel = add_lag_features(panel, features_config)
    panel = add_trend_features(panel)
    panel = add_climatology_features(panel)
    panel = add_horizon_features(panel)
    panel = add_target_month_encoding(panel)

    train_out = panel.loc[panel["is_train"]].drop(columns=["is_train"]).reset_index(drop=True)
    test_out = (
        panel.loc[~panel["is_train"]]
        .drop(columns=["is_train", "target"])
        .reset_index(drop=True)
    )

    feature_columns = [c for c in train_out.columns if c not in NON_FEATURE_COLUMNS]
    return train_out, test_out, feature_columns


def main() -> None:
    base_config = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    features_config = yaml.safe_load((PROJECT_ROOT / "configs" / "features.yaml").read_text())

    raw_dir = PROJECT_ROOT / base_config["paths"]["raw_dir"]
    processed_dir = PROJECT_ROOT / base_config["paths"]["processed_dir"]

    train_out, test_out, feature_columns = build_features(raw_dir, features_config)

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_parquet(processed_dir / "train_features.parquet", index=False)
    test_out.to_parquet(processed_dir / "test_features.parquet", index=False)

    feature_columns_path = PROJECT_ROOT / "configs" / "feature_columns.yaml"
    feature_columns_path.write_text(
        yaml.safe_dump({"feature_columns": feature_columns}, sort_keys=False)
    )

    print(f"train_features.parquet : {train_out.shape}")
    print(f"test_features.parquet  : {test_out.shape}")
    print(f"feature_columns ({len(feature_columns)}) -> {feature_columns_path}")


if __name__ == "__main__":
    main()
