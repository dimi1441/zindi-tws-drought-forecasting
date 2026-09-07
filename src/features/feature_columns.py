"""Source unique du jeu de 33 features modèle retenu (cf. JOURNAL.md 2026-09-07) : exclut les 5
colonnes de tendance long terme qui aident le MAE spatial mais le dégradent en temporel — le
schéma temporel étant celui qui ressemble au vrai test Zindi (mêmes cellules, mois futurs).

`configs/feature_columns.yaml` est régénéré automatiquement à 33 ou 38 colonnes par chaque run de
`src/features/pipeline.py` selon les features actives : tout code de modélisation doit passer par
`load_model_feature_columns` plutôt que lire ce YAML directement, pour ne pas répéter le bug du
2026-09-07 (une soumission avait été générée par erreur avec les 38 colonnes avant ce correctif).
"""

from pathlib import Path

import yaml

EXCLUDED_FEATURE_COLUMNS = {
    "TWS_t_climatology_count",
    "TWS_t_diff1_expanding_mean",
    "TWS_t_diff1_expanding_mean_count",
    "TWS_t_diff12_expanding_mean",
    "TWS_t_diff12_expanding_mean_count",
}


def load_model_feature_columns(config_dir: Path) -> list[str]:
    all_columns = yaml.safe_load((config_dir / "feature_columns.yaml").read_text())[
        "feature_columns"
    ]
    return [c for c in all_columns if c not in EXCLUDED_FEATURE_COLUMNS]
