# TWS Forecasting Challenge (Zindi)

Prédiction à court terme (1 à 7 mois) du Total Water Storage (TWS) à l'échelle globale,
à partir d'observations GRACE/GRACE-FO et de covariables climatiques (SPEI, humidité du sol).

Voir [doc/INFOS.md](doc/INFOS.md) pour l'énoncé du challenge et
[doc/PROJECT_BRIEF.md](doc/PROJECT_BRIEF.md) pour la méthodologie complète suivie sur ce projet.

## Structure du projet

```
data/
  raw/         # Train.csv, Test.csv, SampleSubmission.csv (versionnés via DVC, pas Git)
  interim/     # données intermédiaires (Parquet)
  processed/   # jeux de features finaux (Parquet)
notebooks/     # exploration et prototypage
src/
  features/    # feature engineering (lags, voisinage spatial, saisonnalité, horizon)
  validation/  # splits temporel / spatial / combiné
  models/      # entraînement, inférence
configs/       # configs YAML (Hydra)
models/        # modèles entraînés sérialisés
reports/       # rapports d'analyse (data understanding, erreurs, biais)
submissions/   # fichiers de soumission générés
docs/decisions/ # Architecture Decision Records (ADR)
doc/           # documents fournis par l'organisateur + brief méthodologique
```

## Setup

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
dvc pull                                          # récupère les données depuis le remote local
```

## Règles anti-leakage (rappel)

- Seules les informations disponibles à `t` ou avant peuvent être utilisées pour prédire `t+1`.
- Exception : le mois calendaire de `t+1` (encodage cyclique) est autorisé.
- `lat`, `lon`, `ID` ne doivent jamais être des features du modèle — uniquement des clés de
  groupement (lags par cellule, GroupKFold spatial, voisinage).

Voir la checklist complète dans [doc/PROJECT_BRIEF.md](doc/PROJECT_BRIEF.md#35-checklist-anti-leakage-systématique).

## Baseline de référence

Persistance (`TWS_{t+1} ≈ TWS_t`) sur split temporel 80/20 :

| | MAE | RMSE | R² |
|---|---|---|---|
| Persistance | 0.455 | 0.662 | 0.358 |
| HistGBR (starter notebook) | 0.429 | 0.607 | 0.460 |

Tout modèle doit améliorer sensiblement cette baseline.
