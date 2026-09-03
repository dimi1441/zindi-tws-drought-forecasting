# Journal de projet

Non livré au jury — mémoire de travail pour comprendre plus tard pourquoi telle décision a été
prise, et base d'écriture pour les 400 mots de la section Trustworthiness.

## 2026-09-03 — Phase 0 : mise en place

- Lecture du brief complet (`doc/PROJECT_BRIEF.md`), de l'énoncé (`doc/INFOS.md`) et du
  starter notebook. Le starter avait déjà été exécuté une fois par l'utilisateur : baseline
  HistGradientBoostingRegressor (features simples : TWS_t, month_sin/cos, SPEI x4,
  SOIL_MOISTURE_t) sur split temporel 80/20 → MAE 0.429 / RMSE 0.607 / R² 0.460, contre
  persistance MAE 0.455 / RMSE 0.662 / R² 0.358. Gain modeste (~6% MAE) : confirme qu'il y a
  de la marge, notamment via lags multiples, voisinage spatial et gestion explicite de
  l'horizon variable (leviers identifiés dans le brief comme majeurs).
- Décisions de mise en place actées avec l'utilisateur :
  - Repo GitHub **privé** créé maintenant (passage en public prévu au moment de la
    publication finale, cf. section 8 du brief).
  - Remote DVC **local** (dossier hors du repo Git, sur ce PC) — pas de dépendance à un
    compte cloud pour démarrer vite.
  - Dépendances installées dans un **`.venv`** dédié (pas d'installation globale), figées via
    `pip freeze` dans `requirements.txt` après installation initiale.
- Structure de dossiers créée conformément à la section 5 (Phase 0) du brief :
  `data/{raw,interim,processed}`, `notebooks/`, `src/{features,validation,models}`,
  `configs/`, `models/`, `reports/`, `docs/decisions/`, `submissions/`.
- Données brutes déplacées dans `data/raw/` ; notebook starter déplacé dans
  `notebooks/00_starter_baseline.ipynb` et chemins corrigés (`DATA_DIR`, `SUBMISSION_PATH`).
- Environnement : `lightgbm`, `dvc`, `mlflow`, `pandas`, `pyarrow`, `scikit-learn`, `hydra`
  étaient déjà disponibles ; `xgboost`, `catboost`, `shap`, `codecarbon`, `pytest`
  installés en plus dans le `.venv`.

### Prochaine étape

Phase 1 — exploration (`notebooks/01_exploration.ipynb`) : mesurer empiriquement la
distribution des horizons dans le test (essentiel pour la Phase 3, masquage augmenté),
cartographier couverture spatiale et manquants, analyser saisonnalité et corrélations à
différents lags.
