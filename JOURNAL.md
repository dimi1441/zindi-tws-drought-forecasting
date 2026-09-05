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

## 2026-09-04 — Phase 1 : exploration

- `notebooks/01_exploration.ipynb` + `reports/data_understanding.md` produits et commités
  (8 sections : chargement, stats descriptives, valeurs manquantes, horizon effectif,
  couverture spatiale, saisonnalité, corrélations aux lags, synthèse), avec 7 figures dans
  `reports/`.
- Grille spatiale parfaitement stable entre train (2 154 021 lignes, 15 715 cellules, 138 mois
  2002-05→2015-08) et test (280 961 lignes, mêmes 15 715 cellules, 18 mois non contigus
  2015-09→2018-12). 1 157 cellules (7,4 %) ont une couverture temporelle incomplète en train —
  à surveiller pour les splits de la Phase 4.
- Pas de `NaN` explicite : le masquage de `TWS_t` (66,5 % des lignes test, conforme au brief)
  se traduit par une valeur substituée signalée uniquement par `TWS_t_masked`. Règle actée :
  ne jamais lire `TWS_t` en test sans vérifier ce booléen.
- **Horizon effectif mesuré sans fuite** (uniquement à partir de `TWS_t_masked` et `time`,
  jamais des valeurs de `TWS_t`), selon la formule du brief §2.4. Plage confirmée 1-7,
  moyenne ≈ 2,71, distribution fortement décroissante (`{1: 94048, 2: 62576, 3: 46777,
  4: 31076, 5: 15560, 6: 15479, 7: 15445}`). Cette distribution devra être reproduite telle
  quelle par le masquage augmenté de la Phase 3 (voir correction du 2026-09-04 ci-dessous :
  plusieurs épisodes de masquage par cellule, pas un cutoff unique).
- Cycles saisonniers en opposition de phase entre hémisphères nord/sud (~6 mois de décalage)
  → décision : la climatologie de la Phase 2 doit être calculée par cellule × mois calendaire,
  jamais globalement.
- Corrélations avec `target` : persistance pure `TWS_t` = 0,803 (confirme la pertinence de la
  baseline naïve), décroissance progressive jusqu'à lag12 = 0,371 (encore informatif → justifie
  une feature interannuelle `TWS_t − TWS_{t-12}`). Côté covariables, les fenêtres SPEI longues
  (12, 6 mois) sont plus corrélées que les courtes (1 mois) — cohérent avec l'inertie
  hydrologique du TWS.
- Écart constaté par rapport au processus décrit au brief §8.2 : ce journal n'avait pas été mis
  à jour à l'issue de la session Phase 1 (commit `ee9b6f0`) ; entrée ajoutée a posteriori le
  même jour pour combler le trou avant d'enchaîner sur la Phase 2.

## 2026-09-04 — Corrections sur la Phase 1 (masquage)

En répondant à des questions de compréhension sur le starter notebook et le masquage, deux
erreurs ont été trouvées dans les livrables Phase 1 et corrigées (notebook, rapport, mémoire
projet) :

- **`TWS_t` masqué = vrai `NaN`, pas une valeur substituée.** Le rapport affirmait à tort que le
  masquage produisait une valeur numérique de remplacement. Vérifié directement sur
  `data/raw/Test.csv` : 186 913 `NaN` sur `TWS_t`, exactement égal au nombre de lignes
  `TWS_t_masked == True`. L'erreur se voyait déjà dans la propre sortie du notebook (tableau
  `isna()` juste au-dessus de l'affirmation erronée) — non recoupée au moment de la rédaction.
- **Le masquage n'est pas cumulatif (pas de cutoff unique et définitif par cellule).** Analyse
  ajoutée dans `01_exploration.ipynb` §4 : 99,9 % des cellules (15 707 / 15 715) ont plusieurs
  segments masqués distincts sur les 18 mois de test (en moyenne ~4 trous séparés, max 5), avec
  retour à un `TWS_t` observé entre deux trous. Seules 8 cellules suivent le schéma « masqué une
  fois, masqué pour toujours ». Conforme au brief §2.3 (« séquences de trous »).
- Impact sur la Phase 3 : le masquage augmenté devra simuler **plusieurs épisodes de masquage
  par cellule** (chacun cumulatif localement), pas un unique cutoff terminal — sous peine de
  sous-représenter le pattern réel du test.
- Notebook ré-exécuté de bout en bout après correction (`jupyter nbconvert --execute --inplace`)
  pour garantir que les sorties affichées correspondent au code actuel.

## 2026-09-04 — Affinement : le masquage suit les vraies interruptions de la mission GRACE

Suite à une question sur la plausibilité historique des trous de données, analyse plus poussée
dans `01_exploration.ipynb` §4, qui affine (sans l'invalider) la correction précédente :

- **Le masquage opère par mois calendaire, pas par cellule indépendamment.** Le taux de masquage
  par mois de test est quasi binaire : ~0 % ou 99,6-100 %, jamais intermédiaire. 12 des 18 mois
  de test sont des mois "gap", 6 sont des mois "bons".
- **Confirmation côté train** : 22 mois sur 160 possibles sont totalement absents des lignes du
  train (pas `NaN`, la ligne n'existe pas). Ces mois coïncident avec des interruptions réelles et
  documentées de GRACE (mise en service 2002-2003, puis gaps liés à la dégradation des batteries
  à partir de 2011, de plus en plus fréquents). Le trou de 13 mois entre les deux derniers blocs
  du test correspond à la vraie transition GRACE → GRACE-FO (fin 2017 → mi-2018).
- **Décision pour la Phase 3** (remplace la recommandation précédente "plusieurs épisodes par
  cellule") :
  1. Masquer par **mois entier** (toutes les cellules d'un mois choisi, pas un tirage
     indépendant par cellule) — reproduit le vrai mécanisme.
  2. Calibrer la fréquence des gaps simulés sur les 22 mois absents du train, avec un taux
     **croissant dans le temps** (rare avant 2011, fréquent après) plutôt qu'un taux constant.
- Rapport et notebook mis à jour et ré-exécutés en conséquence.

### Prochaine étape

Phase 2 — pipeline de features (`src/features/`) : lags temporels (1, 3, 6, 12 mois),
climatologie cellule/mois + anomalies, feature d'horizon explicite
(`months_since_last_observed_tws`), voisinage spatial (3×3/5×5). Datasets enrichis versionnés
DVC en Parquet, stage `feature_engineering` dans `dvc.yaml`.

## 2026-09-05 — Phase 2 : pipeline de features

Implémenté en mode plan (validé par l'utilisateur avant codage) : `src/features/io.py`,
`temporal_lags.py`, `seasonal.py`, `target_month_encoding.py`, `horizon_features.py`,
`pipeline.py` + `configs/features.yaml`, `dvc.yaml` (nouveau, stage `feature_engineering`),
`tests/` (11 tests, tous passants), ADR `docs/decisions/0002-feature-pipeline-leakage-safety.md`.

- **Voisinage spatial retiré de cette phase**, reporté à la Phase 5 itération D (décision de
  l'utilisateur en cours de conception) — le brief prévoit déjà cette itération séparément de la
  baseline, donc ce n'est pas une perte, juste un séquencement "commencer simple, itérer" (§4.3).
- **Panel combiné train+test** (`io.build_cell_timeline`) : toutes les features temporelles
  regardent l'historique réel d'une cellule sur train ET test (mêmes 15 715 cellules), pas
  seulement les 18 mois épars du test — sans ça les lags de test auraient été en grande partie
  faux/trous. Détail dans l'ADR 0002.
- **Climatologie causale par construction** (`expanding().mean().shift(1)` par cellule × mois
  calendaire) plutôt que "recalculée fold par fold" : plus stricte, ne peut pas fuiter même sans
  machinerie de fold. Vérifié que `pandas.expanding().mean()` ignore nativement les `NaN` (mois
  absents ou masqués) sans les propager — pas de code spécial nécessaire, testé dans
  `tests/test_seasonal.py`.
- **Horizon vectorisé** (`.where(~masked)` + `.groupby(cellule).ffill()`) en remplacement de la
  boucle Python du notebook Phase 1 — testé en régression contre la distribution déjà validée à
  la main (`{1: 94048, ..., 7: 15445}`), résultat identique.
- **Feature ajoutée par rapport au brief** : `last_observed_tws` (pas seulement le compte de mois
  écoulés `months_since_last_observed_tws`, mais aussi la dernière vraie valeur connue) — permet
  une persistance depuis le dernier point connu plutôt que l'imputation par constante globale du
  starter des organisateurs.
- Correction en cours de route : `pd.concat` avec une colonne `target` toute-`NaN` sur test
  déclenchait un `FutureWarning` pandas (dtype inference dépréciée) — corrigé en utilisant
  `float("nan")` au lieu de `pd.NA`.
- `configs/feature_columns.yaml` marqué `cache: false` dans `dvc.yaml` (contrairement aux deux
  Parquet) pour rester un fichier Git normal, lisible en diff — utile pour suivre l'évolution des
  features dans le temps.
- Résultats : `train_features.parquet` (2 154 021 × 39), `test_features.parquet` (280 961 × 38),
  33 colonnes dans `feature_columns.yaml` (aucune de `lat`, `lon`, `ID`, `sample_id`).
  `dvc repro feature_engineering` reproductible depuis un état propre.

### Prochaine étape

Phase 3 — masquage augmenté (`src/features/mask_augmentation.py`) : simuler sur le train les
trous réels de la mission GRACE identifiés en Phase 1 (masquage par mois entier, pas par
cellule indépendamment ; fréquence croissante dans le temps, calibrée sur les 22 mois absents du
train). Une fois le train augmenté, réappliquer `horizon_features.add_horizon_features` dessus
(même fonction, juste un nom de colonne de masquage différent) pour obtenir une vraie
distribution d'horizons côté train, cohérente avec celle du test.

## 2026-09-05 — Phase 3 : masquage augmenté (mode dynamique uniquement)

Implémenté en mode plan (deux allers-retours de révision avec l'utilisateur avant codage — voir
décisions ci-dessous, qui divergent de ce que ce journal anticipait après la Phase 2).

- **Décision 1** : masquage par **mois calendaires entiers**, jamais par cellule indépendamment
  — confirme et applique la découverte de Phase 1 (le vrai masquage GRACE opère par mois, pas
  par cellule). Le texte du brief §4.2, qui décrivait un tirage indépendant par ligne/cellule, a
  été **corrigé** en conséquence (demande explicite de l'utilisateur : "corrige l'erreur du brief
  sur le chapitre cité").
- **Décision 2** (revirement par rapport au premier plan proposé) : **aucune version statique
  figée dans un Parquet** — l'utilisateur a rejeté la proposition initiale (masque tiré une fois,
  mis en cache sur disque comme les autres features Phase 2). Motif donné : un entraînement sur
  un masque unique figé ne généralise pas aussi bien qu'un entraînement exposé à plusieurs
  tirages différents. Mode dynamique uniquement : `pipeline.build_features(..., masking_config,
  rng)` recalcule tout en mémoire (lags, climatologie, horizon) à chaque appel, rien n'est
  persisté.
- **Décision 3** (clarification de portée) : la Phase 3 livre uniquement le **mécanisme
  réutilisable** (une fonction rejouable avec un `rng` différent à chaque appel) — la vraie
  boucle multi-tirages (ensemble de modèles, ou callback d'époque pour un modèle itératif type
  LSTM/TCN) est explicitement laissée à la Phase 4/5, qui n'existe pas encore. Confirmé après
  avoir proposé de construire la boucle maintenant et que l'utilisateur ait préféré respecter la
  séparation des phases du brief.
- **Détail architectural important** : la climatologie d'une ligne de test dépend de l'historique
  complet de la cellule (années antérieures, y compris train) — donc masquer le train affecte
  aussi les features de test (climatologie/anomalie, jamais `TWS_t`/`TWS_t_masked` du test qui
  restent le vrai masquage). Train ET test doivent donc être reconstruits ensemble à chaque
  tirage pour rester cohérents entre eux.
- `build_features(raw_dir, features_config)` (sans masquage) reste rétrocompatible — vérifié :
  `dvc repro feature_engineering` produit les mêmes fichiers qu'avant (seules les dépendances de
  `dvc.lock` ont changé, pas les sorties), les 11 tests Phase 2 restent verts.
- 7 nouveaux tests (`tests/test_mask_augmentation.py`) : isolation train/test, mois entier masqué
  pour toutes les cellules, `target` jamais modifié, reproductibilité par seed, deux seeds
  donnent des résultats différents, test d'intégration sur données réelles.
- Vérifié sur données réelles (2 seeds différentes) : distributions d'horizon différentes à
  chaque tirage, plage jusqu'à 13-37 mois (plus large que le 1-7 du test, car des mois masqués
  consécutifs peuvent s'accumuler sur plusieurs tirages — pas un problème, juste un effet
  secondaire du tirage aléatoire indépendant par mois).
- ADR `docs/decisions/0003-augmented-masking-mechanism.md`.

## 2026-09-05 — Correction de process : `dvc push` oublié depuis la Phase 2

Constat (question de l'utilisateur) : `dvc.yaml`/`dvc.lock` étaient bien tenus à jour et commités
à chaque fois, mais **`dvc push` n'avait jamais été exécuté depuis la Phase 2** — les CSV bruts
avaient été poussés vers le remote local en Phase 0, mais pas `train_features.parquet` ni
`test_features.parquet`. Concrètement, les commits `fffda00` (Phase 2) et `5f50c46` (Phase 3) ont
chacun ajouté/modifié ces outputs sans push correspondant vers
`C:\Users\user\Documents\PERSO\ZINDI\dvc-storage-drought-zindi`.

- Rattrapé le 2026-09-05 : `dvc push` exécuté (2 fichiers), `dvc data status --not-in-remote`
  confirme "No changes." depuis.
- Règle ajoutée dans `CLAUDE.md` (nouveau fichier) : tout commit qui touche un output DVC doit
  être suivi d'un `dvc push`, pas seulement d'un commit git de `dvc.lock`.

### Prochaine étape

Phase 4 — baseline et validation (`src/validation/`) : splits temporel (rolling/expanding-window)
et spatial (GroupKFold par cellule/bassin versant), métriques (MAE, RMSE, R²) par fold, baseline
de persistance + fallback climatologie, tracking MLflow. C'est là que sera construite la
première vraie boucle d'entraînement qui décidera comment exploiter le mécanisme de masquage
dynamique de la Phase 3 (combien de tirages, ensemble ou non).

## 2026-09-05 — Phase 4 : baseline et validation

Implémenté en mode plan (deux allers-retours de clarification avec l'utilisateur avant codage :
mécanique précise du découpage temporel `TimeSeriesSplit` vérifiée sur les vraies dates, et
confirmation que les folds ne servent qu'à estimer la performance, jamais à limiter les données du
modèle final).

- `src/validation/splits.py` : `temporal_splits` (fenêtre expansive, `TimeSeriesSplit` sur les
  mois réellement présents — vérifié : 5 folds de ~23 mois de validation chacun, de 2004-09 à
  2015-08), `spatial_splits` (`GroupKFold` par blocs de 10°, vérifié 282 blocs non vides sur les
  vraies données).
- `src/validation/metrics.py` : MAE/RMSE/R² par fold + agrégation moyenne ± écart-type.
- `src/validation/baselines.py` : persistance (`last_observed_tws` → fallback
  `TWS_t_climatology_mean`, aucun entraînement) et GBR simple (reprend `make_model()` du starter
  à l'identique, mêmes 8 features, pour comparaison directe avec la référence connue).
- `src/validation/run_baselines.py` : un seul tirage de masquage (Phase 3, seed=42) appliqué
  avant les splits — la validation voit donc un vrai mélange d'horizons, pas seulement
  l'horizon 1 dégénéré du train non masqué. 4 runs MLflow (2 baselines × 2 schémas), détail par
  fold exporté en CSV (`reports/fold_detail_*.csv`) et attaché comme artefact MLflow.
- **Résultats sur données réelles** (train masqué, seed=42) :

  | Baseline | Schéma | MAE moyen | RMSE moyen | R² moyen |
  |---|---|---|---|---|
  | Persistance+climato | temporel | 0,423 | 0,592 | 0,509 |
  | GBR simple | temporel | 0,421 | 0,576 | 0,546 |
  | Persistance+climato | spatial | 0,422 | 0,598 | 0,568 |
  | GBR simple | spatial | 0,413 | 0,562 | 0,619 |

  Cohérent avec la référence starter (MAE 0,429 non masqué) malgré une tâche plus dure (train
  masqué) — le GBR bat systématiquement la persistance, et le schéma spatial est plus facile que
  le temporel (interpoler spatialement vs extrapoler vers un futur jamais vu), un écart qui a du
  sens plutôt qu'un signal de surapprentissage.
- 11 nouveaux tests (`test_splits.py`, `test_metrics.py`, `test_baselines.py`), 29 au total, tous
  verts.

### Prochaine étape

Phase 5 — modélisation itérative : itérations A à G du brief (features simples → lags → 
climatologie/anomalies → voisinage spatial (Phase 5 aussi, différé depuis Phase 2) → masquage
augmenté (décider ici combien de tirages/ensemble, question laissée ouverte en Phase 3-4) →
covariables externes → autres modèles). SHAP et CodeCarbon dès qu'un modèle mérite analyse.

## 2026-09-05 — Phase 5, itération B+C : lags + climatologie + horizon

Scope choisi avec l'utilisateur parmi plusieurs options (voisinage spatial, ensemble de
masquage, ceci) : réutiliser le harnais de la Phase 4 tel quel, juste en remplaçant les 8
features simples du GBR par les 33 features déjà calculées en Phase 2 (`configs/
feature_columns.yaml`) — mesure directe du gain des lags/climatologie/horizon sans construire de
nouveau module.

- `src/validation/baselines.py` généralisé : `fit_predict_gbr(fit_df, val_df, feature_columns)`
  accepte n'importe quel jeu de features ; `fit_predict_simple_gbr` devient un cas particulier
  (garde le comportement/la signature d'avant, aucune régression sur les tests existants).
- `run_baselines.py` ajoute une 3e baseline `full_features_gbr` (mêmes hyperparamètres GBR,
  toutes les features Phase 2/3) à côté des 2 existantes, sur les mêmes 2 schémas de split.
- **Résultat sur données réelles** (même tirage de masquage seed=42) :

  | Baseline | Temporel (MAE / R²) | Spatial (MAE / R²) |
  |---|---|---|
  | Persistance+climato | 0,423 / 0,509 | 0,422 / 0,568 |
  | GBR simple (8 features) | 0,421 / 0,546 | 0,413 / 0,619 |
  | **GBR toutes features (33)** | **0,384 / 0,623** | **0,366 / 0,694** |

  Gain net et cohérent dans les deux schémas : **-9 % de MAE** (temporel) et **-11 %** (spatial)
  par rapport au GBR simple, R² en hausse de ~0,55→0,62 et ~0,62→0,69. Confirme que les lags,
  la climatologie et l'horizon (Phase 2/3) apportent une valeur réelle, pas seulement pour la
  cohérence anti-leakage mais pour la performance mesurée.
- 6 runs MLflow au total maintenant dans l'expérience `tws-forecasting` (3 baselines × 2
  schémas). Suite de tests inchangée (29 tests, tous verts).

### Prochaine étape

Phase 5 restante : voisinage spatial (itération D, le "levier majeur" du brief, toujours pas
implémenté), décision sur le nombre de tirages de masquage/ensemble (itération E), covariables
externes ERA5 si autorisation clarifiée (itération F), autres modèles (CatBoost/XGBoost,
itération G).

## 2026-09-05 — Phase 5 (suite) : LightGBM + early stopping calé sur nos folds

Question de l'utilisateur : le GBR actuel (`HistGradientBoostingRegressor`, `max_iter=300` fixe)
fait-il de l'early stopping ? Vérifié : `early_stopping="auto"` est le défaut sklearn (actif dès
que le train dépasse 10 000 lignes), mais il pioche un tirage **aléatoire** de 10 % de `fit_df`
pour sa validation interne — sans respecter la chronologie ni les blocs spatiaux. Testé sur le
dernier fold temporel : `n_iter_ = 300 = max_iter`, l'early stopping ne s'est jamais déclenché
(le modèle n'a peut-être pas fini de converger à 300 itérations).

- Ajout de `src/validation/splits.py::temporal_holdout`/`spatial_holdout` : découpe interne
  *one-shot* de `fit_df` (jamais `val_df`, qui resterait sinon utilisé à la fois pour arrêter
  l'entraînement et pour rapporter la métrique — biais optimiste). `temporal_holdout` réserve la
  dernière tranche de mois de `fit_df` ; `spatial_holdout` tire au sort une fraction des blocs de
  10°. Chacun branché sur le schéma de validation externe correspondant
  (`INNER_HOLDOUT_BY_SCHEME`).
- `baselines.py::fit_predict_lightgbm_early_stopping` : LightGBM (déjà dans
  `requirements.txt`, recommandé en premier par le brief §7.4 — plus que `HistGBR`, qui n'était
  qu'un choix de convenance du starter), jusqu'à 3000 itérations, arrêt anticipé après 50 rounds
  sans amélioration sur la validation interne. Mêmes hyperparamètres (max_depth, min_samples,
  L2) que le GBR pour isoler l'effet du changement.
- **Résultat sur données réelles** (même tirage de masquage seed=42) :

  | Modèle | Temporel (MAE / R²) | Spatial (MAE / R²) |
  |---|---|---|
  | GBR (300 itérations fixes) | 0,3838 / 0,623 | 0,3660 / 0,694 |
  | **LightGBM + early stopping** | 0,3836 / 0,623 (quasi identique) | **0,3492 / 0,720** (-4,7 % MAE) |

  Gain **dépendant du schéma** : en temporel, aucune différence réelle (300 itérations
  suffisaient déjà) ; en spatial, gain net. LightGBM+early-stopping ne fait jamais moins bien que
  le GBR fixe, et parfois nettement mieux — bon candidat pour devenir le modèle par défaut des
  prochaines itérations.
- **Ajout suite à une question de l'utilisateur** ("le temps d'entraînement est-il loggué ?") :
  non, seulement de façon implicite (MLflow enregistre `start_time`/`end_time` de chaque run,
  mais rien d'explicite, rien par fold). Ajouté `fit_predict_seconds` par fold (colonne dans
  `reports/fold_detail_*.csv`) + `fit_predict_seconds_total`/`_mean` en métriques MLflow. Pas un
  vrai suivi d'empreinte carbone (CodeCarbon, toujours prévu pour plus tard, brief §6.4), mais un
  point de comparaison chiffré en attendant.
- 4 nouveaux tests (`test_fit_predict_lightgbm_early_stopping_runs_and_never_trains_on_val_df`,
  `test_temporal_holdout_val_is_strictly_after_train`,
  `test_spatial_holdout_never_splits_a_block_and_is_disjoint`, + 1), 32 au total, tous verts.
- 8 runs MLflow au total dans `tws-forecasting` (4 baselines × 2 schémas).

## 2026-09-05 — Saut anticipé en Phase 7 : première soumission

Décision de l'utilisateur : sauter directement à la Phase 7 (soumission), revenir aux Phases 5
(reste des itérations, dont un modèle itératif LSTM/TCN) et 6 (analyse d'erreurs) plus tard.

- Discussion importante avant de coder : l'utilisateur a fait remarquer qu'un GBM (arbres) n'a
  pas de notion d'"époque" contrairement à un réseau de neurones — entraîner plusieurs modèles
  sur des tirages de masquage différents et moyenner leurs prédictions serait du **bagging**, pas
  "le masquage dynamique par époque" prévu au brief (§5), qui suppose un modèle itératif. Décision
  finale : **pas de bagging**, un seul tirage de masquage (seed=42, même seed que
  `run_baselines.py` pour rester comparable), LightGBM+early-stopping (déjà validé meilleur en
  Phase 5). Le vrai mécanisme dynamique par époque attendra un modèle itératif, en Phase 5.
- `src/generate_submission.py` (nouveau) : réutilise tel quel `build_features(...,
  masking_config, rng)` et `fit_predict_lightgbm_early_stopping(train_df, test_df,
  feature_columns, temporal_holdout, return_model=True)` — aucun nouveau mécanisme de
  modélisation, uniquement de l'orchestration. `return_model=True` ajouté à
  `fit_predict_lightgbm_early_stopping` (paramètre optionnel, défaut `False`, aucune régression)
  pour récupérer `model.booster_.best_iteration` à des fins de traçabilité.
- Bug corrigé en cours de route : `model.booster_.best_iteration_` (avec underscore final,
  convention scikit-learn) n'existe pas sur l'objet `Booster` de LightGBM — c'est
  `best_iteration` (sans underscore). Corrigé dans le script et la docstring.
- **Résultat** : `submissions/submission.csv` généré (280 961 lignes, colonnes `ID`/`Target`,
  même ordre d'IDs que `SampleSubmission.csv`, aucun NaN, jamais tout à zéro,
  `Target` ∈ [-2,09 ; 2,18], moyenne -0,11, écart-type 0,65 — cohérent avec l'échelle standardisée
  de `TWS_t` observée depuis la Phase 1). `best_iteration` LightGBM = 70 (sur ce tirage précis,
  arrêt bien avant la limite de 3000).
- Run MLflow `final_submission` loggué (seed, nb lignes train/test, `best_iteration`,
  `submission.csv` en artefact). Nouveau stage `dvc.yaml: generate_submission` (déterministe à
  seed fixée — `random_state=42` dans LightGBM).
- **Rappel explicite pour la suite** : cette soumission n'utilise ni le voisinage spatial
  (itération D, toujours différée), ni les covariables externes (itération F), ni un modèle
  itératif (itération G) — c'est un premier jalon déposable, pas la version finale. Priorité
  performance d'abord (règle #9 du brief) : à améliorer en revenant sur les Phases 5/6.

### Prochaine étape

Retour aux Phases 5/6 comme convenu : voisinage spatial (itération D), analyse d'erreurs par
horizon/zone/saison (Phase 6), puis modèle itératif LSTM/TCN (occasion d'exercer le vrai
masquage dynamique par époque, jamais encore testé faute de modèle adapté).

## 2026-09-05 — Bagging de GBR sur tirages de masquage différents (demande explicite utilisateur)

Contrairement au bagging évoqué puis écarté lors du saut en Phase 7 (qui visait à simuler le
"masquage dynamique par époque" du brief — jugé inadapté à un GBM sans notion d'époque),
l'utilisateur redemande ici du bagging pour lui-même, comme technique de réduction de variance
indépendante de cette question : plusieurs `HistGradientBoostingRegressor` (réplique exacte de
`make_gbr_pipeline()`), chacun entraîné sur 100 % du train mais avec un tirage de masquage
augmenté différent (mécanisme dynamique de la Phase 3, un `rng` différent par membre), prédictions
moyennées. Demande explicite : GBR uniquement (pas de mix avec LightGBM), seeds fixées avec soin
pour la reproductibilité.

- Clarifié avec l'utilisateur avant codage (`AskUserQuestion`) : 3 modèles, tous GBR (pas de mix
  LightGBM/GBR contrairement à l'idée initiale), seeds explicites **42/43/44** (42 = même premier
  tirage que `run_baselines.py`/l'ancienne soumission, pour rester comparable). Pas de bootstrap
  des lignes en plus — la seule source de diversité entre membres est le tirage de masquage
  (mois de trous différents), pas un sous-échantillonnage aléatoire des lignes.
- `src/validation/baselines.py::fit_predict_bagged_gbr` : moyenne les prédictions de N
  `fit_predict_gbr` déjà existants, chacun recevant une paire `(fit_df, val_df)` propre à son
  tirage de masquage — construites par l'appelant en appliquant le même masque
  temporel/spatial à chacun des N `train_df` obtenus via `build_features(..., rng=seed_i)` (les
  lignes/mois sont identiques quel que soit le tirage, seule `TWS_t` et les features dérivées
  diffèrent, donc les masques booléens de split se réutilisent tels quels).
- `src/validation/run_bagging.py` (nouveau) : reproduit le harnais de `run_baselines.py`
  (`temporal_splits`/`spatial_splits`, 5 folds chacun) pour rester directement comparable aux
  baselines déjà loggées.
- **Résultat sur données réelles** (seeds 42/43/44) :

  | Modèle | Temporel (MAE / R²) | Spatial (MAE / R²) |
  |---|---|---|
  | GBR simple, un seul tirage (33 features) | 0,384 / 0,623 | 0,366 / 0,694 |
  | LightGBM + early stopping, un seul tirage | 0,384 / 0,623 | 0,349 / 0,720 |
  | **Bagging 3×GBR (seeds 42/43/44)** | **0,372 ± 0,035 / 0,648** | **0,360 ± 0,016 / 0,704** |

  Le bagging bat le GBR seul sur les deux schémas (-3 % MAE temporel, -2 % spatial) et bat
  LightGBM en temporel, mais reste derrière LightGBM en spatial (0,360 contre 0,349) — le
  bagging de GBR n'est donc pas strictement le meilleur choix sur toute la ligne, mais reste un
  gain net par rapport au GBR simple. Piste ouverte pour une prochaine itération : appliquer le
  même bagging à LightGBM plutôt qu'au GBR, pour cumuler les deux gains.
- `src/generate_submission.py` réécrit pour utiliser ce bagging (3 GBR, seeds 42/43/44,
  réentraînés sur 100 % du train comme toujours pour la soumission finale) au lieu du LightGBM
  mono-tirage de la Phase 7. `submissions/submission.csv` régénéré (280 961 lignes, aucun NaN,
  aucun zéro, `Target` ∈ [-2,35 ; 2,34] — cohérent avec l'échelle standardisée de `TWS_t`).
- 1 nouveau test (`test_fit_predict_bagged_gbr_averages_individual_member_predictions`, vérifie
  que le bagging renvoie exactement la moyenne arithmétique des prédictions individuelles), 34 au
  total, tous verts.
- `dvc status` avait détecté le changement de deps (`baselines.py`, `generate_submission.py`) et
  de sortie (`submissions/submission.csv`) ; `dvc commit generate_submission -f` utilisé pour
  enregistrer l'état déjà produit et vérifié (plutôt que de relancer `dvc repro`, redondant vu que
  le script venait d'être exécuté avec les mêmes seeds) — `dvc status` confirme "Data and
  pipelines are up to date." ensuite. `dvc push` à faire dans la foulée de ce commit (règle
  `CLAUDE.md`).

### Prochaine étape

Toujours en attente (inchangé depuis la Phase 7) : voisinage spatial (itération D), analyse
d'erreurs (Phase 6), covariables externes (itération F), modèle itératif (itération G). Nouvelle
piste identifiée ce jour : bagging appliqué à LightGBM (au lieu du GBR) pour cumuler le gain du
bagging et celui déjà mesuré de LightGBM+early-stopping en spatial.
