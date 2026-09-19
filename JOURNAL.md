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

### Prochaine étape (2026-09-05)

Toujours en attente (inchangé depuis la Phase 7) : voisinage spatial (itération D), analyse
d'erreurs (Phase 6), covariables externes (itération F), modèle itératif (itération G). Nouvelle
piste identifiée ce jour : bagging appliqué à LightGBM (au lieu du GBR) pour cumuler le gain du
bagging et celui déjà mesuré de LightGBM+early-stopping en spatial.

## 2026-09-06 — Correction du calcul des features de test à l'inférence

En expliquant à l'utilisateur comment se passe l'inférence sur le jeu de test, un bug a été
trouvé : `generate_submission.py` calculait les features dérivées de test (lags, climatologie,
horizon) à partir de l'historique de train *masqué* par le même tirage augmenté utilisé pour
entraîner chaque membre du bag — dégradant artificiellement les features de test avec des trous
fictifs, alors qu'à l'inférence l'historique réel et complet de train est disponible. Le masquage
augmenté est une technique d'entraînement (exposer le modèle à des trous simulés pour la
robustesse), pas une dégradation qui doit aussi s'appliquer aux features qu'on calcule pour la
vraie prédiction finale.

- Corrigé : les features de test sont maintenant calculées **une seule fois, sans aucun
  masquage** (`build_features(...)` sans `masking_config`/`rng`), partagées par les 3 membres du
  bag — seul l'entraînement (`train_df`) varie encore par tirage de masquage.
- **Impact mesuré** (comparaison ancien/nouveau `submission.csv`) : ~51 % des lignes de test ont
  changé de plus de 0,01 (écart absolu moyen 0,02, max 0,55 sur l'échelle standardisée ~[-2, 2])
  — un vrai changement, pas cosmétique.
- Le même schéma affecte probablement `run_baselines.py`/`run_bagging.py` (le `val_df` de la CV
  provient du même `train_df` masqué que `fit_df`) — délibérément laissé tel quel pour ne pas
  invalider les chiffres de comparaison déjà enregistrés ; seul le chemin de soumission finale a
  été corrigé. À revisiter si les chiffres de CV doivent être fiables au chiffre près plus tard.

## 2026-09-07 — Nouvelles features : tendance de long terme + comptes de fiabilité

Discussion approfondie avec l'utilisateur sur la nature des features existantes (lags = point
précis, rolling mean = fenêtre fixe, climatologie = seule feature à fenêtre croissante/tout
l'historique). Proposition retenue : ajouter deux variables de tendance de long terme, cumulées
sur tout l'historique de la cellule (comme la climatologie mais sans regroupement par mois
calendaire), appliquées aux variations plutôt qu'au niveau brut (pour éviter la redondance avec la
climatologie) — `TWS_t_diff1_expanding_mean` et `TWS_t_diff12_expanding_mean`. Plus, sur suggestion
de l'utilisateur : un compte `_count` pour chaque moyenne à fenêtre variable (dont
`TWS_t_climatology_count`, rétrofitté), pour que le modèle puisse discount une moyenne calculée
sur peu de données.

- Nouveau module `src/features/trend.py`, `seasonal.py` étendu (`TWS_t_climatology_count`),
  câblé dans `pipeline.py` juste après `add_lag_features`. 33 → 38 features
  (`configs/feature_columns.yaml`, régénéré automatiquement à chaque run du pipeline — **donc
  tout code qui le lit doit explicitement décider s'il veut les 5 nouvelles colonnes ou non**).
- 8 nouveaux tests (`test_trend.py` + extension `test_seasonal.py`), 38 au total, tous verts.
- **Résultat 5-fold CV réel (les 3 familles de modèles)** :

  | Modèle | Temporel 33→38 | Spatial 33→38 |
  |---|---|---|
  | GBR toutes features | 0,384 → 0,387 (pire) | 0,366 → 0,361 (mieux) |
  | LightGBM + early stopping | 0,384 → 0,386 (pire) | 0,349 → **0,339** (mieux, meilleur spatial à ce jour) |
  | Bag de 3 GBR | **0,372** → 0,377 (pire) | 0,360 → 0,355 (mieux) |

  Motif cohérent partout : les nouvelles features aident systématiquement en spatial, nuisent
  légèrement en temporel. **Décision : ne pas les utiliser pour la soumission finale** — le vrai
  test Zindi ressemble beaucoup plus au schéma temporel (mêmes cellules, mois futurs) qu'au
  spatial, et c'est justement le schéma où elles nuisent. Gardées pour une future itération
  orientée spatial.
- **Bug causé par cette régénération automatique** : `generate_submission.py` lisait
  `feature_columns` directement depuis `configs/feature_columns.yaml` sans filtre — une
  soumission a été générée par erreur avec le bag à 38 features (0,377) sans que ce choix soit
  délibéré. Corrigé avec un `EXCLUDED_FEATURE_COLUMNS` explicite dans `generate_submission.py`.

## 2026-09-06/07 — ANN avec masquage par époque (résultat négatif, mais instructif)

L'utilisateur voulait exercer le vrai mécanisme de masquage dynamique par époque du brief (§4.2),
qui suppose un modèle itératif — jamais fait jusqu'ici (GBM n'a pas de notion d'époque, cf.
bagging du 2026-09-05). Planifié d'abord pour un LSTM (voir plan approuvé via EnterPlanMode :
`nn.LSTM` causal uniquement, recalcul tensoriel des features par époque pour éviter de rejouer le
pipeline pandas 50-150+ fois, holdout à deux niveaux) — **bloqué sur l'installation de `torch`** :
l'index de roues CUDA de `download.pytorch.org` est joignable pour une requête rapide mais expire
sur la récupération complète de l'index par pip, 2 sessions/tentatives séparées. Une seule
tentative fraîche par session, conformément à la règle de retry établie en Phase 0 — abandonné
sans boucler.

Architecture repensée à deux reprises avec l'utilisateur avant de coder quoi que ce soit :
1. **LSTM → ANN simple** : TWS est un processus lisse/saisonnier où les features déjà construites
   (lags/climatologie) captent déjà la majorité du signal mesurable (gain Phase 5 itération B+C) —
   un ANN memoryless + features explicites isole proprement "est-ce que la diversité de masquage
   par époque aide" de "est-ce qu'une architecture séquentielle aide", et a une surface de risque
   de fuite bien plus petite qu'un LSTM (aucune récurrence/padding à sécuriser).
2. **PyTorch → `sklearn.neural_network.MLPRegressor`** : `torch` toujours pas installé ;
   `partial_fit()` correspond exactement au besoin (une itération de plus par appel) sans boucle
   d'entraînement custom.

Implémentation réelle (pas un script jetable) dans `src/models/ann/{feature_tensors,
dynamic_features,train}.py` :
- `feature_tensors.py` : scaffold statique `(n_cells, T_max)` par cellule, calculé une seule fois,
  avec un index de bucket climatologique précalculé pour vectoriser le regroupement par mois sans
  `groupby` pandas.
- `dynamic_features.py` : réimplémentation tensorielle de toutes les features dérivées de
  `TWS_t`, chacune testée en **parité exacte** contre la fonction pandas correspondante
  (`tests/test_ann_dynamic_features.py`) — plus un test de **causalité au niveau des features**
  (perturber le masquage d'un mois tardif ne doit rien changer aux lignes antérieures d'une
  cellule), la vraie preuve anti-fuite pour ce nouveau code (un modèle memoryless n'a de risque de
  fuite que dans le recalcul de features, jamais dans le réseau).
- `train.py` : holdout à deux niveaux, prétraitement (`SimpleImputer` + `StandardScaler`) ajusté
  **une seule fois** sur la vue réelle non masquée, jamais par époque — même principe que la
  correction du 06/09.
- 13 nouveaux tests (`test_ann_*.py`), 48 au total, tous verts.

**Résultat sur données réelles, un seul split temporel (pas encore de 5-fold)** :

| Configuration | MAE | Meilleur epoch |
|---|---|---|
| 33 features, LR=1e-3, masque variable | 0,438 | 2/23 |
| 38 features, LR=1e-3, masque variable | 0,440 | 2/23 |
| 33 features, LR=1e-3, masque **fixe** (diagnostic) | 0,438 | 2/23 |
| 33 features, **LR=1e-4**, masque variable | 0,437 | 14/45 |

Diagnostic mené avec l'utilisateur : le meilleur epoch trouvé dès l'époque 2 suggérait une
instabilité d'entraînement. Un masquage fixe (comme le GBR) donne un résultat quasi identique —
**élimine le masquage par époque comme cause**. Un taux d'apprentissage 10× plus faible corrige
bien la stabilité (meilleur epoch repoussé à 14 au lieu de 2) mais ne change presque rien au score
final. **Conclusion : l'ANN plafonne bien au-dessus de tous les GBM testés** (meilleur GBM : 0,339
spatial / 0,372 temporel) — cohérent avec le fait bien documenté que le gradient boosting domine
généralement les MLP simples sur données tabulaires. La diversité de masquage par époque, en soi,
ne compense pas ce désavantage structurel. Le bag de GBR reste le meilleur modèle. LSTM/TCN pas
abandonné, juste déprioritisé.

## 2026-09-07 — Correction du format de soumission (rejet par la plateforme Zindi)

L'utilisateur signale que sa soumission est rejetée par Zindi ; quelqu'un dans le chat de la
compétition suggère un problème de connexion. Vérification directe (diff des fichiers plutôt que
suppositions) contre `data/raw/SampleSubmission.csv` : deux différences réelles trouvées, aucune
liée à un problème de connexion — (1) `submission.csv` était écrit avec des fins de ligne Windows
CRLF (`\r\n`, comportement par défaut de pandas `to_csv` sous Windows) alors que le fichier
Zindi utilise du LF (`\n`) pur — cause classique de rejet par un validateur d'upload strict ; (2)
précision à 17 chiffres significatifs (défaut pandas) contre des entiers simples dans l'exemple —
pas forcément fatal mais inutilement volumineux.

- Corrigé avec `to_csv(..., lineterminator="\n", float_format="%.6f")` dans
  `generate_submission.py`. **Confirmé par l'utilisateur : le nouvel upload a fonctionné.**
- Fichier `submissions/submission_2dp.csv` généré à la demande (précision à 2 décimales), en plus
  du fichier principal à 6 décimales — pour réduire davantage la taille si nécessaire.
- Nouveau `scripts/submit_to_zindi.py` : soumission directe via l'API Zindi avec un timeout
  configurable (repli en cas de connexion instable). Endpoint/contrat vérifiés contre le code
  source d'un client Zindi tiers non officiel (`github.com/KameniAlexNea/zindi`), pas la
  documentation officielle Zindi — signalé comme tel dans le script.

## Phase 5, bug de colonnes de features propagé + courbe de bagging (N=8 retenu) — 2026-09-07

En discutant avec l'utilisateur de combien de GBR utiliser dans le bag (au-delà des 3 fixés le
2026-09-05), deux choses trouvées avant de coder l'expérience :

**Bug trouvé — même piège que le 2026-09-07 (soumission à 38 features), pas encore corrigé
partout** : `run_baselines.py` et `run_bagging.py` lisaient `configs/feature_columns.yaml`
directement, sans filtrer les 5 colonnes de tendance long terme — alors que ce fichier est
régénéré à 38 colonnes par le dernier run de `pipeline.py`. Seul `generate_submission.py` avait
été corrigé ce jour-là. Si `run_bagging.py` avait été relancé tel quel, il aurait silencieusement
utilisé 38 features au lieu des 33 retenues, invalidant toute comparaison avec les chiffres déjà
enregistrés (0.372/0.360). **Corrigé en centralisant** : nouveau `src/features/feature_columns.py`
(`EXCLUDED_FEATURE_COLUMNS` + `load_model_feature_columns()`), utilisé maintenant par les 3
scripts (`generate_submission.py`, `run_baselines.py`, `run_bagging.py`) — plus jamais de lecture
directe du YAML pour construire le jeu de features modèle. 48 tests toujours verts après ce
refactor.

**Stratégie retenue pour choisir N** (au lieu d'un nombre arbitraire) : nouveau
`src/validation/run_bagging_curve.py` — entraîne un pool de 15 GBR une seule fois par fold (un par
tirage de masquage, seeds 42..56) sur le schéma **temporel** uniquement (celui qui ressemble au
vrai test Zindi — le spatial n'a pas été dupliqué pour limiter le coût de calcul), puis évalue la
moyenne cumulative des N premiers membres pour N=1..15 à partir des mêmes prédictions (pas de
réentraînement par valeur de N). Ajout de `fit_predict_bagged_gbr_members()` dans `baselines.py`
pour exposer les prédictions individuelles (`fit_predict_bagged_gbr` refactorisé dessus, zéro
régression, confirmé par le test existant).

**Résultat réel (5-fold CV, schéma temporel)** :

| N membres | MAE moyen | Écart-type inter-folds |
|---|---|---|
| 1 | 0,3838 | 0,049 |
| 2 | 0,3725 | 0,039 |
| 3 (ancien réglage) | 0,3717 | 0,039 |
| 5 | 0,3699 | 0,038 |
| 8 | **0,3683** (meilleur) | 0,036 |
| 15 | 0,3689 | 0,037 |

N=3 reproduit exactement le 0,372 déjà connu (bonne vérification croisée que le fix du bug de
colonnes n'a rien cassé). Motif : vrai saut 1→2 membres (-3%), amélioration continue mais faible
jusqu'à N≈8, puis plateau bruité jusqu'à N=15 (aucune tendance claire) — **l'écart-type
inter-folds (~0,037) est ~10× plus grand que l'écart entre N=3 et N=8 (0,0034)**, cohérent avec
l'hypothèse de départ : la seule source de diversité entre membres est le tirage de masquage (pas
de bootstrap des lignes), donc corrélation plus forte entre membres et plateau plus rapide qu'un
bagging classique.

**Décision (choix utilisateur parmi 3 options proposées)** : **N=8** — meilleur MAE observé, coût
de calcul encore trivial pour une génération de soumission ponctuelle (pas un job répété), malgré
un gain marginal vs N=3 proche du bruit de mesure. `BAGGING_SEEDS` mis à jour à `[42..49]` dans
`generate_submission.py` et `run_bagging.py`. **Confirmé par un rerun officiel de `run_bagging.py` avec N=8** (mêmes seeds, harnais complet
5-fold, les deux schémas) : MAE 0,3683 temporel (identique au chiffre de la courbe — bonne
vérification croisée) / 0,3576 spatial (léger mieux que le 0,360 à N=3, cohérent avec la
tendance générale du bagging).

### Prochaine étape (2026-09-07)

Voisinage spatial (itération D), analyse d'erreurs (Phase 6), covariables externes (itération F).
`run_baselines.py`/`run_bagging.py` pourraient bénéficier de la même correction d'inférence que
`generate_submission.py` (val_df calculé sur train masqué) si des chiffres de CV plus précis sont
nécessaires plus tard. Nouvelle soumission (N=8) générée, committée et poussée (DVC) — format à 2
décimales sur demande explicite de l'utilisateur, pas de re-vérification byte-for-byte cette fois
(la correction CRLF de l'incident précédent reste, elle, bien en place dans le code). Pas encore
confirmé par un upload Zindi réussi à ce stade.

## `torch` installé (CPU) — 2026-09-07, débloque l'itération G

Après deux échecs (Phase 0-adjacent, puis à nouveau lors de l'ANN du 2026-09-06/07 — l'index de
roues CUDA de `download.pytorch.org` reachable en requête rapide mais timeout sur la récupération
complète par pip), tentative avec l'**index CPU-only** (`--index-url
https://download.pytorch.org/whl/cpu`, beaucoup plus petit que l'index CUDA complet) au lieu du
défaut : réussie du premier coup, une seule tentative bornée conformément à
[[feedback-retry-limits]]. `torch==2.14.0+cpu` installé et importable (`torch.cuda.is_available()
== False`, attendu — pas de GPU dédié nécessaire pour ce projet). Figé dans `requirements.in`
(avec commentaire sur l'index spécial requis, absent de PyPI standard) et `requirements.txt`
(régénéré via `pip freeze`, diff propre : +torch, +sympy, +mpmath uniquement).

**Débloque** : le LSTM/TCN de l'itération G (roadmap), déprioritisé deux fois faute de cet
install — pas encore commencé, juste rendu possible à nouveau.

## Score Zindi confirmé : 0,75 (bag de 8 GBR, moyenne simple, taux fixe) — 2026-09-08

Premier retour réel de la plateforme sur la soumission générée le 2026-09-07 (N=8, seeds 42-49,
taux de masquage fixe — avant l'essai de taux variable par membre). **Score 0,75** — métrique
Zindi où **plus bas = mieux** (confirmé par l'utilisateur, corrigé après une première lecture
erronée de ma part qui supposait l'inverse). Donc **moins bon** que le repère indicatif du brief
(~0,70, "atteint honnêtement avec ERA5 et sans features interdites", §10) — il reste de la marge
avant d'égaler ce repère, qui utilise en plus des covariables externes qu'on n'utilise pas encore
(itération F). Métrique Zindi exacte non documentée dans le brief (juste "non spécifiée") — pas
de lien direct établi avec nos MAE de CV (0,368 temporel), mais la direction d'amélioration est
maintenant claire : chercher à faire **baisser** ce score.

## Diagnostic : spécialisation horizon des membres du bag (taux variable) — 2026-09-08

Suite à l'observation que le train ne voit que ~10-16% de mois masqués contre 67% du vrai test
(cf. plus haut), et à l'essai de `RATE_MULTIPLIERS` (0,5x-2x par membre, testé le 2026-09-07 :
MAE quasi identique en moyenne simple, 0,3704 vs 0,3683 taux fixe — dans le bruit). Avant de
construire une pondération conditionnelle par ligne (piste discutée avec l'utilisateur), test de
rentabilité : `src/validation/diagnose_bag_horizon_specialization.py` (diagnostic seul, pas
branché à la production) — entraîne les 8 membres (mêmes seeds/multiplicateurs) mais évalue tous
sur un **même** panel de validation de référence (masquage à taux de base, seed 1000 dédié,
jamais utilisé à l'entraînement) pour comparer les membres sur exactement les mêmes lignes/
horizon, comme au vrai inférence.

**Résultat réel (5 folds temporels, ~312k lignes horizon=1 / ~312k lignes horizon>1 au total,
horizon>1 concentré sur les folds 1/4/5)** :

| Membre (multiplicateur) | MAE horizon=1 | MAE horizon>1 |
|---|---|---|
| 42 (0,50x) | 0,3665 | **0,5041** |
| 43 (0,71x) | 0,3670 | 0,4729 |
| 44 (0,93x) | 0,3672 | 0,4540 |
| 45 (1,14x) | 0,3668 | 0,4565 |
| 46 (1,36x) | 0,3674 | 0,4464 |
| 47 (1,57x) | 0,3663 | 0,4619 |
| 48 (1,79x) | 0,3670 | 0,4485 |
| 49 (2,00x) | 0,3679 | 0,4513 |

Sur horizon=1 (`TWS_t` disponible) : tous les membres indiscernables (spread 0,0016) — le
multiplicateur ne change rien ici. Sur horizon>1 (`TWS_t` masqué, **66,5% du vrai test** d'après
la Phase 1) : **effet de seuil, pas un dégradé continu** — le membre à 0,5x (quasiment aucune
exposition au masquage à l'entraînement) est nettement pire (0,504) que tous les autres
(0,446-0,473), mais au-delà d'environ 0,9-1x aucune tendance claire à continuer de s'améliorer
(46 à 1,36x légèrement meilleur que 49 à 2x — différence dans le bruit).

**Décision** : pas besoin de pondération conditionnelle par ligne (complexité non justifiée) —
le vrai problème est plus simple, le bas de la plage actuelle (0,5x) est nocif sans bénéfice
compensatoire. Prochaine étape proposée à l'utilisateur : remonter le plancher de
`RATE_MULTIPLIERS` (ex. 1x-2x au lieu de 0,5x-2x) et refaire le CV officiel — pas encore fait,
en attente de confirmation.

## Bag à 7 membres (sans seed 42) : 0,75 → 0,74 sur Zindi — 2026-09-08

Action immédiate suite au diagnostic ci-dessus (plutôt que de refaire la courbe complète ou
remonter `RATE_MULTIPLIERS`) : `src/generate_submission_no42.py` (nouveau, réutilise
`generate_submission()` refactorisée en fonction paramétrable dans `generate_submission.py`) —
bag à 7 membres, exclut simplement le seed 42 (0,5x), garde les 7 autres seeds/multiplicateurs
inchangés. **Confirmé sur la plateforme Zindi : 0,74 (mieux que 0,75)** — première confirmation
externe que le diagnostic horizon/multiplicateur reflète un vrai effet, pas un artefact de CV.

**Cache par membre ajouté à cette occasion** (demande utilisateur, car le premier test "excluons
le seed 42" a nécessité de réentraîner les 7 autres membres depuis zéro, ~19 min perdues) :
`generate_submission()` sauvegarde maintenant chaque modèle entraîné (`models/bag_member_seed{
seed}_mult{multiplier}.joblib`, via `joblib`) et ses prédictions sur le test réel (`reports/
bag_member_predictions/`), clé sur (seed, multiplicateur) -- toute future variante (exclure/
inclure/repondérer des membres) recombine ces caches au lieu de réentraîner. `models/` déjà
gitignoré (`models/*` sauf `.gitkeep`) en anticipation de ce genre d'artefact -- à suivre via DVC
comme `data/processed/` une fois qu'il contient des fichiers réels (pas encore fait : le run du
2026-09-08 qui a produit `submission_no42.csv` utilisait encore l'ancien code sans cache, lancé
avant ce correctif).

**Prochaine étape** : le plancher de `RATE_MULTIPLIERS` (1x-2x au lieu de 0,5x-2x) reste à tester
-- pourrait capturer un gain similaire à "juste exclure le 0,5x" sans réduire le bag à 7 membres.
`submission.csv` (officielle) n'a pas encore été remplacée par la version à 7 membres -- à décider
avec l'utilisateur.

## Essai de remontée du plancher `RATE_MULTIPLIERS` : résultat non fiable — 2026-09-08

Suite à la demande utilisateur de tester 1x-2x et 1x-3x (au lieu d'exclure le seed 42) :
`src/validation/compare_rate_ranges.py` (nouveau), même principe que `run_bagging.py` mais boucle
sur plusieurs plages de multiplicateurs pour les mêmes 8 seeds.

**Résultat brut (5-fold CV temporel)** : 0,5x-2x → 0,3704 (actuel) ; 1x-2x → 0,3729 (pire) ; 1x-3x
→ 0,3821 (encore pire). En apparence, remonter le plancher serait contre-productif.

**Mais ce résultat n'est pas fiable** — trouvé en regardant le détail par fold : les folds 1-3
bougent à peine (0,3545→0,3555 etc.) mais les folds 4-5 (les plus tardifs, années à fort taux de
base) se dégradent nettement (0,386→0,414 et 0,431→0,459). Cause : `compare_rate_ranges.py`
réutilise le schéma de `run_bagging.py` où **les lignes de validation de chaque fold sont
elles-mêmes masquées selon le multiplicateur du membre** — donc augmenter le multiplicateur rend
aussi l'examen de validation plus dur en même temps que l'entraînement, exactement le biais que
`diagnose_bag_horizon_specialization.py` avait été conçu pour éviter (masquage de référence fixe,
partagé entre tous les membres). Le vrai test Zindi, lui, a une difficulté fixe (66,5% masqué) —
ne devient pas plus dur selon notre réglage d'entraînement.

**Décision** : ne pas refaire cette comparaison (retour sur investissement incertain, ~1h de
calcul pour un résultat qu'il faudrait de toute façon re-vérifier avec la bonne méthode). Le vrai
signal fiable dont on dispose reste le diagnostic à jeu de validation partagé (membres stables de
~0,9x à 2x, seul 0,5x nuisible) et la confirmation Zindi (0,75→0,74 en excluant le 0,5x). On s'en
tient à l'exclusion simple plutôt que de pousser la piste "remonter le plancher".

## `submission.csv` officielle : bag à 7 membres (sans seed 42) — 2026-09-08

Sur demande explicite de l'utilisateur, `submission.csv` (suivi DVC) est remplacé par le contenu
déjà généré et confirmé sur Zindi (`submission_no42.csv`, identique bit à bit — pas de
réentraînement, juste promu). `src/generate_submission.py` mis à jour : `BAGGING_SEEDS`/
`RATE_MULTIPLIERS` de production dérivent maintenant des 8 valeurs originales moins le premier
élément (`_ORIGINAL_BAGGING_SEEDS[1:]`), pas un nouveau `linspace` sur 7 points (qui aurait donné
des multiplicateurs différents de ceux déjà testés/confirmés). `src/generate_submission_no42.py`
supprimé (son `BAGGING_SEEDS[1:]` aurait maintenant coupé le seed 43 au lieu du 42 -- devenu faux
depuis que la production exclut déjà 42). `dvc commit -f` sur `generate_submission` et
`feature_engineering` (dep `src/features` modifié par les fichiers de cette session), puis
`dvc push`.

## Phase 5 itération D : voisinage spatial (premier essai) — 2026-09-09

Reprise de l'itération D (différée depuis la Phase 2, "levier majeur" du brief §4) après une
discussion approfondie avec l'utilisateur sur comment l'implémenter correctement pour CE dataset
précis, pas la proposition littérale du brief. Trois désaccords/corrections successifs avec la
proposition initiale, tous par l'utilisateur ou déclenchés par une vérification directe des
données :

1. **Pas un carré fixe 3x3/5x5** (proposition du brief) : vérifié directement sur `Train.csv` --
   grille irrégulière (140 latitudes espacées de 1°, mais 358 longitudes avec des trous ponctuels
   à 3°) et sparse (15 715 cellules sur 50 120 possibles, 31% peuplé -- probablement terres
   émergées seulement). Un décalage fixe en degrés manquerait souvent le vrai voisin, surtout
   près des côtes. **Remplacé par** : le voisin direct = la cellule existante la plus proche dans
   chaque direction cardinale (même longitude pour nord/sud, même latitude pour est/ouest) --
   `build_direct_neighbor_lookup`, statique (ne dépend pas du temps).

2. **Colonnes séparées par direction, pas une moyenne** (intuition utilisateur : l'eau ruisselle,
   donc une direction peut être structurellement plus informative qu'une autre -- une moyenne
   imposerait une symétrie a priori qui empêcherait un GBR/LightGBM de découvrir cette asymétrie
   par lui-même).

3. **Dernière valeur connue du voisin, jamais sa valeur brute à `t`** (correction utilisateur
   déterminante) : le masquage réel est quasi-global par mois (Phase 1 -- chaque mois est masqué
   à ~0% ou ~99,6-100% des cellules, jamais entre les deux, panne satellite mondiale). Donc quand
   `TWS_t` d'une cellule est masqué (horizon>1, 66,5% du vrai test), son voisin l'est presque
   certainement aussi, au même mois -- une feature "valeur brute du voisin à t" serait NaN pile
   quand on en a besoin. Solution retenue : réutiliser `last_observed_tws`/
   `months_since_last_observed_tws` (déjà calculés, causals, Phase 2) mais appliqués à la cellule
   voisine plutôt qu'à soi-même -- même pendant une panne synchronisée, chaque voisin garde une
   dernière valeur connue différente (info spatiale réelle), exploitable à tout horizon.

**Implémentation** : nouveau module `src/features/spatial_neighborhood.py`
(`build_direct_neighbor_lookup` + `add_spatial_neighborhood_features`), câblé dans `pipeline.py`
juste après `add_horizon_features` (dépendance directe). 8 nouvelles colonnes :
`TWS_neighbor_{N,S,E,W}_last_observed` et `TWS_neighbor_{N,S,E,W}_months_since_last_observed`.
4 nouveaux tests (`test_spatial_neighborhood.py`) : lookup correct sur grille à trous, propagation
de la dernière valeur (pas la valeur brute), et **causalité** (perturber le masquage d'un voisin à
un mois tardif ne change rien aux lignes antérieures de la cellule qui l'observe -- même principe
que les tests de causalité de l'ANN). 53 tests au total.

**Coût mesuré** : pipeline complet (train+test réels) en ~124s, taux de NaN faible (1,2-2,6%
selon la direction -- la plupart des cellules ont un voisin direct dans chaque direction).

**Évaluation** (`src/validation/compare_spatial_neighborhood.py`) : comparaison CV temporelle 33
vs 41 features (+ les 8 de voisinage) sur le bag de production actuel (7 membres, seeds 43-49) --
même méthode que l'évaluation des features de tendance long terme (2026-09-07).

**Résultat réel (5-fold CV temporel)** : MAE 0,3718 (sans) → 0,3713 (avec), gain de -0,0005
(~-0,14%) mais **cohérent sur les 5 folds** (chaque fold s'améliore, aucun ne se dégrade) --
signal réel, pas du bruit, mais bien plus petit que le "levier majeur" annoncé par le brief.
Cohérent avec la limite structurelle anticipée : les voisins n'aident vraiment que sur horizon=1
(33,5% du test), pas sur horizon>1 (66,5%, où le voisin est aussi masqué). **Décision : adopté en
production** (rien à exclure, `EXCLUDED_FEATURE_COLUMNS` inchangé -- les 8 colonnes de voisinage
n'y sont pas, donc incluses par défaut). `configs/feature_columns.yaml` régénéré (38→46 colonnes),
`submission.csv` régénéré avec le bag à 7 membres + ces 41 features (cache par membre activé
cette fois, modèles + prédictions sauvegardés dans `models/`/`reports/bag_member_predictions/`).

**Diagnostic overfitting/underfitting demandé par l'utilisateur suite au gain faible** : mesuré
directement l'écart MAE train vs validation par fold (jamais fait jusqu'ici dans ce projet) :

| Fold | Train MAE | Val MAE | Écart |
|---|---|---|---|
| 1 | 0,3348 | 0,3570 | 0,022 |
| 2 | 0,3352 | 0,3379 | 0,003 |
| 3 | 0,3345 | 0,3431 | 0,009 |
| 4 | 0,3349 | 0,3716 | 0,037 |
| 5 | 0,3397 | 0,4176 | 0,078 |

**Pas de l'overfitting classique** : le MAE train reste quasi constant (~0,335) malgré la fenêtre
expansive (fold 5 a bien plus de données que fold 1, mais un train MAE comparable -- pas de signe
de mémorisation croissante). L'écart se creuse spécifiquement aux folds 4-5, dont la fenêtre de
validation tombe sur 2011+/2015 -- exactement les années où (a) les vraies pannes GRACE sont les
plus fréquentes (Phase 1) et (b) notre masquage augmenté applique un taux délibérément plus élevé
(35%/50% vs 5%, `configs/features.yaml`). **Conclusion : plafond de difficulté de la tâche sur ces
folds, pas un problème d'ajustement du modèle** -- cohérent avec l'observation Phase 5 que
LightGBM (modèle nettement plus capable) n'améliorait quasiment pas le MAE temporel non plus.
Implication pour la suite : le réglage d'hyperparamètres a probablement peu de marge ; les vrais
leviers restent des features qui survivent au manquant (climatologie du voisin plutôt que
dernière valeur observée) ou une vraie source d'info externe (ERA5, itération F).

## Idée d'experts par horizon (bagging spécialisé) — explorée puis mise de côté — 2026-09-09

Discussion approfondie avec l'utilisateur : entraîner des membres spécialisés par horizon
(nombre de trous consécutifs), routés à l'inférence selon l'horizon réel de la ligne de test
(déterministe, pas un poids appris). Plusieurs itérations de raffinement :
1. 7 membres experts (un par horizon 1-7) — écarté après calcul réel des volumes disponibles :
   horizon=7 ne représente que 0,7% des lignes d'un tirage typique (~15 900 lignes), concentrées
   sur un seul mois calendaire (une seule "saison" vue par ce membre).
2. Raffinement boosting/cascade (M1 sur 100% des données, M2..M7 corrigent les résidus sur des
   sous-ensembles progressivement plus durs) — répond au "je ne veux rien perdre".
3. Simplifié par l'utilisateur à 2 modèles : Modèle A (masquage calibré sur les vraies
   statistiques de rafales du train, plafonné à 3 mois) / Modèle B (mêmes emplacements de rafales,
   prolongées à 4-7 mois selon la distribution du test), routage déterministe par horizon réel.

**Vérification empirique clé, à la demande de l'utilisateur** : croisement horizon × mois cible
réel dans le test (`data/processed/test_features.parquet`) — **les horizons 5, 6 et 7 du vrai
test proviennent tous d'un seul et même événement** : la rafale de 6 mois consécutifs masqués
2017-01→2017-06 (transition GRACE→GRACE-FO), produisant une progression continue d'horizon 2→7 à
mesure qu'on avance dans cette rafale unique. Le train, lui, n'a jamais connu de rafale réelle
>3 mois (10 rafales mesurées, toutes de 2 ou 3 mois, cf. entrée précédente).

**Critique demandée par l'utilisateur avant tout code** — failles identifiées :
1. Calibrer la distribution de Modèle B sur le test revient à ajuster un modèle sur un
   échantillon de taille 1 (un seul événement historique).
2. Une rafale artificiellement étirée n'est pas nécessairement équivalente statistiquement à la
   vraie transition satellite (phénomènes physiques potentiellement différents).
3. Modèle A, recalibré sur le taux *naturel* du train (moins de trous que la config actuelle,
   déjà biaisée vers le régime du test depuis la Phase 3), risque de régresser sur la majorité du
   test qu'il gère (horizon 1-3 = 72,4% des lignes réelles) pour un gain incertain sur les 27,6%
   restants.
4. **Point rédhibitoire** : le train n'ayant jamais de rafale réelle >3 mois, Modèle B ne peut
   jamais être validé par CV contre une vraie ligne horizon>3 -- seulement du synthétique contre
   du synthétique. Contrairement à toutes les décisions prises aujourd'hui (validées par CV avant
   de soumettre), cette piste n'offre aucun garde-fou avant de dépenser une soumission Zindi réelle.
5. Rupture artificielle au point de routage (horizon 3→4) sans justification physique.
6. `months_since_last_observed_tws` est déjà une feature du modèle unique actuel -- un GBR peut
   déjà conditionner ses prédictions dessus via ses splits, sans dupliquer l'architecture.

**Décision : mise de côté** (pas abandonnée définitivement, mais pas la prochaine action) --
alternative plus sûre suggérée si le signal horizon doit être exploité : pondérer plus fort les
lignes horizon≥4 dans la fonction de perte d'un seul modèle unifié, qui reste validable par le CV
existant, plutôt que diviser en deux pipelines.

## Phase 6 : analyse d'erreur du modèle de production — jamais faite jusqu'ici — 2026-09-09/10

Décidé de prioriser l'analyse d'erreur (§6 du brief, jamais réalisée) avant toute nouvelle
architecture -- pour orienter les décisions par la donnée plutôt que par l'hypothèse, cohérent
avec le pattern de toute la session (vérifier avant de construire). Nouveau
`src/validation/error_analysis.py` : prédictions **hors-échantillon** (out-of-fold) des 5 folds
temporels sur le modèle de production réel (bag à 7 membres, 41 features incluant le voisinage
spatial) -- 1 795 014 lignes au total, ventilées par zone climatique, bande de latitude, saison,
niveau de TWS, et horizon (ajouté au périmètre du brief, jugé indispensable vu les découvertes
récentes). `lat` utilisée uniquement comme clé de regroupement post-hoc (jamais une feature,
conforme au brief §3.2). Bug rencontré et corrigé en cours de route : `pd.cut` sans labels produit
des `Interval` non sérialisables en parquet par pyarrow -- corrigé (`.astype(str)`) et code
réordonné pour que l'affichage des résultats ne dépende plus de la sauvegarde finale.

**Résultat le plus important, et le plus fiable statistiquement (échantillons ~359k par case)** :

| Niveau de TWS | MAE | n |
|---|---|---|
| Très bas (sécheresse) | **0,498** | 359 003 |
| Très haut (inondation) | 0,433 | 359 002 |
| Bas | 0,319 | 359 003 |
| Moyen | 0,304 | 359 002 |
| Haut | 0,302 | 359 004 |

Le modèle est ~1,6× moins bon sur les extrêmes (surtout la sécheresse) que sur les valeurs
normales. Pour un challenge de **détection de sécheresse**, c'est le résultat le plus actionnable
de toute la session : le modèle rate le plus précisément là où sa valeur pratique compte le plus.
Matériel direct pour la section biais du rapport de confiance (§6.1).

**Par zone climatique** : tropicale 0,390 (pire) > tempérée 0,373 > polaire 0,326 (meilleure).
**Par bande de latitude** : pires bandes (-50,-30] à 0,409 et (50,70] à 0,400 ; meilleures (70,90]
à 0,257 et (30,50] à 0,339 -- tropiques/subtropicaux des deux hémisphères systématiquement plus
durs que les hautes latitudes. **Par saison** (ajustée par hémisphère) : été pire (0,406),
printemps meilleur (0,350).

**Par horizon** : confirme la tendance générale (1-2 ≈ 0,365, 3 ≈ 0,463) mais motif non monotone
suspect (horizon 4 à 0,823, pire que horizon 5 à 0,462) et échantillons horizon 6/7/8 minuscules
(32 à 43 lignes) -- probable confusion avec les folds tardifs (plus durs) ou le niveau de TWS
extrême plutôt qu'un vrai effet horizon=4 isolé. Pas assez fiable pour agir dessus tel quel.

Sauvegardé : `reports/error_analysis_by_{climate_zone,lat_band,season,tws_level,horizon}.csv` et
`reports/error_analysis_oof_predictions.parquet` (1,79M lignes, prédictions + métadonnées
complètes pour creuser plus tard).

### Prochaine étape (2026-09-10)

Pas encore décidé avec l'utilisateur : creuser le biais TWS-extrême (ex. pondération de la perte
par niveau de TWS, ou un modèle/objectif quantile), clarifier la question ERA5 Final/ERA5T avec
les organisateurs, ou la vérification d'hyperparamètres légère toujours en attente. Voisinage
spatial (commit du 2026-09-09, `41 features`) toujours pas confirmé sur Zindi après régénération
de `submission.csv` avec ce jeu de features -- upload à faire. Changements de cette session
(`spatial_neighborhood.py`, `error_analysis.py`, refactor `pipeline.py`/`mask_augmentation.py`)
pas encore committés.

## 2026-09-19 — Push GitHub/DVC en retard + trou de tracking MLflow

- **Push GitHub/DVC oublié depuis plusieurs sessions** : la branche `main` locale était 17
  commits en avance sur `origin/main` (rien poussé depuis un moment), et `dvc data status
  --not-in-remote` montrait 3 fichiers jamais poussés vers le remote DVC local
  (`train_features.parquet`, `test_features.parquet`, `submissions/submission.csv`). Commité
  (`a459b65`, tout le travail non suivi : voisinage spatial, analyse d'erreur, recherche
  d'hyperparamètres, importance par permutation), poussé sur les deux (`git push` + `dvc push`).
- **MLflow non lancable** : `mlflow ui` plantait (500 sur `/`, `AttributeError: module 'anyio' has
  no attribute 'from_thread'`). Cause : `anyio==4.15.0` (pinné dans `requirements.txt`, tiré par
  `mlflow==3.15.2`) a changé son système de lazy-import et n'expose plus `from_thread` comme
  attribut du module -- casse le shim WSGI déprécié de Starlette que le serveur UI de MLflow
  utilise encore sous Windows. Fix : downgrade vers `anyio==4.11.0` (dernière version non yankée
  qui expose encore l'attribut), repinné dans `requirements.txt`.
- **Trou de tracking MLflow constaté par l'utilisateur** : `search_hyperparameters.py`,
  `feature_importance.py` et `error_analysis.py` (écrits lors d'une session précédente, jamais
  committés jusqu'à aujourd'hui) n'appelaient jamais `mlflow`, contrairement à la convention déjà
  en place dans `run_baselines.py`/`run_bagging.py` (`mlflow.start_run` + `log_param`/
  `log_metrics`/`log_artifact`). Corrigé dans les 3 scripts. Les runs déjà exécutés (résultats
  déjà dans `reports/*.csv`, recalcul coûteux évité) ont été **rattrapés dans `mlflow.db`** via un
  script ponctuel (non versionné) qui relit ces CSV et les logue comme des runs tagués
  `backfilled=true` -- distinguables des futurs runs live si besoin de comparer la fidélité.
