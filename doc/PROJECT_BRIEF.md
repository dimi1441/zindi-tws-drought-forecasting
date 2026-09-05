# Project Brief — TWS Forecasting Challenge

> Document de référence à destination de Claude Code (et de tout collaborateur technique).
> Il résume le contexte du challenge, la nature du problème, les erreurs à éviter,
> les étapes de réalisation et les livrables attendus. À lire en priorité avant tout
> travail de code sur ce projet.

---

## 1. Contexte du challenge

### 1.1 Organisation

Challenge international co-organisé par un consortium prestigieux : ITU (AI for Good), Commission européenne, Copernicus, European and Global Drought Observatories, King's Institute for AI, ESA, WMO, ECMWF, CGIAR Climate Action, Fraunhofer HHI, IWMI, UNCCD.

Objectif : prédire à court terme le **Total Water Storage (TWS)** à l'échelle globale pour appuyer la détection précoce des sécheresses.

### 1.2 Le TWS et la mission GRACE

- **TWS (Total Water Storage)** : quantité totale d'eau stockée à la surface et sous la surface terrestre (eaux souterraines, humidité du sol, eaux de surface, neige).
- **GRACE / GRACE-FO** : mission satellitaire NASA/DLR qui mesure les variations du champ de gravité terrestre pour en déduire les variations de TWS. Seule source directe et globale de cette variable.
- **Contrainte opérationnelle** : les données GRACE sont publiées avec un délai de **2 à 3 mois** (parfois plus), ce qui limite leur usage en temps quasi réel — d'où la motivation du challenge.

### 1.3 Enjeu

Fournir des estimations mensuelles anticipées de TWS pour appuyer la sécurité alimentaire, la planification agricole et les réponses humanitaires en zones vulnérables (notamment Sahel, Corne de l'Afrique).

---

## 2. Problème à résoudre

### 2.1 Formulation

Prédire la valeur **`target`** pour chaque ligne du fichier de test, où :
- **`target` = TWS au mois `t+1`** pour un couple (localisation, mois `t`).

### 2.2 Structure du dataset

Fichiers fournis : `Train.csv`, `Test.csv`, `SampleSubmission.csv`.
Chaque ligne = un échantillon `(time, lat, lon)`, identifié par `ID` au format `YYYYMMDD_lat_lon`.

**Colonnes disponibles** :

| Colonne | Description |
|---|---|
| `TWS_t` | TWS du mois courant. **Masqué (~66,5 % des lignes de test)** |
| `SPEI_01_t`, `SPEI_03_t`, `SPEI_06_t`, `SPEI_12_t` | Indices standardisés de sécheresse/humidité (fenêtres 1, 3, 6, 12 mois) |
| `SOIL_MOISTURE_t` | Humidité du sol au mois `t` |
| `month_sin`, `month_cos` | Encodage cyclique du mois de `t` |
| `target` | TWS au mois `t+1` (train uniquement — variable cible) |
| `TWS_t_masked` | Booléen (test uniquement) : `True` si `TWS_t` a été masqué |
| `time`, `lat`, `lon`, `ID` | Identifiants |

### 2.3 Le masquage : deux fonctions superposées

Le masquage de `TWS_t` sur ~66,5 % des lignes de test remplit deux rôles :

1. **Anti-triche** : sans masquage, le `TWS_t` d'une ligne de test au mois `t` révélerait directement la target d'une autre ligne (au mois `t-1` pour la même cellule, dont la target = TWS de `t`).
2. **Simulation du retard GRACE** : reproduire la contrainte opérationnelle réelle.

**Conséquence** : sur une même cellule, plusieurs mois consécutifs peuvent être masqués, créant des séquences de trous.

### 2.4 Horizon effectif de prédiction : 1 à 7 mois

Bien que le libellé soit « one-month ahead », **l'horizon effectif varie de 1 à 7 mois** selon les lignes.

Pour chaque ligne de test datée `t` (cible = `t+1`), l'horizon effectif est défini comme :
```
horizon_effectif = (t+1) − (dernier mois où TWS_t est observé pour cette cellule)
```

Cela signifie que le modèle doit s'adapter à des horizons variables, sans qu'un horizon lui soit explicitement donné.

---

## 3. Règles à respecter absolument

### 3.1 Anti-leakage (règle centrale)

> **Pour prédire au mois `t`, seules les informations disponibles à `t` ou avant peuvent être utilisées.**
> Aucune valeur future, ni directement, ni via un calcul qui la contient en douce.

**Exception unique** : le mois calendaire de `t+1` peut être utilisé (via encodage cyclique), puisque le calendrier est connu à l'avance et ne constitue pas une observation future.

### 3.2 Usage des coordonnées `lat` / `lon`

Position officielle des organisateurs (confirmée le 19 août dans le chat du challenge) :

**Interdit** :
- Passer `lat`, `lon`, `ID` (cellule), ou tout **encodage dérivé** (embeddings, buckets, bandes de latitude, etc.) comme feature prédictive au modèle.

**Permis** :
- Utiliser `(lat, lon)` comme **identifiant de groupe** (calcul de lags par cellule, GroupKFold spatial, définition de voisinages).
- Utiliser les **valeurs de TWS des cellules voisines** au mois `t` ou avant.
- Faire du **filtrage spatial** et de l'**agrégation spatiale** (moyennes locales, gradients, anomalies de voisinage).

**Corollaire important** : ne pas transformer `lat` en features géographiques dérivées (bande climatique par latitude, etc.) — c'est considéré comme un encodage de coordonnée. Utiliser plutôt des variables physiques indépendantes (climatologie de température, précipitations moyennes, indice d'aridité) provenant de sources autorisées.

### 3.3 Données externes autorisées

- Covariables Copernicus : European and Global Drought Observatories, Copernicus Climate Data Store.
- **ERA5** (réanalyse ECMWF) : autorisée mais une question reste ouverte : peut-on utiliser `ERA5 Final` (retard 2-3 mois) pour reconstituer ce que `ERA5T` (retard ~5 jours mais non archivé pour la période) aurait fourni ? À trancher explicitement avec les organisateurs si utilisé.
- **Interdit** : tout produit externe dérivé de GRACE/TWS.
- Toute donnée externe doit être **documentée** et **respecter la contrainte temporelle** (disponibilité à `t`).

### 3.4 Interpolations dangereuses

- ❌ `pandas.interpolate(method='linear')` par défaut → utilise passé ET futur.
- ❌ Interpolation par spline sur toute la série.
- ❌ `fillna` par la moyenne globale (inclut le futur).
- ✅ `ffill` (forward fill) — n'utilise que le passé.
- ✅ Moyenne des N mois précédents.
- ✅ Estimation par un modèle entraîné uniquement sur le passé.

### 3.5 Checklist anti-leakage systématique

À vérifier pour chaque nouvelle feature ou transformation :

- [ ] Aucune valeur d'une variable au mois `t+1` (hors mois calendaire) n'est utilisée.
- [ ] Les lags et moyennes mobiles ne regardent que le passé (`t` inclus, jamais au-delà).
- [ ] Les interpolations sont unidirectionnelles (ffill, pas linéaire bidirectionnelle).
- [ ] La climatologie est calculée sur une période fixe antérieure, ou refaite fold par fold.
- [ ] Le masquage simulé en entraînement suit la même logique cumulative que le test.
- [ ] La validation croisée respecte l'ordre temporel (pas de futur dans le train).
- [ ] Toute statistique par cellule (climatologie, moyenne locale) est recalculée fold par fold.
- [ ] Les cellules spatialement corrélées (même bassin versant) sont regroupées dans un même fold.
- [ ] Les voisins spatiaux respectent le masquage : un voisin masqué à `t` reste NaN, jamais rempli par une valeur future.

---

## 4. Stratégie technique

### 4.1 Feature engineering

**Features temporelles (par cellule, en respectant l'ordre chronologique)** :
- Lags de TWS : `TWS_{t-1}`, `TWS_{t-2}`, ..., `TWS_{t-k}` (jusqu'à 6-12 mois).
- Différences : `TWS_t − TWS_{t-1}` (vitesse), `TWS_t − TWS_{t-12}` (interannuel).
- Moyennes mobiles : 3, 6, 12 mois.
- Feature d'horizon : `months_since_last_observed_tws` (essentielle pour que le modèle calibre sa confiance).

**Features saisonnières** :
- `target_month_sin`, `target_month_cos` : encodage cyclique du mois cible `t+1`.
- Climatologie : moyenne historique du TWS par cellule × mois calendaire, calculée sur période fixe.
- Anomalies saisonnières : `TWS_t − climatologie(cellule, mois de t)`.

**Features climatiques** :
- Utiliser SPEI, humidité du sol au mois `t` et aux mois passés, **si disponibles**.
- Construire lags et moyennes mobiles similaires.

**Features spatiales de voisinage** (levier majeur du challenge) :
- Pour chaque cellule, définir un voisinage (par exemple carré 3×3 ou 5×5).
- Moyenne, écart-type, gradient du TWS des voisins à `t`, `t-1`, `t-2`.
- Anomalie locale : `TWS_t (cellule) − moyenne des voisins`.
- Même chose pour SPEI et humidité du sol.

**Features géographiques statiques (avec précaution)** :
- **Ne PAS** utiliser `lat`, `lon`, ni d'encodage dérivé comme feature.
- Éventuellement des variables physiques indépendantes (climatologie de température ERA5, précipitations moyennes) provenant de sources autorisées.

### 4.2 Gestion de l'horizon variable (1-7 mois)

**Principe** : un seul modèle qui apprend à s'adapter, pas 7 modèles séparés.

**Deux leviers** :

1. **Feature explicite d'horizon** (`months_since_last_observed_tws`).
2. **Masquage augmenté à l'entraînement** :
   - **Correction (2026-09-05, Phase 3)** : le masquage réel n'est **pas** indépendant par
     cellule. Mesuré sur le test (Phase 1) : le taux de masquage par mois est quasi binaire
     (~0 % ou ~99,6-100 %, jamais intermédiaire), et confirmé par les 22 mois du train totalement
     absents (pas masqués — la ligne n'existe pas), qui coïncident avec des pannes GRACE
     documentées (dégradation des batteries à partir de 2011, transition GRACE→GRACE-FO fin
     2017-2018). Le masquage augmenté doit donc couper des **mois calendaires entiers pour
     (quasi) toutes les cellules à la fois**, jamais cellule par cellule indépendamment — sinon
     un modèle apprendrait une béquille ("mes voisins ont des données même quand je n'en ai pas")
     qui n'existe jamais en réalité et qui deviendrait trompeuse dès que des features de
     voisinage spatial seraient ajoutées.
   - **Mode dynamique uniquement** : le masque n'est jamais figé sur disque, il est retiré à
     chaque appel (nouvel ensemble de mois masqués à chaque tirage) — un entraînement exposé à
     plusieurs tirages différents (ensemble de modèles, ou callback d'époque pour un modèle
     itératif) généralise mieux qu'un masque unique figé une fois pour toutes.
   - Fréquence calibrée sur la forme historique (croissante dans le temps, concentrée après 2011)
     mais rapprochée du régime, plus dur, observé en test — voir
     `docs/decisions/0003-augmented-masking-mechanism.md` et
     `configs/features.yaml: masking.target_gap_rate_by_period`.
   - Distribution cible mesurée sur le test (utilisation de `TWS_t_masked` uniquement, pas des
     valeurs — pas de leakage) : `{1: 94048, 2: 62576, 3: 46777, 4: 31076, 5: 15560, 6: 15479,
     7: 15445}` (voir `reports/data_understanding.md`). L'histogramme résultant du masquage par
     mois entier ne la reproduit pas exactement (effet émergent, pas contrôlé ligne à ligne) —
     accepté comme point de départ à affiner en Phase 5 (itération E).

### 4.3 Choix du modèle

Approche recommandée : **commencer simple, itérer**.

- **Baseline forte** : Persistance (`y_pred = TWS_t` quand disponible, climatologie sinon).
- **Modèles principaux** : Gradient boosting (LightGBM, XGBoost, CatBoost) — robustes, rapides, gèrent les NaN nativement.
- **Optionnels** (si le gain sur GBM plafonne) : LSTM, TCN, Temporal Fusion Transformer.

**Contrainte de soutenabilité** : préférer les modèles légers si l'écart de performance ne justifie pas le surcoût.

### 4.4 Validation croisée

Deux schémas complémentaires **indispensables** :

**A) Split temporel (obligatoire)** — rolling-window ou expanding-window. Le train contient uniquement des mois antérieurs à ceux de la validation. Évite le leakage temporel.

**B) GroupKFold spatial** — chaque cellule appartient à un seul fold. Cellules du même bassin versant / région corrélée regroupées dans le même fold. Évalue la robustesse spatiale exigée par le challenge.

**Bonus** : schéma spatio-temporel combiné (localisations hold-out ET mois futurs hold-out) pour la mesure la plus honnête.

Rapporter : moyenne ± écart-type sur les folds. L'écart entre les deux schémas indique un éventuel surapprentissage local.

---

## 5. Étapes de réalisation (roadmap)

### Phase 0 — Mise en place (1-2 jours)

- Structure de dossiers : `data/{raw,interim,processed}/`, `notebooks/`, `src/`, `models/`, `configs/`, `reports/`.
- Environnement Python figé (`requirements.txt` avec versions exactes, ou `pyproject.toml`).
- Git initialisé avec `.gitignore` (données lourdes exclues).
- **DVC** : `dvc init`, remote configuré, `dvc add` sur les CSV bruts.
- **MLflow** : experiment `tws-forecasting` créé.
- Seeds globales fixées et documentées.

### Phase 1 — Exploration (2-3 jours)

- Notebook `01_exploration.ipynb` :
  - Statistiques descriptives.
  - **Mesure empirique de la distribution des horizons dans le test** (essentielle pour phase 3).
  - Cartes de couverture spatiale et de manquants.
  - Analyse saisonnière et corrélations à différents lags.
- Rapport court `reports/data_understanding.md`.

### Phase 2 — Pipeline de features (3-5 jours)

- Module `src/features/` :
  - `temporal_lags.py`
  - `spatial_neighborhood.py`
  - `seasonal.py` (climatologie + anomalies)
  - `target_month_encoding.py`
  - `horizon_features.py`
- Orchestration `src/features/pipeline.py`.
- Datasets enrichis → `data/processed/` (format Parquet, versionnés DVC).
- `dvc.yaml` avec stage `feature_engineering`.

### Phase 3 — Masquage augmenté (1-2 jours) — fait le 2026-09-05

- `src/features/mask_augmentation.py` : **mode dynamique uniquement** (décision du 2026-09-05,
  voir `docs/decisions/0003-augmented-masking-mechanism.md`) — pas de version statique figée sur
  disque, le masque (mois calendaires entiers, pas par cellule) est retiré à chaque appel via
  `pipeline.build_features(..., masking_config, rng)`. La boucle multi-tirages (ensemble de
  modèles ou callback d'époque) est laissée à la Phase 4/5.
- Distribution cible (mesurée sur le test en Phase 1) documentée dans `reports/data_understanding.md`
  et rappelée en §4.2 ci-dessus ; fréquence de masquage configurée dans
  `configs/features.yaml: masking.target_gap_rate_by_period`.

### Phase 4 — Baseline et validation (2 jours)

- Reproduire la baseline du notebook fourni.
- Baseline naïve : persistance pure + fallback climatologie.
- Module `src/validation/` : splits temporel, spatial, combiné.
- Métriques par fold : MAE, RMSE, R².
- Chaque run baseline tracé dans MLflow.

### Phase 5 — Modélisation itérative (1-2 semaines)

Itérations progressives, chacune = un ou plusieurs runs MLflow tagués :

- **A** : baseline avec features simples (TWS, SPEI, humidité, mois cyclique, horizon).
- **B** : + lags temporels et différences.
- **C** : + climatologie et anomalies saisonnières.
- **D** : + features de voisinage spatial (levier majeur).
- **E** : + masquage augmenté à l'entraînement.
- **F** : + covariables externes (ERA5, si autorisation clarifiée).
- **G** : autres modèles (CatBoost, XGBoost, éventuellement LSTM/TCN).

Intégrer SHAP et CodeCarbon dès qu'un modèle mérite analyse (dès itération C ou D).

### Phase 6 — Analyse d'erreurs et robustesse (2-3 jours)

- Erreur par **horizon effectif** (1-7 mois).
- Erreur par **zone climatique** (tropicale, tempérée, aride, polaire).
- Erreur par **saison**.
- Erreur en fonction du niveau moyen de TWS.
- Cartes d'erreur spatiale.
- SHAP global + explications locales sur cas représentatifs.

Cette phase produit le matériel brut pour la section « biais » du rapport final.

### Phase 7 — Soumission et documentation (2-3 jours)

- Script `src/generate_submission.py` : réentraînement sur train complet, production de `submission.csv`.
- **Document Trustworthiness (400 mots)** — voir section 6.
- README complet avec instructions de reproduction.

### Phase 8 — Nettoyage et publication (1-2 jours)

- Licence claire (MIT ou Apache 2.0).
- Notebook de démonstration.
- Push GitHub avec DVC remote accessible.

**Calendrier total réaliste : 4 à 6 semaines de travail sérieux.**

---

## 6. Documentation à remettre (Trustworthiness Evaluation)

**Rappel important** : cette section n'est demandée qu'aux **top 10 participants**. Priorité absolue à la performance du modèle pour y accéder.

Format : **maximum 100 mots par section**, quatre sections, **400 mots total**. Ton attendu : synthétique, factuel, honnête sur les limites.

### 6.1 Data & Model Bias (max 100 mots)

À couvrir :
- Biais identifiés (spatial, temporel, environnemental).
- Outils/méthodes d'évaluation (AI Fairness 360 mentionné, mais analyse par sous-groupes plus adaptée à une régression continue).
- Biais résiduels non corrigés et leur impact.

**Matériel à collecter en phase 6** : MAE par continent, par bande latitudinale, par horizon, par saison.

### 6.2 Model Transparency (max 100 mots)

À couvrir :
- Outils utilisés (SHAP, LIME).
- Features les plus influentes (TWS lag, anomalies saisonnières, voisinage, SPEI).
- Une observation inattendue ou intéressante.

**Matériel à collecter en phase 5-6** : SHAP TreeExplainer sur modèle final, importance globale + explications locales sur cas représentatifs, éventuellement SHAP par horizon.

### 6.3 Approach Reusability (max 100 mots)

À couvrir :
- Structure modulaire et paramétrable (configs YAML, Hydra si utilisé).
- Un choix concret qui renforce la flexibilité.
- Une limite honnête.

**Matériel à préparer dès phase 2** : pipeline découplé feature/model/eval, configs séparées, `dvc repro` reproductible.

### 6.4 Sustainability and Efficiency (max 100 mots)

À couvrir :
- Outil de mesure utilisé (CodeCarbon, ML CO2 Impact) + chiffres réels.
- Optimisations (Parquet vs CSV, cache DVC, choix modèle léger).
- Compromis complexité/efficacité assumé (avec exemple chiffré).

**Matériel à collecter en phase 5** : CodeCarbon activé sur runs finaux, comparaison chiffrée entre 2 architectures testées.

### 6.5 Livrables techniques additionnels

- **Repo GitHub public** propre (structure, README, licence).
- **Pipeline DVC** reproductible (`dvc repro` de bout en bout).
- **MLflow tracking** exportable ou hébergé pour permettre au jury de naviguer.
- **`submission.csv`** au format attendu.

---

## 7. Stack technique et outils

### 7.1 Outils imposés par le contexte projet

- **DVC** : versioning données + pipeline (`dvc.yaml`).
- **MLflow** : tracking des expériences (params, métriques, artefacts).

### 7.2 Outils recommandés pour la Trustworthiness

- **SHAP** (TreeExplainer pour gradient boosting) → transparence.
- **CodeCarbon** (3 lignes de code, tourne en tâche de fond) → soutenabilité.
- **AI Fairness 360** : mentionné par le challenge mais peu adapté à une régression continue. Documenter le choix de l'analyse manuelle par sous-groupes à la place.

### 7.3 Outils recommandés pour la qualité du code

- **Hydra** ou **OmegaConf** pour les configs YAML.
- **pytest** pour tests unitaires sur les fonctions critiques (calcul de lags, climatologie, masquage).
- **type hints** systématiques.
- **docstrings** format numpy/google.
- **mkdocs** ou **sphinx** pour doc HTML si le temps le permet.
- **Parquet** (via pyarrow) plutôt que CSV pour les données intermédiaires.

### 7.4 Bibliothèques ML principales

- `pandas`, `numpy`, `scikit-learn` (base).
- `lightgbm` ou `xgboost` ou `catboost` (modèles principaux).
- `shap` (interprétabilité).
- `codecarbon` (empreinte carbone).
- `pyarrow` (Parquet).
- Optionnels : `pytorch` ou `pytorch-lightning` si LSTM/TCN, `torchvision` non requis.

---

## 8. Pratiques transversales

### 8.1 Reproductibilité

- Toutes les seeds fixées (`random`, `numpy`, framework ML).
- Versions figées (`==`, pas `>=`).
- Aucune étape manuelle non documentée : tout dans `dvc.yaml` + `Makefile`.
- Container Docker en bonus si le temps le permet.

### 8.2 Journal de projet

Fichier `JOURNAL.md` versionné dans Git, mis à jour à chaque session :
- Ce qui a été fait, appris, décidé.
- Pourquoi telle décision plutôt qu'une autre.
- Base d'écriture pour les 400 mots finaux.

Non livré au jury, mais indispensable pour la mémoire du projet.

### 8.3 Décisions architecturales

Format ADR (Architecture Decision Record) dans `docs/decisions/` :
- Un fichier court par décision majeure (choix du modèle, choix du split, gestion des voisinages...).
- Contexte, options considérées, décision, justification.
- Bonus énorme pour la crédibilité méthodologique.

---

## 9. Points d'attention spécifiques pour Claude Code

Quelques rappels utiles pour tout travail de code sur ce projet :

1. **Toujours grouper par cellule avant de calculer un lag ou une moyenne mobile.** Sinon les valeurs de cellules différentes se mélangent.
   ```python
   df.groupby(['lat', 'lon']).shift(k)  # ✅
   df.shift(k)                          # ❌
   ```

2. **Ne jamais passer `lat`, `lon`, `ID`, ou tout dérivé comme feature au modèle.** Ces colonnes servent uniquement à grouper / valider / définir des voisinages.

3. **Toute nouvelle feature doit passer la checklist anti-leakage** (section 3.5) avant d'être ajoutée au pipeline.

4. **La distribution des horizons dans le train doit refléter celle du test.** À mesurer d'abord (phase 1), à reproduire ensuite (phase 3).

5. **Format Parquet systématiquement** pour les fichiers intermédiaires (lecture plus rapide, plus compact, préserve les types).

6. **Chaque expérience = un run MLflow taggué** avec suffisamment de contexte pour être compréhensible plus tard.

7. **Chaque changement majeur du pipeline = un nouveau stage `dvc.yaml`** pour garantir la reproductibilité.

8. **Le mois calendaire de `t+1` est autorisé** (encodage cyclique) — c'est la seule information sur `t+1` utilisable.

9. **Priorité performance d'abord, documentation ensuite.** La section Trustworthiness n'est demandée qu'aux top 10 ; il faut d'abord y arriver.

10. **Écrire d'abord, coder ensuite.** Un paragraphe de deux lignes dans `JOURNAL.md` avant chaque décision non triviale.

---

## 10. Références rapides

- Notebook de départ fourni par les organisateurs : `StarterNotebook.ipynb`.
- Baseline à battre : persistance simple (`TWS_{t+1} ≈ TWS_t`).
- Score de référence indicatif atteint honnêtement par un participant : **~0.70** avec ERA5 et sans features interdites.
- Deadline et règlement complet : à consulter sur la plateforme du challenge.

---

*Fin du document.*
