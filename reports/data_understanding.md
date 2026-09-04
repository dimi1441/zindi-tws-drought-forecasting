# Data Understanding — Phase 1

Résumé des résultats du notebook [`notebooks/01_exploration.ipynb`](../notebooks/01_exploration.ipynb).

## Vue d'ensemble

| | Train | Test |
|---|---|---|
| Lignes | 2 154 021 | 280 961 |
| Cellules (lat/lon) | 15 715 | 15 715 (identiques au train — 0 cellule exclusive de part et d'autre) |
| Mois couverts | 138 (2002-05 → 2015-08) | 18 mois répartis entre 2015-09 et 2018-12 (pas de couverture continue) |

La grille spatiale est parfaitement stable entre train et test : aucune cellule n'apparaît
uniquement dans l'un des deux fichiers. 1 157 cellules (7,4 %) ont une couverture temporelle
incomplète en train (dernier mois observé avant 2015-08) — à surveiller lors du split temporel
(Phase 4).

## Statistiques descriptives

`TWS_t` et les covariables (`SPEI_01/03/06/12_t`, `SOIL_MOISTURE_t`) sont déjà standardisées
(moyenne ≈ 0, écart-type ≈ 1), vraisemblablement par les organisateurs. Les distributions
train/test se superposent bien (voir `fig_distributions.png`) — pas de dérive de distribution
flagrante entre les deux périodes.

## Valeurs manquantes et masquage

- `TWS_t` est **explicitement `NaN`** pour les lignes masquées en test (186 913 NaN, exactement
  égal au nombre de lignes `TWS_t_masked == True`) ; aucune autre colonne n'a de manquants
  (train comme test). *(Correction 2026-09-04 : une version précédente de ce rapport affirmait à
  tort que le masquage produisait une valeur substituée plutôt qu'un vrai `NaN` — contredit par la
  mesure `isna()` elle-même ; voir le notebook §3.)*
- **66,5 % des lignes de test ont `TWS_t` masqué (= NaN)** — cohérent avec le chiffre annoncé dans
  le brief (~66,5 %).
- Règle stricte : ne jamais utiliser `TWS_t` en test sans vérifier `TWS_t_masked` au préalable (ou,
  de façon équivalente en pratique ici, `TWS_t.isna()`).

## Horizon effectif (mesure sans fuite)

Calculé uniquement à partir de `TWS_t_masked` (jamais des valeurs), selon la définition du brief :
`horizon_effectif = (t+1) − (dernier mois où TWS_t est observé pour cette cellule)`.

Distribution empirique sur le test :

| Horizon (mois) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Nb lignes | 94 048 | 62 576 | 46 777 | 31 076 | 15 560 | 15 479 | 15 445 |

- Min = 1, Max = 7, moyenne ≈ 2,71 — confirme exactement la plage 1-7 mois annoncée par le brief.
- La distribution est fortement décroissante (33,5 % des lignes en horizon 1, seulement 5,5 %
  chacune pour les horizons 5 à 7).
- Cette distribution doit être **reproduite dans le masquage augmenté à l'entraînement**
  (Phase 3).

### Structure du masquage : de vrais trous de la mission GRACE, partagés par tout le globe

*(Deux corrections successives le 2026-09-04. D'abord : le masquage n'est pas « cumulatif » —
une fois masqué, une cellule ne reste pas masquée jusqu'à la fin du test ; 99,9 % des cellules ont
plusieurs segments masqués distincts. Ensuite, affinement : ces segments ne sont pas générés
indépendamment cellule par cellule — voir ci-dessous.)*

- **Le masquage opère au niveau du mois calendaire, pas de la cellule.** Le taux de masquage par
  mois de test est quasi binaire : soit ~0 % (mois "bon"), soit 99,6-100 % (mois "gap") — jamais
  de valeur intermédiaire. 12 des 18 mois de test sont des mois "gap", 6 sont des mois "bons"
  (2015-09, 2016-01, 2016-06, 2016-12, 2018-07, 2018-11).
- **Le train confirme la même origine** : 22 mois calendaires sur 160 possibles (2002-05 →
  2015-08) sont **totalement absents** des lignes du train (pas de `NaN` — la ligne n'existe pas).
  Ces mois coïncident avec des interruptions réelles et documentées de la mission GRACE :
  mise en service du satellite (2002-06 → 2002-08, 2003-06 → 2003-07), puis gaps récurrents de
  1-2 mois liés à la dégradation des batteries à partir de 2011 (2011-01 → 2014-08), de plus en
  plus fréquents avec le temps. Le trou de 13 mois entre les deux derniers blocs du test
  (2017-06 → 2018-07) correspond à la vraie transition GRACE → GRACE-FO.
- **Conclusion** : ce ne sont pas des trous synthétiques ni un bruit aléatoire par cellule — c'est
  la vraie chronologie d'indisponibilité de la mission, de fréquence croissante dans le temps.
- **Conséquences pour la Phase 3 (masquage augmenté)** :
  1. **Masquer par mois, pas par cellule indépendamment** : choisir un mois du train et masquer
     `TWS_t` pour (quasi) toutes ses cellules d'un coup, pour reproduire le vrai mécanisme (un
     mois entier de données GRACE indisponibles) plutôt qu'un bruit indépendant par cellule qui
     ne correspond à rien de réel.
  2. **Calibrer la fréquence sur les vrais gaps, avec un taux croissant dans le temps** : utiliser
     la fréquence des 22 mois absents du train (~14 % des mois, concentrés après 2011) plutôt
     qu'un taux constant — les gaps étaient rares avant 2011 et de plus en plus fréquents ensuite.

## Couverture spatiale

Grille quasi complète et régulière, correspondant aux continents (pas de cellules océaniques —
cohérent avec un produit GRACE de stockage d'eau terrestre). Voir `fig_spatial_coverage.png` et
`fig_temporal_completeness_map.png`.

## Analyse saisonnière

Cycle saisonnier moyen clairement marqué (`fig_seasonal_cycle.png`). En décomposant par bande de
latitude (`fig_seasonal_by_latband.png`) :

- Hémisphères nord et sud présentent des cycles **opposés en phase** (décalage d'environ 6 mois),
  cohérent avec les saisons hydrologiques inversées.
- La bande tropicale a un cycle plus atténué.

→ Justifie une climatologie **par cellule et par mois** (Phase 2, `seasonal.py`) plutôt qu'une
climatologie globale.

## Corrélations à différents lags

| Lag | Corrélation avec `target` |
|---|---|
| `TWS_t` (lag 0, persistance) | **0,803** |
| `TWS_t_lag1` | 0,720 |
| `TWS_t_lag3` | 0,629 |
| `TWS_t_lag6` | 0,510 |
| `TWS_t_lag12` | 0,371 |

`TWS_t` (persistance pure) reste de loin le meilleur prédicteur linéaire, ce qui confirme la
pertinence de la baseline de persistance. La corrélation décroît avec le lag mais reste
significative à 12 mois → justifie une feature interannuelle `TWS_t − TWS_{t-12}` en Phase 2.

Corrélation des covariables avec `target` (hors lags de TWS) :

| Variable | Corrélation |
|---|---|
| `SPEI_12_t` | 0,380 |
| `SPEI_06_t` | 0,368 |
| `SOIL_MOISTURE_t` | 0,320 |
| `SPEI_03_t` | 0,285 |
| `SPEI_01_t` | 0,204 |

Les fenêtres SPEI longues (12, 6 mois) sont plus informatives que les fenêtres courtes pour
prédire `target` — cohérent avec l'inertie hydrologique du stockage d'eau (TWS intègre les
anomalies climatiques sur plusieurs mois).

## Implications pour la suite

1. **Phase 2 (features)** — prioriser : lags temporels (1, 3, 6, 12 mois), climatologie
   cellule/mois + anomalies, feature d'horizon explicite (`months_since_last_observed_tws`),
   voisinage spatial.
2. **Phase 3 (masquage augmenté)** — reproduire la distribution horizon mesurée ci-dessus via des
   **mois entiers masqués simultanément pour (quasi) toutes les cellules** (pas un tirage
   indépendant par cellule), avec une fréquence de gaps croissante dans le temps, calibrée sur les
   22 mois absents du train (~14 % des mois, concentrés après 2011).
3. **Phase 4 (validation)** — tenir compte des 1 157 cellules à couverture temporelle incomplète
   lors de la construction des splits.
4. **Baseline à battre** : persistance pure (`TWS_{t+1} ≈ TWS_t`), déjà fortement corrélée
   (r = 0,803) — le modèle final doit apporter un gain net au-delà de cette heuristique simple.
