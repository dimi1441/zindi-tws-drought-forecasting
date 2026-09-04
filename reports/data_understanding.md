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

- Aucune valeur `NaN` explicite dans les colonnes numériques : le masquage de `TWS_t` en test se
  traduit par une valeur substituée, signalée uniquement par le booléen `TWS_t_masked`.
- **66,5 % des lignes de test ont `TWS_t` masqué** — cohérent avec le chiffre annoncé dans le
  brief (~66,5 %).
- Règle stricte : ne jamais utiliser `TWS_t` en test sans vérifier `TWS_t_masked` au préalable.

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
  (Phase 3), de façon cumulative (on masque toujours les N mois les plus récents d'une cellule,
  jamais des mois isolés).

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
2. **Phase 3 (masquage augmenté)** — reproduire la distribution horizon mesurée ci-dessus
   (masquage cumulatif des N mois les plus récents, N tiré selon cette distribution empirique).
3. **Phase 4 (validation)** — tenir compte des 1 157 cellules à couverture temporelle incomplète
   lors de la construction des splits.
4. **Baseline à battre** : persistance pure (`TWS_{t+1} ≈ TWS_t`), déjà fortement corrélée
   (r = 0,803) — le modèle final doit apporter un gain net au-delà de cette heuristique simple.
