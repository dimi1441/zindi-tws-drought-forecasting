# ADR 0002 — Pipeline de features : panel combiné et climatologie causale par construction

**Date** : 2026-09-05
**Statut** : Acceptée

## Contexte

Phase 2 du brief (`doc/PROJECT_BRIEF.md` §4.1, §5) : construire `src/features/` pour produire
des datasets enrichis à partir de `Train.csv`/`Test.csv`. Contrainte centrale du challenge :
aucune feature ne doit utiliser d'information future par rapport au mois `t` qu'elle décrit
(§3.1). Le voisinage spatial (`spatial_neighborhood.py`) est reporté à la Phase 5, itération D
(décision prise en cours de conception, voir `JOURNAL.md` 2026-09-05) — cet ADR ne couvre donc
que les 4 modules effectivement implémentés : lags temporels, climatologie, encodage du mois
cible, et horizon effectif.

## Options considérées

**Lags/climatologie sur test seul vs panel combiné train+test** : calculer les features de test
uniquement à partir des 18 mois de test disponibles aurait produit des séries trouées et de
mauvaise qualité, alors que test partage exactement les mêmes 15 715 cellules que train (Phase 1).

**Climatologie fold-safe** : soit (a) une climatologie fixe calculée une fois sur tout le train
et recalculée à chaque fold de la Phase 4 (option suggérée par la checklist anti-leakage §3.5),
soit (b) une climatologie causale par construction, valable ligne par ligne sans machinerie de
fold.

**Horizon effectif** : réutiliser tel quel le calcul en boucle Python du notebook d'exploration
(Phase 1), ou le réécrire en version vectorisée pour le pipeline.

## Décision

- **Panel long combiné** (`io.build_cell_timeline`) : concatène train et test triés par
  `(lat, lon, time)`, avec un flag `is_train`. Tous les modules de features opèrent sur ce panel ;
  `TWS_t` masqué de test reste `NaN` (jamais rempli). `pipeline.py` re-sépare train/test à la fin.
- **Climatologie causale par construction** : `expanding().mean().shift(1)` par
  `(cellule, mois calendaire)`, trié par année. Chaque ligne n'utilise que les années strictement
  antérieures — vérifié que `pandas.Series.expanding().mean()` ignore nativement les `NaN` sans
  les propager, donc un mois absent ou masqué est simplement sauté, sans code spécial.
- **Horizon vectorisé** (`horizon_features.add_horizon_features`) : `time`/`TWS_t` mis à `NaN`
  là où masqué (`.where(~masked)`), puis propagés vers l'avant par cellule (`.groupby().ffill()`).
  Testé en régression contre la distribution déjà validée à la main en Phase 1
  (`{1: 94048, 2: 62576, ..., 7: 15445}`) — résultat identique.
- **Feature ajoutée par rapport au brief** : `last_observed_tws` (la dernière valeur réelle
  connue, pas seulement le compte de mois écoulés) — permet au modèle une persistance depuis le
  dernier point connu plutôt qu'une imputation par une constante globale (défaut identifié dans
  le notebook starter des organisateurs).

## Justification

- Le panel combiné est la seule façon de donner à test un historique réel (les lags/climatologie
  de ses premières lignes proviennent légitimement de la fin du train), sans dupliquer la logique
  dans deux code paths séparés pour train et pour test.
- La climatologie causale par construction est **plus stricte** que "recalculée fold par fold" :
  elle ne peut pas fuiter même si on oublie de la refaire à la Phase 4, et elle est plus simple à
  raisonner (une seule implémentation, jamais deux — une pour Phase 2, une pour la validation).
- La vectorisation de l'horizon est nécessaire pour la performance (panel ~2,4M lignes contre
  281k en Phase 1) ; le test de régression garantit qu'elle ne change pas silencieusement le
  résultat déjà validé à la main.

## Limite assumée

La climatologie des toutes premières années d'une cellule est `NaN` ou basée sur très peu
d'années (peu de données au début de la série) — accepté comme un compromis honnête plutôt qu'une
climatologie complète qui fuiterait le futur. LightGBM/CatBoost gèrent le NaN nativement, donc
aucune imputation compensatoire n'est ajoutée.
