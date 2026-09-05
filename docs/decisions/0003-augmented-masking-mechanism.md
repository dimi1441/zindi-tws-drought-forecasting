# ADR 0003 — Masquage augmenté : mois entiers, mode dynamique uniquement

**Date** : 2026-09-05
**Statut** : Acceptée

## Contexte

Sur le train de la Phase 2, l'horizon effectif est dégénéré à 1 partout (aucune ligne masquée),
alors que 66,5 % du test l'est avec un horizon 1-7. Le modèle n'a jamais vu d'exemple d'horizon
>1 pendant l'entraînement. Le brief §4.2 propose de corriger ça en tirant, pour chaque ligne de
train, un nombre aléatoire de mois à masquer indépendamment par cellule.

## Options considérées

**Granularité du masquage** : indépendant par cellule (texte du brief) vs mois calendaire entier
pour (quasi) toutes les cellules à la fois (mécanisme réel identifié en Phase 1 : le taux de
masquage par mois de test est quasi binaire, ~0 % ou ~99,6-100 %, jamais intermédiaire ; confirmé
par les 22 mois totalement absents du train, qui coïncident avec les pannes GRACE documentées).

**Persistance** : calculer le masquage augmenté une fois et le figer dans un Parquet (comme les
autres features de la Phase 2) vs ne jamais le persister et le retirer à chaque appel (mode
dynamique du brief §5, initialement prévu pour des modèles itératifs).

## Décision

- **Mois calendaires entiers**, jamais cellule par cellule indépendamment. Le texte du brief §4.2
  est corrigé en conséquence dans `doc/PROJECT_BRIEF.md` (voir commit associé).
- **Mode dynamique uniquement, pas de version statique.** `mask_augmentation.py` n'écrit jamais
  de fichier : `pipeline.build_features(..., masking_config, rng)` recalcule tout en mémoire à
  chaque appel. Un `rng` différent → un tirage de mois masqués différent → un train différent.
  Motif : un entraînement exposé à plusieurs tirages différents (ensemble de modèles, ou callback
  d'époque pour un modèle itératif) généralise mieux qu'un entraînement sur un unique masque figé
  — décision de l'utilisateur, contre la proposition initiale de ce plan qui prévoyait un mode
  statique.
- **Portée de cette phase limitée au mécanisme réutilisable.** La vraie boucle multi-tirages
  (combien de fois appeler `build_features` avec quel `rng`, comment ensembler ou brancher sur des
  époques) est laissée à la Phase 4/5, qui n'existe pas encore — cohérent avec la séparation des
  phases du brief (Phase 3 = feature engineering, Phase 4/5 = validation/modélisation).
- **Calibration de fréquence** : forme historique (croissante dans le temps, ~13,8 % des mois du
  train sont de vrais trous, concentrés après 2011) mais échelle rapprochée du régime du test en
  fin de train (67 % de ses mois sont des trous) — sinon le modèle serait sous-exposé au régime
  qu'il affronte réellement à l'inférence. Exposée en config
  (`configs/features.yaml: masking.target_gap_rate_by_period`), pas figée en dur : le brief prévoit
  déjà une itération de modélisation dédiée ("Phase 5, itération E") pour l'évaluer et l'ajuster.
- **Rétrocompatibilité** : `build_features` sans `masking_config`/`rng` (défaut) reproduit
  exactement le comportement Phase 2 — `dvc repro feature_engineering` et les 11 tests existants
  restent inchangés, vérifié après implémentation (seules les dépendances de `dvc.lock` ont
  changé, pas les sorties).

## Justification

- Le mécanisme par mois entier évite qu'un modèle apprenne une béquille qui n'existe jamais en
  vrai ("mes voisins ont des données même quand je n'en ai pas") — risque qui deviendrait actif
  dès que la Phase 5 ajoutera des features de voisinage spatial (actuellement absentes).
- Ne jamais persister le masquage force la Phase 4/5 à construire son harnais d'entraînement
  autour de tirages multiples dès le départ, plutôt que de devoir migrer plus tard depuis un
  pipeline habitué à un seul train figé.

## Limite assumée

L'histogramme d'horizon résultant sur le train n'est pas calé pixel par pixel sur celui du test
(effet émergent du choix des mois coupés, pas contrôlé ligne à ligne) — accepté comme un point de
départ raisonnable à affiner en Phase 5 plutôt qu'une cible à atteindre exactement maintenant.
