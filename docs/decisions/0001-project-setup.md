# ADR 0001 — Mise en place du projet (Phase 0)

**Date** : 2026-09-03
**Statut** : Acceptée

## Contexte

Démarrage du challenge Zindi "TWS Forecasting". Le brief méthodologique
(`doc/PROJECT_BRIEF.md`) impose : versioning des données (DVC), tracking des expériences
(MLflow), environnement Python figé, repo Git structuré.

## Options considérées

**Environnement Python** : installation globale vs `.venv` dédié.
**Repo GitHub** : public dès le départ vs privé puis public au moment de la publication.
**Remote DVC** : stockage local vs cloud (Google Drive / S3).

## Décision

- `.venv` dédié au projet, dépendances figées via `pip freeze` → `requirements.txt`.
- Repo GitHub **privé** créé immédiatement sous le compte `dimi1441`, passage en public
  différé à la Phase 8 (publication), une fois le code et les résultats stabilisés.
- Remote DVC **local** (dossier hors du repo Git, sur disque local) : aucune dépendance à un
  compte cloud externe pour commencer, données volumineuses (Train.csv ~276 Mo) versionnées
  sans coût ni configuration réseau.

## Justification

- `.venv` isole les dépendances du projet du reste de l'environnement Python de la machine
  (plusieurs installations Python 3.11/3.12 coexistent) — condition de reproductibilité
  explicitement demandée (section 8.1 du brief).
- Repo privé d'abord évite d'exposer du code de compétition en cours de développement,
  tout en respectant l'obligation finale de publication publique (règlement du challenge).
- Remote DVC local est la solution la plus rapide à mettre en place ; un remote cloud
  pourra être ajouté ultérieurement sans perte d'historique DVC si la collaboration ou la
  sauvegarde distante devient nécessaire.

## Limite assumée

Le remote DVC local ne protège pas contre une perte de la machine locale (pas de sauvegarde
distante). Acceptable pour l'instant ; à revisiter si le projet devient critique ou collaboratif.
