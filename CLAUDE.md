# Instructions pour Claude Code — TWS Forecasting (Zindi)

Lire `doc/PROJECT_BRIEF.md` en priorité avant tout travail de code sur ce projet : contexte du
challenge, règles anti-leakage, roadmap complet par phase.

## Environnement Python

Toujours utiliser `.venv/Scripts/python.exe` (jamais le `python` du PATH global) pour lancer le
pipeline, les tests ou `dvc repro` — un interpréteur différent a été utilisé par erreur en début
de Phase 2 (a fonctionné par coïncidence, mais mauvais environnement).

## DVC

Après tout commit qui modifie un fichier suivi par DVC comme output (`data/processed/*.parquet`,
futurs modèles dans `models/`), exécuter **`dvc push`** vers le remote local en plus du commit
git — pas seulement committer `dvc.yaml`/`dvc.lock`.

**Oubli constaté sur les commits `fffda00` (Phase 2) et `5f50c46` (Phase 3)** : `dvc.lock` était
bien à jour et commité, mais les données correspondantes n'avaient jamais été poussées vers
`C:\Users\user\Documents\PERSO\ZINDI\dvc-storage-drought-zindi`. Rattrapé le 2026-09-05 (voir
`JOURNAL.md` à cette date). Vérifier avec `dvc data status --not-in-remote` en cas de doute.

## Journal de projet

Toute décision non triviale se consigne dans `JOURNAL.md` (brief §8.2), avant ou juste après le
travail correspondant — pas seulement dans les messages de commit.
