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

**Récidive constatée le 2026-09-19** : même oubli, 3 fichiers (`train_features.parquet`,
`test_features.parquet`, `submissions/submission.csv`) jamais poussés malgré cette règle déjà
écrite ici. La documentation seule n'a pas suffi — vérifier systématiquement avec `dvc data
status --not-in-remote` après chaque `git push`, pas seulement s'en souvenir.

## MLflow

Tout script qui lance une expérimentation (recherche d'hyperparamètres, comparaison de features,
analyse d'erreur, tout ce qui boucle sur des folds/configs pour produire une métrique) doit logger
vers MLflow avec la même convention que `run_baselines.py`/`run_bagging.py` :
`mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])` +
`mlflow.set_experiment(base_config["mlflow"]["experiment_name"])` une fois, puis un
`mlflow.start_run(run_name=...)` par expérimentation/config avec `log_param(s)`, `log_metrics`,
et `log_artifact` pour tout CSV sauvé dans `reports/`. Ne pas se contenter d'un `print()` +
`to_csv()` — même pour un script jugé "juste une vérification rapide".

**Oubli constaté le 2026-09-19** : `search_hyperparameters.py`, `feature_importance.py` et
`error_analysis.py` (écrits lors d'une session antérieure) sauvegardaient leurs résultats
uniquement en CSV dans `reports/`, sans jamais appeler `mlflow`, contrairement à la convention
déjà en place ailleurs dans le projet. Corrigé et rattrapé dans `mlflow.db` (runs tagués
`backfilled=true`) — voir `JOURNAL.md` à cette date et commit `04351f4`.

## Journal de projet

Toute décision non triviale se consigne dans `JOURNAL.md` (brief §8.2), avant ou juste après le
travail correspondant — pas seulement dans les messages de commit.
