"""Baselines Phase 4 : persistance + fallback climatologie (formule pure), et GBR simple
(réplique le modèle du starter) — évaluées à travers le harnais de validation.
"""

import time
from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Mêmes features que `00_starter_baseline.ipynb::feature_names` — comparaison directe possible
# avec la référence connue (MAE 0.429, RMSE 0.607, R² 0.460 sur un split temporel non masqué).
SIMPLE_FEATURES = [
    "TWS_t",
    "month_sin",
    "month_cos",
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
]


def predict_persistence_climatology(df: pd.DataFrame) -> np.ndarray:
    """Persistance pure (`last_observed_tws`), fallback sur la climatologie cellule/mois si le
    dernier point connu n'a pas encore de valeur (tout début d'une série). Aucun entraînement."""
    prediction = df["last_observed_tws"].fillna(df["TWS_t_climatology_mean"])
    return prediction.to_numpy()


def make_gbr_pipeline() -> Pipeline:
    """Reprend `make_model()` du starter (`00_starter_baseline.ipynb`) à l'identique — mêmes
    hyperparamètres, quel que soit le jeu de features utilisé."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "gbr",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=300,
                    max_depth=8,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_predict_gbr(
    fit_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    """Entraîne `make_gbr_pipeline()` sur `fit_df` et prédit sur `val_df`, pour un jeu de
    features quelconque (ex : `SIMPLE_FEATURES` ou `configs/feature_columns.yaml` en entier)."""
    model = make_gbr_pipeline()
    X_fit = fit_df[feature_columns].to_numpy(dtype=np.float32)
    y_fit = fit_df["target"].to_numpy(dtype=np.float32)
    X_val = val_df[feature_columns].to_numpy(dtype=np.float32)

    model.fit(X_fit, y_fit)
    return model.predict(X_val)


def fit_predict_simple_gbr(fit_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """`fit_predict_gbr` restreint aux 8 features simples du starter — réplique exacte pour
    comparaison directe avec la référence connue (MAE 0.429, RMSE 0.607, R² 0.460)."""
    return fit_predict_gbr(fit_df, val_df, SIMPLE_FEATURES)


def fit_predict_bagged_gbr_members(
    fit_dfs: list[pd.DataFrame], val_dfs: list[pd.DataFrame], feature_columns: list[str]
) -> tuple[np.ndarray, list[float]]:
    """Comme `fit_predict_bagged_gbr`, mais retourne les prédictions de chaque membre séparément
    (shape `(n_models, n_val)`) plutôt que leur moyenne, plus le temps fit+predict de chacun.
    Sert à évaluer des moyennes cumulatives sur les N premiers membres sans réentraîner — courbe
    de bagging, Phase 5 itération E (cf. `run_bagging_curve.py`)."""
    predictions = []
    seconds = []
    for fit_df, val_df in zip(fit_dfs, val_dfs):
        start = time.perf_counter()
        predictions.append(fit_predict_gbr(fit_df, val_df, feature_columns))
        seconds.append(time.perf_counter() - start)
    return np.stack(predictions, axis=0), seconds


def fit_predict_bagged_gbr(
    fit_dfs: list[pd.DataFrame], val_dfs: list[pd.DataFrame], feature_columns: list[str]
) -> np.ndarray:
    """Bagging (Phase 5, demande utilisateur explicite du 2026-09-05) : moyenne des prédictions
    de plusieurs `make_gbr_pipeline()` (mêmes hyperparamètres que le starter, `random_state=42`
    fixe sur chaque membre — la diversité vient du masquage augmenté, pas du hasard interne du
    modèle, pour isoler l'effet mesuré). `fit_dfs`/`val_dfs` : une paire par tirage de masquage
    (même lignes/mêmes mois pour tous les tirages, seules `TWS_t` et les features dérivées
    diffèrent) — construites par l'appelant en appliquant le même masque fit/val à chaque
    `build_features(..., rng=seed_i)`.
    """
    predictions, _ = fit_predict_bagged_gbr_members(fit_dfs, val_dfs, feature_columns)
    return predictions.mean(axis=0)


def fit_predict_lightgbm_early_stopping(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    inner_holdout_fn: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]],
    n_estimators: int = 3000,
    early_stopping_rounds: int = 50,
    return_model: bool = False,
):
    """LightGBM avec un vrai early stopping calé sur nos propres folds, pas le tirage aléatoire
    interne de sklearn (`HistGradientBoostingRegressor(early_stopping="auto")`, qui pioche 10 %
    de `fit_df` sans respecter la chronologie ni les blocs spatiaux).

    `inner_holdout_fn` découpe `fit_df` en (train interne, validation interne pour l'arrêt
    anticipé) — jamais `val_df`, qui reste intact pour le calcul final de la métrique. Sinon on
    entraînerait ET on évaluerait sur le même jeu, ce qui biaiserait l'estimation en optimiste.
    Mêmes hyperparamètres que `make_gbr_pipeline()` (max_depth, min_samples/child, L2) pour isoler
    l'effet du passage à plus d'itérations + un early stopping propre.

    `return_model=True` retourne `(predictions, model)` au lieu de `predictions` seul — utile pour
    inspecter `model.booster_.best_iteration` (ex : génération de soumission).
    """
    inner_train_mask, inner_val_mask = inner_holdout_fn(fit_df)
    inner_fit = fit_df.loc[inner_train_mask]
    inner_val = fit_df.loc[inner_val_mask]

    X_fit = inner_fit[feature_columns].to_numpy(dtype=np.float32)
    y_fit = inner_fit["target"].to_numpy(dtype=np.float32)
    X_inner_val = inner_val[feature_columns].to_numpy(dtype=np.float32)
    y_inner_val = inner_val["target"].to_numpy(dtype=np.float32)
    X_val = val_df[feature_columns].to_numpy(dtype=np.float32)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=8,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_inner_val, y_inner_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    predictions = model.predict(X_val)
    if return_model:
        return predictions, model
    return predictions
