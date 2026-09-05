"""Baselines Phase 4 : persistance + fallback climatologie (formule pure), et GBR simple
(réplique le modèle du starter) — évaluées à travers le harnais de validation.
"""

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


def make_simple_gbr() -> Pipeline:
    """Reprend `make_model()` du starter (`00_starter_baseline.ipynb`) à l'identique."""
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


def fit_predict_simple_gbr(fit_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """Entraîne `make_simple_gbr()` sur `fit_df` et prédit sur `val_df`."""
    model = make_simple_gbr()
    X_fit = fit_df[SIMPLE_FEATURES].to_numpy(dtype=np.float32)
    y_fit = fit_df["target"].to_numpy(dtype=np.float32)
    X_val = val_df[SIMPLE_FEATURES].to_numpy(dtype=np.float32)

    model.fit(X_fit, y_fit)
    return model.predict(X_val)
