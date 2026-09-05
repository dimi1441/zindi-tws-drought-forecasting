"""Métriques de validation : MAE, RMSE, R² par fold, agrégées en moyenne ± écart-type."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    """Reprend `compute_metrics` du starter notebook (`00_starter_baseline.ipynb`) à l'identique."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Moyenne et écart-type de chaque métrique across folds (checklist anti-leakage §3.5 :
    rapporter moyenne ± écart-type par schéma de validation)."""
    keys = fold_metrics[0].keys()
    aggregated = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics])
        aggregated[f"{key}_mean"] = float(values.mean())
        aggregated[f"{key}_std"] = float(values.std())
    return aggregated
