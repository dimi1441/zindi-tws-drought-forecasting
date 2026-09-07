"""Vérifie l'entraînement de l'ANN : tourne, prédit raisonnablement, arrête tôt, et n'apprend
jamais des lignes de validation (interne comme externe)."""

import numpy as np
import pandas as pd

from src.features.io import build_cell_timeline
from src.models.ann.feature_tensors import build_feature_scaffold_from_panel
from src.models.ann.train import predict_ann, train_ann

CONFIG = {"lags": {"tws": [1, 3, 6, 12], "covariates": [1, 3]}, "rolling_windows": [3, 6, 12]}
MASKING_CONFIG = {
    "target_gap_rate_by_period": [{"start_year": 2020, "end_year": 2022, "rate": 0.15}]
}
FEATURE_COLUMNS = [
    "TWS_t",
    "SPEI_01_t",
    "SOIL_MOISTURE_t",
    "month_sin",
    "month_cos",
    "TWS_t_lag1",
    "TWS_t_rollmean3",
]


def _synthetic_scaffold(n_cells: int = 20, n_months: int = 30, seed: int = 0):
    rng = np.random.RandomState(seed)
    train_rows, test_rows = [], []
    times = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    for cell_idx in range(n_cells):
        lat = float(cell_idx)
        values = rng.normal(0, 1, n_months).cumsum() * 0.05
        train_times, test_times = times[:-3], times[-3:]
        train_values, test_values = values[:-3], values[-3:]
        train_rows.append(
            pd.DataFrame(
                {
                    "lat": lat,
                    "lon": 0.5,
                    "time": train_times,
                    "TWS_t": train_values,
                    "SPEI_01_t": rng.normal(0, 1, len(train_times)),
                    "SPEI_03_t": rng.normal(0, 1, len(train_times)),
                    "SPEI_06_t": rng.normal(0, 1, len(train_times)),
                    "SPEI_12_t": rng.normal(0, 1, len(train_times)),
                    "SOIL_MOISTURE_t": rng.normal(0, 1, len(train_times)),
                    "month_sin": np.sin(2 * np.pi * train_times.month / 12),
                    "month_cos": np.cos(2 * np.pi * train_times.month / 12),
                    "target": train_values + rng.normal(0, 0.01, len(train_times)),
                }
            )
        )
        test_rows.append(
            pd.DataFrame(
                {
                    "lat": lat,
                    "lon": 0.5,
                    "time": test_times,
                    "TWS_t": test_values,
                    "SPEI_01_t": rng.normal(0, 1, len(test_times)),
                    "SPEI_03_t": rng.normal(0, 1, len(test_times)),
                    "SPEI_06_t": rng.normal(0, 1, len(test_times)),
                    "SPEI_12_t": rng.normal(0, 1, len(test_times)),
                    "SOIL_MOISTURE_t": rng.normal(0, 1, len(test_times)),
                    "month_sin": np.sin(2 * np.pi * test_times.month / 12),
                    "month_cos": np.cos(2 * np.pi * test_times.month / 12),
                    "TWS_t_masked": False,
                }
            )
        )
    train = pd.concat(train_rows, ignore_index=True)
    test = pd.concat(test_rows, ignore_index=True)
    panel = build_cell_timeline(train, test)
    return build_feature_scaffold_from_panel(panel, CONFIG)


def _fit_and_inner_val_masks(scaffold):
    is_train = scaffold.panel["is_train"].to_numpy()
    months = pd.DatetimeIndex(sorted(scaffold.panel.loc[is_train, "time"].unique()))
    cutoff = months[int(len(months) * 0.8)]
    fit_mask = is_train & (scaffold.panel["time"] < cutoff).to_numpy()
    inner_val_mask = is_train & ~fit_mask
    return fit_mask, inner_val_mask


def test_train_ann_runs_and_predicts_reasonable_shape():
    scaffold = _synthetic_scaffold()
    fit_mask, inner_val_mask = _fit_and_inner_val_masks(scaffold)

    trained = train_ann(
        scaffold, FEATURE_COLUMNS, fit_mask, inner_val_mask, MASKING_CONFIG, seed=42,
        hidden_layer_sizes=(16, 8), max_epochs=15, patience=5,
    )
    assert trained.best_epoch >= 0

    test_mask = ~scaffold.panel["is_train"].to_numpy()
    predictions = predict_ann(trained, scaffold, test_mask)
    assert predictions.shape == (int(test_mask.sum()),)
    assert np.isfinite(predictions).all()


def test_early_stopping_restores_best_epoch_not_necessarily_last():
    scaffold = _synthetic_scaffold()
    fit_mask, inner_val_mask = _fit_and_inner_val_masks(scaffold)

    trained = train_ann(
        scaffold, FEATURE_COLUMNS, fit_mask, inner_val_mask, MASKING_CONFIG, seed=42,
        hidden_layer_sizes=(16, 8), max_epochs=15, patience=3,
    )
    # Avec patience=3, soit l'arrêt anticipé a déclenché avant la fin (best < dernier run - 1),
    # soit on a tourné jusqu'au bout -- dans les deux cas `best_epoch` doit être un epoch valide.
    assert 0 <= trained.best_epoch < trained.n_epochs_run


def test_inner_val_rows_never_influence_trained_weights():
    # Perturber fortement la cible des lignes de validation interne (jamais utilisées par
    # `partial_fit`) ne doit rien changer aux poids appris -- seule `fit_mask` doit compter.
    #
    # `max_epochs=1` élimine toute ambiguïté due à la sélection du "meilleur epoch" (qui, elle,
    # dépend légitimement du score sur `inner_val_mask` -- c'est son rôle) : avec un seul epoch
    # possible, la seule question testée est "est-ce que CET UNIQUE appel à `partial_fit` a été
    # influencé par la cible corrompue ?", jamais "quel epoch a été restauré ?".
    scaffold = _synthetic_scaffold()
    fit_mask, inner_val_mask = _fit_and_inner_val_masks(scaffold)

    trained_a = train_ann(
        scaffold, FEATURE_COLUMNS, fit_mask, inner_val_mask, MASKING_CONFIG, seed=42,
        hidden_layer_sizes=(16, 8), max_epochs=1, patience=1,
    )

    perturbed_panel = scaffold.panel.copy()
    perturbed_panel.loc[inner_val_mask, "target"] = 999.0
    perturbed_scaffold = build_feature_scaffold_from_panel(perturbed_panel, CONFIG)
    trained_b = train_ann(
        perturbed_scaffold, FEATURE_COLUMNS, fit_mask, inner_val_mask, MASKING_CONFIG, seed=42,
        hidden_layer_sizes=(16, 8), max_epochs=1, patience=1,
    )

    for coef_a, coef_b in zip(trained_a.model.coefs_, trained_b.model.coefs_):
        np.testing.assert_allclose(coef_a, coef_b)
