"""Vérifie les baselines : fallback persistance->climatologie, et que les GBR s'entraînent."""

import numpy as np
import pandas as pd

from src.validation.baselines import (
    SIMPLE_FEATURES,
    fit_predict_bagged_gbr,
    fit_predict_gbr,
    fit_predict_lightgbm_early_stopping,
    fit_predict_simple_gbr,
    predict_persistence_climatology,
)
from src.validation.splits import temporal_holdout


def test_persistence_uses_last_observed_when_available():
    df = pd.DataFrame({"last_observed_tws": [0.5, np.nan], "TWS_t_climatology_mean": [0.9, 0.9]})
    prediction = predict_persistence_climatology(df)
    assert prediction[0] == 0.5


def test_persistence_falls_back_to_climatology_when_missing():
    df = pd.DataFrame({"last_observed_tws": [0.5, np.nan], "TWS_t_climatology_mean": [0.9, 0.7]})
    prediction = predict_persistence_climatology(df)
    assert prediction[1] == 0.7


def _synthetic_fit_val(n: int = 200, n_months: int = 20):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({col: rng.normal(size=n) for col in SIMPLE_FEATURES})
    df["target"] = df["TWS_t"] + rng.normal(scale=0.01, size=n)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    df["time"] = rng.choice(months, size=n)
    split = int(n * 0.75)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def test_fit_predict_simple_gbr_runs_and_predicts_reasonable_shape():
    fit_df, val_df = _synthetic_fit_val()
    y_pred = fit_predict_simple_gbr(fit_df, val_df)
    assert y_pred.shape == (len(val_df),)
    # Le signal de test est TWS_t + bruit faible : le modele doit largement battre une moyenne
    # naive (variance quasi nulle attendue sur l'erreur si le modele a bien appris TWS_t ~ target).
    mae = np.abs(y_pred - val_df["target"].to_numpy()).mean()
    assert mae < 0.5


def test_fit_predict_lightgbm_early_stopping_runs_and_never_trains_on_val_df():
    fit_df, val_df = _synthetic_fit_val(n=2000, n_months=40)
    y_pred = fit_predict_lightgbm_early_stopping(
        fit_df, val_df, SIMPLE_FEATURES, temporal_holdout,
        n_estimators=200, early_stopping_rounds=10,
    )
    assert y_pred.shape == (len(val_df),)
    mae = np.abs(y_pred - val_df["target"].to_numpy()).mean()
    assert mae < 0.5


def test_fit_predict_bagged_gbr_averages_individual_member_predictions():
    # Deux tirages "de masquage" différents simulés par deux jeux fit/val distincts (mêmes
    # colonnes, valeurs différentes) : le bag doit renvoyer exactement la moyenne des deux GBR
    # entraînés séparément avec les mêmes hyperparamètres (make_gbr_pipeline, random_state=42 fixe).
    fit_df_a, val_df_a = _synthetic_fit_val(n=200, n_months=20)
    fit_df_b, val_df_b = _synthetic_fit_val(n=200, n_months=20)

    bagged_pred = fit_predict_bagged_gbr(
        [fit_df_a, fit_df_b], [val_df_a, val_df_b], SIMPLE_FEATURES
    )
    pred_a = fit_predict_gbr(fit_df_a, val_df_a, SIMPLE_FEATURES)
    pred_b = fit_predict_gbr(fit_df_b, val_df_b, SIMPLE_FEATURES)

    assert bagged_pred.shape == (len(val_df_a),)
    np.testing.assert_allclose(bagged_pred, (pred_a + pred_b) / 2)
