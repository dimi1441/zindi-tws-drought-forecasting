"""Vérifie les baselines : fallback persistance->climatologie, et que le GBR simple s'entraîne."""

import numpy as np
import pandas as pd

from src.validation.baselines import (
    SIMPLE_FEATURES,
    fit_predict_simple_gbr,
    predict_persistence_climatology,
)


def test_persistence_uses_last_observed_when_available():
    df = pd.DataFrame({"last_observed_tws": [0.5, np.nan], "TWS_t_climatology_mean": [0.9, 0.9]})
    prediction = predict_persistence_climatology(df)
    assert prediction[0] == 0.5


def test_persistence_falls_back_to_climatology_when_missing():
    df = pd.DataFrame({"last_observed_tws": [0.5, np.nan], "TWS_t_climatology_mean": [0.9, 0.7]})
    prediction = predict_persistence_climatology(df)
    assert prediction[1] == 0.7


def _synthetic_fit_val():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame(
        {col: rng.normal(size=n) for col in SIMPLE_FEATURES}
    )
    df["target"] = df["TWS_t"] + rng.normal(scale=0.01, size=n)
    return df.iloc[:150].reset_index(drop=True), df.iloc[150:].reset_index(drop=True)


def test_fit_predict_simple_gbr_runs_and_predicts_reasonable_shape():
    fit_df, val_df = _synthetic_fit_val()
    y_pred = fit_predict_simple_gbr(fit_df, val_df)
    assert y_pred.shape == (len(val_df),)
    # Le signal de test est TWS_t + bruit faible : le modele doit largement battre une moyenne
    # naive (variance quasi nulle attendue sur l'erreur si le modele a bien appris TWS_t ~ target).
    mae = np.abs(y_pred - val_df["target"].to_numpy()).mean()
    assert mae < 0.5
