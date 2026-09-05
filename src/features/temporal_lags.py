"""Lags, différences et moyennes mobiles, calculés par cellule (jamais sur le dataframe entier)."""

import pandas as pd

COVARIATE_COLUMNS = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]


def add_lag_features(panel_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Ajoute lags, différences et moyennes mobiles au panel (trié cellule/temps).

    Tout est calculé via `groupby(["lat", "lon"])` pour ne jamais mélanger deux cellules.
    Les moyennes mobiles incluent le point courant `t` (légitime, ce n'est pas une observation
    future) ; seul le futur (`t+1` et au-delà) est exclu par construction (fenêtre arrière).
    """
    df = panel_df.copy()
    grouped = df.groupby(["lat", "lon"], sort=False)

    tws_lags = config["lags"]["tws"]
    for k in tws_lags:
        df[f"TWS_t_lag{k}"] = grouped["TWS_t"].shift(k)

    if 1 in tws_lags:
        df["TWS_t_diff1"] = df["TWS_t"] - df["TWS_t_lag1"]
    if 12 in tws_lags:
        df["TWS_t_diff12"] = df["TWS_t"] - df["TWS_t_lag12"]

    for w in config["rolling_windows"]:
        df[f"TWS_t_rollmean{w}"] = grouped["TWS_t"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )

    covariate_lags = config["lags"]["covariates"]
    for col in COVARIATE_COLUMNS:
        if col not in df.columns:
            continue
        for k in covariate_lags:
            df[f"{col}_lag{k}"] = grouped[col].shift(k)

    return df
