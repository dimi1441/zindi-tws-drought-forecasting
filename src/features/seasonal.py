"""Climatologie (cellule x mois calendaire) et anomalie saisonnière, sans fuite temporelle.

La climatologie d'une ligne n'utilise que les années strictement antérieures pour la même
cellule et le même mois calendaire (`expanding().mean().shift(1)`). Un mois calendaire absent
d'une année donnée (pas de ligne) ou masqué (`TWS_t` = NaN) est naturellement ignoré par
`expanding().mean()`, qui saute les NaN sans les propager — pas de traitement spécial requis.
"""

import pandas as pd


def add_climatology_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `TWS_t_climatology_mean` et `TWS_t_seasonal_anomaly` au panel.

    Suppose `panel_df` déjà trié par (lat, lon, time) : au sein d'un groupe (cellule, mois
    calendaire), les lignes apparaissent alors naturellement dans l'ordre des années croissantes,
    condition nécessaire pour que `expanding().shift(1)` exclue bien l'année courante.
    """
    df = panel_df.copy()
    month_of_year = df["time"].dt.month

    grouped = df.groupby(["lat", "lon", month_of_year], sort=False)["TWS_t"]
    df["TWS_t_climatology_mean"] = grouped.transform(
        lambda s: s.expanding().mean().shift(1)
    )
    df["TWS_t_seasonal_anomaly"] = df["TWS_t"] - df["TWS_t_climatology_mean"]

    return df
