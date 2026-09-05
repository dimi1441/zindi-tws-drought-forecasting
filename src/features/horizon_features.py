"""Horizon effectif et dernière valeur `TWS_t` connue, calculés sans fuite.

Généralise la formule validée en Phase 1 (`horizon = (t+1) - dernier mois observé`), vectorisée
via `where` + `groupby(cellule).ffill()` au lieu de la boucle Python du notebook d'exploration
(correcte mais trop lente pour tourner sur le panel combiné train+test, ~2,4M lignes).

Réutilisable telle quelle par la Phase 3 : il suffit de lui passer le nom d'une colonne de
masquage augmenté (simulé sur le train) au lieu de la colonne de masquage réelle du test.
"""

import pandas as pd


def add_horizon_features(
    panel_df: pd.DataFrame, masked_col: str = "TWS_t_masked"
) -> pd.DataFrame:
    """Ajoute `months_since_last_observed_tws` et `last_observed_tws` au panel.

    Suppose `panel_df` trié par (lat, lon, time). Sur une cellule sans aucune ligne masquée
    (ex : train en Phase 2, avant le masquage augmenté de la Phase 3), le résultat est dégénéré
    mais cohérent : horizon = 1 partout, `last_observed_tws` = `TWS_t` de la ligne elle-même.
    """
    df = panel_df.copy()
    masked = df[masked_col].fillna(False).astype(bool)
    group_keys = [df["lat"], df["lon"]]

    last_observed_time = df["time"].where(~masked).groupby(group_keys).ffill()
    last_observed_tws = df["TWS_t"].where(~masked).groupby(group_keys).ffill()

    target_time = df["time"] + pd.DateOffset(months=1)
    months_since = (target_time.dt.year - last_observed_time.dt.year) * 12 + (
        target_time.dt.month - last_observed_time.dt.month
    )

    df["months_since_last_observed_tws"] = months_since
    df["last_observed_tws"] = last_observed_tws
    return df
