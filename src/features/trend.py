"""Tendance de long terme (ajout du 2026-09-07, demande utilisateur) : moyenne cumulée, sur tout
l'historique d'une cellule, des variations `TWS_t_diff1`/`TWS_t_diff12` déjà calculées par
`temporal_lags.add_lag_features`. Contrairement à la climatologie (qui regroupe par mois
calendaire), on moyenne ici sur TOUS les pas précédents sans distinction de mois — ça capture une
dérive de long terme (la cellule se dessèche/se recharge en moyenne dans le temps), complémentaire
au niveau saisonnier typique que donne la climatologie.

Chaque moyenne est accompagnée d'un compte (même mécanique causale, `expanding().count().shift(1)`)
: le nombre d'observations valides ayant servi à la moyenne, pour que le modèle puisse apprendre à
moins faire confiance à une tendance calculée sur peu de données (même logique que
`TWS_t_climatology_count`/`months_since_last_observed_tws`).
"""

import pandas as pd

# Colonne source (déjà produite par `temporal_lags.add_lag_features`) -> nom de la moyenne cumulée.
TREND_SOURCE_TO_MEAN_COLUMN = {
    "TWS_t_diff1": "TWS_t_diff1_expanding_mean",
    "TWS_t_diff12": "TWS_t_diff12_expanding_mean",
}


def add_trend_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `TWS_t_diff{1,12}_expanding_mean` et leurs `_count`.

    Suppose `panel_df` déjà trié par (lat, lon, time) et déjà passé par `add_lag_features`
    (colonnes `TWS_t_diff1`/`TWS_t_diff12` présentes). N'exclut que le pas courant de sa propre
    moyenne (`shift(1)`), jamais un mois calendaire précis — à la différence de la climatologie,
    aucun regroupement par mois ici.
    """
    df = panel_df.copy()
    grouped = df.groupby(["lat", "lon"], sort=False)

    for source_col, mean_col in TREND_SOURCE_TO_MEAN_COLUMN.items():
        count_col = f"{mean_col}_count"
        df[mean_col] = grouped[source_col].transform(lambda s: s.expanding().mean().shift(1))
        df[count_col] = grouped[source_col].transform(lambda s: s.expanding().count().shift(1))

    return df
