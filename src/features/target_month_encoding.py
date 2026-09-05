"""Encodage cyclique du mois calendaire de `t+1` — seule information sur le futur autorisée.

Le calendrier est connu à l'avance (ce n'est pas une observation), donc le mois de `t+1` peut
être utilisé sans enfreindre la règle anti-leakage (brief §3.1).
"""

import numpy as np
import pandas as pd


def add_target_month_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `target_month_sin` et `target_month_cos` (mois calendaire de `t+1`)."""
    df = df.copy()
    target_month = (df["time"] + pd.DateOffset(months=1)).dt.month
    df["target_month_sin"] = np.sin(2 * np.pi * target_month / 12)
    df["target_month_cos"] = np.cos(2 * np.pi * target_month / 12)
    return df
