"""Masquage augmenté du train — mode dynamique uniquement.

Simule des mois GRACE entiers indisponibles (fidèle au vrai mécanisme identifié en Phase 1 :
masquage par mois calendaire pour ~toutes les cellules à la fois, jamais cellule par cellule
indépendamment). Aucune version figée sur disque : chaque appel avec un `rng` différent produit
un tirage différent, à recalculer à chaque fois par l'appelant (voir `pipeline.build_features`).
"""

import pandas as pd


def _rate_for_year(year: int, target_gap_rate_by_period: list[dict]) -> float:
    for period in target_gap_rate_by_period:
        if period["start_year"] <= year <= period["end_year"]:
            return period["rate"]
    return 0.0


def scale_gap_rate_by_period(
    target_gap_rate_by_period: list[dict], multiplier: float, cap: float = 0.9
) -> list[dict]:
    """Multiplie chaque taux de période par `multiplier`, plafonné à `cap` (demande utilisateur
    du 2026-09-07 : varier la difficulté du masquage par membre du bag, pas seulement les mois
    tirés, pour se rapprocher du régime bien plus dur du vrai test — 67% de ses mois sont des
    trous contre ~10-16% vus par les membres actuels à taux fixe). Le plafond évite qu'un membre
    à fort multiplicateur se retrouve avec quasiment plus d'historique du tout sur les dernières
    années."""
    return [
        {**period, "rate": min(period["rate"] * multiplier, cap)}
        for period in target_gap_rate_by_period
    ]


def select_gap_months(
    existing_months: pd.Series, target_gap_rate_by_period: list[dict], rng
) -> set[pd.Timestamp]:
    """Tire les mois calendaires qui deviennent des trous simulés.

    Un tirage de Bernoulli indépendant par mois existant, au taux de la période qui contient son
    année (`configs/features.yaml: masking.target_gap_rate_by_period`). Un mois absent de
    `existing_months` ne peut évidemment pas être tiré (rien à masquer).
    """
    months = pd.DatetimeIndex(sorted(set(existing_months)))
    gap_months = set()
    for month in months:
        rate = _rate_for_year(month.year, target_gap_rate_by_period)
        if rng.random() < rate:
            gap_months.add(month)
    return gap_months


def apply_augmented_masking(panel_df: pd.DataFrame, gap_months: set[pd.Timestamp]) -> pd.DataFrame:
    """Masque `TWS_t` pour toutes les lignes de train dont le mois est un trou simulé.

    Ne touche jamais les lignes de test (masquage réel, jamais réécrit) ni `target` (la vraie
    valeur à prédire reste toujours connue en train — seule la feature `TWS_t` du mois masqué
    devient artificiellement inconnue, exactement comme un vrai trou GRACE le ferait).
    """
    df = panel_df.copy()
    is_gap_month = df["time"].isin(gap_months)
    to_mask = df["is_train"] & is_gap_month
    df.loc[to_mask, "TWS_t"] = float("nan")
    df.loc[to_mask, "TWS_t_masked"] = True
    return df
