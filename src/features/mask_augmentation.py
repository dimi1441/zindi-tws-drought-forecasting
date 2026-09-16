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


def find_consecutive_streaks(gap_months: set[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    """Regroupe un ensemble de mois-trous en rafales de mois consécutifs, triées par date.

    Sert de base aux experts par horizon (2026-09-09) : Modèle A plafonne chaque rafale à 3 mois
    (régime "normal", proche des vraies rafales du train -- jamais plus de 3 mois consécutifs
    observés historiquement), Modèle B en prolonge une partie pour viser 4-7 mois (régime du
    test, où une seule vraie rafale -- la transition GRACE/GRACE-FO -- atteint 6 mois).
    """
    months = sorted(gap_months)
    streaks: list[list[pd.Timestamp]] = []
    current: list[pd.Timestamp] = []
    for month in months:
        if current and (month.year - current[-1].year) * 12 + (month.month - current[-1].month) == 1:
            current.append(month)
        else:
            if current:
                streaks.append(current)
            current = [month]
    if current:
        streaks.append(current)
    return streaks


def cap_streak_lengths(gap_months: set[pd.Timestamp], max_length: int) -> set[pd.Timestamp]:
    """Retire les mois excédentaires d'une rafale qui dépasserait `max_length` mois consécutifs
    (garde les `max_length` premiers mois de chaque rafale, démasque le reste). Sert au régime
    "normal" de Modèle A (2026-09-09) : empêche qu'un tirage produise par hasard une rafale plus
    longue que ce que le train a réellement connu (jamais plus de 3 mois consécutifs, cf.
    JOURNAL.md 2026-09-09)."""
    return {month for streak in find_consecutive_streaks(gap_months) for month in streak[:max_length]}


def extend_gap_streaks(
    gap_months: set[pd.Timestamp],
    existing_months: pd.Series,
    target_lengths: list[int],
    rng,
) -> set[pd.Timestamp]:
    """Prolonge chaque rafale de `gap_months` jusqu'à une longueur cible tirée dans
    `target_lengths` (ex. `[4, 5, 6, 7]` pour matcher la distribution d'horizon du test), en
    ajoutant les mois consécutifs suivants réellement présents dans `existing_months`. Sert au
    régime "dur" de Modèle B (2026-09-09) -- même emplacement de rafale que Modèle A (même
    tirage de base), juste étendu, pour préserver la diversité saisonnière naturelle des rafales
    plutôt que de choisir des positions arbitraires.

    Une rafale trop proche de la fin de l'historique (pas assez de mois suivants) ou dont
    l'extension chevaucherait une autre rafale déjà masquée est laissée inchangée -- jamais de
    fusion accidentelle de deux rafales distinctes.
    """
    all_months = pd.DatetimeIndex(sorted(set(existing_months)))
    streaks = find_consecutive_streaks(gap_months)
    extended = set(gap_months)

    for streak in streaks:
        target_length = int(rng.choice(target_lengths))
        n_to_add = target_length - len(streak)
        if n_to_add <= 0:
            continue

        last_month = streak[-1]
        position = all_months.get_loc(last_month)
        candidates = all_months[position + 1 : position + 1 + n_to_add]
        if len(candidates) < n_to_add or any(c in gap_months for c in candidates):
            continue  # pas assez de mois restants, ou chevauchement avec une autre rafale

        extended.update(candidates)

    return extended


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
