"""Vérifie le voisinage spatial : identification des voisins directs sur une grille irrégulière,
propagation de la dernière valeur connue du voisin (pas sa valeur brute à `t`), et causalité."""

import numpy as np
import pandas as pd

from src.features.horizon_features import add_horizon_features
from src.features.spatial_neighborhood import (
    add_spatial_neighborhood_features,
    build_direct_neighbor_lookup,
)


def test_direct_neighbor_lookup_on_irregular_grid_with_gaps():
    # Grille avec un trou délibéré (aucune cellule à lat=1.5,lon=1.5) pour vérifier que le
    # "voisin direct" saute bien au prochain existant plutôt que de supposer un voisin fictif.
    cells = pd.DataFrame(
        {
            "lat": [0.5, 1.5, 2.5, 0.5, 0.5],
            "lon": [0.5, 0.5, 0.5, 1.5, 2.5],
        }
    )
    lookup = build_direct_neighbor_lookup(cells).set_index(["lat", "lon"])

    # (0.5, 0.5) : nord = (1.5,0.5), est = (0.5,1.5), pas de sud/ouest.
    row = lookup.loc[(0.5, 0.5)]
    assert (row["lat_N"], row["lon_N"]) == (1.5, 0.5)
    assert (row["lat_E"], row["lon_E"]) == (0.5, 1.5)
    assert pd.isna(row["lat_S"]) and pd.isna(row["lon_W"])

    # (1.5, 0.5) : sud = (0.5,0.5), nord = (2.5,0.5), pas de voisin est/ouest (colonne isolée).
    row = lookup.loc[(1.5, 0.5)]
    assert (row["lat_S"], row["lon_S"]) == (0.5, 0.5)
    assert (row["lat_N"], row["lon_N"]) == (2.5, 0.5)
    assert pd.isna(row["lon_E"]) and pd.isna(row["lon_W"])

    # (0.5, 2.5) : ouest = (0.5,1.5), pas de voisin est (bord de la zone).
    row = lookup.loc[(0.5, 2.5)]
    assert (row["lat_W"], row["lon_W"]) == (0.5, 1.5)
    assert pd.isna(row["lon_E"])


def _three_cell_panel() -> pd.DataFrame:
    # A=(0.5,0.5) au centre, B=(1.5,0.5) voisin nord de A, C=(0.5,1.5) voisin est de A.
    # Ni B ni C n'ont de voisin autre que A (bord de zone) -- vérifie les NaN attendus.
    times = pd.date_range("2020-01-01", periods=3, freq="MS")
    rows = []
    for lat, lon, tws, masked in [
        (0.5, 0.5, [1.0, None, None], [False, True, True]),
        (1.5, 0.5, [2.0, 2.1, 2.2], [False, False, False]),
        (0.5, 1.5, [3.0, None, 3.2], [False, True, False]),
    ]:
        for t, v, m in zip(times, tws, masked):
            rows.append({"lat": lat, "lon": lon, "time": t, "TWS_t": v, "TWS_t_masked": m})
    return pd.DataFrame(rows)


def test_neighbor_features_use_last_observed_not_raw_value_at_t():
    panel = add_horizon_features(_three_cell_panel())
    result = add_spatial_neighborhood_features(panel)

    cell_a = result[(result["lat"] == 0.5) & (result["lon"] == 0.5)].sort_values("time")

    # Voisin nord de A = B : dernière valeur connue de B à chaque mois (B n'est jamais masqué).
    assert cell_a["TWS_neighbor_N_last_observed"].tolist() == [2.0, 2.1, 2.2]
    assert cell_a["TWS_neighbor_N_months_since_last_observed"].tolist() == [1, 1, 1]

    # Voisin est de A = C : C est masqué au mois 2 -> dernière valeur connue reste celle du mois 1.
    assert cell_a["TWS_neighbor_E_last_observed"].tolist() == [3.0, 3.0, 3.2]
    assert cell_a["TWS_neighbor_E_months_since_last_observed"].tolist() == [1, 2, 1]

    # Pas de voisin sud/ouest pour A -> NaN, jamais une valeur inventée.
    assert cell_a["TWS_neighbor_S_last_observed"].isna().all()
    assert cell_a["TWS_neighbor_W_last_observed"].isna().all()


def test_neighbor_features_never_leak_neighbor_raw_value_when_neighbor_is_masked():
    # A n'est jamais masqué, mais son voisin est (C) l'est au mois 2 -- la feature de A ne doit
    # jamais contenir 3.1 (une valeur imaginaire) ni la vraie valeur future si C avait été
    # démasqué plus tard : seule la dernière valeur PASSÉE connue de C doit apparaître.
    panel = add_horizon_features(_three_cell_panel())
    result = add_spatial_neighborhood_features(panel)
    cell_a = result[(result["lat"] == 0.5) & (result["lon"] == 0.5)].sort_values("time")
    assert cell_a["TWS_neighbor_E_last_observed"].tolist() == [3.0, 3.0, 3.2]


def test_neighbor_features_are_causal_perturbing_later_mask_does_not_change_earlier_rows():
    # Preuve anti-fuite directe (même principe que les tests ANN du 2026-09-07) : changer si un
    # voisin est masqué à un mois TARDIF ne doit rien changer aux lignes ANTÉRIEURES de la
    # cellule qui le regarde.
    baseline_panel = _three_cell_panel()
    perturbed_panel = baseline_panel.copy()
    # Démasque C au mois 2 (dernière ligne du bloc C, cf. _three_cell_panel : mois index 1).
    is_c_month2 = (
        (perturbed_panel["lat"] == 0.5)
        & (perturbed_panel["lon"] == 1.5)
        & (perturbed_panel["time"] == pd.Timestamp("2020-02-01"))
    )
    perturbed_panel.loc[is_c_month2, "TWS_t_masked"] = False
    perturbed_panel.loc[is_c_month2, "TWS_t"] = 99.0

    baseline_result = add_spatial_neighborhood_features(add_horizon_features(baseline_panel))
    perturbed_result = add_spatial_neighborhood_features(add_horizon_features(perturbed_panel))

    def cell_a_before_month2(df: pd.DataFrame) -> pd.DataFrame:
        cell_a = df[(df["lat"] == 0.5) & (df["lon"] == 0.5)].sort_values("time")
        return cell_a[cell_a["time"] < pd.Timestamp("2020-02-01")]

    pd.testing.assert_frame_equal(
        cell_a_before_month2(baseline_result).reset_index(drop=True),
        cell_a_before_month2(perturbed_result).reset_index(drop=True),
    )
