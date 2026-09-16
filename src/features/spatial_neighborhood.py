"""Voisinage spatial (Phase 5 itération D, "levier majeur" du brief, différée 3 fois depuis la
Phase 2) : pour chaque cellule, ses 4 voisins directs (nord/sud/est/ouest), sous forme de
**dernière valeur connue** de chacun -- jamais la valeur brute du voisin à `t`.

**Pourquoi pas `TWS_t` du voisin directement** (discussion utilisateur du 2026-09-08/09) : le
masquage réel est un phénomène quasi-global par mois (Phase 1 -- chaque mois est masqué à ~0% ou
~99,6-100% des cellules, jamais entre les deux, panne satellite qui touche tout le monde en même
temps). Donc quand `TWS_t` d'une cellule est masqué (horizon>1, 66,5% du vrai test), son voisin
l'est presque certainement aussi, au même mois -- une colonne "valeur du voisin à t" serait NaN
pile quand on en a le plus besoin. Solution : réutiliser la même logique causale que
`last_observed_tws`/`months_since_last_observed_tws` (`horizon_features.py`), appliquée à chaque
voisin plutôt qu'à la cellule elle-même -- même pendant une panne synchronisée, chaque voisin
garde une dernière valeur connue différente (l'eau qu'il avait avant la panne), une vraie info
spatiale qui reste exploitable à tout horizon, juste de plus en plus ancienne.

**Pourquoi 4 voisins séparés plutôt qu'une moyenne** : une moyenne de voisinage impose une
hypothèse de symétrie (toutes les directions comptent pareil) avant même de voir les données. En
gardant les 4 directions comme colonnes séparées, un modèle d'arbres (GBR/LightGBM) peut
apprendre lui-même qu'une direction est plus informative qu'une autre -- pertinent si l'eau
ruisselle globalement dans une direction dominante (intuition utilisateur).

**Pourquoi un voisin "direct le plus proche existant" plutôt qu'un carré fixe 3x3/5x5 (proposition
initiale du brief)** : la grille réelle est irrégulière et sparse (140 latitudes espacées de 1°
mais 358 longitudes avec des trous ponctuels à 3°, seulement 31% de la grille théorique peuplée --
probablement les terres émergées seulement). Un décalage fixe en degrés manquerait souvent son
voisin réel ; on cherche à la place la cellule existante la plus proche dans chaque direction
cardinale (même longitude pour nord/sud, même latitude pour est/ouest).

Doit être appelé **après** `add_horizon_features` (dépend de `last_observed_tws`/
`months_since_last_observed_tws` déjà présents sur le panel).
"""

import pandas as pd

DIRECTIONS = ("N", "S", "E", "W")


def build_direct_neighbor_lookup(cells: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque cellule (lat, lon) unique, trouve son voisin direct dans chacune des 4
    directions cardinales : la cellule existante la plus proche dans la même colonne (nord/sud,
    même longitude) ou la même ligne (est/ouest, même latitude). `NaN` si aucun voisin n'existe
    dans cette direction (bord de la zone couverte, ex : côte, île isolée).

    Retourne une ligne par cellule unique, avec `lat_{dir}`/`lon_{dir}` pour chaque direction.
    """
    cells = cells[["lat", "lon"]].drop_duplicates().reset_index(drop=True)

    by_lon = cells.sort_values(["lon", "lat"]).copy()
    by_lon["lat_N"] = by_lon.groupby("lon")["lat"].shift(-1)
    by_lon["lat_S"] = by_lon.groupby("lon")["lat"].shift(1)

    by_lat = cells.sort_values(["lat", "lon"]).copy()
    by_lat["lon_E"] = by_lat.groupby("lat")["lon"].shift(-1)
    by_lat["lon_W"] = by_lat.groupby("lat")["lon"].shift(1)

    lookup = cells.merge(by_lon[["lat", "lon", "lat_N", "lat_S"]], on=["lat", "lon"], how="left")
    lookup = lookup.merge(by_lat[["lat", "lon", "lon_E", "lon_W"]], on=["lat", "lon"], how="left")

    # Nord/Sud gardent la même longitude que la cellule d'origine ; Est/Ouest la même latitude.
    lookup["lon_N"] = lookup["lon"]
    lookup["lon_S"] = lookup["lon"]
    lookup["lat_E"] = lookup["lat"]
    lookup["lat_W"] = lookup["lat"]

    return lookup


def add_spatial_neighborhood_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `TWS_neighbor_{N,S,E,W}_last_observed` et
    `TWS_neighbor_{N,S,E,W}_months_since_last_observed` (8 colonnes) au panel.

    Voisinage statique (ne dépend pas du temps) : calculé une fois sur les cellules uniques, puis
    joint à chaque ligne (cellule, mois). Réutilise `last_observed_tws`/
    `months_since_last_observed_tws` déjà calculés par `add_horizon_features` -- aucune nouvelle
    logique causale, juste un changement de cellule source (celle du voisin plutôt que soi-même).
    """
    df = panel_df.copy()
    neighbor_lookup = build_direct_neighbor_lookup(df)

    # Capturé une seule fois, avant la boucle : chaque direction lit les colonnes causales
    # d'origine de la cellule, jamais une colonne de voisin déjà ajoutée par une direction
    # précédente (pas de contamination croisée entre directions).
    self_observed = df[["lat", "lon", "time", "last_observed_tws", "months_since_last_observed_tws"]]

    for direction in DIRECTIONS:
        lat_col, lon_col = f"lat_{direction}", f"lon_{direction}"
        cell_to_neighbor = neighbor_lookup[["lat", "lon", lat_col, lon_col]].rename(
            columns={lat_col: "neighbor_lat", lon_col: "neighbor_lon"}
        )
        df = df.merge(cell_to_neighbor, on=["lat", "lon"], how="left")

        neighbor_values = self_observed.rename(
            columns={
                "lat": "neighbor_lat",
                "lon": "neighbor_lon",
                "last_observed_tws": f"TWS_neighbor_{direction}_last_observed",
                "months_since_last_observed_tws": (
                    f"TWS_neighbor_{direction}_months_since_last_observed"
                ),
            }
        )
        df = df.merge(neighbor_values, on=["neighbor_lat", "neighbor_lon", "time"], how="left")
        df = df.drop(columns=["neighbor_lat", "neighbor_lon"])

    return df
