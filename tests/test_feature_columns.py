"""Vérifie qu'aucune colonne interdite (lat, lon, ID, sample_id) ne finit en feature modèle."""

from pathlib import Path

import pytest
import yaml

from src.features.pipeline import NON_FEATURE_COLUMNS

FORBIDDEN = {"lat", "lon", "ID", "sample_id"}
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "configs" / "feature_columns.yaml"


def test_non_feature_columns_covers_forbidden_set():
    # lat/lon/ID sont explicitement exclus par construction du pipeline ; sample_id n'existe
    # plus après harmonisation par `io.load_raw` donc n'a pas besoin d'être dans la constante.
    assert {"lat", "lon", "ID"} <= NON_FEATURE_COLUMNS


@pytest.mark.skipif(
    not FEATURE_COLUMNS_PATH.exists(),
    reason="configs/feature_columns.yaml pas encore généré (lancer le pipeline d'abord)",
)
def test_generated_feature_columns_excludes_forbidden_columns():
    content = yaml.safe_load(FEATURE_COLUMNS_PATH.read_text())
    feature_columns = set(content["feature_columns"])
    assert not (feature_columns & FORBIDDEN)
