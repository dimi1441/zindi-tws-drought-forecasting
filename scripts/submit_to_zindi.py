"""Soumission directe à l'API Zindi, avec un timeout explicite (contournement du problème de
connexion signalé dans le chat de la compétition). Endpoint vérifié via le code source du paquet
tiers non officiel `KameniAlexNea/zindi` (github.com/KameniAlexNea/zindi/blob/master/zindi/
platform_api.py) -- pas la documentation officielle Zindi, donc à considérer comme "très
probablement correct" plutôt que garanti ; ajuster si la réponse renvoie une erreur inattendue.

Ne jamais committer un token en dur ici -- toujours via la variable d'environnement.

Usage :
    set ZINDI_AUTH_TOKEN=...   (PowerShell : $env:ZINDI_AUTH_TOKEN = "...")
    python scripts/submit_to_zindi.py --challenge-id <id> --file submissions/submission.csv --comment "Bag GBR 33 features"
"""

import argparse
import os
from pathlib import Path

import requests

API_BASE = "https://api.zindi.africa/v1/competitions"
DEFAULT_TIMEOUT_SECONDS = 900  # 15 min -- largement au-dessus du défaut habituel, pour une connexion lente/instable


def submit(challenge_id: str, file_path: Path, comment: str, timeout: int) -> None:
    auth_token = os.environ.get("ZINDI_AUTH_TOKEN")
    if not auth_token:
        raise SystemExit(
            "ZINDI_AUTH_TOKEN n'est pas défini. Le trouver sur zindi.africa (page de compte / "
            "onglet API de la compétition) et le mettre dans une variable d'environnement, "
            "jamais en dur dans ce fichier."
        )

    url = f"{API_BASE}/{challenge_id}/submissions"
    with open(file_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": (file_path.name, f, "text/csv")},
            data={"comment": comment},
            headers={"auth_token": auth_token},
            timeout=timeout,
        )

    print(response.status_code)
    print(response.text)
    response.raise_for_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-id", required=True, help="Identifiant de la compétition (visible dans l'URL zindi.africa/competitions/<challenge-id>)")
    parser.add_argument("--file", type=Path, default=Path("submissions/submission.csv"))
    parser.add_argument("--comment", default="Submission")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    submit(args.challenge_id, args.file, args.comment, args.timeout)
