import json
from pathlib import Path
from datetime import datetime, timezone

CLEAN_PATH = Path("../data/clean_events.json")

# Date minimale autorisée
MIN_DATE = datetime(2025, 2, 17, tzinfo=timezone.utc)

# -------------------------------
# Fixture pour charger les données
# -------------------------------
import pytest

@pytest.fixture
def data():
    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------------
# 1️⃣ Dataset non vide
# -------------------------------
def test_dataset_not_empty(data):
    assert len(data) > 0, "Le dataset est vide"
    print(" Dataset non vide", len(data))

# -------------------------------
# 2️⃣ Tous les événements sont à Paris
# -------------------------------
def test_city_is_paris(data):
    for event in data:
        city = event.get("location_city", "")
        assert city == "Paris", f"Ville incorrecte trouvée : {city}"
    print(" Tous les événements sont à Paris")

# -------------------------------
# 3️⃣ Toutes les dates sont récentes
# -------------------------------
def test_dates_within_one_year(data):
    for event in data:
        date_str = event.get("firstdate_begin")

        assert date_str, "Date manquante"

        try:
            event_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            raise AssertionError(f"Date invalide : {date_str}")

        assert event_date >= MIN_DATE, f"Événement trop ancien : {event_date}"
    print(" Toutes les dates sont dans la limite des 12 derniers mois")