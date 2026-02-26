# test_faiss_search.py

import json
import numpy as np
import pytest
import faiss
from pathlib import Path
from mistralai import Mistral
from dotenv import load_dotenv
import os

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = BASE_DIR / "data/faiss_index.index"
METADATA_PATH = BASE_DIR / "data/faiss_metadata.json"
MODEL_NAME = "mistral-embed"
TOP_K = 5

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("Clé API Mistral manquante")

client = Mistral(api_key=MISTRAL_API_KEY)

# -----------------------------
# FIXTURES
# -----------------------------
@pytest.fixture(scope="module")
def index():
    return faiss.read_index(str(FAISS_INDEX_PATH))

@pytest.fixture(scope="module")
def metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def embed_query(query: str) -> np.ndarray:
    response = client.embeddings.create(
        model=MODEL_NAME,
        inputs=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")

def search(query: str, index, metadata, top_k=TOP_K):
    query_vector = embed_query(query)
    distances, indices = index.search(np.array([query_vector]), top_k)
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        result = metadata[idx]
        result["distance"] = float(distance)
        results.append(result)
    return results

# -----------------------------
# TESTS
# -----------------------------
def test_search_returns_results(index, metadata):
    # Query represente quelques exemples de requêtes pour tester l'index
    # query = "concert jazz Paris"
    # query = "exposition art contemporain"
    # query = "festival gratuit famille"
    query = "conférence intelligence artificielle"
    results = search(query, index, metadata)
    
    # Vérifie qu'on a bien TOP_K résultats
    assert len(results) == TOP_K, f"Expected {TOP_K} results, got {len(results)}"
    
    # Vérifie que chaque résultat contient bien les champs attendus
    for result in results:
        assert "metadata" in result
        assert "title_fr" in result["metadata"]
        assert "location_city" in result["metadata"]
        assert "firstdate_begin" in result["metadata"]
        assert "distance" in result
        # distance doit être un float positif
        assert isinstance(result["distance"], float)
        assert result["distance"] >= 0

def test_search_index(index, metadata):
    # Query represente quelques exemples de requêtes pour tester l'index
    # query = "concert jazz Paris"
    # query = "exposition art contemporain"
    # query = "festival gratuit famille"
    query = "conférence intelligence artificielle"
    results = search(query, index, metadata)
    
    # Affichage comme dans ton script original
    print(f"\nRequête test : {query}\n")
    for i, result in enumerate(results, 1):
        title = result["metadata"].get("title_fr", "")
        city = result["metadata"].get("location_city", "")
        date = result["metadata"].get("firstdate_begin", "")
        distance = result["distance"]

        print(f"Résultat {i}")
        print(f"Titre : {title}")
        print(f"Ville : {city}")
        print(f"Date : {date}")
        print(f"Distance : {distance}")
        print("-" * 50)

        # Assertions : vérifie juste que les clés existent et distance est un float positif
        assert "title_fr" in result["metadata"]
        assert "location_city" in result["metadata"]
        assert "firstdate_begin" in result["metadata"]
        assert isinstance(distance, float)
        assert distance >= 0

    # Vérifie qu'on a bien TOP_K résultats
    assert len(results) == TOP_K, f"Expected {TOP_K} results, got {len(results)}"