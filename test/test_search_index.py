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
MODEL_NAME = "mistral-embed"    # Nom du modèle d’embedding utilisé chez Mistral
TOP_K = 5    # Nombre de résultats à retourner

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()    # Charge les variables d'environnement depuis le fichier .env
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("Clé API Mistral manquante")    # Si la clé API Mistral est manquante, on raise une erreur

client = Mistral(api_key=MISTRAL_API_KEY)    # Crée un client Mistral avec la clé API Mistral, pour interagir avec l’API Mistral

# -----------------------------
# FIXTURES
# -----------------------------
@pytest.fixture(scope="module")
def index():
    """
    Charge l'index FAISS une seule fois pour tous les tests du module.
    """
    return faiss.read_index(str(FAISS_INDEX_PATH))

@pytest.fixture(scope="module")
def metadata():
    """
    Charge le fichier metadata.json une seule fois pour tous les tests.
    """
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def embed_query(query: str) -> np.ndarray:
    """
    Envoie une requête texte au modèle Mistral pour obtenir son embedding vectoriel.
    Retourne un numpy array en float32 (format attendu par FAISS).
    """
    response = client.embeddings.create(
        model=MODEL_NAME,
        inputs=[query]
    )
    # On récupère le vecteur et on le convertit en float32
    return np.array(response.data[0].embedding).astype("float32")

def search(query: str, index, metadata, top_k=TOP_K):
    """
    Effectue une recherche vectorielle :
    1. Transforme la requête en embedding
    2. Interroge l’index FAISS
    3. Récupère les métadonnées correspondantes
    4. Ajoute la distance à chaque résultat
    """
    # Génère le vecteur de la requête
    query_vector = embed_query(query)
    # FAISS attend un tableau 2D → on encapsule dans un array
    distances, indices = index.search(np.array([query_vector]), top_k)
    # On initialise la liste des résultats
    results = []
    # On parcourt les indices retournés par FAISS
    for idx, distance in zip(indices[0], distances[0]):
        # Récupère les métadonnées correspondant à l’index
        result = metadata[idx]
        # Ajoute la distance (score de similarité) à chaque résultat
        result["distance"] = float(distance)
        # Ajoute le résultat à la liste des résultats
        results.append(result)
    return results


def test_search_index(index, metadata):
    """
    Test avec affichage console :
    - Permet de visualiser les résultats
    - Vérifie également les champs obligatoires
    """

    # Query represente quelques exemples de requêtes pour tester l'index
    # query = "concert jazz Paris"
    # query = "exposition art contemporain"
    # query = "festival gratuit famille"
    query = "conférence intelligence artificielle"
    # Lance la recherche
    results = search(query, index, metadata)

    # Affichage comme dans ton script original
    print(f"\nRequête test : {query}\n")

    # Vérifie qu'on a bien TOP_K résultats
    assert len(results) == TOP_K, f"Expected {TOP_K} results, got {len(results)}"
    
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
        assert "metadata" in result
        assert "title_fr" in result["metadata"]
        assert "location_city" in result["metadata"]
        assert "firstdate_begin" in result["metadata"]
        assert isinstance(distance, float)
        assert distance >= 0

    # # Vérifie qu'on a bien TOP_K résultats
    # assert len(results) == TOP_K, f"Expected {TOP_K} results, got {len(results)}"