# Tester la recherche dans l'index

import json
import numpy as np
import faiss
import os
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral


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
# LOAD INDEX & METADATA
# -----------------------------
def load_index():
    return faiss.read_index(str(FAISS_INDEX_PATH))


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# EMBED QUERY
# -----------------------------
def embed_query(query):
    response = client.embeddings.create(
        model=MODEL_NAME,
        inputs=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")


# -----------------------------
# SEARCH
# -----------------------------
def search(query, index, metadata, top_k=TOP_K):
    query_vector = embed_query(query)

    distances, indices = index.search(
        np.array([query_vector]), 
        top_k
    )

    results = []

    for idx, distance in zip(indices[0], distances[0]):
        result = metadata[idx]
        result["distance"] = float(distance)
        results.append(result)

    return results


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Chargement de l'index FAISS...")
    index = load_index()

    print("Chargement des métadonnées...")
    metadata = load_metadata()

    # Query represente quelques exemples de requêtes pour tester l'index
    # query = "concert jazz Paris"
    # query = "exposition art contemporain"
    # query = "festival gratuit famille"
    query = "conférence intelligence artificielle"
    print(f"\nRequête test : {query}\n")

    results = search(query, index, metadata)

    for i, result in enumerate(results, 1):
        print(f"Résultat {i}")
        print(f"Titre : {result['metadata'].get('title_fr')}")
        print(f"Ville : {result['metadata'].get('location_city')}")
        print(f"Date : {result['metadata'].get('firstdate_begin')}")
        print(f"Distance : {result['distance']}")
        print("-" * 50)
