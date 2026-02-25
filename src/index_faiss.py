# Script de construction de l'index FAISS

import json
import numpy as np
import faiss
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = BASE_DIR / "data/embeddings_events.json"

FAISS_INDEX_PATH = BASE_DIR / "data/faiss_index.index"
METADATA_PATH = BASE_DIR / "data/faiss_metadata.json"


# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
def load_embeddings():
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# BUILD INDEX
# -----------------------------
def build_faiss_index(data):
    print("Préparation des vecteurs...")

    embeddings = []
    metadata = []

    for item in data:

        if "embedding" not in item:
            continue

        embeddings.append(item["embedding"])

        # on retire le vecteur pour ne garder que les infos utiles
        metadata.append({
            "chunk_id": item["chunk_id"],
            "uid": item["uid"],
            "text": item["text"],
            "metadata": item["metadata"]
        })

    # conversion en numpy float32
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    print(f"Dimension des vecteurs : {dimension}")
    print(f"Nombre de vecteurs : {embeddings.shape[0]}")

    print("Création de l'index FAISS (IndexFlatL2)...")
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("Index construit avec succès.")

    return index, metadata


# -----------------------------
# SAVE
# -----------------------------
def save_index(index):
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Index sauvegardé : {FAISS_INDEX_PATH}")


def save_metadata(metadata):
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Métadonnées sauvegardées : {METADATA_PATH}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Chargement des embeddings...")
    data = load_embeddings()
    print(f"{len(data)} chunks chargés")

    index, metadata = build_faiss_index(data)

    save_index(index)
    save_metadata(metadata)

    print("Index FAISS prêt.")
