# Script de construction de l'index FAISS

import json              # Pour lire et écrire les fichiers JSON
import numpy as np       # Pour manipuler les vecteurs sous forme de tableaux numériques
import faiss             # Librairie spécialisée dans la recherche de similarité vectorielle
from pathlib import Path # Pour gérer proprement les chemins de fichiers

# -----------------------------
# CONFIG
# -----------------------------
# BASE_DIR correspond au dossier racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = BASE_DIR / "data/embeddings_events.json"

FAISS_INDEX_PATH = BASE_DIR / "data/faiss_index.index"  # Fichier de sortie pour l'index FAISS
METADATA_PATH = BASE_DIR / "data/faiss_metadata.json"   # Fichier de sortie pour les métadonnées associées aux vecteurs


# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
# Charge le fichier JSON contenant les embeddings.
def load_embeddings():
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# BUILD INDEX
# -----------------------------
def build_faiss_index(data):
    print("Préparation des vecteurs...")

    embeddings = []  # Liste qui contiendra uniquement les vecteurs numériques
    metadata = []    # Liste qui contiendra uniquement les informations descriptives

    # Parcours de chaque chunk
    for item in data:
        # Vérification qu’un embedding est bien présent sinon on passe au chunk suivant
        if "embedding" not in item:
            continue    # On ignore les entrées invalides

        embeddings.append(item["embedding"])  # On ajoute le vecteur d'embedding à la liste embeddings

        # On retire le vecteur pour ne garder que les infos utiles. Donc on stocke les métadonnées sans le vecteur
        metadata.append({
            "chunk_id": item["chunk_id"],
            "uid": item["uid"],
            "text": item["text"],
            "metadata": item["metadata"]
        })

    # Conversion des embeddings en tableau numpy. FAISS exige un format float32
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]  # On récupère la dimension des vecteurs (nombre de dimensions)
    print(f"Dimension des vecteurs : {dimension}")
    print(f"Nombre de vecteurs : {embeddings.shape[0]}")  # On récupère le nombre de vecteurs

    print("Création de l'index FAISS (IndexFlatL2)...")  # On crée l'index FAISS
    # Création d’un index FAISS utilisant la distance L2 (distance euclidienne)
    # IndexFlatL2 = recherche exacte (pas approximative)
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)  # Ajout de tous les vecteurs dans l’index 

    print("Index construit avec succès.")

    return index, metadata


# -----------------------------
# SAVE
# -----------------------------
def save_index(index):
    """
    Sauvegarde l’index FAISS sur disque. Cela permet de ne pas le reconstruire à chaque lancement.
    """
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Index sauvegardé : {FAISS_INDEX_PATH}")


def save_metadata(metadata):
    """
    Sauvegarde les métadonnées dans un fichier JSON.
    Ces données permettront d’associer les résultats de recherche à leurs textes originaux.
    """
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Métadonnées sauvegardées : {METADATA_PATH}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    '''
    FAISS sert à retrouver les passages les plus proches d’une question grâce à la similarité vectorielle.
    L’index contient uniquement les vecteurs.
    Le fichier metadata permet de récupérer le texte associé aux vecteurs trouvés.
    Ensuite, on envoie ces textes à Mistral pour générer une réponse pertinente.
    '''
    
    print("Chargement des embeddings...")
    # Chargement des embeddings depuis le fichier JSON
    data = load_embeddings()
    print(f"{len(data)} chunks chargés")
    # Construction de l’index FAISS + extraction des métadonnées
    index, metadata = build_faiss_index(data)
    # Sauvegarde de l’index et des métadonnées
    save_index(index)
    save_metadata(metadata)

    print("Index FAISS prêt.")
