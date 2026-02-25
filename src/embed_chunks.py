# Script de génération des embeddings pour la vectorisation

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
import time

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data/chunks_events.json"
EMBEDDINGS_PATH = BASE_DIR / "data/embeddings_events.json"

EMBEDDING_MODEL = "mistral-embed" # nom du modèle utilisé pour générer les embeddings
BATCH_SIZE = 50  # nombre de chunks envoyés par requête à l'API Mistral

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()  # charge les variables d'environnement depuis le fichier .env
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # récupère la clé API Mistral depuis le fichier .env

if not MISTRAL_API_KEY:
    raise ValueError("Clé API Mistral manquante")  # si la clé API Mistral est manquante, on raise une erreur

client = Mistral(api_key=MISTRAL_API_KEY)  # crée un client Mistral avec la clé API Mistral, pour interagir avec l’API Mistral

# -----------------------------
# LOAD CHUNKS
# -----------------------------
def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_chunks(chunks, batch_size=BATCH_SIZE):
    # chunks = liste de chunks à traiter
    # batch_size = nombre de chunks à envoyer par requête
    print(f"Génération des embeddings par batch de {batch_size} chunks...")

    total_chunks = len(chunks)  # total_chunks = nombre total de chunks à traiter


    # On découpe la liste complète en plusieurs batches de batch_size
    # on boucle de 0 à total_chunks par batch_size
    for i in range(0, total_chunks, batch_size):
        # On récupère les chunks à traiter en parallèle
        batch = chunks[i:i + batch_size]  # prend les chunks de l’index i jusqu’à i + batch_size
        # batch est donc une sous-liste de chunks, c'est ce qu’on va envoyer à l’API Mistral en une seule requête
        texts = [chunk["text"] for chunk in batch]  # texts = liste de textes à traiter en parallèle, donc extraction de tous les textes à envoyer à l’API.
        # [chunk["text"] for chunk in batch] : pour chaque chunk dans la liste batch, on récupère le texte du chunk

        print(f"Traitement des chunks {i} à {i + len(batch) - 1}...")  # print(f"Traitement des chunks {i} à {i + len(batch) - 1}...")

        success = False  # success = booléen pour vérifier si l'embedding a été généré avec succès
        attempts = 0  # attempts = nombre de tentatives pour générer l'embedding

        # On répète la tentative de génération de l'embedding jusqu'à 3 fois
        while not success and attempts < 3:
            try:
                response = client.embeddings.create(  # client.embeddings.create() envoie les textes à Mistral pour obtenir leurs embeddings.
                    model=EMBEDDING_MODEL,  # model = nom du modèle utilisé pour générer les embeddings
                    inputs=texts  # inputs = liste de textes à traiter en parallèle
                )
                # response.data contient les embeddings générés par Mistral

                embeddings = [item.embedding for item in response.data]  # on recupere le vecteur d'embedding de chaque chunk grâce à item.embedding
                # embeddings = liste de vecteurs d'embedding

                # ATTRIBUTION DES EMBEDDINGS A CHAQUE CHUNK
                # enumerate() renvoie un tuple contenant l'index et la valeur de chaque élément de la liste batch
                # j = index du chunk dans le batch
                # chunk = chunk à traiter
                for j, chunk in enumerate(batch):
                    chunk["embedding"] = embeddings[j]  # on ajoute le vecteur d'embedding à la liste chunks

                success = True  # on passe le booléen success à True pour indiquer que l'embedding a été généré avec succès

            except Exception as e:
                attempts += 1  # on incrémente le nombre de tentatives
                print(f"Erreur (tentative {attempts}/3): {e}")
                print("Attente de 5 secondes avant nouvelle tentative...")  # on attend 5 secondes avant de réessayer
                time.sleep(5)

        if not success:
            print(f"Batch {i}-{i+len(batch)-1} échoué après 3 tentatives.")  # on affiche le batch qui a échoué
            raise Exception("Arrêt du script pour éviter perte de données.")  # on arrête le script pour éviter perte de données    

        # petite pause pour éviter rate limit
        time.sleep(1)  # on attend 1 seconde avant de passer au batch suivant

    return chunks  # on retourne la liste de chunks avec les vecteurs d'embedding

# -----------------------------
# SAVE
# -----------------------------
# save_embeddings() sauvegarde les vecteurs d'embedding dans embeddings_events.json
def save_embeddings(data):
    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:  # on ouvre le fichier embeddings_events.json en mode écriture
        json.dump(data, f, ensure_ascii=False)  # json.dump() écrit les vecteurs d'embedding dans le fichier. ensure_ascii=False : garde les caractères spéciaux (é, à, ü) lisibles.

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Chargement des chunks...")
    chunks = load_chunks()  # load_chunks() charge les chunks dans la variable chunks
    print(f"{len(chunks)} chunks chargés")

    enriched_chunks = embed_chunks(chunks)  # embed_chunks() génère les vecteurs d'embedding pour les chunks

    save_embeddings(enriched_chunks)  # save_embeddings() sauvegarde les vecteurs d'embedding dans embeddings_events.json

    print("Embeddings sauvegardés")
    print(f"Shape embeddings : {len(enriched_chunks)}")