# Script de découpage en chunks de texte pour la vectorisation

import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_PATH = BASE_DIR / "data/clean_events.json"
CHUNK_PATH = BASE_DIR / "data/chunks_events.json"

# Paramètres chunking
CHUNK_SIZE = 800    # nombre de caractères par chunk maximum
CHUNK_OVERLAP = 150    # nombre de caractères de recouvrement entre les chunks

# ouvre le fichier JSON clean_events.json et charge les données dans la variable events
def load_clean_data():
    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    # lit le fichier JSON avec json.load() et retourne la liste des événements, chaque événement est un dictionnaire Python

# On crée un splitter configuré avec la taille et le chevauchement définis
def chunk_events(events):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # On initialise la liste chunked_data qui contiendra tous les chunk
    chunked_data = []
    # On parcourt chaque événement dans la liste events
    for event in events:
        # On récupère le texte pour l'embedding et l'UID de l'événement
        text = event.get("text_for_embedding", "")
        uid = event.get("uid")

        # Si le texte ou l'UID est manquant, on passe à l'événement suivant
        if not text or not uid:
            continue

        # On découpe le texte en chunks en utilisant le splitter
        # split_text(text) découpe le texte d’un événement en plusieurs morceaux, appelés chunks
        chunks = splitter.split_text(text)

        # text_for_embedding est remplacé par text
        # On parcourt chaque chunk dans la liste chunks
        # i = l'index du chunk dans la liste chunks
        # chunk = le texte du chunk
        # enumerate() renvoie un tuple contenant l'index et la valeur de chaque élément de la liste chunks
        for i, chunk in enumerate(chunks):
            # On ajoute le chunk à la liste chunked_data
            chunked_data.append({
                "chunk_id": f"{uid}_{i}",
                "uid": uid,
                "text": chunk,
                "metadata": {
                    "uid": uid,
                    "canonicalurl": event.get("canonicalurl", ""),
                    "title_fr": event.get("title_fr", ""),
                    "description_fr": event.get("description_fr", ""),
                    "longdescription_fr": event.get("longdescription_fr", ""),
                    "conditions_fr": event.get("conditions_fr", ""),
                    "updatedat": event.get("updatedat", ""),
                    "daterange_fr": event.get("daterange_fr", ""),
                    "firstdate_begin": event.get("firstdate_begin", ""),
                    "firstdate_end": event.get("firstdate_end", ""),
                    "lastdate_begin": event.get("lastdate_begin", ""),
                    "lastdate_end": event.get("lastdate_end", ""),
                    "timings": event.get("timings", ""),
                    "accessibility": event.get("accessibility", ""),
                    "accessibility_fr": event.get("accessibility_fr", ""),
                    "location_name": event.get("location_name", ""),
                    "location_address": event.get("location_address", ""),
                    "location_district": event.get("location_district", ""),
                    "location_postalcode": event.get("location_postalcode", ""),
                    "location_city": event.get("location_city", ""),
                    "location_department": event.get("location_department", ""),
                    "location_region": event.get("location_region", ""),
                    "location_countrycode": event.get("location_countrycode", ""),
                    "location_phone": event.get("location_phone", ""),
                    "location_website": event.get("location_website", ""),
                    "location_access_fr": event.get("location_access_fr", ""),
                    "age_min": event.get("age_min", ""),
                    "age_max": event.get("age_max", "")
                    
                }
            })
            

    return chunked_data


def save_chunks(data):
    CHUNK_PATH.parent.mkdir(parents=True, exist_ok=True) # parents=True → crée aussi tous les dossiers parents manquants. exist_ok=True → ne renvoie pas d’erreur si le dossier existe déjà
    with open(CHUNK_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    # json.dump : écrit tous les chunks dans chunks_events.json. ensure_ascii=False : garde les caractères spéciaux (é, à, ü) lisibles. indent=4 : formate le JSON avec 4 espaces pour qu’il soit facile à lire


if __name__ == "__main__":
    print("Chargement des événements nettoyés...")
    events = load_clean_data() # load_clean_data() charge les données nettoyées dans la variable events
    print(f"{len(events)} événements chargés.")

    print("Découpage en chunks...")
    chunked = chunk_events(events) # chunk_events() découpe les événements en chunks et les stocke dans la variable chunked

    save_chunks(chunked) # save_chunks() sauvegarde les chunks dans chunks_events.json

    print(f"{len(chunked)} chunks générés et sauvegardés.")
