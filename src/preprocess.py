import json
import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# -------------------------------
# Définition des chemins de fichiers
# -------------------------------

RAW_PATH = Path("../data/raw_events.json")  # Chemin vers le fichier JSON brut (non nettoyé)
CLEAN_PATH = Path("../data/clean_events.json")  # Chemin vers le fichier JSON nettoyé (prêt pour la vectorisation)

# -------------------------------
# Champs texte éditoriaux à nettoyer
# -------------------------------
# Ce sont les champs contenant du texte susceptible de contenir du HTML ou des retours à la ligne qu'on souhaite nettoyer.

TEXT_FIELDS_TO_CLEAN = [
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "conditions_fr",
    "keywords_fr",
    "imagecredits",
    "location_imagecredits",
    "location_description_fr",
    "location_access_fr",
    "originagenda_title",
    "country_fr",
    "location_tags"
]

# -------------------------------
# Fonction pour nettoyer le HTML et les espaces
# -------------------------------
def clean_html(value):
    """
    Nettoie une chaîne de caractères en :
    - supprimant les balises HTML si présentes
    - remplaçant les retours à la ligne par des espaces
    - supprimant les espaces multiples
    """
    # Si ce n'est pas une chaîne de caractères, on renvoie tel quel
    if not isinstance(value, str):
        return value

    # Si la chaîne ne contient pas de balises HTML
    if "<" not in value and ">" not in value:
        value = re.sub(r"[\r\n]+", " ", value)  # On remplace les retours à la ligne par un espace
        return re.sub(r"\s+", " ", value).strip()   # On réduit les espaces multiples à un seul espace

    # Si contient du HTML → on parse (analyse) le HTML
    soup = BeautifulSoup(value, "html.parser")
    text = soup.get_text(separator=" ")  # On extrait le texte du HTML

    # Supprimer les retours ligne
    text = re.sub(r"[\r\n]+", " ", text)  # On remplace les retours à la ligne par un espace

    # Nettoyage espaces
    text = re.sub(r"\s+", " ", text).strip()  # On réduit les espaces multiples à un seul espace

    return text  # On renvoie le texte nettoyé


# -------------------------------
# Fonction pour nettoyer tous les champs
# -------------------------------
def clean_all_fields(data):
    """
    Nettoie récursivement toutes les chaînes de caractères d'un dictionnaire ou d'une liste.
    - Si c'est un dict, on nettoie chaque valeur
    - Si c'est une liste, on nettoie chaque élément
    - Si c'est une chaîne, on applique clean_html
    - Sinon, on renvoie tel quel
    """
    if isinstance(data, dict):
        return {k: clean_all_fields(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_all_fields(item) for item in data]
    elif isinstance(data, str):
        return clean_html(data)
    else:
        return data


# -------------------------------
# Fonction pour valider et formater une date
# -------------------------------
def validate_date(date_str):
    """
    Vérifie si une date est valide et la convertit en format ISO standard.
    - date_str : chaîne de caractères représentant une date
    - Renvoie : date formatée "YYYY-MM-DDTHH:MM:SS+00:00" ou None si invalide
    """
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")) # fromisoformat ne supporte pas "Z" donc on remplace par "+00:00" pour UTC
        return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00") # on formate la date en "YYYY-MM-DDTHH:MM:SS+00:00"
    except Exception:
        return None


# -------------------------------
# Fonction pour charger les données brutes
# -------------------------------
def load_raw_data():
    """
    Charge les données brutes depuis le fichier JSON.
    """
    with open(RAW_PATH, "r", encoding="utf-8") as f:  # On ouvre le fichier JSON en lecture
        return json.load(f)


# -------------------------------
# Fonction principale pour nettoyer et prétraiter les événements
# -------------------------------
def preprocess_events(events):
    """
    Prétraite une liste d'événements en :
    - nettoyant les champs texte
    - validant les dates
    - supprimant les doublons
    - construisant un champ "text_for_embedding" prêt pour l'IA
    """

    clean_events = []  # On initialise une liste pour stocker les événements nettoyés
    seen_hashes = set()  # On initialise un ensemble pour stocker les hashs des événements déjà traités, pour éviter les doublons

    for event in events:

        # NETTOYAGE COMPLET DE TOUS LES CHAMPS
        #event = clean_all_fields(event)

        # Nettoyage uniquement des champs texte éditoriaux
        for field in TEXT_FIELDS_TO_CLEAN:
            if field in event and event[field]:
                event[field] = clean_html(event[field])

        # Vérification des champs obligatoires
        uid = event.get("uid")
        title = event.get("title_fr", "")
        short_desc = event.get("description_fr", "")
        long_desc = event.get("longdescription_fr", "")

        # On ignore l'événement si UID ou titre ou description manquent
        if not uid or not title or (not short_desc and not long_desc):
            continue

        # Validation et formatage des dates
        firstdate_begin = validate_date(event.get("firstdate_begin"))
        firstdate_end = validate_date(event.get("firstdate_end"))
        lastdate_begin = validate_date(event.get("lastdate_begin"))
        lastdate_end = validate_date(event.get("lastdate_end"))

        # On ignore l'événement si date de début manquante
        if not firstdate_begin:
            continue

        # Création d'un hash pour détecter les doublons exacts
        event_hash = (
            uid, title, short_desc, long_desc,
            firstdate_begin, firstdate_end,
            lastdate_begin, lastdate_end,
            event.get("location_name"),
            event.get("location_address"),
            event.get("location_city")
        )

        # On ignore l'événement si le hash existe déjà
        if event_hash in seen_hashes:
            continue
        seen_hashes.add(event_hash)
        
        # Construction du texte pour embeddings

        text_for_embedding = f"""Titre : {title}
Description : {short_desc}
Détails : {long_desc}
Lieu : {event.get("location_name", "")}
Adresse : {event.get("location_address", "")}
Date début : {firstdate_begin}
Date fin : {firstdate_end}"""

        
        # Remplacer les retours ligne par un point
        text_for_embedding = re.sub(r"[\r\n]+", ". ", text_for_embedding)
        # Nettoyer espaces multiples
        text_for_embedding = re.sub(r"\s+", " ", text_for_embedding).strip()
        # Éviter les doubles points accidentels
        text_for_embedding = re.sub(r"\.\s*\.", ".", text_for_embedding)


        # # Remplacer None par ""
        # clean_event = {k: (v if v is not None else "") for k, v in event.items()}

        # clean_event["text_for_embedding"] = text_for_embedding
        # clean_event["firstdate_begin"] = firstdate_begin
        # clean_event["firstdate_end"] = firstdate_end
        # clean_event["lastdate_begin"] = lastdate_begin
        # clean_event["lastdate_end"] = lastdate_end

        # Création du dictionnaire final pour l'événement
        clean_event = {
            "uid": uid,
            "canonicalurl": event.get("canonicalurl", ""),
            "title_fr": title,
            "description_fr": short_desc,
            "longdescription_fr": long_desc,
            "conditions_fr": event.get("conditions_fr", ""),
            "updatedat": event.get("updatedat", ""),
            "daterange_fr": event.get("daterange_fr", ""),
            "firstdate_begin": firstdate_begin,
            "firstdate_end": firstdate_end,
            "lastdate_begin": lastdate_begin,
            "lastdate_end": lastdate_end,
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
            "age_max": event.get("age_max", ""),
            "text_for_embedding": text_for_embedding
        }

        # Remplacer tous les None par ""
        clean_event = {k: (v if v is not None else "") for k, v in clean_event.items()}

        # On ajoute l'événement nettoyé à la liste des événements nettoyés
        clean_events.append(clean_event)

    return clean_events

# -------------------------------
# Fonction pour sauvegarder les données nettoyées
# -------------------------------
def save_clean_data(data):
    """
    Sauvegarde les données nettoyées dans un fichier JSON.
    """
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)  # On crée le dossier parent si il n'existe pas
    with open(CLEAN_PATH, "w", encoding="utf-8") as f:  # On ouvre le fichier JSON en écriture
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    print("Nettoyage des données...")  # Message informatif
    raw_data = load_raw_data()  # Appel de la fonction pour charger les données brutes
    clean_data = preprocess_events(raw_data)  # Appel de la fonction pour nettoyer les événements
    save_clean_data(clean_data)  # Sauvegarde les données nettoyées dans le fichier JSON
    print(f"{len(clean_data)} événements nettoyés sauvegardés.")  # Message de confirmation
