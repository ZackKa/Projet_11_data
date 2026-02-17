import requests  # Pour faire des requêtes HTTP vers l'API
import json      # Pour manipuler et sauvegarder les données au format JSON
from datetime import datetime, timedelta  # Pour manipuler les dates si nécessaire
from pathlib import Path  # Pour gérer les chemins de fichiers de manière portable

# URL de l'API OpenDataSoft pour les événements publics OpenAgenda
BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

# Chemin local où les données brutes vont être sauvegardées
DATA_PATH = Path("../data/raw_events.json")


def fetch_all_events(city="Paris"):
    """
    Récupère tous les événements récents (moins d'un an) pour une ville donnée
    en utilisant la pagination pour contourner la limite par requête.
    Paramètre :
        city : nom de la ville pour filtrer les événements (par défaut "Paris")
    
    Retour :
        all_results : liste de tous les événements récupérés
    """
    # Date de fin = aujourd'hui
    end_date = datetime.now().date()
    # Date de début = il y a 1 an
    start_date = end_date - timedelta(days=365)

    all_results = []    # Liste qui contiendra tous les résultats cumulés
    limit = 100   # Limite de résultats par requête 
    offset = 0     # Offset = position à partir de laquelle récupérer les événements

    # Boucle infinie jusqu'à ce qu'on ait récupéré tous les événements
    while True:
        # Paramètres de la requête API
        params = {
            "limit": limit,
            "offset": offset,
            "where": f"location_city='{city}' AND firstdate_begin >= date'{start_date}'"
        }

        response = requests.get(BASE_URL, params=params)    # Requête GET à l'API
        if response.status_code != 200:
            raise Exception(f"Erreur API : {response.status_code}") # En cas d'erreur, on stoppe et on affiche le code

        data = response.json()    # Conversion de la réponse JSON en dictionnaire Python
        results = data.get("results", [])    # Récupération de la liste des événements de cette page

        if not results:
            break    # Si cette page est vide, on a tout récupéré → on sort de la boucle

        all_results.extend(results)    # Ajout des résultats de cette page à la liste totale
        offset += limit    # On passe à la page suivante en incrémentant l'offset

        # Si on atteint ou dépasse le nombre total d'événements disponibles
        # 'total_count' est fourni par l'API et correspond au nombre total d'événements répondant au filtre
        if offset >= data.get("total_count", 0):
            break

        print(f"Récupérés jusqu'à {offset} événements...")    # Message pour suivre l'avancée

    print(f"Total récupéré : {len(all_results)} événements")    # Affichage du nombre total d'événements récupérés
    return all_results    # Retourne la liste de tous les événements récupérés

# Fonction pour sauvegarder les données récupérées dans un fichier local
def save_raw_data(data):
    """
    Sauvegarde les données JSON dans un fichier local
    Paramètre :
        data : à sauvegarder
    """

    # Crée le dossier parent si il n'existe pas (../data)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Ouvre le fichier en mode écriture ("w") avec encodage UTF-8
    # 'with' assure la fermeture automatique du fichier après l'écriture
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        # json.dump convertit l'objet Python en JSON et l'écrit dans le fichier
        # ensure_ascii=False : garde les caractères spéciaux (é, à, ü) lisibles
        # indent=4 : formate le JSON avec 4 espaces pour une lecture facile


if __name__ == "__main__":
    print("Récupération des événements...")  # Message informatif
    events = fetch_all_events(city="Paris")        # Appel de la fonction pour récupérer les événements
    save_raw_data(events)                   # Sauvegarde les données récupérées dans le fichier JSON
    print("Données sauvegardées dans data/raw_events.json")  # Confirmation de sauvegarde
