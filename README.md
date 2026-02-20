# POC RAG – Puls-Events
## Étape 1 : Configuration de l’environnement de développement

Ce projet s’inscrit dans la réalisation d’un Proof of Concept (POC) d’un système RAG (Retrieval-Augmented Generation) pour l’entreprise Puls-Events.

Le système utilisera :

- LangChain pour l’orchestration

- Mistral API via mistralai

- FAISS (CPU) pour la base vectorielle

- OpenAgenda API pour la récupération des événements

## Objectif

Mettre en place un environnement Python reproductible contenant toutes les dépendances nécessaires au développement du système RAG.

### Prérequis

- Python 

- Conda (Anaconda)


## 🧪 Création de l’environnement virtuel
### 1️ Création
```bash
conda create -n puls_env python=3.10
```
### 2️ Activation
```bash
conda activate puls_env
```
## Installation des dépendances

Les dépendances sont centralisées dans le fichier requirements.txt.

Installation :
```bash
pip install --upgrade pip
```
```bash
pip install -r requirements.txt
```


## Vérification de l’installation

Un fichier test_env.py est fourni pour vérifier que toutes les dépendances sont correctement installées.

Exécution du test :
```bash
python test_env.py
```

Si tout est correctement installé, le terminal affichera :

- Confirmation que toutes les librairies sont importées

- Les versions principales des packages

Exemple de sortie attendue :
```bash
Toutes les librairies sont importées avec succès !
Versions :
LangChain : X.X.X
Faiss : X.X.X
pandas : X.X.X
...
```

Si une erreur apparaît, vérifier que :

- L’environnement puls_env est bien activé

- Les dépendances ont bien été installées avec pip install -r requirements.txt

## Import des données OpenAgenda

Pour récupérer les événements publics récents (moins d’un an) et les sauvegarder localement, utilisez le script fourni (fetch_events.py).

### Fonctionnalités du script

- Récupère les événements via l’API OpenAgenda

- Filtre par ville (par défaut : Paris) et par date (moins d’un an)

- Sauvegarde les événements dans data/raw_events.json

Exécution :
```bash
python fetch_events.py
```

Après exécution, vous devriez voir un message :
```bash
Récupération des événements...
Récupérés jusqu'à 100 événements...
...
Total récupéré : XXX événements
Données sauvegardées dans data/raw_events.json
```

Les données sont alors prêtes pour les étapes suivantes (prétraitement et vectorisation).

## Prétraitement et nettoyage des données

Un script `preprocess_events.py` a été développé pour nettoyer et préparer les événements récupérés afin de les rendre exploitables pour la vectorisation et le système RAG.

### Fonctionnalités principales

1. Nettoyage des champs texte

- Suppression du HTML, des retours à la ligne et des espaces multiples dans les champs éditoriaux tels que :
`title_fr`, `description_fr`, `longdescription_fr`, `conditions_fr`, `keywords_fr`, `location_description_fr`, etc.

2. Validation des dates

- Conversion des dates en format ISO standard (`YYYY-MM-DDTHH:MM:SS+00:00`).

- Suppression des événements sans date valide.

3. Suppression des doublons

- Détection basée sur un hash combinant UID, titre, description, dates et lieu.

4. Construction d’un champ `text_for_embedding`

- Consolidation du titre, description, détails, lieu et dates dans un champ unique prêt pour la vectorisation NLP.

- Nettoyage supplémentaire des retours à la ligne et des espaces multiples.

5. Gestion des valeurs manquantes

- Remplacement automatique de toutes les valeurs `None` par des chaînes vides pour éviter des erreurs lors de la vectorisation.

### Résultat

- Les événements nettoyés sont sauvegardés dans :
```bash
data/clean_events.json
```
- Le fichier contient uniquement des événements récents, complets et sans doublons, avec un texte consolidé prêt pour la création de la base vectorielle.

### Exemple d’exécution
```bash
python preprocess_events.py
```

Message attendu :
``` bash
Nettoyage des données...
XXX événements nettoyés sauvegardés.
```

### Tests unitaires avec Pytest

Pour garantir que le prétraitement a produit un dataset correct et exploitable, un script de tests test_clean_events.py a été développé avec pytest.

#### Objectifs des tests

- Vérifier que le dataset n’est pas vide.

- Vérifier que tous les événements sont situés à Paris.

- Vérifier que toutes les dates des événements sont récentes (supérieures à la date minimale définie).

#### Exécution des tests
```bash
pytest test_clean_events.py
```
Le succès de tous les tests confirme que le dataset nettoyé est fiable et prêt pour les étapes suivantes (vectorisation et RAG).