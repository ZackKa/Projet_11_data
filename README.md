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