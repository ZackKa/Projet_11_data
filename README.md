# POC RAG – Puls-Events

## Présentation du projet

Ce projet constitue un Proof of Concept (POC) pour l’entreprise Puls-Events.
L’objectif est de démontrer la faisabilité d’un système de recommandation d’événements culturels basé sur une architecture RAG (Retrieval-Augmented Generation).

Le périmètre géographique retenu est : Paris.
Les événements intégrés sont récents (moins d’un an) ou à venir.

## Étape 1 : Configuration de l’environnement de développement

Le POC permet :

- De récupérer les événements publics récents via l’API OpenAgenda

- De nettoyer et structurer ces données pour un usage NLP

- De vectoriser les descriptions avec un modèle Mistral

- D’indexer les vecteurs dans FAISS pour une recherche rapide par similarité

- De générer des recommandations personnalisées via un chatbot orchestré avec LangChain

Le système utilisera :

- LangChain pour l’orchestration

- Mistral API via mistralai

- FAISS pour la base vectorielle

- OpenAgenda API pour la récupération des événements

## Objectif

Mettre en place un environnement Python reproductible contenant toutes les dépendances nécessaires au développement du système RAG.

## Architecture du système

Pipeline :
```
OpenAgenda API
→ Nettoyage & normalisation
→ Chunking
→ Embeddings
→ Indexation FAISS
→ Retriever LangChain
→ LLM Mistral
→ Réponse utilisateur
```
### Structure
```
PULS-EVENTS/
├─ data/
│  ├─ raw_events.json
│  ├─ clean_events.json
│  ├─ chunks_events.json
│  ├─ embeddings_events.json
│  ├─ faiss_index.index
│  └─ faiss_metadata.json
├─ src/
│  ├─ fetch_events.py
│  ├─ preprocess.py
│  ├─ chunking.py
│  ├─ embed_chunks.py
|  ├─ index_faiss.py
|  ├─ build_vector_db.py (script qui lance toutes les étapes précédentes en une commande)
│  └─ rag.py
├─ test/
│  ├─ test_env.py
│  ├─ test_filtre_localisation.py
│  └─ test_search_index.py
├─ .env (à créer)
├─ requirements.txt
└─ README.md
```
### Prérequis

- Python 3.10

- Conda (Anaconda)


## Création de l’environnement virtuel
### 1️ Création
```bash
conda create -n puls_env python=3.10
```
### 2️ Activation
```bash
conda activate puls_env
```
## Installation des dépendances

Les dépendances sont centralisées dans le fichier `requirements.txt`.

Installation :
```bash
pip install --upgrade pip
```
```bash
pip install -r requirements.txt
```
Attention si requirements provoque une erreur, il faut changer la ligne qui ressemble à `packaging @ file:///C:/...` dans le fichier `requirements.txt`, par juste cette ligne `packaging`.


## Vérification de l’installation

Un fichier `test_env.py` (dans le dossier test) est fourni pour vérifier que toutes les dépendances sont correctement installées.

Exécution du test sur le chemin du dossier test:
```bash
pytest -v -s test_env.py
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

- L’environnement `puls_env` est bien activé

- Les dépendances ont bien été installées avec `pip install -r requirements.txt`

## Étape 2 - Import des données OpenAgenda

Pour récupérer les événements publics récents (moins d’un an) et les sauvegarder localement, utilisez le script fourni (`fetch_events.py`).

### Fonctionnalités du script

- Récupère les événements via l’API OpenAgenda

- Filtre par ville (par défaut : Paris) et par date (moins d’un an)

- Sauvegarde les événements dans `data/raw_events.json`


Exécution :
Créer un dossier `data` a la racine du projet puis lancer la commande suivante
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

## Étape 3 - Prétraitement et nettoyage des données

Un script `preprocess.py` a été développé pour nettoyer et préparer les événements récupérés afin de les rendre exploitables pour la vectorisation et le système RAG.

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
python preprocess.py
```

Message attendu :
``` bash
Nettoyage des données...
XXX événements nettoyés sauvegardés.
```

### Tests unitaires avec Pytest

Pour garantir que le prétraitement a produit un dataset correct et exploitable, un script de tests `test_filtre_localisation.py` a été développé avec pytest.

#### Objectifs des tests

- Vérifier que le dataset n’est pas vide.

- Vérifier que tous les événements sont situés à Paris.

- Vérifier que toutes les dates des événements sont récentes (supérieures à la date minimale définie).

#### Exécution des tests
```bash
pytest -v -s test_filtre_localisation.py
```
Le succès de tous les tests confirme que le dataset nettoyé est fiable et prêt pour les étapes suivantes (vectorisation et RAG).

## Étape 4 : Découpage en chunks (Chunking)

Après avoir nettoyé les événements, il est nécessaire de découper les textes consolidés (`text_for_embedding`) en morceaux plus petits, appelés chunks, pour faciliter la génération d’embeddings et la recherche vectorielle.

### Objectifs

- Gérer les limites de taille des modèles NLP.

- Optimiser la recherche sémantique dans la base FAISS.

- Conserver le contexte via un chevauchement entre les chunks.

### Fichier et script

Script : `chunking.py`

Fichier produit : `data/chunks_events.json`

### Fonctionnement

1. Charge les événements nettoyés depuis `clean_events.json`.

2. Utilise `RecursiveCharacterTextSplitter` de LangChain avec :

 - `chunk_size = 800` caractères

 - `chunk_overlap = 150` caractères

3. Pour chaque événement, génère des chunks et conserve les métadonnées associées (titre, description, dates, lieu…).

4. Sauvegarde tous les chunks dans `chunks_events.json`.

Exemple d’exécution
```bash
python chunking.py
```
Message attendu :
```bash
Chargement des événements nettoyés...
XXX événements chargés.
Découpage en chunks...
YYY chunks générés et sauvegardés.
```

## Étape 5 : Génération des embeddings

Chaque chunk doit être transformé en vecteur numérique représentatif de son contenu, appelé embedding, afin de permettre une recherche par similarité dans la base FAISS.

### Objectifs

- Transformer les textes en vecteurs compréhensibles par l’IA.

- Préparer les données pour l’index vectoriel.

### Fichier et script

Script : `embed_chunks.py`

Fichier produit : `data/embeddings_events.json`

### Note

Créer une clé API sur mistral et la mettre dans un fichier `.env` avec `MISTRAL_API_KEY=Votre_clé_api`

### Fonctionnement

1. Charge les chunks depuis `chunks_events.json`.

2. Utilise le modèle Mistral via son API pour générer les embeddings.

3. Traite les chunks par batches (ici 50 par batch) pour optimiser les requêtes API.

4. Réessaie jusqu’à 3 fois en cas d’erreur pour éviter la perte de données.

5. Ajoute chaque embedding au chunk correspondant.

6. Sauvegarde le résultat final dans `embeddings_events.json`.

Exemple d’exécution
```bash
python embed_chunks.py
```
Message attendu :
```bash
Chargement des chunks...
XXX chunks chargés
Génération des embeddings par batch de 50 chunks...
Embeddings sauvegardés
Shape embeddings : XXX
```

## Étape 6 : Construction de l’index FAISS

Après avoir généré les embeddings des chunks, il est nécessaire de créer un index vectoriel FAISS pour permettre une recherche rapide par similarité.

### Objectifs

- Indexer tous les vecteurs d’embedding pour un accès efficace.

- Conserver les métadonnées associées à chaque chunk (titre, description, date, lieu…).

- Préparer la base pour des recherches sémantiques et des systèmes RAG.

### Fichiers et scripts

Script : `index_faiss.py`

Fichiers produits :

- `data/faiss_index.index` → index FAISS des vecteurs

- `data/faiss_metadata.json` → métadonnées associées aux chunks

### Fonctionnement

1. Charge les embeddings depuis embeddings_events.json.

2. Crée un index FAISS `IndexFlatL2` (distance euclidienne) pour tous les vecteurs.

3. Conserve uniquement les métadonnées pertinentes pour chaque chunk.

4. Sauvegarde l’index FAISS et le fichier de métadonnées.

Exemple d’exécution
```bash
python index_faiss.py
```
Message attendu :
```bash
Chargement des embeddings...
XXX chunks chargés
Préparation des vecteurs...
Dimension des vecteurs : 1536
Nombre de vecteurs : XXX
Création de l'index FAISS (IndexFlatL2)...
Index construit avec succès.
Index sauvegardé : data/faiss_index.index
Métadonnées sauvegardées : data/faiss_metadata.json
Index FAISS prêt.
```

## Étape 7 : Test de recherche dans l’index

Une fois l’index créé, il est recommandé de tester la recherche par requête pour vérifier que tout fonctionne correctement.

### Objectifs

- Vérifier l’intégrité de l’index et des embeddings.

- Tester des requêtes exemples pour retrouver les événements pertinents.

- Obtenir les résultats avec leur distance et métadonnées.

### Fichier et script

Script : `test_search_index.py` dans le dossier `test`

### Fonctionnement

1. Charge l’index FAISS (`faiss_index.index`) et les métadonnées (`faiss_metadata.json`).

2. Génère l’embedding d’une requête utilisateur via le modèle Mistral.

3. Cherche les `TOP_K` chunks les plus proches dans l’index.

4. Retourne les résultats avec :

- titre de l’événement

- ville

- date

- distance vectorielle

### Requêtes de test

Dans le script, plusieurs requêtes sont déjà présentes et commentées :
```bash 
# query = "concert jazz Paris"
# query = "exposition art contemporain"
# query = "festival gratuit famille"
query = "conférence intelligence artificielle"
```
Vous pouvez décommenter une requête à la fois pour tester différents types d’événements.

Vous pouvez également remplacer query par votre propre chaîne de recherche pour expérimenter avec vos propres requêtes.

Exemple d’exécution
```bash
pytest -v -s test_search_index.py
```
Message attendu :
```bash
Chargement de l'index FAISS...
Chargement des métadonnées...
Requête test : conférence intelligence artificielle

Résultat 1
Titre : Conférence IA & Innovation
Ville : Paris
Date : 2026-03-15T09:00:00+00:00
Distance : 0.0123
--------------------------------------------------
Résultat 2
Titre : Atelier Machine Learning
Ville : Paris
Date : 2026-04-01T14:00:00+00:00
Distance : 0.0251
...
```

## Étape 8 : Automatisation du pipeline avec `build_vector_db.py`

Pour simplifier et centraliser le processus de préparation des données et de construction de la base vectorielle, toutes les étapes précédentes sont automatisées dans un script unique.
Avant d’exécuter ce script, assurez-vous que le dossier `data/` existe et que `.env` contient `MISTRAL_API_KEY` avec votre clé.
```bash
python build_vector_db.py
```
Ce script exécute automatiquement :

- La récupération des événements (`fetch_events.py`)

- Le prétraitement et le nettoyage des événements (`preprocess.py`)

- Le découpage des textes en chunks (`chunking.py`)

- La génération des embeddings pour chaque chunk (`embed_chunks.py`)

- La création de l’index FAISS et la sauvegarde des métadonnées associées (`index_faiss.py`)

À l’issue de l’exécution, une base vectorielle FAISS complète et prête à l’emploi est produite, incluant tous les chunks avec leurs embeddings et métadonnées. Cette base est directement exploitable pour la recherche sémantique et pour le système RAG.

Cette automatisation garantit un pipeline reproductible, cohérent et rapide, minimisant les erreurs et facilitant la mise à jour des événements.


## Étape 9 : Intégration RAG et Chatbot

Cette étape consiste à exploiter la base vectorielle FAISS construite précédemment pour fournir un assistant capable de recommander des événements culturels via un système RAG (Retrieval-Augmented Generation).

Le script est `rag.py`. Il permet de :

- Charger l’index FAISS et les métadonnées associées (`faiss_index.index` et `faiss_metadata.json`).

- Reconstruire les documents à partir des chunks indexés.

- Créer un vectorstore LangChain pour l’interrogation par similarité.

- Configurer un LLM Mistral pour la génération de réponses basées sur les documents récupérés.

- Définir un prompt structuré pour répondre uniquement sur les événements culturels à Paris, en affichant : `Titre, Date, Lieu et Description.`

- Construire une chaîne RAG combinant le retriever et le LLM.

- Fournir une interface interactive où l’utilisateur peut poser des questions et obtenir des réponses contextualisées.

### Fonctionnement

- Le retriever recherche les chunks les plus pertinents pour chaque question.

- Le LLM génère une réponse structurée en utilisant uniquement les informations extraites des chunks récupérés.

- Les sources utilisées pour générer la réponse sont affichées pour plus de transparence.

Exemple d’exécution : 

!!! Assurez-vous que `build_vector_db.py` a été exécuté et que les fichiers `faiss_index.index` et `faiss_metadata.json` sont présents avant de lancer `rag.py`.
```bash
python rag.py
```
Exemple de sortie attendue :
```bash
Chatbot événements Paris prêt !
Tape 'exit' pour quitter.

Votre question : Quels concerts de jazz ont lieu à Paris ce mois-ci ?

Réponse :

Titre : Jazz Night au Sunset
Date : 2026-03-12T20:00:00+00:00
Lieu : Sunset Jazz Club, 3 Rue des Lombards, Paris
Description : Soirée jazz avec le groupe XYZ. Information non disponible pour certains détails.

Sources utilisées :
- Jazz Night au Sunset
- Festival Jazz à la Villette

--------------------------------------------------
```
Objectifs de cette étape

- Permettre des recherches sémantiques et des recommandations contextuelles.

- Garantir que le LLM ne génère des réponses que sur les événements culturels à Paris, en utilisant les données indexées.

- Fournir une interface interactive simple pour tester et exploiter la base vectorielle.

## Lancer le projet complet (Rappel)

Rappel : Pour faire simple, au lieu de lancer chaque script du dossier Src l'un après l'autre, une solution plus simple est disponible.
Après création d'un dossier data et de l'ajout de la clé API Mistral dans un fichier `.env`, les commandes à exécuter sont :

1. Construire la base vectorielle :
```bash
   python build_vector_db.py
```
2. Lancer le chatbot :
```bash
   python rag.py
```

## Conclusion

Ce POC démontre la faisabilité d’un système de recommandation d’événements culturels basé sur une architecture RAG.

Grâce à la récupération, au nettoyage et à la vectorisation des données, à l’indexation FAISS et à l’intégration avec un LLM Mistral orchestré par LangChain, il est possible de fournir des recommandations personnalisées et contextuelles pour des événements à Paris. 

Ce projet établit une base solide pour étendre le système à d’autres villes, types d’événements ou modèles NLP (Natural Language Processing, ou traitement automatique du langage naturel, qui permet aux ordinateurs de comprendre, analyser et générer du langage humain), et constitue un exemple concret de l’utilisation de l’IA pour enrichir l’expérience utilisateur dans le domaine culturel.