# Execution de tous le processus jusqu'à l'index

# subprocess sert à exécuter des commandes externes depuis ce script Python
import subprocess

subprocess.run(["python", "fetch_events.py"])   # Lance la récupération des données
subprocess.run(["python", "preprocess.py"])   # Lance le nettoyage et la préparation des données
subprocess.run(["python", "chunking.py"])   # Lance le découpage des données en petits morceaux (chunks)
subprocess.run(["python", "embed_chunks.py"])   # Lance la génération des embeddings (vecteurs numériques représentatifs du contenu des chunks)
subprocess.run(["python", "index_faiss.py"])   # Lance la création de l'index (FAISS)

print("Base vectorielle reconstruite avec succès.")