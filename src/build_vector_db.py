# Execution de tous le processus jusqu'à l'index

import subprocess

subprocess.run(["python", "fetch.py"])
subprocess.run(["python", "preprocess.py"])
subprocess.run(["python", "chunk.py"])
subprocess.run(["python", "embedding.py"])
subprocess.run(["python", "index.py"])

print("Base vectorielle reconstruite avec succès.")