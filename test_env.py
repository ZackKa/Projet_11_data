# test_env.py
try:
    import langchain
    import faiss
    import mistralai
    import pandas
    import requests
    import dotenv
    import tiktoken
    import pytest

    print("Toutes les librairies sont importées avec succès !")
    print("Versions :")
    print("LangChain :", langchain.__version__)
    print("Faiss :", faiss.__version__)
    print("pandas :", pandas.__version__)
    print("requests :", requests.__version__)
    print("tiktoken :", tiktoken.__version__)
    print("mistralai importé avec succès")
except Exception as e:
    print("Erreur lors de l'importation :", e)
