import os
import json
from pathlib import Path
import faiss
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = BASE_DIR / "data/faiss_index.index"
FAISS_METADATA_PATH = BASE_DIR / "data/faiss_metadata.json"

load_dotenv()   # Chargement des variables d’environnement (.env)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")   # Récupération de la clé API Mistral

if not MISTRAL_API_KEY:
    raise ValueError("Clé API Mistral manquante")   # Si la clé API Mistral est manquante, on raise une erreur

# -----------------------------
# EMBEDDINGS
# -----------------------------
# Modèle d’embedding utilisé pour transformer les questions en vecteurs
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=MISTRAL_API_KEY
)

# -----------------------------
# LOAD FAISS MANUEL
# -----------------------------
print("Chargement de l'index FAISS...")
# Lecture du fichier index sauvegardé précédemment
index = faiss.read_index(str(FAISS_INDEX_PATH))

print("Chargement des métadonnées...")
# Lecture du fichier JSON metadata contenant les textes associés aux vecteurs
with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# -----------------------------
# RECONSTRUCTION DES DOCUMENTS
# -----------------------------
# Création d'une liste de documents LangChain à partir des métadonnées car langchain ne peut pas utiliser directement les index FAISS
documents = [
    Document(page_content=item["text"], metadata=item["metadata"])
    for item in metadata
]

# -----------------------------
# BUILD VECTORSTORE LANGCHAIN
# -----------------------------

# Création d'un docstore compatible LangChain, il assure la correspondance entre les index FAISS et les documents (chunk) LangChain
# stocke les documents en mémoire
docstore = InMemoryDocstore(
    {str(i): doc for i, doc in enumerate(documents)}
)

# Le docstore contient les documents. FAISS retourne juste des positions. 
# Le mapping fait le lien entre la position FAISS et le document réel pour que le LLM sache quel texte utiliser.
# Mapping index FAISS → docstore. Le mapping permet de relier les positions FAISS aux documents réels.
index_to_docstore_id = {i: str(i) for i in range(len(documents))}

# Construction du vectorstore FAISS compatible LangChain
# combine l’index FAISS, les embeddings et le docstore pour LangChain. Il permet de faire la recherche par similarité.
vectorstore = FAISS(
    embedding_function=embeddings,      # fonction pour vectoriser les requêtes
    index=index,                        # index FAISS chargé
    docstore=docstore,                  # stockage des documents
    index_to_docstore_id=index_to_docstore_id
)


# Création du retriever. retriever va chercher les 5 documents les plus proches de la question via la distance vectorielle
# search_type="similarity" → recherche par distance vectorielle
# k=5 → on récupère les 5 documents les plus proches
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# -----------------------------
# LLM
# -----------------------------

# Création du LLM Mistral
llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=MISTRAL_API_KEY,
    temperature=0.3   # température pour contrôler la créativité de la réponse. faible température = réponses plus stables, haute température = réponses plus créatives
)


# Prompt système : cadre le comportement du modèle. Il donne les instructions au modèle.
system_prompt = (
    "Tu es un assistant qui recommande des événements culturels à Paris. " 
    "Utilise uniquement le contexte fourni pour répondre. " 
    "Pour chaque événement proposé, tu dois afficher obligatoirement : "
    "Titre, Date, Lieu (nom + adresse si disponible) et Description moyennement courte. "
    "Si une information est absente dans le contexte, indique 'Information non disponible'. "
    "Si l'information n'est pas présente, dis que tu ne sais pas, et informe que tu ne peux répondre que sur les événements culturels à Paris, et que pour d'autres questions, il faut un autre outil de réponse. " 
    "Contexte: {context}"
)

# Création du template de conversation. Il contient le prompt système et le prompt utilisateur.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), 
        ("human", "{input}"),  # {input} est la question de l'utilisateur
    ]
)

# -----------------------------
# CHAIN
# -----------------------------

# Création de la chaîne pour combiner les documents. Elle combine le LLM et le prompt pour générer la réponse à partir d’un ou plusieurs documents.
# create_stuff_documents_chain() est une fonction de LangChain qui permet de créer une chaîne de traitement des documents.
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# Création de la chaîne RAG. Elle combine le retriever et la chaîne de documents.
# Chaîne complète RAG :
# 1. Retriever → récupère les documents pertinents
# 2. LLM → génère la réponse à partir du contexte
qa_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("\n Chatbot événements Paris prêt !")
    print("Tape 'exit' pour quitter.\n")

    while True:
        # Récupération de la question utilisateur
        query = input("Votre question : ")

        if query.lower() == "exit": # convertit la question en minuscules pour comparer
            break # si la question est "exit", on quitte la boucle

        # Recherche des documents pertinents dans FAISS via le retriever
        # invoke est une méthode standard dans LangChain qui permet d’exécuter un objet “callable”. On lui donne une entrée, il renvoie un résultat
        docs = retriever.invoke(query)

        # Génération de la réponse via la chaîne RAG
        response = qa_chain.invoke({"input": query})

        print("\nRéponse :\n")
        # print(response["result"])
        print(response["answer"])   # affiche la réponse générée par le modèle LLM

        print("\nSources utilisées :")
        # Boucle sur chaque document dans la liste 'docs'
        for doc in docs:
            # Récupère le titre du document depuis les métadonnées
            print("-", doc.metadata.get("title_fr", "Sans titre"))

        # Affiche une ligne de séparation de 60 tirets + un saut de ligne avant et après
        print("\n" + "-"*60 + "\n")
        print(response.keys())  # Affiche les clés disponibles dans l'objet 'response' (utile pour debug)

# En résumé :
# Charger FAISS et ses métadonnées.
# Transformer les documents en objets LangChain.
# Créer un vectorstore avec embeddings Mistral.
# Créer un retriever pour trouver les documents pertinents.
# Créer un LLM (ChatMistralAI) avec un prompt système.
# Combiner retriever + LLM dans une chaîne RAG (qa_chain).
# Lancer le chatbot interactif
