# import os
# import json
# from pathlib import Path
# import faiss
# from dotenv import load_dotenv

# from langchain_community.vectorstores import FAISS
# from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
# #from langchain.chains import RetrievalQA
# #from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain_classic.chains import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.documents import Document
# from langchain.prompts import PromptTemplate
# from langchain_community.docstore.in_memory import InMemoryDocstore

# # -----------------------------
# # CONFIG
# # -----------------------------
# BASE_DIR = Path(__file__).resolve().parent.parent
# FAISS_INDEX_PATH = BASE_DIR / "data/faiss_index.index"
# FAISS_METADATA_PATH = BASE_DIR / "data/faiss_metadata.json"

# load_dotenv()
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# if not MISTRAL_API_KEY:
#     raise ValueError("Clé API Mistral manquante")

# # -----------------------------
# # EMBEDDINGS
# # -----------------------------
# embeddings = MistralAIEmbeddings(
#     model="mistral-embed",
#     mistral_api_key=MISTRAL_API_KEY
# )

# # -----------------------------
# # LOAD FAISS MANUEL
# # -----------------------------
# print("Chargement de l'index FAISS...")
# index = faiss.read_index(str(FAISS_INDEX_PATH))

# print("Chargement des métadonnées...")
# with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
#     metadata = json.load(f)

# # -----------------------------
# # RECONSTRUCTION DES DOCUMENTS
# # -----------------------------
# documents = [
#     Document(page_content=item["text"], metadata=item["metadata"])
#     for item in metadata
# ]

# # -----------------------------
# # BUILD VECTORSTORE LANGCHAIN
# # -----------------------------

# # Création d'un docstore compatible LangChain
# docstore = InMemoryDocstore(
#     {str(i): doc for i, doc in enumerate(documents)}
# )

# # Mapping index FAISS → docstore
# index_to_docstore_id = {i: str(i) for i in range(len(documents))}

# # Construction du vectorstore
# vectorstore = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=docstore,
#     index_to_docstore_id=index_to_docstore_id
# )

# # vectorstore = FAISS(
# #     embedding_function=embeddings,
# #     index=index,
# #     docstore=dict(zip(range(len(documents)), documents)),
# #     index_to_docstore_id={i: i for i in range(len(documents))}
# # )

# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 5}
# )


# # -----------------------------
# # LLM
# # -----------------------------

# llm = ChatMistralAI(
#     model="mistral-small-latest",
#     mistral_api_key=MISTRAL_API_KEY,
#     temperature=0.3
# )

# # -----------------------------
# # PROMPT
# # -----------------------------
# template = """
# Tu es un assistant qui recommande des événements culturels à Paris.
# Pour chaque question de l’utilisateur, utilise uniquement les informations disponibles dans {context}.
# Réponds de manière claire, structurée et adaptée aux préférences de l'utilisateur.

# Question :
# {question}

# Si aucune information pertinente n’est disponible, indique-le.

# """

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"]
# )

# # -----------------------------
# # CHAIN
# # -----------------------------

# # Création de la chaîne pour combiner les documents
# question_answer_chain = create_stuff_documents_chain(
#     llm=llm,
#     prompt=prompt
# )

# # Création de la chaîne RAG
# qa_chain = create_retrieval_chain(
#     retriever=retriever,
#     combine_documents_chain=question_answer_chain,
#     return_source_documents=True
# )

# # qa_chain = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     retriever=retriever,
# #     chain_type="stuff",
# #     chain_type_kwargs={"prompt": prompt},
# #     return_source_documents=True
# # )

# # -----------------------------
# # MAIN
# # -----------------------------
# if __name__ == "__main__":

#     print("\n🤖 Chatbot événements Paris prêt !")
#     print("Tape 'exit' pour quitter.\n")

#     while True:
#         query = input("Votre question : ")

#         if query.lower() == "exit":
#             break

#         # response = qa_chain.invoke({"query": query})
#         response = qa_chain.invoke({"input": query})

#         print("\nRéponse :\n")
#         print(response["result"])
#         print("\nSources utilisées :")
#         for doc in response["source_documents"]:
#             print("-", doc.metadata.get("title_fr", "Sans titre"))
#         print("\n" + "-"*60 + "\n")



import os
import json
from pathlib import Path
import faiss
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
#from langchain.chains import RetrievalQA
#from langchain.chains.retrieval_qa.base import RetrievalQA
#from langchain.prompts import PromptTemplate
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

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("Clé API Mistral manquante")

# -----------------------------
# EMBEDDINGS
# -----------------------------
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=MISTRAL_API_KEY
)

# -----------------------------
# LOAD FAISS MANUEL
# -----------------------------
print("Chargement de l'index FAISS...")
index = faiss.read_index(str(FAISS_INDEX_PATH))

print("Chargement des métadonnées...")
with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# -----------------------------
# RECONSTRUCTION DES DOCUMENTS
# -----------------------------
documents = [
    Document(page_content=item["text"], metadata=item["metadata"])
    for item in metadata
]

# -----------------------------
# BUILD VECTORSTORE LANGCHAIN
# -----------------------------

# Création d'un docstore compatible LangChain
docstore = InMemoryDocstore(
    {str(i): doc for i, doc in enumerate(documents)}
)

# Mapping index FAISS → docstore
index_to_docstore_id = {i: str(i) for i in range(len(documents))}

# Construction du vectorstore
vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# vectorstore = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=dict(zip(range(len(documents)), documents)),
#     index_to_docstore_id={i: i for i in range(len(documents))}
# )

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# -----------------------------
# LLM
# -----------------------------

llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=MISTRAL_API_KEY,
    temperature=0.3
)

# -----------------------------
# PROMPT
# -----------------------------
# system_prompt = (
#     "Tu es un assistant qui recommande des événements culturels à Paris. "
#     "Utilise uniquement le contexte fourni pour répondre. "
#     "Si l'information n'est pas présente, dis que tu ne sais pas. "
#     "Contexte: {context}"
# )

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"]
# )

system_prompt = (
    "Tu es un assistant qui recommande des événements culturels à Paris. " 
    "Utilise uniquement le contexte fourni pour répondre. " 
    "Pour chaque événement proposé, tu dois afficher obligatoirement : "
    "Titre, Date, Lieu (nom + adresse si disponible) et Description moyennement courte. "
    "Si une information est absente dans le contexte, indique 'Information non disponible'. "
    "Si l'information n'est pas présente, dis que tu ne sais pas, et informe que tu ne peux répondre que sur les événements culturels à Paris, et que pour d'autres questions, il faut un autre outil de réponse. " 
    "Contexte: {context}"
)


# "Si l'information n'est pas présente, dis que tu ne sais pas, et informe que tu ne peux répondre que sur les événements culturels à Paris, et que pour d'autres questions, il faut un autre outil de réponse."
    
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# -----------------------------
# CHAIN
# -----------------------------

# Création de la chaîne pour combiner les documents
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# Création de la chaîne RAG

qa_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# qa_chain = create_retrieval_chain(
#     retriever=retriever,
#     combine_documents_chain=question_answer_chain,
#     return_source_documents=True
# )

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt},
#     return_source_documents=True
# )

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("\n🤖 Chatbot événements Paris prêt !")
    print("Tape 'exit' pour quitter.\n")

    while True:
        query = input("Votre question : ")

        if query.lower() == "exit":
            break

        # docs = retriever._get_relevant_documents(query)
        # response = qa_chain.invoke({"query": query})
        docs = retriever.invoke(query)

        response = qa_chain.invoke({"input": query})

        print("\nRéponse :\n")
        # print(response["result"])
        print(response["answer"])

        print("\nSources utilisées :")
        for doc in docs:
            print("-", doc.metadata.get("title_fr", "Sans titre"))

        print("\n" + "-"*60 + "\n")
        print(response.keys())
