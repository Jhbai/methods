import uuid
import chromadb
from chromadb.utils import embedding_functions

# ----- ChromaDB loads SentenceTransformer ----- #
PATH = "./LLM/all-MiniLM-L6-v2" # ref: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2; other ref: https://www.kaggle.com/models/google/embeddinggemma/
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=PATH)
client = chromadb.PersistentClient(path="./chroma_db")

# ----- Create Collection ----- #
collection_name = "my_local_model_collection"
collection = client.get_or_create_collection(name=collection_name, embedding_function=local_ef)  # Using local emb function

# ---- The function that packages memory ----- #
def add_memory(_type, message):
    docs = [message]
    meta = [{"source": "user", "type": _type}]
    ids = [str(uuid.uuid4())]
    collection.add(documents=docs, metadatas=meta, ids=ids)
    return "Knowledge stored !!"

# ----- Fetch the knowledge from chromaDB ----- #
def get_memory(message):
    result = list()
    results = collection.query(query_texts=[message], n_results=10)
    if results:
        for i, doc in enumerate(results.get('documents', [[]])[0]):
            dist = results.get('distances', [[]])[0][i]
            meta = results.get('metadatas', [[]])[0][i]
            print(f"[結果 {i}] meta: {meta}; distance: {dist}")
            result += [doc]
    return result
