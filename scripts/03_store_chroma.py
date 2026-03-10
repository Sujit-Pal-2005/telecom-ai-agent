import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# CONFIGURATION

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "chunks", "text_chunks.json")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "vectorstore")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# MAIN FUNCTION

def store_in_chroma(input_path, persist_directory):

    print("Loading text chunks...")

    with open(input_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    print(f"Total chunks loaded: {len(texts)}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(texts)

    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_or_create_collection(name="telecom_docs")

    print("Storing embeddings into ChromaDB...")

    ids = [str(i) for i in range(len(texts))]

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids
    )

    print("Data stored successfully!")
    print(f"Database location: {persist_directory}")


if __name__ == "__main__":
    store_in_chroma(INPUT_PATH, PERSIST_DIRECTORY)