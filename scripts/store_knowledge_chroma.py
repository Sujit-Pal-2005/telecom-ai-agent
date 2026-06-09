import os
import chromadb
from sentence_transformers import SentenceTransformer

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "knowledge")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "vectorstore")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "telecom_knowledge"

def load_and_chunk_knowledge():
    chunks = []
    metadata_list = []
    
    files = {
        "weak_signal.txt": "Weak Signal",
        "congestion.txt": "Network Congestion",
        "handoff.txt": "Handoff / Handover Failure",
        "recommendation.txt": "Standard Recommendations"
    }
    
    for filename, category in files.items():
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found in {KNOWLEDGE_DIR}. Skipping.")
            continue
            
        print(f"Processing {filename}...")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split by double newline to separate sections/paragraphs
        raw_sections = content.split("\n\n")
        
        current_section_title = category
        for section in raw_sections:
            section = section.strip()
            if not section:
                continue
                
            # If the section looks like a header, we can prepend it or track it
            if section.startswith("===") or (section.isupper() and len(section) < 50):
                current_section_title = section.replace("===", "").strip()
                continue
                
            # Keep meaningful sections
            chunks.append(section)
            metadata_list.append({
                "category": category,
                "section": current_section_title,
                "source": filename
            })
            
    print(f"Total chunks extracted: {len(chunks)}")
    return chunks, metadata_list

def store_knowledge():
    chunks, metadata_list = load_and_chunk_knowledge()
    if not chunks:
        print("No chunks to store.")
        return
        
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Generating embeddings...")
    embeddings = model.encode(chunks)
    
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Get or create the collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    print(f"Adding {len(chunks)} documents to Chroma collection '{COLLECTION_NAME}'...")
    
    ids = [f"knowledge_{i}" for i in range(len(chunks))]
    
    # Store documents, embeddings, and metadata
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata_list,
        ids=ids
    )
    
    print("Knowledge stored in ChromaDB successfully!")

if __name__ == "__main__":
    store_knowledge()
