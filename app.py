import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# LOAD ENV VARIABLES
load_dotenv()

# GEMINI CLIENT
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# PAGE CONFIG
st.set_page_config(page_title="Telecom AI Agent", layout="wide")

# LOAD EMBEDDING MODEL
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# LOAD CHROMADB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "vectorstore")

client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

collection = client.get_or_create_collection(
    name="telecom_logs"
)

# SIDEBAR
st.sidebar.title("⚙ Settings")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

st.sidebar.markdown("**Model:** Gemini 2.0 Flash")
st.sidebar.markdown("**Vector DB:** ChromaDB")

# CHAT STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# MAIN UI
st.title("📡 Telecom Network AI Agent")
st.caption("RAG using ChromaDB + Gemini")

# SHOW CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# USER INPUT
if user_query := st.chat_input("Ask about telecom logs or call drop issues..."):

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    # EMBEDDING
    query_embedding = embedding_model.encode(user_query).tolist()

    # VECTOR SEARCH
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    docs = results["documents"][0]
    context = "\n\n".join(docs)

    # RAG PROMPT
    rag_prompt = f"""
You are a senior telecom network analyst.

Telecom Logs:
{context}

User Question:
{user_query}

Provide structured output:

Root Cause:
Evidence:
Recommended Solution:
"""

    # GEMINI CALL
    with st.chat_message("assistant"):
        with st.spinner("Analyzing telecom logs..."):

            response = client_gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=rag_prompt
            )

            answer = response.text

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )