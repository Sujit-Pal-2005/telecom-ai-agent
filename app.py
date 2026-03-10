import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# LOAD ENV VARIABLES
load_dotenv()

# PAGE CONFIG
st.set_page_config(
    page_title="Telecom AI Agent",
    page_icon="📡",
    layout="wide"
)

# LOAD GROQ CLIENT
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# CACHE EMBEDDING MODEL
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# CACHE VECTOR DATABASE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "vectorstore")

@st.cache_resource
def load_vector_db():

    client = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY
    )

    collection = client.get_or_create_collection(
        name="telecom_docs"
    )

    return collection

collection = load_vector_db()


# SIDEBAR
st.sidebar.title("⚙ Settings")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

st.sidebar.markdown("**Model:** Llama-3.1-8B (Groq)")
st.sidebar.markdown("**Embedding:** MiniLM")
st.sidebar.markdown("**Vector DB:** ChromaDB")

# CHAT STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# APP TITLE
st.title("📡 Telecom Network AI Agent")
st.caption("RAG + Streaming LLM for Telecom Incident Analysis")

# DISPLAY CHAT HISTORY
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# VECTOR SEARCH TOOL
def query_vector_db(query, top_k=80):

    # Encode query (batch for efficiency)
    query_embedding = embedding_model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )

    docs = results["documents"][0]
    scores = results["distances"][0]

    # Optional filtering by similarity
    filtered_docs = []
    for doc, score in zip(docs, scores):
        if score < 1.2:   
            filtered_docs.append(doc)

    return filtered_docs

# STREAM LLM RESPONSE
def stream_llm(prompt):

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a telecom network troubleshooting expert."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        stream=True
    )

    for chunk in completion:

        delta = chunk.choices[0].delta.content

        if delta:
            yield delta


# USER INPUT
if user_query := st.chat_input("Ask about telecom network issues..."):

    # save user message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # assistant response
    with st.chat_message("assistant"):

        with st.spinner("Analyzing telecom logs..."):

            # VECTOR SEARCH
            logs = query_vector_db(user_query)

            context = "\n".join(logs)

            # PROMPT
#             prompt = f"""
# You are a senior telecom network analyst.

# User Query:
# {user_query}

# Relevant Telecom Logs:
# {context}

# Generate a structured telecom incident report.

# Issue Summary:
# Root Cause:
# Evidence from Logs:
# Recommended Resolution Steps:
# """
            prompt = f"""
You are a senior telecom network analyst.

User Query:
{user_query}

Relevant Telecom Logs:
{context}

TASK:
Analyze all logs collectively and produce ONE final consolidated telecom incident report.

IMPORTANT RULES:
- Generate ONLY ONE report
- Do NOT repeat reports
- Do NOT create multiple regions
- If multiple logs exist, summarize them into a single analysis

Output Format (STRICT):

Region:
Observation:
Root Cause:
Suggested Resolution:
"""
        # STREAM RESPONSE
        response = st.write_stream(
            stream_llm(prompt)
        )

    # SAVE RESPONSE
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )