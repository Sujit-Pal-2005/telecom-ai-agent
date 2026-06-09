import os
import re
import datetime
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configuration
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "telecom_knowledge"

class TelecomHybridAgent:
    def __init__(self):
        # Initialize Groq client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Initialize embedding model
        print("Loading embedding model in agent...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initialize Chroma client
        print("Loading ChromaDB client...")
        self.chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.knowledge_collection = self.chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        
        # SQL Database configuration
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "3306")
        self.db_user = os.getenv("DB_USER", "root")
        self.db_password = os.getenv("DB_PASSWORD", "")
        self.db_name = os.getenv("DB_NAME", "telecom_db")
        self.engine = None
        self._init_db_engine()

    def _init_db_engine(self):
        try:
            db_uri = f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            self.engine = create_engine(db_uri)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Successfully connected to MySQL database engine.")
        except Exception as e:
            print(f"Warning: Failed to connect to MySQL database: {e}")
            print("SQL queries will fail to execute unless a database connection is established.")
            self.engine = None

    def reconnect_db(self, host, port, user, password, db):
        self.db_host = host
        self.db_port = port
        self.db_user = user
        self.db_password = password
        self.db_name = db
        self._init_db_engine()
        return self.engine is not None

    def is_telecom_related(self, query: str) -> tuple[bool, str]:
        prompt = f"""
You are an AI guardrail assistant for a Telecom Network Operations Center (NOC).
Your task is to classify if the following User Query is related to telecom network operations, logs, base stations, cells, towers, handover/handoffs, signal strength, network congestion, call drops, or related recommendations.

User Query: "{query}"

Respond strictly in the following JSON format:
{{
  "is_telecom_related": true/false,
  "reason": "A brief 1-sentence explanation of why it is or is not telecom-related."
}}
"""
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a precise classifier that outputs only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            response_text = completion.choices[0].message.content
            result = json.loads(response_text)
            
            is_related = result.get("is_telecom_related", False)
            reason = result.get("reason", "No reason provided.")
            return is_related, reason
        except Exception as e:
            print(f"Error checking telecom intent: {e}")
            # Fallback to true to avoid blocking the user if Groq fails
            return True, "Fallback: proceeding with query."

    def generate_sql(self, query: str) -> str:
        current_date = datetime.date.today().strftime("%Y-%m-%d")
        
        prompt = f"""
You are an expert Text-to-SQL translator for a MySQL database containing telecom incident logs.
Generate a valid SQL query to retrieve data that answers the user's request.

Database Schema:
Table: `network_metrics`
Columns:
- `id` (INT, Primary Key)
- `Region` (VARCHAR) - Name of the city/region (e.g. 'Mumbai', 'Chennai', 'Bengaluru', 'Guwahati', 'Delhi', etc.)
- `Tower_ID` (VARCHAR) - Tower identifier (e.g. 'T287', 'T0165')
- `Date` (DATE) - Date of the log in YYYY-MM-DD format
- `Call_Drops` (INT) - Number of call drops recorded
- `Signal_Strength` (INT) - Signal strength in dBm. E.g., -95 is worse than -70. (Values range from -50 to -115)
- `Congestion_Level` (VARCHAR) - Traffic congestion level ('Low', 'Medium', 'High')
- `Handoff_Failure` (INT) - Handover failure percentage (values from 0 to 100)
- `Notes` (TEXT) - Remarks about the log (e.g. 'Power fluctuation', 'Maintenance activity', 'Heavy user load')

Rules for SQL generation:
1. ONLY return the executable SQL statement. Do not include markdown code block formatting (```sql ... ```) and do not explain the query.
2. The user queries may have slight variations in region names. Write queries using case-insensitive matching where appropriate (e.g., LOWER(Region) = 'chennai' or Region LIKE '%chennai%').
3. For comparisons of signal strength, remember that more negative values mean a WORSE signal (e.g. ORDER BY Signal_Strength ASC will give the worst signal strength).
4. If relative time is mentioned, use the current reference date of: {current_date}.
5. Limit results to a maximum of 25 rows unless the user specifically asks for more, to prevent context overflow.
6. Do NOT invent columns. Use only the columns in the schema.

User Request: "{query}"

Executable SQL Query:
"""
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a precise text-to-SQL translator. Output ONLY the raw SQL query. No markdown, no commentary."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            sql_query = completion.choices[0].message.content.strip()
            # Clean up markdown formatting if the model still generated it
            sql_query = re.sub(r"^```sql\s*", "", sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r"^```\s*", "", sql_query)
            sql_query = re.sub(r"\s*```$", "", sql_query)
            return sql_query.strip()
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return ""

    def execute_sql(self, sql_query: str) -> tuple[pd.DataFrame, str]:
        if not self.engine:
            return pd.DataFrame(), "Database connection is not initialized. Please configure MySQL first."
        
        try:
            print(f"Executing SQL: {sql_query}")
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(sql_query), conn)
            
            if df.empty:
                return df, "No matching records found in the database metrics."
            
            # Format context string
            formatted_rows = []
            for idx, row in df.iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                formatted_rows.append(f"- Row {idx+1}: {row_str}")
            
            context = "\n".join(formatted_rows)
            return df, context
        except Exception as e:
            error_msg = f"Error executing SQL query: {e}"
            print(error_msg)
            return pd.DataFrame(), error_msg

    def retrieve_vector_context(self, query: str, top_k: int = 3) -> str:
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            results = self.knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            if not docs:
                return "No domain knowledge found in vector store."
                
            formatted_docs = []
            for doc, meta, dist in zip(docs, metadatas, distances):
                # Filter out extremely poor matches if needed (lower cosine distance means higher similarity)
                if dist < 1.4:
                    source = meta.get("source", "Unknown source")
                    category = meta.get("category", "General")
                    section = meta.get("section", "")
                    doc_str = f"[{category} Knowledge - Source: {source} (Section: {section})]\n{doc}"
                    formatted_docs.append(doc_str)
                    
            if not formatted_docs:
                return "No highly relevant domain knowledge found in vector store."
                
            return "\n\n---\n\n".join(formatted_docs)
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return "Failed to retrieve domain knowledge from ChromaDB."

    def generate_final_report(self, query: str, sql_query: str, sql_context: str, vector_context: str):
        prompt = f"""
You are a Lead Telecom Network Operations Center (NOC) Analyst.
You are tasked with generating a comprehensive, professional incident analysis and resolution report based on database metrics and vector domain knowledge.

User Query:
"{query}"

Retrieved Database Metrics (from SQL Query: `{sql_query}`):
---
{sql_context}
---

Retrieved Telecom Domain Knowledge (from Vector DB):
---
{vector_context}
---

INSTRUCTIONS:
1. Synthesize the structured database metrics with the unstructured domain knowledge to explain the root cause and provide actionable recommendations.
2. Structure your response in a professional, markdown-formatted report containing the following sections:
   - ** Incident Overview**: Clear summary of the issue, affected regions/towers, and timestamp/dates of occurrence.
   - ** Database Metric Analysis**: Present the metrics (call drops, signal strength, congestion, handoffs) in a clear markdown table. Summarize key indicators.
   - ** Root Cause Analysis (RCA)**: Connect the data patterns to the telecom guidelines (e.g. correlation between low RSRP and high call drops, or high PRB utilization and congestion levels) and explain what is causing the degradation.
   - ** Actionable Recommendation Plan**: Provide short-term parameter tuning suggestions (e.g., Event A3 Hysteresis, Time-to-Trigger, Cell Individual Offset, electrical tilt adjustments, pilot power boosts) and long-term resolutions (e.g., backhaul upgrades, site densification) based on the domain knowledge.
3. Be precise, technical, and quantitative. Avoid vague statements. If some metrics or facts are missing, state what is unknown.
"""
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert telecom network analyst who generates detailed, technical, and highly structured incident reports."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                stream=True
            )
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"Error generating final report: {e}"
