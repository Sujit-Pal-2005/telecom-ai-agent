import os
import streamlit as st
from agent import TelecomHybridAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config - Simple and centered
st.set_page_config(
    page_title="Telecom AI Agent",
    page_icon="📡",
    layout="wide"
)

# Initialize Agent in Session State
@st.cache_resource
def get_agent():
    return TelecomHybridAgent()

try:
    agent = get_agent()
    agent_loaded = True
except Exception as e:
    st.error(f"Failed to initialize the Telecom AI Agent: {e}")
    agent_loaded = False

st.title("Telecom AI Agent")

# Sidebar options
st.sidebar.header("Options")
if st.sidebar.button("Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if agent_loaded:
    st.sidebar.subheader("System Status")
    groq_status = " Connected" if os.getenv("GROQ_API_KEY") else " Missing Groq Key"
    db_status = " Connected" if agent.engine is not None else " Disconnected"
    st.sidebar.write(f"**Groq API:** {groq_status}")
    st.sidebar.write(f"**MySQL DB:** {db_status}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input and AI Response Pipeline
if user_query := st.chat_input("Ask a telecom question..."):
    # Show user query immediately
    with st.chat_message("user"):
        st.markdown(user_query)


    # Save user query to history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate Response
    with st.chat_message("assistant"):
        if not agent_loaded:
            st.error("Agent is not initialized. Please configure your .env file.")
        else:
            with st.spinner("Analyzing..."):
                # 1. Classification
                is_related, classification_reason = agent.is_telecom_related(user_query)
                
                if not is_related:
                    refusal = f"**Guardrail Active**: This agent is optimized specifically for telecom network operations.\n\n*Reason: {classification_reason}*"
                    st.markdown(refusal)
                    st.session_state.messages.append({"role": "assistant", "content": refusal})
                else:
                    # 2. SQL RAG Pipeline
                    sql_query = agent.generate_sql(user_query)
                    sql_df, sql_context = agent.execute_sql(sql_query)

                    # 3. Vector DB Domain Knowledge Retrieval
                    vector_context = agent.retrieve_vector_context(user_query, top_k=3)

                    # 4. Ingest previous chat summary as context if available
                    

                    # 5. Stream the final response
                    response_placeholder = st.empty()
                    full_response = ""
                    for response_chunk in agent.generate_final_report(
                        query=user_query,
                        sql_query=sql_query,
                        sql_context=sql_context,
                        vector_context=vector_context
                    ):
                        full_response += response_chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

                    # Save response to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})