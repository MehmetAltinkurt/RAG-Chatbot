# Streamlit UI goes here
import streamlit as st
from rag_chatbot.file_handler import get_text_from_file
from rag_chatbot.rag_pipeline import RAGPipeline

import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

# Load API key from environment variable or manually set here
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize RAGPipeline only once using Streamlit session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(api_key=TOGETHER_API_KEY)
#print(st.session_state.rag.retriever.vector_store.text_chunks)
st.title("📄🧠 RAG QA Chatbot")
st.markdown("Upload documents (PDF or DOCX) and ask questions about their content.")

# File uploader
uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx"], accept_multiple_files=True)

all_text_chunks = []

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        _, ext = os.path.splitext(uploaded_file.name)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            text = get_text_from_file(tmp_file_path, ext)
            chunks = text.split("\\n\\n")  # Naive chunking; can be replaced with a better splitter
            #todo: better chunking with langchain
            all_text_chunks.extend(chunks)
            st.success(f"Processed: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")

    if all_text_chunks:
        st.session_state.rag.build_knowledge_base(all_text_chunks)
        st.session_state.chunks = all_text_chunks

# Chat interface
st.subheader("Ask a question")
user_question = st.text_input("Your question:")

if user_question and "chunks" in st.session_state:
    with st.spinner("Retrieving answer..."):
        answer = st.session_state.rag.answer_question(user_question)
        st.markdown(f"**Answer:** {answer}")