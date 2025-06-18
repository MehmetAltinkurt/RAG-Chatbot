# Streamlit UI goes here
import streamlit as st
from rag_chatbot.file_handler import get_text_from_file
from rag_chatbot.rag_pipeline import RAGPipeline
from rag_chatbot.text_splitter import chunk_with_langchain

import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

# Load API key from environment variable or manually set here
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize RAGPipeline only once using Streamlit session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(api_key=TOGETHER_API_KEY)

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}

st.title("📄🧠 RAG QA Chatbot")
st.markdown("Upload documents (PDF or DOCX) and ask questions about their content.")

# File uploader
uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    new_chunks = []
    new_files_uploaded = False

    for uploaded_file in uploaded_files:
        _, ext = os.path.splitext(uploaded_file.name)
        if uploaded_file.name not in st.session_state.processed_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            try:
                text = get_text_from_file(tmp_file_path, ext)
                chunks = chunk_with_langchain(text)
                new_chunks.extend(chunks)
                st.success(f"Processed: {uploaded_file.name}")
                st.session_state.processed_files.add(uploaded_file.name)
                new_files_uploaded = True
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")

    if new_files_uploaded:
        st.session_state.chunks.extend(new_chunks)
        st.session_state.rag.build_knowledge_base(st.session_state.chunks)

# Chat input
st.subheader("Ask a question")
user_question = st.text_input("Your question:")

if user_question and "chunks" in st.session_state and st.session_state.chunks:
    with st.spinner("Retrieving answer..."):
        if user_question in st.session_state.qa_cache:
            answer = st.session_state.qa_cache[user_question]
        else:
            answer = st.session_state.rag.answer_question(user_question)
            st.session_state.qa_cache[user_question] = answer
        st.markdown(f"**Answer:** {answer}")