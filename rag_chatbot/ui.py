import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import streamlit_authenticator as stauth

from rag_chatbot.file_handler import get_text_from_file
from rag_chatbot.text_splitter import chunk_with_langchain
from rag_chatbot.retriever_manager import RetrieverManager
from rag_chatbot.llm_manager import LLMManager

import tempfile
from dotenv import load_dotenv
load_dotenv()

import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

def reset(args):
    for key in list(st.session_state.keys()):
        if key != "file_uploader_key":
            del st.session_state[key]
    st.session_state["file_uploader_key"] += 1
    st.session_state["question_input"] = ""

try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state.get('authentication_status'):
    authenticator.logout("Logout","sidebar", callback=reset)
    st.sidebar.write(f'Welcome *{st.session_state.get("name")}*')

    # Load API key
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

    if st.button("🔄 Reset App"):
        for key in list(st.session_state.keys()):
            if (key != "file_uploader_key") & (key != "authentication_status"):
                del st.session_state[key]
        st.session_state["file_uploader_key"] += 1
        st.session_state["question_input"] = ""

    # Sidebar model selection
    llm_mode = st.sidebar.radio("LLM Mode", ["Online (Together.ai)", "Offline (Local LLM)"])
    use_local_llm = llm_mode == "Offline (Local LLM)"
    together_model = st.sidebar.text_input("Together.ai Model", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    local_model = st.sidebar.text_input("Local LLM Model", "Qwen/Qwen2.5-0.5B-Instruct")# "mistralai/Mistral-7B-Instruct-v0.1"

    # Initialize retriever once
    if "retriever_manager" not in st.session_state:
        st.session_state.retriever_manager = RetrieverManager()
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    # Initialize session state
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("processed_files", set())
    st.session_state.setdefault("qa_cache", {})
    st.session_state.setdefault("chunk_metadata", [])

    # Detect and reinit LLM if needed
    current_model = local_model if use_local_llm else together_model
    if (
        "llm_manager" not in st.session_state or
        "last_llm_settings" not in st.session_state or
        st.session_state.last_llm_settings != {"mode": llm_mode, "model": current_model}
    ):
        st.session_state.llm_manager = LLMManager(
            model_name=current_model,
            api_key=TOGETHER_API_KEY,
            use_local_llm=use_local_llm
        )
        st.session_state.last_llm_settings = {"mode": llm_mode, "model": current_model}
        st.sidebar.success(f"✅ Loaded {current_model} ({llm_mode})")

    # Sidebar top_k selection
    top_k = st.sidebar.slider("Top-K Chunks", min_value=1, max_value=20, value=5, step=1)

    # Upload and process files
    st.title("📄🧠 RAG QA Chatbot")
    st.markdown("Upload documents (PDF or DOCX) and ask questions about their content.")

    uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx"], accept_multiple_files=True, key=st.session_state["file_uploader_key"])

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
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(f"Processed: {uploaded_file.name}")
                    new_files_uploaded = True
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
        if new_files_uploaded:
            for i, chunk in enumerate(new_chunks):
                st.session_state.chunk_metadata.append({
                    "chunk": chunk,
                    "source": uploaded_file.name,
                    "split_id": f"{uploaded_file.name}_{i}"
                })
            st.session_state.chunks = [item["chunk"] for item in st.session_state.chunk_metadata]
            st.session_state.retriever_manager.build_knowledge_base(st.session_state.chunks)

    # Q&A
    st.subheader("Ask a question")
    user_question = st.text_input("Your question:", key="question_input")
    if user_question and st.session_state.chunks:
        with st.spinner("Retrieving answer..."):
            retrieved_chunks = st.session_state.retriever_manager.retrieve_context(user_question, top_k)
            doc_chunks = []
            for item in st.session_state.chunk_metadata:
                if item["chunk"] in retrieved_chunks:
                    doc_chunks.append(item)
            if user_question in st.session_state.qa_cache:
                answer = st.session_state.qa_cache[user_question]
            else:
                context = "\n\n".join(retrieved_chunks)
                prompt = (
                f"You are a helpful assistant. Use only the context below to answer the question clearly and concisely. "
                f"Do not include extra phrases like 'Let me know if I should continue'. "
                f"Respond in a single paragraph without repeating the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_question}\n\n"
                f"Answer:"
            )
                answer = st.session_state.llm_manager.answer(prompt)
                st.session_state.qa_cache[user_question] = answer

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### 💬 Answer")
                st.markdown(answer)
            with col2:
                st.markdown("### 📄 Resources")
                for chunk_data in doc_chunks:
                    st.markdown(f"[{chunk_data['split_id']}]\n{chunk_data['chunk']}")
                    st.markdown("---")
elif st.session_state.get('authentication_status') is False:
    st.error('Username/password is incorrect')
elif st.session_state.get('authentication_status') is None:
    st.warning('Please enter your username and password')
