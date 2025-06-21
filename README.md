
# 🧠 RAG QA Chatbot Application

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents (PDF or DOCX), ask questions in natural language, and receive accurate answers based on the content of the uploaded documents — all powered by a vector database and local or API-based LLMs.

---

## 🚀 Features

- ✅ Upload multiple documents (PDF, DOCX)
- ✅ Text chunking using LangChain's `RecursiveCharacterTextSplitter`
- ✅ Embedding with `sentence-transformers`
- ✅ Vector storage using FAISS with persistent caching
- ✅ Question-answering using:
  - 🔌 Together.ai API
  - 💻 Local HuggingFace LLMs (with CPU/GPU support)
- ✅ Select LLM and model at runtime via sidebar
- ✅ Adjustable top-k chunk retrieval slider
- ✅ Cached answers for faster repeated queries
- ✅ Reset App button to clear everything
- ✅ Chunk-source tracking and grouped display by document
- ✅ Works offline or online
- ✅ Fully containerized with Docker

---

## 🧑‍💻 How to Run

### 🔹 1. Clone the repository (optional for local dev)

```bash
git clone https://github.com/MehmetAltinkurt/rag-chatbot.git
cd rag-chatbot
```

---

### 🔹 2. Run with Docker (recommended)

You can pull and run the prebuilt image directly from Docker Hub:

```bash
docker pull mehmetaltinkurt/rag-chatbot:v2
docker run -p 8501:8501 --env-file .env mehmetaltinkurt/rag-chatbot:v2
```

To enable GPU support (if using `nvidia-docker`):

```bash
docker run --gpus all -p 8501:8501 --env-file .env mehmetaltinkurt/rag-chatbot:v2
```

---

## 🧪 Local Development

### 🔹 Install dependencies

```bash
pip install -r requirements.txt
```

Or with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 🔹 Set up `.env`

Create a `.env` file in your project root:

```env
TOGETHER_API_KEY=your_together_api_key
HUGGINGFACE_HUB_TOKEN=your_huggingface_token
```

---

### 🔹 Run the app locally

```bash
streamlit run rag_chatbot/ui.py
```

---

## 🤖 Run in Google Colab

1. Mount secrets securely:

```python
from google.colab import userdata
import os

os.environ["TOGETHER_API_KEY"] = userdata.get("TOGETHER_API_KEY")
os.environ["HUGGINGFACE_HUB_TOKEN"] = userdata.get("HUGGINGFACE_HUB_TOKEN")
```

2. Launch the app:

```python
!git clone https://github.com/MehmetAltinkurt/rag-chatbot.git
%cd rag-chatbot
!pip install -r requirements.txt
!streamlit run rag_chatbot/ui.py &
from pyngrok import ngrok
print(ngrok.connect(8501))
```

---

## 🧪 Usage Tips

- Use the sidebar to:
  - Choose LLM mode (Together.ai or Local)
  - Set the model name
  - Adjust Top-K chunks retrieved
- Uploaded files and answers are cached
- 💬 Answer and 📄 Source Chunks are shown side-by-side
- Click **Reset App** to clear files, chunks, and history

---

## 📁 Project Structure

```
rag_chatbot/
├── ui.py                  # Streamlit UI logic
├── file_handler.py        # PDF/DOCX parsing
├── text_splitter.py       # Chunking logic
├── vector_store.py        # FAISS logic
├── retriever.py           # Retriever wrapper
├── retriever_manager.py   # Session-safe retriever state
├── llm_api.py             # Together.ai completion
├── local_llm.py           # Transformers-based local LLM
├── llm_manager.py         # LLM mode switcher
data/                      # Stores chunks and FAISS index
```

---

## 📦 Docker Image

- 🐳 DockerHub: [`mehmetaltinkurt/rag-chatbot:v2`](https://hub.docker.com/r/mehmetaltinkurt/rag-chatbot)
- Includes full environment and runs offline or with Together.ai

---

## 🧾 License

MIT License
