
# 🧠 RAG QA Chatbot Application

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents (PDF or DOCX), ask questions in natural language, and receive accurate answers based on the content of the uploaded documents — all powered by a vector database and LLMs.

---

## 🚀 Features

- ✅ Upload multiple documents (PDF, DOCX)
- ✅ Text chunking using LangChain's `RecursiveCharacterTextSplitter`
- ✅ Embedding with `sentence-transformers`
- ✅ Vector storage using FAISS with incremental updates
- ✅ Question-answering with Together.ai API (e.g., Mistral-7B-Instruct)
- ✅ Caching of previous questions to speed up repeated queries
- ✅ Persistent FAISS index and chunk storage between sessions
- ✅ Clean and simple Streamlit UI
- ✅ Fully containerized with Docker

---

## 📦 Installation (Local Dev)

```bash
# Clone the repository
git clone https://github.com/MehmetAltinkurt/rag-chatbot.git
cd rag-chatbot

# Set up Python environment
pip install -r requirements.txt

# Create a .env file
echo "TOGETHER_API_KEY=your_api_key_here" > .env

# Run the app
streamlit run rag_chatbot/ui.py
```

---

## 🐳 Run from DockerHub

```bash
docker run -p 8501:8501 mehmetaltinkurt/rag-chatbot:v1
```
---

## 🔑 Environment Variables

| Variable          | Description                  |
|------------------|------------------------------|
| `TOGETHER_API_KEY` | Your Together.ai API key     |

---

## 🧪 How It Works

1. Upload documents (PDF or DOCX)
2. Each document is parsed and split into overlapping chunks
3. Chunks are embedded and added to the FAISS index
4. When a question is asked:
   - Top-10 relevant chunks are retrieved
   - A prompt is built and sent to the LLM (Together.ai)
   - The answer is generated and returned

---

## 📁 Project Structure

```
rag_chatbot/
├── ui.py                  # Streamlit UI
├── file_handler.py        # PDF/DOCX reading
├── text_splitter.py       # LangChain chunking
├── vector_store.py        # FAISS with persistent chunks
├── retriever.py           # Retrieval logic
├── llm_api.py             # Together.ai interaction
├── rag_pipeline.py        # Orchestrates RAG pipeline
data/                      # Stored FAISS index + chunks
Dockerfile
requirements.txt
.env
```

---

## ✅ Evaluation Highlights

- 🔍 **Relevant Document Retrieval**: LangChain splitter + FAISS for fast retrieval
- 🧠 **LLM Output**: Controlled prompt format + stop sequences
- 🧪 **Tested**: Dockerized and tested with 100+ page documents

---

## 📬 Submission Info

This project was developed as part of a technical interview task. For more information, see the original task.

---

## 📜 License

MIT License

