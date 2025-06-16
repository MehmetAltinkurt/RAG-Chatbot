# Retrieve relevant chunks from vector DB
from rag_chatbot.vector_store import VectorStore

class Retriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.vector_store = VectorStore(model_name)

    def index_documents(self, text_chunks):
        #Index the provided text chunks.
        self.vector_store.build_index(text_chunks)

    def retrieve(self, query, top_k=5):
        #Retrieve top_k relevant text chunks for the query.
        return self.vector_store.search(query, top_k)

    def save(self, path="data/faiss.index"):
        #Save the current FAISS index to disk.
        self.vector_store.save_index(path)

    def load(self, chunks, path="data/faiss.index"):
        #Load FAISS index and associate it with the provided chunks.
        self.vector_store.load_index(chunks, path)