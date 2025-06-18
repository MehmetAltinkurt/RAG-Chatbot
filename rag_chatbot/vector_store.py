# FAISS or other vector DB integration
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []

    def embed_text(self, texts):
        #Generate embeddings for a list of text chunks.
        return self.model.encode(texts, show_progress_bar=True)

    def build_index(self, chunks):
        #Build or update FAISS index by adding only new chunks.
        # Only process new chunks
        new_chunks = chunks[len(self.text_chunks):]
        if not new_chunks:
            return  # nothing new to add

        new_embeddings = self.embed_text(new_chunks)

        if self.index is None:
            # First-time setup
            dim = new_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(np.array(new_embeddings).astype("float32"))
        self.text_chunks.extend(new_chunks)

    def search(self, query, top_k=5):
        #Retrieve top_k relevant chunks for a query.
        if not self.index:
            raise ValueError("FAISS index not built.")
        query_embedding = self.embed_text([query])
        distances, indices = self.index.search(np.array(query_embedding).astype("float32"), top_k)
        return [self.text_chunks[i] for i in indices[0]]

    def save_index(self, path="data/faiss.index"):
        #Save FAISS index to disk.
        if self.index:
            faiss.write_index(self.index, path)

    def load_index(self, chunks, path="data/faiss.index"):
        #Load FAISS index from disk.
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            self.text_chunks = chunks
        else:
            raise FileNotFoundError(f"No FAISS index found at {path}")
    
    def save_text_chunks(self, path="data/text_chunks.json"):
        #Save the associated text chunks to a JSON file.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.text_chunks, f)

    def load_text_chunks(self, path="data/text_chunks.json"):
        #Load text chunks from a JSON file.
        with open(path, "r", encoding="utf-8") as f:
            self.text_chunks = json.load(f)