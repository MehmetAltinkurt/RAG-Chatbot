from rag_chatbot.retriever import Retriever

class RetrieverManager:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.retriever = Retriever(model_name=embedding_model)

    def build_knowledge_base(self, chunks):
        self.retriever.index_documents(chunks)

    def retrieve_context(self, query, top_k):
        return self.retriever.retrieve(query, top_k)

    def save_state(self, index_path="data/faiss.index", chunks_path="data/text_chunks.json"):
        self.retriever.save(index_path)
        self.retriever.vector_store.save_text_chunks(chunks_path)

    def load_state(self, index_path="data/faiss.index", chunks_path="data/text_chunks.json"):
        self.retriever.vector_store.load_text_chunks(chunks_path)
        self.retriever.load(self.retriever.vector_store.text_chunks, index_path)