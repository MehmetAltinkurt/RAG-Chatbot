# Tie together retrieval and generation
from rag_chatbot.retriever import Retriever
from rag_chatbot.llm_api import TogetherLLM
from rag_chatbot.local_llm import LocalLLM

class RAGPipeline:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', #'mistralai/Mixtral-8x7B-Instruct-v0.1',
                 api_key=None):
        self.retriever = Retriever(model_name=embedding_model)
        self.local_llm = LocalLLM(model_name="mistralai/Mistral-7B-Instruct-v0.1")#"crumb/nano-mistral"
        self.together_llm = TogetherLLM(api_key=api_key, model='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free')

    def build_knowledge_base(self, text_chunks):
        #Index the given text chunks in the vector database.
        self.retriever.index_documents(text_chunks)

    def answer_question(self, question, use_local_llm=False, top_k=10):
        #Use RAG to answer a question: retrieve + generate.
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        context = "\\n".join(retrieved_chunks)
        prompt = (
            f"You are a helpful assistant. Use only the context below to answer the question clearly and concisely. "
            f"Do not include extra phrases like 'Let me know if I should continue'. "
            f"Respond in a single paragraph without repeating the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        if use_local_llm:
            return self.local_llm.generate(prompt)
        else:
            return self.llm.generate(prompt)

    def save_state(self, index_path="data/faiss.index", chunks_path="data/text_chunks.json"):
        #Save FAISS index and text chunks.
        self.retriever.save(index_path)
        self.retriever.vector_store.save_text_chunks(chunks_path)

    def load_state(self, index_path="data/faiss.index", chunks_path="data/text_chunks.json"):
        #Load FAISS index and text chunks.
        self.retriever.vector_store.load_text_chunks(chunks_path)
        self.retriever.load(self.retriever.vector_store.text_chunks, index_path)