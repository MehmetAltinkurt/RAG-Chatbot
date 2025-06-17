# Tie together retrieval and generation
from rag_chatbot.retriever import Retriever
from rag_chatbot.llm_api import TogetherLLM

class RAGPipeline:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', #'mistralai/Mixtral-8x7B-Instruct-v0.1',
                 api_key=None):
        self.retriever = Retriever(model_name=embedding_model)
        self.llm = TogetherLLM(api_key=api_key, model=llm_model)

    def build_knowledge_base(self, text_chunks):
        #Index the given text chunks in the vector database.
        self.retriever.index_documents(text_chunks)

    def answer_question(self, question, top_k=5):
        #Use RAG to answer a question: retrieve + generate.
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        context = "\\n".join(retrieved_chunks)
        prompt = f"Use the following context to answer the question.\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\\nAnswer:"
        return self.llm.generate(prompt)

    def save_index(self, path="data/faiss.index"):
        self.retriever.save(path)

    def load_index(self, chunks, path="data/faiss.index"):
        self.retriever.load(chunks, path)