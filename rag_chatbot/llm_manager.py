from rag_chatbot.local_llm import LocalLLM
from rag_chatbot.llm_api import TogetherLLM

class LLMManager:
    def __init__(self, model_name, api_key=None, use_local_llm=False):
        if use_local_llm:
            self.llm = LocalLLM(model_name=model_name)
        else:
            self.llm = TogetherLLM(api_key=api_key, model=model_name)

    def answer(self, prompt):
        return self.llm.generate(prompt)