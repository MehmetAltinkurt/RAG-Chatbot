import os
from autollm import AutoLLM
from huggingface_hub import login
login(os.getenv("HUGGINGFACE_HUB_TOKEN"))

class LocalLLM:
    def __init__(self, model_name="crumb/nano-mistral", max_tokens=256):
        self.llm = AutoLLM(model_name)
        self.max_tokens = max_tokens

    def generate(self, prompt, max_tokens=None):
        return self.llm(prompt)