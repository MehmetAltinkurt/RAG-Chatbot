import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfFolder
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    HfFolder.save_token(hf_token)

class LocalLLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", max_tokens=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto"
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    def generate(self, prompt, max_tokens=None):
        result = self.generator(prompt, max_new_tokens=max_tokens or 256)
        return result[0]["generated_text"]