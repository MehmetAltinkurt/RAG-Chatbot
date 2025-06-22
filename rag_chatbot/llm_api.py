# Interact with LLM API
import requests
import os

class TogetherLLM:
    def __init__(self, api_key=None, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.api_url = "https://api.together.xyz/v1/completions"
        self.model = model

    def generate(self, prompt, max_tokens=512, temperature=0.7):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["\nUser:", "\nQuestion:", "\nAnswer:", "Another question, please.", "instructionFollowing"]
        }

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            print("Error:", response.status_code, response.text)
            return "Sorry, I couldn't process your request.\n" + response.text