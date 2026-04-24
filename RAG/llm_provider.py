import os
import json
import requests
from typing import Generator

class LLMProvider:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.model_name = os.getenv("LLM_MODEL", None)
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

        if self.model_name is None:
            raise ValueError("Model need to be specified")
        

    def generate(self, system_prompt: str, user_prompt: str, stream: bool = False):
        return self._call_ollama(system_prompt, user_prompt, stream)

    def _call_ollama(self, system: str, user: str, stream: bool):
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": stream,
            "options": {"temperature": 0.1, "num_ctx": 4096}
        }

        if stream:
            def generator():
                response = requests.post(url, json=payload, stream=True, timeout=20)
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token: yield token
                        if data.get("done"): break
            return generator()
        else:
            response = requests.post(url, json=payload, timeout=120)
            return response.json()["message"]["content"]
