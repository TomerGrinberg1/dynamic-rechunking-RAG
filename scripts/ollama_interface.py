import requests
import json

import torch

class OllamaInterface:
    def __init__(self, url="http://localhost:11434/api/chat", model_name="llama3.2:1b"):
        self.url = url
        self.model_name = model_name

    def send_message(self, user_message):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        }

        try:
            with requests.post(self.url, json=payload, stream=True) as response:
                response.raise_for_status()

                assistant_reply = ""
                for line in response.iter_lines():
                    if line:
                        chunk = line.decode("utf-8")
                        try:
                            data = json.loads(chunk)
                            assistant_reply += data.get("message", {}).get("content", "")
                        except ValueError:
                            print("Warning: Skipping invalid JSON chunk:", chunk)

                return assistant_reply

        except requests.exceptions.RequestException as e:
            print("HTTP request failed:", e)
            return None

# Example usage
if __name__ == "__main__":
    ollama = OllamaInterface()
    reply = ollama.send_message("What is the capital of France?")
    print("Assistant reply:", reply)