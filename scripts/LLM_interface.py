
from openai import OpenAI

class DeepSeekInterface:
    def __init__(self, api_key, base_url="https://api.deepseek.com", model_name="deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def send_message(self, user_message):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]


        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=1.3,
                stream=False
            )

            # Extract and return the assistant's reply
            return response.choices[0].message.content

        except Exception as e:
            print("Request to DeepSeek failed:", e)
            return None

# Example usage
if __name__ == "__main__":
    deepseek = DeepSeekInterface(api_key="sk-6aa4c88ba3134cf9827695e32e2fd7c9")
    reply = deepseek.send_message("What is the capital of France?")
    print("Assistant reply:", reply)




