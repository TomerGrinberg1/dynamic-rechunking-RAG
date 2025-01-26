
from openai import OpenAI

client = OpenAI(api_key="sk-6aa4c88ba3134cf9827695e32e2fd7c9", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)