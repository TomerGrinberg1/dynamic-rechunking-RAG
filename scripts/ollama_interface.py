import requests
import json  # Import the json module for parsing JSON

url = "http://localhost:11434/api/chat"
model_name = "llama3.1"

# Define the payload
payload = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
}

# Send the request with streaming enabled
try:
    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()  # Check for HTTP errors

        # Collect and assemble the streamed content
        assistant_reply = ""
        for line in response.iter_lines():
            if line:
                # Parse each line as JSON
                chunk = line.decode("utf-8")
                try:
                    data = json.loads(chunk)                      # Append the assistant's response chunk
                    assistant_reply += data.get("message", {}).get("content", "")
                except ValueError:
                    print("Warning: Skipping invalid JSON chunk:", chunk)

        print("Assistant reply:", assistant_reply)

except requests.exceptions.RequestException as e:
    print("HTTP request failed:", e)
