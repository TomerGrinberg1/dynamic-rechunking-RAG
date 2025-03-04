import boto3
import json
from botocore.exceptions import ClientError

def send_message(model_id: str, prompt: str, region_name: str = 'us-east-1') -> str:
    """
    Sends a prompt to the specified Amazon Bedrock model and retrieves the response.

    Args:
        model_id (str): The Amazon Bedrock model ID.
        prompt (str): The input text prompt.
        region_name (str): AWS region where the Bedrock service is hosted. Default is 'us-east-1'.

    Returns:
        str: The model's response text.
    """
    # Initialize the Bedrock Runtime client
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=region_name)

    # Construct the message payload
    messages = [{'role': 'user', 'content': [{'text': prompt}]}]

    try:
        # Send the prompt to the model using the Converse API
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages
        )

        # Extract and return the model's response text
        output_message = response['output']['message']
        response_text = output_message['content'][0]['text']
        return response_text

    except ClientError as e:
        error_message = e.response['Error']['Message']
        print(f"An error occurred: {error_message}")
        return ""

# Small test script
if __name__ == "__main__":
    print("Sending a prompt to the model...")
    model_id = "meta.llama3-8b-instruct-v1:0"
    prompt = "What is the capital of France?"
    print("\nPrompt:\n", prompt)
    response = send_message(model_id, prompt)
    print("\nModel Response:\n", response)
