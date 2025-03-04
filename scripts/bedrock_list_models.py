import boto3
import json

client = boto3.client(service_name="bedrock", region_name="us-east-1")

response = client.list_foundation_models()
# def print_schema(d, indent=0):
#     for key, value in d.items():
#         print(' ' * indent + str(key))
#         if isinstance(value, dict):
#             print_schema(value, indent + 2)
#         elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
#             print(' ' * (indent + 2) + '[list of dicts]')
#             print_schema(value[0], indent + 4)

# print_schema(response)

# Extract and print the model IDs
model_summaries = response.get('modelSummaries', [])
for model in model_summaries:
    print(model.get('modelId'))