from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import json
from  Retrieval import    RetrievalService
from  expansion_service import ExpansionService
from ollama_interface import OllamaInterface
import pandas as pd
import os


def load_data_from_json(file_path):
    """
    Load data from a JSON file.

    :param file_path: Path to the JSON file.
    :return: Data loaded from the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_rag_prompt(question, retrieved_context, k=2):
    """
    Create a RAG prompt dynamically using the top-k retrieved passages.

    :param question: The question to be answered.
    :param retrieved_context: A list of tuples where each tuple contains a passage and its relevance score.
    :param k: Number of passages to include in the prompt.
    :return: A formatted string containing the RAG prompt.
    """
    passages = "\n".join([f"passage {i + 1}: {retrieved_context[i][0]}" for i in range(min(k, len(retrieved_context)))])
    
    return f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            Answer the question based on the given passages below. Keep the answer concise.
            <|eot_id|><|start_header_id|>user<|end_header_id|>

            ## Passages:
            {passages}

            ## Question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>

            ##Answer:
            """


def get_queries(path):
    df = pd.read_parquet(path)
    questions = df['question'].head(1000).tolist()
    return questions


def calculate_threshold(initial_retrieval):
        first_value = initial_retrieval[0][1]
        last_value = initial_retrieval[-1][1]
        # Calculate the difference
        interval_threshold = first_value - last_value
        return interval_threshold
    

    #extract the first 1000 questions and return a list of those 


model = SentenceTransformer('intfloat/e5-base-v2')

# Initialize Pinecone
pinecone_index_name="dynamic"
pinecone_api_key="pcsk_5SLo1E_SPpKBieD8wQRuRf9G7xfYrVWDLWUAkfRC3X5xjMpDhy7j8CH3SEen8kJiNbMjav"
data_path="/home/tomer/learning/dynamic-rechunking-RAG/data/passage_to_location.json"
answer_path="/home/tomer/learning/dynamic-rechunking-RAG/data/answer_json.json"
answer_data=load_data_from_json(answer_path)
retrieval_service = RetrievalService(pinecone_index_name=pinecone_index_name, pinecone_api_key=pinecone_api_key)
expansion_service = ExpansionService()
ollama_service=OllamaInterface()
data=retrieval_service.load_data(data_path)
retrieval_service.index_and_upsert_documents(data, to_index_flag=True, to_upsert_flag=True, batch_size=64)
# queries=get_queries("/home/tomer/learning/dynamic-rechunking-RAG/data/first_1000_rows.parquet")
# n_neighbors_list = [1, 2, 5, 10, 20, 40, 60]

# hyperparams_status_file = "/home/tomer/dynamic-rechunking-RAG/data/hyperparams_status.json"

# if os.path.exists(hyperparams_status_file):
#     with open(hyperparams_status_file, "r") as f:
#         remaining_hyperparams = json.load(f)
# else:
#     remaining_hyperparams = {"n_neighbors": n_neighbors_list}

# for n_neighbors in tqdm(remaining_hyperparams["n_neighbors"], desc="n_neighbors Progress"):
#     answer_data = {}

#     for i, query in tqdm(enumerate(queries), desc=f"Processing queries for n_neighbors={n_neighbors}"):
#         distilled_passages = []
#         initial_retrieval = retrieval_service.search_dense(query, top_k=20)
#         interval_threshold = calculate_threshold(initial_retrieval)
        
#         for initial_passage in initial_retrieval:
#             expanded_passage = expansion_service.expand_passages(
#                 initial_passage[0], n_neighbors=n_neighbors
#             )
#             distilled_passage = expansion_service.query_focused_rechunk(
#                 query, expanded_passage, float(initial_passage[1]) - interval_threshold
#             )
#             distilled_passages.append(distilled_passage)
        
#         rerenaked_passages = expansion_service.rerank_chunks(query, distilled_passages)
#         rag_prompt = create_rag_prompt(query, rerenaked_passages)
#         response = ollama_service.send_message(rag_prompt)
#         answer_data[i] = {"model_response": response}
    
#     filename = f"/home/tomer/learning/dynamic-rechunking-RAG/data/result_n_neighbors_{n_neighbors}.json"
    
#     with open(filename, "w") as json_file:
#         json.dump(answer_data, json_file, indent=4)

#     remaining_hyperparams["n_neighbors"].remove(n_neighbors)
#     with open(hyperparams_status_file, "w") as f:
#         json.dump(remaining_hyperparams, f, indent=4)

    












   






