from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import json
from Retrieval import RetrievalService
from expansion_service import ExpansionService
from ollama_interface import OllamaInterface
import pandas as pd
import os
import nltk
import yaml

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_rag_prompt(question, retrieved_context, k=3):
    passages = "\n".join([f"passage {i + 1}: {retrieved_context[i][0]}" for i in range(min(k, len(retrieved_context)))])
    return f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Answer the question based on the given passages below. Keep the answer concise. use plain text only.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            ## Passages:
            {passages}
            ## Question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            ##Answer:
            """

def get_queries(path):
    df = pd.read_parquet(path)
    return df['question'].tolist()

def get_real_answer_list(path):
    df = pd.read_parquet(path)
    answer_list = df['short_answers'].apply(lambda x: [item for sublist in x for item in sublist]).tolist()
    return answer_list

def calculate_threshold(initial_retrieval):
    first_value = initial_retrieval[0][1]
    last_value = initial_retrieval[-1][1]
    return first_value - last_value

def is_real_answer_included(real_answers: list[str], llm_response: str) -> bool:
    if not real_answers or not llm_response:
        return False
    llm_response = llm_response.strip().lower()
    for real_answer in real_answers:
        if real_answer.strip().lower() in llm_response:
            return True
    return False

# Initialize models and services
model = SentenceTransformer('intfloat/e5-base-v2')
pinecone_index_name = "dynamic"
pinecone_api_key = "pcsk_5SLo1E_SPpKBieD8wQRuRf9G7xfYrVWDLWUAkfRC3X5xjMpDhy7j8CH3SEen8kJiNbMjav"
with open("keys.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        key = config["api_key"]
    except yaml.YAMLError as exc:
        print(exc)
        exit()

data_path = "/home/student/dynamic-rechunking-RAG/data/5000_passage_to_location.json"
retrieval_service = RetrievalService(pinecone_index_name=pinecone_index_name, pinecone_api_key=pinecone_api_key)
expansion_service = ExpansionService()
llm_service = OllamaInterface()
data = retrieval_service.load_data(data_path)
# retrieval_service.index_and_upsert_documents(data, to_index_flag=True, to_upsert_flag=True, batch_size=64)
queries = get_queries("/home/student/dynamic-rechunking-RAG/data/short_answer_5000_rows.parquet")
real_answers = get_real_answer_list('/home/student/dynamic-rechunking-RAG/data/short_answer_5000_rows.parquet')

# Define hyperparameter spaces
n_neighbors_list = [0, 1, 2, 5, 10]
k_list = [3, 5, 10]

# Create combinations for use_knn False and True experiments
use_knn_false_combos = [{"n_neighbors": n} for n in n_neighbors_list]
use_knn_true_combos = [{"n_neighbors": n, "k": k} for n in n_neighbors_list for k in k_list]

# Files for saving hyperparameter status and accuracy (results saved in JSON in the specified output folder)
status_file_path = "/home/student/dynamic-rechunking-RAG/scripts/500_outputs/hyperparams_status.json"
accuracy_file_path = "/home/student/dynamic-rechunking-RAG/scripts/500_outputs/hyperparams_accuracy.json"

if os.path.exists(status_file_path):
    with open(status_file_path, "r") as f:
        remaining_hyperparams = json.load(f)
else:
    remaining_hyperparams = {
        "False": use_knn_false_combos,
        "True": use_knn_true_combos
    }

if os.path.exists(accuracy_file_path):
    with open(accuracy_file_path, "r") as f:
        accuracy_data = json.load(f)
else:
    accuracy_data = {}

# ----- Experiment: use_knn = False -----
for combo in tqdm(remaining_hyperparams["False"], desc="Processing use_knn=False combinations"):
    n_neighbors = combo["n_neighbors"]
    correct_answers = 0
    total_queries = len(queries)
    results = []

    for i, query in tqdm(enumerate(queries), desc=f"Queries for use_knn=False, n_neighbors={n_neighbors}"):
        if i >= 500:
            break
        distilled_passages = []
        initial_retrieval = retrieval_service.search_dense(query, top_k=20)
        interval_threshold = calculate_threshold(initial_retrieval)

        for initial_passage in initial_retrieval:
            expanded_passage = expansion_service.expand_passages(initial_passage[0], n_neighbors=n_neighbors)
            threshold_value = float(initial_passage[1]) - interval_threshold
            distilled_passage = expansion_service.query_focused_rechunk(
                query, expanded_passage, threshold_value, use_knn=False
            )
            distilled_passages.append(distilled_passage)

        rerenaked_passages = expansion_service.rerank_chunks(query, distilled_passages)
        rag_prompt = create_rag_prompt(query, rerenaked_passages)
        response = llm_service.send_message(rag_prompt)

        is_correct = is_real_answer_included(real_answers[i], response)
        if is_correct:
            correct_answers += 1

        result_data = {
            "query": query,
            "llm_response": response,
            "real_answers": real_answers[i],
            "is_correct": is_correct
        }
        results.append(result_data)

        # Save intermediate results per query
        result_file = f"/home/student/dynamic-rechunking-RAG/scripts/500_outputs/results_use_knn_False_n_neighbors_{n_neighbors}.json"
        with open(result_file, "w") as json_file:
            json.dump(results, json_file, indent=4)

    accuracy = correct_answers / total_queries if total_queries > 0 else 0.0
    key_name = f"use_knn_False_n_neighbors_{n_neighbors}"
    accuracy_data[key_name] = accuracy

    with open(accuracy_file_path, "w") as json_file:
        json.dump(accuracy_data, json_file, indent=4)

    remaining_hyperparams["False"].remove(combo)
    with open(status_file_path, "w") as f:
        json.dump(remaining_hyperparams, f, indent=4)

# ----- Experiment: use_knn = True -----
for combo in tqdm(remaining_hyperparams["True"], desc="Processing use_knn=True combinations"):
    n_neighbors = combo["n_neighbors"]
    k = combo["k"]
    correct_answers = 0
    total_queries = len(queries)
    results = []

    for i, query in tqdm(enumerate(queries), desc=f"Queries for use_knn=True, n_neighbors={n_neighbors}, k={k}"):
        if i >= 500:
            break
        distilled_passages = []
        initial_retrieval = retrieval_service.search_dense(query, top_k=10)
        interval_threshold = calculate_threshold(initial_retrieval)

        for initial_passage in initial_retrieval:
            expanded_passage = expansion_service.expand_passages(initial_passage[0], n_neighbors=n_neighbors)
            threshold_value = float(initial_passage[1]) - interval_threshold
            distilled_passage = expansion_service.query_focused_rechunk(
                query, expanded_passage, threshold_value, use_knn=True, k=k
            )
            distilled_passages.append(distilled_passage)

        rerenaked_passages = expansion_service.rerank_chunks(query, distilled_passages)
        rag_prompt = create_rag_prompt(query, rerenaked_passages)
        response = llm_service.send_message(rag_prompt)

        is_correct = is_real_answer_included(real_answers[i], response)
        if is_correct:
            correct_answers += 1

        result_data = {
            "index": i,
            "query": query,
            "llm_response": response,
            "real_answers": real_answers[i],
            "is_correct": is_correct
        }
        results.append(result_data)

        result_file = f"/home/student/dynamic-rechunking-RAG/scripts/500_outputs/results_use_knn_True_n_neighbors_{n_neighbors}_k_{k}.json"
        with open(result_file, "w") as json_file:
            json.dump(results, json_file, indent=4)

    accuracy = correct_answers / total_queries if total_queries > 0 else 0.0
    key_name = f"use_knn_True_n_neighbors_{n_neighbors}_k_{k}"
    accuracy_data[key_name] = accuracy

    with open(accuracy_file_path, "w") as json_file:
        json.dump(accuracy_data, json_file, indent=4)

    remaining_hyperparams["True"].remove(combo)
    with open(status_file_path, "w") as f:
        json.dump(remaining_hyperparams, f, indent=4)
