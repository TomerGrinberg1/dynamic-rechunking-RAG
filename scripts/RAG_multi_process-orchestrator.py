import os
import json
import yaml
import nltk
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import concurrent.futures
from functools import partial

from Retrieval import RetrievalService
from expansion_service import ExpansionService
from LLM_interface import DeepSeekInterface

############################
# Your existing helper functions
############################

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_rag_prompt(question, retrieved_context, k=3):
    passages = "\n".join(
        [f"passage {i + 1}: {retrieved_context[i][0]}" for i in range(min(k, len(retrieved_context)))]
    )
    return f"""
            Answer the following question based on the given passages below: Keep the answer concise, use plain text only.

            ## Passages:
            {passages}

            ## Question: {question}

            ## Answer:
            """

def get_queries(path):
    df = pd.read_parquet(path)
    questions = df['question'].tolist()
    return questions

def get_real_answer_list(path):
    df = pd.read_parquet(path)
    answer_list = df['short_answers'].apply(lambda x: [item for sublist in x for item in sublist]).tolist()
    return answer_list

def calculate_threshold(initial_retrieval):
    first_value = initial_retrieval[0][1]
    last_value = initial_retrieval[-1][1]
    interval_threshold = first_value - last_value
    return interval_threshold

def is_real_answer_included(real_answers, llm_response):
    if not real_answers or not llm_response:
        return False
    llm_response = llm_response.strip().lower()
    for real_answer in real_answers:
        if real_answer.strip().lower() in llm_response:
            return True
    return False

############################
# Multiprocessing worker function
############################
def process_single_query(
    query_info,
    pinecone_index_name,
    pinecone_api_key,
    key,
    data_path
):
    """
    Process a single query in a separate process.

    :param query_info: A tuple (i, query, real_answer_list, n_neighbors).
    :param pinecone_index_name: Pinecone index name.
    :param pinecone_api_key: Pinecone API key.
    :param key: LLM service key.
    :param data_path: Path to the JSON data for retrieval service.

    :return: A dictionary with query result data.
    """
    i, query, real_answer_list, n_neighbors = query_info

    # Re-initialize services here in each process
    retrieval_service = RetrievalService(
        pinecone_index_name=pinecone_index_name,
        pinecone_api_key=pinecone_api_key
    )
    expansion_service = ExpansionService()
    llm_service = DeepSeekInterface(api_key=key)

    # Load or set data, if needed by your retrieval service
    # data = retrieval_service.load_data(data_path)
    
    # Main logic
    initial_retrieval = retrieval_service.search_dense(query, top_k=20)
    interval_threshold = calculate_threshold(initial_retrieval)

    distilled_passages = []
    for initial_passage in initial_retrieval:
        expanded_passage = expansion_service.expand_passages(
            initial_passage[0],
            n_neighbors=n_neighbors
        )
        distilled_passage = expansion_service.query_focused_rechunk(
            query,
            expanded_passage,
            float(initial_passage[1]) - interval_threshold
        )
        distilled_passages.append(distilled_passage)

    # Rerank the distilled passages
    rerenaked_passages = expansion_service.rerank_chunks(query, distilled_passages)
    rag_prompt = create_rag_prompt(query, rerenaked_passages)
    
    # Get LLM response
    response = llm_service.send_message(rag_prompt)
    
    # Check correctness
    is_correct = is_real_answer_included(real_answer_list, response)

    # Return the result so we can aggregate
    return {
        "index": i,
        "query": query,
        "llm_response": response,
        "real_answers": real_answer_list,
        "is_correct": is_correct,
        "n_neighbors": n_neighbors
    }

############################
# Main script
############################
def main():
    # Load your config
    with open("keys.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            key = config["api_key"]
        except yaml.YAMLError as exc:
            print(exc)
            return

    
    # Pinecone configs
    pinecone_index_name = "dynamic"
    pinecone_api_key = "pcsk_5SLo1E_SPpKBieD8wQRuRf9G7xfYrVWDLWUAkfRC3X5xjMpDhy7j8CH3SEen8kJiNbMjav"

    data_path = "/home/tomer/learning/dynamic-rechunking-RAG/data/5000_passage_to_location.json"
    queries = get_queries("/home/tomer/learning/dynamic-rechunking-RAG/data/short_answer_5000_rows.parquet")
    real_answers = get_real_answer_list("/home/tomer/learning/dynamic-rechunking-RAG/data/short_answer_5000_rows.parquet")

    hyperparams_status_file = "/home/tomer/learning/dynamic-rechunking-RAG/data/hyperparams_status.json"
    accuracy_file = "/home/tomer/learning/dynamic-rechunking-RAG/data/hyperparams_accuracy.json"

    if os.path.exists(hyperparams_status_file):
        with open(hyperparams_status_file, "r") as f:
            remaining_hyperparams = json.load(f)
    else:
        # Example list of n_neighbors
        remaining_hyperparams = {"n_neighbors": [0]}

    if os.path.exists(accuracy_file):
        with open(accuracy_file, "r") as f:
            accuracy_data = json.load(f)
    else:
        accuracy_data = {}

    for n_neighbors in tqdm(remaining_hyperparams["n_neighbors"], desc="n_neighbors Progress"):
        # Prepare tasks for parallel processing
        tasks = []
        for i, query in enumerate(queries):
            tasks.append((i, query, real_answers[i], n_neighbors))

        # Run tasks in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            # partial(...) to avoid passing repeated parameters in every submit
            func = partial(
                process_single_query,
                pinecone_index_name=pinecone_index_name,
                pinecone_api_key=pinecone_api_key,
                key=key,
                data_path=data_path
            )

            # Submit all tasks at once
            futures = [executor.submit(func, t) for t in tasks]

            # Collect results with a progress bar
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=f"Queries for n_neighbors={n_neighbors}"):
                res = f.result()
                results.append(res)

        # Now we have all results for the current n_neighbors
        # Let's compute accuracy and also write intermediate results
        total_queries = len(queries)
        correct_answers = sum(r["is_correct"] for r in results)
        accuracy = correct_answers / total_queries if total_queries > 0 else 0.0
        accuracy_data[n_neighbors] = accuracy

        # Write out the intermediate results for this hyperparameter
        result_file = f"/home/tomer/learning/dynamic-rechunking-RAG/data/results_n_neighbors_{n_neighbors}.json"
        with open(result_file, "w") as json_file:
            json.dump(results, json_file, indent=4)

        # Save accuracy
        with open(accuracy_file, "w") as json_file:
            json.dump(accuracy_data, json_file, indent=4)

        # Update the list of remaining hyperparams
        remaining_hyperparams["n_neighbors"].remove(n_neighbors)
        with open(hyperparams_status_file, "w") as f:
            json.dump(remaining_hyperparams, f, indent=4)

    print("All done!")

if __name__ == "__main__":
    main()
