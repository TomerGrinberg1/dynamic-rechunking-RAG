import json
import os
import yaml
from tqdm import tqdm
import pandas as pd

from sentence_transformers import SentenceTransformer
# Import your service interfaces
from Retrieval import RetrievalService
from ollama_interface import OllamaInterface

# Import Together client for LLM evaluation
from together import Together

########################################
# Helper Functions
########################################

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_rag_prompt(question, retrieved_context, k=3):
    # Concatenate the top-k passages into a prompt
    passages = "\n".join(
        [f"passage {i + 1}: {retrieved_context[i][0]}" for i in range(min(k, len(retrieved_context)))]
    )
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Answer the question based on the given passages below. Keep the answer concise. Use plain text only.
<|eot_id|><|start_header_id|>user<|end_header_id|>
## Passages:
{passages}
## Question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
##Answer:
"""
    return prompt

def get_queries(path):
    df = pd.read_parquet(path)
    return df['question'].tolist()

def get_real_answer_list(path):
    df = pd.read_parquet(path)
    # Flatten the list of answers for each query
    answer_list = df['short_answers'].apply(lambda x: [item for sublist in x for item in sublist]).tolist()
    return answer_list

def is_real_answer_included(real_answers, llm_response):
    """
    Simple string-based check: returns True if any real answer (case-insensitive)
    is a substring of the LLM response.
    """
    if not real_answers or not llm_response:
        return False
    llm_response = llm_response.strip().lower()
    for real_answer in real_answers:
        if real_answer.strip().lower() in llm_response:
            return True
    return False

########################################
# LLM Evaluation via Together API
########################################

system_prompt = (
    "You are a language expert skilled in evaluating whether an LLM response correctly "
    "answers a question. The response does not need to be an exact match but should be "
    "considered correct if it conveys the same core meaning."
)

few_shot_examples = [
    {
        "role": "user",
        "content": (
            "LLM Response: ## Arteries \n- Coronary arteries\n- Great cardiac vein\n- Middle cardiac vein\n"
            "- Small cardiac veins\n- Anterior cardiac veins \n\n## Veins \n- Venae cavae (the two largest veins)\n\n"
            "Real Answers:\n- the arteries\n\nDoes the LLM response correctly answer the question, even if not word-for-word? Answer 'yes' or 'no'."
        ),
    },
    {"role": "assistant", "content": "yes"},
    {
        "role": "user",
        "content": (
            "LLM Response: Rick Kriseman.\n\nReal Answers:\n- Incumbent Rick Kriseman\n\n"
            "Does the LLM response correctly answer the question, even if not word-for-word? Answer 'yes' or 'no'."
        ),
    },
    {"role": "assistant", "content": "yes"},
    {
        "role": "user",
        "content": (
            "LLM Response: It comes from the ribs.\n\nReal Answers:\n- The strip loin steak comes from the short loin.\n\n"
            "Does the LLM response correctly answer the question, even if not word-for-word? Answer 'yes' or 'no'."
        ),
    },
    {"role": "assistant", "content": "no"},
]

def check_correctness(llm_response, real_answers):
    """
    Uses the Together API to ask a meta-model whether the LLM response is correct.
    """
    real_answers_text = "\n".join(f"- {ans}" for ans in real_answers)
    user_query = {
        "role": "user",
        "content": (
            f"LLM Response: {llm_response}\n\nReal Answers:\n{real_answers_text}\n\n"
            "Does the LLM response correctly answer the question, even if not word-for-word? Answer 'yes' or 'no'."
        ),
    }
    messages = [{"role": "system", "content": system_prompt}] + few_shot_examples + [user_query]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer

########################################
# Initialization: Keys, Clients & Data
########################################

# Load Together API key from keys.yaml
with open("keys.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        together_key = config["together_key"]
    except yaml.YAMLError as exc:
        print(exc)
        exit()

client = Together(api_key=together_key)

# Initialize your retrieval service and LLM service.
pinecone_index_name = "dynamic"
pinecone_api_key = "pcsk_5SLo1E_SPpKBieD8wQRuRf9G7xfYrVWDLWUAkfRC3X5xjMpDhy7j8CH3SEen8kJiNbMjav"

data_path = "/home/student/dynamic-rechunking-RAG/data/5000_passage_to_location.json"
retrieval_service = RetrievalService(
    pinecone_index_name=pinecone_index_name, pinecone_api_key=pinecone_api_key
)
llm_service = OllamaInterface()

# Load the document data into the retrieval service (if needed)
data = retrieval_service.load_data(data_path)

# Load queries and corresponding real answers.
queries = get_queries("/home/student/dynamic-rechunking-RAG/data/short_answer_5000_rows.parquet")
real_answers = get_real_answer_list("/home/student/dynamic-rechunking-RAG/data/short_answer_5000_rows.parquet")

########################################
# Main Pipeline: Vanilla RAG
########################################

results = []
correct_includes_count = 0
correct_llm_eval_count = 0
total_queries = 0

# Process (for example) 500 queries
for i, query in enumerate(tqdm(queries, desc="Processing queries")):
    if i >= 500:
        break
    total_queries += 1

    # Retrieve top 3 passages without any expansion or rechunking
    retrieved_context = retrieval_service.search_dense(query, top_k=3)

    # Build the prompt from the retrieved passages
    rag_prompt = create_rag_prompt(query, retrieved_context, k=3)

    # Get the LLM answer
    llm_response = llm_service.send_message(rag_prompt)

    # Evaluate via simple inclusion check
    includes_eval = is_real_answer_included(real_answers[i], llm_response)
    if includes_eval:
        correct_includes_count += 1

    # Evaluate via Together API
    llm_eval_response = check_correctness(llm_response, real_answers[i])
    if llm_eval_response == "yes":
        correct_llm_eval_count += 1

    # Store detailed result for this query
    results.append({
        "query": query,
        "real_answers": real_answers[i],
        "llm_response": llm_response,
        "includes_eval": includes_eval,
        "llm_eval": llm_eval_response
    })

# Compute overall accuracy for both evaluation methods.
includes_accuracy = correct_includes_count / total_queries if total_queries > 0 else 0.0
llm_eval_accuracy = correct_llm_eval_count / total_queries if total_queries > 0 else 0.0

accuracy_data = {
    "includes_accuracy": includes_accuracy,
    "llm_eval_accuracy": llm_eval_accuracy
}

########################################
# Save Results
########################################

# Ensure output directory exists
output_dir = "/home/student/dynamic-rechunking-RAG/scripts/500_outputs/"
os.makedirs(output_dir, exist_ok=True)

# Save the detailed results (each with query, real answer, LLM answer, and both evals)
results_file = os.path.join(output_dir, "vanilla_rag_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

# Save the overall accuracy scores in a separate file
accuracy_file = os.path.join(output_dir, "vanilla_rag_accuracy.json")
with open(accuracy_file, "w") as f:
    json.dump(accuracy_data, f, indent=4)

print(f"Results saved to {results_file}")
print(f"Accuracy data saved to {accuracy_file}")
