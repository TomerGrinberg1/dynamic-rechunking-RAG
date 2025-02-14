import os
import json
from glob import glob
from together import Together
from tqdm import tqdm
import yaml

with open("keys.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        together_key = config["together_key"]
    except yaml.YAMLError as exc:
        print(exc)
        exit()
        
client = Together(api_key=together_key)

# Directory path

# System prompt
system_prompt = "You are a language expert skilled in evaluating whether an LLM response correctly answers a question. The response does not need to be an exact match but should be considered correct if it conveys the same core meaning."

# Few-shot examples to guide the model
few_shot_examples = [
    {
        "role": "user",
        "content": "LLM Response: ## Arteries \n- Coronary arteries\n- Great cardiac vein\n- Middle cardiac vein\n- Small cardiac veins\n- Anterior cardiac veins \n\n## Veins \n- Venae cavae (the two largest veins)\n\nReal Answers:\n- the arteries\n\nDoes the LLM response correctly answer the question, even if not word-for-word? Answer 'yes' or 'no'.",
    },
    {"role": "assistant", "content": "yes"},
    {
        "role": "user",
        "content": "LLM Response: Rick Kriseman.\n\nReal Answers:\n- Incumbent Rick Kriseman\n\nDoes the LLM response correctly answer the question, even if not word-for-word? Answer 'yes' or 'no'.",
    },
    {"role": "assistant", "content": "yes"},
    {
        "role": "user",
        "content": "LLM Response: It comes from the ribs.\n\nReal Answers:\n- The strip loin steak comes from the short loin.\n\nDoes the LLM response correctly answer the question, even if not word-for-word? Answer 'yes' or 'no'.",
    },
    {"role": "assistant", "content": "no"},
]

# Function to check if the LLM response is mostly correct
def check_correctness(llm_response, real_answers):
    real_answers_text = "\n".join(f"- {ans}" for ans in real_answers)
    
    user_query = {
        "role": "user",
        "content": f"LLM Response: {llm_response}\n\nReal Answers:\n{real_answers_text}\n\nDoes the LLM response correctly answer the question, even if not word-for-word? Answer 'yes' or 'no'.",
    }

    messages = [{"role": "system", "content": system_prompt}] + few_shot_examples + [user_query]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages
    )

    answer = response.choices[0].message.content.strip().lower()
    return answer

# Process each file and calculate accuracy
accuracy_results = {}


# file_list = [
#     "results_use_knn_True_n_neighbors_0_k_3.json",
#     "results_use_knn_True_n_neighbors_0_k_10.json",
#     "results_use_knn_True_n_neighbors_1_k_5.json",
#     "results_use_knn_True_n_neighbors_2_k_10.json",
#     "results_use_knn_True_n_neighbors_5_k_5.json",
#     "results_use_knn_True_n_neighbors_10_k_3.json"
# ]



 # List of filenames extracted from the image
file_list = [
    "results_use_knn_True_n_neighbors_2_k_5.json",
    "results_use_knn_False_n_neighbors_2.json",
    "results_use_knn_False_n_neighbors_10.json"
]


# Directory path
dir_path = "/home/student/dynamic-rechunking-RAG/scripts/500_outputs/"
output_accuracy_file = os.path.join(dir_path, "llm_accuracy_results.json")

# Load existing accuracy results if the file exists
if os.path.exists(output_accuracy_file):
    with open(output_accuracy_file, "r") as f:
        accuracy_results = json.load(f)
else:
    accuracy_results = {}

for file_name in file_list:
    file_path = os.path.join(dir_path, file_name)
    print(f"Processing: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    
    correct_count = 0
    total = len(data)
    print(f"Total entries: {total}")
    
    eval_results = []  # list to store detailed evaluation results
    
    for entry in tqdm(data):
        llm_eval_response = check_correctness(entry["llm_response"], entry["real_answers"])
        
        # Create a result entry; adjust 'query' if your data uses a different key.
        result_data = {
            "query": entry.get("query", ""),
            "llm_response": entry.get("llm_response", ""),
            "real_answers": entry.get("real_answers", []),
            "llm_eval_response": llm_eval_response
        }
        eval_results.append(result_data)
        
        if llm_eval_response == "yes":
            correct_count += 1

    accuracy = correct_count / total if total > 0 else 0
    # Update the existing accuracy results with the new evaluation for this file
    accuracy_results[file_name] = accuracy

    # Save the updated overall accuracy results to the file
    with open(output_accuracy_file, "w") as f:
        json.dump(accuracy_results, f, indent=4)

    # Save the detailed evaluation results for this file
    eval_file_name = file_name.replace(".json", "_llm_eval.json")
    output_eval_file = os.path.join(dir_path, eval_file_name)
    with open(output_eval_file, "w") as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"Updated accuracy in {output_accuracy_file} and saved eval details to {output_eval_file}")
    
    # Calculate final accuracy from all detailed evaluation files
    final_accuracy_results = {}
    eval_files = glob(os.path.join(dir_path, "*_llm_eval.json"))

    for eval_file in eval_files:
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
        
        correct_count = sum(1 for entry in eval_data if entry["llm_eval_response"] == "yes")
        total = len(eval_data)
        accuracy = correct_count / total if total > 0 else 0
        
        final_accuracy_results[os.path.basename(eval_file)] = accuracy

    # Save the final accuracy results to a new JSON file
    final_accuracy_file = os.path.join(dir_path, "final_llm_accuracy_results.json")
    with open(final_accuracy_file, "w") as f:
        json.dump(final_accuracy_results, f, indent=4)

    print(f"Final accuracy results saved to {final_accuracy_file}")