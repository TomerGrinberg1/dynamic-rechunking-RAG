# Dynamic Rechunking RAG

Dynamic Rechunking RAG is a project aimed at enhancing the performance of Retrieval-Augmented Generation (RAG) by employing query-dependent rechunking post-initial retrieval. This pipeline improves document representation by dynamically expanding and rechunking retrieved documents based on cosine similarity at the sentence level, and reranking them for better alignment with the query.

## Project Overview

The pipeline:
1. **Initial Retrieval:** Retrieve relevant chunks using dense retrieval.
2. **Query-Dependent Rechunking:** Expand and rechunk retrieved chunks based on semantic similarity.
3. **Re-ranking:** Rank the rechunked and expanded passages for relevance.
4. **Dynamic Prompt Creation:** Generate RAG prompts dynamically based on the top-k retrieved passages.
5. **Evaluation:** Assess the pipeline’s accuracy by comparing generated responses to ground-truth answers.

## Key Components

### 1. Data
- **Dataset:** Natural Questions dataset.
- **Files:** 
  - `passage_to_location.json`: Stores document data.
  - `first_5000_rows.parquet`: Contains queries and real answers for evaluation.

### 2. Services
- **RetrievalService:** Handles initial dense retrieval and indexing of documents.
- **ExpansionService:** Dynamically expands and rechunks retrieved passages.
- **OllamaInterface:** Interacts with the language model for RAG prompts.

### 3. Pipeline Steps
#### a. Initial Retrieval
Retrieve the top-20 relevant chunks for a query using the dense retrieval model.

#### b. Query-Dependent Rechunking
- Expand passages by including `N` paragraphs before and after the retrieved chunk.
- Rechunk expanded passages by calculating cosine similarity at the sentence level.

#### c. Re-ranking
Re-rank rechunked passages based on query relevance.

#### d. Prompt Creation
Generate a dynamic RAG prompt with the top-k passages for each query.

#### e. Evaluation
Evaluate the generated response against the real answers to compute accuracy.

## Installation and Setup

### Prerequisites
1. Install Python 3.8+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/dynamic-rechunking-RAG.git
   cd dynamic-rechunking-RAG
   ```

### Configure Pinecone
1. Set up a Pinecone index named `dynamic`.
2. Replace `pinecone_api_key` in the script with your Pinecone API key.

### Run the Pipeline
1. Place the data files in the `data/` directory.
2. Execute the pipeline:
   ```bash
   python RAG_orchestrator.py
   ```

## Output

### Results
- **Accuracy:** Accuracy for each `n_neighbors` parameter is stored in `data/hyperparams_accuracy.json`.
- **Detailed Results:** Query-specific results for each `n_neighbors` are saved as `results_n_neighbors_<n_neighbors>.json`.

### Intermediate Files
- **Indexing Progress:** Stored in `data/hyperparams_status.json`.
- **Expanded Data:** Temporary data files created during the expansion process.

## Evaluation Method
The evaluation determines if any real answers are included in the generated response by checking:
- Case-insensitive match of ground-truth answers in the response.

