import nltk
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
class ExpansionService:
    """
    Handles expansion by neighboring passages and optional query-focused re-chunking.
    """

    def __init__(self, similarity_model_name="intfloat/e5-base-v2"):
        """
        Args:
            dataset_service: Instance of MSMarcoDatasetService
            similarity_model_name: A sentence-transformers model for sentence-level filtering
        """
        nltk.download('punkt', quiet=True)
        self.sim_model = SentenceTransformer(similarity_model_name)

    def expand_passages(self, key: str, n_neighbors: int = 1) -> str:
        """
        Retrieves the passages from a JSON structure using keys formatted as "<row_number>_<passage_number>", 
        then expands around passage_idx by +/- n_neighbors. Joins them into one big text string.
        """
        with open("/home/tomer/learning/dynamic-rechunking-RAG/data/passage_to_location.json", "r") as file:
            all_passages = json.load(file)

        expanded_list = []
        
        # Loop through the range of neighboring passages
        passage_idx=int(key.split("_")[1])
        row_idx=int(key.split("_")[0])
        for idx in range(max(0, passage_idx - n_neighbors), passage_idx + n_neighbors + 1):
            key = f"{row_idx}_{idx}"
            if key in all_passages:
                expanded_list.append(all_passages[key]["passage"])

        # Concatenate the relevant passages
        expanded_text = "\n".join(expanded_list)  # or "\n\n" if you want more separation
        return expanded_text

def query_focused_rechunk(self, query: str, text: str, threshold: float = 0.8, use_knn: bool = False, k: int = 5) -> str:
    """
    Splits the expanded text into sentences, scores each for relevance to the query,
    and either keeps those above a similarity threshold or selects the top-K nearest sentences.
    """
    sentences = nltk.sent_tokenize(text)
    
    # Encode query and sentences
    query_emb = self.sim_model.encode([f"query: {query}"], convert_to_tensor=True)
    sent_embs = self.sim_model.encode(
        [f"passage: {s}" for s in sentences], convert_to_tensor=True
    )
    
    # Compute cosine similarity
    scores = util.cos_sim(query_emb, sent_embs)[0].cpu().numpy()
    
    if use_knn:
        # Select top-K nearest sentences
        top_k_indices = np.argsort(scores)[-k:][::-1]  # Get top-K indices sorted by score
        filtered_sents = [sentences[i] for i in top_k_indices]
    else:
        # Filter sentences based on threshold
        filtered_sents = [s for s, sc in zip(sentences, scores) if sc >= threshold]
    
    return "\n".join(filtered_sents)

    

    def rerank_chunks(self, query: str, passages: list) -> list:
        """
        Reranks a list of passages based on similarity to the query.

        Args:
            query: The query string.
            passages: A list of passage strings to be reranked.

        Returns:
            A list of tuples [(passage, score), ...] sorted by score in descending order.
        """
        # Embed the query
        query_emb = self.sim_model.encode([f"query: {query}"], convert_to_tensor=True)

        # Embed all passages
        passage_embs = self.sim_model.encode(
            [f"passage: {p}" for p in passages],
            convert_to_tensor=True
        )

        # Compute cosine similarities
        scores = util.cos_sim(query_emb, passage_embs)[0].cpu().numpy()

        # Pair passages with their scores and sort by score in descending order
        ranked_passages = sorted(
            zip(passages, scores), key=lambda x: x[1], reverse=True
        )

        return ranked_passages
