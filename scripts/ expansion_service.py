# expansion_service.py

import nltk
from sentence_transformers import SentenceTransformer, util

class ExpansionService:
    """
    Handles expansion by neighboring passages and optional query-focused re-chunking.
    """

    def _init_(self, dataset_service, similarity_model_name="intfloat/e5-base"):
        """
        Args:
            dataset_service: Instance of MSMarcoDatasetService
            similarity_model_name: A sentence-transformers model for sentence-level filtering
        """
        self.dataset_service = dataset_service
        nltk.download('punkt', quiet=True)
        self.sim_model = SentenceTransformer(similarity_model_name)

    def expand_passages(self, row_idx: int, passage_idx: int, n_neighbors: int = 1) -> str:
        """
        Retrieves the row's list of passages, then expands around passage_idx
        by +/- n_neighbors. Joins them into one big text string.
        """
        all_passages = self.dataset_service.get_passages(row_idx)

        start = max(0, passage_idx - n_neighbors)
        end = min(len(all_passages), passage_idx + n_neighbors + 1)

        # Concatenate the relevant passages
        expanded_list = all_passages[start:end]
        expanded_text = "\n".join(expanded_list)  # or "\n\n" if you want more separation
        return expanded_text

    def query_focused_rechunk(self, query: str, text: str, threshold: float = 0.4) -> str:
        """
        Splits the expanded text into sentences, scores each for relevance to the query,
        and keeps only those above a similarity threshold.
        """
        sentences = nltk.sent_tokenize(text)
        
        # If using E5, might prefix the query with "query: " and each passage with "passage: "
        query_emb = self.sim_model.encode([f"query: {query}"], convert_to_tensor=True)
        sent_embs = self.sim_model.encode(
            [f"passage: {s}" for s in sentences],
            convert_to_tensor=True
        )

        # Compute cosine similarity
        scores = util.cos_sim(query_emb, sent_embs)[0].cpu().numpy()

        filtered_sents = [
            s for s, sc in zip(sentences, scores) if sc >= threshold
        ]
        return " ".join(filtered_sents)