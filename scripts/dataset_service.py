from datasets import load_dataset

class MSMarcoDatasetService:
    """
    Manages loading and accessing MS MARCO data from Hugging Face.
    """

    def _init_(self, split="train"):
        """
        Example for the MS MARCO dataset on Hugging Face.
        Adjust 'split' as needed (train, validation, etc.).
        """
        self.dataset = load_dataset("ms_marco", split=split)
        # ^ Adjust if you're using a different config or variant.

    def _len_(self):
        return len(self.dataset)

    def get_item(self, idx: int) -> dict:
        """
        Returns the entire record at index idx with fields like:
          - query
          - answers
          - passages -> { "is_selected": [...], "passage_text": [...], ... }
        """
        return self.dataset[idx]

    def get_passages(self, idx: int) -> list:
        """
        Returns a list of passage texts for the dataset row idx.
        Each row's passages["passage_text"] is a list of strings.
        """
        record = self.dataset[idx]
        return record["passages"]["passage_text"]

    def get_query(self, idx: int) -> str:
        """
        Returns the query string for row idx.
        """
        return self.dataset[idx]["query"]

    def get_answers(self, idx: int) -> list:
        """
        Returns the list of reference answers for row idx.
        """
        return self.dataset[idx]["answers"]