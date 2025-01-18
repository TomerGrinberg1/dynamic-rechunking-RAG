import json
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from sentence_transformers import SentenceTransformer




class RetrievalService:
    def __init__(self,  pinecone_index_name, pinecone_api_key):
        """
        Initialize the RetrievalService.

        :param index_dir: Directory path to store Whoosh index for sparse retrieval.
        :param pinecone_index_name: Name of the Pinecone index for dense retrieval.
        :param pinecone_api_key: Your Pinecone API key.
        :param pinecone_environment: Pinecone environment region.
        """
        self.pinecone_index_name = pinecone_index_name

        # Initialize sentence-transformers model for dense embeddings
        self.model = SentenceTransformer('intfloat/e5-base-v2')
        self.embedding_dim = 768
        # Initialize Pinecone
        self.pc=Pinecone(api_key=pinecone_api_key)

        # Create Pinecone index if it doesn't exist
        print("Creating a Pinecone index...")
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if self.pinecone_index_name not in existing_indexes:
             self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.pinecone_index=self.pc.Index(self.pinecone_index_name)

        # Initialize or create Whoosh index
       
    def load_data(self,path):
        with open(path, "r") as file:
            data = json.load(file)
        return data

    def index_and_upsert_documents(self,data,to_index_flag=False, to_upsert_flag=False, batch_size=64):
        if to_index_flag:
            to_upsert=[]
            print("create embeddings...")
            for key, value in tqdm(data.items()):
                passage = value.get("passage", "")
                document_row = value.get("document_row", None)
                passage_index = value.get("passage_index", None) 
                if passage:
                    embedding = self.model.encode(passage)
                else:
                    embedding = None
                    continue        # Check if the "passages" structure is valid

                    # Prepare upsert data for the current row
                to_upsert.append({
                    "id": str(key),
                    "values": embedding.tolist(),
                    "metadata": {"row_number":document_row, "location": passage_index, "passage":passage}       
                })
        if to_upsert_flag:
            # # Upsert in batches using tqdm for progress tracking
            print("Upserting the embeddings to the Pinecone index...")
            for i in tqdm(range(0, len(to_upsert), batch_size)):
                i_end = min(i + batch_size, len(to_upsert))
                batch = to_upsert[i:i_end]
                self.pinecone_index.upsert(vectors=batch)
            print("Dense indexing completed.")

                    
                # Upsert data to Pinecone

    def search_dense(self, query, top_k=3):
        """
        Perform a dense search using Pinecone.

        :param query: The search query as a string.
        :param top_k: Number of top results to retrieve.
        :return: List of (doc_id, score, metadata) tuples.
        """
        # Generate the embedding for the query question
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # query_embedding = embed_text(query)

        # Perform the query with Pinecone
        try:
            response = self.pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            # Extract the matches with metadata
            matches = response['matches']
            hits = [(match['id'], match['score']) for match in matches]
        except Exception as e:
            print(f"Error querying index: {e}")
            hits = []

        return hits

   
    