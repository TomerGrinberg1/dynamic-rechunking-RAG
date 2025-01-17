import os
import json
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, MultifieldParser,OrGroup
from whoosh import analysis, scoring
from keybert import KeyBERT
import uuid

def truncate_text_to_bytes(text, max_bytes=40960):
    """Truncate text to fit within a specified byte limit."""
    # Start with an initial truncation within a reasonable character limit
    truncated_text = text[:15000]
    # Iteratively reduce size if byte count exceeds the limit
    while len(truncated_text.encode('utf-8')) > max_bytes:
        truncated_text = truncated_text[:-100]  # Remove more characters
    return truncated_text


class RetrievalService:
    def __init__(self, index_dir, pinecone_index_name, pinecone_api_key, pinecone_environment):
        """
        Initialize the RetrievalService.

        :param index_dir: Directory path to store Whoosh index for sparse retrieval.
        :param pinecone_index_name: Name of the Pinecone index for dense retrieval.
        :param pinecone_api_key: Your Pinecone API key.
        :param pinecone_environment: Pinecone environment region.
        """
        self.index_dir = index_dir
        self.pinecone_index_name = pinecone_index_name

        # Initialize sentence-transformers model for dense embeddings
        self.model = SentenceTransformer('intfloat/e5-large')
        self.embedding_dim = 1024
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
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
            schema = Schema(doc_id=ID(stored=True, unique=True), content=TEXT(stored=True, analyzer=analysis.StemmingAnalyzer()))
            self.index_sparse = create_in(index_dir, schema)
        elif not exists_in(index_dir):
            schema = Schema(doc_id=ID(stored=True, unique=True), content=TEXT(stored=True, analyzer=analysis.StemmingAnalyzer()))
            self.index_sparse = create_in(index_dir, schema)
        else:
            self.index_sparse = open_dir(index_dir)



    def index_documents(self, chunked_list, sparse=True, dense=True, batch_size=64):
        # Read documents
        documents = []
        doc_ids = []
        metadata = []
        texts_to_embed = []

        for data in chunked_list:
            title = truncate_text_to_bytes(data.get("title", ""))
            content = truncate_text_to_bytes(data.get("content", ""))
            document = f"title: {title}\n content: {content}"
            doc_id = str(uuid.uuid4())
            documents.append((doc_id, document))
            doc_ids.append(doc_id)
            metadata.append({"title": title, "content": content})
            texts_to_embed.append(document)

        if sparse:
            print("Indexing documents for sparse retrieval using Whoosh...")
            self.index_sparse_documents(documents)

        if dense:
            print("Indexing documents for dense retrieval using Pinecone...")
            self.index_dense_documents(doc_ids, texts_to_embed, metadata, batch_size=batch_size)



    def index_sparse_documents(self, documents):
        """
        Index documents into a sparse (BM25) Whoosh index.

        :param documents: List of (doc_id, content) tuples.
        """
        writer = self.index_sparse.writer()
        for doc_id, content in documents:
            writer.add_document(doc_id=doc_id, content=content)
            print(f"Indexed Document: DocID={doc_id}, Content (excerpt)={content[:100]}...")  # Debug statement
        writer.commit()
        print("Sparse indexing completed.")
    def index_dense_documents(self, doc_ids, texts_to_embed, metadata, batch_size=64):
        """
        Index documents into Pinecone using provided doc_ids, texts, and metadata.

        :param doc_ids: List of document IDs.
        :param texts_to_embed: List of texts to embed.
        :param metadata: List of metadata dictionaries.
        :param batch_size: Batch size for processing documents.
        """
        # Encode texts using sentence-transformers
        embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True, batch_size=batch_size)

        # Prepare list of (id, vector, metadata) dictionaries for upsert
        to_upsert = []
        for doc_id, embedding, meta in zip(doc_ids, embeddings, metadata):
            to_upsert.append({
                "id": doc_id,
                "values": embedding.tolist(),
                "metadata": meta
            })

        # Upsert in batches using tqdm for progress tracking
        print("Upserting the embeddings to the Pinecone index...")
        for i in tqdm(range(0, len(to_upsert), batch_size)):
            i_end = min(i + batch_size, len(to_upsert))
            batch = to_upsert[i:i_end]
            self.pinecone_index.upsert(vectors=batch)
        print("Dense indexing completed.")

        

    def extract_keyphrases(self,kw_model,query, num_keywords=5):
        """Extract significant words or keyphrases from the query."""
        keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
        # Extract only the words from the keywords
        return [word for word, score in keywords]

    def search_sparse(self, query, top_k=3):
        try:
            kw_model = KeyBERT()
            keyphrases = self.extract_keyphrases(kw_model,query)
            keyphrase_query = " OR ".join(keyphrases)
                    
            with self.index_sparse.searcher(weighting=scoring.BM25F()) as searcher:
                parser = QueryParser("content", self.index_sparse.schema, group=OrGroup)
                parsed_query = parser.parse(keyphrase_query)
                print(f"Parsed Query: {parsed_query}")
                results = searcher.search(parsed_query, limit=top_k)
                print(f"Number of results found: {len(results)}")
                hits = [(hit['doc_id'], hit.score) for hit in results]
            return hits
        except Exception as e:
            print(f"An error occurred during sparse search: {e}")
            return []







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

    def hybrid_search(self, query, top_k=3, alpha=0.5):
        """
        Perform a hybrid search combining sparse and dense retrieval.

        :param query: The search query as a string.
        :param top_k: Number of top results to retrieve.
        :param alpha: Weighting factor for dense scores (between 0 and 1).
        :return: List of (doc_id, combined_score) tuples.
        """
        initial_k = 12
        sparse_hits = self.search_sparse(query, top_k=initial_k)
        dense_hits = self.search_dense(query, top_k=initial_k)

        # Normalize scores
        max_sparse_score = max([score for _, score in sparse_hits], default=1)
        max_dense_score = max([score for _, score in dense_hits], default=1)
        sparse_scores = {doc_id: score / max_sparse_score for doc_id, score in sparse_hits}
        dense_scores = {doc_id: score / max_dense_score for doc_id, score in dense_hits}

        # Combine results
        all_doc_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        combined_scores = {}
        for doc_id in all_doc_ids:
            sparse_score = sparse_scores.get(doc_id, 0)
            dense_score = dense_scores.get(doc_id, 0)
            combined_score = (1 - alpha) * sparse_score + alpha * dense_score
            combined_scores[doc_id] = combined_score

        # Sort and return top_k results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def retrieve_document_by_id(self, doc_id):
        """
        Retrieve the actual document text given a document ID.

        :param doc_id: The ID of the document to retrieve.
        :return: The document content.
        """
        # First, try to retrieve from Whoosh index
        with self.index_sparse.searcher() as searcher:
            doc = searcher.document(doc_id=doc_id)
            if doc:
                return doc['content']
        
        # If not found in Whoosh, try to retrieve from Pinecone metadata
        try:
            response = self.pinecone_index.fetch(ids=[doc_id])
            vectors = response.get('vectors', {})
            if doc_id in vectors:
                metadata = vectors[doc_id].get('metadata', {})
                # Reconstruct the document content from metadata
                title = metadata.get('title', '')
                content = metadata.get('content', '')
                result = f"title: {title}\n content: {content}"
                return result
        except Exception as e:
            print(f"Error retrieving document by ID from Pinecone: {e}")
        
        return None
    def retrieve_relevant_documents(self, query, search_method='hybrid'):
        if search_method=='sparse':
            hits=self.search_sparse(query)
        elif search_method=='dense':
            hits=self.search_dense(query)
        elif search_method=='hybrid':
            hits=self.hybrid_search(query)
        relevant_documents=[]
        for doc_id, _ in hits:
            document=self.retrieve_document_by_id(doc_id)
            relevant_documents.append(document)
        return relevant_documents
