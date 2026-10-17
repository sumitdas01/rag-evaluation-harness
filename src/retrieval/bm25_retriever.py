from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.config import config
from .vector_retriever import VectorRetriever

class BM25Retriever:
    def __init__(self, strategy: str = "simple", top_k: int = None):
        self.top_k = top_k or config.retrieval.top_k
        
        # Load all documents from the corresponding vector DB to ensure 
        # BM25 searches the exact same chunk set as the dense retriever
        vector_db = VectorRetriever(strategy=strategy, top_k=1)
        self.documents = vector_db.get_all_documents()
        
        if not self.documents:
            raise ValueError(f"No documents found for BM25 initialization in strategy '{strategy}'")
            
        # Tokenize documents for BM25
        tokenized_corpus = [doc.page_content.lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def retrieve(self, query: str) -> List[Document]:
        tokenized_query = query.lower().split(" ")
        # Get scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:self.top_k]
        
        results = []
        for idx in top_indices:
            # Can also inject score into metadata if needed
            doc = self.documents[idx]
            doc.metadata["bm25_score"] = doc_scores[idx]
            results.append(doc)
            
        return results
