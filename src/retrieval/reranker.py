from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.config import config
from .hybrid_retriever import HybridRetriever

class RerankerRetriever:
    def __init__(self, strategy: str = "simple"):
        self.initial_k = config.reranking.initial_k
        self.final_k = config.reranking.final_k
        
        # Use Hybrid Retriever as the base candidate generator for reranking
        self.base_retriever = HybridRetriever(strategy=strategy, top_k=self.initial_k)
        
        # Initialize CrossEncoder
        self.model = CrossEncoder(config.reranking.model)
        
    def retrieve(self, query: str) -> List[Document]:
        # 1. Get initial candidates
        candidates = self.base_retriever.retrieve(query)
        if not candidates:
            return []
            
        # 2. Prepare pairs for cross-encoder (query, document_text)
        pairs = [[query, doc.page_content] for doc in candidates]
        
        # 3. Predict scores
        scores = self.model.predict(pairs)
        
        # 4. Sort and filter
        # Combine docs and scores
        scored_candidates = list(zip(candidates, scores))
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scored_candidates[:self.final_k]:
            # Optional: ensure score is python float, not numpy float32 for JSON serialization later
            doc.metadata["reranker_score"] = float(score)
            results.append(doc)
            
        return results
