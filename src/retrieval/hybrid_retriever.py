from typing import List, Dict
from langchain_core.documents import Document
from src.config import config
from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever

class HybridRetriever:
    def __init__(self, strategy: str = "simple", top_k: int = None):
        self.top_k = top_k or config.retrieval.top_k
        self.dense_k = config.hybrid.dense_k
        self.bm25_k = config.hybrid.bm25_k
        
        self.vector_retriever = VectorRetriever(strategy=strategy, top_k=self.dense_k)
        self.bm25_retriever = BM25Retriever(strategy=strategy, top_k=self.bm25_k)
        
    def _rrf_score(self, rank: int, k: int = 60) -> float:
        """Reciprocal Rank Fusion score."""
        return 1.0 / (k + rank)
        
    def retrieve(self, query: str) -> List[Document]:
        # 1. Retrieve candidates
        dense_docs = self.vector_retriever.retrieve(query)
        bm25_docs = self.bm25_retriever.retrieve(query)
        
        # 2. RRF Fusion
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Helper to process a list of docs
        def process_docs(docs: List[Document]):
            for rank, doc in enumerate(docs, start=1):
                chunk_id = doc.metadata.get("chunk_id")
                if not chunk_id:
                    # Fallback if chunk_id is missing
                    chunk_id = hash(doc.page_content)
                    doc.metadata["chunk_id"] = chunk_id
                    
                doc_map[chunk_id] = doc
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                rrf_scores[chunk_id] += self._rrf_score(rank)
                
        process_docs(dense_docs)
        process_docs(bm25_docs)
        
        # 3. Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # 4. Return top K
        results = []
        for chunk_id in sorted_chunk_ids[:self.top_k]:
            doc = doc_map[chunk_id]
            doc.metadata["rrf_score"] = rrf_scores[chunk_id]
            results.append(doc)
            
        return results
