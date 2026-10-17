import os
from typing import List, Dict, Any
from pathlib import Path
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import config

class VectorRetriever:
    def __init__(self, strategy: str = "simple", top_k: int = None):
        self.strategy = strategy
        self.top_k = top_k or config.retrieval.top_k
        self.collection_name = f"{strategy}_chunks"
        persist_directory = Path(config.paths.chroma_db) / self.collection_name
        
        if not persist_directory.exists():
            raise FileNotFoundError(f"Vector database not found for strategy {strategy} at {persist_directory}")
            
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding.model)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(persist_directory),
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        
    def retrieve(self, query: str) -> List[Document]:
        """Retrieves documents matching the query."""
        return self.retriever.invoke(query)
        
    def get_all_documents(self) -> List[Document]:
        """Utility to get all documents (used for BM25 init)"""
        # Note: Chroma get() might be slow for massive datasets, 
        # but BM25 requires all docs in memory anyway.
        result = self.vectorstore.get(include=["metadatas", "documents"])
        docs = []
        for doc_text, metadata in zip(result['documents'], result['metadatas']):
            docs.append(Document(page_content=doc_text, metadata=metadata))
        return docs
