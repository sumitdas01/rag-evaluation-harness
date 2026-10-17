from typing import List
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import config
import uuid

class BaseChunker:
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError

    def _assign_chunk_ids(self, chunks: List[Document], prefix: str = "chunk") -> List[Document]:
        for i, chunk in enumerate(chunks):
            doc_id = chunk.metadata.get("document_id", "unknown_doc")
            page = chunk.metadata.get("page", "")
            page_str = f"_page_{page}" if page else ""
            chunk.metadata["chunk_id"] = f"{doc_id}{page_str}_{prefix}_{i}"
        return chunks

class SimpleChunker(BaseChunker):
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunking.simple.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.simple.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        return self._assign_chunk_ids(chunks, prefix="simple")

class SemanticChunkerStrategy(BaseChunker):
    def __init__(self, embeddings_model=None):
        try:
            from langchain_experimental.text_splitter import SemanticChunker
        except ImportError:
            raise ImportError(
                "langchain-experimental is required for SemanticChunker. "
                "Please run: pip install langchain-experimental"
            )

        from langchain_community.embeddings import HuggingFaceEmbeddings
        if not embeddings_model:
            embeddings_model = HuggingFaceEmbeddings(model_name=config.embedding.model)
            
        self.splitter = SemanticChunker(
            embeddings_model, 
            breakpoint_threshold_type=config.chunking.semantic.breakpoint_threshold_type,
            breakpoint_threshold_amount=config.chunking.semantic.breakpoint_threshold_amount
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        # SemanticChunker can be slow, processing doc by doc
        chunks = self.splitter.split_documents(documents)
        return self._assign_chunk_ids(chunks, prefix="semantic")

def get_chunker(strategy: str) -> BaseChunker:
    if strategy == "simple":
        return SimpleChunker()
    elif strategy == "semantic":
        return SemanticChunkerStrategy()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
