import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.hybrid_retriever import HybridRetriever
from langchain_core.documents import Document

@patch('src.retrieval.hybrid_retriever.VectorRetriever')
@patch('src.retrieval.hybrid_retriever.BM25Retriever')
def test_hybrid_retrieval(mock_bm25_class, mock_vector_class):
    # Mock Vector Retriever
    mock_vector = MagicMock()
    mock_vector.retrieve.return_value = [
        Document(page_content="doc A", metadata={"chunk_id": "c1"}),
        Document(page_content="doc B", metadata={"chunk_id": "c2"})
    ]
    mock_vector_class.return_value = mock_vector
    
    # Mock BM25 Retriever
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = [
        Document(page_content="doc B", metadata={"chunk_id": "c2"}),
        Document(page_content="doc C", metadata={"chunk_id": "c3"})
    ]
    mock_bm25_class.return_value = mock_bm25
    
    retriever = HybridRetriever(top_k=2)
    results = retriever.retrieve("test query")
    
    assert len(results) == 2
    # c2 is in both, so it should rank highest
    assert results[0].metadata["chunk_id"] == "c2"
