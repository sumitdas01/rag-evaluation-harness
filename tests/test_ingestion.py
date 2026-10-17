import pytest
from src.ingestion.cleaners import clean_text
from langchain_core.documents import Document
from src.ingestion.chunkers import SimpleChunker

def test_clean_text():
    raw_text = "This   is  a\n\n\ntest\x00 text.   "
    expected = "This is a\n\ntest text."
    assert clean_text(raw_text) == expected

def test_simple_chunker():
    chunker = SimpleChunker(chunk_size=50, chunk_overlap=10)
    docs = [
        Document(
            page_content="This is a long document text that needs to be split into multiple chunks.",
            metadata={"document_id": "doc1"}
        )
    ]
    chunks = chunker.chunk_documents(docs)
    
    assert len(chunks) > 1
    assert "chunk_id" in chunks[0].metadata
    assert chunks[0].metadata["chunk_id"].startswith("doc1")
