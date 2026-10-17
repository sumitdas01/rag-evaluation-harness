import argparse
import os
from pathlib import Path
from tqdm import tqdm
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import config
from src.ingestion.loaders import DocumentLoader
from src.ingestion.chunkers import get_chunker

def build_index(strategy: str, force_rebuild: bool = False):
    """
    Builds the vector index for a given chunking strategy.
    """
    collection_name = f"{strategy}_chunks"
    persist_directory = Path(config.paths.chroma_db) / collection_name

    if persist_directory.exists() and not force_rebuild:
        print(f"Index for strategy '{strategy}' already exists at {persist_directory}.")
        print("Use --force to rebuild.")
        return

    print(f"Loading documents from {config.paths.documents}...")
    loader = DocumentLoader(config.paths.documents)
    documents = loader.load_all()

    if not documents:
        print("No documents found. Please add documents to data/documents/.")
        return

    print(f"Loaded {len(documents)} document pages/files.")
    
    print(f"Chunking documents using '{strategy}' strategy...")
    chunker = get_chunker(strategy)
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Initializing embedding model: {config.embedding.model}...")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding.model)

    print(f"Building ChromaDB index at {persist_directory}...")
    
    # Batch indexing to avoid memory issues with large datasets
    batch_size = 100
    vectorstore = None
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing chunks"):
        batch = chunks[i:i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(persist_directory),
                collection_name=collection_name
            )
        else:
            vectorstore.add_documents(batch)
            
    # Persist the vectorstore if needed (Chroma >= 0.4.x auto-persists in some configs, 
    # but we can explicitly call it just in case if the API allows or rely on auto-persist).
    if hasattr(vectorstore, 'persist'):
        vectorstore.persist()
        
    print(f"Successfully built index for strategy: {strategy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Vector Index for RAG")
    parser.add_argument("--strategy", type=str, choices=["simple", "semantic"], required=True, 
                        help="Chunking strategy to use (simple or semantic)")
    parser.add_argument("--force", action="store_true", help="Force rebuild the index")
    
    args = parser.parse_args()
    build_index(args.strategy, args.force)
