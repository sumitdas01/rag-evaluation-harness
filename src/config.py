import os
import yaml
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChunkingSimpleConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 100

class ChunkingSemanticConfig(BaseModel):
    buffer_size: int = 1
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 90.0

class ChunkingConfig(BaseModel):
    simple: ChunkingSimpleConfig
    semantic: ChunkingSemanticConfig

class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"

class RetrievalConfig(BaseModel):
    top_k: int = 5

class HybridConfig(BaseModel):
    dense_k: int = 10
    bm25_k: int = 10

class RerankingConfig(BaseModel):
    initial_k: int = 20
    final_k: int = 5
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class EvaluationConfig(BaseModel):
    questions: int = 50
    threshold: float = 0.80

class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

class PathsConfig(BaseModel):
    documents: str = "data/documents"
    evaluation_questions: str = "data/evaluation/questions.json"
    chroma_db: str = "chroma"
    results: str = "results"

class AppConfig(BaseModel):
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    hybrid: HybridConfig
    reranking: RerankingConfig
    evaluation: EvaluationConfig
    llm: LLMConfig
    paths: PathsConfig

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Loads configuration from YAML file and environment variables."""
    base_dir = Path(__file__).parent.parent
    full_path = base_dir / config_path
    
    with open(full_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Override with environment variables if present
    if os.getenv("EMBEDDING_MODEL"):
        config_dict["embedding"]["model"] = os.getenv("EMBEDDING_MODEL")
    if os.getenv("RERANKER_MODEL"):
        config_dict["reranking"]["model"] = os.getenv("RERANKER_MODEL")
    if os.getenv("LLM_MODEL"):
        config_dict["llm"]["model"] = os.getenv("LLM_MODEL")
    if os.getenv("EVAL_THRESHOLD"):
        config_dict["evaluation"]["threshold"] = float(os.getenv("EVAL_THRESHOLD"))

    return AppConfig(**config_dict)

# Global configuration object
config = load_config()
