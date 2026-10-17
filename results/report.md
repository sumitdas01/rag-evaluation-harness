# RAG Evaluation Results

**Embedding Model:** all-MiniLM-L6-v2
**LLM Model:** llama-3.3-70b-versatile
**Retrieval Top-K:** 5

## Strategy Comparison

| Strategy | Retrieval Accuracy | Recall@K | MRR | Answer Accuracy | Avg Latency (s) |
|----------|--------------------|----------|-----|-----------------|-----------------|
| hybrid | 96.0% | 90.0% | 90.5% | 74.0% | 0.015 |
| reranking | 98.0% | 95.0% | 97.0% | 70.0% | 0.365 |
| semantic | 96.0% | 50.6% | 78.7% | 60.0% | 0.015 |
| simple | 98.0% | 91.0% | 89.1% | 74.0% | 0.018 |

**Best Strategy:** hybrid (74.0% Answer Accuracy)
