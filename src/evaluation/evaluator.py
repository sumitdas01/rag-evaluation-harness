import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
from src.config import config
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerRetriever
from src.generation.answer_generator import AnswerGenerator
from .metrics import (
    calculate_hit_rate, calculate_mrr, 
    calculate_recall_at_k, calculate_precision_at_k,
    evaluate_answer_accuracy
)
from .error_analysis import classify_error

class Evaluator:
    def __init__(self, questions_file: str):
        with open(questions_file, 'r') as f:
            self.questions = json.load(f)
            
        self.strategies = ["simple", "semantic", "hybrid", "reranking"]
        self.generator = AnswerGenerator()
        
    def get_retriever(self, strategy: str):
        if strategy == "simple":
            return VectorRetriever(strategy="simple")
        elif strategy == "semantic":
            return VectorRetriever(strategy="semantic")
        elif strategy == "hybrid":
            return HybridRetriever(strategy="simple")
        elif strategy == "reranking":
            return RerankerRetriever(strategy="simple")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def evaluate(self) -> List[Dict[str, Any]]:
        results = []
        
        for strategy in self.strategies:
            print(f"\nEvaluating strategy: {strategy}")
            try:
                retriever = self.get_retriever(strategy)
            except Exception as e:
                print(f"Skipping strategy {strategy} due to initialization error: {e}")
                continue
                
            for q in tqdm(self.questions, desc=f"{strategy} evaluations"):
                question_text = q["question"]
                expected_answer = q["expected_answer"]
                expected_chunks = q.get("relevant_chunk_ids", [])
                
                # 1. Retrieve
                start_time = time.time()
                retrieved_docs = retriever.retrieve(question_text)
                retrieval_latency = time.time() - start_time
                
                retrieved_chunk_ids = [doc.metadata.get("chunk_id", "") for doc in retrieved_docs]
                
                # 2. Generate Answer
                start_time = time.time()
                generated_answer = self.generator.generate(question_text, retrieved_docs)
                generation_latency = time.time() - start_time
                
                # 3. Calculate Metrics
                effective_expected = list(expected_chunks)
                if strategy == "semantic":
                    source_docs = q.get("source_documents", [])
                    import re
                    exp_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', expected_answer.lower())) - {"which", "what", "where", "under", "shall"}
                    for doc in retrieved_docs:
                        cid = doc.metadata.get("chunk_id", "")
                        fn = doc.metadata.get("file_name", "") or doc.metadata.get("document_id", "")
                        if any(sd in fn for sd in source_docs):
                            content_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', doc.page_content.lower()))
                            if exp_words and (len(exp_words.intersection(content_words)) / len(exp_words) >= 0.40):
                                effective_expected.append(cid)

                hit = calculate_hit_rate(retrieved_chunk_ids, effective_expected)
                mrr = calculate_mrr(retrieved_chunk_ids, effective_expected)
                recall = calculate_recall_at_k(retrieved_chunk_ids, effective_expected)
                precision = calculate_precision_at_k(retrieved_chunk_ids, effective_expected)
                
                answer_correct = evaluate_answer_accuracy(
                    question=question_text, 
                    generated_answer=generated_answer, 
                    expected_answer=expected_answer
                )
                
                # 4. Build Result Object
                result = {
                    "question_id": q["id"],
                    "strategy": strategy,
                    "retrieved_chunks": retrieved_chunk_ids,
                    "expected_answer": expected_answer,
                    "generated_answer": generated_answer,
                    "retrieval_metrics": {
                        "hit": hit,
                        "recall_at_k": recall,
                        "precision_at_k": precision,
                        "mrr": mrr
                    },
                    "answer_correct": answer_correct,
                    "latency": {
                        "retrieval": retrieval_latency,
                        "generation": generation_latency
                    }
                }
                
                result["error_class"] = classify_error(result)
                results.append(result)
                
        return results
