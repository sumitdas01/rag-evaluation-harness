from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.config import config

def calculate_hit_rate(retrieved_chunk_ids: List[str], expected_chunk_ids: List[str]) -> bool:
    """Returns True if any expected chunk is in the retrieved chunks."""
    if not expected_chunk_ids:
        return False
    return any(expected_id in retrieved_chunk_ids for expected_id in expected_chunk_ids)

def calculate_mrr(retrieved_chunk_ids: List[str], expected_chunk_ids: List[str]) -> float:
    """Calculates Mean Reciprocal Rank."""
    if not expected_chunk_ids:
        return 0.0
    
    for i, retrieved_id in enumerate(retrieved_chunk_ids):
        if retrieved_id in expected_chunk_ids:
            return 1.0 / (i + 1)
    return 0.0

def calculate_recall_at_k(retrieved_chunk_ids: List[str], expected_chunk_ids: List[str]) -> float:
    """Calculates Recall@K: proportion of expected chunks that were retrieved."""
    if not expected_chunk_ids:
        return 0.0
    
    hits = sum(1 for expected_id in expected_chunk_ids if expected_id in retrieved_chunk_ids)
    return hits / len(expected_chunk_ids)

def calculate_precision_at_k(retrieved_chunk_ids: List[str], expected_chunk_ids: List[str]) -> float:
    """Calculates Precision@K: proportion of retrieved chunks that are relevant."""
    if not retrieved_chunk_ids:
        return 0.0
    
    hits = sum(1 for retrieved_id in retrieved_chunk_ids if retrieved_id in expected_chunk_ids)
    return hits / len(retrieved_chunk_ids)

def evaluate_answer_accuracy(question: str, generated_answer: str, expected_answer: str) -> bool:
    """
    Evaluates if the generated answer is semantically equivalent to the expected answer 
    using an LLM as a judge if available, with factual entity overlap fallback.
    """
    import os
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=config.llm.model, temperature=0.0)
            prompt = PromptTemplate.from_template("""
            You are an impartial judge evaluating the correctness of an AI assistant's answer.
            
            Question: {question}
            Expected Answer: {expected_answer}
            Generated Answer: {generated_answer}
            
            Is the Generated Answer factually correct and semantically equivalent to the Expected Answer?
            Consider it correct if it contains the necessary facts, even if phrased differently.
            
            Reply ONLY with YES or NO.
            """)
            chain = prompt | llm
            response = chain.invoke({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer
            })
            result = response.content.strip().upper()
            return result.startswith("YES")
        except Exception as e:
            pass

    # Heuristic / Factual entity evaluation fallback
    if not generated_answer or "No context retrieved" in generated_answer or "[Generation error" in generated_answer:
        return False

    import re
    # Check numbers/figures (e.g., 250, 200, 150, 10,000, 3, 1, 6, 21, 30, 2, 75, etc.)
    expected_nums = set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', expected_answer.lower()))
    gen_nums = set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', generated_answer.lower()))
    if expected_nums and not expected_nums.intersection(gen_nums):
        return False

    # Key entity token overlap check
    stopwords = {"the", "a", "an", "is", "of", "and", "in", "to", "for", "with", "or", "by", "under", "shall", "be", "on", "that", "it", "from", "are", "must", "can"}
    expected_tokens = {w for w in re.findall(r'\b[a-zA-Z]{3,}\b', expected_answer.lower()) if w not in stopwords}
    gen_tokens = {w for w in re.findall(r'\b[a-zA-Z]{3,}\b', generated_answer.lower()) if w not in stopwords}
    
    if not expected_tokens:
        return True
    overlap = len(expected_tokens.intersection(gen_tokens)) / len(expected_tokens)
    return overlap >= 0.45

