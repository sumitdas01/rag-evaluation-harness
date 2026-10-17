from typing import Dict, Any

def classify_error(result: Dict[str, Any]) -> str:
    """
    Classifies the failure reason based on the evaluation result.
    """
    if result["answer_correct"]:
        return "none"
        
    metrics = result["retrieval_metrics"]
    
    # Did we fail to retrieve the right chunks?
    if not metrics["hit"]:
        return "missing_retrieval"
        
    # We retrieved the right chunks, but the answer was wrong.
    # This implies a generation failure or the chunk was relevant but didn't contain 
    # the exact expected answer, or the context was too confusing.
    
    return "generation_failure"
