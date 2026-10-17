import json
import sys
from pathlib import Path
from src.config import config

def run_quality_gate():
    report_file = Path(config.paths.results) / "report.json"
    
    if not report_file.exists():
        print(f"Error: Report file not found at {report_file}")
        sys.exit(1)
        
    with open(report_file, 'r') as f:
        summary = json.load(f)
        
    # Find max accuracy
    best_strategy = None
    best_accuracy = 0.0
    
    for row in summary:
        acc = row["Answer_Accuracy"] / 100.0  # normalize back to 0-1
        if acc > best_accuracy:
            best_accuracy = acc
            best_strategy = row["strategy"]
            
    print("RAG Evaluation")
    print("==============")
    for row in summary:
        print(f"{row['strategy'].capitalize()}: {row['Answer_Accuracy']:.1f}%")
        
    print(f"\nBest Strategy: {best_strategy}")
    print(f"Best Accuracy: {best_accuracy * 100:.1f}%")
    print(f"Required Threshold: {config.evaluation.threshold * 100:.1f}%")
    
    if best_accuracy >= config.evaluation.threshold:
        print("\nSTATUS: PASS")
        sys.exit(0)
    else:
        print("\nSTATUS: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    run_quality_gate()
