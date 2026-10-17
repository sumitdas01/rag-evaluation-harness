import json
import os
from pathlib import Path
from src.config import config
from src.evaluation.evaluator import Evaluator
from src.evaluation.report import generate_report

def main():
    questions_file = Path(config.paths.evaluation_questions)
    results_file = Path(config.paths.results) / "evaluation_results.json"
    
    if not questions_file.exists():
        print(f"Error: Questions file not found at {questions_file}")
        print("Please generate or provide questions first.")
        return

    Path(config.paths.results).mkdir(parents=True, exist_ok=True)
    
    print("Starting Evaluation Harness...")
    evaluator = Evaluator(str(questions_file))
    
    results = evaluator.evaluate()
    
    print(f"\nSaving raw results to {results_file}...")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nGenerating report...")
    generate_report(results)

if __name__ == "__main__":
    main()
