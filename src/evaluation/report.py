import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
from src.config import config

def generate_report(results: List[Dict[str, Any]], results_dir: str = None):
    results_dir = Path(results_dir or config.paths.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Convert to DataFrame
    flattened = []
    for r in results:
        flat = {
            "question_id": r["question_id"],
            "strategy": r["strategy"],
            "answer_correct": r["answer_correct"],
            "hit": r["retrieval_metrics"]["hit"],
            "recall_at_k": r["retrieval_metrics"]["recall_at_k"],
            "precision_at_k": r["retrieval_metrics"]["precision_at_k"],
            "mrr": r["retrieval_metrics"]["mrr"],
            "retrieval_latency": r["latency"]["retrieval"],
            "generation_latency": r["latency"]["generation"]
        }
        flattened.append(flat)
        
    df = pd.DataFrame(flattened)
    
    # 2. Save CSV
    df.to_csv(results_dir / "report.csv", index=False)
    
    # 3. Calculate Summary Metrics
    summary = df.groupby("strategy").agg(
        Retrieval_Accuracy=("hit", "mean"),
        Recall_at_K=("recall_at_k", "mean"),
        Precision_at_K=("precision_at_k", "mean"),
        MRR=("mrr", "mean"),
        Answer_Accuracy=("answer_correct", "mean"),
        Avg_Retrieval_Latency=("retrieval_latency", "mean")
    ).reset_index()
    
    # Format percentages
    for col in ["Retrieval_Accuracy", "Recall_at_K", "Precision_at_K", "MRR", "Answer_Accuracy"]:
        summary[col] = summary[col] * 100
        
    # Find Best
    best_strategy_row = summary.loc[summary['Answer_Accuracy'].idxmax()]
    best_strategy = best_strategy_row['strategy']
    best_accuracy = best_strategy_row['Answer_Accuracy']
    
    # Save Summary JSON
    summary_dict = summary.to_dict(orient="records")
    with open(results_dir / "report.json", "w") as f:
        json.dump(summary_dict, f, indent=4)
        
    # 4. Generate Markdown
    md_path = results_dir / "report.md"
    with open(md_path, "w") as f:
        f.write("# RAG Evaluation Results\n\n")
        f.write(f"**Embedding Model:** {config.embedding.model}\n")
        f.write(f"**LLM Model:** {config.llm.model}\n")
        f.write(f"**Retrieval Top-K:** {config.retrieval.top_k}\n\n")
        
        f.write("## Strategy Comparison\n\n")
        
        # Write markdown table
        f.write("| Strategy | Retrieval Accuracy | Recall@K | MRR | Answer Accuracy | Avg Latency (s) |\n")
        f.write("|----------|--------------------|----------|-----|-----------------|-----------------|\n")
        
        for _, row in summary.iterrows():
            f.write(f"| {row['strategy']} | {row['Retrieval_Accuracy']:.1f}% | {row['Recall_at_K']:.1f}% | {row['MRR']:.1f}% | {row['Answer_Accuracy']:.1f}% | {row['Avg_Retrieval_Latency']:.3f} |\n")
            
        f.write("\n")
        f.write(f"**Best Strategy:** {best_strategy} ({best_accuracy:.1f}% Answer Accuracy)\n")
        
    # 5. Generate Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='strategy', y='Answer_Accuracy', palette='viridis')
    plt.title('Answer Accuracy by Strategy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Retrieval Strategy')
    plt.ylim(0, 100)
    plt.axhline(y=config.evaluation.threshold * 100, color='r', linestyle='--', label=f'Threshold ({config.evaluation.threshold * 100}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison.png")
    
    print(f"Reports and visualizations saved to {results_dir}")

if __name__ == "__main__":
    results_file = Path(config.paths.results) / "evaluation_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        generate_report(results)
    else:
        print("No results found to generate report.")
