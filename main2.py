import json
import numpy as np

def calculate_average_scores():
    # Load the rouge results
    with open('rouge_results.json', 'r', encoding='utf-8') as f:
        rouge_results = json.load(f)
    
    # Metrics to compute averages for
    metrics = ['2-R', '2-F', '3-F', 'SU4-F']
    
    # Initialize dictionaries to store sums and counts
    sums = {metric: 0.0 for metric in metrics}
    counts = {metric: 0 for metric in metrics}
    
    # Calculate sum and count for each metric
    for result in rouge_results:
        for metric in metrics:
            if metric in result:
                sums[metric] += result[metric]
                counts[metric] += 1
    
    # Calculate averages
    averages = {metric: sums[metric] / counts[metric] if counts[metric] > 0 else 0 
                for metric in metrics}
    
    # Print the results
    print("Average ROUGE Scores:")
    for metric, value in averages.items():
        print(f"{metric}: {value:.4f}")
    
    # Save the results to a file
    with open('average_rouge_scores.json', 'w', encoding='utf-8') as f:
        json.dump({
            "average_scores": averages,
            "total_evaluations": counts
        }, f, indent=2)
    
    print(f"\nResults saved to average_rouge_scores.json")

if __name__ == "__main__":
    calculate_average_scores()