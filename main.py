import json
from rouge_score import rouge_scorer

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_rouge_scores(hyp, ref):
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rouge3', 'rouge4'], use_stemmer=True)
    scores = scorer.score(ref, hyp)
    
    return {
        '2-R': scores['rouge2'].recall * 100,
        '2-F': scores['rouge2'].fmeasure * 100,
        '3-F': scores['rouge3'].fmeasure * 100,
        'SU4-F': scores['rouge4'].fmeasure * 100  # Approximating SU4 with ROUGE-4
    }

def evaluate_summaries():
    filtered_summaries = load_json('filtered_summaries.json')
    test_label = load_json('data/raw/test_label.json')
    
    results = []
    
    for paper_id, model_summary in filtered_summaries.items():
        if paper_id in test_label:
            gold_summaries = test_label[paper_id]
            
            for i, gold_summary in enumerate(gold_summaries):
                scores = calculate_rouge_scores(model_summary, gold_summary)
                
                result = {
                    'paper id': paper_id,
                    'gold_summary_ID': f"{paper_id}_{i}",
                }
                result.update(scores)
                results.append(result)
    
    with open('rouge_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed. Results saved to rouge_results.json")

if __name__ == "__main__":
    evaluate_summaries()