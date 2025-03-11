from rouge_score.rouge_scorer import RougeScorer
import torch
import numpy as np

def compute_target_scores(sentences, gold_summary):
    """
    Compute target scores R by averaging ROUGE-1 and ROUGE-2 scores
    for each sentence against the gold summary, then normalizing.
    
    :param sentences: List of extracted sentences.
    :param gold_summary: Gold reference summary.
    :return: Torch tensor of rescaled target scores (probability distribution).
    """
    scorer = RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    scores = []
    
    for sentence in sentences:
        rouge1 = scorer.score(sentence, gold_summary)["rouge1"].fmeasure
        rouge2 = scorer.score(sentence, gold_summary)["rouge2"].fmeasure
        avg_score = (rouge1 + rouge2) / 2
        scores.append(avg_score)
    
    scores = np.array(scores)
    
    # Avoid division by zero by adding a small constant if all scores are zero
    if np.sum(scores) == 0:
        scores += 1e-8
        
    normalized_scores = scores / np.sum(scores)  # Rescale to probability distribution
    return torch.tensor(normalized_scores, dtype=torch.float32)