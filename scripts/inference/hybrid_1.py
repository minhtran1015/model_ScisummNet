import torch
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_encoder import LSTMEncoder
from models.gcn import GCN
from models.scorer import SentenceScorer
from models.summary_generator import hybrid_1
from utils.preprocess import construct_sentence_graph
from utils.get_data import get_extracted_data, get_label_data

# Model dimensions
embedding_dim = 100
hidden_dim = 200
gcn_out_dim = 201

import re

def split_into_sentences(text):
    """
    Split a text passage into individual sentences.
    
    Args:
        text (str): Input text passage
        
    Returns:
        list: List of sentences
    """
    # Handle common abbreviations to prevent false sentence breaks
    text = re.sub(r'(\b\w\.)+\s', lambda m: m.group().replace('.', '@'), text)
    
    # Split by sentence-ending punctuation followed by space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restore periods in abbreviations
    sentences = [s.replace('@', '.') for s in sentences]
    
    # Remove empty sentences
    sentences = [s for s in sentences if s.strip()]
    
    return sentences

# Load trained models
def load_models(checkpoint_path='results/best_model.pt'):
    print(f"Loading models from {checkpoint_path}...")
    
    # Initialize models
    lstm_encoder = LSTMEncoder(embedding_dim, hidden_dim, "glove/glove.6B.100d.txt")
    gcn = GCN(hidden_dim * 2, hidden_dim, gcn_out_dim)
    sentence_scorer = SentenceScorer(gcn_out_dim)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    lstm_encoder.load_state_dict(checkpoint['lstm_encoder_state_dict'])
    gcn.load_state_dict(checkpoint['gcn_state_dict'])
    sentence_scorer.load_state_dict(checkpoint['scorer_state_dict'])
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lstm_encoder.device = device
    lstm_encoder.to(device)
    gcn.to(device)
    sentence_scorer.to(device)
    
    # Set to eval mode
    lstm_encoder.eval()
    gcn.eval()
    sentence_scorer.eval()
    
    return lstm_encoder, gcn, sentence_scorer, device


def compute_salience_scores(sentences, lstm_encoder, gcn, sentence_scorer, device):

    adjacency_matrix = construct_sentence_graph(sentences)
    
    adj_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        sentence_embeddings = lstm_encoder(sentences)
        gcn_output = gcn(sentence_embeddings, adj_tensor)
        scores = sentence_scorer(gcn_output)
        # Convert to probabilities
        probs = torch.softmax(scores, dim=0)
    
    return probs.cpu().numpy()

def generate_summary(sentences, lstm_encoder, gcn, sentence_scorer, device, length_limit=250):
    salience_scores = compute_salience_scores(sentences, lstm_encoder, gcn, sentence_scorer, device)
    summary = hybrid_1(sentences, salience_scores, length_limit)
    return summary

def summarize_paper(paper_id=None, sentences=None, length_limit=250):
    lstm_encoder, gcn, sentence_scorer, device = load_models()
    
    if paper_id:
        data = get_extracted_data().get(paper_id)
        if not data:
            print(f"Paper ID {paper_id} not found in dataset.")
            return None
        
        sentences = data["input_sentences"]
        
    if not sentences:
        print("No sentences provided.")
        return None
    
    summary = generate_summary(sentences, lstm_encoder, gcn, sentence_scorer, device, length_limit)
    
    return summary

if __name__ == "__main__":
    paper_id = "1906.00103"
    summary = summarize_paper(paper_id=paper_id)
    
    if summary:
        print("\n=== Generated Summary ===")
        print(" ".join(summary))
        
        gold_summary = " ".join(get_label_data().get(paper_id, []))
        if gold_summary:
            print("\n=== Gold Summary ===")
            print(gold_summary)
    
    test_passage = "A well-structured manuscript allows readers and reviewers to get excited about the subject matter, to understand and verify the paper's contributions, and to integrate these contributions into a broader context. However, many scientists struggle with producing high-quality manuscripts and are typically untrained in paper writing. Focusing on how readers consume information, we present a set of ten simple rules to help you communicate the main idea of your paper. The editors want to make sure that the paper is significant, and the reviewers want to determine whether the conclusions are justified by the results. The reader wants to quickly understand the conceptual conclusions of the paper before deciding whether to dig into the details, and the writer wants to convey the important contributions to the broadest audience possible while convincing the specialist that the findings are credible. As scientists become increasingly specialized, it becomes more important (and difficult) to strengthen the conceptual links. Communication across disciplinary boundaries can only work when manuscripts are readable, credible, and memorable. The claim that gives significance to your work has to be supported by data and by a logic that gives it credibility. Without carefully planning the paper's logic, writers will often be missing data or missing logical steps on the way to the conclusion. While these lapses are beyond our scope, your scientific logic must be crystal clear to powerfully make your claim. Here we present ten simple rules for structuring papers. The first four rules are principles that apply to all the parts of a paper and further to other forms of communication such as grants and posters. The next four rules deal with the primary goals of each of the main parts of papers. The final two rules deliver guidance on the process—heuristics for efficiently constructing manuscripts."

    # test_sentences = [
    #     "This paper explores the impact of new algorithms on machine translation.",
    #     "The proposed approach combines neural networks with statistical methods.",
    #     "Experiments show significant improvements over baseline methods.",
    #     "The model achieves state-of-the-art performance on standard benchmarks.",
    #     "Future work will focus on multilingual capabilities and efficiency."
    # ]

    test_sentences = split_into_sentences(test_passage)
    
    custom_summary = summarize_paper(sentences=test_sentences)
    
    if custom_summary:
        print("\n=== Custom Text Summary ===")
        print(" ".join(custom_summary))