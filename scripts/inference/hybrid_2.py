import torch
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_encoder import LSTMEncoder
from models.gcn import GCN
from models.scorer import SentenceScorer
from models.summary_generator import hybrid_2
from utils.preprocess import construct_sentence_graph
from utils.get_data import get_extracted_data, get_label_data, get_docs_data, get_citations_data

# Model dimensions
embedding_dim = 100
hidden_dim = 200
gcn_out_dim = 201

def load_models(checkpoint_path='results/best_model.pt'):
    print(f"Loading models from {checkpoint_path}...")
    
    lstm_encoder = LSTMEncoder(embedding_dim, hidden_dim, "glove/glove.6B.100d.txt")
    gcn = GCN(hidden_dim * 2, hidden_dim, gcn_out_dim)
    sentence_scorer = SentenceScorer(gcn_out_dim)
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    lstm_encoder.load_state_dict(checkpoint['lstm_encoder_state_dict'])
    gcn.load_state_dict(checkpoint['gcn_state_dict'])
    sentence_scorer.load_state_dict(checkpoint['scorer_state_dict'])
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lstm_encoder.device = device
    lstm_encoder.to(device)
    gcn.to(device)
    sentence_scorer.to(device)
    
    lstm_encoder.eval()
    gcn.eval()
    sentence_scorer.eval()
    
    return lstm_encoder, gcn, sentence_scorer, device

def separate_abstract_and_cited_text(paper_id):
    docs_data = get_docs_data()
    citations_data = get_citations_data()
    
    if paper_id not in docs_data:
        return None, None
    
    sections = docs_data[paper_id].get("sections", {})
    abstract_sentences = [s["text"] for s in sections.get("Abstract", {}).get("sentences", [])]
    
    extracted_data = get_extracted_data()
    combined_sentences = extracted_data.get(paper_id, {}).get("input_sentences", [])
    
    cited_text_spans = [sent for sent in combined_sentences if sent not in abstract_sentences]
    
    return abstract_sentences, cited_text_spans

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

def generate_summary(abstract, cited_text_spans, lstm_encoder, gcn, sentence_scorer, device, length_limit=250):
    if not cited_text_spans:
        return abstract
    
    salience_scores = compute_salience_scores(cited_text_spans, lstm_encoder, gcn, sentence_scorer, device)
    summary = hybrid_2(abstract, cited_text_spans, salience_scores, length_limit)
    return summary

def summarize_paper(paper_id=None, abstract=None, cited_text_spans=None, length_limit=150):
    lstm_encoder, gcn, sentence_scorer, device = load_models()
    
    if paper_id:
        abstract, cited_text_spans = separate_abstract_and_cited_text(paper_id)
        if not abstract:
            print(f"Paper ID {paper_id} not found in dataset.")
            return None
    
    if not abstract:
        print("No abstract provided.")
        return None
    
    if not cited_text_spans:
        print("No cited text spans found. Returning original abstract.")
        return abstract
    
    summary = generate_summary(abstract, cited_text_spans, lstm_encoder, gcn, sentence_scorer, device, length_limit)
    
    return summary

if __name__ == "__main__":
    paper_id = "1906.00103"
    summary = summarize_paper(paper_id=paper_id)
    
    if summary:
        print("\n=== Generated Hybrid 2 Summary ===")
        print(" ".join(summary))
        
        gold_summary = " ".join(get_label_data().get(paper_id, []))
        if gold_summary:
            print("\n=== Gold Summary ===")
            print(gold_summary)
    
    custom_abstract = [
        "This paper introduces a novel approach to machine translation.",
        "We propose a hybrid method combining statistical and neural techniques.",
        "Our results demonstrate significant improvements over existing methods."
    ]
    
    custom_cited_spans = [
        "The model achieves state-of-the-art performance on standard benchmarks.",
        "Experiments show a 5.2 BLEU score improvement over the baseline.",
        "The approach is particularly effective for low-resource languages.",
        "Future work will focus on multilingual capabilities.",
        "Our error analysis reveals that semantic ambiguity remains challenging."
    ]
    
    custom_summary = summarize_paper(abstract=custom_abstract, cited_text_spans=custom_cited_spans)
    
    if custom_summary:
        print("\n=== Custom Hybrid 2 Summary ===")
        print(" ".join(custom_summary))