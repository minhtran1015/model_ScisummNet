import json
import torch
import numpy as np
from models.lstm_encoder import LSTMEncoder
from models.gcn import GCN
from models.scorer import SentenceScorer

import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Rest of your code remains the same

# Load extracted sentences
with open("data/processed/extracted_sentences.json", "r") as f:
    extracted_data = json.load(f)

# Model parameters
embedding_dim = 384  # Transformer sentence embedding dimension
hidden_dim = 128
gcn_out_dim = 64

# Initialize models
lstm_encoder = LSTMEncoder(embedding_dim, hidden_dim)
gcn = GCN(hidden_dim * 2, hidden_dim, gcn_out_dim)
sentence_scorer = SentenceScorer(gcn_out_dim)

for paper_id, data in extracted_data.items():
    sentences = data["input_sentences"]
    
    if not sentences:
        print(f"Skipping {paper_id}: No sentences found")
        continue

    adjacency_matrix = torch.tensor(data["adjacency_matrix"], dtype=torch.float32)
    
    # Print sizes to debug mismatches
    print(f"Paper ID: {paper_id}")
    print(f"Number of sentences: {len(sentences)}")
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")

    # Encode sentences using LSTM
    with torch.no_grad():
        sentence_embeddings = lstm_encoder(sentences)  # Shape: (N, hidden_dim * 2)

    print(f"Sentence embeddings shape: {sentence_embeddings.shape}")  # Debugging
    
    # Check for mismatches
    if sentence_embeddings.shape[0] != adjacency_matrix.shape[0]:
        print(f"🚨 Mismatch in {paper_id}: {sentence_embeddings.shape[0]} embeddings vs {adjacency_matrix.shape[0]} graph nodes")
        continue  # Skip this entry

    # Pass through GCN
    gcn_output = gcn(sentence_embeddings, adjacency_matrix)
    
    print(f"✅ Processed {paper_id}: GCN Output Shape ->", gcn_output.shape)
    print(f"GCN output std: ", gcn_output.std(dim=0))
    # Estimate salience scores
    salience_scores = sentence_scorer(gcn_output)
    print(f"Salience scores for {paper_id}: {salience_scores}")
    print(f"sentence scorer projection weight: ", sentence_scorer.projection.weight)
