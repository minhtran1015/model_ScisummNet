import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from models.gcn import GCN
from models.lstm_encoder import LSTMEncoder
from models.scorer import SentenceScorer
from utils.preprocess import extract_sentences_for_summarization
from utils.target_scorer import compute_target_scores
from utils.get_data import (
    get_extracted_data,
    get_citations_data, 
    get_docs_data,
    get_label_data
)

# Hyperparams
learning_rate = 0.003
epochs = 10
batch_size = 32

# Model dimensions
embedding_dim = 100
hidden_dim = 128
gcn_out_dim = 64

# Init models
lstm_encoder = LSTMEncoder(embedding_dim, hidden_dim, "glove/glove.6B.100d.txt")
gcn = GCN(hidden_dim * 2, hidden_dim, gcn_out_dim)
sentence_scorer = SentenceScorer(gcn_out_dim)

# Loss and optim
criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(list(lstm_encoder.parameters())
                       + list(gcn.parameters())
                       + list(sentence_scorer.parameters()), lr=learning_rate)

# MPS backend config
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

lstm_encoder.device = device
lstm_encoder.to(device)
gcn.to(device)
sentence_scorer.to(device)

def _get_gold_summary(paper_id):
    return " ".join(get_label_data().get(paper_id, []))

extract_sentences_for_summarization()

# Training loop
for epoch in range(epochs):
    lstm_encoder.train()
    gcn.train()
    sentence_scorer.train()
    
    total_loss = 0
    batch_count = 0
    
    # Process one sample at a time - no batch accumulation
    for i, (paper_id, data) in enumerate(get_extracted_data().items()):
        sentences = data["input_sentences"]
        if not sentences:
            continue

        optimizer.zero_grad() 
            
        adj_matrix = torch.tensor(data["adjacency_matrix"], dtype=torch.float32).to(device)
        gold_summary = _get_gold_summary(paper_id)
        target_scores = compute_target_scores(sentences, gold_summary).to(device)

        # Forward pass
        sentence_embeddings = lstm_encoder(sentences) 
        gcn_output = gcn(sentence_embeddings, adj_matrix)
        scores = sentence_scorer(gcn_output)
        estimated_scores = torch.log_softmax(scores, dim=0)

        # Normalize target scores
        target_scores = target_scores / (target_scores.sum() + 1e-8)
        loss = criterion(estimated_scores, target_scores)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Skipping sample {paper_id} due to NaN/Inf loss")
            continue

        loss.backward()
        
        # Strong gradient clipping
        clip_grad_norm_(
            list(lstm_encoder.parameters()) + 
            list(gcn.parameters()) + 
            list(sentence_scorer.parameters()), 
            max_norm=0.1  # Reduced even further for more stability
        )
        
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        
        if i % 10 == 0:  # Print every 10 samples
            print(f"Sample {i}, Loss: {loss.item():.4f}")

    # Calculate average loss per sample
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'lstm_encoder_state_dict': lstm_encoder.state_dict(),
        'gcn_state_dict': gcn.state_dict(),
        'scorer_state_dict': sentence_scorer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'results/checkpoint-{epoch}.pt')