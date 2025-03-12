import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from torch.nn.utils import clip_grad_norm_
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

os.makedirs("results", exist_ok=True)

# Hyperparams
learning_rate = 0.001
epochs = 50
batch_size = 5
patience = 5
val_ratio = 0.1
dropout_rate = 0.5

# Model dimensions
embedding_dim = 100
hidden_dim = 200
gcn_out_dim = 201

# Init models
lstm_encoder = LSTMEncoder(embedding_dim, hidden_dim, "glove/glove.6B.100d.txt")
gcn = GCN(hidden_dim * 2, hidden_dim, gcn_out_dim, num_layers=2)  # Explicitly set 2 layers
sentence_scorer = SentenceScorer(gcn_out_dim)

# Loss and optim
criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(
    list(lstm_encoder.parameters())
    + list(gcn.parameters())
    + list(sentence_scorer.parameters()),
    lr=learning_rate
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# MPS backend config
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

lstm_encoder.device = device
lstm_encoder.to(device)
gcn.to(device)
sentence_scorer.to(device)

def _get_gold_summary(paper_id):
    return " ".join(get_label_data().get(paper_id, []))

# Prepare data
extract_sentences_for_summarization()

# Split data into training and validation
all_data = list(get_extracted_data().items())
random.shuffle(all_data)
val_size = int(val_ratio * len(all_data))
val_data = all_data[:val_size]
train_data = all_data[val_size:]
print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")

def validate():
    """Run validation and return average loss"""
    lstm_encoder.eval()
    gcn.eval()
    sentence_scorer.eval()
    
    val_loss = 0
    val_count = 0
    
    with torch.no_grad():
        for paper_id, data in val_data:
            sentences = data["input_sentences"]
            if not sentences:
                continue
                
            adj_matrix = torch.tensor(data["adjacency_matrix"], dtype=torch.float32).to(device)
            gold_summary = _get_gold_summary(paper_id)
            target_scores = compute_target_scores(sentences, gold_summary).to(device)
            
            # Forward pass - with dropout disabled in eval mode
            sentence_embeddings = lstm_encoder(sentences)
            gcn_output = gcn(sentence_embeddings, adj_matrix)
            scores = sentence_scorer(gcn_output)
            estimated_scores = torch.log_softmax(scores, dim=0)
            
            # Normalize target scores
            target_scores = target_scores / (target_scores.sum() + 1e-8)
            loss = criterion(estimated_scores, target_scores)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                val_loss += loss.item()
                val_count += 1
    
    return val_loss / val_count if val_count > 0 else float('inf')

best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(epochs):
    lstm_encoder.train()
    gcn.train()
    sentence_scorer.train()
    
    total_loss = 0
    batch_count = 0
    
    random.shuffle(train_data)
    
    for i, (paper_id, data) in enumerate(train_data):
        sentences = data["input_sentences"]
        if not sentences:
            continue

        optimizer.zero_grad() 
            
        adj_matrix = torch.tensor(data["adjacency_matrix"], dtype=torch.float32).to(device)
        gold_summary = _get_gold_summary(paper_id)
        target_scores = compute_target_scores(sentences, gold_summary).to(device)

        sentence_embeddings = lstm_encoder(sentences) 
        sentence_embeddings = torch.nn.functional.dropout(sentence_embeddings, p=dropout_rate, training=True)
        gcn_output = gcn(sentence_embeddings, adj_matrix)
        gcn_output = torch.nn.functional.dropout(gcn_output, p=dropout_rate, training=True)
        scores = sentence_scorer(gcn_output)
        estimated_scores = torch.log_softmax(scores, dim=0)

        # Normalize target scores
        target_scores = target_scores / (target_scores.sum() + 1e-8)
        loss = criterion(estimated_scores, target_scores)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Skipping sample {paper_id} due to NaN/Inf loss")
            continue

        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(
            list(lstm_encoder.parameters()) + 
            list(gcn.parameters()) + 
            list(sentence_scorer.parameters()), 
            max_norm=2.0
        )
        
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        
        if i % 10 == 0:
            print(f"Sample {i}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1}/{epochs}, Avg Train Loss: {avg_train_loss:.4f}")
    
    val_loss = validate()
    print(f"Validation Loss: {val_loss:.4f}")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print(f"New best validation loss: {best_val_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'lstm_encoder_state_dict': lstm_encoder.state_dict(),
            'gcn_state_dict': gcn.state_dict(),
            'scorer_state_dict': sentence_scorer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'train_loss': avg_train_loss,
        }, 'results/best_model.pt')
    else:
        patience_counter += 1
        print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'lstm_encoder_state_dict': lstm_encoder.state_dict(),
        'gcn_state_dict': gcn.state_dict(),
        'scorer_state_dict': sentence_scorer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'train_loss': avg_train_loss,
    }, f'results/checkpoint-{epoch}.pt')
    
    print("-" * 50)