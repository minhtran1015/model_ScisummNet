import torch
import torch.nn as nn
import torch.optim as optim

def compute_loss(predicted_scores, target_scores):
    """
    Computes the cross-entropy loss between target salience scores and predicted scores.
    :param predicted_scores: Tensor of shape (N,), predicted salience scores
    :param target_scores: Tensor of shape (N,), target salience scores (Rouge Score?)
    :return: Cross-entropy loss
    """
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(predicted_scores, target_scores)
    return loss

def train(model, data_loader, optimizer):
    """
    Training loop for the model.
    :param model: The neural network model
    :param data_loader: DataLoader providing input data
    :param optimizer: Optimizer for updating model parameters
    """
    model.train()
    total_loss = 0
    for batch in data_loader:
        sentences, target_scores = batch  # Extract input sentences and target scores
        optimizer.zero_grad()
        
        predicted_scores = model(sentences)  # Forward pass
        loss = compute_loss(predicted_scores, target_scores)  # Compute loss
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)  # Return average loss
