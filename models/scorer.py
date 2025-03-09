import torch
import torch.nn as nn
import torch.nn.functional as F

class SentenceScorer(nn.Module):
    def __init__(self, embedding_dim):
        """
        Scorer model to estimate sentence salience.
        :param embedding_dim: Dimension of sentence embeddings
        """
        super(SentenceScorer, self).__init__()
        self.projection = nn.Linear(embedding_dim, 1, bias=False)  # Learnable parameter v
    
    def forward(self, sentence_embeddings):
        """
        Compute salience scores using softmax normalization.
        :param sentence_embeddings: Tensor of shape (N, D) where N is the number of sentences and D is embedding size
        :return: Normalized salience scores of shape (N,)
        """
        scores = self.projection(sentence_embeddings).squeeze(-1)  # Shape: (N,)
        salience_scores = F.softmax(scores, dim=0)  # Normalize over all sentences
        return salience_scores
