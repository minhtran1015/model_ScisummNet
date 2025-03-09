import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
    
    def forward(self, sentence_embeddings, authority_scores):
        """
        Encode sentences using LSTM and append authority scores.
        :param sentence_embeddings: List of tensor embeddings (variable lengths).
        :param authority_scores: List of authority scores (one per sentence).
        :return: Tensor of encoded sentence representations with authority scores.
        """
        padded_inputs = pad_sequence(sentence_embeddings, batch_first=True)
        packed_output, (hn, cn) = self.lstm(padded_inputs)
        
        # Take the final hidden state
        sentence_representations = torch.cat((hn[-2], hn[-1]), dim=1) if self.lstm.bidirectional else hn[-1]
        
        # Append authority scores as an additional feature
        authority_scores_tensor = torch.tensor(authority_scores, dtype=torch.float32).unsqueeze(1)
        encoded_sentences = torch.cat((sentence_representations, authority_scores_tensor), dim=1)
        
        return encoded_sentences
