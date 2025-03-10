import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

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
        
        # Transformer-based sentence embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
    def get_sentence_embeddings(self, sentences):
        """Generates sentence embeddings using a transformer model."""
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :]  # (N, 384) instead of (1, 384)

    def forward(self, sentences):
        """
        Encode sentences using LSTM.
        :param sentences: List of sentences.
        :return: Tensor of encoded sentence representations.
        """
        # Generate sentence embeddings (N, 384)
        sentence_embeddings = self.get_sentence_embeddings(sentences)  
        
        # Process each sentence separately
        sentence_embeddings = [emb.unsqueeze(0) for emb in sentence_embeddings]  # Convert to list of tensors
        padded_inputs = pad_sequence(sentence_embeddings, batch_first=True)  # Pad sequences

        # Pass through LSTM
        lstm_output, (hn, _) = self.lstm(padded_inputs)

        # Extract final hidden states
        sentence_representations = torch.cat((hn[-2], hn[-1]), dim=1) if self.lstm.bidirectional else hn[-1]
        
        return sentence_representations

