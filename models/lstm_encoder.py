import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.rnn as rnn_utils

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, glove_path, num_layers=1, bidirectional=True):
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
        
        # Load GloVe embeddings
        self.glove_embeddings = self.load_glove_embeddings(glove_path)
        self.embedding_dim = embedding_dim
        self.device = torch.device("cpu")  # Default device is CPU
    
    def load_glove_embeddings(self, file_path):
        embeddings_index = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = vector
        return embeddings_index
    
    def sentence_to_glove_embedding(self, sentence):
        words = sentence.lower().split()
        vectors = [self.glove_embeddings[word] for word in words if word in self.glove_embeddings]
        if len(vectors) == 0:
            return torch.zeros(self.embedding_dim, device=self.device) 
        return torch.tensor(np.mean(vectors, axis=0), dtype=torch.float32, device=self.device)
    
    def forward(self, sentences):
        """
        Encode sentences using GloVe embeddings and LSTM.
        :param sentences: List of sentences.
        :return: Tensor of encoded sentence representations.
        """
        sentence_embeddings = [self.sentence_to_glove_embedding(sent) for sent in sentences]
        sentence_embeddings = rnn_utils.pad_sequence(sentence_embeddings, batch_first=True)  # Pad sequences
        
        packed_output, (hn, cn) = self.lstm(sentence_embeddings.unsqueeze(1))
        
        sentence_representations = torch.cat((hn[-2], hn[-1]), dim=1) if self.lstm.bidirectional else hn[-1]
        
        return sentence_representations