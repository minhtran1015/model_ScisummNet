import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, H, A_tilde):
        """GCN Layer Forward Pass: H(l+1) = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H W)"""
        D_tilde = torch.diag(torch.sum(A_tilde, dim=1) ** -0.5)  # Compute D^(-1/2)
        A_norm = D_tilde @ A_tilde @ D_tilde  # Normalized adjacency matrix
        H_next = A_norm @ H @ self.weight  # Apply propagation and transformation
        return F.relu(H_next)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_layers=2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers.append(GCNLayer(hidden_dim, out_features))
    
    def forward(self, X, A_tilde):
        H = X
        for layer in self.layers:
            H = layer(H, A_tilde)
        return H

if __name__ == "__main__":
    # Example usage with dummy data
    num_nodes = 5
    in_features = 10
    hidden_dim = 16
    out_features = 8
    
    X = torch.rand((num_nodes, in_features))  # Random feature matrix
    A_tilde = torch.eye(num_nodes)  # Identity matrix as adjacency (for self-connections)
    
    model = GCN(in_features, hidden_dim, out_features)
    output = model(X, A_tilde)
    print("GCN Output:", output)
