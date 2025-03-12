import torch
print(torch.backends.mps.is_available())  # Should return True if MPS is available

import torch_geometric
import transformers
import networkx
import spacy

print("PyTorch version:", torch.__version__)
print("Torch Geometric version:", torch_geometric.__version__)
print("Transformers version:", transformers.__version__)
print("NetworkX version:", networkx.__version__)
print("SpaCy version:", spacy.__version__)
