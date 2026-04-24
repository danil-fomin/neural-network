import torch.nn as nn
import torch

class RnnClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        h_fwd = h_n[0]                          
        h_bwd = h_n[1]                          
        h = torch.cat([h_fwd, h_bwd], dim=1)       

        return self.linear(h)  
