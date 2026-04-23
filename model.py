import torch.nn as nn


class RnnClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)        
        _, hidden = self.rnn(x)      
        hidden = hidden.squeeze(0)
           
        return self.linear(hidden)   
