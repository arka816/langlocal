import math

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, d_model, d_vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(d_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the Positional Endoding function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10_000.0) / d_model)
        )
        term = torch.einsum("i,j->ij", position, div_term)

        pe[:, 0::2] = torch.sin(term)
        pe[:, 1::2] = torch.cos(term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
