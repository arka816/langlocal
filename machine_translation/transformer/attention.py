import math

import torch
import torch.nn as nn

from machine_translation.transformer.utils import clones


def attention(query, key, value, mask=None, dropout=None):
    "Compute `Scaled Dot-Product Attention`"
    d_k = query.size(-1)
    scores = torch.einsum("...ik,...jk->...ij", query, key)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.einsum("...ij,...jk->...ik", p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    The linear maps for each head i:
    W_Q_i, W_K_i and W_V_i are all (d_model x d_k) where (d_k = d_v / h) where (h = number of heads).

    But d_v is chosen to be (d_v = d_model).
    So, W_Q_i, W_K_i and W_V_i are all (d_v x d_k). Since there are h heads, the head-wise maps are combined so that
    W_Q, W_K, W_V are all (d_v x d_v) i.e. (d_model x d_model).
    """

    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % heads == 0

        self.d_k = d_model // heads
        self.h = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [
            linear(x)
            .view(nbatches, -1, self.h, self.d_k)
            .transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query,
            key,
            value,
            mask,
            dropout=self.dropout,
        )

        x = (
            x
            .transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        del query, key, value

        return self.linears[-1](x)
