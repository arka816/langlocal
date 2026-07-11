from copy import deepcopy

import torch
import torch.nn as nn


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([deepcopy(module) for _ in range(N)])
