from copy import deepcopy
import os

import torch
import torch.nn as nn


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.tril(torch.ones(attn_shape, dtype=torch.bool))
    return subsequent_mask


def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def load_checkpoint(filepath):
    if not filepath or not os.path.exists(filepath):
        return None

    print(f"Loading checkpoint from {filepath}...", flush=True)

    checkpoint = torch.load(filepath, map_location="cpu")

    return checkpoint


def save_checkpoint(model, optimizer, filepath, epoch, step):
    if not filepath:
        return

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, filepath)

    print(f"Checkpoint successfully saved to {filepath} at Epoch {epoch}, Step {step}", end="\n", flush=True)
