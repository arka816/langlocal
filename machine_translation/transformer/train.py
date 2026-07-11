import time
import os
from dataclasses import dataclass

from machine_translation.data_pipeline.dataset import TranslationDataset
from machine_translation.data_pipeline.collator import DynamicTrimmingCollator
from machine_translation.data_pipeline.batch import Batch
from machine_translation.transformer.loss import Loss
from machine_translation.transformer.transformer import make_model

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import torch_xla.core.xla_model as xm

@dataclass
class TrainState:
    """Track number of steps, examples and tokens processed."""

    step: int = 0       # Steps in the current epoch
    accum_step: int = 0 # Number of gradient accumulation steps
    samples: int = 0    # Total number of examples used
    tokens: int = 0     # Total number of tokens processed


def load_checkpoint(model, optimizer, filepath, device):
    if not filepath or not os.path.exists(filepath):
        return 0, 0

    print(f"Loading checkpoint from {filepath}...")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["step"]


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

    print(f"Checkpoint successfully saved to {filepath} at Epoch {epoch}, Step {step}")

def rate(step, model_size=512, factor=1, warmup=4000):
    if step == 0:
        step = 1

    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    accum_iter=1,
    train_state = TrainState()
):
    """Train a single epoch."""

    start = time.time()

    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )

        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        loss_node.backward()

        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens

        if i % accum_iter == 0:         # Gradient accumulation
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1

        scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f | Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0

        del loss
        del loss_node

    return total_loss/total_tokens, train_state


def train_tpu_single_core(
    src_file,
    tgt_file,
    file_dims,
    model_configs,
    bos_id=0,
    eos_id=1,
    pad_id=2,
    batch_size=256,
    epochs=10,
    loader_workers=0,
    checkpoint_filepath=None,
):
    device = xm.xla_device()
    print(f"Using XLA device: {device}")

    translation_dataset = TranslationDataset(src_file, tgt_file, dims=file_dims)
    collator = DynamicTrimmingCollator(bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)

    loader = DataLoader(
        translation_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=loader_workers,
        drop_last=False
    )

    model = make_model(**model_configs).to(device=device)

    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_id)
    loss = Loss(model.generator, criterion)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=1)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size=model_configs["d_model"])
    )

    model.train()

    start_epoch, start_step = load_checkpoint(model, optimizer, checkpoint_filepath, device)
    
    total_loss = 0.0
    total_tokens = 0

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        for step, batch in enumerate(loader):
            if epoch == start_epoch and step < start_step:
                continue

            optimizer.zero_grad()

            # [CPU] Extract ntokens before moving batch to device
            ntokens = batch.ntokens.item()

            # [TPU] Move batch tensors to XLA device
            batch.to_device(device)

            # [TPU] Forward pass (all tensors on device)
            output = model(
                batch.src, 
                batch.tgt, 
                batch.src_mask, 
                batch.tgt_mask
            )

            # [TPU] Loss computation (output and batch.tgt_y are on device)
            loss_ = loss(output, batch.tgt_y, batch.ntokens)

            # [TPU] Backward pass
            loss_.backward()

            # [TPU] Optimizer step on XLA device
            xm.optimizer_step(optimizer)

            # [TPU] Mark step boundary for XLA computation
            xm.mark_step()

            # [CPU] Scheduler step
            lr_scheduler.step()

            # [CPU] Extract scalar loss value from device
            loss_value = loss_.item()

            # [CPU] Accumulate metrics (Python scalars)
            total_loss += loss_value
            total_tokens += ntokens

            # [CPU] Compute metrics for logging
            elapsed_epoch = time.time() - epoch_start_time
            batch_loss = loss_value / ntokens
            cum_loss = total_loss / total_tokens

            # [CPU] Print progress
            print(
                f"Epoch {epoch+1}/{epochs} | Batch {step+1} | Loss: {batch_loss:.4f} | Cum Loss: {cum_loss:.4f} | Time: {elapsed_epoch:.1f}s",
                end="\r",
                flush=True
            )

        print()  # New line after epoch completes
        save_checkpoint(model, optimizer, filepath=checkpoint_filepath, epoch=epoch + 1, step=0)            
