import time
import os
from dataclasses import dataclass

from machine_translation.data_pipeline.dataset import TranslationDataset, TranslationDatasetFast
from machine_translation.data_pipeline.collator import Collator, DynamicTrimmingCollator
from machine_translation.data_pipeline.batch import Batch
from machine_translation.transformer.loss import Loss
from machine_translation.transformer.transformer import make_model

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from libtpu.sdk import tpumonitoring


@dataclass
class TrainState:
    """Track number of steps, examples and tokens processed."""

    step: int = 0       # Steps in the current epoch
    accum_step: int = 0 # Number of gradient accumulation steps
    samples: int = 0    # Total number of examples used
    tokens: int = 0     # Total number of tokens processed


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
    loader_prefetch_factor=4,
    checkpoint_filepath=None,
    dynamic_padding=False,
    save_every=1000
):
    device = xm.xla_device()
    print(f"Using XLA device: {device}", end="\n", flush=True)

    translation_dataset = TranslationDataset(src_file, tgt_file, dims=file_dims)
    if dynamic_padding:
        collator = DynamicTrimmingCollator(bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
    else:
        collator = Collator(pad_id=pad_id)

    loader = DataLoader(
        translation_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=loader_workers,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=loader_prefetch_factor,
    )

    tpu_loader = pl.MpDeviceLoader(loader, device)

    model = make_model(**model_configs)

    # load check point here
    checkpoint = load_checkpoint(checkpoint_filepath)

    # load model state send model to device
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device)

    # load optimizer state after model has been moved to device
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=1)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_id)
    loss = Loss(model.generator, criterion)

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size=model_configs["d_model"])
    )

    model.train()

    
    total_loss = 0.0
    total_tokens = 0
    if checkpoint:
        start_epoch, start_step = checkpoint["epoch"], checkpoint["step"]
    else:
        start_epoch, start_step = 0, 0

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        for step, batch in enumerate(tpu_loader):
            if epoch == start_epoch and step < start_step:
                continue

            optimizer.zero_grad(set_to_none=True)

            # [TPU] Move batch tensors to XLA device
            batch.to_device(device)

            with torch.autocast(device_type="xla", dtype=torch.bfloat16):
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

            # [CPU] Extract ntokens before moving batch to device
            ntokens = batch.ntokens.item()

            # [CPU] Accumulate metrics (Python scalars)
            total_loss += loss_value
            total_tokens += ntokens

            # [CPU] Compute metrics for logging
            elapsed_epoch = time.time() - epoch_start_time
            batch_loss = loss_value / ntokens
            cum_loss = total_loss / total_tokens

            # [CPU] Save checkpoint after every few batches
            if save_every is not None and step != 0 and step % save_every == 0:
                metric = tpumonitoring.get_metric("duty_cycle_pct")
                print("\nTPU Core Utilization (%):", metric.data(), end="\n", flush=True)

                save_checkpoint(model, optimizer, filepath=checkpoint_filepath, epoch=epoch, step=step + 1)

            # [CPU] Print progress
            print(
                f"\rEpoch {epoch+1}/{epochs} | Batch {step+1} | Loss: {batch_loss:.6f} | Cum Loss: {cum_loss:.6f} | Time: {elapsed_epoch:.1f}s",
                end="",
                flush=True
            )

        print()  # New line after epoch completes
        if save_every is not None and step % save_every != 0:
            save_checkpoint(model, optimizer, filepath=checkpoint_filepath, epoch=epoch + 1, step=0)           


def train_gpu(
    src_file,
    tgt_file,
    file_dims,
    model_configs,
    bos_id=0,
    eos_id=1,
    pad_id=2,
    batch_size=256,
    epochs=10,
    loader_workers=2, # Optimized for T4 (Colab allows 2 worker threads)
    checkpoint_filepath=None,
    dynamic_padding=False,
    save_every=1000
):
    # [GPU] Setup standard CUDA device
    device = torch.device("cuda")
    print(f"Using device: {device}", end="\n", flush=True)

    translation_dataset = TranslationDatasetFast(src_file, tgt_file, dims=file_dims)
    if dynamic_padding:
        collator = DynamicTrimmingCollator(bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
    else:
        collator = Collator(pad_id=pad_id)

    # [GPU Optimization] pin_memory=True speeds up CPU-to-GPU data transfers
    loader = DataLoader(
        translation_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=loader_workers,
        pin_memory=True, 
        drop_last=False,
    )

    model = make_model(**model_configs)

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_filepath)

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to T4 GPU
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=1)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_id)
    loss = Loss(model.generator, criterion)

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size=model_configs["d_model"])
    )

    # [GPU Optimization] Initialize GradScaler for FP16 Mixed Precision
    scaler = torch.amp.GradScaler(device="cuda")

    model.train()

    total_loss = 0.0
    total_tokens = 0

    if checkpoint:
        start_epoch, start_step = checkpoint["epoch"], checkpoint["step"]
    else:
        start_epoch, start_step = 0, 0

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # [GPU Optimization] Use standard loader; pl.MpDeviceLoader is removed
        for step, batch in enumerate(loader):
            if epoch == start_epoch and step < start_step:
                continue

            optimizer.zero_grad(set_to_none=True)

            # [GPU Optimization] Move tensors non-blockingly (works with pin_memory=True)
            batch.to_device(device)

            # [GPU Optimization] Changed from "xla"/bfloat16 to "cuda"/float16 for T4 Tensor Cores
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(
                    batch.src, 
                    batch.tgt, 
                    batch.src_mask, 
                    batch.tgt_mask
                )
                loss_ = loss(output, batch.tgt_y, batch.ntokens)

            # [GPU Optimization] Scale loss and scale gradients during backward pass
            scaler.scale(loss_).backward()

            # [GPU Optimization] Unscale gradients and step optimizer safely
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            loss_value = loss_.item()
            ntokens = batch.ntokens.item()

            total_loss += loss_value
            total_tokens += ntokens

            elapsed_epoch = time.time() - epoch_start_time
            batch_loss = loss_value / ntokens
            cum_loss = total_loss / total_tokens

            # [GPU Optimization] Removed TPU-specific monitoring metrics
            if save_every is not None and step != 0 and step % save_every == 0:
                save_checkpoint(model, optimizer, filepath=checkpoint_filepath, epoch=epoch, step=step + 1)

            print(
                f"\rEpoch {epoch+1}/{epochs} | Batch {step+1} | Loss: {batch_loss:.6f} | Cum Loss: {cum_loss:.6f} | Time: {elapsed_epoch:.1f}s",
                end="",
                flush=True
            )

        print() 
        if save_every is not None and step % save_every != 0:
            save_checkpoint(model, optimizer, filepath=checkpoint_filepath, epoch=epoch + 1, step=0)
