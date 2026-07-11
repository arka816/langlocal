import time
import os
from dataclasses import dataclass

from data_pipeline.dataset import TranslationDataset
from data_pipeline.collator import DynamicTrimmingCollator
from data_pipeline.batch import Batch
from loss import Loss
from transformer import make_model

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp



def tpu_logger(log_func, *items):
    if xm.is_master_ordinal():
        log_func(*items)


def save_checkpoint(model, optimizer, epoch, step, filepath):
    "Saves the model and optimizer states from a multi-core TPU environment."

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    # xm.save ensures only master ordinal (Core 0) writes to the storage
    # while coordinating with all other cores to consolidate the data
    xm.save(checkpoint, filepath)

    tpu_logger(print, f"Checkpoint successfully saved to {filepath} at Epoch {epoch}, Step {step}")


def load_checkpoint(model, optimizer, filepath, device):
    "Loads a checkpoint in a multi-core TPU environment."
    if not filepath or not os.path.exists(filepath):
        tpu_logger(print, f"No checkpoint found at {filepath}. Starting training from scratch.")
        return 0, 0

    tpu_logger(print, f"Loading checkpoint from {filepath}...")

    # ALWAYS load the checkpoint to the CPU first to avoid cross-device memory bugs
    checkpoint = torch.load(filepath, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Synchronize all cores to ensure they don't start until loading is complete
    xm.rendezvous("checkpoint_loaded")

    return checkpoint["epoch"], checkpoint["step"]


def train_per_tpu_core(
    model_configs,
    batch_size=16,
    epochs=10,
    pad_idx=2,
    checkpoint_filepath=None,
):
    "Code to train on multiple TPU cores (on colab)."

    # CRITICAL: Force every parallel process to use the identical random seed
    # This ensures nn.init custom initializations produce identical values across all 8 cores
    torch.manual_seed(42)

    # Set the unique device for this specific core (0 to 7)
    device = xm.xla_device()

    # Initialize dataset
    translation_dataset = TranslationDataset()

    # CRITICAL - each TPU core should get a distinct shard of data
    sampler = DistributedSampler(
        dataset=translation_dataset,
        num_replicas=xm.xrt_world_size(),   # Total cores(8)
        rank=xm.get_ordinal(),              # Current core ID (0-7)
        shuffle=True,
    )

    loader = DataLoader(
        dataset=translation_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        drop_last=False,
    )

    model = make_model(**model_configs).to(device=device)

    # Custom initialization function (first `optimizer_step` call ensures that all cores have identical parameters)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_idx)
    loss = Loss(model.generator, criterion)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.001)

    # Print only from the master core (index 0) to prevent 8x spam
    tpu_logger(print, f"Starting training on {xm.xrt_world_size()} TPU cores...")

    # Load checkpoint
    start_epoch, start_step = load_checkpoint(model, optimizer, checkpoint_filepath)

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)            # Keep shuffling across epochs

        # CRITICAL: Wrap DataLoader with MpDeviceLoader for background TPU memory transfers
        tpu_loader = pl.MpDeviceLoader(loader, device)

        model.train()

        for step, (src_tensors, tgt_tensors) in enumerate(tpu_loader):
            # Skip steps if resuming mid-epoch
            if epoch == start_epoch and step <= start_step:
                continue

            optimizer.zero_grad()

            # INTEGRATION POINT: Wrap raw device tensors into your Batch object
            # Tensors are already on the TPU device via MpDeviceLoader
            batch = Batch(src_tensors, tgt_tensors, pad=pad_idx)

            output = model(
                batch.src, 
                batch.tgt, 
                batch.src_mask, 
                batch.tgt_mask
            )

            loss = loss(output, batch.tgt_y, batch.ntokens)

            loss.backward()

            # Synchronize gradients across all 8 cores and step
            xm.optimizer_step(optimizer)

        
        # Save at the end of every epoch
        if checkpoint_filepath:
            save_checkpoint(model, optimizer, epoch=epoch + 1, step=0, filepath=checkpoint_filepath)
        
