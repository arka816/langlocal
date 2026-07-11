"""Training utilities for Word2Vec models."""

from os.path import join
from typing import Any, Dict, Optional

import argparse
import pickle
import zstandard as zstd
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from word2vec import Word2Vec
from dataset import SentenceDataset, SkipGramCollator
from configs import configs
from logger import logger


def train_word2vec(
    max_window: Optional[int] = None,
    negative_samples: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    learning_rate: Optional[float] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Train a Word2Vec model using a prebuilt vocabulary and sentence dataset.

    Args:
        max_window (Optional[int]): Context window size override. Uses config default if None.
        negative_samples (Optional[int]): Number of negative samples per positive pair.
        embedding_dim (Optional[int]): Dimensionality of the word embeddings.
        learning_rate (Optional[float]): Learning rate for the optimizer.
        num_epochs (Optional[int]): Number of training epochs.
        batch_size (Optional[int]): Batch size for training.

    Returns:
        None: This function saves model checkpoints and final embeddings to disk.
    """
    max_window = max_window or configs['training']['CONTEXT_WINDOW_SIZE']
    negative_samples = negative_samples or configs['training']['NUM_NEGATIVE_SAMPLES']

    embedding_dim = embedding_dim or configs['training']['EMBEDDING_DIM']
    learning_rate = learning_rate or configs['training']['LEARNING_RATE']
    num_epochs = num_epochs or configs['training']['NUM_EPOCHS']
    batch_size = batch_size or configs['training']['BATCH_SIZE']

    # Load Vocabulary
    with open(join(configs['cache']['VOCABULARY_PATH'], "vocabulary.pkl.zst"), "rb") as f:
        decompressor = zstd.ZstdDecompressor()

        with decompressor.stream_reader(f) as reader:
            vocabulary = pickle.load(reader)

    # Initialize the Word2Vec model
    model = Word2Vec(
        vocab_size=len(vocabulary),
        embedding_dim=embedding_dim,
    )

    # Define the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Optionally resume from checkpoint
    start_epoch = 0
    if checkpoint_path:
        logger.info("Received checkpoint path: %s", checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint.get('model_state_dict', {}))
            optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
            # Stored epoch is 1-based (matches previous implementation)
            start_epoch = int(checkpoint.get('epoch', 0))
            logger.info("Resuming training from checkpoint: epoch %d", start_epoch)
        except Exception as e:
            logger.error("Failed to load checkpoint '%s': %s. Starting from scratch.", checkpoint_path, e)

    # Load the dataset
    logger.info("Loading Dataset...")
    dataset = SentenceDataset(sentences_dir=configs['cache']['VOCABULARY_PATH'])

    logger.info("Loading Collator...")
    collator = SkipGramCollator(
        word2idx=vocabulary.index,
        negative_sampler=vocabulary.sample,
        max_window=max_window,
        negative_samples=negative_samples,
    )

    logger.info("Preparing DataLoader...")
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collator,
        num_workers=0,      # No memory explosion
    )

    logger.info("Kicking off training...")

    for epoch in range(start_epoch, num_epochs):
        for center_words, context_words, negative_words in tqdm(loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
            optimizer.zero_grad()

            loss = model(center_words, context_words, negative_words)

            loss.backward()

            optimizer.step()

        logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, num_epochs, loss.item())

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }

        try:
            torch.save(checkpoint, join(configs['cache']['CHECKPOINTS_PATH'], f"word2vec_embedding_{embedding_dim}d_checkpoint.pt"))
        except Exception:
            logger.exception("Checkpoint failed.")

    try:
        torch.save(model.state_dict(), join(configs['cache']['EMBEDDINGS_PATH'], f"word2vec_embeddings_{embedding_dim}d.pt"))
    except Exception:
        logger.exception("Model save failed.")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or resume Word2Vec model')
    parser.add_argument('--checkpoint', '-c', type=str, default=None, help='Path to checkpoint file to resume from')
    args = parser.parse_args()

    train_word2vec(checkpoint_path=args.checkpoint)
