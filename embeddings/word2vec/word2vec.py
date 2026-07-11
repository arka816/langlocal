"""Word2Vec model implementation using skip-gram and negative sampling."""

import torch
import torch.nn as nn
from torch import Tensor
from data.schema import Word

class Word2Vec(nn.Module):
    """A simple Word2Vec model with input and output embeddings."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()

        self.input_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.output_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

    def forward(
        self,
        center_words: Tensor,
        context_words: Tensor,
        negative_words: Tensor,
    ) -> Tensor:
        """Compute the skip-gram loss for a batch of training examples.

        Args:
            center_words (Tensor): Tensor of center word indices, shape (batch_size,).
            context_words (Tensor): Tensor of context word indices, shape (batch_size,).
            negative_words (Tensor): Tensor of negative sample indices, shape (batch_size, num_negative_samples).

        Returns:
            Tensor: The averaged skip-gram loss for the batch.
        """
        center = self.input_embeddings(center_words)        # batch_size x embedding_dim
        positive = self.output_embeddings(context_words)    # batch_size x embedding_dim
        negative = self.output_embeddings(negative_words)   # batch_size x num_negative_samples x embedding_dim

        positive_score = torch.einsum('ij,ij->i', center, positive)     # batch_size
        negative_score = torch.einsum('ij,ikj->ik', center, negative)   # batch_size x num_negative_samples

        positive_loss = torch.log(torch.sigmoid(positive_score))        # batch_size
        negative_loss = torch.log(torch.sigmoid(-negative_score))       # batch_size x num_negative_samples

        loss = - positive_loss - torch.einsum('ij->i', negative_loss)   # batch_size
        loss = loss.mean()  # Average over the batch

        return loss
