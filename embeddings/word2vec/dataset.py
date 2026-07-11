"""Dataset utilities for training Word2Vec models."""

import io
from os.path import join
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import zstandard as zstd

from data.schema import Sentence, Word


class SentenceDataset(Dataset):
    """Load preprocessed sentences from a Zstandard-backed file."""

    def __init__(self, sentences_dir: str) -> None:
        self.sentences: List[Sentence] = []

        with open(join(sentences_dir, "sentences.zst"), "rb") as f:
            decompressor = zstd.ZstdDecompressor()

            with decompressor.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")

                for line in text_stream:
                    line = line.strip("\n")
                    sentence = line.split("\t")
                    if len(sentence) >= 2:
                        self.sentences.append(sentence)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Sentence:
        return self.sentences[idx]


class SkipGramCollator:
    """Collate function for skip-gram batches with negative sampling."""

    class _NegativeSamplePool:
        """Cache negative samples and refill the pool lazily."""

        def __init__(self, sampler: Callable[[int], List[int]], batch_size: int) -> None:
            self._sampler = sampler
            self._batch_size = max(batch_size, 64)
            self._pool: List[int] = []
            self._refill()

        def _refill(self) -> None:
            sampled = self._sampler(self._batch_size)
            self._pool.extend(sampled)

        def draw(self, center: int, count: int) -> List[int]:
            negatives: List[int] = []

            while len(negatives) < count:
                if not self._pool:
                    self._refill()

                candidate = self._pool.pop()
                if candidate != center:
                    negatives.append(candidate)

            return negatives

    def __init__(
        self,
        word2idx: Callable[[Word], int],
        negative_sampler: Callable[[int], List[int]],
        max_window: int = 5,
        negative_samples: int = 5,
    ) -> None:
        self.word2idx = word2idx
        self.max_window = max_window

        self.negative_sampler = negative_sampler
        self.negative_samples = negative_samples

        self._rng = np.random.default_rng(seed=42)
        self._negative_sample_pool = self._NegativeSamplePool(
            sampler=negative_sampler,
            batch_size=max(negative_samples * 1024, 64),
        )

    def __call__(self, batch: List[Sentence]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a batch of sentences into center/context/negative training tensors.

        Args:
            batch (List[Sentence]): A batch of tokenized sentences produced by the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                center words, positive context words, and negative samples tensors.
        """

        centers = []
        positive_contexts = []
        negative_contexts = []

        for sentence in batch:
            ids = [self.word2idx(word) for word in sentence]

            n = len(ids)

            for center_idx in range(n):
                center = ids[center_idx]

                window = self._rng.integers(1, self.max_window)

                left = max(0, center_idx - window)
                right = min(n, center_idx + window + 1)

                for context_idx in range(left, right):
                    if context_idx == center_idx:
                        continue

                    positive = ids[context_idx]
                    negatives = self._negative_sample_pool.draw(
                        center=center,
                        count=self.negative_samples,
                    )

                    centers.append(center)
                    positive_contexts.append(positive)
                    negative_contexts.append(negatives)

        return (
            torch.tensor(centers, dtype=torch.long),
            torch.tensor(positive_contexts, dtype=torch.long),
            torch.tensor(negative_contexts, dtype=torch.long),
        )
