import numpy as np
from collections import Counter
from typing import Dict, Sequence, List, Iterator

from os.path import join
import zstandard as zstd
import pickle

from .schema import Sentence, Word


class Vocabulary:
    """A vocabulary object that tracks word frequencies and sampling distribution."""

    def __init__(self, alpha: float = 0.75, min_frequency: float = 5):
        self.frequencies: Counter[Word] = Counter()
        self.word2idx: Dict[Word, int]  = dict()
        self.idx2word: Dict[int, Word]  = dict()
        self.vocabulary: List[Word]     = []
        self.one_hots: List[int]        = []

        self.distribution: List[float] = []
        self.alpha: float = alpha
        self._rng = np.random.default_rng(seed=42)

        self.min_frequency = min_frequency
        self.cleaned = False


    def add_sentence(self, sentence: Sentence) -> None:
        """Add words from a sentence to the vocabulary."""
        for word in sentence:
            self.frequencies[word] += 1
            if word not in self.word2idx:
                assigned_idx = len(self.word2idx)
                self.word2idx[word] = assigned_idx
                self.idx2word[assigned_idx] = word
                self.vocabulary.append(word)

    def _prune_vocabulary(self) -> None:
        """Remove low-frequency words from the vocabulary."""
        words_to_remove = [word for word in self.vocabulary if self.frequencies[word] < self.min_frequency]

        for word in words_to_remove:
            del self.frequencies[word]

            target_idx = self.word2idx[word]
            last_word = self.vocabulary[-1]
            last_idx = len(self.vocabulary) - 1

            if word != last_word:
                self.vocabulary[target_idx] = last_word
                self.word2idx[last_word] = target_idx
                self.idx2word[target_idx] = last_word

            self.vocabulary.pop()
            del self.word2idx[word]
            del self.idx2word[last_idx]

        self.one_hots = [self.word2idx[word] for word in self.vocabulary]

        self.cleaned = True


    def __contains__(self, word: Word) -> bool:
        """
        Check if a word is in the vocabulary. O(1) search time.

        Args:
            word (Word): The word to check.
        
        Returns:
            bool: True if the word is in the vocabulary, False otherwise.
        """
        return word in self.word2idx
    
    def __len__(self) -> int:
        """
        Get the number of unique words in the vocabulary.
        
        Returns:
            int: The number of unique words in the vocabulary.
        """
        return len(self.vocabulary)
    
    def __iter__(self) -> Iterator[Word]:
        """
        Iterate over the words in the vocabulary.
        
        Yields:
            Word: The next word in the vocabulary.
        """
        for word in self.vocabulary:
            yield word

    def index(self, word: Word) -> int:
        """
        Get the index of a word in the vocabulary. O(1) search time.
        
        Args:
            word (Word): The word to get the index for.
        
        Returns:
            int: The index of the word in the vocabulary.
        """
        return self.word2idx[word]
    
    def __getitem__(self, index: int) -> Word:
        """
        Get the word at the specified index in the vocabulary.
        
        Args:
            index (int): The index of the word to retrieve.
        
        Returns:
            Word: The word at the specified index.
        """
        return self.vocabulary[index]

    def _calculate_probabilities(self) -> None:
        """Calculate the negative sampling distribution from word frequencies."""
        freq = np.array([self.frequencies[word] for word in self.vocabulary])
        mod_freq = freq ** self.alpha
        self.distribution = (mod_freq / mod_freq.sum()).tolist()

    def sample(self, num_samples: int) -> List[int]:
        """Sample words from the vocabulary based on their frequencies.

        Args:
            num_samples (int): The number of words to sample.

        Returns:
            List[Word]: A list of sampled words.
        """
        if not self.cleaned:
            self._prune_vocabulary()

        if not self.distribution:
            self._calculate_probabilities()
        
        sampled_words = self._rng.choice(
            self.one_hots, size=num_samples, p=self.distribution
        )
        return sampled_words.tolist()


def load_vocabulary(vocabulary_dir):
    with open(join(vocabulary_dir, "vocabulary.pkl.zst"), "rb") as f:
        decompressor = zstd.ZstdDecompressor()

        with decompressor.stream_reader(f) as reader:
            vocabulary = pickle.load(reader)

            return vocabulary
