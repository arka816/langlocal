import numpy as np
from typing import Union, List, Callable
import faiss

from data.schema import Word
from data.vocabulary import Vocabulary
from logger import logger

class Embedder:
    def __init__(self, vocabulary: Vocabulary, embedding_path: str):
        self.vocabulary = vocabulary
        self.embedding = np.load(embedding_path, mmap_mode='r')
        self.index = None

    def _embed(self, word: Word):
        index = self.vocabulary.index(word)
        if index:
            return self.embedding[index, :]
        else:
            return None

    def __call__(self, words: List[Word]):
        if type(words) != list:
            raise ValueError("Argument `words` must be of type `list`")
        
        return [self._embed(word) for word in words]

    def build_index(self, embedding_path: str, index_path: str) -> None:
        shape = self.embedding.shape
        dimension = shape[1]

        self.index = faiss.IndexFlatL2(dimension)    # L2 norm based distance

        batch_size = 5000
        for i in range(0, shape[0], batch_size):
            batch = self.embedding[i:min(i + batch_size, shape[0])]

            batch_contiguous = np.ascontiguousarray(batch, dtype='float32')
            self.index.add(batch_contiguous)

        logger.info("Total vectors indexed: %d", self.index.ntotal)

        faiss.write_index(self.index, index_path)

    def _search_index(self, word: Word, k: int=10) -> List[int]:
        query_vector = self._embed(word)
        query_vectors = query_vector.reshape(1, query_vector.shape[0])

        distances, indices = self.index.search(query_vectors, k)
        return indices[0]

    def nearest(self, word: Word, k: int=10) -> List[Word]:
        if self.index is None:
            raise RuntimeError("Index not built yet. Please invoke `build_index` before calling `nearest`.")
        indices = self._search_index(word)
        return [self.vocabulary[index] for index in indices]
