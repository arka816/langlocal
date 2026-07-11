from os.path import join
import numpy as np
from typing import Iterable, Iterator, List, Tuple, Sequence
import zstandard as zstd
import pickle

from .schema import Sentence, TrainingPair, Word
from .loader import CorpusLoader
from .vocabulary import Vocabulary
from .cleanup import CorpusCleanup
from .segment import corpus_to_sentences

from logger import logger


cctx = zstd.ZstdCompressor()

def build_vocabulary(
        file_path: str, 
        min_frequency: int, 
        alpha: float,
        vocabulary_dir: str,
    ) -> None:
    """Build vocabulary and save preprocessing artifacts from raw text.

    Args:
        file_path (str): Path to the raw text file.
        min_frequency (int): Minimum frequency threshold for words.
        alpha (float): Exponent for word sampling distribution.
        vocabulary_dir (str): Directory to store generated vocabulary artifacts.
    """
    corpus: Iterator[str] = CorpusLoader.load_corpus(file_path)
    cleaned_corpus: Iterator[str] = CorpusCleanup.pre_segmentation_cleanup(corpus)
    sentences: Sequence[Sentence] = corpus_to_sentences(cleaned_corpus)
    cleaned_sentences: Sequence[Sentence] = CorpusCleanup.post_segmentation_cleanup(sentences)

    vocabulary = Vocabulary(alpha=alpha, min_frequency=min_frequency)
    
    with open(join(vocabulary_dir, "sentences.zst"), "wb") as of:
        with cctx.stream_writer(of) as compressor:
            for sentence in cleaned_sentences:
                vocabulary.add_sentence(sentence)
                line = "\t".join(sentence) + "\n"
                compressor.write(line.encode("utf-8"))

    vocabulary.sample(1)    # trigger cleaning and frequency calculation before dumping

    with open(join(vocabulary_dir, "vocabulary.pkl.zst"), "wb") as of:
        with cctx.stream_writer(of) as compressor:
            pickle.dump(vocabulary, compressor, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved vocabulary to %s", join(vocabulary_dir, 'vocabulary.pkl.zst'))
