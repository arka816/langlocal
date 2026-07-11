import pysbd
from typing import Iterable, Iterator, Sequence

from .schema import Sentence

segmenter = pysbd.Segmenter(language="en", clean=False)

def corpus_to_sentences(cleaned_lines_iterator: Iterable[str]) -> Iterator[Sentence]:
    """Split cleaned text lines into sentences and tokenize each sentence.

    Args:
        cleaned_lines_iterator (Iterable[str]): Iterator over cleaned text lines.

    Returns:
        Iterator[Sentence]: Tokenized sentences produced from the cleaned text.
    """
    for line in cleaned_lines_iterator:
        sentences: Sequence[str] = segmenter.segment(line)
        for sentence in sentences:
            sentence: Sentence = sentence.split()
            yield sentence
