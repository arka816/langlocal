"""Text cleanup utilities for Word2Vec preprocessing."""

from typing import Iterable, Iterator
import string

from .schema import Sentence

class CorpusCleanup:
    """Clean and normalize raw corpus text and tokenized sentences."""
    markup_tag_pairs = [
        ('<br>', '</br>'),
        ('<p>', '</p>'),
        ('<div>', '</div>'),
        ('<span>', '</span>'),
        ('<a>', '</a>'),
        ('<img>', '</img>'),
        ('<ul>', '</ul>'),
        ('<li>', '</li>'),
        ('<h1>', '</h1>'),
        ('<h2>', '</h2>'),
        ('<h3>', '</h3>'),
        ('<h4>', '</h4>'),
        ('<h5>', '</h5>'),
        ('<h6>', '</h6>'),
    ]

    @staticmethod
    def pre_segmentation_cleanup(lines_iterator: Iterable[str]) -> Iterator[str]:
        """
        Cleans up the text data by removing unwanted characters, tags etc.
        Args:
            lines_iterator (Iterable[str]): An iterator over raw text lines.

        Returns:
            Iterator[str]: The cleaned text lines.
        """
        for line in lines_iterator:
            cleaned = line.strip().replace('\n', ' ').replace('\r', ' ')
            for open_tag, close_tag in CorpusCleanup.markup_tag_pairs:
                cleaned = cleaned.replace(open_tag, ' ').replace(close_tag, ' ')
            yield cleaned

    @staticmethod
    def post_segmentation_cleanup(sentence_iterator: Iterator[Sentence]) -> Iterator[Sentence]:
        """Remove punctuation from each token in a tokenized sentence.

        Args:
            sentence_iterator (Iterator[Sentence]): An iterator over tokenized sentences.

        Returns:
            Iterator[Sentence]: Cleaned tokenized sentences.
        """
        for sentence in sentence_iterator:
            clean_sentence = [word.translate(str.maketrans('', '', string.punctuation)).strip() for word in sentence]
            yield clean_sentence
    