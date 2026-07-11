"""Raw text loader for Word2Vec preprocessing."""

from typing import Iterator

class CorpusLoader:
    """Load raw text lines from a file."""
    
    @staticmethod
    def load_corpus(file_path: str, encoding: str = "utf-8") -> Iterator[str]:
        """Yield raw text lines from a file.

        Args:
            file_path (str): Path to the raw text file to load.
            encoding (str): File encoding used to read the input file.

        Returns:
            Iterator[str]: Raw text lines from the file.
        """
        with open(file_path, "r", encoding=encoding) as file:
            for line in file:
                yield line
