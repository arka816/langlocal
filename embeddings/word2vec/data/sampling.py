import numpy as np
from typing import Counter, Iterable, List, Tuple

from .schema import Word, Sentence, TrainingPair
from .vocabulary import Vocabulary


def sentences_to_data_samples(sentence_iterator: Iterable[Sentence], vocabulary: Vocabulary, context_window_size: int) -> List[TrainingPair]:
    """
    Convert a sequence of sentences into training pairs (target, context) for Word2Vec.

    Args:
        sentence_iterator (Iterable[Sentence]): A sequence of sentences, each represented as a list of words.
        vocabulary (Vocabulary): The vocabulary object containing the words and their indices.
        context_window_size (int): The size of the context window for generating training pairs.

    Returns:
        List[TrainingPair]: A list of training pairs (target, context) for Word2Vec.
    """
    training_pairs: List[TrainingPair] = []
    for sentence in sentence_iterator:
        for index, target_word in enumerate(sentence):
            if target_word not in vocabulary:
                continue
            start_index = max(0, index - context_window_size)
            end_index = min(len(sentence), index + context_window_size + 1)
            context_words: Sequence[Word] = [sentence[i] for i in range(start_index, end_index) if i != index]
            for context_word in context_words:
                if context_word in vocabulary:
                    training_pairs.append((target_word, context_word))
    return training_pairs

def generate_negative_samples(positive_samples: List[TrainingPair], vocabulary: Vocabulary, num_negative_samples: int) -> List[List[Word]]:
    """
    Generate negative samples for the given positive samples.

    Args:
        positive_samples (List[TrainingPair]): List of positive (target, context) pairs.
        vocabulary (Vocabulary): The vocabulary object.
        num_negative_samples (int): Number of negative samples to generate for each positive sample.

    Returns:
        List[List[Word]]: List of negative (target, context) pairs.
    """
    negative_samples = []
    for target, context in positive_samples:
        negative_contexts = vocabulary.sample(num_negative_samples + 1)  # +1 to ensure we can filter out the positive context
        negative_contexts = [word for word in negative_contexts if word != context][:num_negative_samples]  # Filter out the positive context and limit to num_negative_samples
        negative_samples.append(negative_contexts)
    return negative_samples

def encode_samples(data_samples: List[TrainingPair], negative_samples: List[List[Word]], vocabulary: Vocabulary) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode the training samples and negative samples into numpy arrays.

    Args:
        data_samples (List[TrainingPair]): List of positive (target, context) pairs.
        negative_samples (List[List[Word]]): List of negative (target, context) pairs.
        vocabulary (Vocabulary): The vocabulary object.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Encoded target words, context words, and negative words.
    """
    centers = np.array([vocabulary.index(target) for target, _ in data_samples], dtype=np.int32)
    contexts = np.array([vocabulary.index(context) for _, context in data_samples], dtype=np.int32)
    negatives = np.array([[vocabulary.index(neg) for neg in neg_list] for neg_list in negative_samples], dtype=np.int32)
    return centers, contexts, negatives
