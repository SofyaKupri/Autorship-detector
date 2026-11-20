"""Burrows' Delta classifier built with pandas and numpy."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from authorship.books import Chunk
from authorship.exceptions import InsufficientTextError

# Below this many words the frequencies are too noisy to say anything.
MIN_TEXT_WORDS = 50
# Protects against division by zero for words with (almost) no variation.
MIN_STD = 1e-6


class DeltaClassifier:
    """Guesses the author of a text with Burrows' Delta.

    Training learns, for every author, a *style profile*: the average
    z-score of the usage frequency of the most common words. Classification
    computes the same z-scores for an unknown text and measures its distance
    (Delta) to every profile.
    """

    def __init__(self, top_words: int = 150) -> None:
        """Create an untrained classifier.

        Args:
            top_words: Number of most frequent words that describe a style.
        """
        self._top_words = top_words
        self._vocabulary: list[str] = []
        self._mean = np.empty(0)
        self._std = np.empty(0)
        self._profiles = pd.DataFrame()

    @property
    def authors(self) -> list[str]:
        """Names of the authors the model knows (empty before training)."""
        return list(self._profiles.index)

    @property
    def vocabulary(self) -> list[str]:
        """The words used as the style fingerprint, most frequent first."""
        return list(self._vocabulary)

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], top_words: int = 150) -> DeltaClassifier:
        """Factory: create a classifier and train it right away."""
        return cls(top_words).fit(chunks)

    @staticmethod
    def frequencies(words: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
        """Relative frequency of every vocabulary word in ``words``."""
        counts = Counter(words)
        return np.array([counts[word] for word in vocabulary], dtype=float) / len(words)

    def fit(self, chunks: Sequence[Chunk]) -> DeltaClassifier:
        """Learn the style profiles of all authors.

        Raises:
            InsufficientTextError: If the chunks come from fewer than two authors.
        """
        if len({chunk.author for chunk in chunks}) < 2:
            raise InsufficientTextError("need chunks from at least two authors")

        totals: Counter[str] = Counter()
        for chunk in chunks:
            totals.update(chunk.words)
        self._vocabulary = [word for word, _ in totals.most_common(self._top_words)]

        # One row per chunk, one column per word.
        matrix = np.array([self.frequencies(c.words, self._vocabulary) for c in chunks])
        self._mean = matrix.mean(axis=0)
        self._std = np.maximum(matrix.std(axis=0), MIN_STD)
        z_scores = (matrix - self._mean) / self._std

        table = pd.DataFrame(
            z_scores, index=[c.author for c in chunks], columns=self._vocabulary
        )
        self._profiles = table.groupby(level=0).mean()
        return self

    def rank(self, words: Sequence[str]) -> pd.Series:
        """Compare a text with every author.

        Args:
            words: The words of the unknown text, lowercase.

        Returns:
            Delta per author, sorted from the closest (best match) to the farthest.

        Raises:
            RuntimeError: If the model has not been trained.
            InsufficientTextError: If the text is too short.
        """
        if self._profiles.empty:
            raise RuntimeError("the model is not trained; call fit() first")
        if len(words) < MIN_TEXT_WORDS:
            raise InsufficientTextError(
                f"text has {len(words)} words, need at least {MIN_TEXT_WORDS}"
            )

        z_scores = (self.frequencies(words, self._vocabulary) - self._mean) / self._std
        deltas = np.abs(self._profiles.to_numpy() - z_scores).mean(axis=1)
        return pd.Series(deltas, index=self._profiles.index, name="delta").sort_values()


def evaluate(chunks: Sequence[Chunk], top_words: int = 150) -> pd.Series:
    """Measure accuracy with leave-one-book-out testing.

    Each book in turn is hidden, the model is trained on all other books,
    and every chunk of the hidden book is classified. This shows how well the
    method recognises an author on a book it has never seen.

    Args:
        chunks: Labelled chunks of all books.
        top_words: Number of most frequent words that describe a style.

    Returns:
        Share of correctly classified chunks (0..1) per author.

    Raises:
        InsufficientTextError: If no book could be tested (each author needs
            at least two books).
    """
    hits: dict[str, list[bool]] = {}
    for author, book in sorted({(c.author, c.book) for c in chunks}):
        training = [c for c in chunks if (c.author, c.book) != (author, book)]
        remaining_authors = {c.author for c in training}
        if author not in remaining_authors or len(remaining_authors) < 2:
            continue  # nothing left to learn this author from

        model = DeltaClassifier.from_chunks(training, top_words)
        for chunk in chunks:
            if (chunk.author, chunk.book) == (author, book):
                guess = model.rank(chunk.words).index[0]
                hits.setdefault(author, []).append(guess == author)

    if not hits:
        raise InsufficientTextError("evaluation needs at least two books per author")
    return pd.Series({name: float(np.mean(results)) for name, results in hits.items()})
