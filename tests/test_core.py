"""Offline tests: no network access is needed."""

from __future__ import annotations

import io
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import authorship.books as books
from authorship.books import Chunk, get_archive, load_chunks
from authorship.config import load_config
from authorship.exceptions import ConfigError, FetchError, InsufficientTextError, ParseError
from authorship.model import DeltaClassifier, evaluate
from authorship.text import split_into_chunks, tokenize

VOCABULARY = [f"w{chr(97 + i)}" for i in range(40)]


def synthetic_words(probabilities: np.ndarray, count: int, seed: int) -> list[str]:
    """Draw words from a fixed distribution: a stand-in for an author's style."""
    rng = np.random.default_rng(seed)
    return rng.choice(VOCABULARY, size=count, p=probabilities).tolist()


def make_styles() -> dict[str, np.ndarray]:
    """Two authors with clearly different word usage habits."""
    rng = np.random.default_rng(0)
    styles = {}
    for name in ("Alice", "Bob"):
        weights = rng.dirichlet(np.ones(len(VOCABULARY)) * 0.5)
        styles[name] = weights
    return styles


def make_chunks(books_per_author: int = 2) -> list[Chunk]:
    chunks = []
    seed = 100
    for author, probabilities in make_styles().items():
        for book in range(books_per_author):
            for _ in range(4):
                seed += 1
                words = synthetic_words(probabilities, 500, seed)
                chunks.append(Chunk(author, f"{author.lower()}-{book}.txt", tuple(words)))
    return chunks


# --- text parsing ----------------------------------------------------------

def test_tokenize() -> None:
    assert tokenize("Don\u2019t stop - it's 3 o'clock!") == ["don't", "stop", "it's", "o'clock"]


def test_split_into_chunks_drops_leftover() -> None:
    chunks = split_into_chunks(list("abcdefghij"), 4)
    assert chunks == [list("abcd"), list("efgh")]
    assert split_into_chunks(list("abc"), 4) == []


# --- books -----------------------------------------------------------------

def make_archive(folder: Path, files: dict[str, list[str]]) -> None:
    """Create gutenberg.zip with one text file per entry, like the real archive."""
    with zipfile.ZipFile(folder / "gutenberg.zip", "w") as archive:
        for name, words in files.items():
            archive.writestr(f"gutenberg/{name}", " ".join(words))


def test_get_archive_uses_cache_without_network() -> None:
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "gutenberg.zip").write_bytes(b"cached")
        assert get_archive(folder).read_bytes() == b"cached"


def test_load_chunks_from_archive() -> None:
    styles = make_styles()
    with tempfile.TemporaryDirectory() as folder:
        make_archive(
            Path(folder),
            {
                "a-one.txt": synthetic_words(styles["Alice"], 1000, 1),
                "b-one.txt": synthetic_words(styles["Bob"], 1000, 2),
            },
        )
        chunks = load_chunks({"Alice": ["a-one.txt"], "Bob": ["b-one.txt"]}, 250, folder)
    assert len(chunks) == 8
    assert {c.author for c in chunks} == {"Alice", "Bob"}
    assert all(len(c.words) == 250 for c in chunks)


def test_load_chunks_unknown_book() -> None:
    with tempfile.TemporaryDirectory() as folder:
        make_archive(Path(folder), {"a-one.txt": ["word"] * 500})
        with pytest.raises(ParseError):
            load_chunks({"Alice": ["a-one.txt"], "Bob": ["missing.txt"]}, 250, folder)


def test_load_chunks_needs_two_authors() -> None:
    styles = make_styles()
    with tempfile.TemporaryDirectory() as folder:
        make_archive(
            Path(folder),
            {
                "a-one.txt": synthetic_words(styles["Alice"], 1000, 1),
                "b-one.txt": synthetic_words(styles["Bob"], 100, 2),  # too short
            },
        )
        with pytest.raises(InsufficientTextError):
            load_chunks({"Alice": ["a-one.txt"], "Bob": ["b-one.txt"]}, 250, folder)


def test_load_chunks_damaged_archive() -> None:
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "gutenberg.zip").write_bytes(b"this is not a zip file")
        with pytest.raises(FetchError):
            load_chunks({"Alice": ["a.txt"], "Bob": ["b.txt"]}, 250, folder)


class _Response(io.BytesIO):
    """Minimal stand-in for the object returned by urlopen."""

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        pass


def test_download_retries_temporary_errors() -> None:
    attempts: list[str] = []

    def flaky_urlopen(request: urllib.request.Request, timeout: float = 0) -> _Response:
        attempts.append(request.full_url)
        if len(attempts) < 3:  # two temporary failures, then success
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)
        return _Response(b"zip bytes")

    with (
        mock.patch.object(urllib.request, "urlopen", flaky_urlopen),
        mock.patch.object(books.time, "sleep"),
        tempfile.TemporaryDirectory() as folder,
    ):
        path = get_archive(folder)
        assert path.read_bytes() == b"zip bytes"
    assert len(attempts) == 3


def test_download_gives_up_with_helpful_message() -> None:
    def always_busy(request: urllib.request.Request, timeout: float = 0) -> _Response:
        raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

    with (
        mock.patch.object(urllib.request, "urlopen", always_busy),
        mock.patch.object(books.time, "sleep"),
        tempfile.TemporaryDirectory() as folder,
    ):
        try:
            get_archive(folder)
        except FetchError as exc:
            assert "save it as" in str(exc)  # the message tells how to download manually
        else:
            raise AssertionError("FetchError was not raised")


# --- model -----------------------------------------------------------------

def test_classifier_finds_the_right_author() -> None:
    styles = make_styles()
    model = DeltaClassifier.from_chunks(make_chunks(), top_words=30)
    assert sorted(model.authors) == ["Alice", "Bob"]
    assert len(model.vocabulary) == 30

    for author, probabilities in styles.items():
        unknown = synthetic_words(probabilities, 400, seed=999)
        ranking = model.rank(unknown)
        assert ranking.index[0] == author
        assert ranking.iloc[0] < ranking.iloc[1]


def test_classifier_rejects_bad_usage() -> None:
    with pytest.raises(RuntimeError):
        DeltaClassifier().rank(["word"] * 100)
    only_alice = [c for c in make_chunks() if c.author == "Alice"]
    with pytest.raises(InsufficientTextError):
        DeltaClassifier().fit(only_alice)
    model = DeltaClassifier.from_chunks(make_chunks(), top_words=30)
    with pytest.raises(InsufficientTextError):
        model.rank(["wa"] * 10)


def test_evaluate_leave_one_book_out() -> None:
    accuracy = evaluate(make_chunks(), top_words=30)
    assert set(accuracy.index) == {"Alice", "Bob"}
    assert accuracy.min() > 0.9


def test_evaluate_needs_two_books_per_author() -> None:
    with pytest.raises(InsufficientTextError):
        evaluate(make_chunks(books_per_author=1), top_words=30)


# --- config ----------------------------------------------------------------

def _load(text: str):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)


def test_config_valid() -> None:
    config = _load("authors:\n  A: [a1.txt, a2.txt]\n  B: [b1.txt]\nchunk_words: 500\n")
    assert config.authors == {"A": ("a1.txt", "a2.txt"), "B": ("b1.txt",)}
    assert config.chunk_words == 500
    assert config.top_words == 150  # default


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ConfigError):
        _load("authors:\n  A: [a.txt]\n")  # only one author
    with pytest.raises(ConfigError):
        _load("authors:\n  A: [a.txt]\n  B: []\n")
    with pytest.raises(ConfigError):
        _load("authors:\n  A: [notatxtfile]\n  B: [b.txt]\n")
    with pytest.raises(ConfigError):
        _load("authors:\n  A: [a.txt]\n  B: [b.txt]\nchunk_words: 10\n")
