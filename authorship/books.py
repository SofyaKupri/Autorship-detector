from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from authorship.exceptions import FetchError, InsufficientTextError, ParseError
from authorship.text import split_into_chunks, tokenize

logger = logging.getLogger(__name__)

# One zip file with 18 classic books
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"
)
ARCHIVE_NAME = "gutenberg.zip"
USER_AGENT = "authorship-detector/1.0 (student project)"
ATTEMPTS = 3


@dataclass(frozen=True)
class Chunk:
    author: str
    book: str
    words: tuple[str, ...]


def _fetch_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            # 429 (too many requests) and 5xx are temporary: wait and retry.
            temporary = exc.code == 429 or exc.code >= 500
            if not temporary or attempt == ATTEMPTS:
                raise
            logger.warning("HTTP %d from %s, retrying (%d)", exc.code, url, attempt)
            time.sleep(3 * attempt)
        except TimeoutError:
            if attempt == ATTEMPTS:
                raise
            time.sleep(3 * attempt)
    raise OSError(f"no attempts made for {url}")


def get_archive(cache_dir: str | Path, timeout: float = 60.0) -> Path:
    path = Path(cache_dir) / ARCHIVE_NAME
    if path.exists():
        return path

    logger.info("Downloading the book archive (a few MB), this happens only once ...")
    try:
        # URLError, HTTPError and timeouts are all subclasses of OSError.
        raw = _fetch_bytes(ARCHIVE_URL, timeout)
    except OSError as exc:
        raise FetchError(
            f"cannot download {ARCHIVE_URL}: {exc}. "
            f"You can download it in a browser and save it as {path}"
        ) from exc
    if not raw:
        raise FetchError("the downloaded archive is empty")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    except OSError as exc:
        raise FetchError(f"cannot save the archive to {path}: {exc}") from exc
    return path


def read_book(archive: zipfile.ZipFile, name: str) -> str:
    suffix = "/" + name
    for member in archive.namelist():
        if member == name or member.endswith(suffix):
            return archive.read(member).decode("latin-1")  # the encoding used by NLTK

    available = sorted(m.rsplit("/", 1)[-1] for m in archive.namelist() if m.endswith(".txt"))
    raise ParseError(f"book {name!r} is not in the archive; available: {', '.join(available)}")


def load_chunks(
    authors: Mapping[str, Sequence[str]],
    chunk_words: int,
    cache_dir: str | Path,
) -> list[Chunk]:
    archive_path = get_archive(cache_dir)
    chunks: list[Chunk] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for author, names in authors.items():
                for name in names:
                    pieces = split_into_chunks(tokenize(read_book(archive, name)), chunk_words)
                    if not pieces:
                        logger.warning("Book %s is shorter than one chunk, skipped", name)
                    chunks.extend(Chunk(author, name, tuple(piece)) for piece in pieces)
    except zipfile.BadZipFile as exc:
        raise FetchError(f"{archive_path} is not a valid zip file; delete it and retry") from exc

    if len({chunk.author for chunk in chunks}) < 2:
        raise InsufficientTextError("need usable text from at least two authors")
    logger.info("Prepared %d chunks", len(chunks))
    return chunks
