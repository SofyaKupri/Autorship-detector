from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from authorship.exceptions import ConfigError


@dataclass(frozen=True)
class Config:
    authors: dict[str, tuple[str, ...]]
    chunk_words: int = 2000
    top_words: int = 150
    cache_dir: str = "books"


def load_config(path: str | Path) -> Config:
    path = Path(path)
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    config = Config(
        authors=_parse_authors(raw.get("authors")),
        chunk_words=int(raw.get("chunk_words", Config.chunk_words)),
        top_words=int(raw.get("top_words", Config.top_words)),
        cache_dir=str(raw.get("cache_dir", Config.cache_dir)),
    )
    if config.chunk_words < 200:
        raise ConfigError("chunk_words must be at least 200")
    if not 10 <= config.top_words <= 1000:
        raise ConfigError("top_words must be between 10 and 1000")
    return config


def _parse_authors(raw_authors: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw_authors, dict) or len(raw_authors) < 2:
        raise ConfigError("'authors' must be a mapping with at least two authors")

    authors: dict[str, tuple[str, ...]] = {}
    for name, book_ids in raw_authors.items():
        if not isinstance(book_ids, list) or not book_ids:
            raise ConfigError(f"author {name!r} needs a non-empty list of book file names")
        if not all(isinstance(b, str) and b.endswith(".txt") for b in book_ids):
            raise ConfigError(f"author {name!r}: book names must be strings ending with .txt")
        authors[str(name)] = tuple(book_ids)
    return authors
