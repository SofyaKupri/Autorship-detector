from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from authorship.books import load_chunks
from authorship.config import load_config
from authorship.exceptions import AuthorshipError
from authorship.model import DeltaClassifier, evaluate
from authorship.report import format_accuracy, format_ranking
from authorship.text import tokenize

logger = logging.getLogger("authorship")


def read_unknown_text(args: argparse.Namespace) -> str | None:
    """Return the text to classify from --text or a file, or None if not given."""
    if args.text:
        return args.text
    if args.file:
        try:
            return Path(args.file).read_text(encoding="utf-8")
        except OSError as exc:
            raise AuthorshipError(f"cannot read {args.file}: {exc}") from exc
    return None


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    unknown = read_unknown_text(args)

    chunks = load_chunks(config.authors, config.chunk_words, config.cache_dir)

    if args.evaluate:
        print(format_accuracy(evaluate(chunks, config.top_words)))
        print()

    if unknown is not None:
        words = tokenize(unknown)
        model = DeltaClassifier.from_chunks(chunks, config.top_words)
        print(format_ranking(model.rank(words), len(words)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Guess the author of a text by its style.")
    parser.add_argument("file", nargs="?", help="text file to classify")
    parser.add_argument("--text", help="text to classify, given directly")
    parser.add_argument("--evaluate", action="store_true", help="measure accuracy on the books")
    parser.add_argument("--config", default="config.yaml", help="path to YAML config")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    if not (args.file or args.text or args.evaluate):
        parser.error("give a text file, --text, or --evaluate")

    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    try:
        return run(args)
    except AuthorshipError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
