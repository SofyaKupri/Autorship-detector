from __future__ import annotations


class AuthorshipError(Exception):
    """base class for all errors"""


class ConfigError(AuthorshipError):
    """the configuration file is missing or invalid"""


class FetchError(AuthorshipError):
    """a book cannot be downloaded or saved"""


class ParseError(AuthorshipError):
    """a downloaded book does not have the expected structure"""


class InsufficientTextError(AuthorshipError):
    """too little text for a reliable result"""
