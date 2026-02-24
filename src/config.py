"""Application-level settings loaded from environment variables."""

from __future__ import annotations

import os
from typing import List, Optional


class Settings:
    """Top-level application settings derived from environment variables.

    All values are read fresh from the environment on each instantiation,
    which ensures that test monkeypatching of environment variables is
    picked up without caching concerns.
    """

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("API_KEY") or None
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        raw_origins = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins: List[str] = [o.strip() for o in raw_origins.split(",")]


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` instance.

    A new object is created on every call so that changes to environment
    variables (e.g. those made by pytest monkeypatch) are always reflected.
    """
    return Settings()


__all__ = ["Settings", "get_settings"]
