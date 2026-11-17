"""Gemini client setup and shared call helpers."""

from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

from google import genai

DEFAULT_MODEL = os.environ.get("CREATIVE_ANALYTICS_MODEL", "gemini-3.8-flash")

T = TypeVar("T")


def make_client() -> genai.Client:
    """Create a client from GEMINI_API_KEY (or GOOGLE_API_KEY).

    The key is never read from code. In Colab, store it in Secrets and export
    it to the environment before calling this (see the quickstart notebook).
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the GEMINI_API_KEY environment variable before running."
        )
    return genai.Client(api_key=api_key)


def with_retries(fn: Callable[[], T], attempts: int = 3, base_delay: float = 4.0) -> T:
    """Run fn, retrying with exponential backoff on any exception."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(base_delay * 2 ** (attempt - 1))
    raise AssertionError("unreachable")
