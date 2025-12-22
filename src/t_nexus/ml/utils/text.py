"""
Utility functions for text handling.
"""

from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np


_WHITESPACE_RGX = re.compile(r"\s+")
_HEADING_RGX = re.compile(r"^#{1,3}\s+", re.MULTILINE)


def normalize_text(value: str) -> str:
    """
    Lower-case and strip punctuation from a text snippet.

    :param value: Raw text.
    """
    return re.sub(r"[^a-z0-9 ]+", " ", value.lower()).strip()


def chunk_text(text: str, max_words: int = 600) -> List[str]:
    """
    Split *text* into roughly-equal chunks limited by *max_words*.
    """
    tokens = _WHITESPACE_RGX.split(text.strip())
    chunk = []
    chunks: List[str] = []
    for token in tokens:
        chunk.append(token)
        if len(chunk) >= max_words:
            chunks.append(" ".join(chunk).strip())
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk).strip())
    return chunks


def min_max_normalize(array: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to the provided array.

    :param array: NumPy vector or matrix.
    :return: Normalized values. Returns ones if the array is constant.
    """
    if array.size == 0:
        return array
    min_val = float(array.min())
    max_val = float(array.max())
    if np.isclose(max_val, min_val):
        return np.ones_like(array)
    return (array - min_val) / (max_val - min_val)
