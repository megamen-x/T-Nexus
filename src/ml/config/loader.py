"""
YAML loader for :class:`HippoRAGSettings`.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.ml.config.schema import HippoRAGSettings


def _read_yaml(path: Path) -> Dict[str, Any]:
    """
    Read a YAML file from disk and return the parsed dictionary.

    :param path: Location of the YAML document.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file '{path}' is missing.") from exc


def load_settings(path: str | os.PathLike[str]) -> HippoRAGSettings:
    """
    Load HippoRAG settings from a YAML file.

    :param path: Path-like pointing to a YAML file.
    :return: Configured :class:`HippoRAGSettings`.
    """
    raw = _read_yaml(Path(path))
    return HippoRAGSettings.from_dict(raw)
