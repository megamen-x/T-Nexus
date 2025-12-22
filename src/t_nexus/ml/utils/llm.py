"""
LLM-oriented helper utilities (prompt conversions, JSON fixes, etc.).
"""

from __future__ import annotations

import re
from typing import List, Sequence


def convert_format_to_template(
    original_string: str,
    placeholder_mapping: dict[str, str] | None = None,
    static_values: dict[str, str] | None = None,
) -> str:
    """
    Convert ``str.format`` placeholders into ``string.Template`` style ones.
    """
    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}
    pattern = re.compile(r"\{(\w+)\}")

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in static_values:
            return str(static_values[key])
        return f"${{{placeholder_mapping.get(key, key)}}}"

    return pattern.sub(_replace, original_string)


def filter_invalid_triples(triples: Sequence[Sequence[str]]) -> List[List[str]]:
    """
    Remove invalid triples (size != 3) and deduplicate results.
    """
    seen: set[tuple[str, str, str]] = set()
    valid: List[List[str]] = []
    for triple in triples:
        if len(triple) != 3:
            continue
        normalized = tuple(str(x) for x in triple)
        if normalized not in seen:
            seen.add(normalized)
            valid.append(list(normalized))
    return valid
