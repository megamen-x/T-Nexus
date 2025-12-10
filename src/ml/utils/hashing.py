"""
Hash helpers.
"""

from __future__ import annotations

import uuid



def compute_uuid5(content: str, prefix: str = "") -> str:
    """
    Compute a deterministic UUID5 string and prefix it for namespacing.
    """
    digest = uuid.uuid5(uuid.NAMESPACE_URL, content)
    return f"{prefix}{digest}"
