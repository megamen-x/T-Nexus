"""
Base contract for chat-oriented LLM clients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from src.t_nexus.ml.config.schema import LLMSettings


class BaseLLM(ABC):
    """Abstract helper for invoking chat models."""

    def __init__(self, settings: LLMSettings) -> None:
        """Persist settings for subclasses."""
        self.settings = settings

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """
        Run the chat completion call.

        :param messages: OpenAI-style sequence of messages.
        :return: Pair of (response text, metadata dict).
        """
