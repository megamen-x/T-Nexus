"""
Conversation utilities for HippoRAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.t_nexus.ml.config.schema import ConversationSettings
from src.t_nexus.ml.utils import QueryBundle


@dataclass
class ChatTurn:
    """Single chat turn."""

    role: str
    content: str


@dataclass
class ChatHistory:
    """Mutable conversation container."""

    turns: List[ChatTurn] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        """Append a chat turn."""
        self.turns.append(ChatTurn(role=role, content=content))

    def last_user_message(self) -> Optional[str]:
        """Return the latest user utterance if available."""
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None

    def as_strings(self) -> List[str]:
        """Render the turns as readable strings."""
        return [f"{turn.role}: {turn.content}" for turn in self.turns]


class ConversationReducer:
    """
    Reduce chat histories into a single query string.
    """

    def __init__(self, settings: ConversationSettings) -> None:
        """Persist conversation settings."""
        self.settings = settings

    def build_query(
        self, history: ChatHistory, override_mode: Optional[str] = None
    ) -> QueryBundle:
        """
        Build a :class:`QueryBundle` according to the configured mode.
        """
        mode = (override_mode or self.settings.mode).lower()
        if mode == "full_history":
            context = history.as_strings()[-self.settings.max_messages :]
            question = context[-1] if context else ""
            return QueryBundle(question=question, history=context)
        # Default to only using the last user message
        question = history.last_user_message() or ""
        return QueryBundle(question=question, history=[])
