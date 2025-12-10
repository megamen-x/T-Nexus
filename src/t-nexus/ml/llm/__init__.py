"""
Minimal LLM abstractions.
"""

from src.ml.llm.base import BaseLLM
from src.ml.llm.openai import OpenAILLM

__all__ = ["BaseLLM", "OpenAILLM"]
