"""
Prompt template loader compatible with OpenAI-style chat completions.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from string import Template
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplateManager:
    """
    Loads python files from ``prompts/templates`` and exposes them via ``render``.

    Each module must define ``prompt_template`` as either a ``string.Template``,
    a raw string (auto-converted into a ``Template``), or a list of chat messages
    ``[{'role': 'system', 'content': Template(...)}]``.
    """

    role_mapping: Dict[str, str] = field(
        default_factory=lambda: {"system": "system", "user": "user", "assistant": "assistant"}
    )
    templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        current_file_path = os.path.abspath(__file__)
        self.templates_dir = os.path.join(os.path.dirname(current_file_path), "templates")
        self._load_templates()

    def _load_templates(self) -> None:
        """Import template modules and register their prompt definitions."""
        if not os.path.exists(self.templates_dir):
            raise FileNotFoundError(f"Templates directory '{self.templates_dir}' does not exist.")

        package_root = __package__ or "src.ml.prompts"

        for filename in os.listdir(self.templates_dir):
            if not (filename.endswith(".py") and filename != "__init__.py"):
                continue

            script_name = os.path.splitext(filename)[0]
            module_name = f"{package_root}.templates.{script_name}"
            module = importlib.import_module(module_name)

            if not hasattr(module, "prompt_template"):
                raise AttributeError(f"Module '{module_name}' lacks `prompt_template`.")

            prompt_template = module.prompt_template
            logger.debug("Loaded template '%s' from %s", script_name, module_name)

            if isinstance(prompt_template, Template):
                self.templates[script_name] = prompt_template
            elif isinstance(prompt_template, str):
                self.templates[script_name] = Template(prompt_template)
            elif isinstance(prompt_template, list) and all(
                isinstance(item, dict) and {"role", "content"} <= item.keys()
                for item in prompt_template
            ):
                for item in prompt_template:
                    item["role"] = self.role_mapping.get(item["role"], item["role"])
                    if not isinstance(item["content"], Template):
                        item["content"] = Template(item["content"])
                self.templates[script_name] = prompt_template
            else:
                raise TypeError(
                    f"Invalid `prompt_template` in '{module_name}'. Expected Template, str or chat-history list."
                )

    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """
        Render a template by substituting keyword arguments.
        """
        template = self.get_template(name)
        if isinstance(template, Template):
            return template.substitute(**kwargs)
        return [{"role": item["role"], "content": item["content"].substitute(**kwargs)} for item in template]

    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        """Return the raw template object."""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found.")
        return self.templates[name]

    def list_template_names(self) -> List[str]:
        """Return all loaded template names."""
        return list(self.templates.keys())

    def is_template_name_valid(self, name: str) -> bool:
        """Check whether a template name exists."""
        return name in self.templates
