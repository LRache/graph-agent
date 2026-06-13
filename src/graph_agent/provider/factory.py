"""Factories for prompt-specialized LLM nodes."""

from __future__ import annotations

from typing import Any, cast

from .base import LLMProvider, SystemPrompt
from .node import LLMNode


class LLMNodeFactory:
    """Factory for creating prompt-specialized LLM node classes."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        node_cls: type[LLMNode] = LLMNode,
    ) -> None:
        self.provider = provider
        self.node_cls = node_cls

    def create_node_class(
        self,
        class_name: str,
        system_prompt: SystemPrompt = None,
        *,
        provider: LLMProvider | None = None,
    ) -> type[LLMNode]:
        node_cls = self.node_cls
        node_provider = provider if provider is not None else self.provider

        def __init__(
            self: LLMNode,
            name: str,
        ) -> None:
            node_cls.__init__(
                self,
                name,
                node_provider,
                system_prompt=system_prompt,
            )

        namespace: dict[str, Any] = {
            "__init__": __init__,
            "__module__": __name__,
        }
        return cast(type[LLMNode], type(class_name, (node_cls,), namespace))
