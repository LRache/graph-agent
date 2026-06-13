"""Shared LLM provider abstractions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from graph_agent.message import Message

if TYPE_CHECKING:
    from graph_agent.tool import ToolSchema


SystemPrompt = str | None


class LLMProvider(Protocol):
    async def generate(
        self,
        messages: list[Message],
        *,
        tools: Iterable[ToolSchema] | None = None,
        **response_options: Any,
    ) -> Message:
        raise NotImplementedError
