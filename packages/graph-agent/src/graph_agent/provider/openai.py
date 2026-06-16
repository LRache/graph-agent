"""OpenAI provider for graph-agent messages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from graph_agent.message import Message

from .convert import (
    JsonObject,
    message_from_openai_chat_completion,
    message_from_openai_response,
    messages_to_openai_chat_messages,
    messages_to_openai_input,
    response_options_to_openai_chat_options,
    response_options_to_openai_response_options,
)

if TYPE_CHECKING:
    from graph_agent.tool import ToolSchema


class OpenAIProvider:
    """LLM provider backed by the OpenAI Python SDK."""

    provider_name = "openai"

    def __init__(
        self,
        model: str,
        client: AsyncOpenAI | None = None,
        *,
        api: str = "responses",
        tools: Iterable[ToolSchema] | None = None,
        **response_options: Any,
    ) -> None:
        self.model = model
        self.client = client if client is not None else AsyncOpenAI()
        self.api = normalize_api(api)
        self.tools = tuple(tools or ())
        self._response_options = dict(response_options)

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: Iterable[ToolSchema] | None = None,
        **response_options: Any,
    ) -> Message:
        if not isinstance(messages, list):
            raise TypeError("messages must be a list[Message]")

        options = {**self._response_options, **response_options}
        active_tools = tuple(tools) if tools is not None else self.tools
        if active_tools:
            options["tools"] = active_tools

        if self.api == "chat_completions":
            return await self._generate_chat_completion(messages, **options)

        return await self._generate_response(messages, **options)

    async def _generate_response(
        self,
        messages: list[Message],
        **response_options: Any,
    ) -> Message:
        options = response_options_to_openai_response_options(response_options)
        request: JsonObject = {
            "model": self.model,
            "input": messages_to_openai_input(messages),
            **options,
        }

        response = await self.client.responses.create(**request)
        return message_from_openai_response(response, model=self.model)

    async def _generate_chat_completion(
        self,
        messages: list[Message],
        **response_options: Any,
    ) -> Message:
        options = response_options_to_openai_chat_options(response_options)
        request: JsonObject = {
            "model": self.model,
            "messages": messages_to_openai_chat_messages(messages),
            **options,
        }

        response = await self.client.chat.completions.create(**request)
        return message_from_openai_chat_completion(response, model=self.model)


def normalize_api(api: str) -> str:
    if api in {"responses", "response"}:
        return "responses"
    if api in {"chat", "chat_completion", "chat_completions", "chat.completions"}:
        return "chat_completions"
    raise ValueError("api must be 'responses' or 'chat_completions'")
