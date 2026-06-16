"""OpenAI provider conversion helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from openai.types.responses import Response

from graph_agent.message import (
    ContentBlock,
    ContentBlockKind,
    Message,
    MessageRole,
    ToolCallBlock,
    ToolResultBlock,
)


JsonObject = dict[str, Any]
OpenAIInputItem = dict[str, Any]
OpenAIChatMessage = dict[str, Any]
logger = logging.getLogger(__name__)


def messages_to_openai_input(messages: list[Message]) -> list[OpenAIInputItem]:
    items: list[OpenAIInputItem] = []
    for message in messages:
        items.extend(message_to_openai_input_items(message))
    return items


def messages_to_openai_chat_messages(
    messages: list[Message],
) -> list[OpenAIChatMessage]:
    chat_messages: list[OpenAIChatMessage] = []
    for message in messages:
        chat_messages.extend(message_to_openai_chat_messages(message))
    return chat_messages


def response_options_to_openai_response_options(
    response_options: Mapping[str, Any],
) -> JsonObject:
    options = dict(response_options)
    if "tools" in options:
        options["tools"] = tools_to_openai_response_tools(options["tools"])
    return options


def response_options_to_openai_chat_options(
    response_options: Mapping[str, Any],
) -> JsonObject:
    options = dict(response_options)
    if "tools" in options:
        options["tools"] = tools_to_openai_chat_tools(options["tools"])
    return options


def message_to_openai_input_items(message: Message) -> list[OpenAIInputItem]:
    items = openai_reasoning_items_from_message_extra(message)
    block_items: list[OpenAIInputItem] = []
    text_parts: list[str] = []

    if message.role == MessageRole.TOOL:
        for block in message.blocks:
            if block.kind != ContentBlockKind.TOOL_RESULT:
                raise ValueError(
                    "tool role messages must contain only function tool result blocks"
                )

    for block in message.blocks:
        if block.kind == ContentBlockKind.TOOL_CALL:
            if not isinstance(block, ToolCallBlock):
                raise TypeError("expected ToolCallBlock for tool call block kind")
            block_items.append(tool_call_to_openai_item(block))
            continue

        if block.kind == ContentBlockKind.TOOL_RESULT:
            if not isinstance(block, ToolResultBlock):
                raise TypeError("expected ToolResultBlock for tool result block kind")
            block_items.append(tool_result_to_openai_item(block))
            continue

        if block.kind == ContentBlockKind.REASONING:
            # ReasoningBlock is a visible summary; OpenAI reasoning replay lives in Message.extra.
            logger.warning(
                "Ignoring ReasoningBlock while converting message to OpenAI input",
                extra={
                    "provider": "openai",
                    "message_role": message.role.value,
                },
            )
            continue

        text = block_to_input_text(block)
        if text:
            text_parts.append(text)

    if text_parts:
        items.append(
            {
                "type": "message",
                "role": message.role.value,
                "content": "\n".join(text_parts),
            },
        )

    items.extend(block_items)
    return items


def message_to_openai_chat_messages(message: Message) -> list[OpenAIChatMessage]:
    if openai_reasoning_items_from_message_extra(message):
        raise ValueError(
            "Chat Completions API does not support OpenAI reasoning replay"
        )

    if message.role == MessageRole.TOOL:
        messages: list[OpenAIChatMessage] = []
        for block in message.blocks:
            if block.kind != ContentBlockKind.TOOL_RESULT:
                raise ValueError(
                    "tool role messages must contain only function tool result blocks"
                )
            if not isinstance(block, ToolResultBlock):
                raise TypeError("expected ToolResultBlock for tool result block kind")
            messages.append(tool_result_to_openai_chat_message(block))
        return messages

    text_parts: list[str] = []
    tool_calls: list[OpenAIChatMessage] = []

    for block in message.blocks:
        if block.kind == ContentBlockKind.TOOL_CALL:
            if not isinstance(block, ToolCallBlock):
                raise TypeError("expected ToolCallBlock for tool call block kind")
            tool_calls.append(tool_call_to_openai_chat_tool_call(block))
            continue

        if block.kind == ContentBlockKind.TOOL_RESULT:
            raise ValueError(
                "non-tool role messages cannot contain function tool result blocks"
            )

        if block.kind == ContentBlockKind.REASONING:
            logger.warning(
                "Ignoring ReasoningBlock while converting message "
                "to OpenAI chat message",
                extra={
                    "provider": "openai",
                    "message_role": message.role.value,
                },
            )
            continue

        text = block_to_input_text(block)
        if text:
            text_parts.append(text)

    text_content = "\n".join(text_parts)
    chat_message: OpenAIChatMessage = {
        "role": openai_chat_role(message.role),
        "content": text_content,
    }
    if tool_calls:
        chat_message["tool_calls"] = tool_calls
        if not text_content:
            chat_message["content"] = None
    return [chat_message]


def message_from_openai_response(
    response: Response,
    *,
    model: str | None = None,
) -> Message:
    blocks: list[ContentBlock] = []
    reasoning_items: list[OpenAIInputItem] = []
    for item in response.output:
        if item.type == "message":
            blocks.extend(message_output_to_blocks(item))
        elif item.type == "function_call":
            blocks.append(ContentBlock.tool_call(function_call_from_output_item(item)))
        elif item.type == "reasoning":
            reasoning_items.append(reasoning_item_to_openai_input_item(item))
            reasoning_text = reasoning_text_from_item(item)
            if reasoning_text:
                blocks.append(ContentBlock.reasoning(reasoning_text))
        else:
            raise ValueError(f"unsupported OpenAI response output item: {item.type}")

    return Message(
        MessageRole.ASSISTANT,
        tuple(blocks),
        response_meta=response_meta(response, model=model),
        extra=openai_extra(reasoning_items=reasoning_items),
    )


def message_from_openai_chat_completion(
    response: Any,
    *,
    model: str | None = None,
) -> Message:
    choices = field_value(response, "choices", [])
    if not choices:
        raise ValueError("OpenAI chat completion response did not include choices")

    choice = choices[0]
    message = field_value(choice, "message")

    blocks: list[ContentBlock] = []
    blocks.extend(
        chat_message_content_to_blocks(field_value(message, "content", None))
    )

    refusal = field_value(message, "refusal", None)
    if refusal:
        blocks.append(ContentBlock.text_block(refusal))

    for tool_call in field_value(message, "tool_calls", None) or []:
        blocks.append(
            ContentBlock.tool_call(function_call_from_chat_tool_call(tool_call))
        )

    legacy_function_call = field_value(message, "function_call", None)
    if legacy_function_call:
        blocks.append(
            ContentBlock.tool_call(
                function_call_from_legacy_chat_function_call(legacy_function_call)
            )
        )

    return Message(
        MessageRole.ASSISTANT,
        tuple(blocks),
        response_meta=chat_completion_response_meta(response, choice, model=model),
    )


def openai_extra(*, reasoning_items: list[OpenAIInputItem]) -> JsonObject:
    if not reasoning_items:
        return {}
    return {"openai": {"reasoning_items": reasoning_items}}


def openai_reasoning_items_from_message_extra(message: Message) -> list[OpenAIInputItem]:
    openai = message.extra.get("openai")
    if openai is None:
        return []
    if not isinstance(openai, Mapping):
        raise TypeError("message.extra['openai'] must be a mapping")

    reasoning_items = openai.get("reasoning_items", [])
    if reasoning_items is None:
        return []
    if not isinstance(reasoning_items, (list, tuple)):
        raise TypeError("message.extra['openai']['reasoning_items'] must be a list")

    return [reasoning_item_to_openai_input_item(item) for item in reasoning_items]


def reasoning_item_to_openai_input_item(item: Any) -> OpenAIInputItem:
    data = plain_data(item)
    if not isinstance(data, Mapping):
        raise TypeError("OpenAI reasoning item must be a mapping")
    if data.get("type") != "reasoning":
        raise ValueError("OpenAI reasoning item type must be reasoning")
    return {
        str(key): value
        for key, value in data.items()
        if value is not None
    }


def tool_call_to_openai_item(call: ToolCallBlock) -> OpenAIInputItem:
    return {
        "type": "function_call",
        "call_id": call.call_id,
        "name": call.tool_name,
        "arguments": json.dumps(call.arguments),
        "status": "completed",
    }


def tool_result_to_openai_item(result: ToolResultBlock) -> OpenAIInputItem:
    return {
        "type": "function_call_output",
        "call_id": result.call_id,
        "output": tool_result_output(result),
        "status": "completed",
    }


def tool_call_to_openai_chat_tool_call(call: ToolCallBlock) -> OpenAIChatMessage:
    return {
        "id": call.call_id,
        "type": "function",
        "function": {
            "name": call.tool_name,
            "arguments": json.dumps(call.arguments),
        },
    }


def tool_result_to_openai_chat_message(result: ToolResultBlock) -> OpenAIChatMessage:
    return {
        "role": "tool",
        "tool_call_id": result.call_id,
        "content": tool_result_output(result),
    }


def tool_result_output(result: ToolResultBlock) -> str:
    if not result.is_error:
        return result.content
    return json.dumps({"error": result.content or "tool failed"})


def tools_to_openai_response_tools(tools: Any) -> list[JsonObject]:
    return [tool_to_openai_response_tool(tool) for tool in tool_items(tools)]


def tools_to_openai_chat_tools(tools: Any) -> list[JsonObject]:
    return [tool_to_openai_chat_tool(tool) for tool in tool_items(tools)]


def tool_items(tools: Any) -> list[Any]:
    if tools is None:
        return []
    if isinstance(tools, Mapping):
        return [tools]
    if isinstance(tools, (str, bytes)):
        raise TypeError("OpenAI tools must be an iterable of tool schemas")

    try:
        return list(tools)
    except TypeError:
        return [tools]


def tool_to_openai_response_tool(tool: Any) -> JsonObject:
    if isinstance(tool, Mapping):
        tool_data = {str(key): plain_data(value) for key, value in tool.items()}
        if "type" in tool_data:
            return tool_data
        return openai_function_tool_from_schema_data(tool_data)

    return openai_function_tool_from_schema_data(
        {
            "name": field_value(tool, "name"),
            "description": field_value(tool, "description", ""),
            "parameters": plain_data(field_value(tool, "parameters", {})),
        }
    )


def tool_to_openai_chat_tool(tool: Any) -> JsonObject:
    response_tool = tool_to_openai_response_tool(tool)
    if response_tool.get("type") != "function":
        return response_tool
    if "function" in response_tool:
        return response_tool

    function = {
        "name": response_tool["name"],
        "description": response_tool.get("description", ""),
        "parameters": response_tool.get("parameters", {}),
    }
    for key, value in response_tool.items():
        if key not in {"type", "name", "description", "parameters"}:
            function[key] = value
    return {"type": "function", "function": function}


def openai_function_tool_from_schema_data(schema: Mapping[str, Any]) -> JsonObject:
    tool = {
        "type": "function",
        "name": schema["name"],
        "description": schema.get("description", ""),
        "parameters": schema.get("parameters", {}),
    }
    for key, value in schema.items():
        if key not in {"name", "description", "parameters"}:
            tool[str(key)] = value
    return tool


def block_to_input_text(block: ContentBlock) -> str:
    if block.kind == ContentBlockKind.FILE:
        raise ValueError("OpenAIProvider does not support file blocks yet")
    text = block.text()
    if text is not None:
        return text
    raise ValueError(f"unsupported OpenAI input block kind: {block.kind.value}")


def message_output_to_blocks(message: Any) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    for content in message.content:
        if content.type == "output_text":
            blocks.append(ContentBlock.text_block(content.text))
        elif content.type == "refusal":
            blocks.append(ContentBlock.text_block(content.refusal))
        else:
            raise ValueError(f"unsupported OpenAI message content item: {content.type}")
    return blocks


def chat_message_content_to_blocks(content: Any) -> list[ContentBlock]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [ContentBlock.text_block(content)]
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            item_type = field_value(item, "type", None)
            if item_type in {"text", "output_text"}:
                text = field_value(item, "text", "")
                if text:
                    parts.append(text)
                continue
            if item_type == "refusal":
                refusal = field_value(item, "refusal", "")
                if refusal:
                    parts.append(refusal)
                continue
            raise ValueError(f"unsupported OpenAI chat content item: {item_type}")
        if parts:
            return [ContentBlock.text_block("\n".join(parts))]
        return []
    raise ValueError(
        f"unsupported OpenAI chat message content: {type(content).__name__}"
    )


def function_call_from_output_item(item: Any) -> ToolCallBlock:
    if item.type != "function_call":
        raise ValueError(f"unsupported OpenAI tool call item: {item.type}")
    return ToolCallBlock(
        call_id=item.call_id,
        tool_name=item.name,
        arguments=parse_arguments(item.arguments),
    )


def function_call_from_chat_tool_call(item: Any) -> ToolCallBlock:
    item_type = field_value(item, "type", "function")
    if item_type != "function":
        raise ValueError(f"unsupported OpenAI chat tool call item: {item_type}")

    function = field_value(item, "function")
    return ToolCallBlock(
        call_id=field_value(item, "id"),
        tool_name=field_value(function, "name"),
        arguments=parse_arguments(field_value(function, "arguments")),
    )


def function_call_from_legacy_chat_function_call(item: Any) -> ToolCallBlock:
    name = field_value(item, "name")
    return ToolCallBlock(
        call_id=f"function_call:{name}",
        tool_name=name,
        arguments=parse_arguments(field_value(item, "arguments")),
    )


def parse_arguments(raw_arguments: Any) -> JsonObject:
    if not isinstance(raw_arguments, str):
        raise TypeError("OpenAI function call arguments must be a JSON object string")
    try:
        value = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenAI function call arguments must be valid JSON") from exc
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("OpenAI function call arguments must be a JSON object")


def reasoning_text_from_item(item: Any) -> str:
    parts: list[str] = []
    for summary in item.summary:
        if summary.text:
            parts.append(summary.text)
    return "\n".join(parts)


def response_meta(response: Response, *, model: str | None = None) -> JsonObject:
    return {
        "provider": "openai",
        "id": response.id,
        "model": response.model or model,
        "status": response.status,
        "usage": plain_data(response.usage),
    }


def chat_completion_response_meta(
    response: Any,
    choice: Any,
    *,
    model: str | None = None,
) -> JsonObject:
    return {
        "provider": "openai",
        "api": "chat_completions",
        "id": field_value(response, "id", None),
        "model": field_value(response, "model", None) or model,
        "finish_reason": field_value(choice, "finish_reason", None),
        "usage": plain_data(field_value(response, "usage", None)),
    }


def openai_chat_role(role: MessageRole) -> str:
    if role == MessageRole.DEVELOPER:
        return MessageRole.SYSTEM.value
    return role.value


_MISSING = object()


def field_value(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping) and name in value:
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    if default is not _MISSING:
        return default
    raise ValueError(f"OpenAI response object is missing field: {name}")


def plain_data(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return plain_data(value.model_dump())
    raise TypeError(f"unsupported OpenAI metadata value: {type(value).__name__}")


def _expect_block(block: ContentBlock, expected_type: type[Any]) -> Any:
    if not isinstance(block, expected_type):
        raise TypeError(f"expected {expected_type.__name__} block for {block.kind}")
    return block
