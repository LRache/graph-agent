"""Typed messages shared by graph nodes, tools, and model adapters."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar
from uuid import UUID, uuid4

from graph_agent.core.utils import (
    _optional_bool,
    _optional_str,
    _required_list,
    _required_str,
)


MESSAGE_HISTORY_SCHEMA = "graph-agent.message-history.v1"


def _new_uuid() -> str:
    return str(uuid4())


def _normalize_uuid(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("message.uuid must be a string")
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise ValueError("message.uuid must be a valid UUID") from exc


def _repr_call(name: str, *args: Any, **kwargs: Any) -> str:
    parts = [repr(arg) for arg in args]
    parts.extend(f"{key}={value!r}" for key, value in kwargs.items())
    return f"{name}({', '.join(parts)})"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentBlockKind(StrEnum):
    TEXT = "text"
    FILE = "file"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


ROLE_ALLOWED_BLOCK_KINDS = {
    MessageRole.ASSISTANT: frozenset(
        {
            ContentBlockKind.TEXT,
            ContentBlockKind.REASONING,
            ContentBlockKind.TOOL_CALL,
        }
    ),
    MessageRole.TOOL: frozenset({ContentBlockKind.TOOL_RESULT}),
    MessageRole.USER: frozenset({ContentBlockKind.TEXT, ContentBlockKind.FILE}),
    MessageRole.SYSTEM: frozenset({ContentBlockKind.TEXT, ContentBlockKind.FILE}),
    MessageRole.DEVELOPER: frozenset({ContentBlockKind.TEXT, ContentBlockKind.FILE}),
}


@dataclass(frozen=True, kw_only=True)
class ContentBlock:
    kind: ClassVar[ContentBlockKind]

    @classmethod
    def reasoning(cls, text: str, *, signature: str | None = None) -> "ReasoningBlock":
        return ReasoningBlock(text, signature=signature)

    @classmethod
    def text_block(cls, text: str) -> "TextBlock":
        return TextBlock(text)

    @classmethod
    def user_text(cls, text: str) -> "TextBlock":
        return cls.text_block(text)

    @classmethod
    def assistant_text(cls, text: str) -> "TextBlock":
        return cls.text_block(text)

    @classmethod
    def file(cls, file_block: "FileBlock") -> "FileBlock":
        return file_block

    @classmethod
    def tool_call(cls, call: "ToolCallBlock") -> "ToolCallBlock":
        return call

    @classmethod
    def tool_result(cls, result: "ToolResultBlock") -> "ToolResultBlock":
        return result

    @classmethod
    def function_tool_call(cls, call: "ToolCallBlock") -> "ToolCallBlock":
        return cls.tool_call(call)

    @classmethod
    def function_tool_result(cls, result: "ToolResultBlock") -> "ToolResultBlock":
        return cls.tool_result(result)

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self, TextBlock):
            return {"kind": self.kind.value, "text": self.text_value}
        if isinstance(self, ReasoningBlock):
            data: dict[str, Any] = {"kind": self.kind.value, "text": self.text_value}
            if self.signature is not None:
                data["signature"] = self.signature
            return data
        if isinstance(self, FileBlock):
            data = {"kind": self.kind.value}
            for key, value in {
                "file_id": self.file_id,
                "path": self.path,
                "mime_type": self.mime_type,
                "name": self.name,
            }.items():
                if value is not None:
                    data[key] = value
            return data
        if isinstance(self, ToolCallBlock):
            return {
                "kind": self.kind.value,
                "call_id": self.call_id,
                "tool_name": self.tool_name,
                "arguments": _json_object(self.arguments, "tool_call.arguments"),
            }
        if isinstance(self, ToolResultBlock):
            return {
                "kind": self.kind.value,
                "call_id": self.call_id,
                "tool_name": self.tool_name,
                "content": self.content,
                "is_error": self.is_error,
            }
        raise TypeError(f"unsupported content block type: {type(self).__name__}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContentBlock":
        if not isinstance(data, Mapping):
            raise TypeError("content block data must be a JSON object")

        kind = ContentBlockKind(_required_str(data, "kind", "block.kind"))
        if kind == ContentBlockKind.TEXT:
            return TextBlock(_required_str(data, "text", "text_block.text"))
        if kind == ContentBlockKind.REASONING:
            return ReasoningBlock(
                _required_str(data, "text", "reasoning_block.text"),
                signature=_optional_str(data, "signature", "reasoning_block.signature"),
            )
        if kind == ContentBlockKind.FILE:
            return FileBlock(
                file_id=_optional_str(data, "file_id", "file_block.file_id"),
                path=_optional_str(data, "path", "file_block.path"),
                mime_type=_optional_str(data, "mime_type", "file_block.mime_type"),
                name=_optional_str(data, "name", "file_block.name"),
            )
        if kind == ContentBlockKind.TOOL_CALL:
            arguments: dict[str, Any] = {}
            if "arguments" in data:
                arguments = _json_object(data["arguments"], "tool_call.arguments")
            return ToolCallBlock(
                _required_str(data, "call_id", "tool_call.call_id"),
                _required_str(data, "tool_name", "tool_call.tool_name"),
                arguments,
            )
        if kind == ContentBlockKind.TOOL_RESULT:
            return ToolResultBlock(
                _required_str(data, "call_id", "tool_result.call_id"),
                _optional_str(data, "tool_name", "tool_result.tool_name"),
                _optional_str(data, "content", "tool_result.content") or "",
                _optional_bool(data, "is_error", "tool_result.is_error") or False,
            )

        raise ValueError(f"unsupported content block kind: {kind.value}")

    def text(self) -> str | None:
        raise NotImplementedError(f"text() not implemented for {type(self).__name__}")


@dataclass(frozen=True)
class TextBlock(ContentBlock):
    text_value: str

    kind: ClassVar[ContentBlockKind] = ContentBlockKind.TEXT

    def __repr__(self) -> str:
        return _repr_call("TextBlock", self.text_value)

    def text(self) -> str:
        return self.text_value


@dataclass(frozen=True)
class ReasoningBlock(ContentBlock):
    text_value: str
    signature: str | None = None

    kind: ClassVar[ContentBlockKind] = ContentBlockKind.REASONING

    def __repr__(self) -> str:
        kwargs: dict[str, Any] = {}
        if self.signature is not None:
            kwargs["signature"] = self.signature
        return _repr_call("ReasoningBlock", self.text_value, **kwargs)

    def text(self) -> str:
        return self.text_value


@dataclass(frozen=True)
class FileBlock(ContentBlock):
    file_id: str | None = None
    path: str | None = None
    mime_type: str | None = None
    name: str | None = None

    kind: ClassVar[ContentBlockKind] = ContentBlockKind.FILE

    def __repr__(self) -> str:
        kwargs = {
            key: value
            for key, value in {
                "file_id": self.file_id,
                "path": self.path,
                "mime_type": self.mime_type,
                "name": self.name,
            }.items()
            if value is not None
        }
        return _repr_call("FileBlock", **kwargs)


@dataclass(frozen=True)
class ToolCallBlock(ContentBlock):
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    kind: ClassVar[ContentBlockKind] = ContentBlockKind.TOOL_CALL

    def __repr__(self) -> str:
        return _repr_call(
            "ToolCallBlock",
            call_id=self.call_id,
            tool_name=self.tool_name,
            arguments=self.arguments,
        )


@dataclass(frozen=True)
class ToolResultBlock(ContentBlock):
    call_id: str
    tool_name: str | None
    content: str = ""
    is_error: bool = False

    kind: ClassVar[ContentBlockKind] = ContentBlockKind.TOOL_RESULT

    def __repr__(self) -> str:
        kwargs: dict[str, Any] = {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
        }
        if self.content:
            kwargs["content"] = self.content
        if self.is_error:
            kwargs["is_error"] = self.is_error
        return _repr_call("ToolResultBlock", **kwargs)

    def text(self) -> str:
        return self.content


@dataclass(frozen=True)
class Message:
    role: MessageRole
    blocks: tuple[ContentBlock, ...]
    response_meta: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    uuid: str = field(default_factory=_new_uuid)

    def __repr__(self) -> str:
        if self.response_meta is None and not self.extra and len(self.blocks) == 1:
            block = self.blocks[0]
            if isinstance(block, TextBlock):
                if self.role == MessageRole.SYSTEM:
                    return _repr_call("Message.system_text", block.text_value)
                if self.role == MessageRole.USER:
                    return _repr_call("Message.user_text", block.text_value)
                if self.role == MessageRole.ASSISTANT:
                    return _repr_call("Message.assistant_text", block.text_value)
                if self.role == MessageRole.DEVELOPER:
                    return _repr_call("Message.developer", block.text_value)
            if self.role == MessageRole.ASSISTANT and isinstance(block, ToolCallBlock):
                args: list[Any] = [block.call_id, block.tool_name]
                if block.arguments:
                    args.append(block.arguments)
                return _repr_call("Message.tool_call", *args)
            if self.role == MessageRole.TOOL and isinstance(block, ToolResultBlock):
                return _repr_call("Message.tool_result", block)

        kwargs: dict[str, Any] = {
            "role": self.role.value,
            "blocks": self.blocks,
            "uuid": self.uuid,
        }
        if self.response_meta is not None:
            kwargs["response_meta"] = self.response_meta
        if self.extra:
            kwargs["extra"] = self.extra
        return _repr_call("Message", **kwargs)

    def __post_init__(self) -> None:
        role = MessageRole(self.role)
        blocks = tuple(self.blocks)
        allowed_kinds = ROLE_ALLOWED_BLOCK_KINDS[role]

        for block in blocks:
            if not isinstance(block, ContentBlock):
                raise TypeError("message blocks must be ContentBlock instances")

            kind = ContentBlockKind(block.kind)
            if kind not in allowed_kinds:
                raise ValueError(
                    f"{role.value} messages cannot contain {kind.value} blocks"
                )

        object.__setattr__(self, "role", role)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "uuid", _normalize_uuid(self.uuid))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role.value,
            "blocks": [block.to_dict() for block in self.blocks],
            "uuid": self.uuid,
        }
        if self.response_meta is not None:
            data["response_meta"] = _json_object(self.response_meta, "response_meta")
        if self.extra:
            data["extra"] = _json_object(self.extra, "extra")
        return data

    def hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Message":
        if not isinstance(data, Mapping):
            raise TypeError("message data must be a JSON object")

        role = _required_str(data, "role", "message.role")
        blocks_data = _required_list(data, "blocks", "message.blocks")
        blocks = [ContentBlock.from_dict(block) for block in blocks_data]

        response_meta = None
        if "response_meta" in data and data["response_meta"] is not None:
            response_meta = _json_object(
                data["response_meta"],
                "message.response_meta",
            )

        extra: dict[str, Any] = {}
        if "extra" in data:
            extra = _json_object(data["extra"], "message.extra")

        kwargs: dict[str, Any] = {}
        if "uuid" in data:
            kwargs["uuid"] = _required_str(data, "uuid", "message.uuid")

        return cls(
            MessageRole(role),
            tuple(blocks),
            response_meta=response_meta,
            extra=extra,
            **kwargs,
        )

    @classmethod
    def of(
        cls,
        role: MessageRole | str,
        blocks: Iterable[ContentBlock],
        *,
        uuid: str | None = None,
    ) -> "Message":
        if uuid is None:
            return cls(MessageRole(role), tuple(blocks))
        return cls(MessageRole(role), tuple(blocks), uuid=uuid)

    @classmethod
    def system_text(cls, text: str, *, uuid: str | None = None) -> "Message":
        if uuid is None:
            return cls(MessageRole.SYSTEM, (ContentBlock.text_block(text),))
        return cls(MessageRole.SYSTEM, (ContentBlock.text_block(text),), uuid=uuid)

    @classmethod
    def user_text(cls, text: str, *, uuid: str | None = None) -> "Message":
        if uuid is None:
            return cls(MessageRole.USER, (ContentBlock.text_block(text),))
        return cls(MessageRole.USER, (ContentBlock.text_block(text),), uuid=uuid)

    @classmethod
    def assistant_text(
        cls,
        text: str,
        *,
        uuid: str | None = None,
        **extra: Any,
    ) -> "Message":
        if uuid is None:
            return cls(
                MessageRole.ASSISTANT,
                (ContentBlock.text_block(text),),
                extra=extra,
            )
        return cls(
            MessageRole.ASSISTANT,
            (ContentBlock.text_block(text),),
            extra=extra,
            uuid=uuid,
        )

    @classmethod
    def developer(cls, text: str, *, uuid: str | None = None) -> "Message":
        if uuid is None:
            return cls(MessageRole.DEVELOPER, (ContentBlock.text_block(text),))
        return cls(
            MessageRole.DEVELOPER,
            (ContentBlock.text_block(text),),
            uuid=uuid,
        )

    @classmethod
    def tool_call(
        cls,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        uuid: str | None = None,
    ) -> "Message":
        call = ToolCallBlock(call_id, tool_name, arguments or {})
        if uuid is None:
            return cls(MessageRole.ASSISTANT, (ContentBlock.tool_call(call),))
        return cls(
            MessageRole.ASSISTANT,
            (ContentBlock.tool_call(call),),
            uuid=uuid,
        )

    @classmethod
    def tool_result(
        cls,
        result: ToolResultBlock,
        *,
        uuid: str | None = None,
    ) -> "Message":
        if uuid is None:
            return cls(MessageRole.TOOL, (ContentBlock.tool_result(result),))
        return cls(
            MessageRole.TOOL,
            (ContentBlock.tool_result(result),),
            uuid=uuid,
        )

    def tool_calls(self) -> list[ToolCallBlock]:
        return [
            block
            for block in self.blocks
            if isinstance(block, ToolCallBlock)
        ]

    def text(self) -> str:
        parts = [block.text() for block in self.blocks]
        return "\n".join(part for part in parts if part)


def message_to_dict(message: Message) -> dict[str, Any]:
    return message.to_dict()


def message_from_dict(data: Mapping[str, Any]) -> Message:
    return Message.from_dict(data)


def messages_to_dict(messages: Iterable[Message]) -> list[dict[str, Any]]:
    return [message_to_dict(message) for message in messages]


def messages_from_dict(data: Any) -> list[Message]:
    if not isinstance(data, list):
        raise TypeError("messages data must be a JSON array")
    return [message_from_dict(message) for message in data]


def message_history_to_dict(messages: Iterable[Message]) -> dict[str, Any]:
    return {
        "schema": MESSAGE_HISTORY_SCHEMA,
        "messages": messages_to_dict(messages),
    }


def message_history_from_dict(data: Mapping[str, Any]) -> list[Message]:
    if not isinstance(data, Mapping):
        raise TypeError("message history data must be a JSON object")

    schema = _required_str(data, "schema", "message_history.schema")
    if schema != MESSAGE_HISTORY_SCHEMA:
        raise ValueError(f"unsupported message history schema: {schema}")

    messages_data = _required_list(data, "messages", "message_history.messages")
    return messages_from_dict(messages_data)


def content_block_to_dict(block: ContentBlock) -> dict[str, Any]:
    return block.to_dict()


def content_block_from_dict(data: Mapping[str, Any]) -> ContentBlock:
    return ContentBlock.from_dict(data)


def _json_object(value: Any, path: str) -> dict[str, Any]:
    data = _json_value(value, path)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must be a JSON object")
    return data


def _json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be a finite JSON number")
        return value
    if isinstance(value, list):
        return [
            _json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        data: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            data[key] = _json_value(item, f"{path}.{key}")
        return data
    raise TypeError(f"{path} must be JSON-compatible, got {type(value).__name__}")
