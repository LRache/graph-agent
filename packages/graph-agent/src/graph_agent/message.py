"""Typed messages shared by graph nodes, tools, and model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Iterable, Mapping


JsonObject = dict[str, Any]


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
    arguments: JsonObject = field(default_factory=dict)

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

    @classmethod
    def of(cls, role: MessageRole | str, blocks: Iterable[ContentBlock]) -> "Message":
        return cls(MessageRole(role), tuple(blocks))

    @classmethod
    def system_text(cls, text: str) -> "Message":
        return cls(MessageRole.SYSTEM, (ContentBlock.text_block(text),))

    @classmethod
    def user_text(cls, text: str) -> "Message":
        return cls(MessageRole.USER, (ContentBlock.text_block(text),))

    @classmethod
    def assistant_text(cls, text: str, **extra: Any) -> "Message":
        return cls(MessageRole.ASSISTANT, (ContentBlock.text_block(text),), extra=extra)

    @classmethod
    def developer(cls, text: str) -> "Message":
        return cls(MessageRole.DEVELOPER, (ContentBlock.text_block(text),))

    @classmethod
    def tool_call(
        cls,
        call_id: str,
        tool_name: str,
        arguments: JsonObject | None = None,
    ) -> "Message":
        call = ToolCallBlock(call_id, tool_name, arguments or {})
        return cls(MessageRole.ASSISTANT, (ContentBlock.tool_call(call),))

    @classmethod
    def tool_result(cls, result: ToolResultBlock) -> "Message":
        return cls(MessageRole.TOOL, (ContentBlock.tool_result(result),))

    def tool_calls(self) -> list[ToolCallBlock]:
        return [
            block
            for block in self.blocks
            if isinstance(block, ToolCallBlock)
        ]

    def text(self) -> str:
        parts = [block.text() for block in self.blocks]
        return "\n".join(part for part in parts if part)
