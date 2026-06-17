import json
import unittest
from uuid import UUID

from graph_agent import (
    ContentBlock,
    ContentBlockKind,
    FileBlock,
    MESSAGE_HISTORY_SCHEMA,
    Message,
    MessageRole,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    content_block_from_dict,
    content_block_to_dict,
    message_from_dict,
    message_history_from_dict,
    message_history_to_dict,
    message_to_dict,
    messages_from_dict,
    messages_to_dict,
)


UUID_1 = "00000000-0000-4000-8000-000000000001"
UUID_2 = "00000000-0000-4000-8000-000000000002"


class MessageTests(unittest.TestCase):
    def test_content_block_kind_has_only_protocol_block_types(self):
        self.assertEqual(
            set(ContentBlockKind),
            {
                ContentBlockKind.TEXT,
                ContentBlockKind.FILE,
                ContentBlockKind.REASONING,
                ContentBlockKind.TOOL_CALL,
                ContentBlockKind.TOOL_RESULT,
            },
        )

    def test_text_blocks_are_role_independent(self):
        self.assertEqual(Message.system_text("rules").blocks[0].kind, ContentBlockKind.TEXT)
        self.assertEqual(Message.user_text("input").blocks[0].kind, ContentBlockKind.TEXT)
        self.assertEqual(
            Message.assistant_text("output").blocks[0].kind,
            ContentBlockKind.TEXT,
        )

    def test_file_block_has_file_kind(self):
        file_block = FileBlock(file_id="file_1", mime_type="text/plain", name="note.txt")
        block = ContentBlock.file(file_block)

        self.assertIs(block, file_block)
        self.assertEqual(block.kind, ContentBlockKind.FILE)

    def test_content_block_factories_return_concrete_block_objects(self):
        text = ContentBlock.text_block("hello")
        reasoning = ContentBlock.reasoning("thinking")
        call = ContentBlock.tool_call(ToolCallBlock("call_1", "lookup"))
        result = ContentBlock.tool_result(ToolResultBlock("call_1", "lookup"))

        self.assertIsInstance(text, TextBlock)
        self.assertIsInstance(reasoning, ReasoningBlock)
        self.assertIsInstance(call, ToolCallBlock)
        self.assertIsInstance(result, ToolResultBlock)

    def test_message_role_block_kind_constraints_allow_valid_shapes(self):
        valid_messages = [
            Message.of(MessageRole.ASSISTANT, [ContentBlock.text_block("answer")]),
            Message.of(MessageRole.ASSISTANT, [ContentBlock.reasoning("thinking")]),
            Message.of(
                MessageRole.ASSISTANT,
                [ContentBlock.tool_call(ToolCallBlock("call_1", "lookup"))],
            ),
            Message.of(
                MessageRole.TOOL,
                [
                    ContentBlock.tool_result(
                        ToolResultBlock("call_1", "lookup", "ok")
                    )
                ],
            ),
            Message.of(MessageRole.USER, [ContentBlock.text_block("question")]),
            Message.of(MessageRole.SYSTEM, [ContentBlock.file(FileBlock(path="/tmp/rules.txt"))]),
            Message.of(MessageRole.DEVELOPER, [ContentBlock.text_block("debug note")]),
        ]

        self.assertEqual(len(valid_messages), 7)

    def test_message_role_block_kind_constraints_reject_invalid_shapes(self):
        invalid_shapes = [
            (MessageRole.ASSISTANT, ContentBlock.file(FileBlock(path="/tmp/output.txt"))),
            (MessageRole.TOOL, ContentBlock.text_block("raw tool output")),
            (MessageRole.USER, ContentBlock.reasoning("thinking")),
            (MessageRole.SYSTEM, ContentBlock.tool_call(ToolCallBlock("call_1", "lookup"))),
            (MessageRole.DEVELOPER, ContentBlock.tool_result(ToolResultBlock("call_1", "lookup"))),
        ]

        for role, block in invalid_shapes:
            with self.subTest(role=role, kind=block.kind):
                with self.assertRaisesRegex(ValueError, f"{role.value} messages cannot contain"):
                    Message.of(role, [block])

    def test_message_list_prints_compact_readable_repr(self):
        messages = [
            Message.user_text("hello"),
            Message.assistant_text("hi"),
            Message.tool_call("call_1", "lookup", {"query": "weather"}),
        ]

        printed = str(messages)

        self.assertEqual(
            printed,
            "[Message.user_text('hello'), Message.assistant_text('hi'), "
            "Message.tool_call('call_1', 'lookup', {'query': 'weather'})]",
        )
        self.assertNotIn("MessageRole", printed)
        self.assertNotIn("text_value", printed)

    def test_content_blocks_round_trip_through_serialized_dicts(self):
        blocks = [
            ContentBlock.text_block("hello"),
            ContentBlock.reasoning("thinking", signature="sig_1"),
            ContentBlock.file(
                FileBlock(
                    file_id="file_1",
                    path="/tmp/input.txt",
                    mime_type="text/plain",
                    name="input.txt",
                )
            ),
            ContentBlock.tool_call(
                ToolCallBlock(
                    "call_1",
                    "lookup",
                    {"query": "weather", "limit": 2},
                )
            ),
            ContentBlock.tool_result(
                ToolResultBlock("call_1", "lookup", "sunny", is_error=True)
            ),
        ]

        for block in blocks:
            with self.subTest(kind=block.kind):
                self.assertEqual(
                    content_block_from_dict(
                        json.loads(json.dumps(content_block_to_dict(block)))
                    ),
                    block,
                )
                self.assertEqual(
                    ContentBlock.from_dict(json.loads(json.dumps(block.to_dict()))),
                    block,
                )

    def test_messages_round_trip_through_json_serializable_history(self):
        messages = [
            Message.system_text("rules"),
            Message.of(
                MessageRole.USER,
                [
                    ContentBlock.text_block("input"),
                    ContentBlock.file(FileBlock(file_id="file_1", name="input.txt")),
                ],
            ),
            Message(
                MessageRole.ASSISTANT,
                (
                    ContentBlock.reasoning("summary", signature="sig_1"),
                    ContentBlock.text_block("answer"),
                    ContentBlock.tool_call(
                        ToolCallBlock(
                            "call_1",
                            "lookup",
                            {"query": "weather", "units": ["celsius"]},
                        )
                    ),
                ),
                response_meta={
                    "provider": "openai",
                    "id": "resp_1",
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
                extra={
                    "openai": {
                        "reasoning_items": [
                            {
                                "id": "rs_1",
                                "type": "reasoning",
                                "summary": [],
                            }
                        ]
                    }
                },
            ),
            Message.tool_result(
                ToolResultBlock("call_1", "lookup", "sunny", is_error=False)
            ),
        ]

        history_data = message_history_to_dict(messages)
        payload = json.loads(json.dumps(history_data))

        self.assertEqual(payload["schema"], MESSAGE_HISTORY_SCHEMA)
        self.assertEqual(payload["messages"][0]["uuid"], messages[0].uuid)
        self.assertEqual(message_history_from_dict(payload), messages)
        self.assertEqual(messages_from_dict(messages_to_dict(messages)), messages)
        self.assertEqual(Message.from_dict(messages[2].to_dict()), messages[2])
        self.assertEqual(message_from_dict(message_to_dict(messages[3])), messages[3])

    def test_message_uuid_is_generated_serialized_and_preserved(self):
        message = Message.user_text("hello")

        UUID(message.uuid)
        self.assertEqual(message.to_dict()["uuid"], message.uuid)
        self.assertEqual(Message.from_dict(message.to_dict()).uuid, message.uuid)

    def test_message_deserialization_adds_uuid_for_legacy_payloads(self):
        message = Message.from_dict(
            {
                "role": "user",
                "blocks": [{"kind": "text", "text": "hello"}],
            }
        )

        UUID(message.uuid)
        self.assertEqual(message.to_dict()["uuid"], message.uuid)

    def test_message_rejects_invalid_uuid(self):
        with self.assertRaisesRegex(ValueError, "message.uuid"):
            Message.user_text("hello", uuid="not-a-uuid")

    def test_message_hash_is_stable_for_serialized_message_identity(self):
        message = Message.tool_call(
            "call_1",
            "lookup",
            {
                "query": "weather",
                "options": {"limit": 2, "unit": "celsius"},
            },
            uuid=UUID_1,
        )
        same_content = Message.tool_call(
            "call_1",
            "lookup",
            {
                "options": {"unit": "celsius", "limit": 2},
                "query": "weather",
            },
            uuid=UUID_1,
        )
        different_uuid = Message.tool_call(
            "call_1",
            "lookup",
            {
                "query": "weather",
                "options": {"limit": 2, "unit": "celsius"},
            },
            uuid=UUID_2,
        )
        round_tripped = Message.from_dict(json.loads(json.dumps(message.to_dict())))

        self.assertRegex(message.hash(), r"^[0-9a-f]{64}$")
        self.assertEqual(message.hash(), same_content.hash())
        self.assertEqual(message.hash(), round_tripped.hash())
        self.assertNotEqual(message.hash(), different_uuid.hash())
        self.assertNotEqual(
            message.hash(),
            Message.tool_call(
                "call_1",
                "lookup",
                {"query": "forecast"},
                uuid=UUID_1,
            ).hash(),
        )

    def test_message_hash_includes_metadata(self):
        message = Message.assistant_text("answer", uuid=UUID_1)

        self.assertNotEqual(
            message.hash(),
            Message.assistant_text("answer", uuid=UUID_1, trace_id="trace_1").hash(),
        )
        self.assertNotEqual(
            message.hash(),
            Message(
                MessageRole.ASSISTANT,
                (ContentBlock.text_block("answer"),),
                response_meta={"id": "resp_1"},
                uuid=UUID_1,
            ).hash(),
        )

    def test_message_serialization_rejects_non_json_values(self):
        message = Message.assistant_text("answer", values=(1, 2))

        with self.assertRaisesRegex(TypeError, "extra.values"):
            message_to_dict(message)

        with self.assertRaisesRegex(TypeError, "tool_call.arguments keys"):
            message_to_dict(Message.tool_call("call_1", "lookup", {1: "bad"}))

    def test_message_deserialization_preserves_role_block_constraints(self):
        with self.assertRaisesRegex(ValueError, "user messages cannot contain"):
            message_from_dict(
                {
                    "role": "user",
                    "blocks": [
                        {
                            "kind": "tool_call",
                            "call_id": "call_1",
                            "tool_name": "lookup",
                        }
                    ],
                }
            )

    def test_message_history_deserialization_rejects_unknown_schema(self):
        with self.assertRaisesRegex(ValueError, "unsupported message history schema"):
            message_history_from_dict({"schema": "unknown", "messages": []})
