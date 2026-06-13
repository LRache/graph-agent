import unittest

from graph_agent import (
    ContentBlock,
    ContentBlockKind,
    FileBlock,
    Message,
    MessageRole,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)


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
