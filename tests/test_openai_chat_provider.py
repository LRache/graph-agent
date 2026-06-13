import asyncio
import unittest

from graph_agent import (
    ContentBlock,
    Message,
    MessageRole,
    OpenAIProvider,
    ToolResultBlock,
    ToolSchema,
)

from tests.openai_helpers import FakeChatClient, chat_completion, chat_tool_call



class OpenAIChatProviderTests(unittest.TestCase):
    def test_chat_completions_generate_converts_messages_to_request(self):
        client = FakeChatClient(
            chat_completion(content="pong", id="chatcmpl_1", model="test-model")
        )
        provider = OpenAIProvider(
            "test-model",
            client=client,
            api="chat_completions",
            temperature=0,
        )

        result = asyncio.run(
            provider.generate([Message.system_text("rules"), Message.user_text("ping")])
        )

        self.assertEqual(result.role, MessageRole.ASSISTANT)
        self.assertEqual(result.text(), "pong")
        self.assertEqual(result.response_meta["id"], "chatcmpl_1")
        self.assertEqual(result.response_meta["api"], "chat_completions")
        self.assertEqual(result.response_meta["finish_reason"], "stop")
        self.assertEqual(
            client.chat.completions.calls,
            [
                {
                    "model": "test-model",
                    "messages": [
                        {"role": "system", "content": "rules"},
                        {"role": "user", "content": "ping"},
                    ],
                    "temperature": 0,
                }
            ],
        )

    def test_chat_completions_maps_developer_messages_to_system(self):
        client = FakeChatClient(chat_completion(content="done"))
        provider = OpenAIProvider("test-model", client=client, api="chat")

        asyncio.run(provider.generate([Message.developer("developer rules")]))

        self.assertEqual(
            client.chat.completions.calls[0]["messages"],
            [{"role": "system", "content": "developer rules"}],
        )

    def test_chat_completion_tool_call_becomes_tool_call_message(self):
        provider = OpenAIProvider(
            "test-model",
            client=FakeChatClient(
                chat_completion(
                    content=None,
                    tool_calls=[
                        chat_tool_call(
                            "call_1",
                            "get_weather",
                            '{"city": "Paris"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            ),
            api="chat_completions",
        )

        result = asyncio.run(provider.generate([Message.user_text("weather?")]))

        self.assertEqual(len(result.tool_calls()), 1)
        call = result.tool_calls()[0]
        self.assertEqual(call.call_id, "call_1")
        self.assertEqual(call.tool_name, "get_weather")
        self.assertEqual(call.arguments, {"city": "Paris"})
        self.assertEqual(result.response_meta["finish_reason"], "tool_calls")

    def test_chat_completions_serializes_tool_call_and_tool_result_messages(self):
        client = FakeChatClient(chat_completion(content="sunny"))
        provider = OpenAIProvider("test-model", client=client, api="chat_completions")

        asyncio.run(
            provider.generate(
                [
                    Message.tool_call(
                        "call_1",
                        "get_weather",
                        {"city": "Paris"},
                    ),
                    Message.tool_result(
                        ToolResultBlock(
                            call_id="call_1",
                            tool_name="get_weather",
                            content="sunny",
                        )
                    ),
                ]
            )
        )

        self.assertEqual(
            client.chat.completions.calls[0]["messages"],
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "sunny",
                },
            ],
        )

    def test_chat_completions_serializes_tool_schemas_to_tools(self):
        client = FakeChatClient(chat_completion(content="done"))
        provider = OpenAIProvider("test-model", client=client, api="chat_completions")

        asyncio.run(
            provider.generate(
                [Message.user_text("weather?")],
                tools=[
                    ToolSchema(
                        name="get_weather",
                        description="Get the weather.",
                        parameters={
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    )
                ],
            )
        )

        self.assertEqual(
            client.chat.completions.calls[0]["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

    def test_chat_completions_serializes_provider_default_tool_schemas(self):
        client = FakeChatClient(chat_completion(content="done"))
        provider = OpenAIProvider(
            "test-model",
            client=client,
            api="chat_completions",
            tools=[
                ToolSchema(
                    name="get_weather",
                    description="Get the weather.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                )
            ],
        )

        asyncio.run(provider.generate([Message.user_text("weather?")]))

        self.assertEqual(
            client.chat.completions.calls[0]["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

    def test_chat_completions_rejects_reasoning_replay(self):
        client = FakeChatClient(chat_completion(content="done"))
        provider = OpenAIProvider("test-model", client=client, api="chat_completions")
        message = Message(
            MessageRole.ASSISTANT,
            (ContentBlock.text_block("visible assistant text"),),
            extra={
                "openai": {
                    "reasoning_items": [
                        {
                            "id": "rs_1",
                            "summary": [],
                            "type": "reasoning",
                        }
                    ]
                }
            },
        )

        with self.assertRaisesRegex(ValueError, "reasoning replay"):
            asyncio.run(provider.generate([message]))

        self.assertEqual(client.chat.completions.calls, [])
