import asyncio
import unittest

from graph_agent import (
    ContentBlock,
    FileBlock,
    Message,
    MessageRole,
    OpenAIProvider,
    ToolResultBlock,
    ToolSchema,
)

from tests.openai_helpers import (
    FakeClient,
    sdk_custom_tool_call,
    sdk_function_call,
    sdk_message,
    sdk_reasoning,
    sdk_response,
    sdk_text,
)



class OpenAIResponsesProviderTests(unittest.TestCase):
    def test_generate_converts_messages_to_response(self):
        response = sdk_response(
            sdk_message(sdk_text("pong")),
            id="resp_1",
            model="test-model",
            status="completed",
        )
        client = FakeClient(response)
        provider = OpenAIProvider("test-model", client=client, temperature=0)

        result = asyncio.run(
            provider.generate([Message.system_text("rules"), Message.user_text("ping")])
        )

        self.assertEqual(result.role, MessageRole.ASSISTANT)
        self.assertEqual(result.text(), "pong")
        self.assertEqual(result.response_meta["id"], "resp_1")
        self.assertEqual(
            client.responses.calls,
            [
                {
                    "model": "test-model",
                    "input": [
                        {"type": "message", "role": "system", "content": "rules"},
                        {"type": "message", "role": "user", "content": "ping"},
                    ],
                    "temperature": 0,
                }
            ],
        )

    def test_response_function_call_becomes_tool_call_message(self):
        response = sdk_response(
            sdk_function_call("call_1", "get_weather", '{"city": "Paris"}'),
            id="resp_2",
            model="test-model",
            status="completed",
        )
        provider = OpenAIProvider("test-model", client=FakeClient(response))

        result = asyncio.run(provider.generate([Message.user_text("weather?")]))

        self.assertEqual(len(result.tool_calls()), 1)
        call = result.tool_calls()[0]
        self.assertEqual(call.call_id, "call_1")
        self.assertEqual(call.tool_name, "get_weather")
        self.assertEqual(call.arguments, {"city": "Paris"})

    def test_response_reasoning_item_is_stored_in_message_extra(self):
        response = sdk_response(
            sdk_reasoning("rs_1", "looked up the weather", "encrypted-reasoning"),
            sdk_message(sdk_text("sunny")),
        )
        provider = OpenAIProvider("test-model", client=FakeClient(response))

        result = asyncio.run(provider.generate([Message.user_text("weather?")]))

        self.assertEqual(result.text(), "looked up the weather\nsunny")
        self.assertEqual(
            result.extra,
            {
                "openai": {
                    "reasoning_items": [
                        {
                            "id": "rs_1",
                            "summary": [
                                {
                                    "text": "looked up the weather",
                                    "type": "summary_text",
                                }
                            ],
                            "type": "reasoning",
                            "encrypted_content": "encrypted-reasoning",
                            "status": "completed",
                        }
                    ]
                }
            },
        )

    def test_message_extra_reasoning_items_are_replayed_as_input_items(self):
        response = sdk_response(sdk_message())
        client = FakeClient(response)
        provider = OpenAIProvider("test-model", client=client)
        message = Message(
            MessageRole.ASSISTANT,
            (
                ContentBlock.reasoning("do not serialize as text"),
                ContentBlock.text_block("visible assistant text"),
            ),
            extra={
                "openai": {
                    "reasoning_items": [
                        {
                            "id": "rs_1",
                            "summary": [
                                {
                                    "text": "reasoning summary",
                                    "type": "summary_text",
                                }
                            ],
                            "type": "reasoning",
                            "encrypted_content": "encrypted-reasoning",
                            "status": "completed",
                        }
                    ]
                }
            },
        )

        with self.assertLogs("graph_agent.provider.convert", level="WARNING") as logs:
            asyncio.run(provider.generate([message]))

        self.assertIn("Ignoring ReasoningBlock", logs.output[0])

        self.assertEqual(
            client.responses.calls[0]["input"],
            [
                {
                    "id": "rs_1",
                    "summary": [
                        {
                            "text": "reasoning summary",
                            "type": "summary_text",
                        }
                    ],
                    "type": "reasoning",
                    "encrypted_content": "encrypted-reasoning",
                    "status": "completed",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "visible assistant text",
                },
            ],
        )

    def test_tool_call_message_serializes_arguments_as_json(self):
        response = sdk_response(sdk_message())
        client = FakeClient(response)
        provider = OpenAIProvider("test-model", client=client)

        asyncio.run(
            provider.generate([Message.tool_call("call_1", "get_weather", {"city": "Paris"})])
        )

        self.assertEqual(
            client.responses.calls[0]["input"],
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                    "status": "completed",
                }
            ],
        )

    def test_tool_result_message_becomes_function_call_output(self):
        message = Message.tool_result(
            ToolResultBlock(
                call_id="call_1",
                tool_name="get_weather",
                content="sunny",
            )
        )
        response = sdk_response(sdk_message())
        client = FakeClient(response)
        provider = OpenAIProvider("test-model", client=client)

        asyncio.run(provider.generate([message]))

        self.assertEqual(
            client.responses.calls[0]["input"],
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny",
                    "status": "completed",
                }
            ],
        )

    def test_generate_serializes_tool_schemas_to_response_tools(self):
        response = sdk_response(sdk_message())
        client = FakeClient(response)
        provider = OpenAIProvider("test-model", client=client)

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
            client.responses.calls[0]["tools"],
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        )

    def test_provider_serializes_default_tool_schemas_to_response_tools(self):
        response = sdk_response(sdk_message())
        client = FakeClient(response)
        provider = OpenAIProvider(
            "test-model",
            client=client,
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
            client.responses.calls[0]["tools"],
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        )

    def test_generate_rejects_tool_role_text_message(self):
        response = sdk_response(sdk_message())
        client = FakeClient(response)
        provider = OpenAIProvider("test-model", client=client)

        with self.assertRaisesRegex(ValueError, "tool messages cannot contain text blocks"):
            message = Message.of(MessageRole.TOOL, [ContentBlock.user_text("raw tool output")])
            asyncio.run(provider.generate([message]))

        self.assertEqual(client.responses.calls, [])

    def test_generate_rejects_file_blocks_until_openai_file_input_is_supported(self):
        response = sdk_response(sdk_message())
        client = FakeClient(response)
        provider = OpenAIProvider("test-model", client=client)

        with self.assertRaisesRegex(ValueError, "does not support file blocks"):
            asyncio.run(
                provider.generate(
                    [Message.of(MessageRole.USER, [FileBlock(path="/tmp/input.txt")])]
                )
            )

        self.assertEqual(client.responses.calls, [])

    def test_response_function_call_rejects_invalid_json_arguments(self):
        response = sdk_response(sdk_function_call("call_1", "lookup", "not-json"))
        provider = OpenAIProvider("test-model", client=FakeClient(response))

        with self.assertRaisesRegex(ValueError, "valid JSON"):
            asyncio.run(provider.generate([Message.user_text("lookup")]))

    def test_response_function_call_rejects_non_object_arguments(self):
        response = sdk_response(sdk_function_call("call_1", "lookup", "[]"))
        provider = OpenAIProvider("test-model", client=FakeClient(response))

        with self.assertRaisesRegex(ValueError, "JSON object"):
            asyncio.run(provider.generate([Message.user_text("lookup")]))

    def test_response_custom_tool_call_is_rejected(self):
        response = sdk_response(sdk_custom_tool_call("call_1", "shell", "ls"))
        provider = OpenAIProvider("test-model", client=FakeClient(response))

        with self.assertRaisesRegex(ValueError, "unsupported OpenAI response output item"):
            asyncio.run(provider.generate([Message.user_text("run")]))

    def test_generate_rejects_single_message(self):
        provider = OpenAIProvider(
            "test-model",
            client=FakeClient(sdk_response(sdk_message())),
        )

        with self.assertRaisesRegex(TypeError, "messages must be a list"):
            asyncio.run(provider.generate(Message.user_text("weather?")))

        self.assertEqual(
            provider.client.responses.calls,
            [],
        )
