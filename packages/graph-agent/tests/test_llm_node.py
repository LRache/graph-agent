import asyncio
import unittest

from graph_agent import LLMNode, LLMNodeFactory, Message, OpenAIProvider, ToolSchema
from graph_agent.runtime import RunContext

from tests.openai_helpers import FakeClient, sdk_message, sdk_response, sdk_text



class LLMNodeTests(unittest.TestCase):
    def test_llm_node_invokes_provider_with_history_and_upstream_outputs(self):
        response = sdk_response(
            sdk_message(sdk_text("done")),
            id="resp_3",
            model="test-model",
            status="completed",
        )
        provider = OpenAIProvider("test-model", client=FakeClient(response))
        node = LLMNode("llm", provider)

        result = asyncio.run(
            node.invoke(
                RunContext(),
                [Message.user_text("context")],
                {"input": Message.user_text("upstream")},
            )
        )

        self.assertEqual(result.output.text(), "done")
        self.assertEqual(
            provider.client.responses.calls[0]["input"],
            [
                {"type": "message", "role": "user", "content": "context"},
                {"type": "message", "role": "user", "content": "upstream"},
            ],
        )

    def test_llm_node_passes_explicit_tools_to_provider(self):
        response = sdk_response(sdk_message(sdk_text("done")))
        provider = OpenAIProvider("test-model", client=FakeClient(response))
        node = LLMNode("llm", provider)

        asyncio.run(
            node.invoke(
                RunContext(),
                [Message.user_text("weather?")],
                {},
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
            provider.client.responses.calls[0]["tools"],
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

    def test_llm_node_injects_system_prompt_for_provider_only(self):
        response = sdk_response(sdk_message(), id="resp_4", model="test-model")
        provider = OpenAIProvider("test-model", client=FakeClient(response))
        node = LLMNode("critic", provider, system_prompt="Be a careful reviewer.")

        downstream_history = node.prepare_downstream_history(
            {"input": Message.user_text("upstream")},
            [Message.user_text("context")],
        )
        asyncio.run(
            node.invoke(
                RunContext(),
                [Message.user_text("context")],
                {"input": Message.user_text("upstream")},
            )
        )

        self.assertEqual(
            [message.text() for message in downstream_history],
            ["context", "upstream"],
        )
        self.assertEqual(
            provider.client.responses.calls[0]["input"],
            [
                {
                    "type": "message",
                    "role": "system",
                    "content": "Be a careful reviewer.",
                },
                {"type": "message", "role": "user", "content": "context"},
                {"type": "message", "role": "user", "content": "upstream"},
            ],
        )

    def test_llm_node_rejects_non_string_system_prompt(self):
        provider = OpenAIProvider(
            "test-model",
            client=FakeClient(sdk_response(sdk_message())),
        )

        with self.assertRaisesRegex(TypeError, "system_prompt must be a str"):
            LLMNode("critic", provider, system_prompt=Message.system_text("rules"))

    def test_factory_creates_node_class_with_system_prompt(self):
        response = sdk_response(sdk_message(), id="resp_5", model="test-model")
        provider = OpenAIProvider("test-model", client=FakeClient(response))
        factory = LLMNodeFactory(provider)
        TranslatorNode = factory.create_node_class(
            "TranslatorNode",
            system_prompt="Translate to English.",
        )
        node = TranslatorNode("translator")

        asyncio.run(node.invoke(RunContext(), [Message.user_text("bonjour")], {}))

        self.assertIsInstance(node, LLMNode)
        self.assertEqual(TranslatorNode.__name__, "TranslatorNode")
        self.assertEqual(
            provider.client.responses.calls[0]["input"],
            [
                {
                    "type": "message",
                    "role": "system",
                    "content": "Translate to English.",
                },
                {"type": "message", "role": "user", "content": "bonjour"},
            ],
        )

    def test_factory_creates_distinct_node_classes_from_system_prompts(self):
        response = sdk_response(sdk_message(), id="resp_6", model="test-model")
        reviewer_provider = OpenAIProvider("review-model", client=FakeClient(response))
        summarizer_provider = OpenAIProvider("summary-model", client=FakeClient(response))
        factory = LLMNodeFactory(reviewer_provider)
        ReviewerNode = factory.create_node_class(
            "ReviewerNode",
            system_prompt="Review code for correctness.",
        )
        SummarizerNode = factory.create_node_class(
            "SummarizerNode",
            provider=summarizer_provider,
            system_prompt="Summarize briefly.",
        )

        reviewer = ReviewerNode("reviewer")
        summarizer = SummarizerNode("summarizer")
        asyncio.run(reviewer.invoke(RunContext(), [Message.user_text("diff")], {}))
        asyncio.run(summarizer.invoke(RunContext(), [Message.user_text("notes")], {}))

        self.assertNotEqual(ReviewerNode, SummarizerNode)
        self.assertEqual(ReviewerNode.__name__, "ReviewerNode")
        self.assertEqual(SummarizerNode.__name__, "SummarizerNode")
        self.assertEqual(
            reviewer_provider.client.responses.calls[0]["input"][0],
            {
                "type": "message",
                "role": "system",
                "content": "Review code for correctness.",
            },
        )
        self.assertEqual(
            summarizer_provider.client.responses.calls[0]["input"][0],
            {
                "type": "message",
                "role": "system",
                "content": "Summarize briefly.",
            },
        )

    def test_factory_requires_provider_on_init(self):
        with self.assertRaises(TypeError):
            LLMNodeFactory()
