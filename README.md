# graph-agent

A small Python implementation of an agent graph core.

The package currently includes:

- typed `Message` and `ContentBlock` values for user, assistant, developer, and tool flow
- a synchronous `ToolRegistry` / `ToolExecutor` that maps function tool calls to tool-result messages
- an OpenAI provider that converts `Message` inputs into assistant `Message`
  outputs using either the Responses API or Chat Completions API
- graph nodes for simple asynchronous message flow
- named graph edges for node activation and upstream output routing
- an explicit graph start node that receives configured input
- graph results from the last completed node

The `graph-agent` package lives under `packages/graph-agent`. This repository
also contains the `graph-agent-viewer` package under `packages/graph-agent-viewer`
for visualization and viewer-related helpers.

## OpenAI provider

`OpenAIProvider` uses the Responses API by default:

```python
import asyncio

from graph_agent import Message, OpenAIProvider


async def main():
    provider = OpenAIProvider(model="gpt-5.5")
    response = await provider.generate([Message.user_text("Say hello in one sentence.")])
    print(response.text())


asyncio.run(main())
```

Tool schemas can be configured on the provider:

```python
from graph_agent import OpenAIProvider, ToolSchema


provider = OpenAIProvider(
    model="gpt-5.5",
    tools=[
        ToolSchema(
            name="get_weather",
            description="Get the weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    ],
)
```

For older OpenAI-compatible Chat Completions endpoints, choose the chat API
explicitly:

```python
provider = OpenAIProvider(model="gpt-4", api="chat_completions")
```

You can also create reusable node types from different system prompts:

```python
from graph_agent import LLMNodeFactory, OpenAIProvider


review_provider = OpenAIProvider(model="gpt-5.5")
summary_provider = OpenAIProvider(model="gpt-5.5")
factory = LLMNodeFactory(review_provider)

ReviewerNode = factory.create_node_class(
    "ReviewerNode",
    system_prompt="Review code for correctness and missing tests.",
)
SummarizerNode = factory.create_node_class(
    "SummarizerNode",
    provider=summary_provider,
    system_prompt="Summarize the input in three concise bullets.",
)

reviewer = ReviewerNode("reviewer")
summarizer = SummarizerNode("summarizer")
```

## Example

```python
import asyncio

from graph_agent import GraphBuilder, Message, NodeKind, NodeResult


class EchoNode:
    name = "echo"

    def prepare_downstream_history(self, upstream_outputs, history):
        messages = list(history)
        for output in upstream_outputs.values():
            messages.append(output)
        return messages

    async def invoke(self, ctx, history, upstream_outputs, **extra):
        return NodeResult(self, Message.assistant_text(history[-1].text()))

    def kind(self):
        return NodeKind.LLM


graph = (
    GraphBuilder("input_echo")
    .input([Message.user_text("hello")])
    .node(EchoNode())
    .start("echo")
    .build()
)

result = asyncio.run(graph.run())
for message in result.output:
    print(message.text())
```

## Viewer

```python
from graph_agent_viewer.visualization import to_mermaid

print(to_mermaid(graph))
```

To open the React runtime viewer:

```python
from graph_agent_viewer import GraphView

GraphView.run(graph)
```

Enable step mode to pause before each next activation round. Click `Next Step` in
the viewer to continue:

```python
GraphView.run(graph, step_mode=True)
```

There is also a runnable local demo:

```bash
uv run python examples/visualization_mermaid.py
```

The demo builds a small graph and prints a Mermaid flowchart.
