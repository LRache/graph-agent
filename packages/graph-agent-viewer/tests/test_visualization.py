import unittest

from graph_agent import (
    GraphBuilder,
    Message,
    Node,
    NodeKind,
    NodeResult,
    UpstreamOutputs,
)
from graph_agent.runtime import RunContext
from graph_agent_viewer.visualization import to_mermaid


class StaticNode(Node):
    def __init__(self, name: str) -> None:
        self.name = name

    def prepare_downstream_history(
        self,
        upstream_outputs: UpstreamOutputs,
        history: list[Message],
    ) -> list[Message]:
        return list(history)

    async def invoke(
        self,
        ctx: RunContext,
        history: list[Message],
        upstream_outputs: UpstreamOutputs,
        **extra,
    ) -> NodeResult:
        return NodeResult(self, Message.assistant_text(self.name))

    def kind(self) -> NodeKind:
        return NodeKind.LLM


class VisualizationTests(unittest.TestCase):
    def test_to_mermaid_renders_nodes_start_and_edges(self) -> None:
        graph = (
            GraphBuilder("demo")
            .node(StaticNode("a"))
            .node(StaticNode("b"))
            .start("a")
            .edge("a", "b", name="to_b")
            .build()
        )

        self.assertEqual(
            to_mermaid(graph),
            "\n".join(
                [
                    "flowchart TD",
                    '    n0["a<br/>llm"]',
                    '    n1["b<br/>llm"]',
                    "    graph_start((start))",
                    "    graph_start --> n0",
                    '    n0 -- "to_b" --> n1',
                ]
            ),
        )
