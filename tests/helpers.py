from graph_agent import Message, Node, NodeKind, NodeResult, UpstreamOutputs
from graph_agent.runtime import RunContext


class CallableNode(Node):
    def __init__(
        self,
        name,
        handler,
        kind=NodeKind.LLM,
        prepare_downstream_history=None,
    ):
        self.name = name
        self._handler = handler
        self._kind = kind
        self._prepare_downstream_history = prepare_downstream_history

    async def invoke(
        self,
        ctx: RunContext,
        history: list[Message],
        upstream_outputs: UpstreamOutputs,
        **extra,
    ) -> NodeResult:
        self.extra = extra
        return NodeResult(self, self._handler(ctx, history, upstream_outputs))

    def prepare_downstream_history(
        self,
        upstream_outputs: UpstreamOutputs,
        history: list[Message],
    ) -> list[Message]:
        if self._prepare_downstream_history is not None:
            return self._prepare_downstream_history(upstream_outputs, history)

        messages = list(history)
        for output in upstream_outputs.values():
            messages.append(output)
        return messages

    def kind(self) -> NodeKind:
        return self._kind


class StaticProvider:
    async def generate(self, messages, **response_options):
        return Message.assistant_text("static")


class RecordingProvider:
    def __init__(self):
        self.calls = []

    async def generate(self, messages, **response_options):
        self.calls.append((messages, response_options))
        return Message.assistant_text("recorded")
