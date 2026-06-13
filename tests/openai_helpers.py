from types import SimpleNamespace

from openai.types.responses import (
    Response,
    ResponseCustomToolCall,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)


def sdk_response(*output, id="resp_1", model="test-model", status="completed"):
    return Response(
        id=id,
        created_at=0,
        model=model,
        object="response",
        output=list(output),
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status=status,
    )


def sdk_message(*content, id="msg_1", status="completed"):
    return ResponseOutputMessage(
        id=id,
        content=list(content),
        role="assistant",
        status=status,
        type="message",
    )


def sdk_text(text):
    return ResponseOutputText(annotations=[], text=text, type="output_text")


def sdk_function_call(call_id, name, arguments, status="completed"):
    return ResponseFunctionToolCall(
        arguments=arguments,
        call_id=call_id,
        name=name,
        type="function_call",
        status=status,
    )


def sdk_reasoning(id="rs_1", summary_text="thinking", encrypted_content="encrypted"):
    return ResponseReasoningItem(
        id=id,
        summary=[{"text": summary_text, "type": "summary_text"}],
        type="reasoning",
        encrypted_content=encrypted_content,
        status="completed",
    )


def sdk_custom_tool_call(call_id, name, input_value):
    return ResponseCustomToolCall(
        call_id=call_id,
        input=input_value,
        name=name,
        type="custom_tool_call",
    )


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


def chat_completion(
    *,
    content="pong",
    tool_calls=None,
    function_call=None,
    refusal=None,
    id="chatcmpl_1",
    model="test-model",
    finish_reason="stop",
):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        function_call=function_call,
        refusal=refusal,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=id,
        model=model,
        choices=[choice],
        usage={
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    )


def chat_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeChatCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeChat:
    def __init__(self, response):
        self.completions = FakeChatCompletions(response)


class FakeChatClient:
    def __init__(self, response):
        self.chat = FakeChat(response)
