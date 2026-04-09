from llmx.models import Message, GenerateRequest, GenerateResponse, Usage, ToolCall, StreamChunk


def test_message():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_generate_request_defaults():
    req = GenerateRequest(messages=[Message(role="user", content="Hi")])
    assert req.temperature == 0.7
    assert req.max_tokens == 1024
    assert req.extra == {}


def test_generate_response():
    res = GenerateResponse(content="Hi", model="test-model")
    assert res.content == "Hi"
    assert res.model == "test-model"
    assert res.tool_calls == []


def test_usage():
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert usage.total_tokens == 15


def test_tool_call():
    tc = ToolCall(id="1", name="test", arguments={"a": 1})
    assert tc.name == "test"
    assert tc.arguments["a"] == 1


def test_stream_chunk():
    chunk = StreamChunk(delta="Hello", finished=False)
    assert chunk.delta == "Hello"
    assert not chunk.finished