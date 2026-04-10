from llmx.models import (
    Message,
    ToolCall,
    Usage,
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)


def test_message_model():
    m = Message(role="user", content="hi")
    assert m.role == "user"
    assert m.content == "hi"


def test_tool_call_model():
    t = ToolCall(id="1", name="test", arguments={"a": 1})
    assert t.id == "1"
    assert t.name == "test"
    assert t.arguments["a"] == 1


def test_usage_model_defaults():
    u = Usage()
    assert u.prompt_tokens is None


def test_generate_request():
    req = GenerateRequest(messages=[Message(role="user", content="hi")])
    assert len(req.messages) == 1
    assert req.temperature == 0.7


def test_generate_response():
    resp = GenerateResponse(content="ok", model="test")
    assert resp.content == "ok"
    assert resp.model == "test"


def test_stream_chunk():
    c = StreamChunk(delta="hi")
    assert c.delta == "hi"