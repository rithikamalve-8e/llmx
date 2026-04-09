import pytest
from llmx.core import LLMClient
from llmx.models import Message, GenerateRequest


# --- Helpers ---
class DummyProvider:
    def __init__(self):
        self.generate_called = False
        self.stream_called = False

    def generate(self, request):
        self.generate_called = True
        return "mocked-response"

    def stream(self, request):
        self.stream_called = True
        yield from []


# --- _to_request tests ---
def test_to_request_from_string():
    req = LLMClient._to_request(
        "Hello",
        model="test-model",
        system=None,
        temperature=0.5,
        max_tokens=50,
        extra={}
    )

    assert isinstance(req, GenerateRequest)
    assert req.messages[0].content == "Hello"
    assert req.model == "test-model"


def test_to_request_from_message_list():
    messages = [Message(role="user", content="Hi")]

    req = LLMClient._to_request(
        messages,
        model=None,
        system=None,
        temperature=0.7,
        max_tokens=100,
        extra={}
    )

    assert req.messages == messages


def test_to_request_passthrough():
    original = GenerateRequest(messages=[])

    req = LLMClient._to_request(
        original,
        model=None,
        system=None,
        temperature=0.7,
        max_tokens=100,
        extra={}
    )

    assert req is original


def test_to_request_invalid_type():
    with pytest.raises(TypeError):
        LLMClient._to_request(
            123,
            model=None,
            system=None,
            temperature=0.7,
            max_tokens=100,
            extra={}
        )


# --- generate() tests ---
def test_generate_calls_provider(mocker):
    dummy = DummyProvider()

    mocker.patch("llmx.core.load_provider", return_value=dummy)

    client = LLMClient(provider="openai")
    result = client.generate("Hello", model="test-model")

    assert dummy.generate_called is True
    assert result == "mocked-response"


# --- stream() tests ---
def test_stream_calls_provider(mocker):
    dummy = DummyProvider()

    mocker.patch("llmx.core.load_provider", return_value=dummy)

    client = LLMClient(provider="openai")
    list(client.stream("Hello", model="test-model"))

    assert dummy.stream_called is True


# --- use() tests ---
def test_use_returns_new_instance(mocker):
    dummy = DummyProvider()

    mocker.patch("llmx.core.load_provider", return_value=dummy)

    client = LLMClient(provider="openai")
    new_client = client.use("gemini")

    assert new_client is not client
    assert new_client.provider_name == "gemini"
    assert client.provider_name == "openai"


# --- provider property ---
def test_provider_property(mocker):
    dummy = DummyProvider()

    mocker.patch("llmx.core.load_provider", return_value=dummy)

    client = LLMClient(provider="openai")

    assert client.provider is dummy


# --- __repr__ ---
def test_repr(mocker):
    dummy = DummyProvider()

    mocker.patch("llmx.core.load_provider", return_value=dummy)

    client = LLMClient(provider="openai")

    assert "openai" in repr(client)


# --- detect_provider ---
def test_detect_provider_openai(mocker):
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "key"})

    provider = LLMClient._detect_provider()
    assert provider == "openai"


def test_detect_provider_no_env(mocker):
    mocker.patch.dict("os.environ", {}, clear=True)

    with pytest.raises(EnvironmentError):
        LLMClient._detect_provider()