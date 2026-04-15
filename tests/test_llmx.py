"""
Unit tests refactored from the integration/demo script.
Every API call is mocked — no real network requests are made.
Covers all scenarios: simple prompt, options, multi-turn, streaming,
provider switching (Groq / Gemini), tool calling, and all error paths.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from llmx import LLMClient, Message
from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
    ToolCall,
    Usage,
)


# ===========================================================================
# Shared factories
# ===========================================================================

def _make_response(
    content: str = "test response",
    model: str = "gpt-4o",
    usage: Usage | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> GenerateResponse:
    return GenerateResponse(
        content=content,
        model=model,
        usage=usage, #if usage is not None else Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        tool_calls=tool_calls or [],
    )


PROVIDER_CLASS_NAMES = {
    "openai": "OpenAIProvider",
    "groq": "GroqProvider",
    "gemini": "GeminiProvider",
}

def _make_client(provider="openai"):
    class_name = PROVIDER_CLASS_NAMES[provider]
    with patch(f"llmx.providers.{provider}.{class_name}.__init__", return_value=None):
        return LLMClient(provider=provider)


def _attach_mock_provider(client: LLMClient, response: GenerateResponse) -> MagicMock:
    """Replace client._provider with a mock that returns *response*."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value=response)

    async def _stream(_req):
        yield StreamChunk(delta="chunk1", finished=False)
        yield StreamChunk(delta="chunk2", finished=True)

    mock.stream = _stream
    client._provider = mock
    return mock


# ===========================================================================
# 1. Client initialisation
# ===========================================================================

class TestClientInitialisation:

    def test_explicit_openai_provider(self):
        client = _make_client("openai")
        assert client.provider_name == "openai"

    def test_explicit_groq_provider(self):
        client = _make_client("groq")
        assert client.provider_name == "groq"

    def test_explicit_gemini_provider(self):
        client = _make_client("gemini")
        assert client.provider_name == "gemini"

    def test_repr_contains_provider_name(self):
        client = _make_client("openai")
        assert "openai" in repr(client)

    def test_auto_detect_openai_from_env(self):
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY")}
        with patch.dict(os.environ, {**clean, "OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("llmx.providers.openai.OpenAIProvider.__init__", return_value=None):
                client = LLMClient()
        assert client.provider_name == "openai"

    def test_auto_detect_raises_when_no_key_set(self):
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY")}
        with patch.dict(os.environ, clean, clear=True):
            with pytest.raises(EnvironmentError, match="No LLM provider detected"):
                LLMClient()

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMClient(provider="nonexistent")

    def test_missing_openai_sdk_raises_import_error(self):
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError):
                LLMClient(provider="openai")

    def test_missing_groq_sdk_raises_import_error(self):
        with patch.dict(sys.modules, {"groq": None}):
            with pytest.raises(ImportError):
                LLMClient(provider="groq")


# ===========================================================================
# 2. Simple string prompt
# ===========================================================================

class TestSimpleStringPrompt:

    def test_generate_returns_content(self):
        client = _make_client()
        resp = _make_response("Paris")
        _attach_mock_provider(client, resp)

        result = client.generate("What is the capital of France?", model="gpt-4o")

        assert result.content == "Paris"

    def test_generate_returns_correct_model(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response(model="gpt-4o"))

        result = client.generate("What is the capital of France?", model="gpt-4o")

        assert result.model == "gpt-4o"

    def test_generate_returns_usage(self):
        client = _make_client()
        usage = Usage(prompt_tokens=8, completion_tokens=3, total_tokens=11)
        _attach_mock_provider(client, _make_response(usage=usage))

        result = client.generate("What is the capital of France?", model="gpt-4o")

        assert result.usage.prompt_tokens == 8
        assert result.usage.completion_tokens == 3
        assert result.usage.total_tokens == 11

    def test_generate_provider_called_with_correct_request(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("What is the capital of France?", model="gpt-4o")

        mock.generate.assert_called_once()
        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.messages[0].content == "What is the capital of France?"
        assert req.messages[0].role == "user"
        assert req.model == "gpt-4o"

    def test_generate_result_is_generate_response_instance(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        result = client.generate("Hello", model="gpt-4o")

        assert isinstance(result, GenerateResponse)

    def test_generate_empty_content_response(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response(content=""))

        result = client.generate("Hello", model="gpt-4o")

        assert result.content == ""

    def test_generate_long_content_response(self):
        long_text = "word " * 500
        client = _make_client()
        _attach_mock_provider(client, _make_response(content=long_text))

        result = client.generate("Write an essay", model="gpt-4o")

        assert result.content == long_text


# ===========================================================================
# 3. Generate with options (temperature, max_tokens)
# ===========================================================================

class TestGenerateWithOptions:

    def test_temperature_forwarded_to_request(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Write a sentence about oxygen", model="gpt-4o",
                        temperature=0.9, max_tokens=100)

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.temperature == 0.9
        assert req.max_tokens == 100

    def test_zero_temperature(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Hello", model="gpt-4o", temperature=0.0)

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.temperature == 0.0

    def test_max_temperature(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Hello", model="gpt-4o", temperature=2.0)

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.temperature == 2.0

    def test_small_max_tokens(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Hello", model="gpt-4o", max_tokens=1)

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.max_tokens == 1

    def test_extra_kwargs_forwarded(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Hello", model="gpt-4o", top_p=0.8)

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.extra.get("top_p") == 0.8

    def test_default_temperature_is_07(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Hello", model="gpt-4o")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.temperature == 0.7

    def test_default_max_tokens_is_1024(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Hello", model="gpt-4o")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.max_tokens == 1024


# ===========================================================================
# 4. Multi-turn conversation
# ===========================================================================

class TestMultiTurnConversation:

    def _messages(self):
        return [
            Message(role="user",      content="My name is Rithika."),
            Message(role="assistant", content="Nice to meet you, Rithika!"),
            Message(role="user",      content="What's my name?"),
        ]

    def test_all_messages_forwarded(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response("Your name is Rithika."))

        client.generate(self._messages(), model="gpt-4o",
                        system="You are a helpful assistant.")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert len(req.messages) == 3
        assert req.messages[0].role == "user"
        assert req.messages[1].role == "assistant"
        assert req.messages[2].role == "user"

    def test_system_prompt_forwarded(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate(self._messages(), model="gpt-4o",
                        system="You are a helpful assistant.")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.system == "You are a helpful assistant."

    def test_response_content_correct(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response("Your name is Rithika."))

        result = client.generate(self._messages(), model="gpt-4o")

        assert result.content == "Your name is Rithika."

    def test_message_content_preserved(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate(self._messages(), model="gpt-4o")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.messages[0].content == "My name is Rithika."
        assert req.messages[1].content == "Nice to meet you, Rithika!"
        assert req.messages[2].content == "What's my name?"

    def test_single_message_list(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate([Message(role="user", content="Hi")], model="gpt-4o")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert len(req.messages) == 1

    def test_empty_message_list(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate([], model="gpt-4o")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.messages == []

    def test_no_system_prompt_defaults_to_none(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate(self._messages(), model="gpt-4o")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.system is None


# ===========================================================================
# 5. Streaming
# ===========================================================================

class TestStreaming:

    def test_stream_yields_chunks(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        chunks = list(client.stream("Count from 1 to 5 slowly.", model="gpt-4o"))

        assert len(chunks) == 2

    def test_stream_chunk_deltas(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        chunks = list(client.stream("Count from 1 to 5 slowly.", model="gpt-4o"))

        assert chunks[0].delta == "chunk1"
        assert chunks[1].delta == "chunk2"

    def test_stream_last_chunk_is_finished(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        chunks = list(client.stream("Count from 1 to 5 slowly.", model="gpt-4o"))

        assert chunks[-1].finished is True

    def test_stream_first_chunk_not_finished(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        chunks = list(client.stream("test", model="gpt-4o"))

        assert chunks[0].finished is False

    def test_stream_returns_iterator(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        result = client.stream("test", model="gpt-4o")

        assert hasattr(result, "__iter__")

    def test_stream_concatenated_deltas(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        full_text = "".join(
            chunk.delta or ""
            for chunk in client.stream("test", model="gpt-4o")
        )

        assert full_text == "chunk1chunk2"

    def test_stream_with_system_prompt(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        list(client.stream("test", model="gpt-4o", system="Be brief."))

        # stream was called (provider's stream method was invoked)

    def test_stream_with_temperature(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        # Should not raise
        chunks = list(client.stream("test", model="gpt-4o", temperature=0.5))
        assert len(chunks) > 0

    def test_astream_async(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        async def collect():
            return [c async for c in client.astream("test", model="gpt-4o")]

        chunks = asyncio.run(collect())
        assert chunks[0].delta == "chunk1"
        assert chunks[1].delta == "chunk2"

    def test_stream_empty_delta_chunks(self):
        client = _make_client()
        mock = MagicMock()
        mock.generate = AsyncMock(return_value=_make_response())

        async def _stream_with_empty(_req):
            yield StreamChunk(delta="",  finished=False)
            yield StreamChunk(delta="hi", finished=False)
            yield StreamChunk(delta="",  finished=True)

        mock.stream = _stream_with_empty
        client._provider = mock

        chunks = list(client.stream("test", model="gpt-4o"))
        assert len(chunks) == 3
        assert chunks[0].delta == ""
        assert chunks[1].delta == "hi"

    def test_stream_many_chunks(self):
        client = _make_client()
        mock = MagicMock()
        mock.generate = AsyncMock(return_value=_make_response())

        async def _many(_req):
            for i in range(20):
                yield StreamChunk(delta=str(i), finished=(i == 19))

        mock.stream = _many
        client._provider = mock

        chunks = list(client.stream("test", model="gpt-4o"))
        assert len(chunks) == 20
        assert chunks[-1].finished is True


# ===========================================================================
# 6. Groq provider
# ===========================================================================

class TestGroqProvider:

    def test_groq_client_initialises(self):
        client = _make_client("groq")
        assert client.provider_name == "groq"

    def test_groq_generate_returns_response(self):
        client = _make_client("groq")
        _attach_mock_provider(client, _make_response("Hello from Groq!", model="llama3"))

        result = client.generate("Hello from Groq!", model="groq/compound-mini")

        assert result.content == "Hello from Groq!"

    def test_groq_generate_correct_model_forwarded(self):
        client = _make_client("groq")
        mock = _attach_mock_provider(client, _make_response(model="compound-mini"))

        client.generate("Hello", model="groq/compound-mini")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.model == "groq/compound-mini"

    def test_groq_stream_yields_chunks(self):
        client = _make_client("groq")
        _attach_mock_provider(client, _make_response())

        chunks = list(client.stream("test", model="groq/compound-mini"))
        assert len(chunks) > 0

    def test_groq_no_usage_returned(self):
        client = _make_client("groq")
        _attach_mock_provider(client, _make_response(usage=None))

        result = client.generate("Hello", model="groq/compound-mini")

        assert result.usage is None

    def test_groq_with_system_prompt(self):
        client = _make_client("groq")
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Hello", model="groq/compound-mini",
                        system="You are a Groq assistant.")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.system == "You are a Groq assistant."

    def test_groq_provider_error_wrapped_in_runtime_error(self):
        client = _make_client("groq")
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=Exception("Groq upstream failure"))
        client._provider = mock

        with pytest.raises(RuntimeError, match="LLM generation failed"):
            client.generate("Hello", model="groq/compound-mini")

    def test_groq_missing_sdk_raises_import_error(self):
        with patch.dict(sys.modules, {"groq": None}):
            with pytest.raises(ImportError):
                LLMClient(provider="groq")


# ===========================================================================
# 7. Gemini provider (via client.use())
# ===========================================================================

class TestGeminiProvider:

    def _gemini_client(self) -> LLMClient:
        openai_client = _make_client("openai")
        with patch("llmx.providers.gemini.GeminiProvider.__init__", return_value=None):
            return openai_client.use("gemini")

    def test_use_returns_new_client(self):
        openai_client = _make_client("openai")
        with patch("llmx.providers.gemini.GeminiProvider.__init__", return_value=None):
            gemini_client = openai_client.use("gemini")

        assert gemini_client is not openai_client

    def test_use_sets_correct_provider_name(self):
        client = self._gemini_client()
        assert client.provider_name == "gemini"

    def test_gemini_generate_returns_response(self):
        client = self._gemini_client()
        _attach_mock_provider(client, _make_response("Hello from Gemini!", model="gemini-2.5-flash"))

        result = client.generate("Hello from Gemini!", model="gemini-2.5-flash")

        assert result.content == "Hello from Gemini!"

    def test_gemini_model_forwarded(self):
        client = self._gemini_client()
        mock = _attach_mock_provider(client, _make_response(model="gemini-2.5-flash"))

        client.generate("Hello", model="gemini-2.5-flash")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.model == "gemini-2.5-flash"

    def test_gemini_no_usage_field(self):
        client = self._gemini_client()
        _attach_mock_provider(client, _make_response(usage=None))

        result = client.generate("Hello", model="gemini-2.5-flash")

        assert result.usage is None

    def test_gemini_stream_yields_chunks(self):
        client = self._gemini_client()
        _attach_mock_provider(client, _make_response())

        chunks = list(client.stream("test", model="gemini-2.5-flash"))
        assert len(chunks) > 0

    def test_gemini_provider_error_wrapped(self):
        client = self._gemini_client()
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=Exception("Gemini upstream failure"))
        client._provider = mock

        with pytest.raises(RuntimeError, match="LLM generation failed"):
            client.generate("Hello", model="gemini-2.5-flash")

    def test_original_client_unchanged_after_use(self):
        openai_client = _make_client("openai")
        with patch("llmx.providers.gemini.GeminiProvider.__init__", return_value=None):
            openai_client.use("gemini")

        assert openai_client.provider_name == "openai"

    def test_chained_use_calls(self):
        openai_client = _make_client("openai")
        with patch("llmx.providers.groq.GroqProvider.__init__", return_value=None):
            groq_client = openai_client.use("groq")
        with patch("llmx.providers.gemini.GeminiProvider.__init__", return_value=None):
            gemini_client = groq_client.use("gemini")

        assert gemini_client.provider_name == "gemini"
        assert groq_client.provider_name == "groq"


# ===========================================================================
# 8. Tool calling
# ===========================================================================

class TestToolCalling:

    def _tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

    def test_tool_call_returned_in_response(self):
        client = _make_client()
        tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Tokyo"})
        _attach_mock_provider(client, _make_response(content="", tool_calls=[tc]))

        result = client.generate("What's the weather in Tokyo?",
                                 model="gpt-4o", tools=self._tools())

        assert len(result.tool_calls) == 1

    def test_tool_call_name_correct(self):
        client = _make_client()
        tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Tokyo"})
        _attach_mock_provider(client, _make_response(content="", tool_calls=[tc]))

        result = client.generate("What's the weather in Tokyo?",
                                 model="gpt-4o", tools=self._tools())

        assert result.tool_calls[0].name == "get_weather"

    def test_tool_call_arguments_correct(self):
        client = _make_client()
        tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Tokyo"})
        _attach_mock_provider(client, _make_response(content="", tool_calls=[tc]))

        result = client.generate("What's the weather in Tokyo?",
                                 model="gpt-4o", tools=self._tools())

        assert result.tool_calls[0].arguments == {"city": "Tokyo"}

    def test_tool_call_id_preserved(self):
        client = _make_client()
        tc = ToolCall(id="call_abc123", name="get_weather", arguments={"city": "Tokyo"})
        _attach_mock_provider(client, _make_response(content="", tool_calls=[tc]))

        result = client.generate("test", model="gpt-4o", tools=self._tools())

        assert result.tool_calls[0].id == "call_abc123"

    def test_tools_forwarded_to_request(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("test", model="gpt-4o", tools=self._tools())

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.tools == self._tools()

    def test_no_tool_calls_when_model_responds_with_text(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response(
            content="I cannot check the weather.", tool_calls=[]))

        result = client.generate("What's the weather in Tokyo?",
                                 model="gpt-4o", tools=self._tools())

        assert result.tool_calls == []
        assert result.content == "I cannot check the weather."

    def test_multiple_tool_calls_in_response(self):
        client = _make_client()
        tc1 = ToolCall(id="c1", name="get_weather", arguments={"city": "Tokyo"})
        tc2 = ToolCall(id="c2", name="get_weather", arguments={"city": "Paris"})
        _attach_mock_provider(client, _make_response(content="", tool_calls=[tc1, tc2]))

        result = client.generate("Weather in Tokyo and Paris?",
                                 model="gpt-4o", tools=self._tools())

        assert len(result.tool_calls) == 2
        cities = [tc.arguments["city"] for tc in result.tool_calls]
        assert "Tokyo" in cities
        assert "Paris" in cities

    def test_no_tools_param_means_none_in_request(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Hello", model="gpt-4o")

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.tools is None

    def test_empty_tools_list(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        client.generate("Hello", model="gpt-4o", tools=[])

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.tools == []

    def test_tool_call_with_multiple_arguments(self):
        client = _make_client()
        tc = ToolCall(
            id="c1",
            name="book_flight",
            arguments={"origin": "NYC", "destination": "LAX", "date": "2025-01-01"}
        )
        _attach_mock_provider(client, _make_response(content="", tool_calls=[tc]))

        result = client.generate("Book a flight", model="gpt-4o",
                                 tools=[{"type": "function", "function": {"name": "book_flight"}}])

        assert result.tool_calls[0].arguments["origin"] == "NYC"
        assert result.tool_calls[0].arguments["destination"] == "LAX"


# ===========================================================================
# 9. Error handling
# ===========================================================================

class TestErrorHandling:

    def test_invalid_prompt_type_raises_type_error(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        with pytest.raises(TypeError):
            client.generate(12345, model="gpt-4o")

    def test_none_prompt_raises_type_error(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        with pytest.raises(TypeError):
            client.generate(None, model="gpt-4o")

    def test_dict_prompt_raises_type_error(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        with pytest.raises(TypeError):
            client.generate({"role": "user", "content": "hi"}, model="gpt-4o")

    def test_provider_exception_wrapped_in_runtime_error(self):
        client = _make_client()
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=Exception("upstream failure"))
        client._provider = mock

        with pytest.raises(RuntimeError, match="LLM generation failed"):
            client.generate("Hello", model="gpt-4o")

    def test_runtime_error_chained_from_original(self):
        client = _make_client()
        original = Exception("root cause")
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=original)
        client._provider = mock

        with pytest.raises(RuntimeError) as exc_info:
            client.generate("Hello", model="gpt-4o")

        assert exc_info.value.__cause__ is original

    def test_stream_provider_exception_wrapped(self):
        client = _make_client()
        mock = MagicMock()
        mock.generate = AsyncMock(return_value=_make_response())
        mock.stream = MagicMock(side_effect=Exception("stream failure"))
        client._provider = mock

        with pytest.raises(RuntimeError, match="LLM streaming failed"):
            list(client.stream("Hello", model="gpt-4o"))

    def test_value_error_from_provider_not_wrapped(self):
        """TypeError and ValueError bubble up directly — not wrapped in RuntimeError."""
        client = _make_client()
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=ValueError("bad value"))
        client._provider = mock

        with pytest.raises(ValueError):
            client.generate("Hello", model="gpt-4o")

    def test_agenerate_invalid_prompt_raises_type_error(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        with pytest.raises(TypeError):
            asyncio.run(client.agenerate(99))

    def test_astream_invalid_prompt_raises_type_error(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        async def collect():
            return [c async for c in client.astream(99)]

        with pytest.raises(TypeError):
            asyncio.run(collect())


# ===========================================================================
# 10. GenerateRequest passthrough
# ===========================================================================

class TestGenerateRequestPassthrough:

    def test_generate_request_passed_directly(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        req = GenerateRequest(
            messages=[Message(role="user", content="direct")],
            model="gpt-4o",
            temperature=0.1,
        )
        client.generate(req)

        called_req: GenerateRequest = mock.generate.call_args[0][0]
        assert called_req is req

    def test_generate_request_not_mutated(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response())

        req = GenerateRequest(
            messages=[Message(role="user", content="direct")],
            model="gpt-4o",
            temperature=0.1,
        )
        original_temp = req.temperature
        client.generate(req)

        assert req.temperature == original_temp

    def test_generate_request_extra_preserved(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gpt-4o",
            extra={"top_p": 0.95, "presence_penalty": 0.1},
        )
        client.generate(req)

        called_req: GenerateRequest = mock.generate.call_args[0][0]
        assert called_req.extra["top_p"] == 0.95
        assert called_req.extra["presence_penalty"] == 0.1


# ===========================================================================
# 11. Async generate
# ===========================================================================

class TestAsyncGenerate:

    def test_agenerate_string_prompt(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response("async result"))

        result = asyncio.run(client.agenerate("Hello", model="gpt-4o"))

        assert result.content == "async result"

    def test_agenerate_list_prompt(self):
        client = _make_client()
        _attach_mock_provider(client, _make_response("async result"))

        result = asyncio.run(
            client.agenerate([Message(role="user", content="Hello")], model="gpt-4o")
        )

        assert result.content == "async result"

    def test_agenerate_with_system(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        asyncio.run(client.agenerate("Hello", model="gpt-4o", system="Be brief."))

        req: GenerateRequest = mock.generate.call_args[0][0]
        assert req.system == "Be brief."

    def test_agenerate_called_once(self):
        client = _make_client()
        mock = _attach_mock_provider(client, _make_response())

        asyncio.run(client.agenerate("Hello", model="gpt-4o"))

        mock.generate.assert_called_once()

    def test_agenerate_with_tools(self):
        client = _make_client()
        tc = ToolCall(id="c1", name="fn", arguments={})
        mock = _attach_mock_provider(client, _make_response(tool_calls=[tc]))

        result = asyncio.run(client.agenerate(
            "Call a tool", model="gpt-4o",
            tools=[{"type": "function", "function": {"name": "fn"}}]
        ))

        assert len(result.tool_calls) == 1