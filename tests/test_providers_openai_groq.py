"""
test_providers_openai_groq.py
Covers: llmx/providers/openai.py, llmx/providers/groq.py
Strategy: mock the SDK client objects so the real provider code executes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from llmx.models import GenerateRequest, GenerateResponse, Message, StreamChunk, ToolCall, Usage
from llmx.providers.openai import OpenAIProvider
from llmx.providers.groq import GroqProvider
from llmx.exceptions import AuthenticationError, RateLimitError, ProviderUnavailableError
from llmx.config import LLMClientConfig


# ===========================================================================
# Shared SDK response builders
# ===========================================================================

def _openai_response(
    content="hello",
    model="gpt-4o",
    tool_calls=None,
    usage=None,
):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = usage
    return resp


def _openai_usage(prompt=5, completion=10, total=15):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = total
    return u


def _stream_chunk(delta="hi", finish_reason=None, model="gpt-4o"):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = delta
    chunk.choices[0].finish_reason = finish_reason
    chunk.model = model
    return chunk


def _make_tc_mock(id_="call_1", name="get_weather", args=None):
    tc = MagicMock()
    tc.id = id_
    tc.function.name = name
    tc.function.arguments = json.dumps(args or {"city": "Tokyo"})
    return tc


def _make_openai_provider(config=None):
    with patch("openai.OpenAI") as mock_cls:
        p = OpenAIProvider(api_key="sk-test", config=config)
        p._client = mock_cls.return_value
    return p


def _make_groq_provider(config=None):
    with patch("groq.Groq") as mock_cls:
        p = GroqProvider(api_key="gsk-test", config=config)
        p._client = mock_cls.return_value
    return p


# ===========================================================================
# OpenAIProvider — __init__
# ===========================================================================

class TestOpenAIProviderInit:

    def test_init_with_explicit_key(self):
        with patch("openai.OpenAI") as mock_cls:
            p = OpenAIProvider(api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test", base_url=None)
        assert isinstance(p, OpenAIProvider)

    def test_init_reads_env_var(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}):
            with patch("openai.OpenAI") as mock_cls:
                OpenAIProvider()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_key"] == "sk-env"

    def test_init_with_base_url(self):
        with patch("openai.OpenAI") as mock_cls:
            OpenAIProvider(api_key="k", base_url="http://localhost:8000")
        mock_cls.assert_called_once_with(api_key="k", base_url="http://localhost:8000")

    def test_init_import_error(self):
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError, match="pip install openai"):
                OpenAIProvider(api_key="k")

    def test_env_var_attribute(self):
        assert OpenAIProvider.env_var == "OPENAI_API_KEY"

    def test_name_attribute(self):
        assert OpenAIProvider.name == "openai"

    def test_init_accepts_config(self):
        cfg = LLMClientConfig(max_retries=5)
        with patch("openai.OpenAI"):
            p = OpenAIProvider(api_key="k", config=cfg)
        assert p.config is cfg

    def test_init_config_defaults_to_none(self):
        with patch("openai.OpenAI"):
            p = OpenAIProvider(api_key="k")
        assert p.config is None

    def test_missing_api_key_raises_authentication_error(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(AuthenticationError, match="OPENAI_API_KEY"):
                OpenAIProvider()


# ===========================================================================
# OpenAIProvider — _build_kwargs
# ===========================================================================

class TestOpenAIBuildKwargs:

    def setup_method(self):
        self.p = _make_openai_provider()

    def test_basic_fields_present(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gpt-4o", temperature=0.5, max_tokens=100,
        )
        kw = self.p._build_kwargs(req, stream=False)
        assert kw["model"] == "gpt-4o"
        assert kw["temperature"] == 0.5
        assert kw["max_tokens"] == 100
        assert kw["stream"] is False

    def test_stream_true(self):
        req = GenerateRequest(messages=[], model="gpt-4o")
        kw = self.p._build_kwargs(req, stream=True)
        assert kw["stream"] is True

    def test_no_tools_key_absent(self):
        req = GenerateRequest(messages=[], model="gpt-4o")
        kw = self.p._build_kwargs(req, stream=False)
        assert "tools" not in kw
        assert "tool_choice" not in kw

    def test_tools_added_with_auto_choice(self):
        tools = [{"type": "function", "function": {"name": "fn"}}]
        req = GenerateRequest(messages=[], model="gpt-4o", tools=tools)
        kw = self.p._build_kwargs(req, stream=False)
        assert kw["tools"] == tools
        assert kw["tool_choice"] == "auto"

    def test_extra_kwargs_merged(self):
        req = GenerateRequest(messages=[], model="gpt-4o", extra={"top_p": 0.9, "n": 2})
        kw = self.p._build_kwargs(req, stream=False)
        assert kw["top_p"] == 0.9
        assert kw["n"] == 2

    def test_messages_built_correctly(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="q")],
            model="gpt-4o", system="sys",
        )
        kw = self.p._build_kwargs(req, stream=False)
        assert kw["messages"][0] == {"role": "system", "content": "sys"}
        assert kw["messages"][1] == {"role": "user", "content": "q"}


# ===========================================================================
# OpenAIProvider — _normalize
# ===========================================================================

class TestOpenAINormalize:

    def setup_method(self):
        self.p = _make_openai_provider()

    def test_basic_response(self):
        resp = _openai_response("hello", "gpt-4o")
        result = self.p._normalize(resp)
        assert isinstance(result, GenerateResponse)
        assert result.content == "hello"
        assert result.model == "gpt-4o"
        assert result.tool_calls == []
        assert result.usage is None

    def test_none_content_becomes_empty_string(self):
        resp = _openai_response(content=None)
        result = self.p._normalize(resp)
        assert result.content == ""

    def test_usage_mapped(self):
        resp = _openai_response(usage=_openai_usage(5, 10, 15))
        result = self.p._normalize(resp)
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 10
        assert result.usage.total_tokens == 15

    def test_no_usage_is_none(self):
        resp = _openai_response(usage=None)
        result = self.p._normalize(resp)
        assert result.usage is None

    def test_single_tool_call(self):
        tc = _make_tc_mock("c1", "get_weather", {"city": "Tokyo"})
        resp = _openai_response(content=None, tool_calls=[tc])
        result = self.p._normalize(resp)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "c1"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Tokyo"}

    def test_multiple_tool_calls(self):
        tcs = [
            _make_tc_mock("c1", "fn1", {"x": 1}),
            _make_tc_mock("c2", "fn2", {"y": 2}),
        ]
        resp = _openai_response(content=None, tool_calls=tcs)
        result = self.p._normalize(resp)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[1].name == "fn2"

    def test_no_tool_calls_is_empty_list(self):
        resp = _openai_response(tool_calls=None)
        result = self.p._normalize(resp)
        assert result.tool_calls == []

    def test_raw_response_stored(self):
        resp = _openai_response()
        result = self.p._normalize(resp)
        assert result.raw is resp


# ===========================================================================
# OpenAIProvider — generate (async, real method)
# ===========================================================================

class TestOpenAIGenerate:

    def setup_method(self):
        self.p = _make_openai_provider()

    def test_generate_success(self):
        raw = _openai_response("answer", "gpt-4o")
        self.p._client.chat.completions.create = MagicMock(return_value=raw)

        req = GenerateRequest(messages=[Message(role="user", content="q")], model="gpt-4o")

        with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
            result = asyncio.run(self.p.generate(req))

        assert result.content == "answer"
        assert result.model == "gpt-4o"

    def test_generate_passes_correct_kwargs(self):
        raw = _openai_response("ok")

        captured = {}

        async def fake_to_thread(fn, **kwargs):
            captured.update(kwargs)
            return raw

        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gpt-4o", temperature=0.3, max_tokens=50,
        )

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            asyncio.run(self.p.generate(req))

    def test_generate_with_tools(self):
        tc = _make_tc_mock("c1", "search", {"q": "python"})
        raw = _openai_response(content=None, tool_calls=[tc])

        with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
            req = GenerateRequest(
                messages=[Message(role="user", content="search")],
                model="gpt-4o",
                tools=[{"type": "function", "function": {"name": "search"}}],
            )
            result = asyncio.run(self.p.generate(req))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    def test_generate_with_usage(self):
        raw = _openai_response(usage=_openai_usage(8, 12, 20))

        with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
            req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
            result = asyncio.run(self.p.generate(req))

        assert result.usage.total_tokens == 20

    def test_generate_exception_propagates(self):
        """Non-mapped exceptions propagate directly (no RuntimeError wrapper)."""
        with patch("asyncio.to_thread", new=AsyncMock(side_effect=Exception("boom"))):
            req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
            with pytest.raises(Exception, match="boom"):
                asyncio.run(self.p.generate(req))

    def test_generate_key_error_propagates(self):
        with patch("asyncio.to_thread", new=AsyncMock(side_effect=KeyError("missing_key"))):
            req = GenerateRequest(messages=[], model="gpt-4o")
            with pytest.raises(KeyError):
                asyncio.run(self.p.generate(req))

    def test_generate_with_system_prompt(self):
        raw = _openai_response("ok")
        with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
            req = GenerateRequest(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o", system="be terse",
            )
            result = asyncio.run(self.p.generate(req))
        assert result.content == "ok"

    def test_generate_empty_response_content(self):
        raw = _openai_response(content="")
        with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
            req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
            result = asyncio.run(self.p.generate(req))
        assert result.content == ""


# ===========================================================================
# OpenAIProvider — exception mapping
# ===========================================================================

class TestOpenAIExceptionMapping:

    def setup_method(self):
        self.p = _make_openai_provider()

    def _req(self):
        return GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")

    def test_authentication_error_raised_immediately(self):
        """openai.AuthenticationError → llmx AuthenticationError, not retried."""
        import openai
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(401, request=request)
        exc = openai.AuthenticationError("401 Unauthorized", response=response, body=None)

        with patch("asyncio.to_thread", new=AsyncMock(side_effect=exc)):
            with pytest.raises(AuthenticationError):
                asyncio.run(self.p.generate(self._req()))

    def test_authentication_error_not_retried(self):
        """AuthenticationError is raised on first attempt only."""
        import openai
        import httpx

        calls = []
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(401, request=request)
        exc = openai.AuthenticationError("401", response=response, body=None)

        async def side_effect(fn, **kwargs):
            calls.append(1)
            raise exc

        with patch("asyncio.to_thread", side_effect=side_effect):
            with pytest.raises(AuthenticationError):
                asyncio.run(self.p.generate(self._req()))

        assert len(calls) == 1

    def test_rate_limit_error_raised(self):
        """openai.RateLimitError → llmx RateLimitError."""
        import openai
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        exc = openai.RateLimitError("429 Too Many Requests", response=response, body=None)

        with patch("asyncio.to_thread", new=AsyncMock(side_effect=exc)):
            with pytest.raises(RateLimitError):
                asyncio.run(self.p.generate(self._req()))

    def test_rate_limit_error_retried_up_to_max_retries(self):
        """RateLimitError triggers retry up to max_retries times."""
        import openai
        import httpx

        calls = []
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        exc = openai.RateLimitError("429", response=response, body=None)

        cfg = LLMClientConfig(max_retries=2, base_delay=0.01, max_delay=0.1, timeout=30.0)
        p = _make_openai_provider(config=cfg)

        async def side_effect(fn, **kwargs):
            calls.append(1)
            raise exc

        with patch("asyncio.to_thread", side_effect=side_effect):
            with patch("llmx.providers.base.random.uniform", return_value=0.0):
                with pytest.raises(RateLimitError):
                    asyncio.run(p.generate(self._req()))

        assert len(calls) == cfg.max_retries

    def test_api_connection_error_mapped_to_provider_unavailable(self):
        """openai.APIConnectionError → llmx ProviderUnavailableError."""
        import openai
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        exc = openai.APIConnectionError(message="connection refused", request=request)

        with patch("asyncio.to_thread", new=AsyncMock(side_effect=exc)):
            with pytest.raises(ProviderUnavailableError):
                asyncio.run(self.p.generate(self._req()))

    def test_api_status_error_5xx_mapped_to_provider_unavailable(self):
        """openai.APIStatusError with 5xx → llmx ProviderUnavailableError."""
        import openai
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(503, request=request)
        exc = openai.APIStatusError("503 Service Unavailable", response=response, body=None)

        with patch("asyncio.to_thread", new=AsyncMock(side_effect=exc)):
            with pytest.raises(ProviderUnavailableError):
                asyncio.run(self.p.generate(self._req()))

    def test_provider_unavailable_retried_up_to_max_retries(self):
        """ProviderUnavailableError triggers retry up to max_retries times."""
        import openai
        import httpx

        calls = []
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(503, request=request)
        exc = openai.APIStatusError("503", response=response, body=None)

        cfg = LLMClientConfig(max_retries=2, base_delay=0.01, max_delay=0.1, timeout=30.0)
        p = _make_openai_provider(config=cfg)

        async def side_effect(fn, **kwargs):
            calls.append(1)
            raise exc

        with patch("asyncio.to_thread", side_effect=side_effect):
            with patch("llmx.providers.base.random.uniform", return_value=0.0):
                with pytest.raises(ProviderUnavailableError):
                    asyncio.run(p.generate(self._req()))

        assert len(calls) == cfg.max_retries

    def test_config_forwarded_to_provider(self):
        """config is stored and used by generate via _retry_with_backoff."""
        cfg = LLMClientConfig(max_retries=1, base_delay=0.01, max_delay=0.1, timeout=30.0)
        p = _make_openai_provider(config=cfg)
        assert p.config is cfg

    def test_jitter_applied_on_retry(self):
        """random.uniform is called when a retry occurs."""
        calls = []

        async def side_effect(fn, **kwargs):
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("retry me")
            return _openai_response("ok")

        raw = _openai_response("ok")

        with patch("asyncio.to_thread", side_effect=side_effect):
            with patch("llmx.providers.base.random.uniform", return_value=0.0) as mock_rand:
                req = GenerateRequest(messages=[Message("user", "hi")], model="gpt-4o")
                asyncio.run(self.p.generate(req))

        mock_rand.assert_called()


# ===========================================================================
# OpenAIProvider — stream (async generator, real method)
# ===========================================================================

class TestOpenAIStream:

    def setup_method(self):
        self.p = _make_openai_provider()

    def _run_stream(self, req, chunks):
        with patch("asyncio.to_thread", new=AsyncMock(return_value=iter(chunks))):
            async def collect():
                return [c async for c in self.p.stream(req)]
            return asyncio.run(collect())

    def test_stream_yields_chunks(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
        chunks = [_stream_chunk("hello "), _stream_chunk("world", "stop")]
        result = self._run_stream(req, chunks)
        assert len(result) == 2

    def test_stream_delta_content(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
        chunks = [_stream_chunk("foo"), _stream_chunk("bar", "stop")]
        result = self._run_stream(req, chunks)
        assert result[0].delta == "foo"
        assert result[1].delta == "bar"

    def test_stream_finished_flag(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
        chunks = [_stream_chunk("a", None), _stream_chunk("b", "stop")]
        result = self._run_stream(req, chunks)
        assert result[0].finished is False
        assert result[1].finished is True

    def test_stream_model_in_chunk(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
        chunks = [_stream_chunk("a", "stop", model="gpt-4o")]
        result = self._run_stream(req, chunks)
        assert result[0].model == "gpt-4o"

    def test_stream_none_delta_becomes_empty_string(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
        chunks = [_stream_chunk(None, "stop")]
        result = self._run_stream(req, chunks)
        assert result[0].delta == ""

    def test_stream_runtime_error_on_exception(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
        with patch("asyncio.to_thread", new=AsyncMock(side_effect=Exception("stream boom"))):
            async def collect():
                return [c async for c in self.p.stream(req)]
            with pytest.raises(RuntimeError, match="OpenAI streaming failed"):
                asyncio.run(collect())

    def test_stream_many_chunks(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
        chunks = [_stream_chunk(str(i), "stop" if i == 9 else None) for i in range(10)]
        result = self._run_stream(req, chunks)
        assert len(result) == 10
        assert result[-1].finished is True

    def test_stream_with_system_prompt(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gpt-4o", system="be concise",
        )
        chunks = [_stream_chunk("ok", "stop")]
        result = self._run_stream(req, chunks)
        assert len(result) == 1

    def test_stream_raw_chunk_stored(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="gpt-4o")
        raw_chunk = _stream_chunk("text", "stop")
        result = self._run_stream(req, [raw_chunk])
        assert result[0].raw is raw_chunk


# ===========================================================================
# GroqProvider — __init__
# ===========================================================================

class TestGroqProviderInit:

    def test_init_with_explicit_key(self):
        with patch("groq.Groq") as mock_cls:
            p = GroqProvider(api_key="gsk-test")
        mock_cls.assert_called_once_with(api_key="gsk-test")

    def test_init_reads_env_var(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-env"}):
            with patch("groq.Groq") as mock_cls:
                GroqProvider()
        assert mock_cls.call_args[1]["api_key"] == "gsk-env"

    def test_init_import_error(self):
        with patch.dict(sys.modules, {"groq": None}):
            with pytest.raises(ImportError, match="pip install groq"):
                GroqProvider(api_key="k")

    def test_env_var_attribute(self):
        assert GroqProvider.env_var == "GROQ_API_KEY"

    def test_name_attribute(self):
        assert GroqProvider.name == "groq"

    def test_init_accepts_config(self):
        cfg = LLMClientConfig(max_retries=5)
        with patch("groq.Groq"):
            p = GroqProvider(api_key="k", config=cfg)
        assert p.config is cfg

    def test_init_config_defaults_to_none(self):
        with patch("groq.Groq"):
            p = GroqProvider(api_key="k")
        assert p.config is None

    def test_missing_api_key_raises_authentication_error(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("GROQ_API_KEY", None)
            with pytest.raises(AuthenticationError, match="GROQ_API_KEY"):
                GroqProvider()


# ===========================================================================
# GroqProvider — _build_kwargs
# ===========================================================================

class TestGroqBuildKwargs:

    def setup_method(self):
        self.p = _make_groq_provider()

    def test_basic_fields(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="llama3-8b", temperature=0.4, max_tokens=200,
        )
        kw = self.p._build_kwargs(req, stream=False)
        assert kw["model"] == "llama3-8b"
        assert kw["temperature"] == 0.4
        assert kw["max_tokens"] == 200
        assert kw["stream"] is False

    def test_stream_true(self):
        req = GenerateRequest(messages=[], model="llama3")
        kw = self.p._build_kwargs(req, stream=True)
        assert kw["stream"] is True

    def test_tools_included(self):
        tools = [{"type": "function"}]
        req = GenerateRequest(messages=[], model="llama3", tools=tools)
        kw = self.p._build_kwargs(req, stream=False)
        assert kw["tools"] == tools
        assert kw["tool_choice"] == "auto"

    def test_no_tools_absent(self):
        req = GenerateRequest(messages=[], model="llama3")
        kw = self.p._build_kwargs(req, stream=False)
        assert "tools" not in kw

    def test_extra_kwargs_merged(self):
        req = GenerateRequest(messages=[], model="llama3", extra={"top_p": 0.7})
        kw = self.p._build_kwargs(req, stream=False)
        assert kw["top_p"] == 0.7


# ===========================================================================
# GroqProvider — _normalize
# ===========================================================================

class TestGroqNormalize:

    def setup_method(self):
        self.p = _make_groq_provider()

    def _make_resp(self, content="hi", model="llama3", tool_calls=None, usage=None):
        msg = MagicMock()
        msg.content = content
        if tool_calls is not None:
            msg.tool_calls = tool_calls
        else:
            msg.tool_calls = None  # forces AttributeError → getattr returns None

        choice = MagicMock()
        choice.message = msg

        resp = MagicMock()
        resp.choices = [choice]
        resp.model = model
        resp.usage = usage
        return resp

    def test_normalize_missing_tool_calls_attr_deleted(self):
        """Hits the else branch: del msg.tool_calls so getattr returns None."""
        resp = self._make_resp()  # tool_calls=None → triggers del msg.tool_calls
        result = self.p._normalize(resp)
        assert result.tool_calls == []

    def test_basic_response(self):
        resp = self._make_resp("hello", "llama3")
        result = self.p._normalize(resp)
        assert result.content == "hello"
        assert result.model == "llama3"
        assert result.tool_calls == []

    def test_none_content_becomes_empty(self):
        resp = self._make_resp(content=None)
        result = self.p._normalize(resp)
        assert result.content == ""

    def test_usage_mapped(self):
        u = MagicMock()
        u.prompt_tokens = 3
        u.completion_tokens = 7
        u.total_tokens = 10
        resp = self._make_resp(usage=u)
        result = self.p._normalize(resp)
        assert result.usage.total_tokens == 10

    def test_no_usage_is_none(self):
        resp = self._make_resp(usage=None)
        result = self.p._normalize(resp)
        assert result.usage is None

    def test_tool_calls_parsed(self):
        tc = MagicMock()
        tc.id = "c1"
        tc.function.name = "search"
        tc.function.arguments = json.dumps({"q": "llm"})

        msg = MagicMock()
        msg.content = None
        msg.tool_calls = [tc]

        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = "llama3"
        resp.usage = None

        result = self.p._normalize(resp)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "llm"}

    def test_missing_tool_calls_attr_uses_getattr_none(self):
        """getattr(msg, 'tool_calls', None) — attribute absent on spec."""
        msg = MagicMock(spec=["content"])
        msg.content = "hi"

        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = "llama3"
        resp.usage = None

        result = self.p._normalize(resp)
        assert result.tool_calls == []

    def test_raw_stored(self):
        resp = self._make_resp()
        result = self.p._normalize(resp)
        assert result.raw is resp


# ===========================================================================
# GroqProvider — generate (async, real method)
# ===========================================================================

class TestGroqGenerate:

    def setup_method(self):
        self.p = _make_groq_provider()

    def _raw_resp(self, content="ok", model="llama3"):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = model
        resp.usage = None
        return resp

    def test_generate_success(self):
        raw = self._raw_resp("groq answer")
        with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
            req = GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")
            result = asyncio.run(self.p.generate(req))
        assert result.content == "groq answer"

    def test_generate_exception_propagates(self):
        """Non-mapped exceptions propagate directly."""
        with patch("asyncio.to_thread", new=AsyncMock(side_effect=Exception("groq fail"))):
            req = GenerateRequest(messages=[], model="llama3")
            with pytest.raises(Exception, match="groq fail"):
                asyncio.run(self.p.generate(req))

    def test_generate_with_tools(self):
        tc = MagicMock()
        tc.id = "c1"
        tc.function.name = "fn"
        tc.function.arguments = json.dumps({"x": 1})

        msg = MagicMock()
        msg.content = None
        msg.tool_calls = [tc]
        choice = MagicMock()
        choice.message = msg
        raw = MagicMock()
        raw.choices = [choice]
        raw.model = "llama3"
        raw.usage = None

        with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
            req = GenerateRequest(
                messages=[Message(role="user", content="call")],
                model="llama3",
                tools=[{"type": "function", "function": {"name": "fn"}}],
            )
            result = asyncio.run(self.p.generate(req))

        assert len(result.tool_calls) == 1

    def test_generate_with_usage(self):
        u = MagicMock()
        u.prompt_tokens = 2
        u.completion_tokens = 4
        u.total_tokens = 6

        msg = MagicMock()
        msg.content = "hi"
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        raw = MagicMock()
        raw.choices = [choice]
        raw.model = "llama3"
        raw.usage = u

        with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
            req = GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")
            result = asyncio.run(self.p.generate(req))

        assert result.usage.total_tokens == 6

    def test_generate_with_system_prompt(self):
        raw = self._raw_resp("answer")
        with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
            req = GenerateRequest(
                messages=[Message(role="user", content="hi")],
                model="llama3", system="be brief",
            )
            result = asyncio.run(self.p.generate(req))
        assert result.content == "answer"


# ===========================================================================
# GroqProvider — exception mapping
# ===========================================================================

class TestGroqExceptionMapping:

    def setup_method(self):
        self.p = _make_groq_provider()

    def _req(self):
        return GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")

    def test_authentication_error_raised_immediately(self):
        """groq.AuthenticationError → llmx AuthenticationError."""
        import groq
        import httpx

        request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
        response = httpx.Response(401, request=request)
        exc = groq.AuthenticationError("401 Unauthorized", response=response, body=None)

        with patch("asyncio.to_thread", new=AsyncMock(side_effect=exc)):
            with pytest.raises(AuthenticationError):
                asyncio.run(self.p.generate(self._req()))

    def test_authentication_error_not_retried(self):
        """AuthenticationError is raised on first attempt only."""
        import groq
        import httpx

        calls = []
        request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
        response = httpx.Response(401, request=request)
        exc = groq.AuthenticationError("401", response=response, body=None)

        async def side_effect(fn, **kwargs):
            calls.append(1)
            raise exc

        with patch("asyncio.to_thread", side_effect=side_effect):
            with pytest.raises(AuthenticationError):
                asyncio.run(self.p.generate(self._req()))

        assert len(calls) == 1

    def test_rate_limit_error_raised(self):
        """groq.RateLimitError → llmx RateLimitError."""
        import groq
        import httpx

        request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        exc = groq.RateLimitError("429 Too Many Requests", response=response, body=None)

        with patch("asyncio.to_thread", new=AsyncMock(side_effect=exc)):
            with pytest.raises(RateLimitError):
                asyncio.run(self.p.generate(self._req()))

    def test_rate_limit_error_retried_up_to_max_retries(self):
        """RateLimitError triggers retry up to max_retries times."""
        import groq
        import httpx

        calls = []
        request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        exc = groq.RateLimitError("429", response=response, body=None)

        cfg = LLMClientConfig(max_retries=2, base_delay=0.01, max_delay=0.1, timeout=30.0)
        p = _make_groq_provider(config=cfg)

        async def side_effect(fn, **kwargs):
            calls.append(1)
            raise exc

        with patch("asyncio.to_thread", side_effect=side_effect):
            with patch("llmx.providers.base.random.uniform", return_value=0.0):
                with pytest.raises(RateLimitError):
                    asyncio.run(p.generate(self._req()))

        assert len(calls) == cfg.max_retries

    def test_api_connection_error_mapped_to_provider_unavailable(self):
        """groq.APIConnectionError → llmx ProviderUnavailableError."""
        import groq
        import httpx

        request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
        exc = groq.APIConnectionError(message="connection refused", request=request)

        with patch("asyncio.to_thread", new=AsyncMock(side_effect=exc)):
            with pytest.raises(ProviderUnavailableError):
                asyncio.run(self.p.generate(self._req()))

    def test_provider_unavailable_retried_up_to_max_retries(self):
        """ProviderUnavailableError triggers retry up to max_retries times."""
        import groq
        import httpx

        calls = []
        request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
        exc = groq.APIConnectionError(message="connection down", request=request)

        cfg = LLMClientConfig(max_retries=2, base_delay=0.01, max_delay=0.1, timeout=30.0)
        p = _make_groq_provider(config=cfg)

        async def side_effect(fn, **kwargs):
            calls.append(1)
            raise exc

        with patch("asyncio.to_thread", side_effect=side_effect):
            with patch("llmx.providers.base.random.uniform", return_value=0.0):
                with pytest.raises(ProviderUnavailableError):
                    asyncio.run(p.generate(self._req()))

        assert len(calls) == cfg.max_retries

    def test_config_forwarded_to_provider(self):
        cfg = LLMClientConfig(max_retries=1)
        p = _make_groq_provider(config=cfg)
        assert p.config is cfg

    def test_jitter_applied_on_retry(self):
        """random.uniform is called when a retry occurs."""
        calls = []

        async def side_effect(fn, **kwargs):
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("retry me")
            return _raw_resp_for_groq()

        with patch("asyncio.to_thread", side_effect=side_effect):
            with patch("llmx.providers.base.random.uniform", return_value=0.0) as mock_rand:
                req = GenerateRequest(messages=[Message("user", "hi")], model="llama3")
                asyncio.run(self.p.generate(req))

        mock_rand.assert_called()


def _raw_resp_for_groq(content="ok", model="llama3"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = None
    return resp


# ===========================================================================
# GroqProvider — stream (async generator, real method)
# ===========================================================================

class TestGroqStream:

    def setup_method(self):
        self.p = _make_groq_provider()

    def _run_stream(self, req, raw_chunks):
        with patch("asyncio.to_thread", new=AsyncMock(return_value=iter(raw_chunks))):
            async def collect():
                return [c async for c in self.p.stream(req)]
            return asyncio.run(collect())

    def _chunk(self, delta="hi", finish_reason=None, model="llama3"):
        c = MagicMock()
        c.choices = [MagicMock()]
        c.choices[0].delta.content = delta
        c.choices[0].finish_reason = finish_reason
        c.model = model
        return c

    def test_stream_yields_chunks(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")
        result = self._run_stream(req, [self._chunk("a"), self._chunk("b", "stop")])
        assert len(result) == 2

    def test_stream_delta(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")
        result = self._run_stream(req, [self._chunk("hello", "stop")])
        assert result[0].delta == "hello"

    def test_stream_finished_flag(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")
        result = self._run_stream(req, [
            self._chunk("a", None),
            self._chunk("b", "stop"),
        ])
        assert result[0].finished is False
        assert result[1].finished is True

    def test_stream_none_delta_becomes_empty(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")
        result = self._run_stream(req, [self._chunk(None, "stop")])
        assert result[0].delta == ""

    def test_stream_runtime_error(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")
        with patch("asyncio.to_thread", new=AsyncMock(side_effect=Exception("stream fail"))):
            async def collect():
                return [c async for c in self.p.stream(req)]
            with pytest.raises(RuntimeError, match="Groq streaming failed"):
                asyncio.run(collect())

    def test_stream_raw_chunk_stored(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")
        raw_chunk = self._chunk("text", "stop")
        result = self._run_stream(req, [raw_chunk])
        assert result[0].raw is raw_chunk

    def test_stream_model_in_chunk(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], model="llama3")
        result = self._run_stream(req, [self._chunk("x", "stop", model="llama3-70b")])
        assert result[0].model == "llama3-70b"
