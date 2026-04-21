"""
test_core_and_registry.py
Covers: llmx/core.py, llmx/providers/__init__.py, llmx/__init__.py
Strategy: mock provider __init__ to bypass SDK; test the real LLMClient logic.
"""

from __future__ import annotations

import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from llmx import LLMClient, Message
from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
    ToolCall,
    Usage,
)
from llmx.providers import load_provider, PROVIDER_REGISTRY
from llmx.config import LLMClientConfig
from llmx.exceptions import (
    InvalidRequestError,
    LLMXError,
    NoProviderError,
    AmbiguousProviderError,
)
from llmx.providers import resolve_provider


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_response(content="ok", model="gpt-4o", usage=None, tool_calls=None):
    return GenerateResponse(
        content=content,
        model=model,
        usage=usage,
        tool_calls=tool_calls or [],
    )


PROVIDER_CLASS_NAMES = {
    "openai": "OpenAIProvider",
    "groq": "GroqProvider",
    "gemini": "GeminiProvider",
}

def _make_client(provider="openai", config=None):
    class_name = PROVIDER_CLASS_NAMES[provider]
    with patch(f"llmx.providers.{provider}.{class_name}.__init__", return_value=None):
        if config is not None:
            return LLMClient(provider=provider, config=config)
        return LLMClient(provider=provider)


def _attach(client, response, stream_chunks=None):
    """Attach an async mock provider to client."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value=response)

    chunks = stream_chunks or [
        StreamChunk(delta="a", finished=False),
        StreamChunk(delta="b", finished=True),
    ]

    async def _stream(_req):
        for c in chunks:
            yield c

    mock.stream = _stream
    client._provider = mock
    return mock


# ===========================================================================
# providers/__init__.py — PROVIDER_REGISTRY & load_provider
# ===========================================================================

class TestProviderRegistry:

    def test_registry_contains_openai(self):
        assert "openai" in PROVIDER_REGISTRY

    def test_registry_contains_groq(self):
        assert "groq" in PROVIDER_REGISTRY

    def test_registry_contains_gemini(self):
        assert "gemini" in PROVIDER_REGISTRY

    def test_registry_values_are_tuples(self):
        for name, val in PROVIDER_REGISTRY.items():
            assert isinstance(val, tuple)
            assert len(val) == 2

    def test_load_openai(self):
        with patch("llmx.providers.openai.OpenAIProvider.__init__", return_value=None):
            p = load_provider("openai")
        from llmx.providers.openai import OpenAIProvider
        assert isinstance(p, OpenAIProvider)

    def test_load_groq(self):
        with patch("llmx.providers.groq.GroqProvider.__init__", return_value=None):
            p = load_provider("groq")
        from llmx.providers.groq import GroqProvider
        assert isinstance(p, GroqProvider)

    def test_load_gemini(self):
        with patch("llmx.providers.gemini.GeminiProvider.__init__", return_value=None):
            p = load_provider("gemini")
        from llmx.providers.gemini import GeminiProvider
        assert isinstance(p, GeminiProvider)

    def test_load_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            load_provider("fantasy_llm")

    def test_load_unknown_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            load_provider("bad")
        msg = str(exc_info.value)
        assert "openai" in msg

    def test_load_passes_kwargs_to_constructor(self):
        with patch("llmx.providers.openai.OpenAIProvider.__init__", return_value=None) as mock_init:
            load_provider("openai", api_key="sk-test", base_url="http://x")
        mock_init.assert_called_once_with(config=None, api_key="sk-test", base_url="http://x")

    def test_load_passes_config_to_constructor(self):
        cfg = LLMClientConfig(max_retries=5)
        with patch("llmx.providers.openai.OpenAIProvider.__init__", return_value=None) as mock_init:
            load_provider("openai", config=cfg)
        mock_init.assert_called_once_with(config=cfg)


# ===========================================================================
# __init__.py — public exports
# ===========================================================================

class TestPublicAPI:

    def test_all_exports_present(self):
        import llmx
        for name in llmx.__all__:
            assert hasattr(llmx, name), f"Missing: {name}"

    def test_version(self):
        import llmx
        assert llmx.__version__ == "0.1.0"

    def test_llmclient_importable(self):
        from llmx import LLMClient as LC
        assert LC is LLMClient

    def test_all_models_importable(self):
        from llmx import (
            Message, GenerateRequest, GenerateResponse,
            StreamChunk, ToolCall, Usage,
        )
        assert all([Message, GenerateRequest, GenerateResponse, StreamChunk, ToolCall, Usage])

    def test_exceptions_importable(self):
        from llmx import (
            LLMXError, AuthenticationError, RateLimitError,
            ProviderUnavailableError, InvalidRequestError,
            NoProviderError, AmbiguousProviderError,
        )
        assert all([LLMXError, AuthenticationError, RateLimitError,
                    ProviderUnavailableError, InvalidRequestError,
                    NoProviderError, AmbiguousProviderError])

    def test_llmclientconfig_importable(self):
        from llmx import LLMClientConfig
        assert LLMClientConfig is not None


# ===========================================================================
# providers/__init__.py — resolve_provider
# ===========================================================================

class TestResolveProvider:

    def test_resolves_gpt_model_to_openai(self):
        with patch("llmx.providers.openai.OpenAIProvider.__init__", return_value=None):
            p = resolve_provider("gpt-4o")
        from llmx.providers.openai import OpenAIProvider
        assert isinstance(p, OpenAIProvider)

    def test_resolves_gemini_model_to_gemini(self):
        with patch("llmx.providers.gemini.GeminiProvider.__init__", return_value=None):
            p = resolve_provider("gemini-1.5-pro")
        from llmx.providers.gemini import GeminiProvider
        assert isinstance(p, GeminiProvider)

    def test_resolves_llama_model_to_groq(self):
        with patch("llmx.providers.groq.GroqProvider.__init__", return_value=None):
            p = resolve_provider("llama-3.1-70b-versatile")
        from llmx.providers.groq import GroqProvider
        assert isinstance(p, GroqProvider)

    def test_resolves_mixtral_to_groq(self):
        with patch("llmx.providers.groq.GroqProvider.__init__", return_value=None):
            p = resolve_provider("mixtral-8x7b-32768")
        from llmx.providers.groq import GroqProvider
        assert isinstance(p, GroqProvider)

    def test_resolves_o1_to_openai(self):
        with patch("llmx.providers.openai.OpenAIProvider.__init__", return_value=None):
            p = resolve_provider("o1-mini")
        from llmx.providers.openai import OpenAIProvider
        assert isinstance(p, OpenAIProvider)

    def test_raises_no_provider_error_for_unknown_model(self):
        with pytest.raises(NoProviderError, match="No registered provider"):
            resolve_provider("unknown-model-xyz-999")

    def test_no_provider_error_mentions_available(self):
        with pytest.raises(NoProviderError) as exc_info:
            resolve_provider("unknown-model-xyz-999")
        assert "openai" in str(exc_info.value)

    def test_raises_ambiguous_provider_error_on_multi_match(self):
        from llmx.providers.openai import OpenAIProvider
        from llmx.providers.groq import GroqProvider
        from llmx.providers.gemini import GeminiProvider

        with patch.object(OpenAIProvider, "supports_model", return_value=True), \
             patch.object(GroqProvider, "supports_model", return_value=True), \
             patch.object(GeminiProvider, "supports_model", return_value=False):
            with pytest.raises(AmbiguousProviderError, match="Multiple providers"):
                resolve_provider("ambiguous-model")

    def test_ambiguous_error_names_conflicting_providers(self):
        from llmx.providers.openai import OpenAIProvider
        from llmx.providers.groq import GroqProvider
        from llmx.providers.gemini import GeminiProvider

        with patch.object(OpenAIProvider, "supports_model", return_value=True), \
             patch.object(GroqProvider, "supports_model", return_value=True), \
             patch.object(GeminiProvider, "supports_model", return_value=False):
            with pytest.raises(AmbiguousProviderError) as exc_info:
                resolve_provider("ambiguous-model")
        msg = str(exc_info.value)
        assert "openai" in msg and "groq" in msg

    def test_resolve_passes_config_to_provider(self):
        cfg = LLMClientConfig(max_retries=9)
        with patch("llmx.providers.openai.OpenAIProvider.__init__", return_value=None) as mock_init:
            resolve_provider("gpt-4o", config=cfg)
        mock_init.assert_called_once_with(config=cfg)

    def test_resolve_passes_kwargs_to_provider(self):
        with patch("llmx.providers.openai.OpenAIProvider.__init__", return_value=None) as mock_init:
            resolve_provider("gpt-4o", api_key="sk-test")
        mock_init.assert_called_once_with(config=None, api_key="sk-test")


# ===========================================================================
# LLMClient — _to_request
# ===========================================================================

class TestToRequest:

    def test_string_prompt_wrapped_in_user_message(self):
        req = LLMClient._to_request(
            "hello", model="gpt-4", system=None,
            temperature=0.7, max_tokens=1024, extra={},
        )
        assert isinstance(req, GenerateRequest)
        assert req.messages[0].role == "user"
        assert req.messages[0].content == "hello"

    def test_list_prompt_used_directly(self):
        msgs = [Message(role="user", content="hi")]
        req = LLMClient._to_request(
            msgs, model=None, system=None,
            temperature=0.5, max_tokens=512, extra={},
        )
        assert req.messages is msgs

    def test_generate_request_passthrough(self):
        original = GenerateRequest(messages=[Message("user", "yo")])
        result = LLMClient._to_request(
            original, model="x", system=None,
            temperature=0.7, max_tokens=1024, extra={},
        )
        assert result is original

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="prompt must be"):
            LLMClient._to_request(
                123, model=None, system=None,
                temperature=0.7, max_tokens=1024, extra={},
            )

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            LLMClient._to_request(
                None, model=None, system=None,
                temperature=0.7, max_tokens=1024, extra={},
            )

    def test_model_forwarded(self):
        req = LLMClient._to_request(
            "hi", model="gpt-4o", system=None,
            temperature=0.7, max_tokens=1024, extra={},
        )
        assert req.model == "gpt-4o"

    def test_system_forwarded(self):
        req = LLMClient._to_request(
            "hi", model=None, system="be brief",
            temperature=0.7, max_tokens=1024, extra={},
        )
        assert req.system == "be brief"

    def test_temperature_forwarded(self):
        req = LLMClient._to_request(
            "hi", model=None, system=None,
            temperature=0.2, max_tokens=1024, extra={},
        )
        assert req.temperature == 0.2

    def test_max_tokens_forwarded(self):
        req = LLMClient._to_request(
            "hi", model=None, system=None,
            temperature=0.7, max_tokens=128, extra={},
        )
        assert req.max_tokens == 128

    def test_tools_forwarded(self):
        tools = [{"type": "function"}]
        req = LLMClient._to_request(
            "hi", model=None, system=None,
            temperature=0.7, max_tokens=1024, tools=tools, extra={},
        )
        assert req.tools == tools

    def test_extra_forwarded(self):
        req = LLMClient._to_request(
            "hi", model=None, system=None,
            temperature=0.7, max_tokens=1024, extra={"top_p": 0.9},
        )
        assert req.extra["top_p"] == 0.9

    def test_none_extra_becomes_empty_dict(self):
        req = LLMClient._to_request(
            "hi", model=None, system=None,
            temperature=0.7, max_tokens=1024, extra=None,
        )
        assert req.extra == {}

    def test_invalid_request_raises_invalid_request_error(self):
        """validate() is called in _to_request; bad params raise InvalidRequestError."""
        with pytest.raises(InvalidRequestError):
            LLMClient._to_request(
                "hi", model=None, system=None,
                temperature=99.0, max_tokens=1024, extra={},
            )

    def test_invalid_max_tokens_raises_invalid_request_error(self):
        with pytest.raises(InvalidRequestError):
            LLMClient._to_request(
                "hi", model=None, system=None,
                temperature=0.7, max_tokens=0, extra={},
            )


# ===========================================================================
# LLMClient — init, repr, properties, config
# ===========================================================================

class TestLLMClientInit:

    def test_explicit_provider_stored(self):
        c = _make_client("openai")
        assert c.provider_name == "openai"

    def test_provider_property_returns_provider(self):
        c = _make_client("openai")
        assert c.provider is c._provider

    def test_repr_format(self):
        c = _make_client("openai")
        assert repr(c) == "LLMClient(provider='openai')"

    def test_use_returns_new_client(self):
        c1 = _make_client("openai")
        with patch("llmx.providers.groq.GroqProvider.__init__", return_value=None):
            c2 = c1.use("groq")
        assert c2 is not c1
        assert c2.provider_name == "groq"

    def test_limiter_created_at_init(self):
        from aiolimiter import AsyncLimiter
        c = _make_client()
        assert isinstance(c._limiter, AsyncLimiter)

    def test_use_does_not_mutate_original(self):
        c1 = _make_client("openai")
        with patch("llmx.providers.groq.GroqProvider.__init__", return_value=None):
            c1.use("groq")
        assert c1.provider_name == "openai"

    def test_use_with_kwargs(self):
        c1 = _make_client("openai")
        with patch("llmx.providers.groq.GroqProvider.__init__", return_value=None) as mock_init:
            c1.use("groq", api_key="test")
        mock_init.assert_called_once_with(config=ANY, api_key="test")

    def test_accepts_config_parameter(self):
        cfg = LLMClientConfig(max_retries=5, rate_limit=20)
        c = _make_client(config=cfg)
        assert c.config is cfg

    def test_defaults_to_llmclientconfig_when_no_config(self):
        c = _make_client()
        assert isinstance(c.config, LLMClientConfig)

    def test_default_config_has_expected_defaults(self):
        c = _make_client()
        assert c.config.rate_limit == 10
        assert c.config.rate_limit_period == 1.0
        assert c.config.max_retries == 3

    def test_limiter_uses_config_rate_limit(self):
        cfg = LLMClientConfig(rate_limit=5, rate_limit_period=2.0)
        c = _make_client(config=cfg)
        assert c._limiter.max_rate == 5

    def test_limiter_uses_config_rate_limit_period(self):
        cfg = LLMClientConfig(rate_limit=15, rate_limit_period=3.0)
        c = _make_client(config=cfg)
        assert c._limiter.time_period == 3.0

    def test_config_forwarded_to_provider_on_init(self):
        cfg = LLMClientConfig(max_retries=7)
        with patch("llmx.providers.openai.OpenAIProvider.__init__", return_value=None) as mock_init:
            LLMClient(provider="openai", config=cfg)
        mock_init.assert_called_once_with(config=cfg)

    def test_provider_none_allowed_at_init(self):
        c = LLMClient(provider=None)
        assert c._provider is None
        assert c.provider_name is None

    def test_provider_none_repr(self):
        c = LLMClient(provider=None)
        assert repr(c) == "LLMClient(provider=None)"

    def test_explicit_provider_overrides_resolver(self):
        c = _make_client("openai")
        assert c._provider is not None
        assert c.provider_name == "openai"

    def test_resolve_returns_explicit_provider_regardless_of_model(self):
        c = _make_client("openai")
        resolved = c._resolve(None)
        assert resolved is c._provider

    def test_resolve_raises_value_error_when_no_provider_and_no_model(self):
        c = LLMClient(provider=None)
        with pytest.raises(ValueError, match="model must be specified"):
            c._resolve(None)

    def test_resolve_calls_resolve_provider_for_unknown_provider(self):
        c = LLMClient(provider=None)
        mock_provider = MagicMock()
        with patch("llmx.core.LLMClient._resolve", return_value=mock_provider):
            resolved = c._resolve("gpt-4o")
        assert resolved is mock_provider

    def test_resolver_caches_resolved_provider(self):
        c = LLMClient(provider=None)
        mock_provider = MagicMock()
        with patch("llmx.providers.resolve_provider", return_value=mock_provider) as mock_res:
            c._resolve("gpt-4o")
            c._resolve("gpt-4o")
        mock_res.assert_called_once()

    def test_resolver_resolves_at_generate_time(self):
        c = LLMClient(provider=None)
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=_make_response("resolved"))
        with patch("llmx.providers.resolve_provider", return_value=mock_provider):
            result = asyncio.run(c.agenerate("hi", model="gpt-4o"))
        assert result.content == "resolved"

    def test_no_provider_error_bubbles_from_agenerate(self):
        c = LLMClient(provider=None)
        with patch("llmx.providers.resolve_provider", side_effect=NoProviderError("no match")):
            with pytest.raises(NoProviderError):
                asyncio.run(c.agenerate("hi", model="unknown-xyz"))

    def test_ambiguous_provider_error_bubbles_from_agenerate(self):
        c = LLMClient(provider=None)
        with patch("llmx.providers.resolve_provider", side_effect=AmbiguousProviderError("ambig")):
            with pytest.raises(AmbiguousProviderError):
                asyncio.run(c.agenerate("hi", model="ambig-model"))


# ===========================================================================
# LLMClient — agenerate
# ===========================================================================

class TestAgenerate:

    def test_string_prompt(self):
        c = _make_client()
        _attach(c, _make_response("hello"))
        result = asyncio.run(c.agenerate("hi", model="gpt-4o"))
        assert result.content == "hello"

    def test_list_prompt(self):
        c = _make_client()
        _attach(c, _make_response("hi"))
        result = asyncio.run(c.agenerate([Message("user", "q")], model="gpt-4o"))
        assert result.content == "hi"

    def test_generate_request_passthrough(self):
        c = _make_client()
        mock = _attach(c, _make_response("ok"))
        req = GenerateRequest(messages=[Message("user", "q")])
        asyncio.run(c.agenerate(req))
        called = mock.generate.call_args[0][0]
        assert called is req

    def test_invalid_prompt_raises_type_error(self):
        c = _make_client()
        _attach(c, _make_response())
        with pytest.raises(TypeError):
            asyncio.run(c.agenerate(999))

    def test_provider_exception_wrapped(self):
        c = _make_client()
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=Exception("upstream"))
        c._provider = mock
        with pytest.raises(RuntimeError, match="LLM generation failed"):
            asyncio.run(c.agenerate("hi"))

    def test_value_error_not_wrapped(self):
        c = _make_client()
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=ValueError("bad"))
        c._provider = mock
        with pytest.raises(ValueError):
            asyncio.run(c.agenerate("hi"))

    def test_type_error_not_wrapped(self):
        c = _make_client()
        _attach(c, _make_response())
        with pytest.raises(TypeError):
            asyncio.run(c.agenerate(123))

    def test_runtime_error_chained(self):
        c = _make_client()
        cause = Exception("root")
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=cause)
        c._provider = mock
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(c.agenerate("hi"))
        assert exc_info.value.__cause__ is cause

    def test_awaits_coroutine_result(self):
        """Cover line 109: provider.generate returns a plain coroutine."""
        c = _make_client()
        expected = _make_response("coro-result")

        async def coro_gen(req):
            return expected

        mock = MagicMock()
        mock.generate = coro_gen
        c._provider = mock

        result = asyncio.run(c.agenerate("hi"))
        assert result.content == "coro-result"

    def test_plain_sync_return_value(self):
        """Cover line 109 else branch: provider.generate returns a plain non-coroutine."""
        c = _make_client()
        expected = _make_response("plain")

        mock = MagicMock()
        mock.generate = MagicMock(return_value=expected)  # plain sync return
        c._provider = mock

        result = asyncio.run(c.agenerate("hi"))
        assert result.content == "plain"

    def test_with_model_and_system(self):
        c = _make_client()
        mock = _attach(c, _make_response())
        asyncio.run(c.agenerate("hi", model="gpt-4o", system="be terse"))
        req = mock.generate.call_args[0][0]
        assert req.model == "gpt-4o"
        assert req.system == "be terse"

    def test_with_tools(self):
        c = _make_client()
        tc = ToolCall(id="c1", name="fn", arguments={})
        mock = _attach(c, _make_response(tool_calls=[tc]))
        result = asyncio.run(c.agenerate(
            "call", model="gpt-4o",
            tools=[{"type": "function", "function": {"name": "fn"}}],
        ))
        assert len(result.tool_calls) == 1

    def test_rate_limiter_acquired(self):
        c = _make_client()
        mock = _attach(c, _make_response())
        mock_limiter = MagicMock()
        mock_limiter.acquire = AsyncMock()
        c._limiter = mock_limiter
        asyncio.run(c.agenerate("hi"))
        mock_limiter.acquire.assert_called_once()

    def test_invalid_request_error_bubbles_up_unwrapped(self):
        """InvalidRequestError from validate() must NOT be wrapped in RuntimeError."""
        c = _make_client()
        _attach(c, _make_response())
        # temperature=99.0 will fail validate()
        with pytest.raises(InvalidRequestError):
            asyncio.run(c.agenerate("hi", temperature=99.0))

    def test_validate_called_automatically_on_agenerate(self):
        """validate() is triggered via _to_request for every agenerate call."""
        c = _make_client()
        _attach(c, _make_response())
        req = GenerateRequest(messages=[], max_tokens=-1)
        with pytest.raises(InvalidRequestError):
            asyncio.run(c.agenerate(req))

    def test_llmx_error_bubbles_unwrapped(self):
        """Any LLMXError subclass passes through agenerate without RuntimeError wrap."""
        from llmx.exceptions import ProviderUnavailableError
        c = _make_client()
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=ProviderUnavailableError("down"))
        c._provider = mock
        with pytest.raises(ProviderUnavailableError):
            asyncio.run(c.agenerate("hi"))


# ===========================================================================
# LLMClient — astream
# ===========================================================================

class TestAstream:

    def test_yields_chunks(self):
        c = _make_client()
        _attach(c, _make_response())

        async def collect():
            return [ch async for ch in c.astream("hi")]

        chunks = asyncio.run(collect())
        assert chunks[0].delta == "a"
        assert chunks[1].delta == "b"

    def test_last_chunk_finished(self):
        c = _make_client()
        _attach(c, _make_response())

        async def collect():
            return [ch async for ch in c.astream("hi")]

        chunks = asyncio.run(collect())
        assert chunks[-1].finished is True

    def test_invalid_prompt_raises_type_error(self):
        c = _make_client()
        _attach(c, _make_response())

        async def collect():
            return [ch async for ch in c.astream(42)]

        with pytest.raises(TypeError):
            asyncio.run(collect())

    def test_provider_exception_wrapped(self):
        c = _make_client()
        mock = MagicMock()
        mock.stream = MagicMock(side_effect=Exception("stream-fail"))
        c._provider = mock

        async def collect():
            return [ch async for ch in c.astream("hi")]

        with pytest.raises(RuntimeError, match="LLM streaming failed"):
            asyncio.run(collect())

    def test_sync_iterator_fallback(self):
        """Cover the else branch: provider.stream returns a sync iterator."""
        c = _make_client()
        sync_chunks = [StreamChunk(delta="x"), StreamChunk(delta="y", finished=True)]
        mock = MagicMock()
        mock.stream = MagicMock(return_value=iter(sync_chunks))
        c._provider = mock

        async def collect():
            return [ch async for ch in c.astream("hi")]

        chunks = asyncio.run(collect())
        assert chunks[0].delta == "x"
        assert chunks[1].finished is True

    def test_rate_limiter_acquired(self):
        c = _make_client()
        _attach(c, _make_response())
        mock_limiter = MagicMock()
        mock_limiter.acquire = AsyncMock()
        c._limiter = mock_limiter

        async def collect():
            return [ch async for ch in c.astream("hi")]

        asyncio.run(collect())
        mock_limiter.acquire.assert_called_once()

    def test_value_error_not_wrapped(self):
        c = _make_client()
        _attach(c, _make_response())

        async def collect():
            return [ch async for ch in c.astream(None)]

        with pytest.raises(TypeError):
            asyncio.run(collect())

    def test_invalid_request_error_bubbles_up_unwrapped(self):
        """InvalidRequestError from validate() must NOT be wrapped in RuntimeError."""
        c = _make_client()
        _attach(c, _make_response())

        async def collect():
            return [ch async for ch in c.astream("hi", temperature=99.0)]

        with pytest.raises(InvalidRequestError):
            asyncio.run(collect())

    def test_validate_called_automatically_on_astream(self):
        """validate() is triggered via _to_request for every astream call."""
        c = _make_client()
        _attach(c, _make_response())
        req = GenerateRequest(messages=[], max_tokens=0)

        async def collect():
            return [ch async for ch in c.astream(req)]

        with pytest.raises(InvalidRequestError):
            asyncio.run(collect())

    def test_llmx_error_bubbles_unwrapped(self):
        """LLMXError from provider.stream is re-raised without wrapping in RuntimeError."""
        from llmx.exceptions import ProviderUnavailableError
        c = _make_client()
        mock = MagicMock()
        mock.stream = MagicMock(side_effect=ProviderUnavailableError("service down"))
        c._provider = mock

        async def collect():
            return [ch async for ch in c.astream("hi")]

        with pytest.raises(ProviderUnavailableError):
            asyncio.run(collect())


# ===========================================================================
# LLMClient — sync wrappers: generate() and stream()
# ===========================================================================

class TestSyncWrappers:

    def test_generate_returns_response(self):
        c = _make_client()
        _attach(c, _make_response("sync answer"))
        result = c.generate("hi", model="gpt-4o")
        assert result.content == "sync answer"

    def test_generate_is_generate_response(self):
        c = _make_client()
        _attach(c, _make_response())
        assert isinstance(c.generate("hi"), GenerateResponse)

    def test_generate_all_params(self):
        c = _make_client()
        mock = _attach(c, _make_response())
        c.generate("hi", model="gpt-4o", system="s", temperature=0.3,
                   max_tokens=100, tools=[{"type": "function"}])
        req = mock.generate.call_args[0][0]
        assert req.system == "s"
        assert req.temperature == 0.3
        assert req.max_tokens == 100

    def test_stream_returns_iterator(self):
        c = _make_client()
        _attach(c, _make_response())
        result = c.stream("hi")
        assert hasattr(result, "__iter__")

    def test_stream_chunks_collected(self):
        c = _make_client()
        _attach(c, _make_response())
        chunks = list(c.stream("hi"))
        assert len(chunks) == 2

    def test_stream_delta_values(self):
        c = _make_client()
        _attach(c, _make_response())
        chunks = list(c.stream("hi"))
        assert chunks[0].delta == "a"
        assert chunks[1].delta == "b"

    def test_stream_all_params(self):
        c = _make_client()
        _attach(c, _make_response())
        chunks = list(c.stream("hi", model="gpt-4o", system="s",
                                temperature=0.5, max_tokens=50))
        assert len(chunks) > 0

    def test_generate_invalid_prompt_raises(self):
        c = _make_client()
        _attach(c, _make_response())
        with pytest.raises(TypeError):
            c.generate(999)

    def test_stream_invalid_prompt_raises(self):
        c = _make_client()
        _attach(c, _make_response())
        with pytest.raises(TypeError):
            list(c.stream(999))

    def test_generate_provider_error_wrapped(self):
        c = _make_client()
        mock = MagicMock()
        mock.generate = AsyncMock(side_effect=Exception("fail"))
        c._provider = mock
        with pytest.raises(RuntimeError):
            c.generate("hi")

    def test_stream_provider_error_wrapped(self):
        c = _make_client()
        mock = MagicMock()
        mock.stream = MagicMock(side_effect=Exception("fail"))
        c._provider = mock
        with pytest.raises(RuntimeError):
            list(c.stream("hi"))

    def test_generate_raises_in_async_context(self):
        c = _make_client()
        _attach(c, _make_response())

        async def call_sync():
            return c.generate("hi")

        with pytest.raises(RuntimeError, match="running event loop"):
            asyncio.run(call_sync())

    def test_stream_raises_in_async_context(self):
        c = _make_client()
        _attach(c, _make_response())

        async def call_sync():
            return list(c.stream("hi"))

        with pytest.raises(RuntimeError, match="running event loop"):
            asyncio.run(call_sync())
