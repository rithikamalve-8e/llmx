"""
test_core_and_registry.py
Covers: llmx/core.py, llmx/providers/__init__.py, llmx/__init__.py
Strategy: mock provider __init__ to bypass SDK; test the real LLMClient logic.
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import importlib

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

def _make_client(provider="openai"):
    class_name = PROVIDER_CLASS_NAMES[provider]
    with patch(f"llmx.providers.{provider}.{class_name}.__init__", return_value=None):
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
        mock_init.assert_called_once_with(api_key="sk-test", base_url="http://x")


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


# ===========================================================================
# LLMClient — _detect_provider
# ===========================================================================

class TestDetectProvider:

    def _clean_env(self):
        return {k: v for k, v in os.environ.items()
                if k not in ("OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY")}

    def test_detects_openai(self):
        with patch.dict(os.environ, {**self._clean_env(), "OPENAI_API_KEY": "sk-x"}, clear=True):
            with patch.dict("llmx.providers.PROVIDER_REGISTRY",
                            {"openai": PROVIDER_REGISTRY["openai"]}, clear=True):
                name = LLMClient._detect_provider()
        assert name == "openai"

    def test_detects_groq(self):
        with patch.dict(os.environ, {**self._clean_env(), "GROQ_API_KEY": "gsk-x"}, clear=True):
            with patch.dict("llmx.providers.PROVIDER_REGISTRY",
                            {"groq": PROVIDER_REGISTRY["groq"]}, clear=True):
                name = LLMClient._detect_provider()
        assert name == "groq"

    def test_detects_gemini(self):
        with patch.dict(os.environ, {**self._clean_env(), "GEMINI_API_KEY": "gai-x"}, clear=True):
            with patch.dict("llmx.providers.PROVIDER_REGISTRY",
                            {"gemini": PROVIDER_REGISTRY["gemini"]}, clear=True):
                name = LLMClient._detect_provider()
        assert name == "gemini"

    def test_raises_when_no_key_set(self):
        with patch.dict(os.environ, self._clean_env(), clear=True):
            with pytest.raises(EnvironmentError, match="No LLM provider detected"):
                LLMClient._detect_provider()

    def test_provider_with_no_env_var_attr_skipped(self):
        fake_cls = MagicMock(spec=[])  # no env_var attr → skipped cleanly
        real_import = importlib.import_module

        def selective_import(name):
            mod = real_import(name)
            if name == "llmx.providers.openai":
                fake_mod = MagicMock()
                fake_mod.OpenAIProvider = fake_cls
                return fake_mod
            return mod

        fake_registry = {"fake": ("llmx.providers.openai", "OpenAIProvider")}

        with patch.dict(os.environ, self._clean_env(), clear=True):
            with patch("importlib.import_module", side_effect=selective_import):
                with patch.dict("llmx.providers.PROVIDER_REGISTRY", fake_registry, clear=True):
                    with pytest.raises(EnvironmentError):
                        LLMClient._detect_provider()


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


# ===========================================================================
# LLMClient — init, repr, properties
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
    
    def test_get_limiter_reuses_same_loop(self):
        c = _make_client()
        async def get_twice():
            l1 = c._get_limiter()
            l2 = c._get_limiter()
            return l1, l2
        l1, l2 = asyncio.run(get_twice())
        assert l1 is l2

    def test_use_does_not_mutate_original(self):
        c1 = _make_client("openai")
        with patch("llmx.providers.groq.GroqProvider.__init__", return_value=None):
            c1.use("groq")
        assert c1.provider_name == "openai"

    def test_use_with_kwargs(self):
        c1 = _make_client("openai")
        with patch("llmx.providers.groq.GroqProvider.__init__", return_value=None) as mock_init:
            c1.use("groq", api_key="test")
        mock_init.assert_called_once_with(api_key="test")


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
        c._get_limiter = MagicMock(return_value=mock_limiter)
        asyncio.run(c.agenerate("hi"))
        mock_limiter.acquire.assert_called_once()

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
        c._get_limiter = MagicMock(return_value=mock_limiter)

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