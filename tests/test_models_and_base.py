"""
test_models_and_base.py
Covers: llmx/models.py, llmx/providers/base.py
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    Message,
    StreamChunk,
    ToolCall,
    Usage,
)
from llmx.providers.base import BaseProvider
from llmx.exceptions import (
    InvalidRequestError,
    AuthenticationError,
    RateLimitError,
    ProviderUnavailableError,
)


# ===========================================================================
# Concrete subclass — calls the real abstract bodies (covers lines 25, 29)
# ===========================================================================

class ConcreteProvider(BaseProvider):
    name = "concrete"

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return False  # test stub — never matches any model

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        pass  # intentional

    def stream(self, request: GenerateRequest):
        pass  # intentional


# ===========================================================================
# models.py — Message
# ===========================================================================

class TestMessage:
    def test_user_role(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_assistant_role(self):
        m = Message(role="assistant", content="hi")
        assert m.role == "assistant"

    def test_system_role(self):
        m = Message(role="system", content="be helpful")
        assert m.role == "system"

    def test_empty_content(self):
        m = Message(role="user", content="")
        assert m.content == ""

    def test_multiline_content(self):
        m = Message(role="user", content="line1\nline2")
        assert "\n" in m.content


# ===========================================================================
# models.py — ToolCall
# ===========================================================================

class TestToolCall:
    def test_basic_fields(self):
        tc = ToolCall(id="tc1", name="get_weather", arguments={"city": "NY"})
        assert tc.id == "tc1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NY"}

    def test_empty_arguments(self):
        tc = ToolCall(id="x", name="noop", arguments={})
        assert tc.arguments == {}

    def test_nested_arguments(self):
        tc = ToolCall(id="1", name="fn", arguments={"a": {"b": 1}})
        assert tc.arguments["a"]["b"] == 1


# ===========================================================================
# models.py — Usage
# ===========================================================================

class TestUsage:
    def test_all_none_defaults(self):
        u = Usage()
        assert u.prompt_tokens is None
        assert u.completion_tokens is None
        assert u.total_tokens is None

    def test_with_values(self):
        u = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20
        assert u.total_tokens == 30

    def test_partial_values(self):
        u = Usage(prompt_tokens=5)
        assert u.prompt_tokens == 5
        assert u.completion_tokens is None


# ===========================================================================
# models.py — GenerateRequest
# ===========================================================================

class TestGenerateRequest:
    def test_defaults(self):
        req = GenerateRequest(messages=[])
        assert req.temperature == 0.7
        assert req.max_tokens == 1024
        assert req.system is None
        assert req.tools is None
        assert req.extra == {}
        assert req.model is None

    def test_full_construction(self):
        msgs = [Message(role="user", content="hi")]
        req = GenerateRequest(
            messages=msgs,
            model="gpt-4",
            temperature=0.3,
            max_tokens=512,
            system="be terse",
            tools=[{"type": "function"}],
            extra={"top_p": 0.9},
        )
        assert req.model == "gpt-4"
        assert req.system == "be terse"
        assert req.extra["top_p"] == 0.9

    def test_extra_default_factory_is_independent(self):
        r1 = GenerateRequest(messages=[])
        r2 = GenerateRequest(messages=[])
        r1.extra["key"] = "val"
        assert "key" not in r2.extra

    def test_tool_calls_default_factory_is_independent(self):
        r1 = GenerateResponse(content="a", model="m")
        r2 = GenerateResponse(content="b", model="m")
        r1.tool_calls.append(ToolCall(id="1", name="f", arguments={}))
        assert len(r2.tool_calls) == 0


# ===========================================================================
# models.py — GenerateRequest.validate()
# ===========================================================================

class TestGenerateRequestValidate:

    def test_validate_passes_valid_request(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gpt-4",
            temperature=0.7,
            max_tokens=1024,
        )
        req.validate()  # must not raise

    def test_validate_allows_model_none(self):
        req = GenerateRequest(messages=[], model=None)
        req.validate()  # None is allowed

    def test_validate_allows_temperature_at_boundary_0(self):
        req = GenerateRequest(messages=[], temperature=0.0)
        req.validate()

    def test_validate_allows_temperature_at_boundary_2(self):
        req = GenerateRequest(messages=[], temperature=2.0)
        req.validate()

    def test_validate_allows_temperature_int_in_range(self):
        req = GenerateRequest(messages=[], temperature=1)  # int, in range
        req.validate()

    def test_validate_allows_empty_messages_list(self):
        req = GenerateRequest(messages=[])
        req.validate()  # empty list is still a list

    def test_validate_raises_messages_not_a_list(self):
        req = GenerateRequest(messages=[])
        req.messages = "not a list"  # type: ignore
        with pytest.raises(InvalidRequestError, match="messages"):
            req.validate()

    def test_validate_raises_messages_is_dict(self):
        req = GenerateRequest(messages=[])
        req.messages = {"role": "user"}  # type: ignore
        with pytest.raises(InvalidRequestError):
            req.validate()

    def test_validate_raises_temperature_below_0(self):
        req = GenerateRequest(messages=[], temperature=-0.1)
        with pytest.raises(InvalidRequestError, match="temperature"):
            req.validate()

    def test_validate_raises_temperature_above_2(self):
        req = GenerateRequest(messages=[], temperature=2.1)
        with pytest.raises(InvalidRequestError, match="temperature"):
            req.validate()

    def test_validate_raises_max_tokens_zero(self):
        req = GenerateRequest(messages=[], max_tokens=0)
        with pytest.raises(InvalidRequestError, match="max_tokens"):
            req.validate()

    def test_validate_raises_max_tokens_negative(self):
        req = GenerateRequest(messages=[], max_tokens=-1)
        with pytest.raises(InvalidRequestError, match="max_tokens"):
            req.validate()

    def test_validate_raises_empty_string_model(self):
        req = GenerateRequest(messages=[], model="")
        with pytest.raises(InvalidRequestError, match="model"):
            req.validate()

    def test_validate_raises_is_invalid_request_error(self):
        req = GenerateRequest(messages=[], max_tokens=-5)
        with pytest.raises(InvalidRequestError):
            req.validate()


# ===========================================================================
# models.py — GenerateResponse
# ===========================================================================

class TestGenerateResponse:
    def test_defaults(self):
        r = GenerateResponse(content="hello", model="gpt-4")
        assert r.usage is None
        assert r.tool_calls == []
        assert r.raw is None

    def test_full_construction(self):
        usage = Usage(total_tokens=50)
        tc = ToolCall(id="1", name="fn", arguments={})
        raw = object()
        r = GenerateResponse(
            content="hi", model="m", usage=usage, tool_calls=[tc], raw=raw
        )
        assert r.usage.total_tokens == 50
        assert len(r.tool_calls) == 1
        assert r.raw is raw

    def test_empty_content(self):
        r = GenerateResponse(content="", model="m")
        assert r.content == ""


# ===========================================================================
# models.py — StreamChunk
# ===========================================================================

class TestStreamChunk:
    def test_defaults(self):
        c = StreamChunk(delta="a")
        assert c.finished is False
        assert c.model is None
        assert c.raw is None

    def test_finished_true(self):
        c = StreamChunk(delta="", finished=True, model="gpt-4")
        assert c.finished is True
        assert c.model == "gpt-4"

    def test_empty_delta(self):
        c = StreamChunk(delta="")
        assert c.delta == ""


# ===========================================================================
# base.py — abstract method bodies (lines 25, 29)
# ===========================================================================

class TestAbstractMethodBodies:
    """Call abstract bodies directly to hit the `pass` lines."""

    def test_generate_body_returns_none(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")])
        result = BaseProvider.generate(None, req)  # type: ignore[arg-type]
        assert result is None

    def test_stream_body_returns_none(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")])
        result = BaseProvider.stream(None, req)  # type: ignore[arg-type]
        assert result is None

    def test_concrete_generate_returns_none(self):
        p = ConcreteProvider()
        req = GenerateRequest(messages=[Message(role="user", content="hi")])
        assert p.generate(req) is None

    def test_concrete_stream_returns_none(self):
        p = ConcreteProvider()
        req = GenerateRequest(messages=[Message(role="user", content="hi")])
        assert p.stream(req) is None


# ===========================================================================
# base.py — _build_messages (lines 70-84)
# ===========================================================================

class TestBuildMessages:
    def setup_method(self):
        self.p = ConcreteProvider()

    def test_no_system(self):
        req = GenerateRequest(messages=[Message(role="user", content="hello")])
        msgs = self.p._build_messages(req)
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_with_system_prepended(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            system="be helpful",
        )
        msgs = self.p._build_messages(req)
        assert msgs[0] == {"role": "system", "content": "be helpful"}
        assert msgs[1] == {"role": "user", "content": "hi"}

    def test_multiple_messages(self):
        req = GenerateRequest(messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
            Message(role="user", content="q2"),
        ])
        msgs = self.p._build_messages(req)
        assert len(msgs) == 3
        assert msgs[1]["role"] == "assistant"

    def test_empty_messages_no_system(self):
        req = GenerateRequest(messages=[])
        assert self.p._build_messages(req) == []

    def test_empty_messages_with_system(self):
        req = GenerateRequest(messages=[], system="sys")
        msgs = self.p._build_messages(req)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_system_none_not_prepended(self):
        req = GenerateRequest(messages=[Message(role="user", content="hi")], system=None)
        msgs = self.p._build_messages(req)
        assert msgs[0]["role"] == "user"

    def test_message_fields_mapped_correctly(self):
        req = GenerateRequest(messages=[Message(role="user", content="test content")])
        msgs = self.p._build_messages(req)
        assert msgs[0]["content"] == "test content"


# ===========================================================================
# base.py — _retry_with_backoff
# ===========================================================================

class TestRetryWithBackoff:
    def setup_method(self):
        self.p = ConcreteProvider()

    def test_success_on_first_attempt(self):
        async def fn():
            return "ok"

        result = asyncio.run(self.p._retry_with_backoff(fn, retries=3))
        assert result == "ok"

    def test_success_after_one_connection_error(self):
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise ConnectionError("temporary")
            return "recovered"

        result = asyncio.run(
            self.p._retry_with_backoff(fn, retries=3, base_delay=0.01)
        )
        assert result == "recovered"
        assert len(calls) == 2

    def test_success_after_timeout_error(self):
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("slow")
            return "recovered"

        result = asyncio.run(
            self.p._retry_with_backoff(fn, retries=3, base_delay=0.01)
        )
        assert result == "recovered"

    def test_raises_after_all_retries_connection_error(self):
        async def fn():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            asyncio.run(
                self.p._retry_with_backoff(fn, retries=2, base_delay=0.01)
            )

    def test_raises_after_all_retries_timeout_error(self):
        async def fn():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            asyncio.run(
                self.p._retry_with_backoff(fn, retries=3, base_delay=0.01)
            )

    def test_asyncio_timeout_triggers_retry(self):
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 3:
                await asyncio.sleep(100)
            return "done"

        result = asyncio.run(
            self.p._retry_with_backoff(
                fn, retries=3, base_delay=0.01, timeout=0.05
            )
        )
        assert result == "done"

    def test_non_retryable_exception_bubbles_immediately(self):
        calls = []

        async def fn():
            calls.append(1)
            raise AuthenticationError("bad key")

        with pytest.raises(AuthenticationError, match="bad key"):
            asyncio.run(
                self.p._retry_with_backoff(fn, retries=3, base_delay=0.01)
            )
        # Only called once — did not retry
        assert len(calls) == 1

    def test_max_delay_caps_sleep(self):
        """Verify large attempt count doesn't exceed max_delay (no error expected)."""
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("retry me")
            return "done"

        result = asyncio.run(
            self.p._retry_with_backoff(fn, retries=3, base_delay=0.01, max_delay=0.02)
        )
        assert result == "done"

    def test_early_raise_on_last_attempt_connection_error(self):
        """Last attempt exhausted: re-raises the original exception."""
        calls = []

        async def fn():
            calls.append(1)
            raise ConnectionError("dead")

        with pytest.raises(ConnectionError, match="dead"):
            asyncio.run(
                self.p._retry_with_backoff(fn, retries=2, base_delay=0.01)
            )
        assert len(calls) == 2

    def test_post_loop_raise_via_asyncio_timeout(self):
        """asyncio.TimeoutError path exhausts all retries and re-raises."""
        calls = []

        async def fn():
            calls.append(1)
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(
                self.p._retry_with_backoff(fn, retries=2, base_delay=0.01, timeout=0.02)
            )
        assert len(calls) == 2

    def test_arbitrary_exceptions_are_retried(self):
        """Non-non_retryable exceptions (e.g. OSError) are retried by default."""
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise OSError("transient")
            return "ok"

        result = asyncio.run(
            self.p._retry_with_backoff(fn, retries=3, base_delay=0.01)
        )
        assert result == "ok"
        assert len(calls) == 2

    # ------------------------------------------------------------------
    # New tests for custom exceptions, jitter, and config
    # ------------------------------------------------------------------

    def test_authentication_error_not_retried(self):
        """AuthenticationError is re-raised immediately — never retried."""
        calls = []

        async def fn():
            calls.append(1)
            raise AuthenticationError("bad key")

        with pytest.raises(AuthenticationError):
            asyncio.run(
                self.p._retry_with_backoff(fn, retries=3, base_delay=0.01)
            )
        assert len(calls) == 1

    def test_rate_limit_error_is_retried(self):
        """RateLimitError is retried."""
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise RateLimitError("too many requests")
            return "ok"

        result = asyncio.run(
            self.p._retry_with_backoff(fn, retries=3, base_delay=0.01)
        )
        assert result == "ok"
        assert len(calls) == 2

    def test_provider_unavailable_error_is_retried(self):
        """ProviderUnavailableError is retried."""
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ProviderUnavailableError("service down")
            return "ok"

        result = asyncio.run(
            self.p._retry_with_backoff(fn, retries=3, base_delay=0.01)
        )
        assert result == "ok"
        assert len(calls) == 2

    def test_rate_limit_error_exhausts_retries(self):
        async def fn():
            raise RateLimitError("always limited")

        with pytest.raises(RateLimitError):
            asyncio.run(
                self.p._retry_with_backoff(fn, retries=2, base_delay=0.01)
            )

    def test_provider_unavailable_error_exhausts_retries(self):
        async def fn():
            raise ProviderUnavailableError("always down")

        with pytest.raises(ProviderUnavailableError):
            asyncio.run(
                self.p._retry_with_backoff(fn, retries=2, base_delay=0.01)
            )

    def test_config_overrides_retries(self):
        from llmx.config import LLMClientConfig

        calls = []
        config = LLMClientConfig(max_retries=2, base_delay=0.01, max_delay=1.0, timeout=30.0)

        async def fn():
            calls.append(1)
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            asyncio.run(
                self.p._retry_with_backoff(
                    fn, retries=99,  # overridden by config
                    config=config,
                )
            )
        assert len(calls) == 2  # config.max_retries=2

    def test_config_overrides_base_delay_and_max_delay(self):
        from llmx.config import LLMClientConfig

        config = LLMClientConfig(max_retries=2, base_delay=0.01, max_delay=1.0, timeout=30.0)
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("retry")
            return "ok"

        with patch("llmx.providers.base.random.uniform", return_value=0.0):
            result = asyncio.run(
                self.p._retry_with_backoff(fn, config=config)
            )
        assert result == "ok"

    def test_jitter_applied_to_backoff_delay(self):
        """Full jitter: random.uniform is called with (0, cap) where cap = min(base * 2^attempt, max_delay)."""
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("retry")
            return "ok"

        with patch("llmx.providers.base.random.uniform", return_value=0.0) as mock_rand:
            asyncio.run(
                self.p._retry_with_backoff(fn, retries=3, base_delay=0.01, max_delay=1.0)
            )

        # attempt=0, rate_limit_multiplier=1.0 → cap = min(0.01 * 1, 1.0) = 0.01
        mock_rand.assert_called_with(0, 0.01)
