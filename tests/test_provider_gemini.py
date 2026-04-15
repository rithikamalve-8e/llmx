"""
test_provider_gemini.py
Covers: llmx/providers/gemini.py
Strategy: mock google.genai at the module level so real provider code runs.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmx.models import GenerateRequest, GenerateResponse, Message, StreamChunk, ToolCall
from llmx.providers.gemini import GeminiProvider


# ===========================================================================
# Helpers — build a realistic google.genai mock
# ===========================================================================

def _make_genai_mock():
    """
    Return a MagicMock that mimics google.genai well enough for
    the real GeminiProvider code to execute all branches.
    """
    genai = MagicMock()

    # types sub-module
    types = MagicMock()

    def _content(role, parts):
        obj = MagicMock()
        obj.role = role
        obj.parts = parts
        return obj

    def _part(text):
        obj = MagicMock()
        obj.text = text
        return obj

    def _config(**kw):
        obj = MagicMock()
        for k, v in kw.items():
            setattr(obj, k, v)
        return obj

    types.Content = MagicMock(side_effect=_content)
    types.Part = MagicMock(side_effect=_part)
    types.GenerateContentConfig = MagicMock(side_effect=_config)

    genai.types = types
    genai.Client = MagicMock()

    return genai


def _make_provider(genai_mock=None):
    """Construct GeminiProvider with a patched google.genai."""
    if genai_mock is None:
        genai_mock = _make_genai_mock()

    with patch.dict(
        sys.modules,
        {
            "google": MagicMock(genai=genai_mock),
            "google.genai": genai_mock,
            "google.genai.types": genai_mock.types,
        },
    ):
        p = GeminiProvider(api_key="test-key")

    # Attach mock references so tests can control behaviour
    p._genai = genai_mock
    p._client = genai_mock.Client.return_value
    return p, genai_mock


def _inject_types(p, genai_mock):
    """Context manager helper: patches sys.modules so _prepare can import types."""
    return patch.dict(
        sys.modules,
        {
            "google": MagicMock(genai=genai_mock),
            "google.genai": genai_mock,
            "google.genai.types": genai_mock.types,
        },
    )


def _raw_response(text="gemini answer", candidates=None):
    resp = MagicMock()
    resp.text = text
    resp.candidates = candidates if candidates is not None else []
    return resp


# ===========================================================================
# __init__
# ===========================================================================

class TestGeminiInit:

    def test_init_with_explicit_key(self):
        genai = _make_genai_mock()
        with patch.dict(sys.modules, {
            "google": MagicMock(genai=genai),
            "google.genai": genai,
            "google.genai.types": genai.types,
        }):
            p = GeminiProvider(api_key="my-key")
        genai.Client.assert_called_once_with(api_key="my-key")

    def test_init_reads_env_var(self):
        genai = _make_genai_mock()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}):
            with patch.dict(sys.modules, {
                "google": MagicMock(genai=genai),
                "google.genai": genai,
                "google.genai.types": genai.types,
            }):
                GeminiProvider()
        genai.Client.assert_called_once_with(api_key="env-key")

    def test_init_import_error(self):
        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="pip install google-genai"):
                GeminiProvider(api_key="k")

    def test_env_var_attribute(self):
        assert GeminiProvider.env_var == "GEMINI_API_KEY"

    def test_name_attribute(self):
        assert GeminiProvider.name == "gemini"

    def test_genai_stored_on_instance(self):
        p, genai = _make_provider()
        assert p._genai is genai

    def test_client_stored_on_instance(self):
        p, genai = _make_provider()
        assert p._client is genai.Client.return_value


# ===========================================================================
# _prepare
# ===========================================================================

class TestGeminiPrepare:

    def test_basic_user_message(self):
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[Message(role="user", content="hello")],
            model="gemini-pro",
        )
        with _inject_types(p, genai):
            model_name, contents, config = p._prepare(req)

        assert model_name == "gemini-pro"
        assert len(contents) == 1

    def test_system_prompt_in_config(self):
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gemini-pro",
            system="be concise",
        )
        with _inject_types(p, genai):
            _, _, config = p._prepare(req)

        assert config.system_instruction == "be concise"

    def test_no_system_instruction_is_none(self):
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gemini-pro",
        )
        with _inject_types(p, genai):
            _, _, config = p._prepare(req)

        assert config.system_instruction is None

    def test_system_role_message_extracted(self):
        """Messages with role='system' are removed from contents, merged into instruction."""
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[
                Message(role="system", content="sys instruction"),
                Message(role="user", content="question"),
            ],
            model="gemini-pro",
        )
        with _inject_types(p, genai):
            _, contents, config = p._prepare(req)

        assert len(contents) == 1
        assert "sys instruction" in config.system_instruction

    def test_system_prompt_and_system_message_merged(self):
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[
                Message(role="system", content="from message"),
                Message(role="user", content="hi"),
            ],
            model="gemini-pro",
            system="from param",
        )
        with _inject_types(p, genai):
            _, _, config = p._prepare(req)

        assert "from param" in config.system_instruction
        assert "from message" in config.system_instruction

    def test_raises_if_no_non_system_messages(self):
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[Message(role="system", content="only system")],
            model="gemini-pro",
        )
        with _inject_types(p, genai):
            with pytest.raises(ValueError, match="at least one non-system message"):
                p._prepare(req)

    def test_raises_if_messages_empty(self):
        p, genai = _make_provider()
        req = GenerateRequest(messages=[], model="gemini-pro")
        with _inject_types(p, genai):
            with pytest.raises(ValueError):
                p._prepare(req)

    def test_assistant_role_mapped_to_model(self):
        captured_roles = []

        def capture_content(role, parts):
            captured_roles.append(role)
            obj = MagicMock()
            obj.role = role
            obj.parts = parts
            return obj

        p, genai = _make_provider()
        genai.types.Content.side_effect = capture_content

        req = GenerateRequest(
            messages=[
                Message(role="user", content="q"),
                Message(role="assistant", content="a"),
                Message(role="user", content="q2"),
            ],
            model="gemini-pro",
        )
        with _inject_types(p, genai):
            p._prepare(req)

        assert "model" in captured_roles
        assert "user" in captured_roles
        assert "assistant" not in captured_roles

    def test_user_role_unchanged(self):
        captured_roles = []

        def capture(role, parts):
            captured_roles.append(role)
            obj = MagicMock()
            obj.role = role
            return obj

        p, genai = _make_provider()
        genai.types.Content.side_effect = capture

        req = GenerateRequest(
            messages=[Message(role="user", content="q")],
            model="gemini-pro",
        )
        with _inject_types(p, genai):
            p._prepare(req)

        assert captured_roles == ["user"]

    def test_temperature_in_config(self):
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gemini-pro", temperature=0.2,
        )
        with _inject_types(p, genai):
            _, _, config = p._prepare(req)

        assert config.temperature == 0.2

    def test_max_tokens_in_config(self):
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gemini-pro", max_tokens=256,
        )
        with _inject_types(p, genai):
            _, _, config = p._prepare(req)

        assert config.max_output_tokens == 256

    def test_extra_kwargs_passed_to_config(self):
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gemini-pro",
            extra={"top_p": 0.95},
        )
        with _inject_types(p, genai):
            _, _, config = p._prepare(req)

        assert config.top_p == 0.95

    def test_multiple_user_messages_all_in_contents(self):
        p, genai = _make_provider()
        req = GenerateRequest(
            messages=[
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="q2"),
            ],
            model="gemini-pro",
        )
        with _inject_types(p, genai):
            _, contents, _ = p._prepare(req)

        assert len(contents) == 3


# ===========================================================================
# _normalize
# ===========================================================================

class TestGeminiNormalize:

    def setup_method(self):
        self.p, _ = _make_provider()

    def test_basic_text_response(self):
        resp = _raw_response("gemini answer")
        result = self.p._normalize(resp, "gemini-pro")
        assert result.content == "gemini answer"
        assert result.model == "gemini-pro"
        assert result.tool_calls == []

    def test_none_text_becomes_empty_string(self):
        resp = _raw_response(text=None)
        result = self.p._normalize(resp, "gemini-pro")
        assert result.content == ""

    def test_no_text_attr_becomes_empty_string(self):
        resp = MagicMock(spec=["candidates"])
        resp.candidates = []
        result = self.p._normalize(resp, "gemini-pro")
        assert result.content == ""

    def test_tool_call_extracted(self):
        fc = MagicMock()
        fc.name = "calculate"
        fc.args = {"x": 1, "y": 2}

        part = MagicMock()
        part.function_call = fc

        candidate = MagicMock()
        candidate.content.parts = [part]

        resp = _raw_response(text="", candidates=[candidate])
        result = self.p._normalize(resp, "gemini-pro")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calculate"
        assert result.tool_calls[0].id == "calculate"
        assert result.tool_calls[0].arguments == {"x": 1, "y": 2}

    def test_multiple_tool_calls(self):
        def _make_part(name, args):
            fc = MagicMock()
            fc.name = name
            fc.args = args
            part = MagicMock()
            part.function_call = fc
            return part

        candidate = MagicMock()
        candidate.content.parts = [
            _make_part("fn1", {"a": 1}),
            _make_part("fn2", {"b": 2}),
        ]

        resp = _raw_response(text="", candidates=[candidate])
        result = self.p._normalize(resp, "gemini-pro")
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "fn1"
        assert result.tool_calls[1].name == "fn2"

    def test_part_without_function_call_skipped(self):
        part = MagicMock(spec=["text"])  # no function_call attr
        candidate = MagicMock()
        candidate.content.parts = [part]

        resp = _raw_response(text="hi", candidates=[candidate])
        result = self.p._normalize(resp, "gemini-pro")
        assert result.tool_calls == []

    def test_part_with_falsy_function_call_skipped(self):
        part = MagicMock()
        part.function_call = None

        candidate = MagicMock()
        candidate.content.parts = [part]

        resp = _raw_response(text="hi", candidates=[candidate])
        result = self.p._normalize(resp, "gemini-pro")
        assert result.tool_calls == []

    def test_empty_candidates_no_tool_calls(self):
        resp = _raw_response(candidates=[])
        result = self.p._normalize(resp, "gemini-pro")
        assert result.tool_calls == []

    def test_index_error_swallowed(self):
        """candidates[0] IndexError is silently caught."""
        resp = _raw_response(text="ok")
        resp.candidates = []  # will cause IndexError in loop
        result = self.p._normalize(resp, "gemini-pro")
        assert result.tool_calls == []

    def test_attribute_error_swallowed(self):
        candidate = MagicMock(spec=[])  # no content attr → AttributeError
        resp = _raw_response(text="ok", candidates=[candidate])
        result = self.p._normalize(resp, "gemini-pro")
        assert result.tool_calls == []

    def test_raw_stored(self):
        resp = _raw_response()
        result = self.p._normalize(resp, "gemini-pro")
        assert result.raw is resp

    def test_model_name_propagated(self):
        resp = _raw_response()
        result = self.p._normalize(resp, "gemini-2.5-flash")
        assert result.model == "gemini-2.5-flash"


# ===========================================================================
# generate (async, real method)
# ===========================================================================

class TestGeminiGenerate:

    def setup_method(self):
        self.p, self.genai = _make_provider()

    def test_generate_success(self):
        raw = _raw_response("answer")
        with _inject_types(self.p, self.genai):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
                req = GenerateRequest(
                    messages=[Message(role="user", content="hi")],
                    model="gemini-pro",
                )
                result = asyncio.run(self.p.generate(req))

        assert result.content == "answer"
        assert result.model == "gemini-pro"

    def test_generate_value_error_propagates(self):
        """ValueError from _prepare (no non-system messages) re-raises directly."""
        with _inject_types(self.p, self.genai):
            req = GenerateRequest(
                messages=[Message(role="system", content="only sys")],
                model="gemini-pro",
            )
            with pytest.raises(ValueError):
                asyncio.run(self.p.generate(req))

    def test_generate_runtime_error_on_other_exception(self):
        with _inject_types(self.p, self.genai):
            with patch("asyncio.to_thread", new=AsyncMock(side_effect=Exception("api error"))):
                req = GenerateRequest(
                    messages=[Message(role="user", content="hi")],
                    model="gemini-pro",
                )
                with pytest.raises(RuntimeError, match="Gemini generation failed"):
                    asyncio.run(self.p.generate(req))

    def test_generate_with_system_prompt(self):
        raw = _raw_response("answer")
        with _inject_types(self.p, self.genai):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
                req = GenerateRequest(
                    messages=[Message(role="user", content="hi")],
                    model="gemini-pro", system="be terse",
                )
                result = asyncio.run(self.p.generate(req))
        assert result.content == "answer"

    def test_generate_with_tool_call_response(self):
        fc = MagicMock()
        fc.name = "fn"
        fc.args = {"k": "v"}
        part = MagicMock()
        part.function_call = fc
        candidate = MagicMock()
        candidate.content.parts = [part]
        raw = _raw_response(text="", candidates=[candidate])

        with _inject_types(self.p, self.genai):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
                req = GenerateRequest(
                    messages=[Message(role="user", content="call fn")],
                    model="gemini-pro",
                )
                result = asyncio.run(self.p.generate(req))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "fn"

    def test_generate_calls_to_thread_with_correct_fn(self):
        raw = _raw_response("ok")
        calls = []

        async def fake_to_thread(fn, **kwargs):
            calls.append(fn)
            return raw

        with _inject_types(self.p, self.genai):
            with patch("asyncio.to_thread", side_effect=fake_to_thread):
                req = GenerateRequest(
                    messages=[Message(role="user", content="hi")],
                    model="gemini-pro",
                )
                asyncio.run(self.p.generate(req))

        assert len(calls) == 1

    def test_generate_empty_text_response(self):
        raw = _raw_response(text="")
        with _inject_types(self.p, self.genai):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=raw)):
                req = GenerateRequest(
                    messages=[Message(role="user", content="hi")],
                    model="gemini-pro",
                )
                result = asyncio.run(self.p.generate(req))
        assert result.content == ""


# ===========================================================================
# stream (async generator, real method)
# ===========================================================================

class TestGeminiStream:

    def setup_method(self):
        self.p, self.genai = _make_provider()

    def _run_stream(self, req, raw_chunks):
        with _inject_types(self.p, self.genai):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=iter(raw_chunks))):
                async def collect():
                    return [c async for c in self.p.stream(req)]
                return asyncio.run(collect())

    def _chunk(self, text="hi"):
        c = MagicMock()
        c.text = text
        return c

    def test_stream_yields_text_chunks(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")], model="gemini-pro"
        )
        result = self._run_stream(req, [self._chunk("hello "), self._chunk("world")])
        deltas = [c.delta for c in result if not c.finished]
        assert "hello " in deltas
        assert "world" in deltas

    def test_stream_sentinel_finished_chunk(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")], model="gemini-pro"
        )
        result = self._run_stream(req, [self._chunk("hi")])
        assert result[-1].finished is True
        assert result[-1].delta == ""

    def test_stream_none_text_becomes_empty_delta(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")], model="gemini-pro"
        )
        c = MagicMock()
        c.text = None
        result = self._run_stream(req, [c])
        # First result is the chunk with empty delta, last is sentinel
        assert result[0].delta == ""

    def test_stream_chunk_no_text_attr(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")], model="gemini-pro"
        )
        c = MagicMock(spec=[])  # no text attribute
        result = self._run_stream(req, [c])
        assert result[0].delta == ""

    def test_stream_runtime_error(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")], model="gemini-pro"
        )
        with _inject_types(self.p, self.genai):
            with patch("asyncio.to_thread", new=AsyncMock(side_effect=Exception("stream err"))):
                async def collect():
                    return [c async for c in self.p.stream(req)]
                with pytest.raises(RuntimeError, match="Gemini streaming failed"):
                    asyncio.run(collect())

    def test_stream_many_chunks(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")], model="gemini-pro"
        )
        chunks = [self._chunk(str(i)) for i in range(10)]
        result = self._run_stream(req, chunks)
        # 10 content chunks + 1 sentinel
        assert len(result) == 11
        assert result[-1].finished is True

    def test_stream_raw_chunk_stored(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")], model="gemini-pro"
        )
        raw_chunk = self._chunk("text")
        result = self._run_stream(req, [raw_chunk])
        assert result[0].raw is raw_chunk

    def test_stream_with_system_prompt(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")],
            model="gemini-pro", system="be concise",
        )
        result = self._run_stream(req, [self._chunk("ok")])
        assert len(result) >= 1

    def test_stream_empty_no_chunks(self):
        req = GenerateRequest(
            messages=[Message(role="user", content="hi")], model="gemini-pro"
        )
        result = self._run_stream(req, [])
        # Only the sentinel
        assert len(result) == 1
        assert result[0].finished is True