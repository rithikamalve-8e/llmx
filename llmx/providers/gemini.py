from __future__ import annotations

import os
from typing import AsyncIterator, TYPE_CHECKING

from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
    ToolCall,
)
from llmx.providers.base import BaseProvider
from llmx.exceptions import (
    AuthenticationError,
    RateLimitError,
    ProviderUnavailableError,
    ContextLengthExceededError,
    QuotaExceededError,
)

if TYPE_CHECKING:
    from llmx.config import LLMClientConfig

import logging
import asyncio

from llmx.observability import observe, lf

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    name = "gemini"
    env_var = "GEMINI_API_KEY"

    _MODEL_PREFIXES = ("gemini-", "models/gemini-")

    @classmethod
    def supports_model(cls, model: str) -> bool:
        m = model.lower()
        return any(m.startswith(p) for p in cls._MODEL_PREFIXES)

    def __init__(
        self,
        api_key: str | None = None,
        config: "LLMClientConfig | None" = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("pip install google-genai") from exc

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise AuthenticationError(
                "Gemini API key not found. Pass api_key= or set the GEMINI_API_KEY environment variable."
            )
        self.config = config
        self._genai = genai
        self._client = genai.Client(api_key=resolved_key)

    #core

    @observe(as_type="generation", name="gemini.generate")
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        _lf = lf()
        if _lf:
            _lf.update_current_generation(
                model=request.model,
                model_parameters={"temperature": request.temperature, "max_tokens": request.max_tokens},
                input=[{"role": m.role, "content": m.content} for m in request.messages],
            )

        async def _call():
            try:
                from google.api_core import exceptions as _gexc
            except ImportError:
                _gexc = None

            try:
                model_name, contents, config = self._prepare(request)
                resp = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                result = self._normalize(resp, model_name)
                logger.debug(
                    "generate completed",
                    extra={"provider": "gemini", "model": model_name},
                )
                if _lf:
                    _lf.update_current_generation(output=result.content)
                return result
            except ValueError:
                logger.exception("Invalid Gemini request", extra={"provider": "gemini", "model": request.model})
                raise
            except Exception as e:
                if _gexc is not None:
                    if isinstance(e, _gexc.Unauthenticated):
                        raise AuthenticationError(str(e)) from e
                    if isinstance(e, _gexc.ResourceExhausted):
                        if "quota" in str(e).lower():
                            raise QuotaExceededError(str(e)) from e
                        raise RateLimitError(str(e)) from e
                    if isinstance(e, _gexc.InvalidArgument) and (
                        "token" in str(e).lower() or "context" in str(e).lower()
                    ):
                        raise ContextLengthExceededError(str(e)) from e
                    if isinstance(e, _gexc.ServiceUnavailable):
                        raise ProviderUnavailableError(str(e)) from e
                raise

        return await self._retry_with_backoff(_call, config=self.config)

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        try:
            model_name, contents, config = self._prepare(request)

            stream = await asyncio.to_thread(
                self._client.models.generate_content_stream,
                model=model_name,
                contents=contents,
                config=config,
            )

            for chunk in stream:
                text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
                yield StreamChunk(delta=text, raw=chunk)

            yield StreamChunk(delta="", finished=True)

        except Exception as e:
            logger.exception(
                "stream failed",
                extra={"provider": "gemini", "model": request.model, "exc_type": type(e).__name__},
            )
            raise RuntimeError("Gemini streaming failed") from e

    # helpers

    def _prepare(self, request: GenerateRequest):
        from google.genai import types

        model_name = request.model

        # collect system prompt
        system_parts = []
        if request.system:
            system_parts.append(request.system)

        non_system = []
        for m in request.messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                non_system.append(m)

        if not non_system:
            raise ValueError("need at least one non-system message")

        def _role(r: str) -> str:
            return "model" if r == "assistant" else "user"

        contents = [
            types.Content(
                role=_role(m.role),
                parts=[types.Part(text=m.content)]
            )
            for m in non_system
        ]

        # TODO: request.tools (OpenAI-style JSON Schema dicts) are not forwarded.
        # Translating them to Gemini FunctionDeclaration objects requires a
        # non-trivial schema conversion layer that is out of scope here.
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction="\n".join(system_parts) if system_parts else None,
            **request.extra,
        )

        return model_name, contents, config

    def _normalize(self, resp, model_name: str) -> GenerateResponse:
        content = resp.text if hasattr(resp, "text") and resp.text else ""

        tool_calls: list[ToolCall] = []
        try:
            for i, part in enumerate(resp.candidates[0].content.parts):
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"{fc.name}_{i}",
                            name=fc.name,
                            arguments=dict(fc.args),
                        )
                    )
        except (AttributeError, IndexError):
            pass

        return GenerateResponse(
            content=content,
            model=model_name,
            tool_calls=tool_calls,
            raw=resp,
        )