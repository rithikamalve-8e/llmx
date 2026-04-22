from __future__ import annotations

import os
from typing import TYPE_CHECKING

from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
    ToolCall,
    Usage,
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

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    name = "groq"
    env_var = "GROQ_API_KEY"

    _MODEL_PREFIXES = ("llama", "mixtral-", "gemma", "whisper-", "deepseek-", "qwen", "groq")

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
            from groq import Groq
        except ImportError as exc:
            raise ImportError("pip install groq") from exc

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise AuthenticationError(
                "Groq API key not found. Pass api_key= or set the GROQ_API_KEY environment variable."
            )
        self.config = config
        self._client = Groq(api_key=resolved_key)

    #core

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        async def _call():
            try:
                import groq as _groq
                kwargs = self._build_kwargs(request, stream=False)
                resp = await asyncio.to_thread(
                    self._client.chat.completions.create, **kwargs
                )
                return self._normalize(resp)
            except _groq.AuthenticationError as e:
                raise AuthenticationError(str(e)) from e
            except _groq.RateLimitError as e:
                if getattr(e, "code", None) == "insufficient_quota" or "quota" in str(e).lower():
                    raise QuotaExceededError(str(e)) from e
                raise RateLimitError(str(e)) from e
            except _groq.BadRequestError as e:
                if getattr(e, "code", None) == "context_length_exceeded" or "context" in str(e).lower():
                    raise ContextLengthExceededError(str(e)) from e
                raise
            except _groq.APIConnectionError as e:
                raise ProviderUnavailableError(str(e)) from e

        return await self._retry_with_backoff(_call, config=self.config)

    async def stream(self, request: GenerateRequest):
        kwargs = self._build_kwargs(request, stream=True)

        try:
            stream = await asyncio.to_thread(
                self._client.chat.completions.create, **kwargs
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                finished = chunk.choices[0].finish_reason is not None

                yield StreamChunk(
                    delta=delta,
                    finished=finished,
                    model=chunk.model,
                    raw=chunk,
                )
        except Exception as e:
            logger.exception("Groq stream failed")
            raise RuntimeError("Groq streaming failed") from e

    #helpers

    def _build_kwargs(self, request: GenerateRequest, stream: bool) -> dict:
        kwargs = {
            "model": request.model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
            **request.extra,
        }

        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"

        return kwargs

    def _normalize(self, resp) -> GenerateResponse:
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []

        if getattr(msg, "tool_calls", None):
            import json

            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        usage = None
        if resp.usage:
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            )

        return GenerateResponse(
            content=msg.content or "",
            model=resp.model,
            usage=usage,
            tool_calls=tool_calls,
            raw=resp,
        )
