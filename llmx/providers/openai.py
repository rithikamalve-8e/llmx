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


class OpenAIProvider(BaseProvider):
    name = "openai"
    env_var = "OPENAI_API_KEY"

    _MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-", "text-embedding-", "text-davinci-")

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return any(model.startswith(p) for p in cls._MODEL_PREFIXES)

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        config: "LLMClientConfig | None" = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise AuthenticationError(
                "OpenAI API key not found. Pass api_key= or set the OPENAI_API_KEY environment variable."
            )
        self.config = config
        self._client = OpenAI(api_key=resolved_key, base_url=base_url)

    #core

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        async def _call():
            try:
                import openai as _openai
                kwargs = self._build_kwargs(request, stream=False)
                resp = await asyncio.to_thread(
                    self._client.chat.completions.create, **kwargs
                )
                result = self._normalize(resp)
                logger.debug(
                    "generate completed",
                    extra={
                        "provider": "openai",
                        "model": result.model,
                        "prompt_tokens": result.usage.prompt_tokens if result.usage else None,
                        "completion_tokens": result.usage.completion_tokens if result.usage else None,
                    },
                )
                return result
            except KeyError:
                logger.exception("Missing API key or config", extra={"provider": "openai"})
                raise
            except _openai.AuthenticationError as e:
                raise AuthenticationError(str(e)) from e
            except _openai.RateLimitError as e:
                if getattr(e, "code", None) == "insufficient_quota" or "quota" in str(e).lower():
                    raise QuotaExceededError(str(e)) from e
                raise RateLimitError(str(e)) from e
            except _openai.BadRequestError as e:
                if getattr(e, "code", None) == "context_length_exceeded" or "context_length" in str(e).lower():
                    raise ContextLengthExceededError(str(e)) from e
                raise
            except _openai.APIConnectionError as e:
                raise ProviderUnavailableError(str(e)) from e
            except _openai.APIStatusError as e:
                if e.status_code >= 500:
                    raise ProviderUnavailableError(str(e)) from e
                raise

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
            logger.exception(
                "stream failed",
                extra={"provider": "openai", "model": request.model, "exc_type": type(e).__name__},
            )
            raise RuntimeError("OpenAI streaming failed") from e

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

        if msg.tool_calls:
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
