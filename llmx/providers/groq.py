from __future__ import annotations

import os
from typing import Iterator

from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
    ToolCall,
    Usage,
)
from llmx.providers.base import BaseProvider

import logging
import asyncio

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    name = "groq"
    env_var="GROQ_API_KEY"

    def __init__(self, api_key: str | None = None)-> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError("pip install groq") from exc

        self._client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    #core

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            kwargs = self._build_kwargs(request, stream=False)
            resp = await asyncio.to_thread(
                self._client.chat.completions.create, **kwargs
            )
            return self._normalize(resp)
        except Exception as e:
            logger.exception("Groq generate failed")
            raise RuntimeError("Groq generation failed") from e

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

    def _build_kwargs(self, request: GenerateRequest, stream: bool)-> dict:
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

    def _normalize(self, resp)-> GenerateResponse:
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