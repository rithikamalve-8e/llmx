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




class OpenAIProvider(BaseProvider):
    name = "openai"
    env_var="OPENAI_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    )-> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        self._client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=base_url,
        )

    #core

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        kwargs = self._build_kwargs(request, stream=False)
        resp = self._client.chat.completions.create(**kwargs)
        return self._normalize(resp)

    def stream(self, request: GenerateRequest) -> Iterator[StreamChunk]:
        kwargs = self._build_kwargs(request, stream=True)

        with self._client.chat.completions.create(**kwargs) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                finished = chunk.choices[0].finish_reason is not None

                yield StreamChunk(
                    delta=delta,
                    finished=finished,
                    model=chunk.model,
                    raw=chunk,
                )
        

    #helpers

    def _build_kwargs(self, request: GenerateRequest, stream: bool) -> dict:
        kwargs ={
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