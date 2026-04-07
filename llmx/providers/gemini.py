from __future__ import annotations

import os
from typing import Iterator

from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
    ToolCall,
)
from llmx.providers.base import BaseProvider




class GeminiProvider(BaseProvider):
    name = "gemini"
    env_var="GEMINI_API_KEY"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("pip install google-genai") from exc

        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    #core

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        model_name, contents, config = self._prepare(request)
        resp = self._client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        return self._normalize(resp, model_name)

    def stream(self, request: GenerateRequest) -> Iterator[StreamChunk]:
        model_name, contents, config = self._prepare(request)
        for chunk in self._client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config,
        ):
            text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
            yield StreamChunk(delta=text, raw=chunk)

        yield StreamChunk(delta="", finished=True)

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
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=fc.name,
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