from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)


class BaseProvider(ABC):
    # name used for registry
    name: str = "base"

    #core

    @abstractmethod
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        pass

    @abstractmethod
    def stream(self, request: GenerateRequest) -> Iterator[StreamChunk]:
        pass

    #helpers

    def _build_messages(self, request: GenerateRequest) -> list[dict]:
        msgs: list[dict] = []

        if request.system:
            msgs.append({
                "role": "system",
                "content": request.system
            })

        for m in request.messages:
            msgs.append({
                "role": m.role,
                "content": m.content
            })

        return msgs