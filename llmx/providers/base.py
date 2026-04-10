from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, AsyncIterator, Callable, Type, Tuple, Any
import asyncio
import logging

from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    # name used for registry
    name: str = "base"

    # core

    @abstractmethod
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        pass

    @abstractmethod
    def stream(self, request: GenerateRequest) -> Iterator[StreamChunk] | AsyncIterator[StreamChunk]:
        pass

    # retry helper

    async def _retry_with_backoff(
        self,
        fn: Callable[[], Any],
        *,
        retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        retry_exceptions: Tuple[Type[Exception], ...] = (TimeoutError, ConnectionError),
    ):
        last_exc: Exception | None = None

        for attempt in range(retries):
            try:
                return await fn()
            except retry_exceptions as e:
                last_exc = e

                if attempt == retries - 1:
                    logger.exception("Max retries reached")
                    raise

                delay = min(base_delay * (2 ** attempt), max_delay)

                logger.warning(
                    f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{retries}) due to: {e}"
                )

                await asyncio.sleep(delay)

        if last_exc:
            raise last_exc

    # helpers

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