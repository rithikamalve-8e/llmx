from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, AsyncIterator, Callable, Type, Tuple, Any, TYPE_CHECKING
import asyncio
import logging
import random

from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)
from llmx.exceptions import AuthenticationError, RateLimitError, ProviderUnavailableError

if TYPE_CHECKING:
    from llmx.config import LLMClientConfig

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    # name used for registry
    name: str = "base"

    # capability

    @classmethod
    @abstractmethod
    def supports_model(cls, model: str) -> bool:
        """Return True if this provider can handle the given model identifier."""
        ...

    # core

    @abstractmethod
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        pass

    @abstractmethod
    def stream(self, request: GenerateRequest) -> Iterator[StreamChunk] | AsyncIterator[StreamChunk]:
        pass

    #retry helper

    async def _retry_with_backoff(
        self,
        fn: Callable[[], Any],
        *,
        retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        timeout: float = 30.0,
        retry_exceptions: Tuple[Type[Exception], ...] = (TimeoutError, ConnectionError),
        config: "LLMClientConfig | None" = None,
    ):
        if config is not None:
            retries = config.max_retries
            base_delay = config.base_delay
            max_delay = config.max_delay
            timeout = config.timeout

        last_exc: Exception | None = None

        for attempt in range(retries):
            try:
                return await asyncio.wait_for(fn(), timeout=timeout)

            except AuthenticationError:
                raise

            except (RateLimitError, ProviderUnavailableError) as e:
                last_exc = e

            except retry_exceptions as e:
                last_exc = e

            except asyncio.TimeoutError as e:
                last_exc = e

            if attempt == retries - 1:
                logger.exception("Max retries reached")
                raise last_exc

            delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 0.1 * base_delay)

            logger.warning(
                f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{retries}) due to: {last_exc}"
            )

            await asyncio.sleep(delay)

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
