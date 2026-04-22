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
from llmx.exceptions import (
    AuthenticationError,
    RateLimitError,
    ProviderUnavailableError,
    InvalidRequestError,
    ContextLengthExceededError,
    QuotaExceededError,
)

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
        config: "LLMClientConfig | None" = None,
    ) -> Any:
        if retries <= 0:
            raise ValueError("retries must be >= 1")

        if config is not None:
            retries = config.max_retries
            base_delay = config.base_delay
            max_delay = config.max_delay
            timeout = config.timeout

        non_retryable = (
            AuthenticationError,
            InvalidRequestError,
            ContextLengthExceededError,
            QuotaExceededError,
            
        )

        last_exc: Exception | None = None

        for attempt in range(retries):
            is_rate_limit = False
            try:
                return await asyncio.wait_for(fn(), timeout=timeout)
            
            except asyncio.CancelledError:
                raise

            except non_retryable:
                raise

            except Exception as e:
                last_exc = e
                is_rate_limit = isinstance(e, RateLimitError)

            if attempt == retries - 1:
                if last_exc is None:
                    raise RuntimeError("Retry loop exited without capturing an exception")
                raise last_exc

            rate_limit_multiplier = 3.0 if is_rate_limit else 1.0
            cap = min(base_delay * (2 ** attempt) * rate_limit_multiplier, max_delay)
            delay = random.uniform(0, cap)  # full jitter

            logger.warning(
                "Retrying in %.2fs (attempt %d/%d) | exc_type=%s msg=%s",
                delay, attempt + 1, retries, type(last_exc).__name__, last_exc,
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
