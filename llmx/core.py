from __future__ import annotations

import asyncio
import logging
from typing import Iterator, AsyncIterator

logger = logging.getLogger(__name__)

from llmx.models import (
    GenerateRequest,
    GenerateResponse,
    Message,
    StreamChunk,
)
from llmx.providers.base import BaseProvider
from llmx.providers import load_provider
from llmx.config import LLMClientConfig
from llmx.exceptions import LLMXError, InvalidRequestError
from dotenv import load_dotenv

_dotenv_loaded = False


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        config: LLMClientConfig | None = None,
        **provider_kwargs,
    ) -> None:
        global _dotenv_loaded
        if not _dotenv_loaded:
            load_dotenv()
            _dotenv_loaded = True

        self.config = config or LLMClientConfig()
        self._provider_kwargs = provider_kwargs
        self._resolved_cache: dict[str, BaseProvider] = {}

        if provider is not None:
            self._provider: BaseProvider | None = load_provider(
                provider, config=self.config, **provider_kwargs
            )
            self.provider_name: str | None = provider
        else:
            self._provider = None
            self.provider_name = None

    def _resolve(self, model: str | None) -> BaseProvider:
        """Return the provider to use for this request.

        If an explicit provider was supplied at init, always use it.
        Otherwise resolve by capability: find the unique registered provider
        that reports supports_model(model) == True.
        """
        if self._provider is not None:
            return self._provider

        if model is None:
            raise ValueError(
                "model must be specified when LLMClient is created without an explicit provider "
                "(needed for capability-based resolution)"
            )

        if model not in self._resolved_cache:
            from llmx.providers import resolve_provider
            self._resolved_cache[model] = resolve_provider(
                model, config=self.config, **self._provider_kwargs
            )
        return self._resolved_cache[model]

    # sync

    def generate(
        self,
        prompt: str | list[Message] | GenerateRequest,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        **extra,
    ) -> GenerateResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "LLMClient.generate() cannot be called from a running event loop. "
                "Use 'await client.agenerate()' instead."
            )

        return asyncio.run(
            self.agenerate(
                prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **extra,
            )
        )

    def stream(
        self,
        prompt: str | list[Message] | GenerateRequest,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **extra,
    ) -> Iterator[StreamChunk]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "LLMClient.stream() cannot be called from a running event loop. "
                "Use 'async for chunk in client.astream()' instead."
            )

        async def _runner():
            return [
                chunk
                async for chunk in self.astream(
                    prompt,
                    model=model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
            ]

        return iter(asyncio.run(_runner()))

    # async

    async def agenerate(
        self,
        prompt: str | list[Message] | GenerateRequest,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        **extra,
    ) -> GenerateResponse:
        try:
            request = self._to_request(
                prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                extra=extra,
            )

            provider = self._resolve(request.model)
            logger.debug(
                "generate request",
                extra={
                    "provider": self.provider_name or type(provider).__name__,
                    "model": request.model,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "message_count": len(request.messages),
                },
            )

            result = provider.generate(request)

            if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                return await result

            return result

        except (TypeError, ValueError, InvalidRequestError):
            logger.exception(
                "Invalid request to generate",
                extra={"provider": self.provider_name, "model": model},
            )
            raise
        except LLMXError:
            raise
        except Exception as e:
            logger.exception(
                "Provider generate failed",
                extra={"provider": self.provider_name, "model": model, "exc_type": type(e).__name__},
            )
            raise RuntimeError("LLM generation failed") from e

    async def astream(
        self,
        prompt: str | list[Message] | GenerateRequest,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **extra,
    ) -> AsyncIterator[StreamChunk]:
        try:
            request = self._to_request(
                prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
            )

            provider = self._resolve(request.model)
            logger.debug(
                "stream request",
                extra={
                    "provider": self.provider_name or type(provider).__name__,
                    "model": request.model,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "message_count": len(request.messages),
                },
            )

            stream = provider.stream(request)

            if hasattr(stream, "__aiter__"):
                async for chunk in stream:
                    yield chunk
            else:
                for chunk in stream:
                    yield chunk

        except (TypeError, ValueError, InvalidRequestError):
            logger.exception(
                "Invalid request to stream",
                extra={"provider": self.provider_name, "model": model},
            )
            raise
        except LLMXError:
            raise
        except Exception as e:
            logger.exception(
                "Provider stream failed",
                extra={"provider": self.provider_name, "model": model, "exc_type": type(e).__name__},
            )
            raise RuntimeError("LLM streaming failed") from e

    @property
    def provider(self) -> BaseProvider | None:
        return self._provider

    def use(self, provider: str, **kwargs) -> "LLMClient":
        return LLMClient(provider=provider, config=self.config, **kwargs)

    def __repr__(self) -> str:
        if self.provider_name is not None:
            return f"LLMClient(provider={self.provider_name!r})"
        return "LLMClient(provider=None)"

    # helper

    @staticmethod
    def _to_request(
        prompt,
        *,
        model,
        system,
        temperature,
        max_tokens,
        tools=None,
        extra,
    ) -> GenerateRequest:
        if isinstance(prompt, GenerateRequest):
            prompt.validate()
            return prompt

        if isinstance(prompt, str):
            messages = [Message(role="user", content=prompt)]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise TypeError(
                f"prompt must be str, list[Message], or GenerateRequest, got {type(prompt).__name__}"
            )

        req = GenerateRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            extra=extra or {},
        )
        req.validate()
        return req
