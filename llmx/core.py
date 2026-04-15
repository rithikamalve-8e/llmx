from __future__ import annotations

import os
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
from aiolimiter import AsyncLimiter


class LLMClient:
    def __init__(self, provider: str | None = None, **provider_kwargs) -> None:
        name = provider or self._detect_provider()
        self._provider: BaseProvider = load_provider(name, **provider_kwargs)
        self.provider_name: str = name
        self._limiter: AsyncLimiter | None = None

    def _get_limiter(self) -> AsyncLimiter:
        return AsyncLimiter(10, 1)

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

            await self._get_limiter().acquire()
            result = self._provider.generate(request)

            if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                return await result

            return result

        except (TypeError, ValueError):
            logger.exception("Invalid input to generate")
            raise
        except Exception as e:
            logger.exception("Provider generate failed")
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

            await self._get_limiter().acquire()
            stream = self._provider.stream(request)

            if hasattr(stream, "__aiter__"):
                async for chunk in stream:
                    yield chunk
            else:
                for chunk in stream:
                    yield chunk

        except (TypeError, ValueError):
            logger.exception("Invalid input to stream")
            raise
        except Exception as e:
            logger.exception("Provider stream failed")
            raise RuntimeError("LLM streaming failed") from e

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    def use(self, provider: str, **kwargs) -> "LLMClient":
        return LLMClient(provider=provider, **kwargs)

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider_name!r})"

    # helper
    from dotenv import load_dotenv
    load_dotenv()
    @staticmethod
    def _detect_provider() -> str:
        from llmx.providers import PROVIDER_REGISTRY
        import importlib
        

        for name, (module_path, class_name) in PROVIDER_REGISTRY.items():
            module = importlib.import_module(module_path)
            provider_cls = getattr(module, class_name)

            env_var = getattr(provider_cls, "env_var", None)
            if env_var and os.environ.get(env_var):
                return name

        raise EnvironmentError(
            "No LLM provider detected. Set any provider API key or pass provider explicitly."
        )

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
            return prompt

        if isinstance(prompt, str):
            messages = [Message(role="user", content=prompt)]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise TypeError(
                f"prompt must be str, list[Message], or GenerateRequest, got {type(prompt).__name__}"
            )

        return GenerateRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            extra=extra or {},
        )