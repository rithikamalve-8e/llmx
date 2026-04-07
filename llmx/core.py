#imports
from __future__ import annotations
import os
from typing import Iterator

from llmx.models import(
    GenerateRequest,
    GenerateResponse,
    Message,
    StreamChunk
)
from llmx.providers.base import BaseProvider
from llmx.providers import load_provider


class LLMClient:
    def __init__(self, provider: str | None = None, **provider_kwargs) -> None:
        name = provider or self._detect_provider()
        self._provider: BaseProvider = load_provider(name, **provider_kwargs)
        self.provider_name: str = name

    # public api
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
        request = self._to_request(
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            extra=extra,
        )
        return self._provider.generate(request)

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
        request = self._to_request(
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )
        return self._provider.stream(request)

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    def use(self, provider: str, **kwargs) -> "LLMClient":
        return LLMClient(provider=provider, **kwargs)

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider_name!r})"

    # helper
    @staticmethod
    def _detect_provider() -> str:
        from llmx.providers import PROVIDER_REGISTRY
        import importlib
        import os

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