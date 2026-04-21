from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmx.providers.base import BaseProvider

#Providers are imported lazily so unused SDKs are never loaded.
PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "openai":  ("llmx.providers.openai",  "OpenAIProvider"),
    "groq":    ("llmx.providers.groq",    "GroqProvider"),
    "gemini":  ("llmx.providers.gemini",  "GeminiProvider"),
}


def load_provider(name: str, config=None, **kwargs) -> "BaseProvider":
    if name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY)
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {available}"
        )
    module_path, class_name = PROVIDER_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config=config, **kwargs)


def resolve_provider(model: str, config=None, **kwargs) -> "BaseProvider":
    """Select and instantiate the unique provider that supports *model*.

    Raises NoProviderError if no provider matches, or AmbiguousProviderError
    if more than one provider claims the model — in that case pass provider=
    explicitly to LLMClient.
    """
    from llmx.exceptions import NoProviderError, AmbiguousProviderError

    matching: list[tuple[str, type]] = []
    for name, (module_path, class_name) in PROVIDER_REGISTRY.items():
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if cls.supports_model(model):
            matching.append((name, cls))

    if not matching:
        available = ", ".join(PROVIDER_REGISTRY)
        raise NoProviderError(
            f"No registered provider supports model '{model}'. "
            f"Pass provider= explicitly or register a provider for this model. "
            f"Available providers: {available}"
        )

    if len(matching) > 1:
        names = [n for n, _ in matching]
        raise AmbiguousProviderError(
            f"Multiple providers support model '{model}': {names}. "
            f"Pass provider= explicitly to LLMClient to resolve the ambiguity."
        )

    name, cls = matching[0]
    return cls(config=config, **kwargs)
