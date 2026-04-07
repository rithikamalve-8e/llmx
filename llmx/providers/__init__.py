from __future__ import annotations
 
from typing import TYPE_CHECKING
 
if TYPE_CHECKING:
    from llmx.providers.base import BaseProvider

#Providers are imported lazily so unused SDKs are never loaded.
PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "openai":  ("llmx.providers.openai",  "OpenAIProvider"),
    "groq":    ("llmx.providers.groq",    "GroqProvider"),
    "gemini":  ("llmx.providers.gemini",  "GeminiProvider"),
}
 
 
 
def load_provider(name: str, **kwargs) -> "BaseProvider":
    if name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY)
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {available}"
        )
    module_path, class_name = PROVIDER_REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
 