from __future__ import annotations


class LLMXError(Exception):
    """Base class for all llmx exceptions."""


class AuthenticationError(LLMXError):
    """Raised when API key is missing, invalid, or rejected by the provider."""


class RateLimitError(LLMXError):
    """Raised when the provider returns a rate limit error."""


class ProviderUnavailableError(LLMXError):
    """Raised when the provider is unreachable or returns a 5xx error."""


class InvalidRequestError(LLMXError):
    """Raised when input validation fails."""


class NoProviderError(LLMXError):
    """Raised when no registered provider supports the requested model."""


class AmbiguousProviderError(LLMXError):
    """Raised when multiple providers claim to support the same model; pass provider= explicitly."""
