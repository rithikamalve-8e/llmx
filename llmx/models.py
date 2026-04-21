from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


#basic

@dataclass
class Message:
    role: str
    content: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


#generation

@dataclass
class GenerateRequest:
    messages: list[Message]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    system: Optional[str] = None
    tools: Optional[list[dict]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        from llmx.exceptions import InvalidRequestError

        if not isinstance(self.messages, list):
            raise InvalidRequestError("messages must be a list")

        if not isinstance(self.temperature, (int, float)) or not (0.0 <= self.temperature <= 2.0):
            raise InvalidRequestError(
                f"temperature must be a float between 0.0 and 2.0, got {self.temperature!r}"
            )

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise InvalidRequestError(
                f"max_tokens must be an integer greater than 0, got {self.max_tokens!r}"
            )

        if self.model is not None and self.model == "":
            raise InvalidRequestError("model must not be an empty string (use None for provider default)")


@dataclass
class GenerateResponse:
    content: str
    model: str
    usage: Optional[Usage] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


@dataclass
class StreamChunk:
    delta: str
    finished: bool = False
    model: Optional[str] = None
    raw: Any = None
