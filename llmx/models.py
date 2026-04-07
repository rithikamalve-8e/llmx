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


