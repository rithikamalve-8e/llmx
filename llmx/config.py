from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMClientConfig:
    timeout: float = 30.0
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    rate_limit: int = 10
    rate_limit_period: float = 1.0
