"""Core data types shared across the SDK."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

TIERS = ("small", "medium", "large")


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str

    @classmethod
    def coerce(cls, value: Any) -> "Message":
        if isinstance(value, Message):
            return value
        if isinstance(value, dict):
            return cls(role=value["role"], content=str(value["content"]))
        raise TypeError(f"Cannot convert {type(value).__name__} to Message")


@dataclass
class GatewayRequest:
    """A provider-neutral request sent to the gateway."""

    messages: List[Message]
    system: Optional[str] = None
    tenant_id: str = "default"
    max_tokens: int = 1024
    model: Optional[str] = None  # force a specific model (bypasses routing)
    tier: Optional[str] = None  # force a tier ("small" | "medium" | "large")
    use_cache: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = [Message.coerce(m) for m in self.messages]
        if not self.messages:
            raise ValueError("GatewayRequest requires at least one message")
        if self.tier is not None and self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")

    @property
    def last_user_message(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return self.messages[-1].content

    @property
    def full_text(self) -> str:
        parts = [self.system or ""] + [m.content for m in self.messages]
        return "\n".join(p for p in parts if p)

    def context_fingerprint(self) -> str:
        """Hash of everything except the final user turn.

        Two requests only share cache entries when their system prompt and prior
        conversation are identical, so a cached answer is never served into a
        different conversational context.
        """
        prior = [asdict(m) for m in self.messages[:-1]]
        blob = json.dumps({"system": self.system or "", "prior": prior}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ProviderResponse:
    text: str
    model: str
    usage: Usage
    stop_reason: Optional[str] = None
    raw: Any = None


@dataclass
class ModelSpec:
    """A model the gateway may route to."""

    name: str
    provider: str
    tier: str
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    max_context_tokens: int = 200_000
    max_output_tokens: int = 8_192
    capabilities: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)  # extra provider params
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"Model {self.name}: tier must be one of {TIERS}")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_mtok
            + output_tokens * self.output_cost_per_mtok
        ) / 1_000_000


@dataclass
class Attempt:
    model: str
    provider: str
    ok: bool
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class GatewayResponse:
    text: str
    model: str
    provider: str
    usage: Usage
    cost_usd: float
    cache_hit: bool = False
    cache_similarity: Optional[float] = None
    saved_usd: float = 0.0
    route_reason: str = ""
    analysis: Optional[Dict[str, Any]] = None
    attempts: List[Attempt] = field(default_factory=list)
    latency_ms: float = 0.0
    request_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
