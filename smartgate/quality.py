"""Response validation for cascade routing.

A cheap model answers first. If its answer fails validation, the gateway escalates to the
next tier up (a "model cascade") and never caches the failed answer. Plug in your own check
(an LLM judge, a classifier, a JSON-schema check) with `Gateway(validator=...)`.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .types import GatewayRequest, ProviderResponse

Validator = Callable[[GatewayRequest, ProviderResponse], Optional[str]]

_REFUSAL = re.compile(
    r"\b(i\s*(?:am|'m)?\s*(?:can(?:'|no)?t|cannot|unable to|not able to)\s+"
    r"(?:help|assist|answer|provide|do that|comply)|as an ai\b|i don'?t have (?:access|the ability))",
    re.I)
_TEMPLATE_LEFTOVER = re.compile(r"\{\{[^}]{1,40}\}\}")


def default_validator(request: GatewayRequest, response: ProviderResponse) -> Optional[str]:
    """Return a reason string if the answer should be escalated, else None."""
    text = (response.text or "").strip()
    if not text:
        return "empty response"
    if _REFUSAL.search(text[:400]):
        return "refusal-style answer"
    if _TEMPLATE_LEFTOVER.search(text):
        return "unfilled template placeholder"
    return None
