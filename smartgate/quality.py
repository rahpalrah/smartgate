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


# --------------------------------------------------------------------------- grounding
_FACT_PATTERNS = [
    re.compile(r"\b\d+(?:\s*(?:-|to|–)\s*\d+)?\s*(?:business\s+|working\s+)?(?:minutes?|hours?|days?|weeks?|months?)\b", re.I),
    re.compile(r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|euros|pounds|usd|eur|gbp)\b)", re.I),
    re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
               r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?\b", re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b[A-Z0-9]*\d[A-Z0-9]{5,}\b"),  # tracking / reference / order-like codes
]


def _norm_fact(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower())


class GroundingValidator:
    """Flags answers that state specific facts the model could not know.

    A fact (a duration, amount, date or reference number) in the answer must appear in the
    conversation, the system prompt, or `known_facts` (e.g. your published policies:
    "30 days", "5-7 business days"). Otherwise the answer is treated as invented.
    Best for support, finance and healthcare bots; not for maths or general knowledge.
    """

    def __init__(self, known_facts: Optional[list] = None) -> None:
        self.known = {_norm_fact(f) for f in (known_facts or [])}

    def __call__(self, request: GatewayRequest, response: ProviderResponse) -> Optional[str]:
        context = _norm_fact(request.full_text)
        for rx in _FACT_PATTERNS:
            for m in rx.finditer(response.text or ""):
                fact = _norm_fact(m.group())
                digits = re.sub(r"\D", "", fact)
                if fact in self.known or fact in context or (digits and digits in re.sub(r"\D", " ", context).split()):
                    continue
                return f"unsupported specific: {m.group().strip()!r}"
        return None


class LLMJudgeValidator:
    """Score answers with any judge: `judge(request, answer_text) -> float in [0, 1]`.

    Wrap a stronger model (or a fine-tuned classifier) as the judge. Answers scoring below
    `min_score` escalate. The score is also exposed for adaptive routing via `last_score`.
    """

    PROMPT = ("Rate how well the ANSWER resolves the CUSTOMER message, from 0 (useless, wrong or "
              "invented facts) to 10 (correct, specific, actionable). Reply with the number only.\n"
              "CUSTOMER: {question}\nANSWER: {answer}")

    def __init__(self, judge: Callable[[GatewayRequest, str], float], min_score: float = 0.6) -> None:
        self.judge = judge
        self.min_score = min_score
        self.last_score: Optional[float] = None

    def __call__(self, request: GatewayRequest, response: ProviderResponse) -> Optional[str]:
        self.last_score = float(self.judge(request, response.text or ""))
        return None if self.last_score >= self.min_score else f"judge score {self.last_score:.2f}"


def chain(*validators: Optional[Validator]) -> Validator:
    """Run validators in order; the first failure wins."""
    active = [v for v in validators if v]

    def run(request: GatewayRequest, response: ProviderResponse) -> Optional[str]:
        for v in active:
            problem = v(request, response)
            if problem:
                return problem
        return None
    return run


# ------------------------------------------------------------------ adaptive routing
class QualityTracker:
    """Running quality per (route key, tier). Feeds adaptive routing.

    The route key is `request.metadata["route_key"]` when set (e.g. a classified intent), else
    the analyzer's task type. Scores come from validator outcomes (pass=1, fail=0), from
    judges, and from `Gateway.feedback()` (user thumbs-up/down, human review).
    """

    def __init__(self, min_score: float = 0.7, min_samples: int = 20, window: int = 200) -> None:
        import threading
        from collections import defaultdict, deque
        self.min_score = min_score
        self.min_samples = min_samples
        self._scores = defaultdict(lambda: deque(maxlen=window))
        self._lock = threading.Lock()

    def record(self, key: str, tier: str, score: float) -> None:
        with self._lock:
            self._scores[(key, tier)].append(max(0.0, min(1.0, float(score))))

    def mean(self, key: str, tier: str) -> Optional[float]:
        s = self._scores.get((key, tier))
        return (sum(s) / len(s)) if s and len(s) >= self.min_samples else None

    def floor_tier(self, key: str, tiers=("small", "medium", "large")) -> Optional[str]:
        """Lowest tier not proven inadequate for this key (None = no evidence yet)."""
        floor = None
        for i, tier in enumerate(tiers[:-1]):
            m = self.mean(key, tier)
            if m is not None and m < self.min_score:
                floor = tiers[i + 1]
            else:
                break
        return floor

    def report(self) -> dict:
        with self._lock:
            return {f"{k}|{t}": {"n": len(v), "mean": round(sum(v) / len(v), 3)}
                    for (k, t), v in sorted(self._scores.items()) if v}
