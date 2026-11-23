"""Prompt analysis: complexity scoring, task classification, PII and cacheability.

The default analyzer is heuristic, fast (<1 ms) and dependency free. Swap in an
ML/LLM classifier by subclassing `PromptAnalyzer` or passing `extra_scorers`.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .types import GatewayRequest

# --------------------------------------------------------------------------- PII
PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "mrn": re.compile(r"\b(?:MRN|medical record(?: number)?)[:#\s]*\d{5,}\b", re.I),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Phone numbers, standard library only. A match needs a leading "+country-code", parentheses or
# separators: bare digit runs are order/invoice/tracking numbers far more often than phone numbers.
#  * "+CC ..." numbers are accepted for any assigned ITU country calling code (table below).
#  * National formats (no "+") are recognized for the regions in PHONE_REGIONS (default: US).
# Built-in national patterns: US, CA, GB, IN, DE, FR, AU, ES. Other countries are detected in
# their international "+CC" form.
CALLING_CODES = frozenset((
    "1 7 20 27 30 31 32 33 34 36 39 40 41 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58 60 61 62 63 64 65 66 "
    "81 82 84 86 90 91 92 93 94 95 98 "
    "211 212 213 216 218 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 "
    "241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 260 261 262 263 264 265 266 267 "
    "268 269 290 291 297 298 299 350 351 352 353 354 355 356 357 358 359 370 371 372 373 374 375 376 377 378 "
    "380 381 382 383 385 386 387 389 420 421 423 500 501 502 503 504 505 506 507 508 509 590 591 592 593 594 "
    "595 596 597 598 599 670 672 673 674 675 676 677 678 679 680 681 682 683 685 686 687 688 689 690 691 692 "
    "850 852 853 855 856 880 886 960 961 962 963 964 965 966 967 968 970 971 972 973 974 975 976 977 992 993 "
    "994 995 996 998").split())
PHONE_REGIONS: List[str] = ["US"]
_US_PHONE = re.compile(r"(?<![\w+])(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\d)")
_INTL_PHONE = re.compile(r"(?<![\w+])\+\s?\d{1,3}(?:[\s.-]?\(?\d{1,5}\)?){1,6}(?![\w])")
_NATIONAL_PHONE: Dict[str, List["re.Pattern"]] = {
    "US": [_US_PHONE],
    "CA": [_US_PHONE],
    "GB": [re.compile(r"(?<![\w+])0\d{2,4}[ -]\d{3,4}(?:[ -]\d{3,4})?(?!\d)")],
    "IN": [re.compile(r"(?<![\w+])0?[6-9]\d{4}[ -]\d{5}(?!\d)")],
    "DE": [re.compile(r"(?<![\w+])0\d{2,4}[ /-]\d{3,8}(?:[ -]\d{1,4})?(?!\d)")],
    "FR": [re.compile(r"(?<![\w+])0[1-9](?:[ .-]\d{2}){4}(?!\d)")],
    "AU": [re.compile(r"(?<![\w+])0[2-478][ -]?\d{4}[ -]\d{4}(?!\d)"),
           re.compile(r"(?<![\w+])04\d{2}[ -]\d{3}[ -]\d{3}(?!\d)")],
    "ES": [re.compile(r"(?<![\w+])[6-9]\d{2}[ -]\d{2}[ -]\d{2}[ -]\d{2}(?!\d)")],
}


def set_phone_regions(regions: List[str]) -> None:
    """Countries whose *national* formats (e.g. GB "020 7946 0958") should be detected."""
    PHONE_REGIONS[:] = [r.upper() for r in regions]


def _valid_international(text: str) -> bool:
    """E.164 shape (8-15 digits) whose leading digits are an assigned country calling code."""
    digits = "".join(ch for ch in text if ch.isdigit())
    if not 8 <= len(digits) <= 15:
        return False
    return any(digits[:n] in CALLING_CODES and len(digits) - n >= 5 for n in (1, 2, 3))


def _phone_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in _INTL_PHONE.finditer(text):
        if _valid_international(m.group()):
            spans.append(m.span())
    for region in PHONE_REGIONS:
        for rx in _NATIONAL_PHONE.get(region, []):
            for m in rx.finditer(text):
                if 8 <= sum(ch.isdigit() for ch in m.group()) <= 15:
                    spans.append(m.span())
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


# Maximal runs of digits joined by single spaces/dashes; card numbers are searched inside them.
_DIGIT_RUN = re.compile(r"\d(?:[ -]?\d)*")


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _card_spans(text: str) -> List[Tuple[int, int]]:
    """Luhn-valid 13-19 digit windows inside digit runs, aligned to group boundaries.

    "0 4111 1111 1111 1111" -> finds the card after the stray "0" (a greedy regex misses it),
    while "1234 5678 9012 3456" is not split mid-group to manufacture a Luhn match.
    """
    spans: List[Tuple[int, int]] = []
    for run in _DIGIT_RUN.finditer(text):
        groups = [(m.start() + run.start(), m.end() + run.start()) for m in re.finditer(r"\d+", run.group())]
        n, i = len(groups), 0
        while i < n:
            match = None
            for j in range(n - 1, i - 1, -1):  # longest window first
                digits = "".join(text[a:b] for a, b in groups[i:j + 1])
                if 13 <= len(digits) <= 19 and _luhn_ok(digits):
                    match = j
                    break
            if match is None:
                i += 1
            else:
                spans.append((groups[i][0], groups[match][1]))
                i = match + 1
    return spans


def detect_pii(text: str) -> List[str]:
    found = ["credit_card"] if _card_spans(text) else []
    found += [kind for kind, pattern in PII_PATTERNS.items() if pattern.search(text)]
    if _phone_spans(text):
        found.append("phone")
    return found


def redact_pii(text: str) -> str:
    # Cards first so the phone pattern cannot consume part of a card number.
    for start, end in reversed(_card_spans(text)):
        text = text[:start] + "[REDACTED_CREDIT_CARD]" + text[end:]
    for kind, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{kind.upper()}]", text)
    for start, end in reversed(_phone_spans(text)):
        text = text[:start] + "[REDACTED_PHONE]" + text[end:]
    return text


# ---------------------------------------------------------------- task signals
TASK_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "code": ("code", "function", "bug", "debug", "refactor", "compile", "python", "javascript",
             "sql", "api", "class ", "stack trace", "exception", "regex", "implement"),
    "math": ("calculate", "solve", "equation", "integral", "derivative", "probability",
             "prove", "theorem", "matrix", "statistics"),
    "reasoning": ("why", "analyze", "analyse", "compare", "evaluate", "trade-off", "tradeoff",
                  "pros and cons", "implications", "strategy", "reason", "assess", "critique"),
    "creative": ("write a story", "poem", "slogan", "creative", "brainstorm", "tagline", "lyrics"),
    "extraction": ("extract", "parse", "list all", "pull out", "find all", "identify the"),
    "summarization": ("summarize", "summarise", "tl;dr", "tldr", "key points", "recap", "condense"),
    "classification": ("classify", "categorize", "categorise", "label", "sentiment", "is this",
                       "which category"),
    "translation": ("translate", "translation", "in spanish", "in french", "in german"),
    "qa": ("what is", "who is", "when did", "where is", "how many", "define", "what are",
           "how do i", "how can i", "how to", "where can i", "is there", "do you"),
}
# Word-boundary regexes so "api" doesn't match "capital", "label" doesn't match "labelled" etc.
TASK_REGEX: Dict[str, List[re.Pattern]] = {
    task: [re.compile(r"(?<!\w)" + re.escape(kw.strip()) + r"(?!\w)", re.I) for kw in kws]
    for task, kws in TASK_KEYWORDS.items()
}
TASK_COMPLEXITY_PRIOR = {
    "classification": 0.10, "extraction": 0.20, "translation": 0.25, "qa": 0.15,
    "summarization": 0.30, "creative": 0.40, "code": 0.60, "reasoning": 0.60,
    "math": 0.65, "general": 0.25,
}
MULTI_STEP_PATTERNS = re.compile(
    r"(step[- ]by[- ]step|first.*then|\b\d+\.\s|\bthen\b.*\bfinally\b|multi-?step|"
    r"in detail|comprehensive|thorough|end[- ]to[- ]end|architecture|design a)",
    re.I | re.S,
)
# Time-sensitive = the correct answer changes over time, so a cached answer can go stale.
# A weak time word alone ("I need help NOW", "I ordered it TODAY") is not enough; it counts only
# together with a request for something that varies (status, price, ETA, availability...).
_STRONG_TIME = re.compile(
    r"\b(?:latest|breaking news|real[- ]time|stock prices?|share prices?|exchange rates?|weather|forecast|"
    r"outage|(?:is|are|was|went)(?:n't| not)?\s+(?:\w+\s+){0,2}down|(?:server|service|site|website|app)\s+down|"
    r"live\s+(?:score|status|updates?|feed|results?|traffic)|(?:flight|train|delivery|order)\s+status)\b", re.I)
_WEAK_TIME = re.compile(
    r"\b(?:now|right now|today|tonight|tomorrow|yesterday|currently|current|recent(?:ly)?|at the moment|"
    r"as of|this (?:week|month|year|morning|afternoon|evening|weekend))\b", re.I)
_VARIES = re.compile(
    r"\b(?:status|(?:any|an|no|status|order|delivery)\s+updates?|updates?\s+(?:on|about|regarding|for)|news|outage|down|"
    r"(?:is|are)\s+(?:\w+\s+){0,3}working|operational|available|availability|open|closed|prices?|rates?|cost|score|"
    r"delay(?:ed|s)?|eta|arriv(?:e|es|ing|al)|when (?:will|does|do|is|are)|how long|where(?:'s| is| are)|"
    r"any (?:update|news|word)|still waiting|what(?:'s| is) happening)\b", re.I)
# Kept for backwards compatibility: the old, broader keyword list.
TIME_SENSITIVE = re.compile(
    r"\b(today|tonight|now|current(?:ly)?|latest|this (?:week|month|year)|right now|"
    r"recent(?:ly)?|breaking|live|real[- ]time|stock price|weather|yesterday|tomorrow)\b",
    re.I,
)


def is_time_sensitive(text: str) -> bool:
    return bool(_STRONG_TIME.search(text) or (_WEAK_TIME.search(text) and _VARIES.search(text)))
PERSONAL_CONTEXT = re.compile(r"\b(my|our|i am|i'm|me)\b", re.I)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for English, never zero)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


@dataclass
class PromptAnalysis:
    complexity: float  # 0.0 (trivial) .. 1.0 (very hard)
    task_type: str
    estimated_input_tokens: int
    pii: List[str] = field(default_factory=list)
    time_sensitive: bool = False
    personal: bool = False
    cacheable: bool = True
    signals: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


Scorer = Callable[[GatewayRequest, "PromptAnalysis"], Optional[float]]


class PromptAnalyzer:
    """Heuristic analyzer. Override `analyze` or add `extra_scorers` to customize.

    Each extra scorer receives (request, draft_analysis) and may return a complexity
    delta in [-1, 1] that is added to the score — useful for industry-specific
    signals (e.g. "contract clause" => +0.2 for legal).
    """

    def __init__(
        self,
        extra_scorers: Optional[List[Scorer]] = None,
        domain_keywords: Optional[Dict[str, float]] = None,
        time_sensitive_fn: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.time_sensitive_fn = time_sensitive_fn or is_time_sensitive
        self.extra_scorers = list(extra_scorers or [])
        # keyword -> complexity boost, e.g. {"differential diagnosis": 0.3}
        self.domain_keywords = {k.lower(): v for k, v in (domain_keywords or {}).items()}

    def classify_task(self, text: str) -> Tuple[str, Dict[str, int]]:
        hits = {task: sum(1 for rx in rxs if rx.search(text)) for task, rxs in TASK_REGEX.items()}
        best = max(hits.items(), key=lambda kv: (kv[1], TASK_COMPLEXITY_PRIOR[kv[0]]))
        return (best[0] if best[1] > 0 else "general"), hits

    def analyze(self, request: GatewayRequest) -> PromptAnalysis:
        user_text = request.last_user_message
        full_text = request.full_text
        tokens = estimate_tokens(full_text)
        task, hits = self.classify_task(user_text)

        signals: Dict[str, float] = {}
        signals["task_prior"] = TASK_COMPLEXITY_PRIOR[task]
        # Longer prompts tend to be harder; log-scaled, saturates around ~8k tokens.
        signals["length"] = min(0.25, math.log10(max(estimate_tokens(user_text), 1)) / 16)
        signals["multi_step"] = 0.15 if MULTI_STEP_PATTERNS.search(user_text) else 0.0
        signals["code_block"] = 0.10 if "```" in user_text else 0.0
        signals["multi_intent"] = min(0.15, 0.05 * max(0, sum(1 for v in hits.values() if v) - 1))
        signals["questions"] = min(0.10, 0.03 * max(0, user_text.count("?") - 1))
        signals["conversation_depth"] = min(0.10, 0.02 * (len(request.messages) - 1))
        signals["constraints"] = min(
            0.10, 0.03 * len(re.findall(r"\b(must|should|ensure|without|only|exactly|at least)\b", user_text, re.I))
        )
        lower = user_text.lower()
        domain_boost = sum(v for k, v in self.domain_keywords.items() if k in lower)
        if domain_boost:
            signals["domain"] = domain_boost

        complexity = sum(signals.values())
        pii = detect_pii(full_text)
        time_sensitive = self.time_sensitive_fn(user_text)
        personal = bool(PERSONAL_CONTEXT.search(user_text))

        analysis = PromptAnalysis(
            complexity=0.0,
            task_type=task,
            estimated_input_tokens=tokens,
            pii=pii,
            time_sensitive=time_sensitive,
            personal=personal,
            cacheable=not time_sensitive and not pii,
            signals=signals,
        )
        for scorer in self.extra_scorers:
            delta = scorer(request, analysis)
            if delta:
                signals[getattr(scorer, "__name__", "custom")] = float(delta)
                complexity += float(delta)

        analysis.complexity = round(max(0.0, min(1.0, complexity)), 4)
        return analysis
