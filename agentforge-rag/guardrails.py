"""
AgentForge-RAG guardrails — Author: Ashutosh Sahoo

Zero-dependency input/output safety:
  * prompt-injection detection (pattern based, unit-testable)
  * PII redaction (email, Indian phone, Aadhaar, PAN, credit card)

These are defence-in-depth heuristics, not a complete safety system.
In production, layer them with a classifier model + human review queues.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- injection
INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above|your)\s+instructions",
    r"forget\s+(everything|all|your\s+instructions)",
    r"you\s+are\s+now\s+(a\s+)?(different|new|unrestricted|dan|jailbroken)",
    r"system\s*prompt",
    r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"show\s+me\s+your\s+(instructions|prompt|rules)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+(rules|restrictions|guardrails)",
    r"bypass\s+(your\s+)?(safety|guardrails|filters|restrictions)",
    r"jailbreak",
    r"developer\s+mode",
    r"override\s+(your\s+)?(safety|instructions|rules)",
    r"do\s+not\s+follow\s+(your\s+)?(rules|instructions)",
    r"pretend\s+(you\s+are|to\s+be)\s+.*(unrestricted|no\s+rules)",
    r"output\s+(the\s+)?(raw\s+)?(system|initial)\s+(prompt|message)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> dict:
    """Return {'blocked': bool, 'matched': [pattern, ...]}."""
    if not text:
        return {"blocked": False, "matched": []}
    matched = [p.pattern for p in _COMPILED if p.search(text)]
    return {"blocked": bool(matched), "matched": matched}


def is_injection(text: str) -> bool:
    return detect_injection(text)["blocked"]


# ---------------------------------------------------------------- PII
# Order matters: cards / Aadhaar / PAN before generic phone/number patterns.
_PII_RULES: list[tuple[str, re.Pattern, str]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    # Credit cards: 13-19 digits, optionally separated by space/dash (checked loosely)
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CARD]"),
    # Aadhaar: 12 digits as 4-4-4 groups (or contiguous, handled by card rule order-safe length check below)
    ("AADHAAR", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"), "[REDACTED_AADHAAR]"),
    # PAN: ABCDE1234F
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[REDACTED_PAN]"),
    # Indian mobile: +91 / 0 prefix optional, 10 digits starting 6-9
    ("PHONE", re.compile(r"(?:\+91[\-\s]?|0)?[6-9]\d{9}\b"), "[REDACTED_PHONE]"),
    # Generic international-ish phone fallback
    ("PHONE", re.compile(r"\+\d{1,3}[\-\s]\d{6,12}\b"), "[REDACTED_PHONE]"),
]


def _luhn_ok(digits: str) -> bool:
    if not (13 <= len(digits) <= 19):
        return False
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact_pii(text: str) -> str:
    """Redact PII. Credit-card-like sequences are only redacted if Luhn-valid
    or clearly grouped (4-4-4-4), to avoid eating ordinary long numbers/dates.
    Aadhaar 4-4-4 grouped numbers run before the generic card rule in effect
    because a 12-digit grouped number is not Luhn-redacted as a card."""
    if not text:
        return text
    out = text
    for kind, pat, repl in _PII_RULES:
        if kind == "CREDIT_CARD":
            def _sub(m: re.Match) -> str:
                raw = m.group(0)
                digits = re.sub(r"\D", "", raw)
                # 4-4-4-4 grouped or Luhn-valid -> card; 4-4-4 grouped -> Aadhaar handled below
                if len(digits) == 12 and " " in raw:
                    return raw  # leave for AADHAAR rule
                if _luhn_ok(digits):
                    return repl
                return raw
            out = pat.sub(_sub, out)
        else:
            out = pat.sub(repl, out)
    return out


def find_pii(text: str) -> list[str]:
    """Return the kinds of PII present (for logging / tests)."""
    found: list[str] = []
    redacted = redact_pii(text)
    for tag in ("EMAIL", "CARD", "AADHAAR", "PAN", "PHONE"):
        if f"[REDACTED_{tag}]" in redacted:
            found.append(tag)
    return found


def guard_input(text: str) -> dict:
    """Full input guard: injection check + redacted copy for downstream use."""
    inj = detect_injection(text)
    return {
        "blocked": inj["blocked"],
        "matched_patterns": inj["matched"],
        "redacted_text": redact_pii(text),
        "pii_found": find_pii(text),
    }
