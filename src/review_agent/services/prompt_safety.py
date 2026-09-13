"""Heuristic detection of prompt-injection phrasing in untrusted PR content."""

# Case-insensitive phrases that resemble an attempt to override the
# reviewer's system instructions. Kept short and literal.
_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all prior",
    "you are now",
    "SYSTEM:",
    "new instructions:",
    "disregard the above",
]


def scan_for_injection_patterns(text: str) -> list[str]:
    """Return which known suspicious phrases appear in text, case-insensitively.

    This is a heuristic, not a guarantee: it flags phrasing that resembles a
    prompt-injection attempt so downstream code (prompt_builder.py) can note
    it, but it does not claim to catch every attempt, and it never blocks or
    removes anything by itself. It is one signal that informs prompt
    construction and validation - it is not the security boundary on its
    own; that boundary is the untrusted-content delimiting the prompt itself
    provides. Per AGENTS.md's rule against logging raw PR content, a caller
    must only ever record which pattern matched (this function's return
    value) - never the surrounding text.
    """
    if not text:
        return []
    lowered = text.lower()
    return [pattern for pattern in _INJECTION_PATTERNS if pattern.lower() in lowered]
