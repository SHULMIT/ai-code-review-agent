"""Normalizes a model's raw text response before it is parsed as JSON.

Models commonly wrap structured output in markdown formatting (a ```json
code fence, or a bare ``` fence) even when explicitly instructed to respond
with JSON only. Observed in practice against a real local model: once the
prompt includes more context (e.g. a cited documentation snippet), the same
model also tends to wrap its findings array in a single named object key -
{"findings": [...]}, {"results": [...]}, or similar - instead of returning
the bare array the prompt asks for. This module normalizes both cases so
callers can parse the actual content underneath - it is a normalization
step, not a validation step. Whether the parsed value is a well-formed list
of Finding-shaped objects is still decided entirely by finding_validation.py.
"""

import json
import re
from typing import Any

_CODE_FENCE_RE = re.compile(r"^```[ \t]*\w*[ \t]*\r?\n(.*?)\r?\n?```$", re.DOTALL)


class ModelOutputParsingError(Exception):
    """Raised when a model's raw text cannot be parsed as JSON, even after fence-stripping."""


def extract_json_from_model_output(raw_text: str) -> Any:
    """Strip a surrounding markdown code fence (if present) and parse the result as JSON.

    Only strips a fence that wraps the entire trimmed text - a leading
    ```json, ```JSON, or bare ``` line and a trailing ``` line - never
    backticks appearing elsewhere inside the content. Raises
    ModelOutputParsingError with the underlying parse error if the result
    still is not valid JSON, so a genuinely malformed response is never
    silently swallowed into None or an empty result. The parsed value then
    passes through _unwrap_findings_object() before being returned.
    """
    candidate = raw_text.strip()
    match = _CODE_FENCE_RE.match(candidate)
    if match:
        candidate = match.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except ValueError as exc:
        raise ModelOutputParsingError(f"model output is not valid JSON: {exc}") from exc

    return _unwrap_findings_object(parsed)


def _unwrap_findings_object(parsed: Any) -> Any:
    """Unwrap a single-key {"<name>": [...]} wrapper some models use instead of a bare array.

    A bare list or any non-dict value (int, str, bool, None, ...) passes
    through unchanged - validating that the result is actually a list of
    findings is finding_validation.py's job, not this function's. Only a
    dict is inspected here, and only one shape is ever unwrapped: exactly
    one key whose value is a list, regardless of what that key is named
    (models have been observed using "findings", "results", and others -
    the key name itself carries no meaning worth hardcoding). A dict with
    more than one key, or a single key whose value isn't a list, is a
    genuine format mismatch - it is rejected with the actual shape shown
    rather than guessed at further.
    """
    if not isinstance(parsed, dict):
        return parsed

    if len(parsed) == 1:
        (only_value,) = parsed.values()
        if isinstance(only_value, list):
            return only_value

    raise ModelOutputParsingError(
        "model output is a JSON object that isn't a single-key array wrapper "
        f"(expected {{\"<name>\": [...]}}), got: {parsed!r}"
    )
