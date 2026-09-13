"""Client for the configured Model Service - a local Ollama instance for now,
or the Gemini API when settings.model_backend is "gemini" (experiment/
gemini-reviewer-v2 only).

Swapping to the real DataNet Model Service later is expected to only
require changing settings.model_service_url: this module already talks to
it purely through Settings, and never hardcodes a host. call_model() is the
one seam every reviewer node, the parser, validator, and dedup call through -
none of them know or care which backend actually served a given prompt,
since both backends return the same ModelReviewOutput shape.
"""

import asyncio
from weakref import WeakKeyDictionary

import httpx
from pydantic import ValidationError

from review_agent.config import Settings
from review_agent.models.model_output import ModelReviewOutput


class ModelServiceError(Exception):
    """Raised when the Model Service cannot be called or its response cannot be parsed.

    reason_code is a stable machine-readable label for structured logging
    (see review_agent.logging.log_event) and for _call_gemini's retry
    decision - never shown to a GitHub user directly, since this exception's
    own message is already the safe, human-readable text (see _call_gemini).
    """

    def __init__(self, message: str, *, reason_code: str = "model_connection_failed") -> None:
        super().__init__(message)
        self.reason_code = reason_code


# One gate per event loop rather than a module-level singleton: a Semaphore's
# waiters belong to the loop that awaited them, and every asyncio.run - each
# test among them - creates a new loop. Keyed weakly so a finished loop's
# gate is collected along with it.
_CONCURRENCY_GATES: "WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]]" = (
    WeakKeyDictionary()
)


def _concurrency_gate(limit: int) -> asyncio.Semaphore:
    """Return this event loop's gate, admitting at most `limit` calls at once."""

    loop = asyncio.get_running_loop()
    cached = _CONCURRENCY_GATES.get(loop)
    if cached is None or cached[0] != limit:
        cached = (limit, asyncio.Semaphore(limit))
        _CONCURRENCY_GATES[loop] = cached
    return cached[1]


# Characters per token, calibrated against qwen2.5-coder:7b's own reported
# prompt_eval_count rather than assumed:
#
#   content            chars    real tokens    chars/token
#   prompt markdown    5,332          1,117           4.8
#   Python source      9,382          2,134           4.4
#   unified diff      16,980          6,609           2.6
#   YAML                 831            232           3.6
#
# Calibrated to the diff figure because the diff dominates a real reviewer
# prompt - the instructions are ~5k characters against a median 118k of diff.
# It therefore over-estimates prose, which is the safe direction for a guard:
# refusing a prompt that would have fit is a recorded error, while accepting
# one that does not fit is the silent truncation this replaces.
_CHARACTERS_PER_TOKEN = 2.5

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _estimate_token_count(prompt: str) -> int:
    """Approximate a prompt's token count from its length.

    Replaces a word count, which undercounted by up to 3.3x on diff content -
    "+    result_1 = process(doc_1, config)" is five words and roughly
    sixteen tokens. That is why the max_model_input_tokens guard never fired
    on a prompt that the server then quietly truncated.
    """

    return int(len(prompt) / _CHARACTERS_PER_TOKEN)


async def call_model(
    prompt: str, settings: Settings, client: httpx.AsyncClient | None = None
) -> ModelReviewOutput:
    """Send a prompt to the configured Model Service and return its structured output.

    Rejects an over-budget prompt before making any HTTP call. Every other
    failure - timeout, connection error, a non-200 response, or an
    unparseable body - is wrapped in ModelServiceError; no raw httpx
    exception ever escapes this function.

    client lets a caller pass in one pooled httpx.AsyncClient shared across
    an entire review run instead of opening a fresh TCP connection per call
    - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5. When omitted, this
    function falls back to its original self-contained per-call client, so
    every existing caller keeps working unchanged.
    At most settings.max_concurrent_model_calls of these run at once. Waiting
    happens here, before the request is sent, which is the point: a request
    queued inside the Model Service is already spending its
    model_service_timeout_seconds while the model has not started on it, so
    six concurrent reviewers timed out waiting rather than on any real work.
    Held only around the HTTP call, so the graph's parallel fan-out and
    everything else the reviewers do still overlap. The gate applies
    identically regardless of settings.model_backend.

    Dispatches to _call_ollama or _call_gemini based on settings.model_backend
    - the only place that decision is made. Everything before this dispatch
    (the token-budget check) and everything callers do with the result is
    backend-agnostic.
    """

    estimated_tokens = _estimate_token_count(prompt)
    if estimated_tokens > settings.max_model_input_tokens:
        raise ModelServiceError(
            f"Prompt is too large for the Model Service: an estimated "
            f"{estimated_tokens} tokens exceeds the configured limit of "
            f"{settings.max_model_input_tokens}."
        )

    async with _concurrency_gate(settings.max_concurrent_model_calls):
        if settings.model_backend == "gemini":
            return await _call_gemini(prompt, settings, client)
        return await _call_ollama(prompt, settings, client)


async def _call_ollama(
    prompt: str, settings: Settings, client: httpx.AsyncClient | None
) -> ModelReviewOutput:
    payload = {
        "model": settings.model_service_model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": settings.model_service_keep_alive,
        "options": {
            "num_ctx": settings.model_context_tokens,
            # Proven on PR #10 (all-zero diagnostic run): temperature=0 made
            # every one of the 6 reviewers greedily converge on the same
            # empty-array completion for a real, substantial diff - not a
            # parsing/validation bug, the model's own most-likely token
            # sequence. 0.3 keeps seed-based repeatability (below) while
            # giving the model room to not collapse onto the "safe" answer.
            "temperature": 0.3,
            "seed": settings.model_service_seed,
        },
    }

    try:
        if client is not None:
            response = await client.post(
                f"{settings.model_service_url}/api/generate",
                json=payload,
                timeout=settings.model_service_timeout_seconds,
            )
            response.raise_for_status()
        else:
            async with httpx.AsyncClient(
                timeout=settings.model_service_timeout_seconds
            ) as owned_client:
                response = await owned_client.post(
                    f"{settings.model_service_url}/api/generate", json=payload
                )
                response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ModelServiceError(
            f"Model Service returned {exc.response.status_code}.",
            reason_code=_classify_http_status(exc.response.status_code),
        ) from exc
    except httpx.RequestError as exc:
        raise ModelServiceError(
            "Model Service request failed.", reason_code=_classify_request_error(exc)
        ) from exc

    try:
        body = response.json()
        return ModelReviewOutput(raw_text=body["response"])
    except (ValueError, KeyError, TypeError, ValidationError) as exc:
        raise ModelServiceError(
            "Model Service returned an unparseable response.",
            reason_code="model_invalid_response",
        ) from exc


# 429/502/503/504 and connection/timeout failures are the transient cases a
# retry can plausibly fix; everything else (auth, bad request, model not
# found) will fail again identically on retry, so retrying it would only
# delay reporting a real, permanent configuration problem.
_RETRYABLE_REASON_CODES = frozenset(
    {"model_connection_failed", "model_timeout", "model_rate_limited", "model_service_unavailable"}
)
_MAX_GEMINI_ATTEMPTS = 3
_GEMINI_RETRY_BACKOFF_SECONDS = (1, 2)

_SAFE_MESSAGE_BY_REASON_CODE = {
    "model_rate_limited": "Gemini is temporarily rate-limited. Please try again shortly.",
    "model_service_unavailable": "Gemini is temporarily unavailable. Please try again shortly.",
    "model_timeout": "Gemini did not respond in time. Please try again shortly.",
    "model_connection_failed": "Could not reach Gemini. Please try again shortly.",
    "model_not_found": "The configured Gemini model is not available.",
    "model_auth_failed": "Gemini rejected the configured API key.",
    "model_invalid_response": "Gemini returned an unexpected response.",
}


def _classify_http_status(status_code: int) -> str:
    """Map an HTTP status code from either backend to a stable reason_code."""
    if status_code == 429:
        return "model_rate_limited"
    if status_code in (502, 503, 504):
        return "model_service_unavailable"
    if status_code == 404:
        return "model_not_found"
    if status_code in (401, 403):
        return "model_auth_failed"
    return "model_invalid_response"


def _classify_request_error(exc: httpx.RequestError) -> str:
    """httpx.TimeoutException is itself a RequestError subclass, so it must be
    checked first - otherwise every timeout would be misreported as a
    generic connection failure and become retryable for the wrong reason.
    """
    if isinstance(exc, httpx.TimeoutException):
        return "model_timeout"
    return "model_connection_failed"


async def _call_gemini_once(
    prompt: str, settings: Settings, client: httpx.AsyncClient | None
) -> ModelReviewOutput:
    """Make exactly one Gemini API call attempt - no retry logic here.

    The key is read from settings.llm_api_key (GEMINI_API_KEY, via
    .env.local - never committed) and sent only in the x-goog-api-key
    header, never in the URL. Every raised ModelServiceError message is
    already safe to show a GitHub user: it carries only a status code or
    reason_code, never the request, the key, or any raw exception text.
    """
    if settings.llm_api_key is None:
        raise ModelServiceError(
            "Gemini backend selected but no GEMINI_API_KEY is configured.",
            reason_code="model_connection_failed",
        )

    url = f"{_GEMINI_API_BASE}/{settings.llm_model}:generateContent"
    headers = {
        "x-goog-api-key": settings.llm_api_key.get_secret_value(),
        "Content-Type": "application/json",
    }
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        if client is not None:
            response = await client.post(
                url, headers=headers, json=body, timeout=settings.model_service_timeout_seconds,
            )
            response.raise_for_status()
        else:
            async with httpx.AsyncClient(
                timeout=settings.model_service_timeout_seconds
            ) as owned_client:
                response = await owned_client.post(url, headers=headers, json=body)
                response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        reason_code = _classify_http_status(exc.response.status_code)
        raise ModelServiceError(
            _SAFE_MESSAGE_BY_REASON_CODE[reason_code], reason_code=reason_code
        ) from exc
    except httpx.RequestError as exc:
        reason_code = _classify_request_error(exc)
        raise ModelServiceError(
            _SAFE_MESSAGE_BY_REASON_CODE[reason_code], reason_code=reason_code
        ) from exc

    try:
        data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        return ModelReviewOutput(raw_text=text)
    except (ValueError, KeyError, TypeError, IndexError, ValidationError) as exc:
        raise ModelServiceError(
            _SAFE_MESSAGE_BY_REASON_CODE["model_invalid_response"],
            reason_code="model_invalid_response",
        ) from exc


async def _call_gemini(
    prompt: str, settings: Settings, client: httpx.AsyncClient | None
) -> ModelReviewOutput:
    """Call Gemini, retrying a small bounded number of times on transient failures only.

    Only reason codes in _RETRYABLE_REASON_CODES (connection failures,
    timeouts, 429, 502/503/504) are retried, up to _MAX_GEMINI_ATTEMPTS
    total attempts with a short fixed backoff between them - never for
    authentication failures, invalid requests, or a missing model, since
    those fail identically every time and retrying only delays reporting a
    real configuration problem. The last attempt's error is what callers see.
    """
    last_error: ModelServiceError | None = None
    for attempt in range(_MAX_GEMINI_ATTEMPTS):
        try:
            return await _call_gemini_once(prompt, settings, client)
        except ModelServiceError as exc:
            last_error = exc
            is_last_attempt = attempt == _MAX_GEMINI_ATTEMPTS - 1
            if exc.reason_code not in _RETRYABLE_REASON_CODES or is_last_attempt:
                raise
            await asyncio.sleep(_GEMINI_RETRY_BACKOFF_SECONDS[attempt])
    raise last_error  # pragma: no cover - loop always returns or raises above
