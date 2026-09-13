"""Tests for the Model Service client (targeting a local Ollama instance)."""

import json
import asyncio
from pathlib import Path

import httpx
import pytest

from review_agent.config import Settings
from review_agent.tools import model_service_tool
from review_agent.tools.model_service_tool import ModelServiceError, call_model


def _settings(**overrides) -> Settings:
    """Build a valid Settings instance without touching the local .env file."""

    values = {
        "_env_file": None,
        "github_app_id": 123,
        "github_app_private_key_path": Path("secrets/test-private-key.pem"),
        "github_webhook_secret": "development-webhook-secret",
        "model_service_url": "http://localhost:11434",
        "model_service_model_name": "qwen2.5:7b",
        "model_service_timeout_seconds": 5,
    }
    values.update(overrides)
    return Settings(**values)


def _patch_async_client(monkeypatch, transport: httpx.MockTransport) -> None:
    """Make model_service_tool's HTTP calls use a local mock transport."""

    class MockedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockedAsyncClient)


async def test_call_model_returns_the_parsed_response_on_success(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content
        return httpx.Response(200, json={"response": "looks good to me"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    result = await call_model("review this diff", _settings())

    assert result.raw_text == "looks good to me"
    assert captured["url"] == "http://localhost:11434/api/generate"


async def test_call_model_wraps_a_timeout(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError):
        await call_model("review this diff", _settings())


async def test_call_model_wraps_a_connection_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError):
        await call_model("review this diff", _settings())


async def test_call_model_wraps_a_non_200_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError, match="500"):
        await call_model("review this diff", _settings())


async def test_call_model_wraps_an_unparseable_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError):
        await call_model("review this diff", _settings())


async def test_call_model_sends_the_configured_keep_alive(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "ok"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await call_model("review this diff", _settings(model_service_keep_alive="45m"))

    assert captured["body"]["keep_alive"] == "45m"


async def test_call_model_uses_a_provided_client_instead_of_opening_its_own(monkeypatch):
    """When a shared client is passed in, call_model must reuse it, never construct a new one."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"response": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as shared_client:

        def _fail_if_constructed(*args, **kwargs):
            raise AssertionError("a fresh AsyncClient must not be constructed when one is provided")

        monkeypatch.setattr(httpx, "AsyncClient", _fail_if_constructed)

        result = await call_model("review this diff", _settings(), client=shared_client)

    assert result.raw_text == "ok"
    assert captured["url"] == "http://localhost:11434/api/generate"


async def test_call_model_rejects_an_oversized_prompt_before_any_http_call(monkeypatch):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"response": "should never be reached"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))
    oversized_prompt = " ".join("word" for _ in range(10))

    with pytest.raises(ModelServiceError):
        await call_model(oversized_prompt, _settings(max_model_input_tokens=5))

    assert calls["count"] == 0


async def test_call_model_sends_the_configured_context_window(monkeypatch):
    """Without num_ctx the server silently applies its own 2048-token default.

    Measured against the real qwen2.5-coder:7b: a 16,386-token prompt came
    back reporting 2,050 tokens evaluated, and the part discarded is the
    beginning - so the reviewer's instructions went first and the model was
    left with the tail of the diff. Asserting the option is actually on the
    wire is the only way to catch that returning, since a truncated prompt
    still produces a perfectly well-formed response.
    """
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "[]"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(model_service_tool.httpx, "AsyncClient", FakeClient)

    settings = _settings()
    await model_service_tool.call_model("a prompt", settings)

    assert captured["json"]["options"]["num_ctx"] == settings.model_context_tokens
    assert settings.model_context_tokens == 32_768
    assert captured["json"]["options"]["temperature"] == 0.3
    assert captured["json"]["options"]["seed"] == settings.model_service_seed


def test_token_estimate_is_calibrated_against_measured_diff_content():
    """A word count undercounted diff text by 3.3x, so the input guard never fired.

    Calibrated against the model's own reported prompt_eval_count: 16,980
    characters of diff measured 6,609 real tokens. The estimate must land
    close to that and, where it misses, must miss high - refusing a prompt
    that would have fit is a recorded error, while accepting one that does
    not is silent truncation.
    """
    diff = "".join(f"+    result_{i} = process(doc_{i}, config)\n" for i in range(400))
    measured_real_tokens = 6_609

    estimated = model_service_tool._estimate_token_count(diff)

    assert estimated >= measured_real_tokens, "the estimate must never undercount"
    assert estimated <= measured_real_tokens * 1.3, "and must not be wildly pessimistic either"
    assert len(diff.split()) < measured_real_tokens / 3, "the old word count was this far off"


def test_the_input_guard_sits_below_the_context_window():
    """A prompt budget above the window would be refused by nobody and truncated by the server."""
    settings = _settings()

    assert settings.max_model_input_tokens < settings.model_context_tokens


class _ConcurrencyRecordingTransport(httpx.AsyncBaseTransport):
    """Records how many requests are in flight at once, and how many finished."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0
        self.completed = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            # Yields control, so any call not held back by the gate gets its
            # chance to start and be counted while this one is still open.
            await asyncio.sleep(0.05)
            return httpx.Response(200, json={"response": "[]"})
        finally:
            self.in_flight -= 1
            self.completed += 1


# Captured before any test patches httpx.AsyncClient. _patch_async_client
# subclasses whatever httpx.AsyncClient currently is, so patching twice in one
# test nests the classes and the inner __init__ overwrites the outer one's
# transport - the second transport then records nothing at all.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


async def _call_all_six(monkeypatch, limit: int) -> _ConcurrencyRecordingTransport:
    """Fan six reviewer calls out at once against a gate of the given size."""

    transport = _ConcurrencyRecordingTransport()

    class PristineMockedAsyncClient(_REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PristineMockedAsyncClient)

    settings = _settings(max_concurrent_model_calls=limit)
    await asyncio.gather(
        *(call_model(f"reviewer {index} prompt", settings) for index in range(6))
    )
    return transport


async def test_call_model_never_exceeds_the_configured_concurrency(monkeypatch):
    """Six reviewers fan out at once; only max_concurrent_model_calls may be in flight.

    Measured against the real Ollama box, six concurrent 50,000-character
    prompts left two answered and the server closing the other four itself
    after 684s - and those four spent their whole timeout queued inside the
    server rather than being worked on. Holding the extra callers here, before
    the request is sent, is what stops that.
    """
    transport = await _call_all_six(monkeypatch, limit=3)

    assert transport.peak_in_flight == 3, (
        f"the gate admitted {transport.peak_in_flight} calls at once, not 3"
    )
    assert transport.completed == 6, "every reviewer must still get its turn"


async def test_the_concurrency_gate_still_allows_real_overlap(monkeypatch):
    """The gate limits parallelism; it must not collapse it to one at a time.

    A limit that quietly serialised every call would pass the test above just
    as well, and would throw away the fan-out the graph is built around.
    """
    transport = await _call_all_six(monkeypatch, limit=3)

    assert transport.peak_in_flight > 1, "calls ran one at a time - the fan-out was lost"


async def test_a_concurrency_limit_of_one_serialises_calls(monkeypatch):
    """The gate is honoured at its lower bound too, so the setting is a real dial."""
    transport = await _call_all_six(monkeypatch, limit=1)

    assert transport.peak_in_flight == 1
    assert transport.completed == 6


def test_the_gate_is_not_shared_across_event_loops(monkeypatch):
    """Each asyncio.run gets its own gate; a leaked one would deadlock a later loop.

    A module-level Semaphore looks like the obvious implementation and is
    unsound: its waiters belong to the loop that awaited them. This is the
    regression test for that - the second run must behave exactly like the
    first, not hang or admit the wrong number.
    """
    first = asyncio.run(_call_all_six(monkeypatch, limit=3))
    second = asyncio.run(_call_all_six(monkeypatch, limit=3))

    assert first.peak_in_flight == 3
    assert second.peak_in_flight == 3
    assert second.completed == 6


# ---- Gemini backend (experiment/gemini-reviewer-v2) --------------------

_FAKE_GEMINI_KEY = "fake-gemini-key-for-tests-only"


def _gemini_settings(**overrides) -> Settings:
    values = {"model_backend": "gemini", "llm_api_key": _FAKE_GEMINI_KEY, "llm_model": "gemini-3.6-flash"}
    values.update(overrides)
    return _settings(**values)


def _gemini_response(text: str, *, thought_text: str | None = None) -> dict:
    parts = []
    if thought_text is not None:
        parts.append({"text": thought_text, "thought": True})
    parts.append({"text": text})
    return {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP", "index": 0}]}


async def test_call_model_sends_the_key_only_in_a_header_never_the_url(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["header"] = request.headers.get("x-goog-api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_gemini_response("[]"))

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    result = await call_model("review this diff", _gemini_settings())

    assert _FAKE_GEMINI_KEY not in captured["url"]
    assert captured["header"] == _FAKE_GEMINI_KEY
    assert "gemini-3.6-flash" in captured["url"]
    assert captured["body"] == {"contents": [{"parts": [{"text": "review this diff"}]}]}
    assert result.raw_text == "[]"


async def test_call_model_joins_multiple_gemini_parts_and_skips_thought_parts(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response("[real answer]", thought_text="internal reasoning"))

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    result = await call_model("prompt", _gemini_settings())

    assert result.raw_text == "[real answer]"
    assert "internal reasoning" not in result.raw_text


async def _no_delay_sleep(_seconds: float) -> None:
    """Stand in for asyncio.sleep in retry tests so exhausting retries costs no wall-clock time."""


async def test_call_model_wraps_a_gemini_auth_error_without_leaking_the_key_or_retrying(monkeypatch):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError) as exc_info:
        await call_model("prompt", _gemini_settings())

    assert exc_info.value.reason_code == "model_auth_failed"
    assert _FAKE_GEMINI_KEY not in str(exc_info.value)
    assert "invalid key" not in str(exc_info.value)
    assert calls["count"] == 1, "an auth failure must never be retried"


async def test_call_model_wraps_a_gemini_network_failure_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(model_service_tool.asyncio, "sleep", _no_delay_sleep)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError) as exc_info:
        await call_model("prompt", _gemini_settings())

    assert exc_info.value.reason_code == "model_connection_failed"
    assert calls["count"] == model_service_tool._MAX_GEMINI_ATTEMPTS


async def test_call_model_wraps_an_unparseable_gemini_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError) as exc_info:
        await call_model("prompt", _gemini_settings())

    assert exc_info.value.reason_code == "model_invalid_response"


async def test_call_model_retries_a_gemini_503_and_succeeds(monkeypatch):
    """PR #13 hit a real transient Gemini 503 - a bounded retry must recover from it."""
    monkeypatch.setattr(model_service_tool.asyncio, "sleep", _no_delay_sleep)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < model_service_tool._MAX_GEMINI_ATTEMPTS:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json=_gemini_response("[]"))

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    result = await call_model("prompt", _gemini_settings())

    assert result.raw_text == "[]"
    assert calls["count"] == model_service_tool._MAX_GEMINI_ATTEMPTS


async def test_call_model_gives_up_after_repeated_gemini_503s(monkeypatch):
    monkeypatch.setattr(model_service_tool.asyncio, "sleep", _no_delay_sleep)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503, text="service unavailable")

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError) as exc_info:
        await call_model("prompt", _gemini_settings())

    assert exc_info.value.reason_code == "model_service_unavailable"
    assert "temporarily unavailable" in str(exc_info.value)
    assert calls["count"] == model_service_tool._MAX_GEMINI_ATTEMPTS


async def test_call_model_retries_a_gemini_timeout(monkeypatch):
    monkeypatch.setattr(model_service_tool.asyncio, "sleep", _no_delay_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError) as exc_info:
        await call_model("prompt", _gemini_settings())

    assert exc_info.value.reason_code == "model_timeout"


async def test_call_model_retries_a_gemini_429(monkeypatch):
    monkeypatch.setattr(model_service_tool.asyncio, "sleep", _no_delay_sleep)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(429, text="rate limited")

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError) as exc_info:
        await call_model("prompt", _gemini_settings())

    assert exc_info.value.reason_code == "model_rate_limited"
    assert calls["count"] == model_service_tool._MAX_GEMINI_ATTEMPTS


async def test_call_model_does_not_retry_a_gemini_404(monkeypatch):
    """A model-not-found failure will not change on retry, so it must fail once, immediately."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(404, text="model not found")

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(ModelServiceError) as exc_info:
        await call_model("prompt", _gemini_settings())

    assert exc_info.value.reason_code == "model_not_found"
    assert calls["count"] == 1


async def test_call_model_refuses_gemini_backend_with_no_api_key_configured():
    with pytest.raises(ModelServiceError, match="no GEMINI_API_KEY"):
        await call_model("prompt", _gemini_settings(llm_api_key=None))


async def test_call_model_still_uses_ollama_when_backend_is_not_gemini(monkeypatch):
    """Default/Version A behavior is completely unchanged by the Gemini addition."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"response": "[]"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    result = await call_model("prompt", _settings())

    assert "generativelanguage.googleapis.com" not in captured["url"]
    assert result.raw_text == "[]"
