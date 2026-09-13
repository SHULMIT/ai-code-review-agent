"""Tests for the GitHub Checks API client backing the on-demand review trigger."""

import json

import httpx
import pytest

from review_agent.tools.check_run_tool import (
    CHECK_RUN_NAME,
    CheckRunToolError,
    complete_check_run,
    create_or_reset_check_run,
    create_pending_check_run,
    find_check_run_for_sha,
    mark_check_run_in_progress,
    reset_check_run_to_pending,
    update_check_run_progress,
)


def _patch_async_client(monkeypatch, transport: httpx.MockTransport) -> None:
    class _MockedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _MockedAsyncClient)


async def test_create_pending_check_run_creates_a_completed_neutral_check_with_the_action(
    monkeypatch,
):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 111})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    check_run_id = await create_pending_check_run(
        "example-org/example-repo", "abc123", "fake-token"
    )

    assert check_run_id == 111
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.github.com/repos/example-org/example-repo/check-runs"
    assert captured["body"]["name"] == CHECK_RUN_NAME
    assert captured["body"]["head_sha"] == "abc123"
    assert captured["body"]["status"] == "completed"
    assert captured["body"]["conclusion"] == "neutral"
    assert captured["body"]["actions"] == [
        {
            "label": "Run Code Review",
            "description": "Run the DataNet AI code review",
            "identifier": "run_code_review",
        }
    ]


async def test_find_check_run_for_sha_returns_the_existing_id(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["check_name"] == CHECK_RUN_NAME
        return httpx.Response(200, json={"total_count": 1, "check_runs": [{"id": 222}]})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    assert await find_check_run_for_sha("example-org/example-repo", "abc123", "fake-token") == 222


async def test_find_check_run_for_sha_returns_none_when_absent(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"total_count": 0, "check_runs": []})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    assert await find_check_run_for_sha("example-org/example-repo", "abc123", "fake-token") is None


async def test_reset_check_run_to_pending_patches_the_existing_check_run(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 222})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await reset_check_run_to_pending("example-org/example-repo", 222, "fake-token")

    assert captured["method"] == "PATCH"
    assert captured["url"] == "https://api.github.com/repos/example-org/example-repo/check-runs/222"
    assert "head_sha" not in captured["body"]
    assert captured["body"]["conclusion"] == "neutral"


async def test_create_or_reset_check_run_resets_an_existing_check_run_for_the_same_sha(
    monkeypatch,
):
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"total_count": 1, "check_runs": [{"id": 333}]})
        return httpx.Response(200, json={"id": 333})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    check_run_id = await create_or_reset_check_run("example-org/example-repo", "abc123", "fake-token")

    assert check_run_id == 333
    assert requests == ["GET", "PATCH"]


async def test_create_or_reset_check_run_creates_a_new_one_for_a_new_sha(monkeypatch):
    """A fresh push's SHA has no existing Check Run - a new one is created, never reused."""
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"total_count": 0, "check_runs": []})
        return httpx.Response(200, json={"id": 444})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    check_run_id = await create_or_reset_check_run("example-org/example-repo", "def456", "fake-token")

    assert check_run_id == 444
    assert requests == ["GET", "POST"]


async def test_mark_check_run_in_progress_patches_status_only(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 555})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await mark_check_run_in_progress("example-org/example-repo", 555, "fake-token")

    assert captured["method"] == "PATCH"
    assert captured["body"]["status"] == "in_progress"
    assert "actions" not in captured["body"]


async def test_mark_check_run_in_progress_accepts_a_custom_title_and_summary(monkeypatch):
    """The immediate "0/6 reviewers completed" text must reach the same PATCH call."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 555})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await mark_check_run_in_progress(
        "example-org/example-repo", 555, "fake-token",
        title="Code review is running", summary="0/6 reviewers completed",
    )

    assert captured["body"]["status"] == "in_progress"
    assert captured["body"]["output"]["title"] == "Code review is running"
    assert captured["body"]["output"]["summary"] == "0/6 reviewers completed"


async def test_update_check_run_progress_keeps_status_in_progress(monkeypatch):
    """Live per-reviewer updates must never flip the Check Run away from in_progress."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 555})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await update_check_run_progress(
        "example-org/example-repo", 555, "fake-token",
        "Code Review in progress — 3/6 reviewers completed", "✅ Security\n⏳ Architecture",
    )

    assert captured["body"]["status"] == "in_progress"
    assert captured["body"]["output"]["title"] == "Code Review in progress — 3/6 reviewers completed"
    assert "⏳ Architecture" in captured["body"]["output"]["summary"]


async def test_complete_check_run_sends_status_conclusion_and_output_in_one_request(monkeypatch):
    """The final state must reach GitHub as a single atomic PATCH, not status first and
    output separately - otherwise a viewer can briefly see "completed" with stale
    in_progress output, or the reverse, between two round trips."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"id": 555})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await complete_check_run(
        "example-org/example-repo", 555, "success", "Approved", "no findings", "fake-token"
    )

    assert len(calls) == 1
    body = calls[0]
    assert body["status"] == "completed"
    assert body["conclusion"] == "success"
    assert body["output"] == {"title": "Approved", "summary": "no findings"}
    assert body["actions"][0]["identifier"] == "run_code_review"


async def test_complete_check_run_omits_the_action_when_rerun_is_disallowed(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 555})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await complete_check_run(
        "example-org/example-repo", 555, "neutral", "Stale", "moved on", "fake-token",
        allow_rerun=False,
    )

    assert "actions" not in captured["body"]


async def test_check_run_request_wraps_a_non_200_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(CheckRunToolError, match="500"):
        await create_pending_check_run("example-org/example-repo", "abc123", "fake-token")


async def test_check_run_request_wraps_a_network_failure(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(CheckRunToolError):
        await mark_check_run_in_progress("example-org/example-repo", 555, "fake-token")


async def test_find_check_run_wraps_an_unexpected_response_shape(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(CheckRunToolError):
        await find_check_run_for_sha("example-org/example-repo", "abc123", "fake-token")


async def test_create_pending_check_run_wraps_a_response_missing_the_id(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(CheckRunToolError, match="Unexpected response shape"):
        await create_pending_check_run("example-org/example-repo", "abc123", "fake-token")


async def test_create_pending_check_run_uses_a_provided_client_instead_of_opening_its_own(
    monkeypatch,
):
    """When a shared client is passed in, the tool must reuse it, never construct a new one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": 111})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as shared_client:

        def _fail_if_constructed(*args, **kwargs):
            raise AssertionError("a fresh AsyncClient must not be constructed when one is provided")

        monkeypatch.setattr(httpx, "AsyncClient", _fail_if_constructed)

        check_run_id = await create_pending_check_run(
            "example-org/example-repo", "abc123", "fake-token", client=shared_client
        )

    assert check_run_id == 111
