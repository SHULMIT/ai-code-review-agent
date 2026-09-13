import json

import httpx
import pytest

from review_agent.models.review_result import ReviewResult
from review_agent.tools.github_publisher import (
    GitHubPublisherError,
    _github_event_for,
    publish_review,
)


def _review_result(**overrides) -> ReviewResult:
    values = {
        "pr_summary": "Reviewed 1 changed files across 1 components; found 0 blocking, 0 warning, 0 suggestion findings.",
        "architecture_score": 100,
        "code_quality_score": 100,
        "production_readiness_score": 100,
        "blocking_findings": [],
        "warnings": [],
        "suggestions": [],
        "recommendation": "APPROVE",
        "reviewed_head_sha": "abc123",
        "errors": [],
    }
    values.update(overrides)
    return ReviewResult(**values)


def _patch_async_client(monkeypatch, transport: httpx.MockTransport) -> None:
    """Make github_publisher's HTTP calls use a local mock transport."""

    class MockedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockedAsyncClient)


async def test_publish_review_submits_an_approve_event(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 1})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await publish_review(
        "example-org/example-repo", 105, "abc123", _review_result(recommendation="APPROVE"), "fake-token"
    )

    assert captured["url"] == "https://api.github.com/repos/example-org/example-repo/pulls/105/reviews"
    assert captured["body"]["event"] == "APPROVE"
    assert captured["body"]["commit_id"] == "abc123"


async def test_publish_review_uses_a_provided_client_instead_of_opening_its_own(monkeypatch):
    """When a shared client is passed in, publish_review must reuse it, never construct a new one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": 1})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as shared_client:

        def _fail_if_constructed(*args, **kwargs):
            raise AssertionError("a fresh AsyncClient must not be constructed when one is provided")

        monkeypatch.setattr(httpx, "AsyncClient", _fail_if_constructed)

        review_id = await publish_review(
            "example-org/example-repo", 105, "abc123", _review_result(), "fake-token", client=shared_client
        )

    assert review_id == 1


async def test_publish_review_submits_a_comment_event_for_an_incomplete_recommendation(monkeypatch):
    """INCOMPLETE must never surface as APPROVE or REQUEST_CHANGES - COMMENT is the only safe event."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 1})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await publish_review(
        "example-org/example-repo",
        105,
        "abc123",
        _review_result(recommendation="INCOMPLETE", incomplete_categories=["security"]),
        "fake-token",
    )

    assert captured["body"]["event"] == "COMMENT"


def test_github_event_for_maps_all_three_recommendations_explicitly():
    assert _github_event_for("APPROVE") == "APPROVE"
    assert _github_event_for("REQUEST_CHANGES") == "REQUEST_CHANGES"
    assert _github_event_for("INCOMPLETE") == "COMMENT"


def test_github_event_for_raises_on_an_unmapped_recommendation():
    with pytest.raises(GitHubPublisherError, match="no GitHub review event is defined"):
        _github_event_for("SOMETHING_ELSE")


async def test_publish_review_submits_a_request_changes_event(monkeypatch):
    """Confirms the recommendation -> GitHub event mapping is correct in both directions."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 1})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await publish_review(
        "example-org/example-repo",
        105,
        "abc123",
        _review_result(recommendation="REQUEST_CHANGES"),
        "fake-token",
    )

    assert captured["body"]["event"] == "REQUEST_CHANGES"


async def test_publish_review_includes_the_formatted_markdown_body(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 1})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await publish_review(
        "example-org/example-repo", 105, "abc123", _review_result(), "fake-token"
    )

    assert "# DataNet Code Review" in captured["body"]["body"]
    assert "**APPROVE**" in captured["body"]["body"]
    assert captured["body"]["body"].count("https://shulamit-noa.netlify.app/") == 1


async def test_publish_review_wraps_a_422_response(monkeypatch):
    """e.g. commit_id doesn't match any commit actually on the pull request."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"message": "Unprocessable Entity"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubPublisherError, match="422"):
        await publish_review(
            "example-org/example-repo", 105, "abc123", _review_result(), "fake-token"
        )


async def test_publish_review_wraps_a_network_failure(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubPublisherError):
        await publish_review(
            "example-org/example-repo", 105, "abc123", _review_result(), "fake-token"
        )


async def test_publish_review_returns_the_new_review_id_when_creating(monkeypatch):
    """The first publish for a PR has no existing_review_id and creates via POST."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(200, json={"id": 42})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    review_id = await publish_review(
        "example-org/example-repo", 105, "abc123", _review_result(), "fake-token"
    )

    assert review_id == 42


async def test_publish_review_updates_the_existing_review_via_put_when_given_an_id(monkeypatch):
    """A re-run with an unchanged recommendation updates the prior review's body in place."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 555})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    review_id = await publish_review(
        "example-org/example-repo",
        105,
        "abc123",
        _review_result(),
        "fake-token",
        existing_review_id=555,
    )

    assert review_id == 555
    assert captured["method"] == "PUT"
    assert captured["url"] == "https://api.github.com/repos/example-org/example-repo/pulls/105/reviews/555"
    assert set(captured["body"].keys()) == {"body"}
    assert "# DataNet Code Review" in captured["body"]["body"]


async def test_publish_review_falls_back_to_creating_a_new_review_when_the_existing_one_is_gone(
    monkeypatch,
):
    """A stored review ID that 404s on GitHub (e.g. deleted manually) creates a new review instead."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "PUT":
            return httpx.Response(404, json={"message": "Not Found"})
        assert request.method == "POST"
        return httpx.Response(200, json={"id": 999})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    review_id = await publish_review(
        "example-org/example-repo",
        105,
        "abc123",
        _review_result(),
        "fake-token",
        existing_review_id=555,
    )

    assert review_id == 999
    assert calls == ["PUT", "POST"]


async def test_publish_review_does_not_fall_back_on_a_non_404_update_failure(monkeypatch):
    """A real failure updating the existing review (e.g. 500) must not silently create a new one."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(500, json={"message": "Internal Server Error"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubPublisherError, match="500"):
        await publish_review(
            "example-org/example-repo",
            105,
            "abc123",
            _review_result(),
            "fake-token",
            existing_review_id=555,
        )

    assert calls == ["PUT"]


async def test_publish_review_wraps_a_response_missing_the_review_id(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubPublisherError, match="Unexpected response shape"):
        await publish_review(
            "example-org/example-repo", 105, "abc123", _review_result(), "fake-token"
        )


async def test_publish_review_wraps_a_non_integer_review_id(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "not-a-number"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubPublisherError, match="Unexpected response shape"):
        await publish_review(
            "example-org/example-repo", 105, "abc123", _review_result(), "fake-token"
        )


async def test_publish_review_wraps_a_network_failure_during_an_update(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubPublisherError, match="Network error while updating"):
        await publish_review(
            "example-org/example-repo",
            105,
            "abc123",
            _review_result(),
            "fake-token",
            existing_review_id=555,
        )
