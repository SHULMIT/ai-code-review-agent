"""End-to-end failure-safety tests: a review must never be marked completed,
or published to GitHub, unless real analysis actually happened.
"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from review_agent.config import Settings
from review_agent.graph.build_graph import build_graph
from review_agent.graph.state import ReviewState
from review_agent.models.finding import Finding
from review_agent.models.review_input import ReviewInput
from review_agent.models.review_result import ReviewResult
from review_agent.nodes.finalize_review import SCORED_CATEGORIES
from review_agent.services import review_launcher as review_launcher_module
from review_agent.services.event_router import ReviewTrigger
from review_agent.services.review_launcher import (
    build_publish_block_reason,
    build_review_launcher,
)
from review_agent.storage.review_runs import ReviewRun, ReviewRunStatus

_FAKE_SECRET = "ghs_super-secret-installation-token-must-never-leak"


def _finding(**overrides) -> Finding:
    values = {
        "severity": "SUGGESTION",
        "component": "Agent Harness / LangGraph",
        "review_category": "code quality",
        "file": "a.py",
        "line": 7,
        "problem": "p",
        "why_it_matters": "w",
        "suggested_fix": "f",
    }
    values.update(overrides)
    return Finding(**values)


def _review_input(**overrides) -> ReviewInput:
    values = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": "abc123",
        "included_diff": "diff --git a/a.py b/a.py\n@@ -1,2 +1,3 @@\n+print('hi')\n",
        "diff_truncated": False,
        "included_files": ["a.py"],
        "omitted_files": [],
        "repository_context": {"results": []},
        "documentation_context": [],
        "coverage_notes": [],
    }
    values.update(overrides)
    return ReviewInput(**values)


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


def _review_run() -> ReviewRun:
    return ReviewRun(
        id=uuid4(),
        github_delivery_id="delivery-1",
        review_identity="identity",
        repository="example-org/example-repo",
        pr_number=1,
        installation_id=123,
        trigger=ReviewTrigger.PULL_REQUEST_OPENED,
        head_sha="abc123",
        status=ReviewRunStatus.RUNNING,
        created_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


# --- build_publish_block_reason: pure-function unit tests -------------------


def test_build_publish_block_reason_blocks_when_head_sha_is_missing():
    reason = build_publish_block_reason(_review_input(), None, _review_result())

    assert reason is not None
    assert "head SHA" in reason


def test_build_publish_block_reason_blocks_when_diff_is_empty():
    reason = build_publish_block_reason(
        _review_input(included_diff=""), "abc123", _review_result()
    )

    assert reason is not None
    assert "diff content" in reason


def test_build_publish_block_reason_blocks_when_every_reviewer_category_is_incomplete():
    review_result = _review_result(
        recommendation="INCOMPLETE",
        errors=["security reviewer: Model Service request failed."],
        incomplete_categories=sorted(SCORED_CATEGORIES),
    )

    reason = build_publish_block_reason(_review_input(), "abc123", review_result)

    assert reason is not None
    assert "every review category failed" in reason


def test_build_publish_block_reason_allows_a_genuinely_clean_pr():
    reason = build_publish_block_reason(_review_input(), "abc123", _review_result())

    assert reason is None


def test_build_publish_block_reason_allows_a_partial_failure():
    """Only some reviewer categories are incomplete - a real (if incomplete) result still publishes."""
    review_result = _review_result(
        recommendation="INCOMPLETE",
        errors=["security reviewer: Model Service request failed."],
        incomplete_categories=["security", "architecture"],
    )

    reason = build_publish_block_reason(_review_input(), "abc123", review_result)

    assert reason is None


def test_build_publish_block_reason_allows_real_findings_even_when_every_category_is_incomplete():
    """Even a total reviewer wipeout still publishes once there is a real finding to show for it."""
    review_result = _review_result(
        recommendation="INCOMPLETE",
        warnings=[_finding()],
        errors=["quality reviewer: model output is not valid JSON: bad"],
        incomplete_categories=sorted(SCORED_CATEGORIES),
    )

    reason = build_publish_block_reason(_review_input(), "abc123", review_result)

    assert reason is None


# --- run_review_graph wiring: fake graph, recording repository -------------


def _launcher_settings() -> Settings:
    return Settings(
        _env_file=None,
        github_app_id=1,
        github_app_private_key_path=Path("secrets/test-private-key.pem"),
        github_webhook_secret="fake-webhook-secret",
        database_url="postgresql://fake:fake@localhost:5432/fake",
    )


class _FakeTokenProvider:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def get_token(self, installation_id: int) -> str:
        return _FAKE_SECRET


class _FakeGraph:
    def __init__(self, final_state: dict) -> None:
        self._final_state = final_state

    async def ainvoke(self, initial_state: ReviewState) -> dict:
        return self._final_state

    async def astream(self, initial_state: ReviewState, stream_mode: str = "values"):
        yield self._final_state


class _RecordingReviewRunRepository:
    """Stands in for ReviewRunRepository, recording every complete()/fail() call."""

    instances: list["_RecordingReviewRunRepository"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.completed: list[tuple] = []
        self.failed: list[tuple] = []
        _RecordingReviewRunRepository.instances.append(self)

    async def get_last_publish(self, repository, pr_number):
        return None

    async def complete(self, review_run_id, completion) -> None:
        self.completed.append((review_run_id, completion))

    async def fail(self, review_run_id, error_summary) -> None:
        self.failed.append((review_run_id, error_summary))


def _patch_launcher(monkeypatch, final_state: dict):
    """Wire fake token/graph/repository into build_review_launcher().

    Returns (launcher, publish_calls, repo) - the SAME launcher instance the
    test must call _review_worker on, since repo is only reachable through
    the ReviewRunRepository instance this exact build_review_launcher() call
    closed over. Building a second launcher would close over a different,
    unobserved repository instance instead.
    """

    publish_calls = []

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        publish_calls.append((repo, pr_number, head_sha, result, token))
        return 1

    _RecordingReviewRunRepository.instances = []
    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _RecordingReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module,
        "build_graph",
        lambda token, settings, **kwargs: _FakeGraph(final_state),
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)

    launcher = build_review_launcher(_launcher_settings())
    repo = _RecordingReviewRunRepository.instances[-1]
    return launcher, publish_calls, repo


async def test_a_successful_run_is_marked_completed(monkeypatch):
    review_result = _review_result(warnings=[_finding()], recommendation="APPROVE")
    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": "abc123",
        "review_input": _review_input(),
        "review_result": review_result,
    }
    launcher, publish_calls, repo = _patch_launcher(monkeypatch, final_state)

    await launcher._review_worker(_review_run())

    assert len(publish_calls) == 1
    assert len(repo.completed) == 1
    assert repo.failed == []
    _, completion = repo.completed[0]
    assert completion.recommendation == "APPROVE"
    assert len(completion.findings) == 1


async def test_all_reviewers_failed_blocks_publish_and_marks_failed(monkeypatch):
    review_result = _review_result(
        recommendation="INCOMPLETE",
        errors=[
            "security reviewer: Model Service request failed.",
            "architecture reviewer: Model Service request failed.",
        ],
        incomplete_categories=sorted(SCORED_CATEGORIES),
    )
    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": "abc123",
        "review_input": _review_input(),
        "review_result": review_result,
    }
    launcher, publish_calls, repo = _patch_launcher(monkeypatch, final_state)

    await launcher._review_worker(_review_run())

    assert publish_calls == []
    assert repo.completed == []
    assert len(repo.failed) == 1
    assert "every review category failed" in repo.failed[0][1]


async def test_a_partial_failure_still_publishes_as_a_comment_with_real_findings(monkeypatch):
    """Some, but not all, categories are incomplete - this is a real (if partial) result."""
    review_result = _review_result(
        recommendation="INCOMPLETE",
        architecture_score=None,
        warnings=[_finding()],
        errors=["architecture reviewer: Model Service request failed."],
        incomplete_categories=["architecture"],
    )
    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": "abc123",
        "review_input": _review_input(),
        "review_result": review_result,
    }
    launcher, publish_calls, repo = _patch_launcher(monkeypatch, final_state)

    await launcher._review_worker(_review_run())

    assert len(publish_calls) == 1
    _, _, _, published_result, _ = publish_calls[0]
    assert published_result.recommendation == "INCOMPLETE"
    assert published_result.incomplete_categories == ["architecture"]
    assert len(published_result.warnings) == 1
    assert repo.failed == []
    assert len(repo.completed) == 1
    _, completion = repo.completed[0]
    assert completion.recommendation == "INCOMPLETE"


async def test_a_genuinely_clean_pr_still_publishes(monkeypatch):
    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": "abc123",
        "review_input": _review_input(),
        "review_result": _review_result(),
    }
    launcher, publish_calls, repo = _patch_launcher(monkeypatch, final_state)

    await launcher._review_worker(_review_run())

    assert len(publish_calls) == 1
    assert len(repo.completed) == 1
    assert repo.failed == []


async def test_missing_head_sha_blocks_publish_explicitly(monkeypatch):
    """load_pr never fetched a head SHA - caught by an explicit check, not GitHub's own rejection."""

    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": None,
        "review_input": _review_input(included_diff="", head_sha=""),
        "review_result": _review_result(),
    }
    launcher, publish_calls, repo = _patch_launcher(monkeypatch, final_state)

    await launcher._review_worker(_review_run())

    assert publish_calls == []
    assert repo.completed == []
    assert len(repo.failed) == 1
    assert "head SHA" in repo.failed[0][1]


# --- no-secret-leak, exercising the real graph and real error-building code -


def _patch_async_client(monkeypatch, transport: httpx.MockTransport) -> None:
    class MockedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockedAsyncClient)


async def test_no_secret_leaks_when_load_pr_fails_end_to_end(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    _RecordingReviewRunRepository.instances = []
    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _RecordingReviewRunRepository)

    launcher = build_review_launcher(_launcher_settings())
    repo = _RecordingReviewRunRepository.instances[-1]

    await launcher._review_worker(_review_run())

    assert len(repo.failed) == 1
    stored_message = repo.failed[0][1]
    assert _FAKE_SECRET not in stored_message

    # Also inspect load_pr's own raw error text directly (not just the
    # guard's static summary that ended up stored) - this is the actual
    # message-building code load_pr.py/github_tool.py exercised above.
    model_settings = Settings(
        github_app_id=1,
        github_app_private_key_path="secrets/unused.pem",
        github_webhook_secret="unused",
    )
    graph = build_graph(_FAKE_SECRET, model_settings)
    final_state = await graph.ainvoke(ReviewState(repo="example-org/example-repo", pr_number=1))
    for error in final_state["errors"]:
        assert _FAKE_SECRET not in error


async def test_no_secret_leaks_when_all_reviewers_fail_end_to_end(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.github.com" in url:
            if url.endswith("/files") or "/files?" in url:
                return httpx.Response(200, json=[{"filename": "a.py"}])
            accept = request.headers.get("accept", "")
            if "diff" in accept:
                return httpx.Response(
                    200,
                    text=(
                        "diff --git a/a.py b/a.py\n"
                        "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,3 @@\n+print('hi')\n"
                    ),
                )
            return httpx.Response(
                200, json={"head": {"sha": "abc123"}, "title": "t", "body": None}
            )
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    _RecordingReviewRunRepository.instances = []
    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _RecordingReviewRunRepository)

    launcher = build_review_launcher(_launcher_settings())
    repo = _RecordingReviewRunRepository.instances[-1]

    await launcher._review_worker(_review_run())

    assert repo.completed == []
    assert len(repo.failed) == 1
    stored_message = repo.failed[0][1]
    assert _FAKE_SECRET not in stored_message
    assert "every review category failed" in stored_message

    # Also inspect each reviewer's own raw error text directly - the actual
    # message-building code in review_security.py/review_architecture.py/
    # review_quality.py exercised above, not just the guard's own summary.
    model_settings = Settings(
        github_app_id=1,
        github_app_private_key_path="secrets/unused.pem",
        github_webhook_secret="unused",
    )
    graph = build_graph(_FAKE_SECRET, model_settings)
    final_state = await graph.ainvoke(ReviewState(repo="example-org/example-repo", pr_number=1))
    assert len(final_state["errors"]) >= 3
    for error in final_state["errors"]:
        assert _FAKE_SECRET not in error
