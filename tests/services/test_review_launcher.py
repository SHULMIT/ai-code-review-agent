"""Tests for atomic review launch scheduling."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks
import httpx
import pytest

from review_agent.config import Settings
from review_agent.graph.state import ReviewState
from review_agent.models.review_input import ReviewInput
from review_agent.models.review_result import ReviewResult
from review_agent.services import review_launcher as review_launcher_module
from review_agent.services.event_router import ReviewRequest, ReviewTrigger
from review_agent.services.review_launcher import (
    ReviewLauncher,
    build_review_identity,
    build_review_launcher,
)
from review_agent.storage.review_runs import PreviousPublish, ReviewRun, ReviewRunStatus


def _review_request(head_sha: str | None = "abc") -> ReviewRequest:
    """Build one safe routed event for launcher tests."""

    return ReviewRequest(
        repository="example-org/example-repo",
        pr_number=42,
        head_sha=head_sha,
        installation_id=123,
        trigger=ReviewTrigger.PULL_REQUEST_OPENED,
    )


def _review_run(
    status: ReviewRunStatus = ReviewRunStatus.PENDING, check_run_id: int | None = None
) -> ReviewRun:
    """Build one durable run returned by the fake storage boundary."""

    return ReviewRun(
        id=uuid4(),
        github_delivery_id="delivery-1",
        review_identity="identity",
        repository="example-org/example-repo",
        pr_number=42,
        installation_id=123,
        trigger=ReviewTrigger.PULL_REQUEST_OPENED,
        head_sha="abc",
        check_run_id=check_run_id,
        status=status,
        created_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


class FakeReviewRunRepository:
    """Record launcher calls while controlling each storage result."""

    def __init__(self, created_run: ReviewRun | None) -> None:
        self.created_run = created_run
        self.created = []
        self.failed = []

    async def create_pending(self, review_run):
        self.created.append(review_run)
        return self.created_run

    async def mark_running(self, review_run_id):
        if self.created_run is None:
            return None
        return _review_run(ReviewRunStatus.RUNNING)

    async def fail(self, review_run_id, error_summary):
        self.failed.append((review_run_id, error_summary))
        return _review_run(ReviewRunStatus.FAILED)


async def test_launch_schedules_background_work_only_after_atomic_insert() -> None:
    """A successful insert creates one task and exposes its durable run ID."""

    repository = FakeReviewRunRepository(_review_run())

    async def worker(review_run: ReviewRun) -> None:
        return None

    background_tasks = BackgroundTasks()
    launcher = ReviewLauncher(repository, worker)  # type: ignore[arg-type]

    result = await launcher.launch(_review_request(), "delivery-1", background_tasks)

    assert result.accepted is True
    assert result.duplicate is False
    assert result.review_run_id is not None
    assert len(background_tasks.tasks) == 1
    assert repository.created[0].github_delivery_id == "delivery-1"


async def test_duplicate_launch_does_not_schedule_background_work() -> None:
    """A database duplicate returns immediately without invoking a review worker."""

    repository = FakeReviewRunRepository(None)

    async def worker(review_run: ReviewRun) -> None:
        return None

    background_tasks = BackgroundTasks()
    launcher = ReviewLauncher(repository, worker)  # type: ignore[arg-type]

    result = await launcher.launch(_review_request(), "delivery-1", background_tasks)

    assert result.accepted is False
    assert result.duplicate is True
    assert background_tasks.tasks == []


async def test_launch_returns_before_its_background_worker_runs() -> None:
    """The Webhook path can acknowledge acceptance before review analysis begins."""

    repository = FakeReviewRunRepository(_review_run())
    worker_started = asyncio.Event()
    allow_worker_to_finish = asyncio.Event()

    async def worker(review_run: ReviewRun) -> None:
        worker_started.set()
        await allow_worker_to_finish.wait()

    background_tasks = BackgroundTasks()
    launcher = ReviewLauncher(repository, worker)  # type: ignore[arg-type]

    result = await launcher.launch(_review_request(), "delivery-1", background_tasks)

    assert result.accepted is True
    assert worker_started.is_set() is False
    executing_tasks = asyncio.create_task(background_tasks())
    await worker_started.wait()
    allow_worker_to_finish.set()
    await executing_tasks


async def test_worker_failure_is_saved_without_its_raw_exception_message() -> None:
    """Background failure storage keeps an exception type but not untrusted details."""

    repository = FakeReviewRunRepository(_review_run())

    async def failing_worker(review_run: ReviewRun) -> None:
        raise RuntimeError("GITHUB_TOKEN=must-not-be-saved")

    launcher = ReviewLauncher(repository, failing_worker)  # type: ignore[arg-type]

    await launcher._run_review(_review_run())

    assert repository.failed
    assert repository.failed[0][1] == "Review worker failed: RuntimeError"


def _review_input() -> ReviewInput:
    """A minimal ReviewInput with real diff content, so build_publish_block_reason() allows it."""

    return ReviewInput(
        repo="example-org/example-repo",
        pr_number=42,
        head_sha="def456",
        included_diff="diff --git a/a.py b/a.py\n@@ -1,2 +1,3 @@\n+print('hi')\n",
        diff_truncated=False,
        included_files=["a.py"],
        omitted_files=[],
        repository_context={"results": []},
        documentation_context=[],
        coverage_notes=[],
    )


def test_build_review_run_completion_preserves_result_diagnostics():
    review_result = ReviewResult(
        pr_summary="Reviewed 1 changed files across 1 components; found 0 blocking, 0 warning, 0 suggestion findings.",
        architecture_score=None,
        code_quality_score=100,
        production_readiness_score=100,
        blocking_findings=[],
        warnings=[],
        suggestions=[],
        recommendation="INCOMPLETE",
        reviewed_head_sha="abc123",
        errors=["architecture reviewer: Model Service request failed."],
        incomplete_categories=["architecture"],
    )

    completion = review_launcher_module._build_review_run_completion(review_result)

    assert completion.errors == review_result.errors
    assert completion.incomplete_categories == review_result.incomplete_categories


def _launcher_settings() -> Settings:
    """Build valid Settings with a database_url, so build_review_launcher() succeeds."""

    return Settings(
        _env_file=None,
        github_app_id=1,
        github_app_private_key_path=Path("secrets/test-private-key.pem"),
        github_webhook_secret="fake-webhook-secret",
        database_url="postgresql://fake:fake@localhost:5432/fake",
    )


class _FakeTokenProvider:
    """Stands in for InstallationTokenProvider without touching a real private key file."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def get_token(self, installation_id: int) -> str:
        return "fake-installation-token"


class _FakeGraph:
    """Stands in for the compiled LangGraph, returning a fixed final state."""

    def __init__(self, final_state: dict) -> None:
        self._final_state = final_state

    async def ainvoke(self, initial_state: ReviewState) -> dict:
        return self._final_state

    async def astream(self, initial_state: ReviewState, stream_mode: str = "values"):
        yield self._final_state


class _FakeReviewRunRepository:
    """Stands in for ReviewRunRepository so a successful run never opens a real DB connection."""

    def __init__(
        self, *args, previous_publish: PreviousPublish | None = None, **kwargs
    ) -> None:
        self._previous_publish = previous_publish
        self.completions: list = []

    async def get_last_publish(self, repository: str, pr_number: int):
        return self._previous_publish

    async def complete(self, review_run_id, completion) -> None:
        self.completions.append(completion)
        return None

    async def fail(self, review_run_id, error_summary) -> None:
        return None


async def test_run_review_graph_publishes_the_review_result_with_the_correct_event(
    monkeypatch,
) -> None:
    """After a successful graph run, publish_review is called with review_result's recommendation."""

    review_result = ReviewResult(
        pr_summary="Reviewed 1 changed files across 1 components; found 1 blocking, 0 warning, 0 suggestion findings.",
        architecture_score=70,
        code_quality_score=100,
        production_readiness_score=100,
        blocking_findings=[],
        warnings=[],
        suggestions=[],
        recommendation="REQUEST_CHANGES",
        reviewed_head_sha="def456",
        errors=[],
    )
    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 42,
        "head_sha": "def456",
        "review_input": _review_input(),
        "review_result": review_result,
    }

    published_calls = []

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        published_calls.append((repo, pr_number, head_sha, result, token, existing_review_id))
        return 777

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module, "build_graph", lambda token, settings, **kwargs: _FakeGraph(final_state)
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run())

    assert len(published_calls) == 1
    repo, pr_number, head_sha, result, token, existing_review_id = published_calls[0]
    assert repo == "example-org/example-repo"
    assert pr_number == 42
    assert head_sha == "def456"
    assert result.recommendation == "REQUEST_CHANGES"
    assert token == "fake-installation-token"
    assert existing_review_id is None


def _patch_graph_and_publish(monkeypatch, final_state: dict, published_id: int = 777) -> None:
    """Wire a fixed final_state and a no-op publish_review, matching the other tests here."""

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        return published_id

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module, "build_graph", lambda token, settings, **kwargs: _FakeGraph(final_state)
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)


@pytest.mark.parametrize(
    ("recommendation", "expected_conclusion"),
    [
        ("APPROVE", "success"),
        ("REQUEST_CHANGES", "action_required"),
        ("INCOMPLETE", "neutral"),
    ],
)
async def test_run_review_graph_completes_a_requested_check_run_with_the_mapped_conclusion(
    monkeypatch, recommendation: str, expected_conclusion: str
) -> None:
    """A Check Run-triggered run's own Check Run is completed with the agreed conclusion mapping."""

    review_result = ReviewResult(
        pr_summary="Reviewed 1 changed files across 1 components; found 0 blocking, 0 warning, 0 suggestion findings.",
        architecture_score=100,
        code_quality_score=100,
        production_readiness_score=100,
        blocking_findings=[],
        warnings=[],
        suggestions=[],
        recommendation=recommendation,
        reviewed_head_sha="def456",
        errors=[],
    )
    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 42,
        "head_sha": "def456",
        "review_input": _review_input(),
        "review_result": review_result,
    }
    _patch_graph_and_publish(monkeypatch, final_state)

    completed_calls = []

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        completed_calls.append((repo, check_run_id, conclusion))

    monkeypatch.setattr(review_launcher_module, "complete_check_run", fake_complete_check_run)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run(check_run_id=987654))

    assert completed_calls == [("example-org/example-repo", 987654, expected_conclusion)]


async def test_run_review_graph_completes_the_check_run_with_failure_when_publishing_is_blocked(
    monkeypatch,
) -> None:
    """A run with nothing safe to publish still reports failure on its own Check Run."""

    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 42,
        "head_sha": None,  # load_pr never fetched a head SHA - see build_publish_block_reason
        "review_result": ReviewResult(
            pr_summary="s", architecture_score=100, code_quality_score=100,
            production_readiness_score=100, blocking_findings=[], warnings=[],
            suggestions=[], recommendation="APPROVE", reviewed_head_sha="", errors=[],
        ),
    }
    _patch_graph_and_publish(monkeypatch, final_state)

    completed_calls = []

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        completed_calls.append((repo, check_run_id, conclusion))

    monkeypatch.setattr(review_launcher_module, "complete_check_run", fake_complete_check_run)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run(check_run_id=987654))

    assert completed_calls == [("example-org/example-repo", 987654, "failure")]


async def test_run_review_graph_completes_the_check_run_with_failure_on_an_unhandled_exception(
    monkeypatch,
) -> None:
    """A genuinely unexpected exception still reports failure before re-raising for _run_review."""

    class _ExplodingGraph:
        async def astream(self, initial_state, stream_mode="values"):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator; never reached

    completed_calls = []

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        completed_calls.append((repo, check_run_id, conclusion))

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module, "build_graph", lambda token, settings, **kwargs: _ExplodingGraph()
    )
    monkeypatch.setattr(review_launcher_module, "complete_check_run", fake_complete_check_run)

    launcher = build_review_launcher(_launcher_settings())

    with pytest.raises(RuntimeError, match="boom"):
        await launcher._review_worker(_review_run(check_run_id=987654))

    assert completed_calls == [("example-org/example-repo", 987654, "failure")]


async def test_run_review_graph_never_touches_a_check_run_when_none_was_requested(
    monkeypatch,
) -> None:
    """A review launched without a Check Run (e.g. the "/review" comment) must not call the Checks API."""

    review_result = ReviewResult(
        pr_summary="s", architecture_score=100, code_quality_score=100,
        production_readiness_score=100, blocking_findings=[], warnings=[],
        suggestions=[], recommendation="APPROVE", reviewed_head_sha="def456", errors=[],
    )
    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 42,
        "head_sha": "def456",
        "review_input": _review_input(),
        "review_result": review_result,
    }
    _patch_graph_and_publish(monkeypatch, final_state)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("complete_check_run must not be called without a check_run_id")

    monkeypatch.setattr(review_launcher_module, "complete_check_run", fail_if_called)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run())  # check_run_id defaults to None


async def test_run_review_graph_shares_one_http_client_across_the_graph_and_publish(
    monkeypatch,
) -> None:
    """The same pooled httpx.AsyncClient must reach both build_graph and publish_review.

    This is the actual runtime win from RUNTIME-OPTIMIZATION-OPPORTUNITIES.md
    item 5 - one review run reuses one client instead of every HTTP call
    (GitHub, Model Service, embeddings, publish) opening its own.
    """
    captured: dict = {}

    def fake_build_graph(token, settings, **kwargs):
        captured["build_graph_client"] = kwargs.get("client")
        return _FakeGraph(_completed_final_state("APPROVE"))

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        captured["publish_review_client"] = kwargs.get("client")
        return 777

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(review_launcher_module, "build_graph", fake_build_graph)
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run())

    assert isinstance(captured["build_graph_client"], httpx.AsyncClient)
    assert captured["build_graph_client"] is captured["publish_review_client"]


async def test_run_review_graph_does_not_publish_when_review_result_is_missing(
    monkeypatch,
) -> None:
    """A None review_result (should not happen given finalize_review, but is handled) skips publishing."""

    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 42,
        "head_sha": "def456",
        "review_result": None,
    }

    published_calls = []

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        published_calls.append((repo, pr_number, head_sha, result, token, existing_review_id))
        return 777

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module, "build_graph", lambda token, settings, **kwargs: _FakeGraph(final_state)
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run())

    assert published_calls == []


def _completed_final_state(recommendation: str) -> dict:
    """A minimal final graph state that clears build_publish_block_reason for publishing."""

    review_result = ReviewResult(
        pr_summary="Reviewed 1 changed files across 1 components; found 0 blocking, 0 warning, 0 suggestion findings.",
        architecture_score=100,
        code_quality_score=100,
        production_readiness_score=100,
        blocking_findings=[],
        warnings=[],
        suggestions=[],
        recommendation=recommendation,
        reviewed_head_sha="def456",
        errors=[],
    )
    return {
        "repo": "example-org/example-repo",
        "pr_number": 42,
        "head_sha": "def456",
        "review_input": _review_input(),
        "review_result": review_result,
    }


async def test_run_review_graph_reuses_the_existing_review_id_when_recommendation_is_unchanged(
    monkeypatch,
) -> None:
    """A re-run whose recommendation matches the last published one updates that review in place."""

    fake_repository = _FakeReviewRunRepository(
        previous_publish=PreviousPublish(review_id=555, recommendation="APPROVE")
    )
    published_calls = []

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        published_calls.append(existing_review_id)
        return existing_review_id or 999

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", lambda *a, **k: fake_repository)
    monkeypatch.setattr(
        review_launcher_module,
        "build_graph",
        lambda token, settings, **kwargs: _FakeGraph(_completed_final_state("APPROVE")),
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run())

    assert published_calls == [555]
    assert fake_repository.completions[0].last_review_id == 555


async def test_run_review_graph_does_not_reuse_the_existing_review_id_when_recommendation_changed(
    monkeypatch,
) -> None:
    """A changed recommendation always creates a fresh review - GitHub cannot update a review's event."""

    fake_repository = _FakeReviewRunRepository(
        previous_publish=PreviousPublish(review_id=555, recommendation="APPROVE")
    )
    published_calls = []

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        published_calls.append(existing_review_id)
        return existing_review_id or 999

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", lambda *a, **k: fake_repository)
    monkeypatch.setattr(
        review_launcher_module,
        "build_graph",
        lambda token, settings, **kwargs: _FakeGraph(_completed_final_state("REQUEST_CHANGES")),
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run())

    assert published_calls == [None]
    assert fake_repository.completions[0].last_review_id == 999


async def test_run_review_graph_publishes_with_no_existing_id_on_the_first_run(
    monkeypatch,
) -> None:
    """Nothing published for this PR yet - always creates a new review."""

    fake_repository = _FakeReviewRunRepository(previous_publish=None)
    published_calls = []

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        published_calls.append(existing_review_id)
        return 111

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", lambda *a, **k: fake_repository)
    monkeypatch.setattr(
        review_launcher_module,
        "build_graph",
        lambda token, settings, **kwargs: _FakeGraph(_completed_final_state("APPROVE")),
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run())

    assert published_calls == [None]
    assert fake_repository.completions[0].last_review_id == 111


def test_reusable_review_id_returns_none_with_no_prior_publish() -> None:
    review_result = ReviewResult(
        pr_summary="s", architecture_score=100, code_quality_score=100,
        production_readiness_score=100, blocking_findings=[], warnings=[],
        suggestions=[], recommendation="APPROVE", reviewed_head_sha="abc", errors=[],
    )
    assert review_launcher_module._reusable_review_id(None, review_result) is None


def test_reusable_review_id_returns_the_id_when_recommendation_matches() -> None:
    review_result = ReviewResult(
        pr_summary="s", architecture_score=100, code_quality_score=100,
        production_readiness_score=100, blocking_findings=[], warnings=[],
        suggestions=[], recommendation="APPROVE", reviewed_head_sha="abc", errors=[],
    )
    previous_publish = PreviousPublish(review_id=555, recommendation="APPROVE")
    assert review_launcher_module._reusable_review_id(previous_publish, review_result) == 555


def test_reusable_review_id_returns_none_when_recommendation_differs() -> None:
    review_result = ReviewResult(
        pr_summary="s", architecture_score=100, code_quality_score=100,
        production_readiness_score=100, blocking_findings=[], warnings=[],
        suggestions=[], recommendation="REQUEST_CHANGES", reviewed_head_sha="abc", errors=[],
    )
    previous_publish = PreviousPublish(review_id=555, recommendation="APPROVE")
    assert review_launcher_module._reusable_review_id(previous_publish, review_result) is None


class _StreamingFakeGraph:
    """Stands in for the compiled LangGraph, yielding a sequence of state snapshots -
    simulating reviewer_diagnostics arriving one at a time as astream(stream_mode="values")
    would really deliver them.
    """

    def __init__(self, state_updates: list[dict]) -> None:
        self._state_updates = state_updates

    async def astream(self, initial_state: ReviewState, stream_mode: str = "values"):
        for state_update in self._state_updates:
            yield state_update


def _diagnostic(category: str, status: str = "completed"):
    from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic

    return ReviewerDiagnostic(category=category, status=status)


async def test_run_review_graph_pushes_a_live_check_run_update_per_completed_reviewer(
    monkeypatch,
) -> None:
    """Each newly-completed reviewer category triggers exactly one progress update -
    never one per intermediate LangGraph super-step with no new terminal status."""

    review_result = _completed_final_state("APPROVE")["review_result"]
    state_updates = [
        {"reviewer_diagnostics": []},
        {"reviewer_diagnostics": [_diagnostic("security")]},
        {"reviewer_diagnostics": [_diagnostic("security")]},  # no new terminal category - must not repeat
        {"reviewer_diagnostics": [_diagnostic("security"), _diagnostic("architecture", "failed")]},
        {**_completed_final_state("APPROVE"), "reviewer_diagnostics": [
            _diagnostic("security"), _diagnostic("architecture", "failed")
        ]},
    ]

    progress_calls = []

    async def fake_update_check_run_progress(repo, check_run_id, token, title, summary, client=None):
        progress_calls.append((repo, check_run_id, title, summary))

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        return 777

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module, "build_graph", lambda token, settings, **kwargs: _StreamingFakeGraph(state_updates)
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)
    monkeypatch.setattr(review_launcher_module, "update_check_run_progress", fake_update_check_run_progress)
    monkeypatch.setattr(review_launcher_module, "complete_check_run", lambda *a, **k: _noop())

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run(check_run_id=987654))

    # Exactly two meaningful transitions happened: 0->1 completed categories, then 1->2.
    assert len(progress_calls) == 2
    assert "1/6" in progress_calls[0][2]
    assert "2/6" in progress_calls[1][2]


async def _noop():
    return None


async def test_run_review_graph_never_pushes_progress_when_no_check_run_was_requested(
    monkeypatch,
) -> None:
    state_updates = [
        {"reviewer_diagnostics": [_diagnostic("security")]},
        _completed_final_state("APPROVE"),
    ]

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("update_check_run_progress must not be called without a check_run_id")

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        return 777

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module, "build_graph", lambda token, settings, **kwargs: _StreamingFakeGraph(state_updates)
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)
    monkeypatch.setattr(review_launcher_module, "update_check_run_progress", fail_if_called)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run())  # check_run_id defaults to None


async def test_run_review_graph_stamps_a_duration_before_publishing(monkeypatch) -> None:
    """The review_result passed to publish_review must already carry a wall-clock duration."""

    captured: dict = {}

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        captured["duration_seconds"] = result.duration_seconds
        return 777

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module,
        "build_graph",
        lambda token, settings, **kwargs: _FakeGraph(_completed_final_state("APPROVE")),
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run())

    assert captured["duration_seconds"] is not None
    assert captured["duration_seconds"] >= 0


async def test_run_review_graph_reports_a_safe_specific_message_when_publishing_fails(
    monkeypatch, caplog,
) -> None:
    """A publish failure must not be reported as a successful completed review."""
    from review_agent.tools.github_publisher import GitHubPublisherError

    async def failing_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        raise GitHubPublisherError("GitHub returned 500 for POST /reviews")

    completed_calls = []

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        completed_calls.append((repo, check_run_id, conclusion, title, summary))

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module,
        "build_graph",
        lambda token, settings, **kwargs: _FakeGraph(_completed_final_state("APPROVE")),
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", failing_publish_review)
    monkeypatch.setattr(review_launcher_module, "complete_check_run", fake_complete_check_run)

    launcher = build_review_launcher(_launcher_settings())
    with caplog.at_level("ERROR", logger="review_agent"):
        await launcher._review_worker(_review_run(check_run_id=987654))

    assert len(completed_calls) == 1
    repo, check_run_id, conclusion, title, summary = completed_calls[0]
    assert conclusion == "failure"
    assert "Retry now" in summary
    assert "GitHub returned 500" not in summary
    assert "publishing the PR Review to GitHub failed" in summary
    assert any(
        getattr(record, "reason_code", None) == "review_publish_failed" for record in caplog.records
    )


async def test_run_review_graph_publishes_before_completing_the_check_run(monkeypatch) -> None:
    """The full PR Review must be confirmed published before the Check Run's spinner stops."""

    call_order: list[str] = []

    async def fake_publish_review(repo, pr_number, head_sha, result, token, existing_review_id=None, **kwargs):
        call_order.append("publish_review")
        return 777

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        call_order.append("complete_check_run")

    monkeypatch.setattr(review_launcher_module, "InstallationTokenProvider", _FakeTokenProvider)
    monkeypatch.setattr(review_launcher_module, "ReviewRunRepository", _FakeReviewRunRepository)
    monkeypatch.setattr(
        review_launcher_module,
        "build_graph",
        lambda token, settings, **kwargs: _FakeGraph(_completed_final_state("APPROVE")),
    )
    monkeypatch.setattr(review_launcher_module, "publish_review", fake_publish_review)
    monkeypatch.setattr(review_launcher_module, "complete_check_run", fake_complete_check_run)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run(check_run_id=987654))

    assert call_order == ["publish_review", "complete_check_run"]


async def test_run_review_graph_completes_the_check_run_with_a_rich_partial_summary(
    monkeypatch,
) -> None:
    """A partial review's final Check Run text must name the failed categories and say Retry later."""

    review_result = ReviewResult(
        pr_summary="s", architecture_score=None, code_quality_score=100,
        production_readiness_score=100, blocking_findings=[], warnings=[],
        suggestions=[], recommendation="INCOMPLETE", reviewed_head_sha="def456", errors=[],
        incomplete_categories=["architecture"],
    )
    final_state = {
        "repo": "example-org/example-repo",
        "pr_number": 42,
        "head_sha": "def456",
        "review_input": _review_input(),
        "review_result": review_result,
    }
    _patch_graph_and_publish(monkeypatch, final_state)

    completed_calls = []

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        completed_calls.append((conclusion, title, summary))

    monkeypatch.setattr(review_launcher_module, "complete_check_run", fake_complete_check_run)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run(check_run_id=987654))

    assert len(completed_calls) == 1
    conclusion, title, summary = completed_calls[0]
    assert conclusion == "neutral"
    assert title == "Code Review partially completed"
    assert "Review status: Partially completed" in summary
    assert "architecture" in summary
    assert "Retry guidance: Retry later" in summary


async def test_run_review_graph_includes_a_direct_link_to_the_published_review(monkeypatch) -> None:
    """GitHub does not reliably live-refresh the Conversation timeline, so the final
    Check Run must carry a direct link built from the review ID GitHub actually
    returned - not a guess, and not omitted just because it could be built later."""

    _patch_graph_and_publish(monkeypatch, _completed_final_state("APPROVE"), published_id=98765)

    completed_calls = []

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        completed_calls.append(summary)

    monkeypatch.setattr(review_launcher_module, "complete_check_run", fake_complete_check_run)

    launcher = build_review_launcher(_launcher_settings())
    await launcher._review_worker(_review_run(check_run_id=987654))

    assert len(completed_calls) == 1
    summary = completed_calls[0]
    assert "✅ Code Review published" in summary
    assert (
        "[View full review]"
        "(https://github.com/example-org/example-repo/pull/42#pullrequestreview-98765)"
    ) in summary


def test_review_identity_uses_head_sha_or_exact_delivery_id() -> None:
    """The duplicate key never invents a head SHA for issue-comment payloads."""

    assert build_review_identity(_review_request(), "delivery-1") == (
        "example-org/example-repo:42:abc:pull_request.opened"
    )
    assert build_review_identity(_review_request(head_sha=None), "delivery-2") == (
        "delivery:delivery-2"
    )
