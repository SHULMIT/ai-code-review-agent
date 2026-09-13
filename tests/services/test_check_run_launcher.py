"""Tests for the on-demand Check Run trigger's orchestration - creation, staleness, and dedup."""

from pathlib import Path

from fastapi import BackgroundTasks
import pytest

from review_agent.config import Settings

from review_agent.services import check_run_launcher as check_run_launcher_module
from review_agent.services.check_run_launcher import (
    CheckRunLauncher,
    CheckRunLauncherError,
    build_check_run_launcher,
)
from review_agent.services.event_router import (
    CheckRunActionRequest,
    CheckRunCreationRequest,
    ReviewTrigger,
)
from review_agent.services.review_launcher import ReviewLaunchResult
from review_agent.tools.check_run_tool import CheckRunToolError
from review_agent.tools.github_auth import GitHubAuthError
from review_agent.tools.github_tool import GitHubToolError


class _FakeTokenProvider:
    def __init__(self, token: str = "fake-installation-token") -> None:
        self.token = token
        self.installation_ids: list[int] = []

    async def get_token(self, installation_id: int) -> str:
        self.installation_ids.append(installation_id)
        return self.token


class _FakeReviewLauncher:
    """Stands in for ReviewLauncher, recording the exact ReviewRequest it was launched with."""

    def __init__(self, launch_result: ReviewLaunchResult) -> None:
        self.launch_result = launch_result
        self.launch_calls: list[tuple] = []

    async def launch(self, review_request, github_delivery_id, background_tasks, check_run_id=None):
        self.launch_calls.append((review_request, github_delivery_id, check_run_id))
        return self.launch_result


def _creation_request(**overrides) -> CheckRunCreationRequest:
    values = {
        "repository": "example-org/example-repo",
        "pr_number": 42,
        "head_sha": "abc123",
        "installation_id": 123456,
    }
    values.update(overrides)
    return CheckRunCreationRequest(**values)


def _action_request(**overrides) -> CheckRunActionRequest:
    values = {
        "repository": "example-org/example-repo",
        "pr_number": 42,
        "check_run_id": 987654,
        "check_run_head_sha": "abc123",
        "installation_id": 123456,
    }
    values.update(overrides)
    return CheckRunActionRequest(**values)


async def test_handle_pull_request_event_creates_or_resets_the_check_run(monkeypatch):
    captured: dict = {}

    async def fake_create_or_reset_check_run(repo, head_sha, token):
        captured["args"] = (repo, head_sha, token)
        return 111

    monkeypatch.setattr(
        check_run_launcher_module, "create_or_reset_check_run", fake_create_or_reset_check_run
    )

    launcher = CheckRunLauncher(_FakeTokenProvider(), _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    ))

    await launcher.handle_pull_request_event(_creation_request())

    assert captured["args"] == ("example-org/example-repo", "abc123", "fake-installation-token")


async def test_handle_pull_request_event_wraps_a_check_run_tool_failure(monkeypatch):
    async def failing_create_or_reset_check_run(repo, head_sha, token):
        raise CheckRunToolError("GitHub API returned 500")

    monkeypatch.setattr(
        check_run_launcher_module, "create_or_reset_check_run", failing_create_or_reset_check_run
    )

    launcher = CheckRunLauncher(_FakeTokenProvider(), _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    ))

    with pytest.raises(CheckRunLauncherError):
        await launcher.handle_pull_request_event(_creation_request())


async def test_handle_pull_request_event_wraps_a_github_auth_failure(monkeypatch):
    class _FailingTokenProvider:
        async def get_token(self, installation_id: int) -> str:
            raise GitHubAuthError("Unable to create a GitHub App JWT.")

    launcher = CheckRunLauncher(_FailingTokenProvider(), _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    ))

    with pytest.raises(CheckRunLauncherError):
        await launcher.handle_pull_request_event(_creation_request())


async def test_handle_requested_action_launches_the_review_and_marks_in_progress(monkeypatch):
    """A fresh head SHA match launches the existing pipeline and flips the check to in_progress."""
    in_progress_calls: list[tuple] = []

    async def fake_get_pr_head_sha(repo, pr_number, token):
        return "abc123"

    async def fake_mark_check_run_in_progress(repo, check_run_id, token, *, title=None, summary=None):
        in_progress_calls.append((repo, check_run_id, token, title, summary))

    monkeypatch.setattr(check_run_launcher_module, "get_pr_head_sha", fake_get_pr_head_sha)
    monkeypatch.setattr(
        check_run_launcher_module, "mark_check_run_in_progress", fake_mark_check_run_in_progress
    )

    review_launcher = _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, review_run_id="run-1", status="accepted")
    )
    launcher = CheckRunLauncher(_FakeTokenProvider(), review_launcher)
    background_tasks = BackgroundTasks()

    result = await launcher.handle_requested_action(
        _action_request(), "delivery-1", background_tasks
    )

    assert result.accepted is True
    assert len(review_launcher.launch_calls) == 1
    review_request, github_delivery_id, check_run_id = review_launcher.launch_calls[0]
    assert review_request.repository == "example-org/example-repo"
    assert review_request.pr_number == 42
    assert review_request.head_sha == "abc123"
    assert review_request.trigger == ReviewTrigger.CHECK_RUN_REQUESTED
    assert github_delivery_id == "delivery-1"
    assert check_run_id == 987654
    assert len(in_progress_calls) == 1
    repo, called_check_run_id, token, title, summary = in_progress_calls[0]
    assert (repo, called_check_run_id, token) == ("example-org/example-repo", 987654, "fake-installation-token")
    assert title == "Code review is running"
    assert "0/6 reviewers completed" in summary


async def test_handle_requested_action_recovers_a_stale_check_run_with_a_fresh_one(monkeypatch):
    """A PR that moved to a new commit must never be reviewed under its old Check Run -
    but must still end up with a fresh, clickable Check Run for its real current commit."""
    complete_calls: list[tuple] = []
    create_or_reset_calls: list[tuple] = []

    async def fake_get_pr_head_sha(repo, pr_number, token):
        return "def456"  # the PR has since moved past the Check Run's own SHA

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        complete_calls.append((repo, check_run_id, conclusion, kwargs.get("allow_rerun", True)))

    async def fake_create_or_reset_check_run(repo, head_sha, token):
        create_or_reset_calls.append((repo, head_sha))
        return 555111

    monkeypatch.setattr(check_run_launcher_module, "get_pr_head_sha", fake_get_pr_head_sha)
    monkeypatch.setattr(check_run_launcher_module, "complete_check_run", fake_complete_check_run)
    monkeypatch.setattr(
        check_run_launcher_module, "create_or_reset_check_run", fake_create_or_reset_check_run
    )

    review_launcher = _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    )
    launcher = CheckRunLauncher(_FakeTokenProvider(), review_launcher)

    result = await launcher.handle_requested_action(
        _action_request(check_run_head_sha="abc123"), "delivery-1", BackgroundTasks()
    )

    assert result.accepted is False
    assert result.status == "stale"
    assert review_launcher.launch_calls == []
    # The stale Check Run is completed (never left hanging)...
    assert complete_calls == [("example-org/example-repo", 987654, "neutral", False)]
    # ...and a fresh one is created/reset for the PR's real current commit.
    assert create_or_reset_calls == [("example-org/example-repo", "def456")]


async def test_handle_requested_action_logs_the_stale_check_run_reason_code(monkeypatch, caplog):
    """The structured log for a stale Check Run must use reason_code "stale_check_run"."""

    async def fake_get_pr_head_sha(repo, pr_number, token):
        return "def456"

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        return None

    async def fake_create_or_reset_check_run(repo, head_sha, token):
        return 555111

    monkeypatch.setattr(check_run_launcher_module, "get_pr_head_sha", fake_get_pr_head_sha)
    monkeypatch.setattr(check_run_launcher_module, "complete_check_run", fake_complete_check_run)
    monkeypatch.setattr(
        check_run_launcher_module, "create_or_reset_check_run", fake_create_or_reset_check_run
    )

    review_launcher = _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    )
    launcher = CheckRunLauncher(_FakeTokenProvider(), review_launcher)

    with caplog.at_level("INFO", logger="review_agent"):
        await launcher.handle_requested_action(
            _action_request(check_run_head_sha="abc123"), "delivery-1", BackgroundTasks()
        )

    assert any(getattr(record, "reason_code", None) == "stale_check_run" for record in caplog.records)


async def test_handle_requested_action_stale_recovery_does_not_duplicate_an_existing_fresh_check_run(
    monkeypatch,
):
    """A repeated stale click must reset the one existing fresh Check Run, never create another -
    create_or_reset_check_run itself is keyed by head_sha (see check_run_tool.py), so this test
    only needs to confirm the launcher calls it (not a raw create) rather than re-deriving that."""
    calls = {"count": 0}

    async def fake_get_pr_head_sha(repo, pr_number, token):
        return "def456"

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        return None

    async def fake_create_or_reset_check_run(repo, head_sha, token):
        calls["count"] += 1
        return 555111  # the same fresh Check Run every time, per check_run_tool.py's dedup-by-SHA

    monkeypatch.setattr(check_run_launcher_module, "get_pr_head_sha", fake_get_pr_head_sha)
    monkeypatch.setattr(check_run_launcher_module, "complete_check_run", fake_complete_check_run)
    monkeypatch.setattr(
        check_run_launcher_module, "create_or_reset_check_run", fake_create_or_reset_check_run
    )

    launcher = CheckRunLauncher(_FakeTokenProvider(), _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    ))

    for _ in range(3):
        await launcher.handle_requested_action(
            _action_request(check_run_head_sha="abc123"), "delivery-1", BackgroundTasks()
        )

    assert calls["count"] == 3  # each click calls create_or_reset_check_run, never a raw create


async def test_handle_requested_action_reports_a_verification_failure_as_check_run_failure(
    monkeypatch,
):
    complete_calls: list[tuple] = []

    async def failing_get_pr_head_sha(repo, pr_number, token):
        raise GitHubToolError("Network error while fetching PR #42 in example-org/example-repo")

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        complete_calls.append((repo, check_run_id, conclusion))

    monkeypatch.setattr(check_run_launcher_module, "get_pr_head_sha", failing_get_pr_head_sha)
    monkeypatch.setattr(check_run_launcher_module, "complete_check_run", fake_complete_check_run)

    review_launcher = _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    )
    launcher = CheckRunLauncher(_FakeTokenProvider(), review_launcher)

    result = await launcher.handle_requested_action(
        _action_request(), "delivery-1", BackgroundTasks()
    )

    assert result.accepted is False
    assert result.status == "blocked"
    assert review_launcher.launch_calls == []
    assert complete_calls == [("example-org/example-repo", 987654, "failure")]


async def test_handle_requested_action_completes_the_check_run_for_a_duplicate_click(
    monkeypatch,
):
    """A second click on the same PR/head SHA while a review is already running must still
    leave a clear Check Run result - never a silent no-op that looks like nothing happened."""
    in_progress_calls: list[tuple] = []
    complete_calls: list[tuple] = []

    async def fake_get_pr_head_sha(repo, pr_number, token):
        return "abc123"

    async def fake_mark_check_run_in_progress(repo, check_run_id, token):
        in_progress_calls.append((repo, check_run_id))

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        complete_calls.append((repo, check_run_id, conclusion, kwargs.get("allow_rerun", True)))

    monkeypatch.setattr(check_run_launcher_module, "get_pr_head_sha", fake_get_pr_head_sha)
    monkeypatch.setattr(
        check_run_launcher_module, "mark_check_run_in_progress", fake_mark_check_run_in_progress
    )
    monkeypatch.setattr(check_run_launcher_module, "complete_check_run", fake_complete_check_run)

    review_launcher = _FakeReviewLauncher(
        ReviewLaunchResult(accepted=False, duplicate=True, status="duplicate")
    )
    launcher = CheckRunLauncher(_FakeTokenProvider(), review_launcher)

    result = await launcher.handle_requested_action(
        _action_request(), "delivery-2", BackgroundTasks()
    )

    assert result.accepted is False
    assert result.duplicate is True
    assert in_progress_calls == []
    assert complete_calls == [("example-org/example-repo", 987654, "neutral", False)]


async def test_handle_requested_action_still_returns_accepted_when_marking_in_progress_fails(
    monkeypatch,
):
    """A Checks API hiccup while marking in_progress must not stop the already-scheduled review."""

    async def fake_get_pr_head_sha(repo, pr_number, token):
        return "abc123"

    async def failing_mark_check_run_in_progress(repo, check_run_id, token, *, title=None, summary=None):
        raise CheckRunToolError("GitHub API returned 500")

    monkeypatch.setattr(check_run_launcher_module, "get_pr_head_sha", fake_get_pr_head_sha)
    monkeypatch.setattr(
        check_run_launcher_module, "mark_check_run_in_progress", failing_mark_check_run_in_progress
    )

    review_launcher = _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, review_run_id="run-1", status="accepted")
    )
    launcher = CheckRunLauncher(_FakeTokenProvider(), review_launcher)

    result = await launcher.handle_requested_action(
        _action_request(), "delivery-1", BackgroundTasks()
    )

    assert result.accepted is True


async def test_handle_requested_action_swallows_a_failure_completing_a_stale_check_run(
    monkeypatch,
):
    """A Checks API hiccup while reporting staleness must not crash the webhook request."""

    async def fake_get_pr_head_sha(repo, pr_number, token):
        return "def456"

    async def failing_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        raise CheckRunToolError("GitHub API returned 500")

    async def fake_create_or_reset_check_run(repo, head_sha, token):
        return 555111

    monkeypatch.setattr(check_run_launcher_module, "get_pr_head_sha", fake_get_pr_head_sha)
    monkeypatch.setattr(check_run_launcher_module, "complete_check_run", failing_complete_check_run)
    monkeypatch.setattr(
        check_run_launcher_module, "create_or_reset_check_run", fake_create_or_reset_check_run
    )

    launcher = CheckRunLauncher(_FakeTokenProvider(), _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    ))

    result = await launcher.handle_requested_action(
        _action_request(check_run_head_sha="abc123"), "delivery-1", BackgroundTasks()
    )

    assert result.status == "stale"


async def test_handle_requested_action_swallows_a_failure_creating_the_fresh_check_run(
    monkeypatch,
):
    """If the fresh Check Run can't be created, the webhook request still succeeds and the
    stale Check Run's own completion (already sent) is what the user sees - never a crash."""

    async def fake_get_pr_head_sha(repo, pr_number, token):
        return "def456"

    async def fake_complete_check_run(repo, check_run_id, conclusion, title, summary, token, **kwargs):
        return None

    async def failing_create_or_reset_check_run(repo, head_sha, token):
        raise CheckRunToolError("GitHub API returned 500")

    monkeypatch.setattr(check_run_launcher_module, "get_pr_head_sha", fake_get_pr_head_sha)
    monkeypatch.setattr(check_run_launcher_module, "complete_check_run", fake_complete_check_run)
    monkeypatch.setattr(
        check_run_launcher_module, "create_or_reset_check_run", failing_create_or_reset_check_run
    )

    launcher = CheckRunLauncher(_FakeTokenProvider(), _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    ))

    result = await launcher.handle_requested_action(
        _action_request(check_run_head_sha="abc123"), "delivery-1", BackgroundTasks()
    )

    assert result.status == "stale"


def test_build_check_run_launcher_builds_a_working_launcher() -> None:
    """The production factory wires a real token provider around the given review launcher."""

    settings = Settings(
        _env_file=None,
        github_app_id=1,
        github_app_private_key_path=Path("secrets/test-private-key.pem"),
        github_webhook_secret="fake-webhook-secret",
    )
    review_launcher = _FakeReviewLauncher(
        ReviewLaunchResult(accepted=True, duplicate=False, status="accepted")
    )

    launcher = build_check_run_launcher(settings, review_launcher)

    assert isinstance(launcher, CheckRunLauncher)
