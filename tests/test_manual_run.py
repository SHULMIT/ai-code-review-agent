"""Safety tests for the opt-in publishing path in scripts/manual_run.py."""

import importlib.util
import sys
from pathlib import Path

from review_agent.models.review_input import ReviewInput
from review_agent.models.review_result import ReviewResult

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "manual_run.py"
_SPEC = importlib.util.spec_from_file_location("manual_run", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
manual_run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(manual_run)


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


def _result(**overrides) -> dict:
    values = {
        "head_sha": "abc123",
        "review_input": ReviewInput(
            repo="example-org/example-repo",
            pr_number=137,
            head_sha="abc123",
            included_diff="diff --git a/a.py b/a.py\n@@ -1 +1 @@\n+print('ok')\n",
            diff_truncated=False,
            included_files=["a.py"],
            omitted_files=[],
            repository_context={"results": []},
            documentation_context=[],
            coverage_notes=[],
        ),
        "review_result": _review_result(),
    }
    values.update(overrides)
    return values


def test_parse_args_defaults_to_a_dry_run(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["manual_run.py", "137"])

    args = manual_run._parse_args()

    assert args.pr_number == 137
    assert args.publish is False


async def test_publish_path_is_not_reached_without_the_publish_flag(monkeypatch):
    async def should_not_verify(*args, **kwargs):
        raise AssertionError("dry run must not contact GitHub for publishing")

    monkeypatch.setattr(manual_run, "get_pr_head_sha", should_not_verify)

    outcome = await manual_run._publish_if_requested(
        False, "example-org/example-repo", 137, _result(), "fake-token"
    )

    assert outcome.startswith("DRY RUN")


async def test_publish_blocks_when_the_pr_head_changes(monkeypatch):
    published = []

    async def fake_head_sha(*args, **kwargs) -> str:
        return "def456"

    async def fake_publish(*args, **kwargs) -> None:
        published.append(args)

    monkeypatch.setattr(manual_run, "get_pr_head_sha", fake_head_sha)
    monkeypatch.setattr(manual_run, "publish_review", fake_publish)

    outcome = await manual_run._publish_if_requested(
        True, "example-org/example-repo", 137, _result(), "fake-token"
    )

    assert outcome.startswith("PUBLISH BLOCKED: the PR head changed")
    assert published == []


async def test_publish_reuses_the_existing_publisher_after_head_verification(monkeypatch):
    published = []

    async def fake_head_sha(*args, **kwargs) -> str:
        return "abc123"

    async def fake_publish(*args) -> None:
        published.append(args)

    monkeypatch.setattr(manual_run, "get_pr_head_sha", fake_head_sha)
    monkeypatch.setattr(manual_run, "publish_review", fake_publish)

    outcome = await manual_run._publish_if_requested(
        True, "example-org/example-repo", 137, _result(), "fake-token"
    )

    assert outcome.startswith("PUBLISHED")
    assert published == [
        ("example-org/example-repo", 137, "abc123", _result()["review_result"], "fake-token")
    ]


async def test_publish_blocks_when_review_result_is_missing(monkeypatch):
    """The graph never reached finalize_review - there is nothing to publish."""

    async def should_not_verify(*args, **kwargs):
        raise AssertionError("must not verify the PR head when review_result is missing")

    monkeypatch.setattr(manual_run, "get_pr_head_sha", should_not_verify)

    outcome = await manual_run._publish_if_requested(
        True, "example-org/example-repo", 137, _result(review_result=None), "fake-token"
    )

    assert outcome == "PUBLISH BLOCKED: review_result was not computed."


async def test_publish_blocks_on_a_non_head_mismatch_block_reason(monkeypatch):
    """build_publish_block_reason can block for reasons other than a head-SHA change."""

    async def should_not_verify(*args, **kwargs):
        raise AssertionError("must not verify the PR head when the result is already blocked")

    monkeypatch.setattr(manual_run, "get_pr_head_sha", should_not_verify)

    result = _result(
        review_input=ReviewInput(
            repo="example-org/example-repo",
            pr_number=137,
            head_sha="abc123",
            included_diff="",
            diff_truncated=False,
            included_files=[],
            omitted_files=[],
            repository_context={"results": []},
            documentation_context=[],
            coverage_notes=[],
        )
    )

    outcome = await manual_run._publish_if_requested(
        True, "example-org/example-repo", 137, result, "fake-token"
    )

    assert outcome.startswith("PUBLISH BLOCKED: no reviewer produced a usable result")


async def test_publish_blocks_when_get_pr_head_sha_raises(monkeypatch):
    """A GitHub failure while re-verifying the head SHA must block, not crash or fall through."""

    async def failing_head_sha(*args, **kwargs) -> str:
        raise manual_run.GitHubToolError(
            "GitHub API returned 404 for PR #137 in example-org/example-repo"
        )

    async def should_not_publish(*args, **kwargs):
        raise AssertionError("must not publish when the head SHA could not be verified")

    monkeypatch.setattr(manual_run, "get_pr_head_sha", failing_head_sha)
    monkeypatch.setattr(manual_run, "publish_review", should_not_publish)

    outcome = await manual_run._publish_if_requested(
        True, "example-org/example-repo", 137, _result(), "fake-token"
    )

    assert outcome.startswith("PUBLISH BLOCKED: unable to verify the current PR head SHA:")


async def test_publish_fails_when_publish_review_raises(monkeypatch):
    """A publish failure that reaches GitHub must be reported, not silently swallowed."""

    async def fake_head_sha(*args, **kwargs) -> str:
        return "abc123"

    async def failing_publish(*args, **kwargs):
        raise manual_run.GitHubPublisherError(
            "GitHub API returned 422 while publishing a review for PR #137 in example-org/example-repo"
        )

    monkeypatch.setattr(manual_run, "get_pr_head_sha", fake_head_sha)
    monkeypatch.setattr(manual_run, "publish_review", failing_publish)

    outcome = await manual_run._publish_if_requested(
        True, "example-org/example-repo", 137, _result(), "fake-token"
    )

    assert outcome.startswith("PUBLISH FAILED:")
