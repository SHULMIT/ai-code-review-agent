"""Tests for selecting safe, supported GitHub review triggers."""

import json
from pathlib import Path

import pytest

from review_agent.schemas.webhooks import (
    CheckRunWebhookPayload,
    IssueCommentWebhookPayload,
    PullRequestWebhookPayload,
)
from review_agent.services.event_router import (
    ReviewTrigger,
    route_check_run_action,
    route_check_run_creation,
    route_github_event,
)


_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "webhooks"


def _load_fixture(name: str) -> dict[str, object]:
    """Load a saved Webhook payload fixture for routing tests."""

    return json.loads((_FIXTURE_ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "fixture_name", ["pull_request_opened.json", "pull_request_synchronize.json"]
)
def test_pull_request_events_no_longer_create_review_requests(fixture_name: str) -> None:
    """opened/synchronize only create/reset a Check Run now - see route_check_run_creation."""

    payload = PullRequestWebhookPayload.model_validate(_load_fixture(fixture_name))

    assert route_github_event("pull_request", payload) is None


@pytest.mark.parametrize(
    "fixture_name", ["pull_request_opened.json", "pull_request_synchronize.json"]
)
def test_supported_pull_request_actions_create_check_run_creation_requests(
    fixture_name: str,
) -> None:
    """Opened and synchronized PRs request a Check Run for their exact head SHA."""

    payload = PullRequestWebhookPayload.model_validate(_load_fixture(fixture_name))

    request = route_check_run_creation("pull_request", payload)

    assert request is not None
    assert request.repository == "example-org/example-repo"
    assert request.pr_number == 42
    assert request.head_sha == payload.pull_request.head.sha
    assert request.installation_id == payload.installation.id


def test_unsupported_pull_request_action_does_not_request_a_check_run() -> None:
    """Closed and other PR actions never create or reset a Check Run either."""

    payload_data = _load_fixture("pull_request_opened.json")
    payload_data["action"] = "closed"
    payload = PullRequestWebhookPayload.model_validate(payload_data)

    assert route_check_run_creation("pull_request", payload) is None


def test_forked_pull_request_does_not_request_a_check_run() -> None:
    """Fork PRs are rejected for Check Run creation the same way as direct launch used to be."""

    payload_data = _load_fixture("pull_request_opened.json")
    payload_data["repository"]["fork"] = True
    payload = PullRequestWebhookPayload.model_validate(payload_data)

    assert route_check_run_creation("pull_request", payload) is None


def _check_run_payload(**overrides: object) -> CheckRunWebhookPayload:
    payload_data = _load_fixture("check_run_requested_action.json")
    payload_data.update(overrides)
    return CheckRunWebhookPayload.model_validate(payload_data)


def test_run_code_review_action_creates_a_check_run_action_request() -> None:
    """The exact 'Run Code Review' click resolves the PR, Check Run ID, and its SHA."""

    payload = _check_run_payload()

    request = route_check_run_action("check_run", payload)

    assert request is not None
    assert request.repository == "example-org/example-repo"
    assert request.pr_number == 42
    assert request.check_run_id == 987654
    assert request.check_run_head_sha == "a1b2c3d4e5f678901234567890abcdef12345678"
    assert request.installation_id == 123456


def test_a_different_requested_action_identifier_is_ignored() -> None:
    """Only this service's exact action identifier may start a review."""

    payload_data = _load_fixture("check_run_requested_action.json")
    payload_data["requested_action"]["identifier"] = "some_other_action"
    payload = CheckRunWebhookPayload.model_validate(payload_data)

    assert route_check_run_action("check_run", payload) is None


def test_a_check_run_action_other_than_requested_action_is_ignored() -> None:
    """A completed/created/rerequested check_run delivery never starts a review by itself."""

    payload_data = _load_fixture("check_run_requested_action.json")
    payload_data["action"] = "completed"
    payload = CheckRunWebhookPayload.model_validate(payload_data)

    assert route_check_run_action("check_run", payload) is None


def test_a_check_run_with_no_associated_pull_request_is_ignored() -> None:
    """Without a concrete PR number there is nothing safe to review."""

    payload_data = _load_fixture("check_run_requested_action.json")
    payload_data["check_run"]["pull_requests"] = []
    payload = CheckRunWebhookPayload.model_validate(payload_data)

    assert route_check_run_action("check_run", payload) is None


def test_a_check_run_on_a_forked_repository_is_ignored() -> None:
    """Same fork-safety boundary as every other trigger in this router."""

    payload_data = _load_fixture("check_run_requested_action.json")
    payload_data["repository"]["fork"] = True
    payload = CheckRunWebhookPayload.model_validate(payload_data)

    assert route_check_run_action("check_run", payload) is None


def test_review_command_creates_an_issue_comment_request() -> None:
    """Only a newly created exact /review comment triggers a manual review."""

    payload = IssueCommentWebhookPayload.model_validate(
        _load_fixture("issue_comment_created.json")
    )

    request = route_github_event("issue_comment", payload)

    assert request is not None
    assert request.trigger == ReviewTrigger.ISSUE_COMMENT
    assert request.head_sha is None


@pytest.mark.parametrize("comment_body", ["/review please", " /review", "/Review", ""])
def test_non_command_comments_are_ignored(comment_body: str) -> None:
    """Only the exact command prevents accidental review launches."""

    payload_data = _load_fixture("issue_comment_created.json")
    payload_data["comment"]["body"] = comment_body
    payload = IssueCommentWebhookPayload.model_validate(payload_data)

    assert route_github_event("issue_comment", payload) is None


def test_edited_review_command_is_ignored() -> None:
    """Editing a comment cannot cause a second manual review trigger."""

    payload_data = _load_fixture("issue_comment_created.json")
    payload_data["action"] = "edited"
    payload = IssueCommentWebhookPayload.model_validate(payload_data)

    assert route_github_event("issue_comment", payload) is None


def test_review_command_on_a_regular_issue_is_ignored() -> None:
    """The command must reference a pull request, not an ordinary issue."""

    payload_data = _load_fixture("issue_comment_created.json")
    payload_data["issue"].pop("pull_request")
    payload = IssueCommentWebhookPayload.model_validate(payload_data)

    assert route_github_event("issue_comment", payload) is None


def test_forked_destination_or_source_repository_is_ignored() -> None:
    """Fork PRs are not launched until untrusted-context hardening is complete."""

    destination_fork = _load_fixture("pull_request_opened.json")
    destination_fork["repository"]["fork"] = True
    source_fork = _load_fixture("pull_request_opened.json")
    source_fork["pull_request"]["head"]["repo"]["fork"] = True

    assert (
        route_github_event(
            "pull_request", PullRequestWebhookPayload.model_validate(destination_fork)
        )
        is None
    )
    assert (
        route_github_event(
            "pull_request", PullRequestWebhookPayload.model_validate(source_fork)
        )
        is None
    )


def test_unsupported_event_is_ignored() -> None:
    """Authenticated events outside the MVP trigger set do not start work."""

    payload = PullRequestWebhookPayload.model_validate(
        _load_fixture("pull_request_opened.json")
    )

    assert route_github_event("push", payload) is None


def test_unsupported_pull_request_action_is_ignored() -> None:
    """Closed and other PR actions never cause review work to start."""

    payload_data = _load_fixture("pull_request_opened.json")
    payload_data["action"] = "closed"
    payload = PullRequestWebhookPayload.model_validate(payload_data)

    assert route_github_event("pull_request", payload) is None
