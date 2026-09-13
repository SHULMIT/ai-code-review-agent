"""Tests for the typed GitHub Webhook HTTP-boundary models."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from review_agent.schemas.webhooks import (
    IssueCommentWebhookPayload,
    PullRequestWebhookPayload,
)


_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "webhooks"


def _load_fixture(name: str) -> dict[str, object]:
    """Load one saved GitHub delivery fixture without altering its contents."""

    return json.loads((_FIXTURE_ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("fixture_name", "action", "head_sha"),
    [
        ("pull_request_opened.json", "opened", "a1b2c3d4e5f678901234567890abcdef12345678"),
        (
            "pull_request_synchronize.json",
            "synchronize",
            "b2c3d4e5f678901234567890abcdef1234567890",
        ),
    ],
)
def test_pull_request_fixtures_parse_required_review_identifiers(
    fixture_name: str, action: str, head_sha: str
) -> None:
    """Supported pull_request fixtures retain only the review identifiers."""

    payload = PullRequestWebhookPayload.model_validate(_load_fixture(fixture_name))

    assert payload.action == action
    assert payload.repository.full_name == "example-org/example-repo"
    assert payload.installation.id == 123456
    assert payload.number == 42
    assert payload.pull_request.head.sha == head_sha


def test_issue_comment_fixture_parses_pr_number_and_comment_text() -> None:
    """A pull-request issue_comment fixture preserves its command and identifiers."""

    payload = IssueCommentWebhookPayload.model_validate(
        _load_fixture("issue_comment_created.json")
    )

    assert payload.action == "created"
    assert payload.repository.full_name == "example-org/example-repo"
    assert payload.installation.id == 123456
    assert payload.issue.number == 42
    assert payload.issue.pull_request.url.endswith("/pulls/42")
    assert payload.comment.body == "/review"


def test_pull_request_payload_rejects_a_non_string_action_precisely() -> None:
    """The event router, rather than Pydantic, decides which actions are supported."""

    payload = _load_fixture("pull_request_opened.json")
    payload["action"] = 123

    with pytest.raises(ValidationError) as exc_info:
        PullRequestWebhookPayload.model_validate(payload)

    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "string_type",
            "loc": ("action",),
            "msg": "Input should be a valid string",
            "input": 123,
        }
    ]


def test_issue_comment_payload_reports_the_missing_nested_field() -> None:
    """Missing required event fields identify their exact GitHub payload path."""

    payload = _load_fixture("issue_comment_created.json")
    del payload["installation"]["id"]

    with pytest.raises(ValidationError) as exc_info:
        IssueCommentWebhookPayload.model_validate(payload)

    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "missing",
            "loc": ("installation", "id"),
            "msg": "Field required",
            "input": {},
        }
    ]


def test_issue_comment_payload_rejects_a_non_integer_pr_number() -> None:
    """Review identifiers reject coercion so malformed JSON cannot target another PR."""

    payload = _load_fixture("issue_comment_created.json")
    payload["issue"]["number"] = "42"

    with pytest.raises(ValidationError) as exc_info:
        IssueCommentWebhookPayload.model_validate(payload)

    assert exc_info.value.errors(include_url=False) == [
        {
            "type": "int_type",
            "loc": ("issue", "number"),
            "msg": "Input should be a valid integer",
            "input": "42",
        }
    ]
