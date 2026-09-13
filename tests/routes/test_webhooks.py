"""Tests for authenticated GitHub Webhook intake."""

import hashlib
import hmac
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from review_agent.app import create_app
from review_agent.config import Settings
from review_agent.routes import webhooks
from review_agent.services.check_run_launcher import CheckRunLauncherError
from review_agent.services.review_launcher import ReviewLaunchResult


_SECRET = "test-webhook-secret"
_FIXTURES = Path(__file__).parents[1] / "fixtures" / "webhooks"
_PULL_REQUEST_FIXTURE = _FIXTURES / "pull_request_opened.json"
_ISSUE_COMMENT_FIXTURE = _FIXTURES / "issue_comment_created.json"
_CHECK_RUN_FIXTURE = _FIXTURES / "check_run_requested_action.json"


def _body(fixture_path: Path, **overrides: object) -> bytes:
    """Load a fixture, optionally overriding top-level fields, as the exact bytes to sign."""

    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    payload.update(overrides)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _signature(payload_body: bytes, secret: str = _SECRET) -> str:
    """Create a GitHub-compatible SHA-256 signature for the provided bytes."""

    digest = hmac.new(secret.encode("utf-8"), payload_body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class FakeReviewLauncher:
    """Record review-launch requests without starting a real database worker."""

    def __init__(self, launch_result: ReviewLaunchResult | None = None) -> None:
        self.launch_result = launch_result or ReviewLaunchResult(
            accepted=True,
            duplicate=False,
            review_run_id="run-123",
            status="accepted",
        )
        self.launches: list[tuple[object, str]] = []

    async def launch(self, review_request, github_delivery_id, background_tasks, check_run_id=None):
        self.launches.append((review_request, github_delivery_id))
        return self.launch_result


class FakeCheckRunLauncher:
    """Record Check Run trigger requests without calling the real GitHub Checks API."""

    def __init__(
        self,
        launch_result: ReviewLaunchResult | None = None,
        pull_request_event_error: Exception | None = None,
    ) -> None:
        self.launch_result = launch_result or ReviewLaunchResult(
            accepted=True,
            duplicate=False,
            review_run_id="run-123",
            status="accepted",
        )
        self._pull_request_event_error = pull_request_event_error
        self.pull_request_events: list[object] = []
        self.requested_actions: list[tuple[object, str]] = []

    async def handle_pull_request_event(self, request) -> None:
        if self._pull_request_event_error is not None:
            raise self._pull_request_event_error
        self.pull_request_events.append(request)

    async def handle_requested_action(self, request, github_delivery_id, background_tasks):
        self.requested_actions.append((request, github_delivery_id))
        return self.launch_result


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        github_app_id=123,
        github_app_private_key_path="secrets/test-private-key.pem",
        github_webhook_secret=_SECRET,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Create an app whose route receives isolated fake launchers and an isolated secret."""

    monkeypatch.setattr(webhooks, "Settings", lambda: _settings())
    review_launcher = FakeReviewLauncher()
    check_run_launcher = FakeCheckRunLauncher()
    app = create_app(review_launcher=review_launcher, check_run_launcher=check_run_launcher)
    return TestClient(app), review_launcher, check_run_launcher


def test_missing_signature_header_returns_401(client) -> None:
    """Unsigned deliveries are rejected before event selection or payload parsing."""

    response = client[0].post("/webhooks/github", content=_body(_PULL_REQUEST_FIXTURE))

    assert response.status_code == 401
    assert response.json()["error"] == "Unauthorized."


def test_malformed_signature_header_returns_401(client) -> None:
    """Non-SHA-256 GitHub signature headers are rejected."""

    response = client[0].post(
        "/webhooks/github",
        content=_body(_PULL_REQUEST_FIXTURE),
        headers={"X-Hub-Signature-256": "sha1=not-a-sha256-signature"},
    )

    assert response.status_code == 401
    assert response.json()["error"] == "Unauthorized."


def test_signature_created_with_another_secret_returns_401(client) -> None:
    """A valid-looking signature from another installation secret is rejected."""

    payload_body = _body(_PULL_REQUEST_FIXTURE)
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _signature(payload_body, "wrong-secret"),
        },
    )

    assert response.status_code == 401
    assert response.json()["error"] == "Unauthorized."


def test_invalid_signature_never_invokes_payload_parsing(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The authentication boundary prevents malformed deliveries reaching Pydantic."""

    parse_payload = Mock(side_effect=AssertionError("payload parsing must not run"))
    monkeypatch.setattr(webhooks, "parse_github_webhook_payload", parse_payload)

    response = client[0].post(
        "/webhooks/github",
        content=_body(_PULL_REQUEST_FIXTURE),
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
        },
    )

    assert response.status_code == 401
    parse_payload.assert_not_called()


def test_unsupported_signed_event_is_ignored_without_starting_work(client) -> None:
    """Events outside the supported trigger set receive 200 without touching any launcher."""

    payload_body = b'{"zen":"Keep it logically awesome."}'
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "ping",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
    assert client[1].launches == []
    assert client[2].pull_request_events == []
    assert client[2].requested_actions == []


def test_pull_request_opened_creates_or_resets_the_check_run_and_does_not_launch_a_review(
    client,
) -> None:
    """opened no longer starts the Code Review Agent - it only arms the on-demand Check Run."""

    payload_body = _body(_PULL_REQUEST_FIXTURE)
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "delivery-1",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "check_run_created"}
    assert len(client[2].pull_request_events) == 1
    assert client[1].launches == []


def test_pull_request_synchronize_creates_or_resets_the_check_run_and_does_not_launch_a_review(
    client,
) -> None:
    """synchronize behaves exactly like opened - a new push only re-arms the Check Run."""

    payload_body = _body(_PULL_REQUEST_FIXTURE, action="synchronize")
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "delivery-2",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "check_run_created"}
    assert len(client[2].pull_request_events) == 1
    assert client[1].launches == []


def test_pull_request_closed_is_ignored(client) -> None:
    """An action outside opened/synchronize does not touch the Check Run either."""

    payload_body = _body(_PULL_REQUEST_FIXTURE, action="closed")
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
    assert client[2].pull_request_events == []


def test_check_run_creation_failure_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Checks API failure is reported clearly instead of a bare 500."""

    monkeypatch.setattr(webhooks, "Settings", lambda: _settings())
    check_run_launcher = FakeCheckRunLauncher(
        pull_request_event_error=CheckRunLauncherError("Unable to create the Check Run.")
    )
    app = create_app(review_launcher=FakeReviewLauncher(), check_run_launcher=check_run_launcher)
    payload_body = _body(_PULL_REQUEST_FIXTURE)

    response = TestClient(app).post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 502


def test_run_code_review_action_launches_a_review(client) -> None:
    """The exact 'Run Code Review' click reaches the Check Run launcher and is accepted."""

    payload_body = _body(_CHECK_RUN_FIXTURE)
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "check_run",
            "X-GitHub-Delivery": "delivery-3",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "accepted": True,
        "duplicate": False,
        "review_run_id": "run-123",
        "status": "accepted",
    }
    assert len(client[2].requested_actions) == 1
    assert client[2].requested_actions[0][1] == "delivery-3"


def test_run_code_review_action_without_delivery_id_returns_400(client) -> None:
    """A Check Run launch requires GitHub's delivery key for atomic deduplication too."""

    payload_body = _body(_CHECK_RUN_FIXTURE)
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "check_run",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 400
    assert client[2].requested_actions == []


def test_check_run_with_a_different_action_identifier_is_ignored(client) -> None:
    """Only this service's own action identifier may start a review."""

    payload_body = _body(
        _CHECK_RUN_FIXTURE, requested_action={"identifier": "some_other_action"}
    )
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "check_run",
            "X-GitHub-Delivery": "delivery-4",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
    assert client[2].requested_actions == []


def test_duplicate_check_run_action_returns_200_without_a_new_task(client) -> None:
    """The route reports a database duplicate without treating it as an error."""

    client[2].launch_result = ReviewLaunchResult(
        accepted=False, duplicate=True, status="duplicate"
    )
    payload_body = _body(_CHECK_RUN_FIXTURE)

    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "check_run",
            "X-GitHub-Delivery": "delivery-5",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "accepted": False,
        "duplicate": True,
        "review_run_id": None,
        "status": "duplicate",
    }


def test_review_comment_command_still_launches_a_review_directly(client) -> None:
    """The pre-existing "/review" comment trigger is unaffected by the Check Run feature."""

    payload_body = _body(_ISSUE_COMMENT_FIXTURE)
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "issue_comment",
            "X-GitHub-Delivery": "delivery-6",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "accepted": True,
        "duplicate": False,
        "review_run_id": "run-123",
        "status": "accepted",
    }
    assert client[1].launches[0][1] == "delivery-6"
    assert client[2].pull_request_events == []
    assert client[2].requested_actions == []


def test_non_command_comment_is_ignored_without_a_launch(client) -> None:
    """An issue_comment delivery whose body isn't the exact "/review" command starts nothing."""

    payload_body = _body(_ISSUE_COMMENT_FIXTURE, comment={"body": "just a regular comment"})
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "issue_comment",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
    assert client[1].launches == []


def test_review_comment_command_without_delivery_id_returns_400(client) -> None:
    """A direct review-launch trigger still requires GitHub's delivery key."""

    payload_body = _body(_ISSUE_COMMENT_FIXTURE)
    response = client[0].post(
        "/webhooks/github",
        content=payload_body,
        headers={
            "X-GitHub-Event": "issue_comment",
            "X-Hub-Signature-256": _signature(payload_body),
        },
    )

    assert response.status_code == 400
    assert client[1].launches == []
