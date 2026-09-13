"""Pure routing for authenticated GitHub Webhook events."""

from enum import Enum

from pydantic import BaseModel

from review_agent.schemas.webhooks import (
    CheckRunWebhookPayload,
    IssueCommentWebhookPayload,
    PullRequestWebhookPayload,
)

# The exact identifier of the Check Run action that starts a review - must
# match the "identifier" this service attaches when creating the Check Run
# (see tools/check_run_tool.py's RUN_CODE_REVIEW_ACTION).
_RUN_CODE_REVIEW_ACTION_IDENTIFIER = "run_code_review"


class ReviewTrigger(str, Enum):
    """The supported GitHub actions that can request a review run."""

    PULL_REQUEST_OPENED = "pull_request.opened"
    PULL_REQUEST_SYNCHRONIZE = "pull_request.synchronize"
    ISSUE_COMMENT = "issue_comment.review"
    CHECK_RUN_REQUESTED = "check_run.requested_action"


class ReviewRequest(BaseModel):
    """The validated identity required to launch one review attempt."""

    repository: str
    pr_number: int
    head_sha: str | None = None
    installation_id: int
    trigger: ReviewTrigger


class CheckRunCreationRequest(BaseModel):
    """The validated identity required to create or reset the DataNet Code Review Check Run."""

    repository: str
    pr_number: int
    head_sha: str
    installation_id: int


class CheckRunActionRequest(BaseModel):
    """The validated identity required to run Code Review from a Check Run's requested action.

    check_run_head_sha is the SHA the clicked Check Run was created for -
    kept separate from a freshly fetched current head SHA so the launcher can
    detect a PR that has since moved on and refuse to review a stale commit.
    """

    repository: str
    pr_number: int
    check_run_id: int
    check_run_head_sha: str
    installation_id: int


def route_github_event(
    event_name: str | None,
    payload: PullRequestWebhookPayload | IssueCommentWebhookPayload,
) -> ReviewRequest | None:
    """Return a review request only for a supported and safe GitHub trigger."""

    if isinstance(payload, PullRequestWebhookPayload):
        return _route_pull_request(event_name, payload)
    if isinstance(payload, IssueCommentWebhookPayload):
        return _route_issue_comment(event_name, payload)
    return None


def _route_pull_request(
    event_name: str | None, payload: PullRequestWebhookPayload
) -> ReviewRequest | None:
    """pull_request no longer launches a review directly - see route_check_run_creation.

    opened/synchronize now only create or reset the DataNet Code Review
    Check Run; a review only starts once that Check Run's "Run Code Review"
    action is explicitly requested (see route_check_run_action below).
    """
    return None


def route_check_run_creation(
    event_name: str | None, payload: PullRequestWebhookPayload
) -> CheckRunCreationRequest | None:
    """Return a request to create/reset the Check Run for a safe pull_request push.

    Only opened and synchronize matter here - the two actions that mean "a
    head SHA now exists (or changed) and needs its own on-demand Check Run".
    Reuses the same fork-safety filter as the old direct-launch path.
    """
    if event_name != "pull_request" or not _is_safe_pull_request(payload):
        return None
    if payload.action not in {"opened", "synchronize"}:
        return None
    return CheckRunCreationRequest(
        repository=payload.repository.full_name,
        pr_number=payload.number,
        head_sha=payload.pull_request.head.sha,
        installation_id=payload.installation.id,
    )


def route_check_run_action(
    event_name: str | None, payload: CheckRunWebhookPayload
) -> CheckRunActionRequest | None:
    """Return a request to run Code Review, only for the exact "Run Code Review" click.

    Rejects a fork repository the same way pull_request events already do,
    and rejects a check_run with no associated pull request - GitHub omits
    that association when it cannot be determined safely, and this service
    has nothing safe to review without a concrete PR number.
    """
    if (
        event_name != "check_run"
        or payload.action != "requested_action"
        or payload.repository.fork
        or payload.requested_action is None
        or payload.requested_action.identifier != _RUN_CODE_REVIEW_ACTION_IDENTIFIER
    ):
        return None

    pull_requests = payload.check_run.pull_requests
    if not pull_requests:
        return None

    return CheckRunActionRequest(
        repository=payload.repository.full_name,
        pr_number=pull_requests[0].number,
        check_run_id=payload.check_run.id,
        check_run_head_sha=payload.check_run.head_sha,
        installation_id=payload.installation.id,
    )


def _route_issue_comment(
    event_name: str | None, payload: IssueCommentWebhookPayload
) -> ReviewRequest | None:
    if (
        event_name != "issue_comment"
        or payload.action != "created"
        or payload.comment.body != "/review"
        or payload.issue.pull_request is None
        or payload.repository.fork
    ):
        return None

    return ReviewRequest(
        repository=payload.repository.full_name,
        pr_number=payload.issue.number,
        installation_id=payload.installation.id,
        trigger=ReviewTrigger.ISSUE_COMMENT,
    )


def _is_safe_pull_request(payload: PullRequestWebhookPayload) -> bool:
    """Reject deliveries targeting a fork or sourced from an external fork."""

    head_repository = payload.pull_request.head.repo
    return (
        not payload.repository.fork
        and head_repository is not None
        and not head_repository.fork
        and head_repository.full_name == payload.repository.full_name
    )
