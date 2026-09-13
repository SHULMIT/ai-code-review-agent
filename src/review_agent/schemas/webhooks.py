"""Typed, minimal GitHub Webhook payloads accepted at the HTTP boundary."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


PositiveWebhookId = Annotated[int, Field(strict=True, gt=0)]
NonBlankWebhookText = Annotated[str, Field(strict=True, min_length=1)]


class GitHubWebhookModel(BaseModel):
    """Base model that retains only fields the Review Agent needs from GitHub."""

    model_config = ConfigDict(extra="ignore")


class WebhookRepository(GitHubWebhookModel):
    """The repository that received the Webhook delivery."""

    full_name: NonBlankWebhookText
    fork: bool


class WebhookInstallation(GitHubWebhookModel):
    """The GitHub App installation that authorized the delivery."""

    id: PositiveWebhookId


class PullRequestHeadRepository(GitHubWebhookModel):
    """The repository that owns the pull request's source branch."""

    full_name: NonBlankWebhookText
    fork: bool


class PullRequestHead(GitHubWebhookModel):
    """The immutable commit reference for the pull request head."""

    sha: NonBlankWebhookText
    repo: PullRequestHeadRepository | None = None


class PullRequest(GitHubWebhookModel):
    """The pull-request fields required to start a review."""

    head: PullRequestHead


class PullRequestWebhookPayload(GitHubWebhookModel):
    """A supported pull_request delivery with the review target identifiers."""

    action: NonBlankWebhookText
    repository: WebhookRepository
    installation: WebhookInstallation
    number: PositiveWebhookId
    pull_request: PullRequest


class IssuePullRequestReference(GitHubWebhookModel):
    """Proof that an issue_comment delivery belongs to a pull request."""

    url: NonBlankWebhookText


class Issue(GitHubWebhookModel):
    """The issue wrapper that supplies a pull-request number to a comment event."""

    number: PositiveWebhookId
    pull_request: IssuePullRequestReference | None = None


class IssueComment(GitHubWebhookModel):
    """The comment text that can request an explicit review."""

    body: str


class IssueCommentWebhookPayload(GitHubWebhookModel):
    """An issue_comment delivery for a pull request, including its command text.

    GitHub does not include the pull request head SHA in this event. The later
    review launcher obtains the current head SHA through the GitHub API.
    """

    action: NonBlankWebhookText
    repository: WebhookRepository
    installation: WebhookInstallation
    issue: Issue
    comment: IssueComment


class CheckRunRequestedAction(GitHubWebhookModel):
    """The button identifier GitHub reports when a Check Run action is clicked."""

    identifier: NonBlankWebhookText


class CheckRunPullRequestRef(GitHubWebhookModel):
    """A pull request GitHub associates with a Check Run's head SHA.

    GitHub only populates this when the check run's head_sha belongs to an
    open pull request in this same repository - exactly the case this
    service creates Check Runs for, since forked-source PRs are already
    rejected before a Check Run is ever created (see event_router.py).
    """

    number: PositiveWebhookId


class CheckRunObject(GitHubWebhookModel):
    """The Check Run fields required to verify and launch a requested review."""

    id: PositiveWebhookId
    head_sha: NonBlankWebhookText
    pull_requests: list[CheckRunPullRequestRef] = Field(default_factory=list)


class CheckRunWebhookPayload(GitHubWebhookModel):
    """A check_run delivery, including the requested action that can start a review."""

    action: NonBlankWebhookText
    check_run: CheckRunObject
    requested_action: CheckRunRequestedAction | None = None
    repository: WebhookRepository
    installation: WebhookInstallation
