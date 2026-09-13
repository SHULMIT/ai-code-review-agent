"""Bridges the on-demand Check Run trigger to the existing review launcher.

Two responsibilities, kept in one small service since both need the same
GitHub App installation token and the same Check Run tool:

1. On a safe pull_request opened/synchronize push, create or reset the
   "DataNet Code Review" Check Run into its completed/neutral "ready to run"
   state - never starts a review by itself.
2. On that Check Run's "Run Code Review" action being requested, verify the
   PR has not moved to a newer commit since the Check Run was created, mark
   it in_progress, and hand off to the existing ReviewLauncher.launch() -
   the actual review pipeline is never duplicated here.
"""

import logging

from fastapi import BackgroundTasks

from review_agent.config import Settings
from review_agent.logging import log_event
from review_agent.services.event_router import (
    CheckRunActionRequest,
    CheckRunCreationRequest,
    ReviewRequest,
    ReviewTrigger,
)
from review_agent.services.review_formatter import format_check_run_start
from review_agent.services.review_launcher import ReviewLauncher, ReviewLaunchResult
from review_agent.tools.check_run_tool import (
    CheckRunToolError,
    complete_check_run,
    create_or_reset_check_run,
    mark_check_run_in_progress,
)
from review_agent.tools.github_auth import GitHubAuthError, InstallationTokenProvider
from review_agent.tools.github_tool import GitHubToolError, get_pr_head_sha

_LOGGER = logging.getLogger("review_agent")


class CheckRunLauncherError(Exception):
    """Raised when the Check Run trigger mechanism cannot safely proceed."""


class CheckRunLauncher:
    """Create/reset the DataNet Code Review Check Run and launch a review only on request."""

    def __init__(
        self, token_provider: InstallationTokenProvider, review_launcher: ReviewLauncher
    ) -> None:
        """Bind the GitHub App token provider and the existing review launcher to reuse."""

        self._token_provider = token_provider
        self._review_launcher = review_launcher

    async def handle_pull_request_event(self, request: CheckRunCreationRequest) -> None:
        """Create or reset the Check Run for a pull_request opened/synchronize push.

        Never launches a review - see module docstring. Wraps every GitHub
        Auth/Checks API failure into CheckRunLauncherError so the caller can
        report a clear, safe error without a raw exception escaping.
        """
        try:
            token = await self._token_provider.get_token(request.installation_id)
            await create_or_reset_check_run(request.repository, request.head_sha, token)
        except (GitHubAuthError, CheckRunToolError) as exc:
            raise CheckRunLauncherError(
                f"Unable to create the Code Review Check Run for "
                f"{request.repository}#{request.pr_number}: {exc}"
            ) from exc

    async def handle_requested_action(
        self,
        request: CheckRunActionRequest,
        github_delivery_id: str,
        background_tasks: BackgroundTasks,
    ) -> ReviewLaunchResult:
        """Verify the Check Run's SHA is still current, then launch the existing review pipeline.

        Refuses to review a stale commit: if the PR's current head SHA no
        longer matches the SHA this specific Check Run was created for, that
        Check Run is completed with an explicit "this is an older commit"
        message and a fresh Check Run is created/reset for the PR's current
        head SHA (see _recover_from_stale_check_run) - the user is never left
        with only a dead-end message and no way to actually run a review. A
        genuinely concurrent duplicate click (one already pending/running for
        this same PR/head SHA) is caught by review_runs's partial unique
        index inside ReviewLauncher.launch() - that index only covers active
        runs, so a retry after a prior attempt already reached a terminal
        state (failed, completed, skipped) is a new, accepted launch, not a
        duplicate; see migrations/005_allow_retry_after_terminal_review_run.sql.

        Every exit below completes this Check Run (or, for the stale case,
        the fresh one it hands off to) with a clear conclusion before
        returning - never a silent return that leaves the Check Run hanging
        with no explanation.
        """
        token = await self._token_provider.get_token(request.installation_id)

        try:
            current_head_sha = await get_pr_head_sha(request.repository, request.pr_number, token)
        except GitHubToolError:
            await self._safe_complete(
                request.repository, request.check_run_id, token,
                "failure", "Could not verify the pull request",
                "The pull request's current commit could not be checked before starting the "
                "review. Click Run Code Review to try again.",
            )
            log_event(
                _LOGGER, logging.WARNING, "check_run_requested_action_blocked",
                reason_code="verify_failed",
                repository=request.repository, pr_number=request.pr_number,
                check_run_id=request.check_run_id,
            )
            return ReviewLaunchResult(accepted=False, duplicate=False, status="blocked")

        if current_head_sha != request.check_run_head_sha:
            await self._recover_from_stale_check_run(request, token, current_head_sha)
            return ReviewLaunchResult(accepted=False, duplicate=False, status="stale")

        review_request = ReviewRequest(
            repository=request.repository,
            pr_number=request.pr_number,
            head_sha=current_head_sha,
            installation_id=request.installation_id,
            trigger=ReviewTrigger.CHECK_RUN_REQUESTED,
        )
        launch_result = await self._review_launcher.launch(
            review_request, github_delivery_id, background_tasks,
            check_run_id=request.check_run_id,
        )
        if launch_result.accepted:
            await self._safe_mark_in_progress(request.repository, request.check_run_id, token)
        elif launch_result.duplicate:
            await self._safe_complete(
                request.repository, request.check_run_id, token,
                "neutral", "Review already in progress",
                "A DataNet Code Review is already running or queued for this exact commit. "
                "Wait for it to finish, then check this pull request's Check Run for the result.",
                allow_rerun=False,
            )
            log_event(
                _LOGGER, logging.INFO, "check_run_requested_action_blocked",
                reason_code="duplicate_request",
                repository=request.repository, pr_number=request.pr_number,
                check_run_id=request.check_run_id,
            )
        return launch_result

    async def _recover_from_stale_check_run(
        self, request: CheckRunActionRequest, token: str, current_head_sha: str
    ) -> None:
        """Complete a stale Check Run and hand the user a fresh, clickable one.

        create_or_reset_check_run is keyed by head_sha (see check_run_tool.py):
        it resets an existing Check Run for current_head_sha instead of
        creating a new one if a repeated stale click (or a redelivered
        webhook) already produced one, so this can never grow Check Runs
        without bound on repeated clicks - at most one Check Run per real SHA
        ever exists. A failure creating/resetting the fresh Check Run is
        logged and swallowed, not raised: the stale Check Run has already
        been completed with a clear message by that point, so the webhook
        delivery still ends in a safe, visible state even in that case.
        """
        stale_sha = request.check_run_head_sha
        await self._safe_complete(
            request.repository, request.check_run_id, token,
            "neutral", "This check belongs to an older commit",
            f"This Check Run was created for commit {stale_sha[:12]}, but the pull "
            f"request has since moved to {current_head_sha[:12]}. A new DataNet Code "
            "Review check has been created for the current commit - use its "
            "Run Code Review action instead.",
            allow_rerun=False,
        )

        new_check_run_id: int | None = None
        try:
            new_check_run_id = await create_or_reset_check_run(
                request.repository, current_head_sha, token
            )
        except CheckRunToolError:
            pass

        log_event(
            _LOGGER, logging.INFO, "check_run_requested_action_blocked",
            reason_code="stale_check_run",
            repository=request.repository, pr_number=request.pr_number,
            stale_head_sha=stale_sha[:12], head_sha=current_head_sha[:12],
            check_run_id=new_check_run_id,
        )

    async def _safe_complete(
        self,
        repo: str,
        check_run_id: int,
        token: str,
        conclusion: str,
        title: str,
        summary: str,
        *,
        allow_rerun: bool = True,
    ) -> None:
        """Complete the Check Run, swallowing a Checks API failure.

        The webhook delivery must still be acknowledged even when GitHub's
        Checks API itself is unreachable - there is nothing more useful this
        request can do about it.
        """
        try:
            await complete_check_run(
                repo, check_run_id, conclusion, title, summary, token, allow_rerun=allow_rerun
            )
        except CheckRunToolError:
            pass

    async def _safe_mark_in_progress(self, repo: str, check_run_id: int, token: str) -> None:
        """Mark the Check Run in_progress with the immediate "0/6 reviewers completed"
        text, swallowing a Checks API failure.

        The review itself has already been scheduled by this point - a
        failed status update must not prevent it from running. This call is
        what stops the Check Run from ever looking like it is still on
        "Ready" after the click was accepted; live per-reviewer updates then
        take over inside review_launcher.py's _run_graph_with_progress.
        """
        title, summary = format_check_run_start()
        try:
            await mark_check_run_in_progress(repo, check_run_id, token, title=title, summary=summary)
        except CheckRunToolError:
            pass


def build_check_run_launcher(settings: Settings, review_launcher: ReviewLauncher) -> CheckRunLauncher:
    """Build the production Check Run launcher, reusing the existing review launcher."""

    token_provider = InstallationTokenProvider(
        app_id=settings.github_app_id,
        private_key_path=settings.github_app_private_key_path,
    )
    return CheckRunLauncher(token_provider, review_launcher)
