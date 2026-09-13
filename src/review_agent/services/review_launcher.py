"""Atomic review-run creation and same-process background execution."""

import logging
import time
from collections.abc import Awaitable, Callable

import httpx
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field

from review_agent.config import Settings
from review_agent.graph.build_graph import build_graph
from review_agent.graph.state import ReviewState
from review_agent.logging import log_event
from review_agent.models.review_input import ReviewInput
from review_agent.models.review_result import ReviewResult
from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic
from review_agent.nodes.finalize_review import SCORED_CATEGORIES
from review_agent.services.event_router import ReviewRequest
from review_agent.services.review_formatter import format_check_run_final, format_check_run_progress
from review_agent.storage.database import Database
from review_agent.storage.review_runs import (
    PreviousPublish,
    ReviewRun,
    ReviewRunCompletion,
    ReviewRunCreate,
    ReviewRunRepository,
)
from review_agent.tools.check_run_tool import (
    CheckRunToolError,
    complete_check_run,
    update_check_run_progress,
)
from review_agent.tools.github_auth import InstallationTokenProvider
from review_agent.tools.github_publisher import GitHubPublisherError, publish_review

# finalize_review.py's exact recommendation strings, mapped to the GitHub
# Check Run conclusion this session agreed on. A recommendation absent here
# (there is none today, but a future one would be) falls back to "neutral"
# in _check_run_conclusion_for rather than raising - a Check Run's own
# conclusion is a status surface, not a place to fail a whole review run
# over an unmapped label.
_CHECK_RUN_CONCLUSION_BY_RECOMMENDATION = {
    "APPROVE": "success",
    "REQUEST_CHANGES": "action_required",
    "INCOMPLETE": "neutral",
}

_LOGGER = logging.getLogger("review_agent")


class ReviewLauncherConfigurationError(Exception):
    """Raised when a review launcher lacks required runtime configuration."""


class ReviewLaunchResult(BaseModel):
    """The HTTP-safe outcome of one atomic launch attempt."""

    accepted: bool
    duplicate: bool
    review_run_id: str | None = None
    status: str = Field(min_length=1)


ReviewWorker = Callable[[ReviewRun], Awaitable[None]]


class ReviewLauncher:
    """Create a pending review atomically and schedule work only on success."""

    def __init__(self, repository: ReviewRunRepository, review_worker: ReviewWorker) -> None:
        """Bind the durable run repository and the background review worker."""

        self._repository = repository
        self._review_worker = review_worker

    async def launch(
        self,
        review_request: ReviewRequest,
        github_delivery_id: str,
        background_tasks: BackgroundTasks,
        check_run_id: int | None = None,
    ) -> ReviewLaunchResult:
        """Atomically insert a pending run and enqueue work only for new records.

        check_run_id is set only by the Check Run "Run Code Review" trigger
        (see check_run_launcher.py) - every other trigger omits it, and
        run_review_graph below only touches a Check Run when it is present.
        """

        review_run = await self._repository.create_pending(
            ReviewRunCreate(
                github_delivery_id=github_delivery_id,
                review_identity=build_review_identity(
                    review_request, github_delivery_id
                ),
                repository=review_request.repository,
                pr_number=review_request.pr_number,
                installation_id=review_request.installation_id,
                trigger=review_request.trigger,
                head_sha=review_request.head_sha,
                check_run_id=check_run_id,
            )
        )
        if review_run is None:
            return ReviewLaunchResult(
                accepted=False,
                duplicate=True,
                status="duplicate",
            )

        background_tasks.add_task(self._run_review, review_run)
        return ReviewLaunchResult(
            accepted=True,
            duplicate=False,
            review_run_id=str(review_run.id),
            status="accepted",
        )

    async def _run_review(self, review_run: ReviewRun) -> None:
        """Mark a review running and record a safe failure if its worker raises."""

        running_review = await self._repository.mark_running(review_run.id)
        if running_review is None:
            return

        try:
            await self._review_worker(running_review)
        except Exception as exc:
            try:
                await self._repository.fail(
                    review_run.id,
                    f"Review worker failed: {type(exc).__name__}",
                )
            except Exception:
                return


def build_review_identity(
    review_request: ReviewRequest, github_delivery_id: str
) -> str:
    """Build the selected database duplicate identity without inventing a head SHA."""

    if review_request.head_sha is None:
        return f"delivery:{github_delivery_id}"
    return ":".join(
        (
            review_request.repository,
            str(review_request.pr_number),
            review_request.head_sha,
            review_request.trigger.value,
        )
    )


def build_publish_block_reason(
    review_input: ReviewInput | None, head_sha: str | None, review_result: ReviewResult
) -> str | None:
    """Return why publishing to GitHub should be blocked, or None when it is safe to publish.

    A review is never published when no real analysis actually happened at
    all: load_pr never fetched a head SHA, gather_context never collected
    any diff content, or every reviewer category is in
    review_result.incomplete_categories (e.g. the Model Service was
    unreachable, so every reviewer call errored) with no findings at all to
    show for it - a true "nothing was analyzed" case. A partial failure -
    some categories incomplete, others (or static_checks) producing real
    findings - is not blocked: it still publishes, as a COMMENT review that
    names the incomplete categories explicitly (see review_formatter.py and
    github_publisher.py's recommendation-to-event mapping). A genuinely
    clean PR - zero findings, zero incomplete categories - is not blocked
    either; it still publishes normally as APPROVE. Checking head_sha and
    included_diff explicitly, rather than relying on GitHub itself
    rejecting a blank commit_id, makes this an intentional policy instead
    of an accidental side effect of the GitHub API's own validation.
    """
    if not head_sha:
        return "no reviewer produced a usable result - PR head SHA was never fetched (load_pr failed)"
    if review_input is None or not review_input.included_diff:
        return "no reviewer produced a usable result - no diff content was available to review"

    has_findings = bool(
        review_result.blocking_findings or review_result.warnings or review_result.suggestions
    )
    every_category_incomplete = set(review_result.incomplete_categories) >= SCORED_CATEGORIES
    if not has_findings and every_category_incomplete:
        return "no reviewer produced a usable result - every review category failed to complete"

    return None


def _build_review_run_completion(
    review_result: ReviewResult, last_review_id: int | None = None
) -> ReviewRunCompletion:
    """Flatten ReviewResult's severity-grouped findings back into one list for storage."""

    return ReviewRunCompletion(
        findings=review_result.blocking_findings + review_result.warnings + review_result.suggestions,
        architecture_score=review_result.architecture_score,
        code_quality_score=review_result.code_quality_score,
        production_readiness_score=review_result.production_readiness_score,
        recommendation=review_result.recommendation,
        errors=review_result.errors,
        incomplete_categories=review_result.incomplete_categories,
        last_review_id=last_review_id,
    )


async def _update_check_run_if_requested(
    review_run: ReviewRun,
    token: str,
    client: httpx.AsyncClient,
    conclusion: str,
    title: str,
    summary: str,
) -> None:
    """Complete the Check Run this run was launched from, if any - never raises.

    Only a Check Run-triggered run has check_run_id set; every other trigger
    (a direct pull_request/issue_comment launch, from before this feature, or
    a future one) leaves it None and this is a no-op. A Check Run update
    failure must never change the review run's own recorded outcome - it is
    purely the trigger/status surface, not the review result itself.
    """
    if review_run.check_run_id is None:
        return
    try:
        await complete_check_run(
            review_run.repository, review_run.check_run_id, conclusion, title, summary,
            token, client=client,
        )
    except CheckRunToolError:
        pass


def _completed_categories(diagnostics: list[ReviewerDiagnostic]) -> frozenset[str]:
    """The set of reviewer categories that have reached a terminal status so far."""
    return frozenset(d.category for d in diagnostics if d.status in ("completed", "failed", "skipped"))


async def _run_graph_with_progress(
    graph, initial_state: ReviewState, review_run: ReviewRun, token: str, client: httpx.AsyncClient
) -> dict:
    """Run the graph via astream, pushing a live Check Run update each time a new
    reviewer reaches a terminal status - never on every intermediate node, so a
    six-reviewer run produces at most six extra GitHub API calls, not one per
    LangGraph super-step.

    A Check Run update failure here is swallowed exactly like every other
    Check Run touch in this module: the review itself must keep running
    regardless of whether GitHub's Checks API is momentarily unavailable.
    """
    final_state: dict = {}
    reported_categories: frozenset[str] = frozenset()
    async for state_update in graph.astream(initial_state, stream_mode="values"):
        final_state = state_update
        if review_run.check_run_id is None:
            continue
        diagnostics = final_state.get("reviewer_diagnostics") or []
        current_categories = _completed_categories(diagnostics)
        if current_categories == reported_categories:
            continue
        reported_categories = current_categories
        title, summary = format_check_run_progress(diagnostics)
        try:
            await update_check_run_progress(
                review_run.repository, review_run.check_run_id, token, title, summary, client
            )
        except CheckRunToolError:
            pass
    return final_state


def _reason_code_for_block(block_reason: str) -> str:
    """Classify a build_publish_block_reason() message into a structured reason_code."""
    if "PR head SHA was never fetched" in block_reason:
        return "github_api_failed"
    if "no diff content was available" in block_reason:
        return "repository_context_failed"
    return "reviewer_failed"


def _safe_message_for_block(block_reason: str) -> str:
    """Translate a block_reason into a safe, human-readable GitHub-facing message.

    block_reason strings are internal and code-review-friendly, not written
    for a PR author - this maps them to the same safe, retry-guidance-bearing
    style as every other terminal outcome (see review_formatter.py).
    """
    reason_code = _reason_code_for_block(block_reason)
    if reason_code == "github_api_failed":
        return "Could not read this pull request from GitHub. Retry now."
    if reason_code == "repository_context_failed":
        return "Could not gather this pull request's changes to review. Retry now."
    return "Code review was not completed: every reviewer failed to run. Retry later."


def _build_review_url(repo: str, pr_number: int, review_id: int) -> str:
    """Build a direct link to one published PR Review from the ID GitHub returned.

    GitHub's own stable, documented anchor for a single review on a PR's
    Conversation page - no extra API call or response field needed beyond
    the review ID publish_review() already returns and review_runs already
    stores as last_review_id.
    """
    return f"https://github.com/{repo}/pull/{pr_number}#pullrequestreview-{review_id}"


def _reusable_review_id(
    previous_publish: PreviousPublish | None, review_result: ReviewResult
) -> int | None:
    """Return the prior review's ID only when it already shows this same recommendation.

    GitHub cannot change an already-submitted review's approve/request-changes
    decision, only its body text - reusing an APPROVE review's ID for a
    REQUEST_CHANGES re-run would update its text while leaving a stale
    Approved banner on the PR. Returning None here makes the caller create a
    fresh review instead, exactly as if nothing had been published before.
    """
    if previous_publish is None:
        return None
    if previous_publish.recommendation != review_result.recommendation:
        return None
    return previous_publish.review_id


def build_review_launcher(settings: Settings) -> ReviewLauncher:
    """Build the production launcher from validated service settings."""

    if settings.database_url is None:
        raise ReviewLauncherConfigurationError("DATABASE_URL is required for review runs.")

    token_provider = InstallationTokenProvider(
        app_id=settings.github_app_id,
        private_key_path=settings.github_app_private_key_path,
    )
    repository = ReviewRunRepository(Database(str(settings.database_url)))

    async def run_review_graph(review_run: ReviewRun) -> None:
        """Start the existing LangGraph foundation, then publish its result to GitHub.

        Publishing is a launcher-level concern, downstream of finalize_review -
        not a graph node. Never publishes when no real analysis happened
        (see build_publish_block_reason) - that case is recorded as a failed
        run instead, and GitHub is never contacted. A publish failure that
        does reach GitHub and still fails is not caught here: it propagates
        to _run_review's existing catch-all, which already marks the review
        run failed and records the exception type. Only once publishing
        actually succeeds is the run marked completed - a review that was
        never delivered to GitHub must never look like a completed one in
        our stored history.

        Looks up the last review published for this PR before publishing -
        see _reusable_review_id - so a re-run whose recommendation hasn't
        changed updates that same review in place instead of accumulating a
        new one on every push. The returned review ID (new or reused) is
        always stored, so the next run can make the same decision.

        One httpx.AsyncClient is opened here and threaded through the whole
        run - the graph's GitHub, Model Service, and embedding calls, and
        the final publish_review call - so a single review run reuses pooled
        connections instead of opening a fresh one per HTTP call. See
        RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5.

        When review_run.check_run_id is set (a Check Run "Run Code Review"
        trigger - see check_run_launcher.py), the whole body below runs
        inside a try/except that completes that same Check Run - success,
        action_required, or neutral by recommendation, failure for every
        other exit including a genuinely unhandled exception - before
        re-raising so _run_review's existing catch-all still records the
        review run's own failure exactly as it did before this feature.
        """

        installation_token = await token_provider.get_token(review_run.installation_id)
        started_at = time.monotonic()

        def _log_blocked(reason_code: str) -> None:
            log_event(
                _LOGGER, logging.ERROR, "check_run_requested_action_blocked",
                reason_code=reason_code,
                repository=review_run.repository, pr_number=review_run.pr_number,
                check_run_id=review_run.check_run_id,
            )

        # Tags which stage a later, single except Exception (below) is
        # catching, without restructuring it into nested try/excepts that
        # would otherwise duplicate the failure-completion/re-raise logic.
        reason_code = "review_graph_failed"

        async with httpx.AsyncClient() as client:
            try:
                graph = build_graph(installation_token, settings, client=client)
                final_state = await _run_graph_with_progress(
                    graph,
                    ReviewState(repo=review_run.repository, pr_number=review_run.pr_number),
                    review_run,
                    installation_token,
                    client,
                )

                # LangGraph's returned state only carries a key for a field once some
                # node has actually set it - on a load_pr failure, for example,
                # "head_sha" is absent entirely, not None. .get(...) everywhere here
                # instead of direct subscripting, so a total upstream failure cannot
                # raise a KeyError instead of going through build_publish_block_reason.
                review_result = final_state.get("review_result")
                if review_result is None:
                    reason = "no reviewer produced a usable result - review_result was never computed"
                    await repository.fail(review_run.id, reason)
                    await _update_check_run_if_requested(
                        review_run, installation_token, client,
                        "failure", "Code Review failed",
                        "Code review could not start: the review pipeline did not produce a "
                        "result. Retry now.",
                    )
                    _log_blocked("review_graph_failed")
                    return

                review_result = review_result.model_copy(
                    update={"duration_seconds": time.monotonic() - started_at}
                )

                block_reason = build_publish_block_reason(
                    final_state.get("review_input"), final_state.get("head_sha"), review_result
                )
                if block_reason is not None:
                    await repository.fail(review_run.id, block_reason)
                    await _update_check_run_if_requested(
                        review_run, installation_token, client,
                        "failure", "Code Review failed", _safe_message_for_block(block_reason),
                    )
                    _log_blocked(_reason_code_for_block(block_reason))
                    return

                previous_publish = await repository.get_last_publish(
                    review_run.repository, review_run.pr_number
                )
                reason_code = "review_publish_failed"
                try:
                    review_id = await publish_review(
                        review_run.repository,
                        review_run.pr_number,
                        final_state.get("head_sha"),
                        review_result,
                        installation_token,
                        existing_review_id=_reusable_review_id(previous_publish, review_result),
                        client=client,
                    )
                except GitHubPublisherError as exc:
                    reason = f"publishing the review to GitHub failed: {type(exc).__name__}"
                    await repository.fail(review_run.id, reason)
                    await _update_check_run_if_requested(
                        review_run, installation_token, client,
                        "failure", "Code Review publishing failed",
                        "Review generation completed, but publishing the PR Review to "
                        "GitHub failed. Retry now.",
                    )
                    _log_blocked("review_publish_failed")
                    return
                reason_code = "review_graph_failed"

                # The full PR Review is now confirmed published - only now is it
                # safe to complete the Check Run and stop its spinner (see
                # HANDOFF.md: publish before complete, never the other order).
                await repository.complete(
                    review_run.id, _build_review_run_completion(review_result, review_id)
                )
                conclusion = _CHECK_RUN_CONCLUSION_BY_RECOMMENDATION.get(
                    review_result.recommendation, "neutral"
                )
                is_partial = bool(review_result.incomplete_categories)
                review_url = _build_review_url(review_run.repository, review_run.pr_number, review_id)
                await _update_check_run_if_requested(
                    review_run, installation_token, client,
                    conclusion,
                    "Code Review partially completed" if is_partial else "Code Review completed",
                    format_check_run_final(review_result, review_url=review_url),
                )
                if is_partial:
                    log_event(
                        _LOGGER, logging.WARNING, "check_run_requested_action_completed",
                        reason_code="partial_review",
                        repository=review_run.repository, pr_number=review_run.pr_number,
                        head_sha=final_state.get("head_sha"), check_run_id=review_run.check_run_id,
                        incomplete_categories=review_result.incomplete_categories,
                    )
                else:
                    log_event(
                        _LOGGER, logging.INFO, "check_run_requested_action_completed",
                        repository=review_run.repository, pr_number=review_run.pr_number,
                        head_sha=final_state.get("head_sha"), check_run_id=review_run.check_run_id,
                    )
            except Exception:
                await _update_check_run_if_requested(
                    review_run, installation_token, client,
                    "failure", "Code Review failed",
                    "The review run failed unexpectedly. Retry later.",
                )
                _log_blocked(reason_code)
                raise

    return ReviewLauncher(repository, run_review_graph)
