"""Publishes a finished review as an official GitHub Pull Request Review.

Uses GitHub's Review API (POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews)
rather than a plain issue comment, so the outcome shows up in the PR's
"Reviewers" section with a colored Approved/Changes-requested status,
instead of being just another comment in the thread.

A re-run whose recommendation is unchanged from the last published review
updates that review's body in place via GitHub's
PUT /repos/{owner}/{repo}/pulls/{pull_number}/reviews/{review_id}, instead of
creating another one - this is the only field that endpoint can change.
GitHub does not allow updating an already-submitted review's `event` (its
approve/request-changes decision), only its body text: there is no API to
flip an existing review from APPROVE to REQUEST_CHANGES or back. The caller
(review_launcher.py) is therefore responsible for only ever passing
existing_review_id when the new recommendation matches the one already
published under that review ID; when the recommendation differs, or nothing
has been published for this PR yet, a fresh review is created instead so the
PR's Reviewers section always shows an accurate, current decision.
"""

import httpx

from review_agent.models.review_result import ReviewResult
from review_agent.services.review_formatter import format_review_body

_API_BASE = "https://api.github.com"


class GitHubPublisherError(Exception):
    """Raised when publishing a review to GitHub fails, for any reason."""


class _ExistingReviewNotFoundError(Exception):
    """Internal signal only: the stored review ID no longer exists on GitHub.

    Never escapes publish_review() - it is caught there and triggers a
    fallback to creating a new review, exactly like having no prior review
    at all (e.g. it was deleted manually on GitHub since the last run).
    """


def _github_event_for(recommendation: str) -> str:
    """Map ReviewResult.recommendation to GitHub's review `event` field, exhaustively.

    INCOMPLETE maps to COMMENT, never APPROVE or REQUEST_CHANGES: a partial
    result - some reviewer category never ran - must never look like a
    formal pass or fail, since nobody actually checked the categories that
    are missing. Written as an explicit if/elif/else rather than a dict
    lookup so a recommendation value with no mapping raises a clear error
    here instead of a bare KeyError further down.
    """
    if recommendation == "APPROVE":
        return "APPROVE"
    if recommendation == "REQUEST_CHANGES":
        return "REQUEST_CHANGES"
    if recommendation == "INCOMPLETE":
        return "COMMENT"
    raise GitHubPublisherError(
        f"no GitHub review event is defined for recommendation {recommendation!r}"
    )


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def publish_review(
    repo: str,
    pr_number: int,
    head_sha: str,
    review_result: ReviewResult,
    token: str,
    existing_review_id: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> int:
    """Submit review_result as an official GitHub PR Review, returning its review ID.

    Creates a new review tied to head_sha when existing_review_id is None -
    the first publish for this PR, or the caller determined the
    recommendation changed since the last one. When existing_review_id is
    given, updates that review's body in place instead of creating another
    one; if the stored review no longer exists on GitHub (deleted manually,
    a 404), falls back to creating a new one rather than failing outright.
    Never lets a raw httpx exception escape - every failure is wrapped in
    GitHubPublisherError, matching github_tool.py's pattern.

    client lets a caller pass in one pooled httpx.AsyncClient shared across
    an entire review run instead of opening a fresh TCP connection per call
    - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5. When omitted, this
    function falls back to its original self-contained per-call client.

    Note: GitHub blocks a *user* from approving their own pull request
    through a personal token, but that restriction is tied to user
    identity, not App identity - a GitHub App-authenticated review is
    expected to work regardless of who opened the PR. This has not been
    verified against a real PR yet; confirm it empirically before relying
    on it in production.
    """
    headers = _headers(token)

    if existing_review_id is not None:
        try:
            return await _update_existing_review(
                repo, pr_number, existing_review_id, review_result, headers, client
            )
        except _ExistingReviewNotFoundError:
            pass

    return await _create_review(repo, pr_number, head_sha, review_result, headers, client)


async def _create_review(
    repo: str,
    pr_number: int,
    head_sha: str,
    review_result: ReviewResult,
    headers: dict[str, str],
    client: httpx.AsyncClient | None = None,
) -> int:
    """Create a new GitHub PR Review via POST and return its ID.

    commit_id is included explicitly so the review is tied to the exact
    commit this run actually reviewed, not silently attached to whatever
    the PR's current head happens to be if it changed mid-review.
    """
    url = f"{_API_BASE}/repos/{repo}/pulls/{pr_number}/reviews"
    payload = {
        "commit_id": head_sha,
        "body": format_review_body(review_result),
        "event": _github_event_for(review_result.recommendation),
    }

    try:
        if client is not None:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        else:
            async with httpx.AsyncClient() as owned_client:
                response = await owned_client.post(url, headers=headers, json=payload)
                response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise GitHubPublisherError(
            f"GitHub API returned {exc.response.status_code} while publishing "
            f"a review for PR #{pr_number} in {repo}"
        ) from exc
    except httpx.RequestError as exc:
        raise GitHubPublisherError(
            f"Network error while publishing a review for PR #{pr_number} in {repo}: {exc}"
        ) from exc

    return _extract_review_id(response, repo, pr_number)


async def _update_existing_review(
    repo: str,
    pr_number: int,
    review_id: int,
    review_result: ReviewResult,
    headers: dict[str, str],
    client: httpx.AsyncClient | None = None,
) -> int:
    """Update an existing GitHub PR Review's body in place via PUT and return its ID.

    GitHub's update endpoint accepts only `body` - there is no `event`
    field, since a submitted review's approve/request-changes decision
    cannot be changed after the fact. The caller already confirmed the
    recommendation is unchanged before passing a review_id in at all.
    """
    url = f"{_API_BASE}/repos/{repo}/pulls/{pr_number}/reviews/{review_id}"
    payload = {"body": format_review_body(review_result)}

    try:
        if client is not None:
            response = await client.put(url, headers=headers, json=payload)
            response.raise_for_status()
        else:
            async with httpx.AsyncClient() as owned_client:
                response = await owned_client.put(url, headers=headers, json=payload)
                response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise _ExistingReviewNotFoundError() from exc
        raise GitHubPublisherError(
            f"GitHub API returned {exc.response.status_code} while updating "
            f"review {review_id} for PR #{pr_number} in {repo}"
        ) from exc
    except httpx.RequestError as exc:
        raise GitHubPublisherError(
            f"Network error while updating review {review_id} for PR #{pr_number} in {repo}: {exc}"
        ) from exc

    return _extract_review_id(response, repo, pr_number)


def _extract_review_id(response: httpx.Response, repo: str, pr_number: int) -> int:
    """Parse and validate the review ID from a GitHub Review API response."""
    try:
        review_id = response.json()["id"]
        if not isinstance(review_id, int) or isinstance(review_id, bool):
            raise ValueError("review id is not an integer")
        return review_id
    except (KeyError, TypeError, ValueError) as exc:
        raise GitHubPublisherError(
            f"Unexpected response shape while publishing a review for PR #{pr_number} in {repo}: {exc}"
        ) from exc
