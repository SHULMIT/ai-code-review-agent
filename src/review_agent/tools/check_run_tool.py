"""Thin client for the GitHub Checks API - the on-demand review trigger surface.

Kept separate from github_tool.py/github_publisher.py: those manage PR data
and the published PR Review respectively, while this module only manages the
"DataNet Code Review" Check Run used to start and report on a run.
"""

import httpx

_API_BASE = "https://api.github.com"

CHECK_RUN_NAME = "DataNet Code Review"
# identifier must match event_router.py's _RUN_CODE_REVIEW_ACTION_IDENTIFIER.
RUN_CODE_REVIEW_ACTION = {
    "label": "Run Code Review",
    "description": "Run the DataNet AI code review",
    "identifier": "run_code_review",
}


class CheckRunToolError(Exception):
    """Raised when creating or updating the DataNet Code Review Check Run fails."""


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def _request(
    method: str,
    url: str,
    token: str,
    json_body: dict,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Send one Checks API request and return its parsed JSON body.

    Every failure - timeout, connection error, a non-2xx response, or an
    unparseable body - is wrapped in CheckRunToolError; no raw httpx
    exception ever escapes this function.
    """
    try:
        if client is not None:
            response = await client.request(method, url, headers=_headers(token), json=json_body)
            response.raise_for_status()
        else:
            async with httpx.AsyncClient() as owned_client:
                response = await owned_client.request(
                    method, url, headers=_headers(token), json=json_body
                )
                response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise CheckRunToolError(
            f"GitHub API returned {exc.response.status_code} for {method} {url}"
        ) from exc
    except httpx.RequestError as exc:
        raise CheckRunToolError(f"Network error calling {method} {url}: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise CheckRunToolError(f"Unexpected response shape from {method} {url}: {exc}") from exc


async def find_check_run_for_sha(
    repo: str, head_sha: str, token: str, client: httpx.AsyncClient | None = None
) -> int | None:
    """Return the existing DataNet Code Review Check Run's ID for this exact SHA, or None.

    A Check Run's head_sha cannot be changed after creation, so this only
    ever finds a Check Run already created for head_sha itself - never one
    for a different commit on the same PR.
    """
    url = f"{_API_BASE}/repos/{repo}/commits/{head_sha}/check-runs"
    params = {"check_name": CHECK_RUN_NAME}
    try:
        if client is not None:
            response = await client.get(url, headers=_headers(token), params=params)
            response.raise_for_status()
        else:
            async with httpx.AsyncClient() as owned_client:
                response = await owned_client.get(url, headers=_headers(token), params=params)
                response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise CheckRunToolError(
            f"GitHub API returned {exc.response.status_code} while listing Check Runs for {head_sha}"
        ) from exc
    except httpx.RequestError as exc:
        raise CheckRunToolError(
            f"Network error while listing Check Runs for {head_sha}: {exc}"
        ) from exc

    try:
        check_runs = response.json()["check_runs"]
        if not check_runs:
            return None
        check_run_id = check_runs[0]["id"]
        if not isinstance(check_run_id, int) or isinstance(check_run_id, bool):
            raise ValueError("check run id is not an integer")
        return check_run_id
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckRunToolError(
            f"Unexpected response shape while listing Check Runs for {head_sha}: {exc}"
        ) from exc


def _pending_check_run_body(head_sha: str | None = None) -> dict:
    """The completed/neutral 'ready to run' body shared by create and reset."""
    body: dict = {
        "name": CHECK_RUN_NAME,
        "status": "completed",
        "conclusion": "neutral",
        "output": {
            "title": "Ready to run",
            "summary": "Click **Run Code Review** below to start an AI review of this pull request.",
        },
        "actions": [RUN_CODE_REVIEW_ACTION],
    }
    if head_sha is not None:
        body["head_sha"] = head_sha
    return body


async def create_pending_check_run(
    repo: str, head_sha: str, token: str, client: httpx.AsyncClient | None = None
) -> int:
    """Create a new completed/neutral Check Run carrying the "Run Code Review" action."""
    url = f"{_API_BASE}/repos/{repo}/check-runs"
    body = await _request("POST", url, token, _pending_check_run_body(head_sha), client)
    return _extract_check_run_id(body, repo)


async def reset_check_run_to_pending(
    repo: str, check_run_id: int, token: str, client: httpx.AsyncClient | None = None
) -> None:
    """Reset an existing Check Run back to the completed/neutral 'ready to run' state."""
    url = f"{_API_BASE}/repos/{repo}/check-runs/{check_run_id}"
    await _request("PATCH", url, token, _pending_check_run_body(), client)


async def create_or_reset_check_run(
    repo: str, head_sha: str, token: str, client: httpx.AsyncClient | None = None
) -> int:
    """Create a Check Run for head_sha, or reset an existing one for that exact SHA.

    A new head_sha (a fresh push) always gets its own new Check Run, never a
    repurposed one from an older commit - a Check Run's head_sha cannot be
    changed after creation. Resetting only applies to a redelivered event for
    a SHA that already has one.
    """
    existing_id = await find_check_run_for_sha(repo, head_sha, token, client)
    if existing_id is not None:
        await reset_check_run_to_pending(repo, existing_id, token, client)
        return existing_id
    return await create_pending_check_run(repo, head_sha, token, client)


async def mark_check_run_in_progress(
    repo: str,
    check_run_id: int,
    token: str,
    client: httpx.AsyncClient | None = None,
    *,
    title: str = "Running Code Review...",
    summary: str = "The DataNet AI code review is running now.",
) -> None:
    """Move a Check Run to in_progress right as its review starts.

    title/summary default to a generic message but are meant to be
    overridden with the immediate "0/6 reviewers completed" progress text
    (see review_formatter.py) - the point of calling this at all, rather
    than leaving the Check Run on "Ready", is that the user must see instant
    confirmation their click was received.
    """
    await update_check_run_progress(repo, check_run_id, token, title, summary, client)


async def update_check_run_progress(
    repo: str,
    check_run_id: int,
    token: str,
    title: str,
    summary: str,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Update an already-in_progress Check Run's visible text without changing its status.

    Used for live reviewer-by-reviewer progress: same PATCH endpoint as
    mark_check_run_in_progress, called again each time a new reviewer
    reports back, so GitHub's spinner keeps showing but the text reflects
    how many of the six reviewers have finished.
    """
    url = f"{_API_BASE}/repos/{repo}/check-runs/{check_run_id}"
    body = {
        "status": "in_progress",
        "output": {"title": title, "summary": summary},
    }
    await _request("PATCH", url, token, body, client)


async def complete_check_run(
    repo: str,
    check_run_id: int,
    conclusion: str,
    title: str,
    summary: str,
    token: str,
    client: httpx.AsyncClient | None = None,
    *,
    allow_rerun: bool = True,
) -> None:
    """Mark a Check Run completed with the given conclusion and output text.

    allow_rerun reattaches the "Run Code Review" action so the same Check Run
    can be triggered again - left off only when re-running this exact Check
    Run would never be useful (a stale, superseded head SHA already has its
    own current Check Run to use instead).
    """
    url = f"{_API_BASE}/repos/{repo}/check-runs/{check_run_id}"
    body: dict = {
        "status": "completed",
        "conclusion": conclusion,
        "output": {"title": title, "summary": summary},
    }
    if allow_rerun:
        body["actions"] = [RUN_CODE_REVIEW_ACTION]
    await _request("PATCH", url, token, body, client)


def _extract_check_run_id(body: dict, repo: str) -> int:
    try:
        check_run_id = body["id"]
        if not isinstance(check_run_id, int) or isinstance(check_run_id, bool):
            raise ValueError("check run id is not an integer")
        return check_run_id
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckRunToolError(
            f"Unexpected response shape while creating a Check Run in {repo}: {exc}"
        ) from exc
