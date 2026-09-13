from typing import Optional

import httpx
from pydantic import BaseModel, ValidationError

_API_BASE = "https://api.github.com"
_FILES_PAGE_SIZE = 100


class PRData(BaseModel):
    head_sha: str
    diff: str
    changed_files: list[str]
    title: str
    description: Optional[str] = None


class GitHubToolError(Exception):
    """Raised when fetching PR data from the GitHub API fails, for any reason."""


def _github_headers(token: str) -> dict[str, str]:
    """Build the GitHub API headers shared by pull-request requests."""

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def get_pr_head_sha(
    repo: str, pr_number: int, token: str, client: httpx.AsyncClient | None = None
) -> str:
    """Fetch the PR's current head SHA for a pre-publish consistency check.

    client lets a caller pass in one pooled httpx.AsyncClient shared across
    an entire review run instead of opening a fresh TCP connection per call
    - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5. When omitted, this
    function falls back to its original self-contained per-call client.
    """

    url = f"{_API_BASE}/repos/{repo}/pulls/{pr_number}"
    try:
        if client is not None:
            response = await client.get(url, headers=_github_headers(token))
            response.raise_for_status()
        else:
            async with httpx.AsyncClient() as owned_client:
                response = await owned_client.get(url, headers=_github_headers(token))
                response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise GitHubToolError(
            f"GitHub API returned {exc.response.status_code} for PR #{pr_number} in {repo}"
        ) from exc
    except httpx.RequestError as exc:
        raise GitHubToolError(
            f"Network error while fetching PR #{pr_number} in {repo}: {exc}"
        ) from exc

    try:
        head_sha = response.json()["head"]["sha"]
        if not isinstance(head_sha, str) or not head_sha:
            raise ValueError("head SHA is missing or invalid")
        return head_sha
    except (KeyError, TypeError, ValueError) as exc:
        raise GitHubToolError(
            f"Unexpected response shape while fetching PR #{pr_number} in {repo}: {exc}"
        ) from exc


async def get_pr(
    repo: str, pr_number: int, token: str, client: httpx.AsyncClient | None = None
) -> PRData:
    """Fetch a pull request using a GitHub App installation access token.

    Raises GitHubToolError on any request failure or unexpected response shape;
    never lets a raw httpx exception escape.

    client lets a caller pass in one pooled httpx.AsyncClient shared across
    an entire review run instead of opening a fresh TCP connection per call
    - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5. When omitted, this
    function falls back to its original self-contained per-call client.
    """
    base_headers = _github_headers(token)
    pr_url = f"{_API_BASE}/repos/{repo}/pulls/{pr_number}"

    async def _fetch(http_client: httpx.AsyncClient) -> tuple[httpx.Response, httpx.Response, list[object]]:
        metadata_response = await http_client.get(pr_url, headers=base_headers)
        metadata_response.raise_for_status()

        diff_response = await http_client.get(
            pr_url,
            headers={**base_headers, "Accept": "application/vnd.github.v3.diff"},
        )
        diff_response.raise_for_status()

        files: list[object] = []
        page = 1
        while True:
            files_response = await http_client.get(
                f"{pr_url}/files",
                headers=base_headers,
                params={"per_page": _FILES_PAGE_SIZE, "page": page},
            )
            files_response.raise_for_status()
            page_files = files_response.json()
            if not isinstance(page_files, list):
                raise GitHubToolError(
                    f"Unexpected response shape while fetching PR #{pr_number} in {repo}: "
                    "files response is not a list"
                )
            files.extend(page_files)
            if len(page_files) < _FILES_PAGE_SIZE:
                break
            page += 1
        return metadata_response, diff_response, files

    try:
        if client is not None:
            metadata_response, diff_response, files = await _fetch(client)
        else:
            async with httpx.AsyncClient() as owned_client:
                metadata_response, diff_response, files = await _fetch(owned_client)
    except httpx.HTTPStatusError as exc:
        raise GitHubToolError(
            f"GitHub API returned {exc.response.status_code} for PR #{pr_number} in {repo}"
        ) from exc
    except httpx.RequestError as exc:
        raise GitHubToolError(
            f"Network error while fetching PR #{pr_number} in {repo}: {exc}"
        ) from exc

    try:
        metadata = metadata_response.json()
        return PRData(
            head_sha=metadata["head"]["sha"],
            diff=diff_response.text,
            changed_files=[f["filename"] for f in files],
            title=metadata["title"],
            description=metadata.get("body"),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise GitHubToolError(
            f"Unexpected response shape while fetching PR #{pr_number} in {repo}: {exc}"
        ) from exc
