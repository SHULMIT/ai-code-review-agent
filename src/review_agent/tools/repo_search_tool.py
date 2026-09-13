"""Repository search tool: shallow-clones a PR's exact commit and searches it for a query string."""

import asyncio
import base64
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

_SNIPPET_MAX_CHARS = 150
_BINARY_SNIFF_BYTES = 8_000

# Mirror Settings' defaults in config.py so search_repository() is safe to
# call without threading a Settings instance through every node/graph layer.
# Keep these in sync with config.py if those defaults ever change.
DEFAULT_GIT_COMMAND_TIMEOUT_SECONDS = 60
DEFAULT_MAX_REPOSITORY_SCAN_FILES = 10_000
DEFAULT_MAX_REPOSITORY_FILE_BYTES = 1_000_000
DEFAULT_MAX_REPOSITORY_SEARCH_RESULTS = 500


class SearchResult(BaseModel):
    file: str
    line: int
    snippet: str


class RepoSearchOutcome(BaseModel):
    """One search_repository() call's matches plus how complete the scan was.

    truncated and results_capped signal a partial-but-successful search, not
    a failure: a caller can still use `results` while recording the
    limitation instead of assuming the context is complete.
    """

    results: list[SearchResult]
    total_file_count: int
    scanned_file_count: int
    truncated: bool
    results_capped: bool
    skipped_large_files: list[str]
    skipped_binary_files: list[str]


class RepoSearchToolError(Exception):
    """Raised when shallow-cloning or searching a repository fails, or a Git command times out, for any reason."""


@asynccontextmanager
async def cloned_repository(
    repo: str,
    head_sha: str,
    token: str,
    *,
    timeout_seconds: float = DEFAULT_GIT_COMMAND_TIMEOUT_SECONDS,
) -> AsyncIterator[Path]:
    """Shallow-clone `repo` at `head_sha` into a temp dir, yielding its path to search.

    The clone step of search_repository(), split out so a caller searching
    for more than one query against the same commit (e.g. gather_context_node
    searching once per import term) can reuse a single clone instead of
    re-cloning per query - cloning is a real network fetch, not free. Use
    search_cloned_repository() one or more times inside this context.
    Cleans up the temp dir exactly once, when the context exits, regardless
    of how many searches ran inside it or whether the clone itself failed.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="review_agent_repo_search_"))
    try:
        await _shallow_clone(repo, head_sha, token, tmp_dir, timeout_seconds)
        yield tmp_dir
    except RepoSearchToolError:
        raise
    except Exception as exc:
        raise RepoSearchToolError(f"repo search failed for {repo}@{head_sha[:12]}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def search_cloned_repository(
    root: Path,
    query: str | Sequence[str],
    *,
    max_scan_files: int = DEFAULT_MAX_REPOSITORY_SCAN_FILES,
    max_file_bytes: int = DEFAULT_MAX_REPOSITORY_FILE_BYTES,
    max_results: int = DEFAULT_MAX_REPOSITORY_SEARCH_RESULTS,
) -> RepoSearchOutcome:
    """Search an already-cloned checkout (from cloned_repository()) for `query`.

    The search step of search_repository(), usable directly against a
    shared clone. Bounded exactly as search_repository() documents: at most
    max_scan_files files are scanned, any file over max_file_bytes or
    detected as binary is skipped without being read, and collection stops
    at max_results matches.
    """
    queries = [query] if isinstance(query, str) else query
    return _search_text_files(root, queries, max_scan_files, max_file_bytes, max_results)


async def search_repository(
    repo: str,
    head_sha: str,
    queries: Sequence[str],
    token: str,
    *,
    timeout_seconds: float = DEFAULT_GIT_COMMAND_TIMEOUT_SECONDS,
    max_scan_files: int = DEFAULT_MAX_REPOSITORY_SCAN_FILES,
    max_file_bytes: int = DEFAULT_MAX_REPOSITORY_FILE_BYTES,
    max_results: int = DEFAULT_MAX_REPOSITORY_SEARCH_RESULTS,
) -> RepoSearchOutcome:
    """Shallow-clone `repo` at `head_sha` into a temp dir and search its text files for every term in `queries`.

    Takes all terms at once, deliberately. gather_context extracts several
    import names from one diff, and every one of them targets the same repo
    at the same commit - so searching them one call at a time re-cloned the
    identical commit once per term (four Git subprocesses and a real network
    fetch each) and re-read every file in the repository once per term. One
    clone and one pass per file produce exactly the same matches.

    Bounded so a large or hostile PR cannot block the agent indefinitely:
    every Git command is killed after timeout_seconds, at most max_scan_files
    files are scanned, any file over max_file_bytes or detected as binary is
    skipped without being read, and collection stops at max_results matches.
    The temp dir is always removed afterward, whether the search succeeds,
    times out, or fails.

    An empty `queries` returns an empty outcome without cloning: there is
    nothing to look for, so there is no reason to pay for a network fetch.
    """
    if not queries:
        return RepoSearchOutcome(
            results=[],
            total_file_count=0,
            scanned_file_count=0,
            truncated=False,
            results_capped=False,
            skipped_large_files=[],
            skipped_binary_files=[],
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="review_agent_repo_search_"))
    try:
        await _shallow_clone(repo, head_sha, token, tmp_dir, timeout_seconds)
        return _search_text_files(tmp_dir, queries, max_scan_files, max_file_bytes, max_results)
    except RepoSearchToolError:
        raise
    except Exception as exc:
        raise RepoSearchToolError(f"repo search failed for {repo}@{head_sha[:12]}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _git_environment(auth_header: str | None = None) -> dict[str, str]:
    """Build an isolated Git environment without user hooks, filters, or config."""
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("GIT_CONFIG_"):
            del environment[key]

    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull

    if auth_header is not None:
        environment["GIT_CONFIG_COUNT"] = "1"
        environment["GIT_CONFIG_KEY_0"] = "http.extraHeader"
        environment["GIT_CONFIG_VALUE_0"] = auth_header

    return environment


async def _run_git(
    args: list[str], cwd: Path, timeout_seconds: float, auth_header: str | None = None
) -> None:
    """Run a Git command with authentication kept out of its argument list.

    Kills the process and raises RepoSearchToolError if it runs longer than
    timeout_seconds, so an unresponsive Git host cannot hang a review run.
    """
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        env=_git_environment(auth_header),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise RepoSearchToolError(f"git {args[0]} timed out after {timeout_seconds}s") from exc

    if process.returncode != 0:
        # Deliberately no stderr/argv in this message - the auth header lives
        # in the git invocation and must never end up in a log or exception.
        raise RepoSearchToolError(f"git {args[0]} failed with exit code {process.returncode}")


async def _shallow_clone(
    repo: str, head_sha: str, token: str, dest: Path, timeout_seconds: float
) -> None:
    """Init a repo at `dest` and fetch only `head_sha` at depth 1 - no full clone."""
    url = f"https://github.com/{repo}.git"
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    auth_header = f"AUTHORIZATION: basic {basic}"

    await _run_git(["init"], dest, timeout_seconds)
    await _run_git(["remote", "add", "origin", url], dest, timeout_seconds)
    await _run_git(["fetch", "--depth", "1", "origin", head_sha], dest, timeout_seconds, auth_header)
    await _run_git(["checkout", "--detach", "FETCH_HEAD"], dest, timeout_seconds)


def _is_binary_content(sample: bytes) -> bool:
    """Heuristically detect binary content via a null byte in a leading sample.

    Cheap and dependency-free: text source files never contain null bytes,
    while most binary formats (images, archives, compiled artifacts) do
    within their first few KB. Replaces the previous read_text(errors=
    "ignore"), which silently mangled binary content into bogus matches
    instead of skipping it.
    """
    return b"\x00" in sample


def _search_text_files(
    root: Path,
    queries: Sequence[str],
    max_scan_files: int,
    max_file_bytes: int,
    max_results: int,
) -> RepoSearchOutcome:
    """Search text files under root for any term in queries, bounded by the configured limits.

    One pattern is built per term and every line is tested against all of
    them in a single pass, so each file is opened and read exactly once no
    matter how many terms there are. A line matching more than one term
    still yields a single result - the caller only needs to know the line
    is relevant, not which term made it so.
    """
    query_patterns = [re.compile(rf"\b{re.escape(query)}\b", re.IGNORECASE) for query in queries]
    candidate_files = [
        path for path in sorted(root.rglob("*")) if path.is_file() and ".git" not in path.parts
    ]
    files_to_scan = candidate_files[:max_scan_files]

    results: list[SearchResult] = []
    results_capped = False
    skipped_large_files: list[str] = []
    skipped_binary_files: list[str] = []

    for path in files_to_scan:
        relative_path = path.relative_to(root).as_posix()

        try:
            file_size = path.stat().st_size
        except OSError:
            continue
        if file_size > max_file_bytes:
            skipped_large_files.append(relative_path)
            continue

        try:
            with path.open("rb") as binary_probe:
                sample = binary_probe.read(_BINARY_SNIFF_BYTES)
        except OSError:
            continue
        if _is_binary_content(sample):
            skipped_binary_files.append(relative_path)
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in query_patterns):
                if len(results) >= max_results:
                    results_capped = True
                    break
                results.append(
                    SearchResult(
                        file=relative_path,
                        line=line_no,
                        snippet=line.strip()[:_SNIPPET_MAX_CHARS],
                    )
                )
        if results_capped:
            break

    return RepoSearchOutcome(
        results=results,
        total_file_count=len(candidate_files),
        scanned_file_count=len(files_to_scan),
        truncated=len(candidate_files) > max_scan_files,
        results_capped=results_capped,
        skipped_large_files=skipped_large_files,
        skipped_binary_files=skipped_binary_files,
    )
