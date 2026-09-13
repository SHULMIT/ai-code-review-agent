from pathlib import Path

from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component
from review_agent.models.review_input import ReviewInput
from review_agent.nodes import gather_context as gather_context_module
from review_agent.nodes.gather_context import make_gather_context_node
from review_agent.tools.doc_retrieval_tool import DocRetrievalToolError, DocumentChunk
from review_agent.tools.repo_search_tool import RepoSearchOutcome, RepoSearchToolError, SearchResult


def _outcome(
    results: list[SearchResult] | None = None,
    *,
    total_file_count: int = 1,
    scanned_file_count: int = 1,
    truncated: bool = False,
    results_capped: bool = False,
    skipped_large_files: list[str] | None = None,
    skipped_binary_files: list[str] | None = None,
) -> RepoSearchOutcome:
    """Build a RepoSearchOutcome with the fields a "nothing was limited" call would return."""

    return RepoSearchOutcome(
        results=results or [],
        total_file_count=total_file_count,
        scanned_file_count=scanned_file_count,
        truncated=truncated,
        results_capped=results_capped,
        skipped_large_files=skipped_large_files or [],
        skipped_binary_files=skipped_binary_files or [],
    )


async def test_gather_context_node_success(monkeypatch):
    async def fake_search_repository(repo, head_sha, query, token, **kwargs):
        return _outcome([SearchResult(file="a.py", line=1, snippet=f"match for {query}")])

    def fake_retrieve_documentation(components, docs_root):
        return [DocumentChunk(source_path="docs/x.md", heading="Architecture", snippet="...")]

    monkeypatch.setattr(gather_context_module, "search_repository", fake_search_repository)
    monkeypatch.setattr(gather_context_module, "retrieve_documentation", fake_retrieve_documentation)

    node = make_gather_context_node(token="fake-token", local_docs_root=Path("docs"))
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff="+from project.client import Client\n+import os\n",
    )

    result = await node(state)

    assert len(result["repository_context"]["results"]) == 1
    assert len(result["documentation_context"]["results"]) == 1
    assert "repository search succeeded: 1 result(s) for 1 term(s)" in result["errors"]


async def test_gather_context_node_no_import_terms_skips_repo_search(monkeypatch):
    called = {"count": 0}

    async def fake_search_repository(*args, **kwargs):
        called["count"] += 1
        return _outcome()

    def fake_retrieve_documentation(components, docs_root):
        return []

    monkeypatch.setattr(gather_context_module, "search_repository", fake_search_repository)
    monkeypatch.setattr(gather_context_module, "retrieve_documentation", fake_retrieve_documentation)

    node = make_gather_context_node(token="fake-token", local_docs_root=Path("docs"))
    state = ReviewState(
        repo="example-org/example-repo", pr_number=1, head_sha="abc123", diff="no imports here\n"
    )

    result = await node(state)

    assert called["count"] == 0
    assert result["repository_context"]["results"] == []
    assert "repository search not attempted: no import terms found in diff" in result["errors"]


def test_extract_import_terms_skips_common_or_too_short_modules():
    """Generic stdlib imports should not turn into noisy repository searches."""

    terms = gather_context_module._extract_import_terms(
        "+import os, re, sys, io, json, time\n"
        "+from typing import Optional\n"
        "+from project.client import Client\n"
    )

    assert terms == ["project.client"]


async def test_gather_context_node_tool_error_lands_in_errors(monkeypatch):
    async def failing_search_repository(*args, **kwargs):
        raise RepoSearchToolError("clone failed")

    def failing_retrieve_documentation(components, docs_root):
        raise DocRetrievalToolError("disk exploded")

    monkeypatch.setattr(gather_context_module, "search_repository", failing_search_repository)
    monkeypatch.setattr(gather_context_module, "retrieve_documentation", failing_retrieve_documentation)

    node = make_gather_context_node(token="fake-token", local_docs_root=Path("docs"))
    state = ReviewState(
        repo="example-org/example-repo", pr_number=1, head_sha="abc123", diff="+import project.client\n"
    )

    result = await node(state)

    assert result["repository_context"]["results"] == []
    assert result["documentation_context"]["results"] == []
    assert len(result["errors"]) == 2


async def test_gather_context_node_records_a_truncated_scan_as_a_partial_context_note(monkeypatch):
    async def fake_search_repository(repo, head_sha, query, token, **kwargs):
        return _outcome(
            [SearchResult(file="a.py", line=1, snippet="match")],
            total_file_count=1200,
            scanned_file_count=500,
            truncated=True,
        )

    def fake_retrieve_documentation(components, docs_root):
        return []

    monkeypatch.setattr(gather_context_module, "search_repository", fake_search_repository)
    monkeypatch.setattr(gather_context_module, "retrieve_documentation", fake_retrieve_documentation)

    node = make_gather_context_node(token="fake-token", local_docs_root=Path("docs"))
    state = ReviewState(
        repo="example-org/example-repo", pr_number=1, head_sha="abc123", diff="+import project.client\n"
    )

    result = await node(state)

    assert result["repository_context"]["results"] != []
    assert "errors" in result
    assert "repository search truncated: scanned 500/1200 files due to configured limit" in result["errors"]


async def test_gather_context_node_records_skipped_large_and_binary_files(monkeypatch):
    async def fake_search_repository(repo, head_sha, query, token, **kwargs):
        return _outcome(
            skipped_large_files=["vendor/bundle.js"],
            skipped_binary_files=["assets/logo.png"],
        )

    def fake_retrieve_documentation(components, docs_root):
        return []

    monkeypatch.setattr(gather_context_module, "search_repository", fake_search_repository)
    monkeypatch.setattr(gather_context_module, "retrieve_documentation", fake_retrieve_documentation)

    node = make_gather_context_node(token="fake-token", local_docs_root=Path("docs"))
    state = ReviewState(
        repo="example-org/example-repo", pr_number=1, head_sha="abc123", diff="+import project.client\n"
    )

    result = await node(state)

    assert "repository search skipped 1 file(s) exceeding the size limit" in result["errors"]
    assert "repository search skipped 1 binary file(s)" in result["errors"]


async def test_gather_context_node_caps_combined_results_across_search_terms(monkeypatch):
    """The result budget is shared across import terms, not reset for each one."""

    calls: list[int] = []

    async def fake_search_repository(repo, head_sha, query, token, *, max_results, **kwargs):
        calls.append(max_results)
        # Each call reports exhausting whatever budget it was given.
        results = [
            SearchResult(file=f"{query}.py", line=index + 1, snippet="match")
            for index in range(max_results)
        ]
        return _outcome(results, results_capped=True)

    def fake_retrieve_documentation(components, docs_root):
        return []

    monkeypatch.setattr(gather_context_module, "search_repository", fake_search_repository)
    monkeypatch.setattr(gather_context_module, "retrieve_documentation", fake_retrieve_documentation)
    monkeypatch.setattr(gather_context_module, "DEFAULT_MAX_REPOSITORY_SEARCH_RESULTS", 5)

    node = make_gather_context_node(token="fake-token", local_docs_root=Path("docs"))
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff="+import alpha\n+import beta\n+import gamma\n",
    )

    result = await node(state)

    # The first call gets the full budget (5); once it reports capped and the
    # budget is spent, later terms are skipped entirely rather than searched.
    assert calls == [5]
    assert len(result["repository_context"]["results"]) == 5
    assert "repository search results capped due to configured limit" in result["errors"]


async def test_gather_context_node_populates_review_input(monkeypatch):
    async def fake_search_repository(repo, head_sha, query, token, **kwargs):
        return _outcome([SearchResult(file="a.py", line=1, snippet="match")])

    def fake_retrieve_documentation(components, docs_root):
        return [DocumentChunk(source_path="docs/x.md", heading="Architecture", snippet="...")]

    monkeypatch.setattr(gather_context_module, "search_repository", fake_search_repository)
    monkeypatch.setattr(gather_context_module, "retrieve_documentation", fake_retrieve_documentation)

    node = make_gather_context_node(token="fake-token", local_docs_root=Path("docs"))
    diff = (
        "diff --git a/a.py b/a.py\n"
        "index 0000000..1111111 100644\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1,2 @@\n"
        "+import project.client\n"
    )
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff=diff,
        changed_files=["a.py"],
        components=[Component.AGENT_HARNESS],
    )

    result = await node(state)

    review_input = result["review_input"]
    assert isinstance(review_input, ReviewInput)
    assert review_input.repo == "example-org/example-repo"
    assert review_input.pr_number == 1
    assert review_input.head_sha == "abc123"
    assert review_input.included_files == ["a.py"]
    assert review_input.diff_truncated is False
    assert len(review_input.repository_context["results"]) == 1
    assert len(review_input.documentation_context) == 1
    assert review_input.documentation_context[0].source_path == "docs/x.md"
    assert review_input.coverage_notes == ["repository search succeeded: 1 result(s) for 1 term(s)"]
