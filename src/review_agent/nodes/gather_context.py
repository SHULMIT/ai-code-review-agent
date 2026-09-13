import re
from pathlib import Path

from review_agent.graph.state import ReviewState
from review_agent.models.review_input import build_review_input
from review_agent.tools.doc_retrieval_tool import DocRetrievalToolError, retrieve_documentation
from review_agent.tools.repo_search_tool import (
    DEFAULT_MAX_REPOSITORY_SEARCH_RESULTS,
    RepoSearchToolError,
    search_repository,
)

_MAX_SEARCH_TERMS = 5
_IMPORT_LINE_RE = re.compile(r"^\+(?:from\s+(\S+)\s+import\s+.+|import\s+(.+))$")
_MIN_SEARCH_TERM_LENGTH = 3
# These imports occur in a disproportionate number of files, so searching
# them mostly contributes incidental context rather than PR-specific evidence.
# The list is deliberately small: it is a noise-control heuristic, not a
# complete taxonomy of unhelpful modules.
_LOW_VALUE_IMPORT_TERMS = {"os", "re", "sys", "io", "json", "time", "typing"}


def _extract_import_terms(diff: str) -> list[str]:
    """Pull Python import targets out of added diff lines, deduped, capped at 5."""
    terms: list[str] = []
    for line in (diff or "").splitlines():
        match = _IMPORT_LINE_RE.match(line)
        if not match:
            continue
        if match.group(1):
            terms.append(match.group(1))
        else:
            for name in match.group(2).split(","):
                name = name.strip().split(" as ")[0].strip()
                if name:
                    terms.append(name)

    deduped: list[str] = []
    for term in terms:
        root_term = term.split(".", 1)[0].lower()
        if len(root_term) < _MIN_SEARCH_TERM_LENGTH or root_term in _LOW_VALUE_IMPORT_TERMS:
            continue
        if term not in deduped:
            deduped.append(term)
        if len(deduped) >= _MAX_SEARCH_TERMS:
            break
    return deduped


def _partial_repository_context_notes(
    *,
    truncated: bool,
    scanned_file_count: int,
    total_file_count: int,
    results_capped: bool,
    skipped_large_files: list[str],
    skipped_binary_files: list[str],
) -> list[str]:
    """Describe repository-search limitations as recorded limitations, not failures."""
    notes: list[str] = []
    if truncated:
        notes.append(
            "repository search truncated: scanned "
            f"{scanned_file_count}/{total_file_count} files due to configured limit"
        )
    if results_capped:
        notes.append("repository search results capped due to configured limit")
    if skipped_large_files:
        notes.append(
            f"repository search skipped {len(skipped_large_files)} file(s) exceeding the size limit"
        )
    if skipped_binary_files:
        notes.append(f"repository search skipped {len(skipped_binary_files)} binary file(s)")
    return notes


# LangGraph nodes must accept only `state` - external dependencies like the token and docs path are bound here via closure before the node is registered.
def make_gather_context_node(
    token: str,
    local_docs_root: Path,
    max_repository_context_characters: int | None = None,
    excluded_diff_paths: tuple[str, ...] = (),
):
    """Build the gather_context node with the GitHub token and docs path bound via closure."""

    async def gather_context_node(state: ReviewState) -> dict:
        """Search the repo for each import term in the diff, and the local docs for the PR's components.

        Repository search and documentation retrieval are independent: a
        failure in one never blocks or discards the other's results. All
        import terms are searched against one shared clone of the repo -
        search_repository() takes every term in one call and clones
        state.head_sha exactly once per call to this node, not once per
        term, since every term targets the same commit. Search results
        across all import terms share one combined
        max_repository_search_results budget; once it is spent, remaining
        terms are skipped rather than searched. A search bounded by any
        configured limit (timeout, scan/file/result caps, binary skipping)
        is recorded as a partial-context note in errors, not a failure. A
        failure cloning the repo itself is recorded the same way and skips
        every term - there is nothing left to search without it.
        Finally, builds review_input from this node's own fresh context
        (not the pre-node state), so it reflects what this call collected.
        """
        errors: list[str] = []

        repo_results: list[dict] = []
        truncated = False
        results_capped = False
        scanned_file_count = 0
        total_file_count = 0
        skipped_large_files: list[str] = []
        skipped_binary_files: list[str] = []

        # One search for every term, not one search per term. Each term used
        # to trigger its own shallow clone of the same repo at the same
        # commit - up to _MAX_SEARCH_TERMS network fetches and full-tree
        # scans per review run, all fetching identical content. The matches
        # are the same either way; only the work is different.
        #
        # One consequence worth naming: results now arrive in file order with
        # terms interleaved, where before they arrived grouped by term. That
        # changes which results survive when max_results caps the list - not
        # which lines match.
        search_terms = _extract_import_terms(state.diff or "")
        if search_terms:
            try:
                outcome = await search_repository(
                    state.repo,
                    state.head_sha,
                    search_terms,
                    token,
                    max_results=DEFAULT_MAX_REPOSITORY_SEARCH_RESULTS,
                )
            except RepoSearchToolError as exc:
                errors.append(f"repository search failed: {exc}")
            else:
                repo_results.extend(r.model_dump() for r in outcome.results)
                truncated = outcome.truncated
                results_capped = outcome.results_capped
                skipped_large_files.extend(outcome.skipped_large_files)
                skipped_binary_files.extend(outcome.skipped_binary_files)
                if outcome.truncated:
                    scanned_file_count = outcome.scanned_file_count
                    total_file_count = outcome.total_file_count
                errors.append(
                    "repository search succeeded: "
                    f"{len(outcome.results)} result(s) for {len(search_terms)} term(s)"
                )
        else:
            errors.append("repository search not attempted: no import terms found in diff")

        errors.extend(
            _partial_repository_context_notes(
                truncated=truncated,
                scanned_file_count=scanned_file_count,
                total_file_count=total_file_count,
                results_capped=results_capped,
                skipped_large_files=skipped_large_files,
                skipped_binary_files=skipped_binary_files,
            )
        )

        doc_results = []
        try:
            docs = retrieve_documentation(state.components or [], local_docs_root)
            doc_results = [d.model_dump() for d in docs]
        except DocRetrievalToolError as exc:
            errors.append(str(exc))

        result: dict = {
            "repository_context": {"results": repo_results},
            "documentation_context": {"results": doc_results},
        }
        if errors:
            result["errors"] = errors

        updated_state = state.model_copy(
            update={
                "repository_context": result["repository_context"],
                "documentation_context": result["documentation_context"],
                "errors": state.errors + errors,
            }
        )
        build_kwargs = {}
        if max_repository_context_characters is not None:
            build_kwargs["max_repository_context_characters"] = max_repository_context_characters
        if excluded_diff_paths:
            build_kwargs["excluded_diff_paths"] = excluded_diff_paths
        result["review_input"] = build_review_input(updated_state, **build_kwargs)
        return result

    return gather_context_node
