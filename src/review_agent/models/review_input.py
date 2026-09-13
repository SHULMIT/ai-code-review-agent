"""ReviewInput: the typed, budgeted object that carries a PR's diff, changed
files, repository/documentation context, and known coverage gaps into the
review nodes. Reviewer nodes should read only from this object, not raw
ReviewState fields, so every fact they see is explicitly grounded instead of
assumed complete.
"""

import fnmatch
import re
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from review_agent.models.finding import Component
from review_agent.tools.doc_retrieval_tool import DocumentChunk

if TYPE_CHECKING:
    # Only needed for the type hint below; a real import would create a
    # cycle, since graph/state.py imports ReviewInput as a field type.
    from review_agent.graph.state import ReviewState

_DIFF_FILE_HEADER_RE = re.compile(r"^diff --git a/.+? b/(.+)$")
_REPOSITORY_COVERAGE_NOTE_PREFIX = "repository search "

# Mirrors Settings.max_diff_characters' default in config.py - see that
# default's comment for why 45_000/15_000 are sized against
# max_model_input_tokens together. Kept as a local constant (like
# repo_search_tool.py's DEFAULT_* values) so build_review_input() stays a
# pure function of ReviewState instead of reading environment configuration
# itself; keep in sync if config.py's default ever changes.
DEFAULT_MAX_DIFF_CHARACTERS = 45_000
# Repository search results are supplementary evidence, not the primary diff.
DEFAULT_MAX_REPOSITORY_CONTEXT_CHARACTERS = 15_000

# Generated, locked, or built artifacts are noise for every reviewed
# repository, not a property of one deployment's config - excluded from the
# diff budget unconditionally, in addition to whatever excluded_diff_paths a
# deployment configures via Settings. A lockfile especially can be large
# enough to consume the entire diff budget by itself and, in changed_files
# order, silently omit every real source file that would otherwise follow it.
_DEFAULT_EXCLUDED_GENERATED_PATTERNS = (
    "*.lock",
    "*package-lock.json",
    "*yarn.lock",
    "*pnpm-lock.yaml",
    "*go.sum",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*/dist/*",
    "*/build/*",
    "*/node_modules/*",
)


class ReviewInput(BaseModel):
    """A budgeted snapshot of one PR review's diff, context, and known coverage gaps.

    Every field here is already within budget. repository_context holds the
    search results that fit max_repository_context_characters, not the full
    set with a note about them - build_review_input used to select the
    subset, use it only to phrase a coverage note, discard it, and store the
    unbudgeted list, leaving prompt_builder to re-derive the identical
    selection when rendering each of the six reviewer prompts.

    That was worth fixing for correctness rather than for the six repeated
    selections: the object claimed a budget had been applied while carrying
    results that had not been filtered, and what actually reached a prompt
    was decided somewhere else entirely. The two agreed only because both
    called the same function with the same arguments - a coincidence held up
    by discipline, not by structure. A budgeted object now contains what it
    says it contains.
    """

    repo: str
    pr_number: int
    head_sha: str
    included_diff: str
    diff_truncated: bool
    included_files: list[str]
    omitted_files: list[str]
    repository_context: dict
    documentation_context: list[DocumentChunk]
    coverage_notes: list[str]
    max_repository_context_characters: int = DEFAULT_MAX_REPOSITORY_CONTEXT_CHARACTERS


def build_review_input(
    state: "ReviewState",
    max_diff_characters: int = DEFAULT_MAX_DIFF_CHARACTERS,
    max_repository_context_characters: int = DEFAULT_MAX_REPOSITORY_CONTEXT_CHARACTERS,
    excluded_diff_paths: tuple[str, ...] = (),
) -> ReviewInput:
    """Derive a budgeted ReviewInput from a ReviewState snapshot.

    Applies max_diff_characters as a deterministic budget over
    state.changed_files in GitHub's returned order: each file's diff hunk is
    included whole or omitted whole, never truncated mid-hunk. Repository
    search context is rendered up to its own whole-result character budget;
    the original results remain in the typed input, while coverage_notes
    makes truncation explicit. Documentation context is carried through.
    coverage_notes
    consolidates every partial/missing-context signal already known - Task
    10's "repository search ..." notes already recorded in state.errors, plus
    a documentation-coverage note derived directly from state.components and
    state.documentation_context, since an empty documentation result is a
    legitimate outcome that never produces an error to extract.

    excluded_diff_paths (glob patterns, e.g. "all_classes_code.txt") removes
    a matching file's hunk from the diff entirely before the budget is ever
    applied - for a repository-specific generated/export dump that re-embeds
    a second, differently-numbered copy of other changed files' source
    inside its own diff, confusing a reviewer about which copy and which
    line numbers are real. Configured per deployment (Settings.excluded_diff_paths),
    never hardcoded here, since which files are "generated noise" is a
    property of the target repository, not of this service.
    """
    included_diff, diff_truncated, included_files, omitted_files = _select_diff_within_budget(
        state.changed_files or [], state.diff or "", max_diff_characters, excluded_diff_paths
    )

    documentation_results = (state.documentation_context or {}).get("results", [])
    documentation_context = [DocumentChunk.model_validate(chunk) for chunk in documentation_results]

    coverage_notes = _extract_repository_coverage_notes(state.errors or [])
    coverage_notes.extend(_documentation_coverage_notes(state.components, documentation_context))
    if omitted_files:
        coverage_notes.append(
            "diff budget: included "
            f"{len(included_files)}/{len(included_files) + len(omitted_files)} "
            f"changed file(s) within the {max_diff_characters}-character budget; "
            f"omitted: {', '.join(omitted_files)}"
        )
    all_results = (state.repository_context or {"results": []}).get("results", [])
    selected_results, repository_context_truncated = select_repository_results_within_budget(
        all_results, max_repository_context_characters
    )
    # Keep the selection, do not re-derive it. What reaches a reviewer prompt
    # is now decided exactly here, once, and the coverage note below describes
    # the very list being stored rather than a separate later decision that
    # merely happened to match.
    repository_context = {"results": selected_results}
    if repository_context_truncated:
        coverage_notes.append(
            "repository context truncated: included "
            f"{len(selected_results)}/{len(all_results)} "
            f"search result(s) within the {max_repository_context_characters}-character budget"
        )

    return ReviewInput(
        repo=state.repo,
        pr_number=state.pr_number,
        head_sha=state.head_sha or "",
        included_diff=included_diff,
        diff_truncated=diff_truncated,
        included_files=included_files,
        omitted_files=omitted_files,
        repository_context=repository_context,
        documentation_context=documentation_context,
        coverage_notes=coverage_notes,
        max_repository_context_characters=max_repository_context_characters,
    )


def split_diff_by_file(diff: str) -> dict[str, str]:
    """Split a unified multi-file diff into each file's own hunk text, keyed by its path.

    Keyed by the path after "b/" in each "diff --git a/... b/..." header
    line, matching the paths GitHub's changed-files API returns.
    """
    file_diffs: dict[str, str] = {}
    current_path: Optional[str] = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_path is not None:
            file_diffs[current_path] = "\n".join(current_lines)

    for line in diff.splitlines():
        match = _DIFF_FILE_HEADER_RE.match(line)
        if match:
            flush()
            current_path = match.group(1)
            current_lines = [line]
        elif current_path is not None:
            current_lines.append(line)
    flush()

    return file_diffs


def _select_diff_within_budget(
    changed_files: list[str],
    diff: str,
    max_diff_characters: int,
    excluded_paths: tuple[str, ...] = (),
) -> tuple[str, bool, list[str], list[str]]:
    """Greedily include each file's whole hunk in changed_files order within max_diff_characters.

    A file's hunk is either fully included or fully omitted - never
    truncated mid-hunk. Once a hunk would exceed the remaining budget, that
    file and every file after it in changed_files order are omitted without
    further consideration (a deterministic stop, not best-fit packing). A
    changed file with no matching hunk in diff (e.g. a binary change), or one
    matching an excluded_paths glob pattern or a built-in generated/lock/build
    pattern (see _DEFAULT_EXCLUDED_GENERATED_PATTERNS), is skipped from both
    lists while still under budget - the same "nothing to include or omit for
    it" treatment, just for a different reason (deliberately out of review
    scope rather than absent from the diff text). Excluding generated files
    first, before the budget walk, is what lets the budget be spent on real
    source instead of being consumed or reordered by a lockfile.
    """
    file_diffs = split_diff_by_file(diff)
    all_excluded_patterns = (*_DEFAULT_EXCLUDED_GENERATED_PATTERNS, *excluded_paths)

    included_files: list[str] = []
    omitted_files: list[str] = []
    included_chunks: list[str] = []
    running_total = 0
    truncated = False

    for index, file_path in enumerate(changed_files):
        hunk = file_diffs.get(file_path)
        if hunk is None:
            continue
        if any(fnmatch.fnmatch(file_path, pattern) for pattern in all_excluded_patterns):
            continue

        addition = len(hunk) + (1 if included_chunks else 0)  # +1 for the joining newline
        if running_total + addition > max_diff_characters:
            truncated = True
            omitted_files.extend(changed_files[index:])
            break

        included_chunks.append(hunk)
        included_files.append(file_path)
        running_total += addition

    return "\n".join(included_chunks), truncated, included_files, omitted_files


def _extract_repository_coverage_notes(errors: list[str]) -> list[str]:
    """Pull Task 10's partial-repository-search notes out of state.errors by their prefix.

    Excludes unrelated hard failures (a Git command that failed outright, an
    invalid component mapping, etc.), which do not share this prefix.
    """
    return [error for error in errors if error.startswith(_REPOSITORY_COVERAGE_NOTE_PREFIX)]


def format_repository_search_result(result: dict) -> str:
    """Render one search result in the exact form included in a reviewer prompt."""

    return f"- {result.get('file')}:{result.get('line')}: {result.get('snippet')}"


def select_repository_results_within_budget(
    results: list[dict], max_characters: int
) -> tuple[list[dict], bool]:
    """Keep whole search-result lines greedily within a deterministic character budget."""

    selected: list[dict] = []
    running_total = 0
    for result in results:
        rendered = format_repository_search_result(result)
        addition = len(rendered) + (1 if selected else 0)
        if running_total + addition > max_characters:
            return selected, True
        selected.append(result)
        running_total += addition
    return selected, False


def _documentation_coverage_notes(
    components: Optional[list[Component]], documentation_context: list[DocumentChunk]
) -> list[str]:
    """Note when no approved documentation was found for this PR's classified components.

    doc_retrieval_tool.py returns an empty result rather than an error when
    nothing is approved yet or nothing matches - a legitimate outcome, not a
    failure - so this is derived directly from state instead of state.errors.
    """
    if documentation_context:
        return []

    if not components:
        return ["no approved documentation available: no component was classified for this PR"]

    component_names = ", ".join(sorted({component.value for component in components}))
    return [f"no approved documentation available for classified component(s): {component_names}"]
