from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component
from review_agent.models.review_input import build_review_input


def _file_diff(path: str, body_lines: int = 3) -> str:
    """Build a minimal, syntactically-plausible single-file unified diff block."""
    lines = [
        f"diff --git a/{path} b/{path}",
        "index 0000000..1111111 100644",
        f"--- a/{path}",
        f"+++ b/{path}",
        f"@@ -1,1 +1,{body_lines} @@",
    ]
    lines.extend(f"+line {i}" for i in range(body_lines))
    return "\n".join(lines)


def test_build_review_input_includes_everything_within_budget():
    files = ["a.py", "b.py"]
    diff = "\n".join(_file_diff(path) for path in files)
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff=diff,
        changed_files=files,
        repository_context={"results": []},
        documentation_context={
            "results": [{"source_path": "docs/x.md", "heading": "X", "snippet": "s"}]
        },
    )

    review_input = build_review_input(state)

    assert review_input.repo == "example-org/example-repo"
    assert review_input.pr_number == 1
    assert review_input.head_sha == "abc123"
    assert review_input.diff_truncated is False
    assert review_input.included_files == files
    assert review_input.omitted_files == []
    for path in files:
        assert f"diff --git a/{path} b/{path}" in review_input.included_diff
    assert len(review_input.documentation_context) == 1
    assert review_input.documentation_context[0].source_path == "docs/x.md"


def test_build_review_input_omits_files_once_budget_is_exceeded():
    files = ["a.py", "b.py", "c.py"]
    hunks = {path: _file_diff(path, body_lines=5) for path in files}
    diff = "\n".join(hunks[path] for path in files)
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff=diff,
        changed_files=files,
    )
    # Exactly enough room for a.py and b.py, joined by one newline; c.py must not fit.
    budget = len(hunks["a.py"]) + 1 + len(hunks["b.py"])

    review_input = build_review_input(state, max_diff_characters=budget)

    assert review_input.diff_truncated is True
    assert review_input.included_files == ["a.py", "b.py"]
    assert review_input.omitted_files == ["c.py"]
    # No partial hunk: the included diff is exactly the whole hunks joined,
    # never a truncated fragment of a hunk that didn't fully fit.
    assert review_input.included_diff == "\n".join([hunks["a.py"], hunks["b.py"]])
    assert hunks["c.py"] not in review_input.included_diff


def test_build_review_input_skips_a_changed_file_with_no_matching_hunk():
    # A binary file (or anything GitHub's diff text has no hunk for) has
    # nothing to include or omit for budget purposes - it is silently
    # skipped rather than counted as truncation.
    files = ["a.py", "image.png"]
    diff = _file_diff("a.py")
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff=diff,
        changed_files=files,
    )

    review_input = build_review_input(state)

    assert review_input.diff_truncated is False
    assert review_input.included_files == ["a.py"]
    assert review_input.omitted_files == []


def test_build_review_input_drops_a_file_matching_an_excluded_diff_path():
    # A repository-specific generated/export dump (configured per deployment,
    # never hardcoded) is removed from the diff entirely, before the
    # character budget is ever applied - not counted as truncation, and
    # invisible to included_files/omitted_files, exactly like a binary file
    # with no matching hunk.
    files = ["a.py", "all_classes_code.txt"]
    diff = "\n".join([_file_diff("a.py"), _file_diff("all_classes_code.txt")])
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff=diff,
        changed_files=files,
    )

    review_input = build_review_input(state, excluded_diff_paths=("all_classes_code.txt",))

    assert review_input.diff_truncated is False
    assert review_input.included_files == ["a.py"]
    assert review_input.omitted_files == []
    assert "all_classes_code.txt" not in review_input.included_diff


def test_build_review_input_excluded_diff_paths_supports_glob_patterns():
    files = ["a.py", "generated/dump.txt"]
    diff = "\n".join([_file_diff("a.py"), _file_diff("generated/dump.txt")])
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff=diff,
        changed_files=files,
    )

    review_input = build_review_input(state, excluded_diff_paths=("generated/*",))

    assert review_input.included_files == ["a.py"]
    assert "generated/dump.txt" not in review_input.included_diff


def test_build_review_input_drops_a_lockfile_by_default_with_no_configured_exclusions():
    # No excluded_diff_paths configured at all: a lockfile must still never
    # reach a reviewer prompt, or consume the diff budget that real source
    # changes need.
    files = ["a.py", "frontend/package-lock.json"]
    diff = "\n".join([_file_diff("a.py"), _file_diff("frontend/package-lock.json")])
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff=diff,
        changed_files=files,
    )

    review_input = build_review_input(state)

    assert review_input.included_files == ["a.py"]
    assert review_input.omitted_files == []
    assert "package-lock.json" not in review_input.included_diff


def test_build_review_input_notes_which_files_the_diff_budget_omitted():
    files = ["a.py", "b.py", "c.py"]
    hunks = {path: _file_diff(path, body_lines=5) for path in files}
    diff = "\n".join(hunks[path] for path in files)
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff=diff,
        changed_files=files,
    )
    budget = len(hunks["a.py"])

    review_input = build_review_input(state, max_diff_characters=budget)

    assert review_input.omitted_files == ["b.py", "c.py"]
    assert any(
        "diff budget: included 1/3" in note and "b.py, c.py" in note
        for note in review_input.coverage_notes
    )


def test_build_review_input_notes_missing_documentation_for_unclassified_component():
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff="",
        changed_files=[],
        components=[Component.UNKNOWN],
        documentation_context={"results": []},
    )

    review_input = build_review_input(state)

    assert review_input.documentation_context == []
    assert any("Unknown" in note for note in review_input.coverage_notes)


def test_build_review_input_reports_repo_truncation_and_missing_docs_together():
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff="",
        changed_files=[],
        components=[Component.AGENT_HARNESS],
        documentation_context={"results": []},
        errors=["repository search truncated: scanned 500/1200 files due to configured limit"],
    )

    review_input = build_review_input(state)

    assert len(review_input.coverage_notes) == 2
    assert any(note.startswith("repository search truncated") for note in review_input.coverage_notes)
    assert any("Agent Harness" in note for note in review_input.coverage_notes)


def test_build_review_input_excludes_unrelated_hard_failures_from_coverage_notes():
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff="",
        changed_files=[],
        components=[Component.AGENT_HARNESS],
        documentation_context={
            "results": [{"source_path": "docs/x.md", "heading": None, "snippet": "s"}]
        },
        errors=["git fetch timed out after 60s", "classify_node: invalid component mapping: bad"],
    )

    review_input = build_review_input(state)

    assert review_input.coverage_notes == []


def test_build_review_input_records_repository_context_truncation():
    results = [
        {"file": "a.py", "line": 1, "snippet": "first useful result"},
        {"file": "b.py", "line": 2, "snippet": "second useful result"},
    ]
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        diff="",
        changed_files=[],
        documentation_context={"results": [{"source_path": "docs/x.md", "heading": None, "snippet": "s"}]},
        repository_context={"results": results},
    )
    first_result_length = len("- a.py:1: first useful result")

    review_input = build_review_input(
        state, max_repository_context_characters=first_result_length
    )

    # The stored results are the ones that fit, not the full set. A
    # ReviewInput is a budgeted object: the list it carries and the coverage
    # note describing it have to be the same decision. Keeping all of them
    # and leaving the selection to whoever rendered the prompt is precisely
    # what allowed the note and the prompt to describe different things.
    assert review_input.repository_context["results"] == [results[0]]
    assert review_input.max_repository_context_characters == first_result_length
    assert any(note.startswith("repository context truncated: included 1/2") for note in review_input.coverage_notes)
