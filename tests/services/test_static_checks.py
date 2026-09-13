"""Tests for the deterministic checks.

Weighted towards the negative cases on purpose. A false positive from a
mechanical check costs more than one from a model, because a reader assumes
a mechanical check is right and stops looking.
"""

from review_agent.models.finding import Severity
from review_agent.services.static_checks import (
    check_change_size,
    check_forbidden_files,
    check_hardcoded_secrets,
    check_lockfile_kept_in_sync,
    check_undocumented_settings,
    parse_added_lines,
    run_static_checks,
)


def _diff(path: str, *added: str, start: int = 1) -> str:
    body = "".join(f"+{line}\n" for line in added)
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +{start},{len(added)} @@\n"
        f"{body}"
    )


# --- diff parsing -------------------------------------------------------------


def test_added_lines_are_numbered_from_the_hunk_header():
    diff = _diff("a.py", "first", "second", start=10)

    assert parse_added_lines(diff) == [("a.py", 10, "first"), ("a.py", 11, "second")]


def test_removed_lines_are_not_reported_as_added():
    diff = "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,1 @@\n-gone\n+kept\n"

    assert [text for _, _, text in parse_added_lines(diff)] == ["kept"]


def test_a_deleted_file_contributes_no_added_lines():
    diff = "--- a/a.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-gone\n"

    assert parse_added_lines(diff) == []


# --- forbidden files ----------------------------------------------------------


def test_a_committed_env_file_is_blocking():
    findings = check_forbidden_files([".env"])

    assert len(findings) == 1
    assert findings[0].severity == Severity.BLOCKING
    assert findings[0].review_category == "security"


def test_a_nested_env_file_is_caught_too():
    assert check_forbidden_files(["services/code-review-agent/.env"])


def test_env_example_is_not_flagged():
    """It is the file other checks ask people to keep updated."""
    assert check_forbidden_files([".env.example", ".env.sample", ".env.template"]) == []


def test_generated_files_are_flagged_as_warnings_not_blocking():
    findings = check_forbidden_files(["src/__pycache__/a.cpython-312.pyc"])

    assert len(findings) == 1
    assert findings[0].severity == Severity.WARNING


def test_ordinary_source_files_are_not_flagged():
    assert check_forbidden_files(["src/app.py", "README.md", "config/mapping.yaml"]) == []


# --- secrets ------------------------------------------------------------------


def test_a_hardcoded_api_key_is_flagged_with_its_line_number():
    diff = _diff("a.py", 'API_KEY = "sk-live-4f2a9c8b7d6e5f"', start=7)

    findings = check_hardcoded_secrets(diff)

    assert len(findings) == 1
    assert findings[0].line == 7
    assert findings[0].severity == Severity.BLOCKING


def test_a_value_read_from_the_environment_is_not_a_secret():
    diff = _diff("a.py", 'API_KEY = os.environ["API_KEY"]', 'token = settings.token')

    assert check_hardcoded_secrets(diff) == []


def test_an_obvious_placeholder_is_not_flagged():
    diff = _diff("a.py", 'API_KEY = "your-api-key-here"', 'SECRET = "changeme-please"')

    assert check_hardcoded_secrets(diff) == []


def test_a_short_value_is_not_flagged():
    """Below the length threshold this is far more likely to be a test fixture."""
    diff = _diff("a.py", 'password = "abc"')

    assert check_hardcoded_secrets(diff) == []


def test_a_removed_secret_line_is_not_flagged():
    """Deleting a credential is the fix, not the problem."""
    diff = "--- a/a.py\n+++ b/a.py\n@@ -1,1 +0,0 @@\n-API_KEY = \"sk-live-4f2a9c8b7d6e5f\"\n"

    assert check_hardcoded_secrets(diff) == []


# --- lockfiles ----------------------------------------------------------------


def test_a_manifest_without_its_lockfile_is_flagged():
    findings = check_lockfile_kept_in_sync(["services/x/pyproject.toml"])

    assert len(findings) == 1
    assert "uv.lock" in findings[0].problem


def test_a_manifest_with_its_lockfile_is_not_flagged():
    assert check_lockfile_kept_in_sync(["services/x/pyproject.toml", "services/x/uv.lock"]) == []


# --- undocumented settings ----------------------------------------------------


def test_a_new_environment_variable_without_a_documented_entry_is_flagged():
    diff = _diff("a.py", 'url = os.environ["SEARCH_BACKEND_URL"]', start=3)

    findings = check_undocumented_settings(diff, ["a.py"])

    assert len(findings) == 1
    assert "SEARCH_BACKEND_URL" in findings[0].problem
    assert findings[0].line == 3


def test_nothing_is_flagged_when_the_example_file_was_updated_too():
    diff = _diff("a.py", 'url = os.environ["SEARCH_BACKEND_URL"]')

    assert check_undocumented_settings(diff, ["a.py", ".env.example"]) == []


def test_the_same_variable_read_twice_is_reported_once():
    diff = _diff("a.py", 'a = os.getenv("MODEL_URL")', 'b = os.getenv("MODEL_URL")')

    assert len(check_undocumented_settings(diff, ["a.py"])) == 1


# --- change size --------------------------------------------------------------


def test_a_small_change_is_not_flagged():
    assert check_change_size(_diff("a.py", "one", "two"), ["a.py"]) == []


def test_too_many_files_is_flagged():
    findings = check_change_size("", [f"f{i}.py" for i in range(41)], max_changed_files=40)

    assert len(findings) == 1
    assert "41 files" in findings[0].problem


def test_too_many_added_lines_is_flagged():
    diff = _diff("a.py", *[f"line {i}" for i in range(50)])

    findings = check_change_size(diff, ["a.py"], max_added_lines=10)

    assert len(findings) == 1


def test_change_size_handles_an_empty_changed_file_list():
    """The finding reads changed_files[0] for its file field - that must not raise when empty."""
    diff = _diff("a.py", "one", "two")

    findings = check_change_size(diff, [], max_added_lines=1)

    assert len(findings) == 1
    assert findings[0].file == ""


# --- composition --------------------------------------------------------------


def test_run_static_checks_combines_every_check():
    diff = _diff("a.py", 'API_KEY = "sk-live-4f2a9c8b7d6e5f"')

    findings = run_static_checks(diff, [".env", "a.py", "pyproject.toml"])

    categories = {finding.review_category for finding in findings}
    assert categories == {"security", "code quality"}
    assert len(findings) == 3


def test_a_clean_change_produces_no_findings():
    """The case that matters most: correct code must come back empty."""
    diff = _diff("src/app.py", "def add(a: int, b: int) -> int:", "    return a + b")

    assert run_static_checks(diff, ["src/app.py", "tests/test_app.py"]) == []
