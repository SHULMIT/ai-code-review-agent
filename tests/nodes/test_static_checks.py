from review_agent.graph.state import ReviewState
from review_agent.models.finding import Severity
from review_agent.nodes import static_checks as static_checks_module
from review_agent.nodes.static_checks import make_static_checks_node


def _state(**overrides) -> ReviewState:
    values = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": "abc123",
        "diff": "--- a/a.py\n+++ b/a.py\n@@ -0,0 +1,1 @@\n+print('hello')\n",
        "changed_files": ["a.py"],
    }
    values.update(overrides)
    return ReviewState(**values)


def test_static_checks_node_returns_findings_for_a_forbidden_file():
    node = make_static_checks_node()

    result = node(_state(changed_files=["a.py", ".env"]))

    assert len(result["findings"]) == 1
    assert result["findings"][0].severity == Severity.BLOCKING
    assert "errors" not in result


def test_static_checks_node_returns_an_empty_list_for_a_clean_change():
    node = make_static_checks_node()

    result = node(_state())

    assert result == {"findings": []}


def test_static_checks_node_records_an_error_when_the_diff_is_missing():
    """load_pr failing upstream must produce a recorded reason, not a crash."""
    node = make_static_checks_node()

    result = node(_state(diff=None, changed_files=None))

    assert result == {"errors": ["static checks: no diff available (upstream failure)"]}


def test_a_bug_inside_a_check_is_recorded_rather_than_raised(monkeypatch):
    """AGENTS.md: a node must never take the run down, including on its own bug."""

    def exploding_checks(diff, changed_files):
        raise ValueError("boom")

    monkeypatch.setattr(static_checks_module, "run_static_checks", exploding_checks)

    node = make_static_checks_node()
    result = node(_state())

    assert result == {"errors": ["static checks: ValueError: boom"]}


def test_static_checks_node_never_marks_a_category_as_failed():
    """These checks are not a reviewer category, so they must not claim one failed."""
    node = make_static_checks_node()

    result = node(_state(changed_files=[".env"]))

    assert "failed_categories" not in result
