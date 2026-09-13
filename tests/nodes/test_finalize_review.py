from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component, Finding
from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic
from review_agent.nodes.finalize_review import make_finalize_review_node


def _finding(**overrides) -> Finding:
    values = {
        "severity": "SUGGESTION",
        "component": "Agent Harness / LangGraph",
        "review_category": "code quality",
        "file": "a.py",
        "line": 7,
        "problem": "p",
        "why_it_matters": "w",
        "suggested_fix": "f",
    }
    values.update(overrides)
    return Finding(**values)


def _state(**overrides) -> ReviewState:
    values = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": "abc123",
        "changed_files": ["a.py", "b.py"],
        "components": [Component.AGENT_HARNESS],
        "findings": [],
    }
    values.update(overrides)
    return ReviewState(**values)


def test_finalize_review_node_produces_a_clean_result_with_no_findings():
    node = make_finalize_review_node()

    result = node(_state())

    review_result = result["review_result"]
    assert review_result.architecture_score == 100
    assert review_result.code_quality_score == 100
    assert review_result.production_readiness_score == 100
    assert review_result.recommendation == "APPROVE"
    assert review_result.blocking_findings == []
    assert review_result.warnings == []
    assert review_result.suggestions == []
    assert review_result.reviewed_head_sha == "abc123"
    assert review_result.errors == []
    assert review_result.pr_summary == (
        "Reviewed 2 changed files across 1 components; found 0 blocking, "
        "0 warning, 0 suggestion findings."
    )


def test_finalize_review_node_reduces_score_but_still_approves_with_only_suggestions():
    node = make_finalize_review_node()
    findings = [
        _finding(severity="SUGGESTION", review_category="code quality"),
        _finding(severity="SUGGESTION", review_category="code quality", line=9),
    ]

    result = node(_state(findings=findings))

    review_result = result["review_result"]
    assert review_result.code_quality_score == 100 - 3 - 3
    assert review_result.architecture_score == 100
    assert review_result.production_readiness_score == 100
    assert review_result.recommendation == "APPROVE"
    assert len(review_result.suggestions) == 2
    assert review_result.blocking_findings == []


def test_finalize_review_node_requests_changes_for_a_single_blocking_finding_regardless_of_scores():
    node = make_finalize_review_node()
    findings = [_finding(severity="BLOCKING", review_category="security")]

    result = node(_state(findings=findings))

    review_result = result["review_result"]
    assert review_result.recommendation == "REQUEST_CHANGES"
    # Only the security-mapped production_readiness_score reflects this finding;
    # architecture and code quality had nothing reported against them.
    assert review_result.architecture_score == 100
    assert review_result.code_quality_score == 100
    assert review_result.production_readiness_score == 70
    assert len(review_result.blocking_findings) == 1


def test_finalize_review_node_scores_each_category_independently():
    """Findings spread across all three categories must not cross-contaminate each other's score."""
    node = make_finalize_review_node()
    findings = [
        _finding(severity="BLOCKING", review_category="architecture", line=1),
        _finding(severity="WARNING", review_category="code quality", line=2),
        _finding(severity="SUGGESTION", review_category="security", line=3),
    ]

    result = node(_state(findings=findings))

    review_result = result["review_result"]
    assert review_result.architecture_score == 100 - 30
    assert review_result.code_quality_score == 100 - 10
    assert review_result.production_readiness_score == 100 - 3
    assert review_result.recommendation == "REQUEST_CHANGES"
    assert len(review_result.blocking_findings) == 1
    assert len(review_result.warnings) == 1
    assert len(review_result.suggestions) == 1


def test_finalize_review_node_floors_a_score_at_zero():
    node = make_finalize_review_node()
    findings = [
        _finding(severity="BLOCKING", review_category="architecture", line=index)
        for index in range(1, 6)
    ]

    result = node(_state(findings=findings))

    assert result["review_result"].architecture_score == 0


def test_finalize_review_node_preserves_upstream_errors():
    node = make_finalize_review_node()
    state = _state(errors=["security reviewer: Model Service request failed."])

    result = node(state)

    assert result["review_result"].errors == ["security reviewer: Model Service request failed."]


def test_finalize_review_node_preserves_reviewer_diagnostics():
    node = make_finalize_review_node()
    diagnostics = [
        ReviewerDiagnostic(
            category="security",
            status="completed",
            findings_count=2,
            duration_seconds=1.25,
        ),
        ReviewerDiagnostic(
            category="rag_search",
            status="skipped",
            duration_seconds=0.0,
        ),
    ]

    result = node(_state(reviewer_diagnostics=diagnostics))

    assert result["review_result"].reviewer_diagnostics == diagnostics


# --- A failed reviewer must never be reported as a passing one -----------------
# Regression cover for a real PR #105 run in which all four reviewers died on a
# Model Service misconfiguration and the result was still APPROVE, 100/100/100.


def test_a_run_where_every_reviewer_failed_is_incomplete_not_approved():
    node = make_finalize_review_node()

    result = node(
        _state(
            failed_categories=["security", "architecture", "code quality", "rag_search"],
        )
    )

    review_result = result["review_result"]
    assert review_result.recommendation == "INCOMPLETE"
    assert review_result.architecture_score is None
    assert review_result.code_quality_score is None
    assert review_result.production_readiness_score is None
    assert review_result.incomplete_categories == [
        "architecture",
        "code quality",
        "rag_search",
        "security",
    ]


def test_only_the_scores_fed_by_a_failed_category_become_unknown():
    """One dead reviewer must not erase the scores the surviving reviewers earned."""
    node = make_finalize_review_node()

    result = node(_state(failed_categories=["code quality"]))

    review_result = result["review_result"]
    assert review_result.code_quality_score is None
    assert review_result.architecture_score == 100
    assert review_result.production_readiness_score == 100
    assert review_result.incomplete_categories == ["code quality"]


def test_one_failed_category_makes_the_whole_production_readiness_score_unknown():
    """production_readiness aggregates several categories; a partial number would look complete."""
    node = make_finalize_review_node()

    result = node(_state(failed_categories=["rag_search"]))

    assert result["review_result"].production_readiness_score is None


def test_a_blocking_finding_still_wins_over_an_incomplete_run():
    """REQUEST_CHANGES is actionable and already correct - INCOMPLETE would only weaken it."""
    node = make_finalize_review_node()

    result = node(
        _state(
            findings=[_finding(severity="BLOCKING", review_category="security")],
            failed_categories=["code quality"],
        )
    )

    review_result = result["review_result"]
    assert review_result.recommendation == "REQUEST_CHANGES"
    assert review_result.incomplete_categories == ["code quality"]


def test_a_genuinely_clean_run_is_still_approved():
    """The INCOMPLETE path must not swallow the ordinary passing case."""
    node = make_finalize_review_node()

    result = node(_state(failed_categories=[]))

    review_result = result["review_result"]
    assert review_result.recommendation == "APPROVE"
    assert review_result.incomplete_categories == []
    assert review_result.architecture_score == 100
