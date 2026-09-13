import pytest
from pydantic import ValidationError

from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component, Finding, Severity


def test_review_state_defaults():
    state = ReviewState(repo="example-org/example-repo", pr_number=42)

    assert state.repo == "example-org/example-repo"
    assert state.pr_number == 42
    assert state.head_sha is None
    assert state.diff is None
    assert state.changed_files is None
    assert state.components is None
    assert state.repository_context is None
    assert state.documentation_context is None
    assert state.findings == []
    assert state.errors == []
    assert state.review_result is None


def test_finding_validates():
    finding = Finding(
        severity=Severity.BLOCKING,
        component=Component.AGENT_HARNESS,
        review_category="architecture",
        file="src/review_agent/nodes/architecture.py",
        line=42,
        problem="Node calls the GitHub API directly instead of through publish_review().",
        why_it_matters="Bypasses the single extension point documented in CLAUDE.md.",
        suggested_fix="Route the call through publish_review() instead.",
    )

    assert finding.severity == Severity.BLOCKING
    assert finding.component == Component.AGENT_HARNESS
    assert finding.line == 42


def test_review_state_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        ReviewState(
            repo="example-org/example-repo",
            pr_number=42,
            findings=[
                {
                    "severity": "CRITICAL",
                    "component": Component.UNKNOWN,
                    "file": "x.py",
                    "problem": "p",
                    "why_it_matters": "w",
                    "suggested_fix": "s",
                }
            ],
        )
