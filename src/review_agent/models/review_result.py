"""ReviewResult: the final, deterministic summary finalize_review.py produces
from a review run's deduplicated findings - what gets published or stored,
not an intermediate LangGraph state field.
"""

from typing import Literal

from pydantic import BaseModel

from review_agent.models.finding import Finding
from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic


class ReviewResult(BaseModel):
    """The scored, grouped outcome of one review run.

    pr_summary is a short, template-generated sentence (never model-generated
    - see finalize_review.py), so it is exactly reproducible from the same
    inputs. blocking_findings/warnings/suggestions group the same
    deduplicated findings by severity, purely for display convenience -
    every Finding still also carries its own severity field.

    incomplete_categories names the reviewer categories that could not run.
    Any category listed there forces recommendation to INCOMPLETE and that
    category's score to None, because a review nobody performed must never
    be reported as a passing one. This was not a theoretical concern: a real
    run against PR #105 lost all four reviewers to a Model Service
    misconfiguration and still produced APPROVE with three scores of 100.
    errors carries the underlying messages, but errors alone cannot be acted
    on - a rejected finding and a dead reviewer both land there.
    reviewer_diagnostics keeps the per-reviewer timing and outcome metadata
    for runtime observability and the compact, safe "Reviewer Results" summary
    included in the published GitHub review.
    """

    pr_summary: str
    architecture_score: int | None
    code_quality_score: int | None
    production_readiness_score: int | None
    blocking_findings: list[Finding]
    warnings: list[Finding]
    suggestions: list[Finding]
    recommendation: Literal["APPROVE", "REQUEST_CHANGES", "INCOMPLETE"]
    reviewed_head_sha: str
    errors: list[str]
    incomplete_categories: list[str] = []
    reviewer_diagnostics: list[ReviewerDiagnostic] = []
    changed_file_count: int = 0
    duration_seconds: float | None = None
