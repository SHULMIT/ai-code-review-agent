"""LangGraph node: compute the final scores and recommendation from deduplicated findings.

The formula and category-to-score mapping implemented here are documented
in DECISION-LOG.md's "Scoring formula and category-to-score mapping
interpreted" entry - the original task spec names three scores without
defining an exact formula, so that entry records the interpretation this
module implements.
"""

from review_agent.graph.state import ReviewState
from review_agent.models.finding import Finding, Severity
from review_agent.models.review_result import ReviewResult

_STARTING_SCORE = 100
_PENALTY_BY_SEVERITY = {
    Severity.BLOCKING: 30,
    Severity.WARNING: 10,
    Severity.SUGGESTION: 3,
}

_ARCHITECTURE_CATEGORIES = {"architecture"}
_CODE_QUALITY_CATEGORIES = {"code quality"}
# Everything about whether the system survives contact with production:
# secrets, grounding, resource limits, and the agent machinery's own failure
# handling. A reviewer category absent from all three of these sets is not
# an error anything raises - its findings simply never reach a score. See
# tests/nodes/test_reviewer_categories_are_scored.py, which fails when a
# reviewer node is added without being mapped here.
_PRODUCTION_READINESS_CATEGORIES = {
    "security",
    "rag_search",
    "performance",
    "agent_harness",
}
SCORED_CATEGORIES = (
    _ARCHITECTURE_CATEGORIES | _CODE_QUALITY_CATEGORIES | _PRODUCTION_READINESS_CATEGORIES
)


def _score_for_categories(
    findings: list[Finding], categories: set[str], failed_categories: set[str]
) -> int | None:
    """Start at 100 and subtract each in-category finding's severity penalty, floored at 0.

    Returns None instead when any category feeding this score failed to run.
    A score of 100 means "looked, found nothing"; a category that never
    executed has no score, and reporting 100 for it states the opposite of
    what happened. production_readiness_score aggregates several categories,
    so one dead reviewer is enough to make the whole score unknown - a
    partial number there would be indistinguishable from a complete one.
    """
    if categories & failed_categories:
        return None
    score = _STARTING_SCORE
    for finding in findings:
        if finding.review_category in categories:
            score -= _PENALTY_BY_SEVERITY[finding.severity]
    return max(score, 0)


def _build_pr_summary(state: ReviewState, findings: list[Finding]) -> str:
    """A short, deterministic, template-generated summary - never model-generated."""
    changed_file_count = len(state.changed_files or [])
    component_count = len(state.components or [])
    blocking_count = sum(1 for finding in findings if finding.severity == Severity.BLOCKING)
    warning_count = sum(1 for finding in findings if finding.severity == Severity.WARNING)
    suggestion_count = sum(1 for finding in findings if finding.severity == Severity.SUGGESTION)
    return (
        f"Reviewed {changed_file_count} changed files across {component_count} "
        f"components; found {blocking_count} blocking, {warning_count} warning, "
        f"{suggestion_count} suggestion findings."
    )


# LangGraph nodes must accept only `state` - kept as a factory for consistency with every other node in this graph (see deduplicate_findings.py), even though this one needs no external dependency bound via closure.
def make_finalize_review_node():
    """Build the finalize_review node."""

    def finalize_review_node(state: ReviewState) -> dict:
        """Compute the three scores and recommendation from state.findings, deterministically.

        state.findings has already been deduplicated by the time this node
        runs (it follows deduplicate_findings in the graph). Never raises.

        The recommendation is decided in this order, and the order matters:

        1. Any BLOCKING finding -> REQUEST_CHANGES. A real blocking issue was
           found and reported, so the run produced an actionable answer even
           if some other reviewer also died.
        2. Otherwise, any failed category -> INCOMPLETE. "No blocking issues
           found" is only meaningful when the reviewers actually looked.
        3. Otherwise -> APPROVE.

        Empty findings alone must never reach APPROVE on their own merits:
        a run where every reviewer failed and a genuinely clean pull request
        both arrive here with zero findings, and state.failed_categories is
        the only thing that tells them apart.
        """
        findings = state.findings
        failed_categories = set(state.failed_categories)

        blocking_findings = [finding for finding in findings if finding.severity == Severity.BLOCKING]
        warnings = [finding for finding in findings if finding.severity == Severity.WARNING]
        suggestions = [finding for finding in findings if finding.severity == Severity.SUGGESTION]

        if blocking_findings:
            recommendation = "REQUEST_CHANGES"
        elif failed_categories:
            recommendation = "INCOMPLETE"
        else:
            recommendation = "APPROVE"

        review_result = ReviewResult(
            pr_summary=_build_pr_summary(state, findings),
            architecture_score=_score_for_categories(
                findings, _ARCHITECTURE_CATEGORIES, failed_categories
            ),
            code_quality_score=_score_for_categories(
                findings, _CODE_QUALITY_CATEGORIES, failed_categories
            ),
            production_readiness_score=_score_for_categories(
                findings, _PRODUCTION_READINESS_CATEGORIES, failed_categories
            ),
            blocking_findings=blocking_findings,
            warnings=warnings,
            suggestions=suggestions,
            recommendation=recommendation,
            reviewed_head_sha=state.head_sha or "",
            errors=state.errors,
            incomplete_categories=sorted(failed_categories),
            reviewer_diagnostics=state.reviewer_diagnostics,
            changed_file_count=len(state.changed_files or []),
        )
        return {"review_result": review_result}

    return finalize_review_node
