"""The ReviewState object passed between every node in the LangGraph workflow.

Fields start as None (or empty, for the accumulating lists) and are filled in
gradually as the review progresses through the graph.
"""

import operator
from typing import Annotated, Optional

from pydantic import BaseModel

from review_agent.models.finding import Component, Finding
from review_agent.models.review_input import ReviewInput
from review_agent.models.review_result import ReviewResult
from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic


class ReplacedFindings(list):
    """Wrap a findings list in this to replace state.findings instead of adding to it.

    findings normally accumulates: each reviewer node's returned list is
    added to whatever is already there, via _merge_findings below. The one
    exception is deduplicate_findings_node, which consolidates the
    accumulated list into a shorter, merged one and must overwrite it
    entirely - returning a plain list there would get added to the existing
    findings by the normal reducer, duplicating every finding instead of
    deduplicating them.
    """


def _merge_findings(existing: list[Finding], update: list[Finding]) -> list[Finding]:
    """Reducer for `findings`: accumulate normally, but replace when `update` is ReplacedFindings."""
    if isinstance(update, ReplacedFindings):
        return list(update)
    return existing + update


class ReviewState(BaseModel):
    """State threaded through every node.

    failed_categories records the review_category of any reviewer that could
    not produce a result at all - no review_input, an empty diff, a Model
    Service error, or an unparseable response. A finding that was merely
    rejected by validation does NOT belong here: that reviewer ran and
    answered, it just answered partly wrong. finalize_review uses this to
    tell "reviewed and clean" apart from "never actually reviewed", which
    errors alone cannot express - a run where every reviewer failed
    otherwise produces the same empty findings list as a genuinely clean PR.

    Reviewers that a future conditional router skips must not appear here:
    a category that was deliberately not applicable did not fail.
    """

    repo: str
    pr_number: int
    head_sha: Optional[str] = None
    diff: Optional[str] = None
    changed_files: Optional[list[str]] = None
    components: Optional[list[Component]] = None
    repository_context: Optional[dict] = None
    documentation_context: Optional[dict] = None
    review_input: Optional[ReviewInput] = None
    findings: Annotated[list[Finding], _merge_findings] = []
    errors: Annotated[list[str], operator.add] = []
    failed_categories: Annotated[list[str], operator.add] = []
    reviewer_diagnostics: Annotated[list[ReviewerDiagnostic], operator.add] = []
    review_result: Optional[ReviewResult] = None
