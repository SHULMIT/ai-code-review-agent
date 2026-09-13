"""ReviewerDiagnostic: one reviewer node's outcome for a single review run.

Observability only - never published to GitHub, never carries a raw model
response or prompt content. Lives beside the other pure data models (not in
graph/state.py) so both state.py and review_result.py can import it without
a circular dependency between them.
"""

from typing import Literal

from pydantic import BaseModel


class ReviewerDiagnostic(BaseModel):
    """One reviewer node's outcome for a single run.

    status is "completed" (ran and returned, even with zero findings),
    "failed" (ran but could not produce a result - no review_input, empty
    diff, a Model Service error, or an unparseable response), or "skipped"
    (a conditional-routing reviewer decided this PR's components make it
    not applicable - not a failure). error carries only the short message
    already recorded in state.errors, never a raw model response or any
    prompt content, so this is always safe to log in full.
    """

    category: str
    status: Literal["completed", "failed", "skipped"]
    raw_findings_count: int = 0
    findings_count: int = 0
    rejected_count: int = 0
    rejection_reasons: list[str] = []
    duration_seconds: float = 0.0
    error: str | None = None
