"""LangGraph node: a single safety-net verification pass when all six specialist
reviewers agree a real, non-trivial diff has zero findings.

Runs after deduplicate_findings, before finalize_review. It never blocks or
downgrades a run on its own failure: a Model Service error, an unparseable
response, or a validation failure here is treated exactly like a clean []
verification - the original all-zero, APPROVE result stands. This node
therefore never returns "failed_categories" (see finalize_review.py:
any non-empty failed_categories flips the recommendation away from APPROVE,
which a transient hiccup in this optional extra check must never do).
"""

import logging
import time
from pathlib import Path

import httpx

from review_agent.config import Settings
from review_agent.graph.state import ReplacedFindings, ReviewState
from review_agent.logging import log_event
from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic
from review_agent.services.finding_deduplication import deduplicate_findings
from review_agent.services.finding_validation import validate_findings
from review_agent.services.model_output_parsing import (
    ModelOutputParsingError,
    extract_json_from_model_output,
)
from review_agent.services.prompt_builder import build_all_zero_verification_prompt
from review_agent.tools.component_classifier import (
    DEFAULT_MAPPING_PATH,
    ComponentMapping,
    load_component_mapping,
)
from review_agent.tools.model_service_tool import ModelServiceError, call_model

_SERVICE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALL_ZERO_VERIFICATION_PROMPT_PATH = _SERVICE_ROOT / "prompts" / "all_zero_verification.md"
REVIEW_CATEGORY = "all_zero_verification"
_LOGGER = logging.getLogger("review_agent")

_EXPECTED_REVIEWER_COUNT = 6
# A general, filename-independent measure of "is there real code here to
# double-check" - counts actual added/removed content lines, not diff/hunk
# headers, so a single-line typo fix or a pure-metadata diff never triggers
# a second model call. Deliberately not based on file extension or path:
# which files count as "source" is a property of what changed, not a
# hardcoded list this service would have to keep in sync with every target
# repository.
_MIN_NON_TRIVIAL_CHANGED_LINES = 3
_DIFF_FILE_HEADER_PREFIXES = ("+++", "---")

# Independent of finding_deduplication.py's own keyword table, which exists
# to arbitrate between findings different specialist reviewers ALREADY
# categorized - a different problem. This exists because this node's single
# prompt spans every category, so nothing stamps one category centrally the
# way every specialist reviewer node does. review_category is still never
# trusted from the model itself (see finding_validation.py's module
# docstring); this is the code-level substitute.
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "security": (
        "secret", "credential", "token", "password", "auth", "permission",
        "injection", "vulnerab", "exploit",
    ),
    "architecture": (
        "boundary", "layer", "coupl", "responsibility", "separation", "dependency",
    ),
    "rag_search": (
        "retriev", "search", "rank", "relevance", "embedding", "grounding",
        "source", "citation", "passage",
    ),
    "performance": (
        "memory", "latency", "timeout", "unbounded", "blocking", "throughput",
        "concurren", "scale",
    ),
    "agent_harness": (
        "graph", "node", "state", "langgraph", "reducer", "workflow",
    ),
    "code quality": (
        "test", "coverage", "duplicat", "naming", "type hint", "documentation",
        "error handling", "exception",
    ),
}
_CATEGORY_FALLBACK = "code quality"


class PromptTemplateError(Exception):
    """Raised when the verification prompt template file cannot be read."""


def load_prompt_template(path: Path) -> str:
    """Read the verification prompt template's content, wrapping any read failure into a clear error."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptTemplateError(f"failed to read prompt template at {path}: {exc}") from exc


def _infer_review_category(finding_text: str) -> str:
    """Deterministically assign a review_category from a verification finding's own text."""
    lowered = finding_text.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return _CATEGORY_FALLBACK


def _stamp_inferred_categories(raw_findings: object) -> object:
    """Inject a deterministically-inferred review_category into each raw finding dict.

    Mirrors finding_validation.stamp_review_category's contract: a non-list
    input, or a non-dict entry within it, is left untouched so
    validate_findings still rejects it with its own clear reason.
    """
    if not isinstance(raw_findings, list):
        return raw_findings
    stamped = []
    for raw_finding in raw_findings:
        if not isinstance(raw_finding, dict):
            stamped.append(raw_finding)
            continue
        text = " ".join(
            str(raw_finding.get(field, "")) for field in ("problem", "why_it_matters", "suggested_fix")
        )
        stamped.append({**raw_finding, "review_category": _infer_review_category(text)})
    return stamped


def _non_trivial_source_changes(included_diff: str) -> bool:
    """Count real added/removed diff content lines, excluding file headers."""
    count = 0
    for line in included_diff.splitlines():
        if line.startswith(_DIFF_FILE_HEADER_PREFIXES):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count >= _MIN_NON_TRIVIAL_CHANGED_LINES


def _all_reviewers_completed(diagnostics: list[ReviewerDiagnostic]) -> bool:
    """True only when exactly the six specialist reviewers ran and all completed.

    A failed or skipped reviewer means the all-zero result is not a genuine
    "six specialists looked and found nothing" - it is already an
    incomplete run, which finalize_review already surfaces on its own
    terms; this extra check must not run on top of that.
    """
    return len(diagnostics) == _EXPECTED_REVIEWER_COUNT and all(
        diagnostic.status == "completed" for diagnostic in diagnostics
    )


# LangGraph nodes must accept only `state` - the Model Service settings and prompt path are bound here via closure before the node is registered.
def make_all_zero_verification_node(
    model_settings: Settings,
    component_mapping: ComponentMapping | None = None,
    client: httpx.AsyncClient | None = None,
    prompt_path: Path = DEFAULT_ALL_ZERO_VERIFICATION_PROMPT_PATH,
):
    """Build the all-zero verification node, reading its prompt template once, at factory time."""

    system_instructions = load_prompt_template(prompt_path)
    rules = (component_mapping or load_component_mapping(DEFAULT_MAPPING_PATH)).rules

    async def all_zero_verification_node(state: ReviewState) -> dict:
        """Challenge a zero-finding result once, only when it's worth challenging.

        Runs only when all six specialist reviewers completed and still
        produced zero findings between them, and the diff is substantial
        enough to be worth a second look. Every other run (any reviewer
        failed or skipped, a finding already exists, or the diff is
        trivial) is a no-op - this node changes nothing, exactly preserving
        today's behavior for every review that already has a real result.
        """
        started_at = time.monotonic()
        call_kwargs = {} if client is None else {"client": client}

        if not _all_reviewers_completed(state.reviewer_diagnostics):
            return {}
        if state.findings:
            return {}
        if state.review_input is None or not _non_trivial_source_changes(
            state.review_input.included_diff
        ):
            log_event(
                _LOGGER, logging.INFO, "all_zero_verification_check",
                reason_code="all_zero_verification",
                repository=state.repo, pr_number=state.pr_number, triggered=False,
            )
            return {}

        log_event(
            _LOGGER, logging.INFO, "all_zero_verification_check",
            reason_code="all_zero_verification",
            repository=state.repo, pr_number=state.pr_number, triggered=True,
        )

        prompt = build_all_zero_verification_prompt(state.review_input, system_instructions)

        def _unavailable(message: str) -> dict:
            """A failure here degrades to "verification unavailable", never to a
            downgraded recommendation - see the module docstring for why
            failed_categories is never returned from this function."""
            return {
                "errors": [f"all_zero_verification: {message}"],
                "reviewer_diagnostics": [
                    ReviewerDiagnostic(
                        category=REVIEW_CATEGORY,
                        status="failed",
                        duration_seconds=time.monotonic() - started_at,
                        error=message,
                    )
                ],
            }

        try:
            model_output = await call_model(prompt, model_settings, **call_kwargs)
        except ModelServiceError as exc:
            return _unavailable(str(exc))

        try:
            raw_findings = extract_json_from_model_output(model_output.raw_text)
        except ModelOutputParsingError as exc:
            return _unavailable(str(exc))

        raw_findings = _stamp_inferred_categories(raw_findings)

        try:
            findings, rejections = validate_findings(
                raw_findings, state.review_input, rules, system_instructions=system_instructions
            )
        except TypeError as exc:
            return _unavailable(str(exc))

        # TEMPORARY (PR #9/#10 model-response investigation) - local-only,
        # never sent to GitHub, never the full prompt or repository source.
        # Remove alongside logging.py's matching fields once the
        # investigation is closed.
        log_event(
            _LOGGER, logging.INFO, "reviewer_model_response_diagnostic",
            reviewer_category=REVIEW_CATEGORY,
            prompt_length=len(prompt),
            raw_response=model_output.raw_text,
            parsed_findings_count=len(raw_findings) if isinstance(raw_findings, list) else -1,
            validation_result=f"{len(findings)} accepted, {len(rejections)} rejected",
            included_changed_files=state.review_input.included_files,
        )

        deduplicated = (
            await deduplicate_findings(findings, model_settings, **call_kwargs) if findings else findings
        )

        result: dict = {
            "findings": ReplacedFindings(deduplicated),
            "reviewer_diagnostics": [
                ReviewerDiagnostic(
                    category=REVIEW_CATEGORY,
                    status="completed",
                    raw_findings_count=len(raw_findings) if isinstance(raw_findings, list) else 0,
                    findings_count=len(deduplicated),
                    rejected_count=len(rejections),
                    rejection_reasons=rejections,
                    duration_seconds=time.monotonic() - started_at,
                )
            ],
        }
        if rejections:
            result["errors"] = [f"all_zero_verification: {reason}" for reason in rejections]
        return result

    return all_zero_verification_node
