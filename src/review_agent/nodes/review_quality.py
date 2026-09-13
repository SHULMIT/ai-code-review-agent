"""LangGraph node: the Code Quality reviewer, checking naming, duplication, error handling, logging, type safety, tests, and documentation."""

import logging
import time
from pathlib import Path

import httpx

from review_agent.config import Settings
from review_agent.graph.state import ReviewState
from review_agent.logging import log_event
from review_agent.models.review_input import split_diff_by_file
from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic
from review_agent.services.finding_validation import stamp_review_category, validate_findings
from review_agent.tools.component_classifier import (
    DEFAULT_MAPPING_PATH,
    ComponentMapping,
    load_component_mapping,
)
from review_agent.services.model_output_parsing import (
    ModelOutputParsingError,
    extract_json_from_model_output,
)
from review_agent.services.prompt_builder import build_review_prompt
from review_agent.tools.model_service_tool import ModelServiceError, call_model

_SERVICE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUALITY_PROMPT_PATH = _SERVICE_ROOT / "prompts" / "code_quality_review.md"
REVIEW_CATEGORY = "code quality"
_LOGGER = logging.getLogger("review_agent")
# TEMPORARY (PR #9 model-response investigation) - see the log_event call below.
_RAG_CLIENT_DIAGNOSTIC_PATH = "datanet_review_recording/rag_client.py"


class PromptTemplateError(Exception):
    """Raised when a reviewer's prompt template file cannot be read."""


def load_prompt_template(path: Path) -> str:
    """Read a prompt template file's content, wrapping any read failure into a clear error."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptTemplateError(f"failed to read prompt template at {path}: {exc}") from exc


# LangGraph nodes must accept only `state` - the Model Service settings and prompt path are bound here via closure before the node is registered.
def make_quality_reviewer_node(
    model_settings: Settings,
    component_mapping: ComponentMapping | None = None,
    client: httpx.AsyncClient | None = None,
    prompt_path: Path = DEFAULT_QUALITY_PROMPT_PATH,
):
    """Build the Code Quality reviewer node, reading its prompt template once, at factory time.

    The template is read here rather than inside quality_reviewer_node so a
    slow or missing prompt file surfaces at graph-build time, not on every
    review run. It currently has no per-request dynamic content, so it is
    used as-is; a future reviewer with dynamic template parts would apply
    simple str.format() to the loaded text inside the node instead of
    re-reading the file.

    client is an optional pooled httpx.AsyncClient shared across a whole
    review run - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5. Only
    passed on to call_model when actually given, so every existing caller
    that omits it keeps call_model's original per-call-client behavior
    unchanged.
    """

    system_instructions = load_prompt_template(prompt_path)
    # Resolved here, at factory time, never inside the node body: build_graph
    # normally passes the mapping it already loaded, and a caller that does
    # not (a focused test) still pays one read per factory rather than one
    # per review - the same rule the prompt template above follows.
    rules = (component_mapping or load_component_mapping(DEFAULT_MAPPING_PATH)).rules

    async def quality_reviewer_node(state: ReviewState) -> dict:
        """Run the Code Quality review category against the PR's grounded review_input.

        Code Quality review currently applies broadly, like Security and
        Architecture - it runs unconditionally once review_input exists.
        Per-component conditional routing across reviewers is a later
        refinement. Every failure path (a missing review_input, an empty
        diff, a Model Service error, an unparseable model response, or a
        rejected finding) is recorded into state.errors instead of raising,
        so this reviewer can never crash the review run.
        """
        started_at = time.monotonic()

        def _failed(message: str) -> dict:
            return {
                "errors": [f"quality reviewer: {message}"],
                "failed_categories": [REVIEW_CATEGORY],
                "reviewer_diagnostics": [
                    ReviewerDiagnostic(
                        category=REVIEW_CATEGORY,
                        status="failed",
                        duration_seconds=time.monotonic() - started_at,
                        error=message,
                    )
                ],
            }

        if state.review_input is None:
            return _failed("no review_input available (upstream failure)")
        if not state.review_input.included_diff:
            return _failed("included_diff is empty, nothing to review")

        prompt = build_review_prompt(
            state.review_input,
            state.components or [],
            review_category=REVIEW_CATEGORY,
            system_instructions=system_instructions,
        )

        call_kwargs = {} if client is None else {"client": client}
        try:
            model_output = await call_model(prompt, model_settings, **call_kwargs)
        except ModelServiceError as exc:
            return _failed(str(exc))

        try:
            raw_findings = extract_json_from_model_output(model_output.raw_text)
        except ModelOutputParsingError as exc:
            return _failed(str(exc))

        raw_findings = stamp_review_category(raw_findings, REVIEW_CATEGORY)

        try:
            findings, rejections = validate_findings(
                raw_findings, state.review_input, rules, system_instructions=system_instructions
            )
        except TypeError as exc:
            return _failed(str(exc))

        # TEMPORARY (PR #9 model-response investigation) - local-only, never
        # sent to GitHub, never the full prompt or repository source. Remove
        # alongside logging.py's matching fields once the investigation is
        # closed.
        _rag_client_hunk = split_diff_by_file(state.review_input.included_diff).get(
            _RAG_CLIENT_DIAGNOSTIC_PATH
        )
        log_event(
            _LOGGER, logging.INFO, "reviewer_model_response_diagnostic",
            reviewer_category=REVIEW_CATEGORY,
            prompt_length=len(prompt),
            raw_response=model_output.raw_text,
            parsed_findings_count=len(raw_findings) if isinstance(raw_findings, list) else -1,
            validation_result=f"{len(findings)} accepted, {len(rejections)} rejected",
            included_changed_files=state.review_input.included_files,
            rag_client_in_prompt=_RAG_CLIENT_DIAGNOSTIC_PATH in state.review_input.included_files,
            diff_total_chars=len(state.review_input.included_diff),
            rag_client_diff_chars=len(_rag_client_hunk) if _rag_client_hunk is not None else None,
        )

        result: dict = {
            "findings": findings,
            "reviewer_diagnostics": [
                ReviewerDiagnostic(
                    category=REVIEW_CATEGORY,
                    status="completed",
                    raw_findings_count=len(raw_findings),
                    findings_count=len(findings),
                    rejected_count=len(rejections),
                    rejection_reasons=rejections,
                    duration_seconds=time.monotonic() - started_at,
                )
            ],
        }
        if rejections:
            result["errors"] = [f"quality reviewer: {reason}" for reason in rejections]
        return result

    return quality_reviewer_node
