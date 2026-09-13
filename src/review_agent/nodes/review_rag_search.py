"""LangGraph node: the RAG and Search grounding reviewer, checking sources, metadata, validation, and grounding."""

import logging
import time
import re
from pathlib import Path

import httpx

from review_agent.config import Settings
from review_agent.graph.state import ReviewState
from review_agent.logging import log_event
from review_agent.models.finding import Component
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
DEFAULT_RAG_SEARCH_PROMPT_PATH = _SERVICE_ROOT / "prompts" / "rag_search_review.md"
# Must stay exactly "rag_search": finalize_review.py already lists this
# string in _PRODUCTION_READINESS_CATEGORIES, so any other spelling would
# silently drop this reviewer's findings out of the production-readiness
# score rather than fail loudly.
REVIEW_CATEGORY = "rag_search"
_LOGGER = logging.getLogger("review_agent")
# TEMPORARY (PR #9 model-response investigation) - see the log_event call below.
_RAG_CLIENT_DIAGNOSTIC_PATH = "datanet_review_recording/rag_client.py"
# This reviewer's checks - source/metadata provenance, grounding, retrieval
# caching - only make sense for code that retrieves, transforms, or serves
# search/document content. See RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 7.
_RELEVANT_COMPONENTS = {Component.SEARCH_SERVICE, Component.INDEXING_PIPELINE}
_SEARCH_OR_RANKING_SIGNAL = re.compile(
    r"\b(?:search|retriev(?:e|al)|rank(?:ing|ed)?|relevance|rerank|embedding|vector|passage)\b",
    re.IGNORECASE,
)


def _reviewer_applies(components: list[Component] | None, included_diff: str) -> bool:
    """Return whether this reviewer's narrow category is relevant to what changed.

    Component.UNKNOWN always counts as applicable, never as a reason to
    skip: component_mapping.yaml's file-path matching is a best-effort
    heuristic (see finding_validation.py), so an unclassified file might
    still be exactly what this reviewer would have flagged. A missing or
    empty components list - classify_node never ran, or found nothing -
    fails open the same way, so an upstream classification problem can
    never silently suppress this entire reviewer category.
    """
    if not components:
        return True
    if Component.UNKNOWN in components:
        return True
    if any(component in _RELEVANT_COMPONENTS for component in components):
        return True
    # A path mapping is deliberately a best-effort classifier. A new
    # directory can contain real search/ranking behavior before its mapping
    # is added, so visible retrieval signals in the changed code must keep
    # this specialist reviewer enabled rather than silently suppressing it.
    return bool(_SEARCH_OR_RANKING_SIGNAL.search(included_diff))


class PromptTemplateError(Exception):
    """Raised when a reviewer's prompt template file cannot be read."""


def load_prompt_template(path: Path) -> str:
    """Read a prompt template file's content, wrapping any read failure into a clear error."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptTemplateError(f"failed to read prompt template at {path}: {exc}") from exc


# LangGraph nodes must accept only `state` - the Model Service settings and prompt path are bound here via closure before the node is registered.
def make_rag_search_reviewer_node(
    model_settings: Settings,
    component_mapping: ComponentMapping | None = None,
    client: httpx.AsyncClient | None = None,
    prompt_path: Path = DEFAULT_RAG_SEARCH_PROMPT_PATH,
):
    """Build the RAG and Search grounding reviewer node, reading its prompt template once, at factory time.

    The template is read here rather than inside rag_search_reviewer_node so
    a slow or missing prompt file surfaces at graph-build time, not on every
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

    async def rag_search_reviewer_node(state: ReviewState) -> dict:
        """Run the RAG and Search grounding review category against the PR's grounded review_input.

        Conditionally routed (see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item
        7 and _reviewer_applies above): skipped entirely - a "skipped"
        reviewer_diagnostics entry only, never an error or a failed
        category, since state.py's reducers treat a deliberate skip as "did
        not apply", not "failed to run" - when state.components names no
        component this reviewer's checks are meaningful for. Every other
        failure path (a missing review_input,
        an empty diff, a Model Service error, an unparseable model
        response, or a rejected finding) is recorded into state.errors
        instead of raising, so this reviewer can never crash the review run.
        """
        started_at = time.monotonic()

        def _failed(message: str) -> dict:
            return {
                "errors": [f"rag search reviewer: {message}"],
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
        if not _reviewer_applies(state.components, state.review_input.included_diff):
            return {
                "reviewer_diagnostics": [
                    ReviewerDiagnostic(
                        category=REVIEW_CATEGORY,
                        status="skipped",
                        duration_seconds=time.monotonic() - started_at,
                    )
                ]
            }
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
            result["errors"] = [f"rag search reviewer: {reason}" for reason in rejections]
        return result

    return rag_search_reviewer_node
