"""Validates raw model-produced findings against the Finding schema and the PR's actual diff.

Rejects findings that are likely hallucinations - an unparseable shape, a
file never touched by the PR, or a line number nowhere near a changed
line - without ever discarding a rejection silently: every dropped finding's
reason is returned alongside the survivors, per AGENTS.md's rule that
failures are recorded, never swallowed.

component is never read from the model's output: it is always derived
deterministically from a finding's file field via component_classifier.py's
classify_file(), the same function nodes/classify.py uses - see
DECISION-LOG.md's "Component derived deterministically" entry for why.
"""

import re
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from review_agent.models.finding import Component, Finding, Severity
from review_agent.models.review_input import ReviewInput, split_diff_by_file
from review_agent.nodes.finalize_review import SCORED_CATEGORIES
from review_agent.tools.component_classifier import ComponentRule, classify_file

# Observed model failure mode (a real reviewer call, on a live PR, reported
# one of architecture_review.md's own example bug patterns verbatim as a
# finding, instead of an actual grounded observation about the diff). Only a
# long, exact substring match is treated as an echo, never a fuzzy/semantic
# one, so a genuine finding that happens to discuss a similar concept in its
# own words is never rejected by this check.
_MIN_PROMPT_ECHO_LENGTH = 40

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
# Public: finding_deduplication.py reuses this same tolerance so two
# findings a few lines apart are treated as the same location consistently
# in both validation and deduplication.
LINE_TOLERANCE = 3
_NON_EMPTY_STRING_FIELDS = ("problem", "why_it_matters", "suggested_fix")
_ABSENCE_CLAIM_RE = re.compile(r"\b(?:missing|absent|absence|no)\b", re.IGNORECASE)
_DOCUMENTATION_CLAIM_RE = re.compile(r"\b(?:docs?|documentation|readme)\b", re.IGNORECASE)

# qwen2.5-coder:7b was observed returning these free-text values instead of
# the exact Severity enum members our prompts explicitly ask for - see
# DECISION-LOG.md's "Severity/component normalization before validation"
# entry. Deliberately conservative: only known synonyms are mapped, nothing
# is guessed.
_SEVERITY_SYNONYMS: dict[str, str] = {
    "blocking": Severity.BLOCKING.value,
    "error": Severity.BLOCKING.value,
    "critical": Severity.BLOCKING.value,
    "high": Severity.BLOCKING.value,
    "warning": Severity.WARNING.value,
    "medium": Severity.WARNING.value,
    "suggestion": Severity.SUGGESTION.value,
    "info": Severity.SUGGESTION.value,
    "information": Severity.SUGGESTION.value,
    "low": Severity.SUGGESTION.value,
}


def _normalize_severity(raw: str) -> str | None:
    """Map a free-text severity value to Finding's exact Severity enum value.

    Case-insensitive and trimmed. Returns None when raw matches no known
    value or synonym, so the caller leaves it untouched and Pydantic
    rejects it with its own clear error, exactly as before this
    normalization step existed.
    """
    return _SEVERITY_SYNONYMS.get(raw.strip().lower())


def _derive_component(file: str | None, rules: list[ComponentRule]) -> Component:
    """Deterministically classify a finding's component from its file field.

    Uses the same classify_file() the classify node itself uses against
    component_mapping.yaml, so a finding's component always matches how the
    rest of the graph would classify that same file. This is never a value
    the model chooses: models were observed repeatedly failing to match
    Component's exact enum values (file paths, generic labels like "Error
    Handling" instead of one of the 7 valid strings) even after a
    normalization attempt - see DECISION-LOG.md. A finding with no file (a
    whole-PR-level concern) has nothing to classify, so it falls back to
    Component.UNKNOWN, the same value classify_file() itself returns when no
    rule matches.
    """
    if not file:
        return Component.UNKNOWN
    return classify_file(file, rules)


def _prepare_raw_finding(raw_finding: Any, rules: list[ComponentRule]) -> Any:
    """Prepare one raw finding for Pydantic validation: normalize severity, derive component.

    severity is only touched when it is a string that normalizes
    successfully - everything else, including the input itself when it is
    not a dict, is left completely untouched, so a finding this cannot help
    still fails Pydantic validation with a clear error. component is always
    overwritten with the deterministically derived value, regardless of
    whatever the raw finding does or does not carry under that key - the
    model is never trusted to self-report this, the same rule
    stamp_review_category already applies to review_category.
    """
    if not isinstance(raw_finding, dict):
        return raw_finding

    normalized = dict(raw_finding)

    raw_severity = normalized.get("severity")
    if isinstance(raw_severity, str):
        mapped_severity = _normalize_severity(raw_severity)
        if mapped_severity is not None:
            normalized["severity"] = mapped_severity

    normalized["component"] = _derive_component(normalized.get("file"), rules).value

    return normalized


def stamp_review_category(raw_findings: Any, review_category: str) -> Any:
    """Inject review_category into each raw finding dict, before Finding validation.

    Which reviewer category produced a finding is decided in code, not
    something the Model Service is asked to report about itself - a
    reviewer node calls this right after parsing the model's raw JSON and
    before validate_findings. A non-list input, or a non-dict entry within
    it, is left untouched here so validate_findings still rejects it with
    its own clear reason instead of this function raising on malformed
    model output.
    """
    if not isinstance(raw_findings, list):
        return raw_findings
    return [
        {**raw_finding, "review_category": review_category} if isinstance(raw_finding, dict) else raw_finding
        for raw_finding in raw_findings
    ]


def validate_findings(
    raw_findings: list[dict],
    review_input: ReviewInput,
    component_rules: Sequence[ComponentRule] = (),
    system_instructions: str | None = None,
) -> tuple[list[Finding], list[str]]:
    """Parse and sanity-check raw model findings, returning (accepted, rejection reasons).

    Takes the already-parsed component rules, not a path to them. Validating
    a finding has nothing to do with the filesystem, and this function used
    to read and parse config/component_mapping.yaml itself on every call -
    six times per review run, once per reviewer, for a file that cannot
    change mid-run. build_graph now loads it once and each reviewer node
    passes it down.

    Defaults to no rules rather than to loading them: a caller that forgets
    to supply them gets Component.UNKNOWN, which is visible in the output,
    instead of a silent filesystem read creeping back into this function.

    Never raises on an individual malformed finding - each raw entry is
    either accepted or recorded as a rejection reason. Only a fundamentally
    wrong input shape (raw_findings not a list) raises, since that means the
    caller did not actually hand this function parsed JSON findings.

    system_instructions is the exact prompt text the reviewer that produced
    these findings was given - passed through so a finding that turns out to
    be a verbatim echo of the reviewer's own instructions, rather than a
    grounded observation about the diff, can be rejected. Optional only so
    every existing caller/test that has no reason to exercise this specific
    check keeps working unchanged.
    """
    if not isinstance(raw_findings, list):
        raise TypeError("raw_findings must be a list of dicts")

    file_diffs = split_diff_by_file(review_input.included_diff)
    rules = list(component_rules)

    accepted: list[Finding] = []
    rejections: list[str] = []

    for index, raw_finding in enumerate(raw_findings):
        try:
            finding = Finding.model_validate(_prepare_raw_finding(raw_finding, rules))
        except ValidationError as exc:
            rejections.append(f"finding[{index}]: failed schema validation: {exc}")
            continue

        reason = _rejection_reason(finding, review_input, file_diffs, system_instructions)
        if reason is not None:
            rejections.append(f"finding[{index}]: {reason}")
            continue

        accepted.append(finding)

    return accepted, rejections


def _rejection_reason(
    finding: Finding,
    review_input: ReviewInput,
    file_diffs: dict[str, str],
    system_instructions: str | None = None,
) -> str | None:
    """Return why a parsed finding should be rejected, or None if it passes every check.

    severity needs no separate emptiness check here - Pydantic already
    rejects any value that isn't a valid Severity member while parsing
    Finding, so an empty value could never have reached here. component
    needs no check at all: it is always deterministically derived (see
    _derive_component), never a value the model can get wrong.

    review_category IS checked here, against SCORED_CATEGORIES: every
    existing reviewer stamps it centrally in code before validation ever
    sees it (stamp_review_category), always one of the six valid strings,
    so this can never reject one of them - it exists for
    all_zero_verification.py, whose findings can span any category and are
    stamped from the finding's own text (see its
    _infer_review_category()), still never trusted from the model itself.
    """
    if finding.review_category not in SCORED_CATEGORIES:
        return f"review_category {finding.review_category!r} is not a recognized scored category"

    for field_name in _NON_EMPTY_STRING_FIELDS:
        if not getattr(finding, field_name).strip():
            return f"{field_name} is empty"

    if system_instructions and _echoes_prompt_instructions(finding, system_instructions):
        return (
            "problem/why_it_matters/suggested_fix matches the reviewer's own prompt "
            "instructions verbatim - likely echoed rather than a real observation"
        )

    if _claims_documentation_absent_despite_context(finding, review_input):
        return (
            "claims documentation is missing although the supplied repository or "
            "documentation context contains documentation evidence"
        )

    if not finding.file:
        # No file specified: a PR-wide finding, not tied to one changed file.
        return None

    if finding.file not in review_input.included_files and finding.file not in review_input.omitted_files:
        return f"file {finding.file!r} does not appear among this PR's changed files"

    if finding.line is not None and finding.file in file_diffs:
        if not _line_is_plausible(finding.line, file_diffs[finding.file]):
            return f"line {finding.line} is not near any changed line in {finding.file!r}"

    return None


def _claims_documentation_absent_despite_context(
    finding: Finding, review_input: ReviewInput
) -> bool:
    """Reject an absence claim contradicted by supplied README or documentation evidence.

    This intentionally does not try to infer that documentation is absent.
    An empty context may be incomplete, so a reviewer can only make an
    absence finding when its own changed-line evidence supports it. The
    validator's narrower job is to stop a categorical claim when the input
    already proves otherwise.
    """

    finding_text = " ".join(
        (finding.problem, finding.why_it_matters, finding.suggested_fix)
    )
    if not (
        _ABSENCE_CLAIM_RE.search(finding_text)
        and _DOCUMENTATION_CLAIM_RE.search(finding_text)
    ):
        return False

    if review_input.documentation_context:
        return True

    repository_results = (review_input.repository_context or {}).get("results", [])
    for result in repository_results:
        path = str(result.get("file", ""))
        if _DOCUMENTATION_CLAIM_RE.search(path):
            return True
    return False


def _echoes_prompt_instructions(finding: Finding, system_instructions: str) -> bool:
    """Return whether any of finding's text fields is a verbatim copy of the reviewer's own prompt.

    Only an exact, long substring match counts - long enough
    (_MIN_PROMPT_ECHO_LENGTH) that a real finding could never coincidentally
    match a full sentence of prompt text by chance. This catches literal
    copying, not paraphrasing; it is a safety net for the specific failure
    already observed, not a general hallucination detector.
    """
    for field_name in _NON_EMPTY_STRING_FIELDS:
        value = getattr(finding, field_name).strip()
        if len(value) >= _MIN_PROMPT_ECHO_LENGTH and value in system_instructions:
            return True
    return False


def _line_is_plausible(line: int, file_diff: str) -> bool:
    """Best-effort check that `line` falls within (or near) a hunk this file's diff touches.

    Not exact-line matching - a small tolerance absorbs the fact that diff
    hunks shift as surrounding code changes. This exists to catch obvious
    hallucinations (a line number nothing in the diff is near), not to gate
    the line-level-comment feature from a future task.
    """
    ranges = _touched_line_ranges(file_diff)
    if not ranges:
        return True
    return any(start - LINE_TOLERANCE <= line <= end + LINE_TOLERANCE for start, end in ranges)


def _touched_line_ranges(file_diff: str) -> list[tuple[int, int]]:
    """Return each hunk's (start, end) new-file line range, parsed from its @@ header."""
    ranges: list[tuple[int, int]] = []
    for line in file_diff.splitlines():
        match = _HUNK_HEADER_RE.match(line)
        if not match:
            continue
        start = int(match.group(1))
        length = int(match.group(2)) if match.group(2) else 1
        ranges.append((start, start + max(length - 1, 0)))
    return ranges
