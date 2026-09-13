"""Renders a ReviewResult into the Markdown body of a GitHub Pull Request
Review, and renders Check Run progress/final text (see check_run_tool.py).

format_review_body() is a pure function of its input: the same ReviewResult
always produces the exact same string, so callers (and snapshot tests) can
rely on it being byte-for-byte stable. Unlike the previous design, the
recommendation word IS shown directly in the body now (as its own
"## Recommendation" section) - GitHub's own review `event` field still
carries the real APPROVE/REQUEST_CHANGES decision (see github_publisher.py),
this is purely for a human scanning the body to see it immediately rather
than only via GitHub's small icon.

Every string built here is either a fixed template, a category name, a
count, or model-authored finding text (problem/why_it_matters/
suggested_fix) - never a raw exception message or internal error string.
That is the safety property section 8 asks for: it is satisfied by
construction, not by a separate sanitization pass, because nothing that
reaches this module was ever allowed to carry internal detail (IPs, keys,
stack traces, paths) in the first place - see model_service_tool.py's
_call_gemini(), which builds already-safe ModelServiceError messages.

The one exception is ReviewerDiagnostic.error, which is free text a
reviewer node wrote for internal diagnostics (see review_security.py and
its five siblings) and can, in one documented case, echo a raw model
response back (model_output_parsing.py's "got: {parsed!r}"). That field is
never rendered directly: _safe_failure_reason() classifies it through a
small allowlist of known-safe substrings and falls back to a generic safe
sentence for anything unrecognized, so an unvetted string can never reach
GitHub through this module.
"""

from review_agent.models.finding import Finding
from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic
from review_agent.models.review_result import ReviewResult

_RECOMMENDATION_DISPLAY = {
    "APPROVE": "APPROVE",
    "REQUEST_CHANGES": "REQUEST CHANGES",
    "INCOMPLETE": "INCOMPLETE",
}
_RECOMMENDATION_EXPLANATION = {
    "APPROVE": "No blocking issues were found. This pull request looks safe to merge as reviewed.",
    "REQUEST_CHANGES": "At least one blocking issue was found. Please address it before merging.",
    "INCOMPLETE": "Not every reviewer category completed, so this result is partial - see below.",
}
_SEVERITY_DISPLAY = {
    "Blocking": ("🔴", "Blocking"),
    "Warnings": ("🟠", "Warnings"),
    "Suggestions": ("🔵", "Suggestions"),
}

# finding_validation.py's validate_findings() always prefixes a rejected
# finding's own reason with "finding[{index}]: ", before a reviewer node adds
# its "{category} reviewer: " prefix on top - this is the only reliable way
# to tell "one finding was discarded as likely inaccurate" apart from "the
# whole category failed to run" (already covered by the incomplete-categories
# warning above) inside review_result.errors, which is a plain list[str].
_FINDING_REJECTION_MARKER = "finding["
_INFORMATION_URL = "https://shulamit-noa.netlify.app/"
_REVIEWER_DISPLAY_NAMES = (
    ("security", "Security"),
    ("architecture", "Architecture"),
    ("code quality", "Code Quality"),
    ("rag_search", "RAG/Search"),
    ("performance", "Performance"),
    ("agent_harness", "Agent Harness"),
)
_STATUS_ICON = {"completed": "✅", "skipped": "✅", "failed": "⚠️"}

# Allowlist, not a blocklist: a ReviewerDiagnostic.error string can come from
# a validation/parsing failure that echoes the model's own raw (untrusted,
# uncurated) output back into the message - see model_output_parsing.py's
# "got: {parsed!r}" branch - so only a recognized, known-safe substring maps
# to its own specific reason. Anything that does not match one of these,
# including that raw-echo case, falls through to _GENERIC_SAFE_FAILURE_REASON
# rather than ever being shown verbatim.
_SAFE_FAILURE_REASON_PATTERNS: tuple[tuple[str, str], ...] = (
    ("rate-limited", "Gemini temporarily rate-limited"),
    ("temporarily unavailable", "Gemini temporarily unavailable"),
    ("did not respond in time", "Gemini did not respond in time"),
    ("could not reach gemini", "Could not reach Gemini"),
    ("model is not available", "Configured model is not available"),
    ("rejected the configured api key", "Gemini rejected the configured API key"),
    ("no gemini_api_key is configured", "Gemini is not configured"),
    ("model service returned", "Model Service returned an error"),
    ("model service request failed", "Could not reach the Model Service"),
    ("unparseable response", "The model returned an unexpected response"),
    ("no review_input available", "No content was available to review"),
    ("included_diff is empty", "No content was available to review"),
)
_GENERIC_SAFE_FAILURE_REASON = "This reviewer did not complete due to an internal error"


def format_review_body(review_result: ReviewResult) -> str:
    """Render a ReviewResult into the full, stable Markdown PR review body."""

    total_findings = (
        len(review_result.blocking_findings)
        + len(review_result.warnings)
        + len(review_result.suggestions)
    )
    diagnostics_by_category = {d.category: d for d in review_result.reviewer_diagnostics}
    completed_reviewers = sum(
        1 for d in review_result.reviewer_diagnostics if d.status in ("completed", "skipped")
    )
    failed_reviewers = sum(1 for d in review_result.reviewer_diagnostics if d.status == "failed")
    is_partial = bool(review_result.incomplete_categories)
    has_findings = total_findings > 0

    sections = ["# DataNet Code Review", ""]

    sections += [
        "## Recommendation",
        "",
        f"**{_RECOMMENDATION_DISPLAY[review_result.recommendation]}**",
        "",
        _RECOMMENDATION_EXPLANATION[review_result.recommendation],
        "",
        _format_status_paragraph(review_result, completed_reviewers, diagnostics_by_category),
        "",
        "### What to do next",
        "",
    ]
    sections += _format_what_to_do_next(has_findings, is_partial)
    sections.append("")

    if is_partial:
        sections.append(
            _format_incomplete_warning(review_result.incomplete_categories, diagnostics_by_category)
        )
        sections.append("")

    rejected_findings_count = _count_rejected_findings(review_result.errors)
    if rejected_findings_count:
        sections.append(_format_rejected_findings_note(rejected_findings_count))
        sections.append("")

    sections += [
        "## Review Summary",
        "",
        f"- Files reviewed: {review_result.changed_file_count}",
        f"- Total findings: {total_findings}",
        f"- Blocking: {len(review_result.blocking_findings)}",
        f"- Warnings: {len(review_result.warnings)}",
        f"- Suggestions: {len(review_result.suggestions)}",
        f"- Reviewers completed: {completed_reviewers}/{len(_REVIEWER_DISPLAY_NAMES)}",
        f"- Reviewers failed: {failed_reviewers}/{len(_REVIEWER_DISPLAY_NAMES)}",
        f"- Review duration: {_format_duration(review_result.duration_seconds)}",
        f"- Review completeness: {'Partial' if is_partial else 'Complete'}",
        "",
    ]

    scores_section = _format_scores_section(review_result)
    if scores_section:
        sections += scores_section + [""]

    sections += [
        "## Findings",
        "",
        _format_finding_section("Blocking", review_result.blocking_findings),
        "",
        _format_finding_section("Warnings", review_result.warnings),
        "",
        _format_finding_section("Suggestions", review_result.suggestions),
        "",
        _format_reviewer_status_table(review_result.reviewer_diagnostics),
        "",
        "## Next Action",
        "",
        _format_next_action(review_result.incomplete_categories, has_findings),
        "",
        "---",
        "",
        "**DataNet Code Review Agent**",
        "Developed by Shulamit & Noa",
        "",
        f"[More information about DataNet Code Review / Report an issue]({_INFORMATION_URL})",
    ]
    return "\n".join(sections)


def _format_status_paragraph(
    review_result: ReviewResult,
    completed_reviewers: int,
    diagnostics_by_category: dict[str, ReviewerDiagnostic],
) -> str:
    """One human-readable sentence stating exactly how much of the review actually ran."""

    total = len(_REVIEWER_DISPLAY_NAMES)
    if not review_result.incomplete_categories:
        return f"All {total} of {total} reviewer categories completed. This review is complete."

    missed_names = ", ".join(
        display_name
        for category, display_name in _REVIEWER_DISPLAY_NAMES
        if category in review_result.incomplete_categories
    )
    return (
        f"{completed_reviewers} of {total} reviewer categories completed. The findings below "
        f"are valid, but this review may have missed issues in {missed_names}."
    )


def _format_what_to_do_next(has_findings: bool, is_partial: bool) -> list[str]:
    """Short, explicit bullets: what a reader should actually do with this review."""
    lines = []
    if has_findings:
        lines.append("- Fix the reported findings.")
    retry_guidance = "Retry later" if is_partial else "No retry needed"
    lines.append(f"- Retry guidance: {retry_guidance}")
    return lines


def _format_next_action(incomplete_categories: list[str], has_findings: bool) -> str:
    """One closing sentence synthesizing what to do, echoed again near the footer."""
    if incomplete_categories:
        count = len(incomplete_categories)
        category_word = "category" if count == 1 else "categories"
        if has_findings:
            return (
                f"Fix the reported findings and retry the review later because {count} "
                f"reviewer {category_word} did not complete."
            )
        return f"Retry the review later because {count} reviewer {category_word} did not complete."
    if has_findings:
        return "Fix the reported findings. No retry needed."
    return "No retry needed."


def _safe_failure_reason(error: str | None) -> str:
    """Classify a reviewer's internal error into a small, known-safe, human-readable reason.

    Allowlist-only: an unrecognized message (which may echo raw, uncurated
    model output - see model_output_parsing.py) never reaches GitHub as-is.
    """
    if not error:
        return _GENERIC_SAFE_FAILURE_REASON
    lowered = error.lower()
    for pattern, safe_reason in _SAFE_FAILURE_REASON_PATTERNS:
        if pattern in lowered:
            return safe_reason
    return _GENERIC_SAFE_FAILURE_REASON


def _format_duration(duration_seconds: float | None) -> str:
    """Render the review's wall-clock duration, or a placeholder when unknown."""
    if duration_seconds is None:
        return "unknown"
    return f"{duration_seconds:.1f}s"


def _format_reviewer_status_table(diagnostics: list[ReviewerDiagnostic]) -> str:
    """Render the six reviewers as a Markdown table: status, finding count, and (for a
    failed reviewer) a safe, human-readable reason - never left as a bare "Failed".
    """

    by_category = {diagnostic.category: diagnostic for diagnostic in diagnostics}
    lines = [
        "## Reviewer Status",
        "",
        "| Reviewer | Status | Findings | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for category, display_name in _REVIEWER_DISPLAY_NAMES:
        diagnostic = by_category.get(category)
        if diagnostic is None:
            lines.append(f"| {display_name} | ⏳ Not reported | — | — |")
            continue
        icon = _STATUS_ICON.get(diagnostic.status, "⏳")
        status_label = f"{icon} {diagnostic.status.capitalize()}"
        findings_cell = "—" if diagnostic.status == "failed" else str(diagnostic.findings_count)
        reason_cell = _safe_failure_reason(diagnostic.error) if diagnostic.status == "failed" else "—"
        lines.append(f"| {display_name} | {status_label} | {findings_cell} | {reason_cell} |")
    return "\n".join(lines)


def _format_incomplete_warning(
    incomplete_categories: list[str], diagnostics_by_category: dict[str, ReviewerDiagnostic]
) -> str:
    """A prominent, near-the-top notice naming exactly which categories never ran, why
    (a safe reason per category), and what to do about it.

    Placed immediately under the recommendation, before the summary, scores,
    or findings: a reader must see that this result is partial before they
    start trusting anything below it.
    """
    failed_lines = []
    for category, display_name in _REVIEWER_DISPLAY_NAMES:
        if category not in incomplete_categories:
            continue
        diagnostic = diagnostics_by_category.get(category)
        reason = _safe_failure_reason(diagnostic.error if diagnostic else None)
        failed_lines.append(f"- {display_name} — {reason}")
    return (
        "⚠️ **Partial review**\n"
        "\n"
        f"Only {len(incomplete_categories)} of {len(_REVIEWER_DISPLAY_NAMES)} reviewer "
        "categories completed. The findings below are valid, but additional issues may "
        "exist in the categories that failed.\n"
        "\n"
        "Failed categories:\n"
        f"{chr(10).join(failed_lines)}\n"
        "\n"
        "Retry guidance: Retry later."
    )


def _count_rejected_findings(errors: list[str]) -> int:
    """Count how many error entries are individual finding-validation rejections."""
    return sum(1 for error in errors if _FINDING_REJECTION_MARKER in error)


def _format_rejected_findings_note(count: int) -> str:
    """A short, safe summary noting findings were discarded, without any raw error text.

    Deliberately never renders the underlying error strings themselves - they
    may contain unfriendly or verbose raw model/validation output not meant
    for a PR reader. The full detail already survives to storage via
    review_result.errors (see review_launcher.py's _build_review_run_completion).
    """
    plural = "finding" if count == 1 else "findings"
    return (
        f"> **Note:** {count} {plural} were rejected during validation "
        "(likely inaccurate or ungrounded) and are not shown above - see "
        "review history for details."
    )


def _format_score(score: int | None) -> str:
    """Render a numeric score, or a clear placeholder when its category never ran."""
    if score is None:
        return "N/A (category did not complete)"
    return f"{score}/100"


def _format_scores_section(review_result: ReviewResult) -> list[str]:
    """Render the Scores section, or omit it entirely when nothing could be scored.

    Architecture and Code Quality map one-to-one onto their own reviewer
    category; Production Readiness is a composite of four other categories
    (Security, RAG/Search, Performance, Agent Harness combined) and is
    labeled as such so it is never mistaken for a seventh reviewer category
    alongside the ones in the Reviewer Status table below. When every score
    is unavailable (every scored category failed), the section is omitted
    rather than shown as three consecutive "N/A" lines.
    """
    scores = (
        review_result.architecture_score,
        review_result.code_quality_score,
        review_result.production_readiness_score,
    )
    if all(score is None for score in scores):
        return []
    return [
        "## Scores",
        "",
        "*Architecture and Code Quality each map to their own reviewer category. "
        "Production Readiness is a combined score across Security, RAG/Search, "
        "Performance, and Agent Harness.*",
        "",
        f"- **Architecture:** {_format_score(review_result.architecture_score)}",
        f"- **Code Quality:** {_format_score(review_result.code_quality_score)}",
        f"- **Production Readiness (combined):** {_format_score(review_result.production_readiness_score)}",
    ]


def _finding_title(problem: str) -> str:
    """Render a concise, ~4-10 word title from a finding's Problem text.

    Takes the first sentence (split at the first ./!/?), then caps it at 10
    words - a pure, deterministic rendering step, not a second model call
    (no LLM is invoked here; see the module docstring on why every reviewer
    node's own detection/prompting logic is out of scope for this module).
    The full, untruncated problem text always still appears in full under
    the finding's own "Problem" heading, so nothing is ever lost - only the
    heading avoids repeating it verbatim when it is long.
    """
    text = problem.strip()
    for sep in (". ", "! ", "? "):
        index = text.find(sep)
        if index != -1:
            text = text[: index + 1]
            break
    text = text.rstrip(".!? ").strip()
    words = text.split()
    if len(words) > 10:
        text = " ".join(words[:10]) + "..."
    return text or "Untitled finding"


def _format_finding_section(title: str, findings: list[Finding]) -> str:
    """Render one severity group as a numbered Markdown section, or a 'none found' line.

    Each finding keeps its own numbered heading with a short generated
    title, then Category/File/Line, then Problem/Why it matters/Suggested
    fix as their own paragraphs - a merge note (when this finding
    consolidates more than one reviewer's report) is its own clearly-labeled
    subsection, not a trailing footnote, so the underlying observations stay
    visible and actionable rather than buried inside one line. Distinct
    defects are never collapsed into one finding here - finding_deduplication.py
    already made that call upstream; this module only renders what it is given.
    """
    emoji, _ = _SEVERITY_DISPLAY[title]
    lines = [f"### {emoji} {title}", ""]
    if not findings:
        lines.append(f"No {title.lower()} found.")
        return "\n".join(lines)

    for index, finding in enumerate(findings, start=1):
        lines += [
            f"#### {index}. {_finding_title(finding.problem)}",
            "",
            f"**Category:** {finding.review_category}  ",
            f"**File:** `{finding.file or '(general)'}`  ",
            f"**Line:** {finding.line if finding.line is not None else '(general)'}",
            "",
            "**Problem**",
            "",
            finding.problem,
            "",
            "**Why it matters**",
            "",
            finding.why_it_matters,
            "",
            "**Suggested fix**",
            "",
            finding.suggested_fix,
        ]
        if finding.merge_note:
            lines += ["", "**Also reported by other reviewers**", "", finding.merge_note]
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------- Check Run text ---

_PENDING_ICON = "⏳"


def format_check_run_start() -> tuple[str, str]:
    """The immediate, instant-feedback text set the moment a click is accepted.

    Every reviewer shows pending - this is rendered before the graph has
    made a single Model Service call, so there is nothing else to report yet.
    """
    checklist = "\n".join(f"{_PENDING_ICON} {name}" for _, name in _REVIEWER_DISPLAY_NAMES)
    summary = f"0/{len(_REVIEWER_DISPLAY_NAMES)} reviewers completed\n\n{checklist}"
    return "Code review is running", summary


def format_check_run_progress(diagnostics_so_far: list[ReviewerDiagnostic]) -> tuple[str, str]:
    """Live progress text: how many of the six reviewers have reported back, and which."""

    by_category = {diagnostic.category: diagnostic for diagnostic in diagnostics_so_far}
    completed = sum(1 for d in diagnostics_so_far if d.status in ("completed", "skipped", "failed"))
    lines = []
    for category, display_name in _REVIEWER_DISPLAY_NAMES:
        diagnostic = by_category.get(category)
        icon = _STATUS_ICON.get(diagnostic.status, _PENDING_ICON) if diagnostic else _PENDING_ICON
        lines.append(f"{icon} {display_name}")
    title = f"Code Review in progress — {completed}/{len(_REVIEWER_DISPLAY_NAMES)} reviewers completed"
    summary = "\n".join(lines)
    return title, summary


def format_check_run_final(review_result: ReviewResult, review_url: str | None = None) -> str:
    """The final Check Run body: recommendation, completeness, counts, and retry guidance.

    output.summary alone must never be only a one-line sentence - this is
    the richer text GitHub renders as the Check Run's own details, distinct
    from (but consistent with) the full published PR Review.

    This function is only ever called by review_launcher.py after
    publish_review() has already succeeded (never on a publish failure or
    any other terminal outcome, which use their own distinct safe
    messages), so a "✅ Code Review published" status line is always
    correct here. GitHub's Checks API does not live-refresh a PR's
    Conversation timeline, so review_url - built from the same review ID
    GitHub just returned from publishing - gives the user a direct link to
    jump straight to it rather than relying on the page to refresh itself.
    review_url is optional: a caller that could not build one (e.g. the ID
    was unavailable for some reason) still gets the rest of this text, just
    without the link line.
    """
    total_findings = (
        len(review_result.blocking_findings)
        + len(review_result.warnings)
        + len(review_result.suggestions)
    )
    completed_reviewers = sum(
        1 for d in review_result.reviewer_diagnostics if d.status in ("completed", "skipped")
    )
    lines = [
        "✅ Code Review published",
        "",
        "The full DataNet Code Review has been published to the PR Conversation.",
    ]
    if review_url:
        lines.append(f"[View full review]({review_url})")
    lines += [
        "",
        f"Recommendation: {_RECOMMENDATION_DISPLAY[review_result.recommendation]}",
        (
            "Review status: Partially completed"
            if review_result.incomplete_categories
            else "Review status: Completed"
        ),
        f"Findings: {total_findings} "
        f"({len(review_result.blocking_findings)} blocking, "
        f"{len(review_result.warnings)} warnings, "
        f"{len(review_result.suggestions)} suggestions)",
        f"Reviewers completed: {completed_reviewers}/{len(_REVIEWER_DISPLAY_NAMES)}",
    ]
    if review_result.incomplete_categories:
        lines.append(
            "Categories that did not complete: " + ", ".join(sorted(review_result.incomplete_categories))
        )
        lines.append("Retry guidance: Retry later")
    return "\n".join(lines)
