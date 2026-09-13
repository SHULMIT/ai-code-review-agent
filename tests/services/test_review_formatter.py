from review_agent.models.finding import Finding
from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic
from review_agent.models.review_result import ReviewResult
from review_agent.services.review_formatter import (
    format_check_run_final,
    format_check_run_progress,
    format_check_run_start,
    format_review_body,
)


def _finding(**overrides) -> Finding:
    values = {
        "severity": "BLOCKING",
        "component": "Agent Harness / LangGraph",
        "review_category": "security",
        "file": "a.py",
        "line": 7,
        "problem": "p",
        "why_it_matters": "w",
        "suggested_fix": "f",
    }
    values.update(overrides)
    return Finding(**values)


def _result(**overrides) -> ReviewResult:
    values = {
        "pr_summary": "Reviewed 1 changed files across 1 components; found 0 blocking, 0 warning, 0 suggestion findings.",
        "architecture_score": 100,
        "code_quality_score": 100,
        "production_readiness_score": 100,
        "blocking_findings": [],
        "warnings": [],
        "suggestions": [],
        "recommendation": "APPROVE",
        "reviewed_head_sha": "abc123",
        "errors": [],
    }
    values.update(overrides)
    return ReviewResult(**values)


def test_format_review_body_for_a_clean_pr_with_no_findings():
    review_result = _result(
        pr_summary=(
            "Reviewed 2 changed files across 1 components; found 0 blocking, "
            "0 warning, 0 suggestion findings."
        ),
        changed_file_count=2,
    )

    body = format_review_body(review_result)
    lines = body.splitlines()

    assert lines[0] == "# DataNet Code Review"
    assert "## Recommendation" in lines
    assert "**APPROVE**" in lines
    assert "All 6 of 6 reviewer categories completed. This review is complete." in lines
    assert "### What to do next" in lines
    assert "- Retry guidance: No retry needed" in lines
    assert "## Review Summary" in lines
    assert "- Files reviewed: 2" in lines
    assert "- Blocking: 0" in lines
    assert "- Warnings: 0" in lines
    assert "- Suggestions: 0" in lines
    assert "- Reviewers failed: 0/6" in lines
    assert "- Review completeness: Complete" in lines
    assert "- **Architecture:** 100/100" in lines
    assert "- **Code Quality:** 100/100" in lines
    assert "- **Production Readiness (combined):** 100/100" in lines
    assert "### 🔴 Blocking" in lines
    assert "No blocking found." in lines
    assert "### 🟠 Warnings" in lines
    assert "No warnings found." in lines
    assert "### 🔵 Suggestions" in lines
    assert "No suggestions found." in lines
    assert "## Reviewer Status" in lines
    assert "## Next Action" in lines
    assert "No retry needed." in lines
    assert "**DataNet Code Review Agent**" in lines
    assert lines[-1] == (
        "[More information about DataNet Code Review / Report an issue]"
        "(https://shulamit-noa.netlify.app/)"
    )


def test_format_review_body_groups_findings_by_severity_with_details():
    review_result = _result(
        pr_summary=(
            "Reviewed 3 changed files across 2 components; found 1 blocking, "
            "1 warning, 1 suggestion findings."
        ),
        architecture_score=70,
        code_quality_score=97,
        blocking_findings=[
            _finding(
                file="a.py",
                line=7,
                problem="Hardcoded secret.",
                why_it_matters="Exposed to anyone with source access.",
                suggested_fix="Read it from configuration instead.",
            )
        ],
        warnings=[
            _finding(
                severity="WARNING",
                review_category="code quality",
                file="b.py",
                line=12,
                problem="No error handling.",
                why_it_matters="A failure could crash the caller.",
                suggested_fix="Wrap the call in try/except.",
            )
        ],
        suggestions=[
            _finding(
                severity="SUGGESTION",
                review_category="code quality",
                file="",
                line=None,
                problem="No tests were added.",
                why_it_matters="Regressions are harder to catch.",
                suggested_fix="Add unit tests.",
            )
        ],
        recommendation="REQUEST_CHANGES",
    )

    body = format_review_body(review_result)

    assert "#### 1. Hardcoded secret" in body
    assert "**File:** `a.py`" in body
    assert "**Line:** 7" in body
    assert "Exposed to anyone with source access." in body
    assert "Read it from configuration instead." in body
    assert "#### 1. No error handling" in body
    assert "**File:** `b.py`" in body
    assert "**Line:** 12" in body
    assert "#### 1. No tests were added" in body
    assert "**File:** `(general)`" in body
    assert "**REQUEST CHANGES**" in body


def test_format_review_body_shows_the_recommendation_word_directly():
    """Unlike the previous design, the recommendation word is now shown in the body itself."""
    review_result = _result(
        architecture_score=70,
        blocking_findings=[_finding()],
        recommendation="REQUEST_CHANGES",
    )

    body = format_review_body(review_result)

    assert "**REQUEST CHANGES**" in body
    assert "APPROVE" not in body


def test_format_review_body_renders_a_general_file_placeholder_when_the_finding_has_no_line():
    review_result = _result(
        code_quality_score=90,
        warnings=[
            _finding(
                severity="WARNING",
                review_category="code quality",
                file="search_helper.py",
                line=None,
                problem="No tests exist for this file.",
                why_it_matters="Regressions are harder to catch.",
                suggested_fix="Add unit tests.",
            )
        ],
    )

    body = format_review_body(review_result)

    assert "**File:** `search_helper.py`" in body
    assert "**Line:** (general)" in body


def test_format_review_body_is_deterministic():
    review_result = _result(
        code_quality_score=97,
        suggestions=[_finding(severity="SUGGESTION", review_category="code quality")],
    )

    assert format_review_body(review_result) == format_review_body(review_result)


def test_format_review_body_shows_all_reviewer_outcomes_and_one_information_link():
    review_result = _result(
        reviewer_diagnostics=[
            ReviewerDiagnostic(category="security", status="completed", findings_count=2),
            ReviewerDiagnostic(category="rag_search", status="skipped"),
            ReviewerDiagnostic(
                category="performance", status="failed", error="Gemini is temporarily unavailable."
            ),
        ],
    )

    body = format_review_body(review_result)

    assert "## Reviewer Status" in body
    assert "| Reviewer | Status | Findings | Reason |" in body
    assert "| --- | --- | ---: | --- |" in body
    assert "| Security | ✅ Completed | 2 | — |" in body
    assert "| RAG/Search | ✅ Skipped | 0 | — |" in body
    assert "| Performance | ⚠️ Failed | — | Gemini temporarily unavailable |" in body
    assert "| Architecture | ⏳ Not reported | — | — |" in body
    assert body.count("https://shulamit-noa.netlify.app/") == 1
    assert body.endswith(
        "[More information about DataNet Code Review / Report an issue](https://shulamit-noa.netlify.app/)"
    )


def test_format_review_body_never_shows_a_bare_failed_status_without_a_reason():
    """A failed reviewer's reason must never be blank, even with no recorded error."""
    review_result = _result(
        reviewer_diagnostics=[ReviewerDiagnostic(category="security", status="failed", error=None)],
    )

    body = format_review_body(review_result)

    assert "| Security | ⚠️ Failed | — | This reviewer did not complete due to an internal error |" in body


def test_format_review_body_never_leaks_a_raw_model_echo_as_a_failure_reason():
    """A validation error that happens to echo raw model output must be classified, not shown verbatim."""
    review_result = _result(
        reviewer_diagnostics=[
            ReviewerDiagnostic(
                category="security",
                status="failed",
                error=(
                    "model output is a JSON object that isn't a single-key array wrapper "
                    "(expected {\"<name>\": [...]}), got: {'unexpected': 'shape', 'secret': 'sk_live_abc'}"
                ),
            )
        ],
    )

    body = format_review_body(review_result)

    assert "sk_live_abc" not in body
    assert "got:" not in body
    assert "This reviewer did not complete due to an internal error" in body


def test_format_review_body_renders_the_markdown_table_header_and_separator_on_their_own_lines():
    """The header/separator/row lines must never merge into one run of glued-together text."""
    review_result = _result()

    body = format_review_body(review_result)

    assert "ReviewerStatusFindings" not in body
    assert "| Reviewer | Status | Findings | Reason |\n| --- | --- | ---: | --- |\n" in body


def test_format_review_body_for_an_incomplete_review_shows_a_prominent_warning():
    review_result = _result(
        architecture_score=None,
        code_quality_score=90,
        production_readiness_score=None,
        warnings=[_finding(severity="WARNING", review_category="code quality")],
        recommendation="INCOMPLETE",
        errors=["architecture reviewer: Model Service request failed."],
        incomplete_categories=["architecture", "security"],
        reviewer_diagnostics=[
            ReviewerDiagnostic(
                category="architecture", status="failed",
                error="Gemini is temporarily unavailable. Please try again shortly.",
            ),
        ],
    )

    body = format_review_body(review_result)
    lines = body.splitlines()

    warning_index = lines.index("⚠️ **Partial review**")
    recommendation_index = lines.index("## Recommendation")
    summary_index = lines.index("## Review Summary")
    assert recommendation_index < warning_index < summary_index
    assert "- Architecture — Gemini temporarily unavailable" in lines
    assert "- Security — This reviewer did not complete due to an internal error" in lines
    assert "**INCOMPLETE**" in body
    assert "2 of 6 reviewer categories completed" in body
    assert "Retry guidance: Retry later" in body


def test_format_review_body_shows_incomplete_warning_alongside_request_changes():
    """Blocking findings do not erase the fact that another reviewer failed."""

    review_result = _result(
        architecture_score=None,
        blocking_findings=[_finding()],
        recommendation="REQUEST_CHANGES",
        errors=["architecture reviewer: Model Service request failed."],
        incomplete_categories=["architecture"],
    )

    body = format_review_body(review_result)

    assert "⚠️ **Partial review**" in body
    assert "- Architecture —" in body
    assert "**REQUEST CHANGES**" in body


def test_format_review_body_never_shows_the_incomplete_warning_for_a_complete_review():
    review_result = _result()

    body = format_review_body(review_result)

    assert "Partial review" not in body


def test_format_review_body_renders_a_null_score_as_not_available():
    review_result = _result(
        architecture_score=None,
        production_readiness_score=None,
        recommendation="INCOMPLETE",
        incomplete_categories=["architecture", "security", "rag_search", "performance", "agent_harness"],
    )

    body = format_review_body(review_result)

    assert "- **Architecture:** N/A (category did not complete)" in body
    assert "- **Code Quality:** 100/100" in body
    assert "- **Production Readiness (combined):** N/A (category did not complete)" in body


def test_format_review_body_omits_the_scores_section_when_nothing_could_be_scored():
    """Three consecutive N/A lines are noise; omit the section entirely when it can't be meaningful."""
    review_result = _result(
        architecture_score=None,
        code_quality_score=None,
        production_readiness_score=None,
        recommendation="INCOMPLETE",
        incomplete_categories=[
            "architecture", "code quality", "security", "rag_search", "performance", "agent_harness",
        ],
    )

    body = format_review_body(review_result)

    assert "## Scores" not in body
    assert "N/A (category did not complete)" not in body


def test_format_review_body_labels_production_readiness_as_a_combined_score():
    """Production Readiness must never look like a seventh reviewer category next to the real six."""
    review_result = _result()

    body = format_review_body(review_result)

    assert "Production Readiness is a combined score" in body
    assert "- **Production Readiness (combined):** 100/100" in body


def test_format_review_body_shows_a_populated_merge_note_as_its_own_visible_detail():
    """A merged finding must visibly say so, and must not bury the underlying observations."""
    review_result = _result(
        architecture_score=70,
        blocking_findings=[
            _finding(
                merge_note=(
                    "Merged 2 independently reported findings: Hardcoded API key.; "
                    "Secret exposed in source (component values disagreed: Model Service, Search Service)"
                )
            )
        ],
        recommendation="REQUEST_CHANGES",
    )

    body = format_review_body(review_result)

    assert "**Also reported by other reviewers**" in body
    assert (
        "Merged 2 independently reported findings: Hardcoded API key.; "
        "Secret exposed in source (component values disagreed: Model Service, Search Service)"
    ) in body


def test_format_review_body_omits_the_merge_note_line_when_not_set():
    """A finding that was never merged must not gain an empty or placeholder note line."""
    review_result = _result(
        architecture_score=70,
        blocking_findings=[_finding()],
        recommendation="REQUEST_CHANGES",
    )

    body = format_review_body(review_result)

    assert "Also reported by other reviewers" not in body


def test_format_review_body_shows_a_safe_rejected_findings_count_without_raw_error_text():
    """Individual finding rejections must surface as a safe count, never the raw error strings."""
    review_result = _result(
        errors=[
            "security reviewer: finding[0]: failed schema validation: 1 validation error for Finding",
            "architecture reviewer: finding[2]: file 'secret_internal_path.py' does not appear among this PR's changed files",
        ],
    )

    body = format_review_body(review_result)

    assert "2 findings were rejected during validation" in body
    assert "failed schema validation" not in body
    assert "secret_internal_path.py" not in body
    assert "1 validation error for Finding" not in body


def test_format_review_body_does_not_show_the_rejected_findings_note_for_a_total_category_failure():
    """A whole-category failure (already shown by the incomplete-categories warning) must not double-count."""
    review_result = _result(
        architecture_score=None,
        code_quality_score=90,
        warnings=[_finding(severity="WARNING", review_category="code quality")],
        recommendation="INCOMPLETE",
        errors=["architecture reviewer: Model Service request failed."],
        incomplete_categories=["architecture"],
    )

    body = format_review_body(review_result)

    assert "⚠️ **Partial review**" in body
    assert "were rejected during validation" not in body


def test_format_review_body_shows_neither_addition_for_a_clean_review():
    """A genuinely clean result (no merges, no errors) shows no partial/rejection noise."""
    review_result = _result()

    body = format_review_body(review_result)

    assert "Also reported by other reviewers" not in body
    assert "were rejected during validation" not in body
    assert "Partial review" not in body


def test_format_review_body_shows_the_review_duration_when_known():
    review_result = _result(duration_seconds=42.7)

    body = format_review_body(review_result)

    assert "- Review duration: 42.7s" in body


def test_format_review_body_shows_an_unknown_duration_placeholder_when_not_stamped():
    review_result = _result(duration_seconds=None)

    body = format_review_body(review_result)

    assert "- Review duration: unknown" in body


def test_format_review_body_generates_a_concise_title_for_a_long_problem_sentence():
    """A long, full-sentence Problem must not be repeated verbatim as the finding heading."""
    long_problem = (
        "The file handle opened with f = open(path) at the top of load_settings is "
        "never closed anywhere in this function, which leaks a file descriptor on "
        "every call and will eventually exhaust the process's descriptor limit."
    )
    review_result = _result(
        blocking_findings=[_finding(problem=long_problem)],
        recommendation="REQUEST_CHANGES",
    )

    body = format_review_body(review_result)
    heading_line = next(line for line in body.splitlines() if line.startswith("#### 1."))

    assert heading_line != f"#### 1. {long_problem}"
    title_words = heading_line.removeprefix("#### 1. ").split()
    assert len(title_words) <= 11  # <=10 words plus a possible "..." token
    assert long_problem in body  # the full Problem text still appears, just not as the heading


def test_format_review_body_next_action_for_a_complete_review_with_findings():
    review_result = _result(
        blocking_findings=[_finding()],
        recommendation="REQUEST_CHANGES",
    )

    body = format_review_body(review_result)

    assert "## Next Action" in body
    assert "Fix the reported findings. No retry needed." in body


def test_format_review_body_next_action_for_a_complete_clean_review():
    review_result = _result()

    body = format_review_body(review_result)

    assert "No retry needed." in body
    assert "Fix the reported findings" not in body


def test_format_review_body_next_action_for_a_partial_review_with_findings():
    review_result = _result(
        architecture_score=None,
        warnings=[_finding(severity="WARNING", review_category="code quality")],
        recommendation="INCOMPLETE",
        incomplete_categories=["architecture", "security", "rag_search", "performance"],
    )

    body = format_review_body(review_result)

    assert (
        "Fix the reported findings and retry the review later because 4 reviewer "
        "categories did not complete."
    ) in body


def test_format_review_body_next_action_for_a_partial_review_with_no_findings():
    review_result = _result(
        architecture_score=None,
        recommendation="INCOMPLETE",
        incomplete_categories=["architecture"],
    )

    body = format_review_body(review_result)

    assert "Retry the review later because 1 reviewer category did not complete." in body


def test_format_review_body_what_to_do_next_retry_guidance_matches_completeness():
    complete = _result()
    partial = _result(
        architecture_score=None, recommendation="INCOMPLETE", incomplete_categories=["architecture"],
    )

    assert "- Retry guidance: No retry needed" in format_review_body(complete)
    assert "- Retry guidance: Retry later" in format_review_body(partial)


# ---- Check Run text (immediate start / live progress / final display) --


def test_format_check_run_start_shows_zero_of_six_and_every_reviewer_pending():
    title, summary = format_check_run_start()

    assert title == "Code review is running"
    assert "0/6 reviewers completed" in summary
    assert summary.count("⏳") == 6


def test_format_check_run_progress_reflects_partial_completion():
    diagnostics = [
        ReviewerDiagnostic(category="security", status="completed"),
        ReviewerDiagnostic(category="architecture", status="failed"),
    ]

    title, summary = format_check_run_progress(diagnostics)

    assert "2/6 reviewers completed" in title
    assert "✅ Security" in summary
    assert "⚠️ Architecture" in summary
    assert "⏳ Code Quality" in summary


def test_format_check_run_progress_never_hides_a_failed_reviewer():
    diagnostics = [ReviewerDiagnostic(category="performance", status="failed")]

    _title, summary = format_check_run_progress(diagnostics)

    assert "⚠️ Performance" in summary


def test_format_check_run_final_shows_recommendation_and_counts_for_a_complete_review():
    review_result = _result(
        blocking_findings=[_finding()],
        recommendation="REQUEST_CHANGES",
        reviewer_diagnostics=[
            ReviewerDiagnostic(category=category, status="completed")
            for category in (
                "security",
                "architecture",
                "code quality",
                "rag_search",
                "performance",
                "agent_harness",
            )
        ],
    )

    text = format_check_run_final(review_result)

    assert "Recommendation: REQUEST CHANGES" in text
    assert "Review status: Completed" in text
    assert "Findings: 1 (1 blocking, 0 warnings, 0 suggestions)" in text
    assert "Reviewers completed: 6/6" in text
    assert "Retry guidance" not in text


def test_format_check_run_final_names_failed_categories_and_gives_retry_guidance():
    review_result = _result(
        architecture_score=None,
        recommendation="INCOMPLETE",
        incomplete_categories=["architecture", "performance"],
    )

    text = format_check_run_final(review_result)

    assert "Review status: Partially completed" in text
    assert "Categories that did not complete: architecture, performance" in text
    assert "Retry guidance: Retry later" in text


def test_format_check_run_final_always_shows_the_published_status_line():
    """This function is only ever called after a successful publish, so the
    "published" status line and its explanatory sentence must always appear -
    GitHub does not reliably live-refresh the Conversation timeline, so the
    Check Run must say plainly that the review is already there."""
    review_result = _result()

    text = format_check_run_final(review_result)
    lines = text.splitlines()

    assert lines[0] == "✅ Code Review published"
    assert "The full DataNet Code Review has been published to the PR Conversation." in text


def test_format_check_run_final_includes_a_direct_link_to_the_published_review():
    review_result = _result()

    text = format_check_run_final(
        review_result,
        review_url="https://github.com/example-org/example-repo/pull/15#pullrequestreview-999",
    )

    assert (
        "[View full review]"
        "(https://github.com/example-org/example-repo/pull/15#pullrequestreview-999)"
    ) in text


def test_format_check_run_final_omits_the_link_line_when_no_url_is_available():
    """The link is only included "when possible" - its absence must not break the rest of the text."""
    review_result = _result()

    text = format_check_run_final(review_result, review_url=None)

    assert "View full review" not in text
    assert "✅ Code Review published" in text
    assert "Recommendation: APPROVE" in text
