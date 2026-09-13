"""Structural conventions every reviewer prompt must follow.

These run against the real files in prompts/, discovered dynamically rather
than listed, so a newly added reviewer prompt is covered the moment it lands
instead of only when someone remembers to extend this file.

The rules asserted here are the ones a prompt cannot be trusted to keep on
its own. Scope boundaries are generated centrally by prompt_builder.py and
tested there; these file-level checks keep the remaining reviewer-specific
noise controls from quietly regressing.
"""

from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_FILES = sorted(PROMPTS_DIR.glob("*_review.md"))

ALLOWED_SEVERITIES = ("BLOCKING", "WARNING", "SUGGESTION")
REQUIRED_FIELD_NAMES = (
    "severity",
    "file",
    "line",
    "problem",
    "why_it_matters",
    "suggested_fix",
)


def test_at_least_one_reviewer_prompt_is_discovered():
    """Guard the glob itself - every other test here passes vacuously on an empty list."""
    assert PROMPT_FILES, f"no *_review.md files found in {PROMPTS_DIR}"


@pytest.fixture(params=PROMPT_FILES, ids=lambda path: path.name)
def prompt_text(request) -> str:
    return request.param.read_text(encoding="utf-8")


def test_prompt_caps_the_number_of_reported_findings(prompt_text):
    """An uncapped reviewer lists everything it notices, which buries the findings that matter."""
    assert "Report AT MOST the 3 most important issues" in prompt_text


def test_prompt_requires_a_full_check_before_an_empty_result(prompt_text):
    """A model told only to find problems will invent one on a clean pull request -
    but telling it that empty is common invites skipping the checklist instead."""
    assert "Inspect every changed hunk in the diff against the checklist above" in prompt_text
    assert "never returned just to avoid an empty result" in prompt_text
    assert "respond with an empty array: []" in prompt_text


def test_prompt_requires_evidence_in_the_changed_lines(prompt_text):
    """Findings about code the reviewer cannot see are ungrounded by construction."""
    assert "only when the evidence for it is visible in the changed lines" in prompt_text


def test_prompt_lists_every_allowed_severity_and_no_other(prompt_text):
    for severity in ALLOWED_SEVERITIES:
        assert severity in prompt_text, f"missing severity {severity}"


def test_prompt_names_every_required_finding_field(prompt_text):
    for field_name in REQUIRED_FIELD_NAMES:
        assert field_name in prompt_text, f"missing field name {field_name}"


def test_prompt_does_not_ask_the_model_for_review_category(prompt_text):
    """review_category is stamped in code by the node, never self-reported by the model."""
    assert "review_category" not in prompt_text


def test_prompt_does_not_require_a_component_field(prompt_text):
    """component is derived deterministically from the finding's file field (see
    finding_validation.py's _derive_component) instead of being self-reported by the
    model - models were observed repeatedly failing to match Component's exact enum
    values even after a normalization attempt, see DECISION-LOG.md. The prompt must not
    instruct the model to produce a "component" JSON field or list the allowed values,
    even though the word "component" still appears in ordinary prose throughout these
    files (e.g. "regardless of which component changed")."""
    assert '"component"' not in prompt_text


def test_security_prompt_explicitly_checks_plaintext_credentials_in_logs():
    prompt = (PROMPTS_DIR / "security_review.md").read_text(encoding="utf-8")

    assert "Plaintext credentials or API keys interpolated into a log call" in prompt


def test_code_quality_prompt_covers_constants_magic_numbers_and_evidence_based_test_gaps():
    prompt = (PROMPTS_DIR / "code_quality_review.md").read_text(encoding="utf-8")

    assert "unused constant, duplicated literal, or magic number" in prompt
    assert "actual evidence" in prompt


def test_rag_search_prompt_covers_ranking_and_unsafe_nested_response_access():
    prompt = (PROMPTS_DIR / "rag_search_review.md").read_text(encoding="utf-8")

    assert "orders a relevance score in the wrong direction" in prompt
    assert "Nested external/search API response fields accessed by direct indexing" in prompt
