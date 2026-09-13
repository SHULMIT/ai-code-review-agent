from pathlib import Path

import pytest

from review_agent.models.finding import Component, Severity
from review_agent.models.review_input import ReviewInput
from review_agent.tools.component_classifier import (
    DEFAULT_MAPPING_PATH,
    load_component_mapping,
)
from review_agent.services.finding_validation import (
    _normalize_severity,
    stamp_review_category,
    validate_findings,
)


def _review_input(**overrides) -> ReviewInput:
    values = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": "abc123",
        "included_diff": (
            "diff --git a/a.py b/a.py\n"
            "index 0000000..1111111 100644\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,3 +1,5 @@\n"
            "+import os\n"
            "+os.system(cmd)\n"
        ),
        "diff_truncated": False,
        "included_files": ["a.py"],
        "omitted_files": ["b.py"],
        "repository_context": {"results": []},
        "documentation_context": [],
        "coverage_notes": [],
    }
    values.update(overrides)
    return ReviewInput(**values)


def _valid_raw_finding(**overrides) -> dict:
    values = {
        "severity": "BLOCKING",
        "review_category": "security",
        "file": "a.py",
        "line": 2,
        "problem": "Command injection via os.system.",
        "why_it_matters": "Untrusted input reaches a shell command.",
        "suggested_fix": "Use subprocess with a list of args, never shell=True.",
    }
    values.update(overrides)
    return values


def test_validate_findings_accepts_a_fully_valid_finding():
    review_input = _review_input()

    accepted, rejections = validate_findings([_valid_raw_finding()], review_input)

    assert len(accepted) == 1
    assert accepted[0].file == "a.py"
    assert rejections == []


def test_validate_findings_rejects_a_nonexistent_file():
    review_input = _review_input()
    raw = _valid_raw_finding(file="never/touched.py", line=None)

    accepted, rejections = validate_findings([raw], review_input)

    assert accepted == []
    assert len(rejections) == 1
    assert "never/touched.py" in rejections[0]


def test_validate_findings_rejects_a_finding_missing_a_required_field():
    review_input = _review_input()
    raw = _valid_raw_finding(problem="   ")

    accepted, rejections = validate_findings([raw], review_input)

    assert accepted == []
    assert "problem is empty" in rejections[0]


def test_validate_findings_rejects_a_finding_that_echoes_prompt_instructions_verbatim():
    """A finding copied verbatim from the reviewer's own prompt is not a real observation."""
    review_input = _review_input()
    system_instructions = (
        "Check specifically for:\n"
        "- A raw database query or driver call written in logic code instead of "
        "in the storage layer that exists for it.\n"
    )
    raw = _valid_raw_finding(
        problem="A raw database query or driver call written in logic code instead of "
        "in the storage layer that exists for it."
    )

    accepted, rejections = validate_findings(
        [raw], review_input, system_instructions=system_instructions
    )

    assert accepted == []
    assert len(rejections) == 1
    assert "echoed" in rejections[0]


def test_validate_findings_accepts_a_finding_merely_similar_to_prompt_instructions():
    """A real finding discussing the same concept in its own words is never rejected by the echo check."""
    review_input = _review_input()
    system_instructions = (
        "Check specifically for:\n"
        "- A raw database query or driver call written in logic code instead of "
        "in the storage layer that exists for it.\n"
    )
    raw = _valid_raw_finding(
        problem="Command injection via os.system.",
    )

    accepted, rejections = validate_findings(
        [raw], review_input, system_instructions=system_instructions
    )

    assert len(accepted) == 1
    assert rejections == []


def test_validate_findings_without_system_instructions_never_runs_the_echo_check():
    """Omitting system_instructions (the default) preserves every existing caller's behavior."""
    review_input = _review_input()
    raw = _valid_raw_finding(problem="Command injection via os.system.")

    accepted, rejections = validate_findings([raw], review_input)

    assert len(accepted) == 1
    assert rejections == []


def test_validate_findings_rejects_a_finding_that_fails_schema_validation():
    review_input = _review_input()
    raw = _valid_raw_finding(severity="NOT_A_REAL_SEVERITY")

    accepted, rejections = validate_findings([raw], review_input)

    assert accepted == []
    assert len(rejections) == 1
    assert "failed schema validation" in rejections[0]


def test_validate_findings_never_silently_drops_a_rejected_finding():
    review_input = _review_input()
    raw_findings = [_valid_raw_finding(), _valid_raw_finding(file="nowhere.py", line=None)]

    accepted, rejections = validate_findings(raw_findings, review_input)

    assert len(accepted) == 1
    assert len(rejections) == 1


def test_validate_findings_accepts_a_finding_with_no_file():
    review_input = _review_input()
    raw = _valid_raw_finding(file="", line=None)

    accepted, rejections = validate_findings([raw], review_input)

    assert len(accepted) == 1
    assert rejections == []


def test_validate_findings_rejects_a_missing_documentation_claim_when_readme_context_exists():
    review_input = _review_input(
        repository_context={
            "results": [{"file": "README.md", "line": 8, "snippet": "How to configure search."}]
        }
    )
    raw = _valid_raw_finding(
        problem="README documentation is missing for this configuration.",
        why_it_matters="Users cannot discover how to configure the service.",
        suggested_fix="Add a README section describing configuration.",
    )

    accepted, rejections = validate_findings([raw], review_input)

    assert accepted == []
    assert "contains documentation evidence" in rejections[0]


def test_validate_findings_allows_a_documentation_claim_when_no_documentation_evidence_exists():
    review_input = _review_input()
    raw = _valid_raw_finding(
        problem="README documentation is missing for this configuration.",
        why_it_matters="Users cannot discover how to configure the service.",
        suggested_fix="Add a README section describing configuration.",
    )

    accepted, rejections = validate_findings([raw], review_input)

    assert len(accepted) == 1
    assert rejections == []


def test_validate_findings_accepts_an_omitted_files_finding_without_a_line_check():
    review_input = _review_input()
    raw = _valid_raw_finding(file="b.py", line=9999)

    accepted, rejections = validate_findings([raw], review_input)

    assert len(accepted) == 1
    assert rejections == []


def test_validate_findings_rejects_a_line_far_from_any_changed_line():
    review_input = _review_input()
    raw = _valid_raw_finding(file="a.py", line=9999)

    accepted, rejections = validate_findings([raw], review_input)

    assert accepted == []
    assert "line 9999" in rejections[0]


def test_validate_findings_accepts_any_line_when_the_files_diff_has_no_parseable_hunk():
    review_input = _review_input(
        included_diff=(
            "diff --git a/a.py b/a.py\n"
            "Binary files a/a.py and b/a.py differ\n"
        )
    )
    raw = _valid_raw_finding(file="a.py", line=9999)

    accepted, rejections = validate_findings([raw], review_input)

    assert len(accepted) == 1
    assert rejections == []


def test_validate_findings_raises_on_a_non_list_input():
    review_input = _review_input()

    with pytest.raises(TypeError):
        validate_findings({"not": "a list"}, review_input)


def test_stamp_review_category_injects_the_field_into_every_dict():
    """Overwrites any existing value too - the model is never trusted to self-report this."""
    raw_findings = [_valid_raw_finding(), {"other": "shape"}]

    stamped = stamp_review_category(raw_findings, "architecture")

    assert all(finding["review_category"] == "architecture" for finding in stamped)


def test_stamp_review_category_leaves_a_non_dict_entry_untouched():
    stamped = stamp_review_category([_valid_raw_finding(), "not a dict", 42], "security")

    assert stamped[1] == "not a dict"
    assert stamped[2] == 42


def test_stamp_review_category_returns_non_list_input_unchanged():
    assert stamp_review_category("not a list", "security") == "not a list"
    assert stamp_review_category(None, "security") is None


# --- severity normalization: mitigates qwen2.5-coder:7b's observed ------------
# --- non-compliant free-text values - see DECISION-LOG.md ---------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("error", "BLOCKING"),
        ("Critical", "BLOCKING"),
        ("HIGH", "BLOCKING"),
        ("warning", "WARNING"),
        ("Medium", "WARNING"),
        ("info", "SUGGESTION"),
        (" Low ", "SUGGESTION"),
        ("suggestion", "SUGGESTION"),
    ],
)
def test_normalize_severity_maps_known_synonyms(raw, expected):
    """Every raw value here was actually observed from qwen2.5-coder:7b during this
    project's PR #137 manual verification run."""
    assert _normalize_severity(raw) == expected


def test_normalize_severity_returns_none_for_an_unrecognized_value():
    assert _normalize_severity("not_a_real_severity") is None


def test_validate_findings_accepts_a_previously_rejected_severity_value_after_normalization():
    """severity="error" was observed for real from qwen2.5-coder:7b in this session's
    PR #137 run (the Agent Harness reviewer), where it caused a schema-validation
    rejection. This isolates that exact value, paired with an otherwise-valid finding,
    to confirm normalization now lets it through.
    """
    review_input = _review_input()
    raw = _valid_raw_finding(severity="error")

    accepted, rejections = validate_findings([raw], review_input)

    assert len(accepted) == 1
    assert accepted[0].severity == Severity.BLOCKING
    assert rejections == []


def test_validate_findings_rejects_a_non_dict_entry_without_crashing_normalization():
    """_prepare_raw_finding must pass a non-dict entry through untouched, not raise."""
    review_input = _review_input()

    accepted, rejections = validate_findings(["not a dict", _valid_raw_finding()], review_input)

    assert len(accepted) == 1
    assert len(rejections) == 1
    assert "failed schema validation" in rejections[0]


# --- component: always derived deterministically, never read from the model - -
# --- see DECISION-LOG.md's "Component derived deterministically" entry --------


def test_validate_findings_derives_component_from_a_matching_mapping_rule():
    """services/agent/* is a real rule in config/component_mapping.yaml, mapping to
    AGENT_HARNESS - this loads the real mapping file rather than a test double, so
    the rules handed to validate_findings are the ones production hands it.

    validate_findings no longer reads that file itself: build_graph loads it once
    and passes the parsed rules down, so the loading happens here, in the caller,
    exactly as it now does in a reviewer node."""
    review_input = _review_input(
        included_diff=(
            "diff --git a/services/agent/orchestrator.py b/services/agent/orchestrator.py\n"
            "index 0000000..1111111 100644\n"
            "--- a/services/agent/orchestrator.py\n"
            "+++ b/services/agent/orchestrator.py\n"
            "@@ -1,3 +1,5 @@\n"
            "+import os\n"
            "+os.system(cmd)\n"
        ),
        included_files=["services/agent/orchestrator.py"],
    )
    raw = _valid_raw_finding(file="services/agent/orchestrator.py", line=2)

    accepted, rejections = validate_findings(
        [raw], review_input, load_component_mapping(DEFAULT_MAPPING_PATH).rules
    )

    assert len(accepted) == 1
    assert accepted[0].component == Component.AGENT_HARNESS
    assert rejections == []


def test_validate_findings_derives_component_unknown_for_a_file_matching_no_rule():
    """"a.py" (this file's default fixture) matches no rule in the real
    config/component_mapping.yaml, so it must derive to Unknown rather than being
    rejected or guessed."""
    review_input = _review_input()
    raw = _valid_raw_finding()

    accepted, rejections = validate_findings([raw], review_input)

    assert len(accepted) == 1
    assert accepted[0].component == Component.UNKNOWN
    assert rejections == []


def test_validate_findings_derives_component_unknown_for_a_finding_with_no_file():
    """A whole-PR-level finding (no file) has nothing to classify."""
    review_input = _review_input()
    raw = _valid_raw_finding(file="", line=None)

    accepted, rejections = validate_findings([raw], review_input)

    assert len(accepted) == 1
    assert accepted[0].component == Component.UNKNOWN


def test_validate_findings_derives_component_unknown_when_the_mapping_file_is_missing():
    """A missing/invalid component_mapping.yaml must degrade to Unknown for every
    finding, not raise - nodes/classify.py already records this exact failure into
    state.errors early in the graph, so this must not crash the reviewer node too."""
    review_input = _review_input()
    raw = _valid_raw_finding()

    # A missing file is turned into an empty rule set by
    # load_component_mapping at graph-build time; this is what every
    # reviewer then receives.
    accepted, rejections = validate_findings(
        [raw], review_input, load_component_mapping(Path("does/not/exist.yaml")).rules
    )

    assert len(accepted) == 1
    assert accepted[0].component == Component.UNKNOWN
    assert rejections == []


def test_validate_findings_ignores_any_component_the_model_still_sends():
    """component is always overwritten with the derived value, exactly like
    stamp_review_category already does for review_category - the model is never
    trusted to self-report either field, even if a raw finding still carries one
    (e.g. from an older cached response)."""
    review_input = _review_input()
    raw = _valid_raw_finding(component="Search Service")

    accepted, rejections = validate_findings([raw], review_input)

    assert len(accepted) == 1
    assert accepted[0].component == Component.UNKNOWN
    assert rejections == []
