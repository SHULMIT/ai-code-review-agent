import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from review_agent.models.finding import Component
from review_agent.tools.component_classifier import ComponentRule, classify_file, load_mapping

SAMPLE_RULES = [
    ComponentRule(pattern="services/mock_agent/*", component=Component.AGENT_HARNESS),
    ComponentRule(pattern="services/mock_search/*", component=Component.SEARCH_SERVICE),
    ComponentRule(pattern="services/frontend/*", component=Component.FRONTEND),
]


def test_classify_file_matches_first_rule():
    result = classify_file("services/mock_agent/search_helper.py", SAMPLE_RULES)
    assert result == Component.AGENT_HARNESS


def test_classify_file_matches_later_rule():
    result = classify_file("services/frontend/app.js", SAMPLE_RULES)
    assert result == Component.FRONTEND


def test_classify_file_no_match_returns_unknown():
    result = classify_file("README.md", SAMPLE_RULES)
    assert result == Component.UNKNOWN


def test_classify_file_double_star_matches_nested_path():
    rules = [ComponentRule(pattern="**/Dockerfile", component=Component.INFRASTRUCTURE)]

    result = classify_file("services/agent/Dockerfile", rules)

    assert result == Component.INFRASTRUCTURE


def test_classify_file_single_star_does_not_cross_slash():
    # services/mock_agent/* must not match a nested subdirectory - only files
    # directly inside mock_agent/, confirming '*' is path-aware, not naive fnmatch.
    result = classify_file("services/mock_agent/sub/other.py", SAMPLE_RULES)
    assert result == Component.UNKNOWN


def test_classify_file_first_matching_rule_wins_when_multiple_rules_match():
    # services/agent/Dockerfile matches both rules below - the general
    # services/agent/* rule (listed first) and the **/Dockerfile rule
    # (listed second). classify_file's docstring promises the rules are
    # matched "in order" with "the first match wins," so the first rule's
    # component must be returned, not the second.
    rules = [
        ComponentRule(pattern="services/agent/*", component=Component.AGENT_HARNESS),
        ComponentRule(pattern="**/Dockerfile", component=Component.INFRASTRUCTURE),
    ]

    result = classify_file("services/agent/Dockerfile", rules)

    assert result == Component.AGENT_HARNESS


def test_load_mapping_reads_the_real_config_file():
    rules = load_mapping(Path("config/component_mapping.yaml"))

    assert isinstance(rules, list)
    assert len(rules) == 12
    assert rules[0] == ComponentRule(
        pattern="services/mock_agent/*", component=Component.AGENT_HARNESS
    )


def test_load_mapping_rejects_an_unknown_component_name():
    with tempfile.TemporaryDirectory(prefix="review_agent_mapping_test_") as tmp_dir:
        mapping_file = Path(tmp_dir) / "component_mapping.yaml"
        mapping_file.write_text(
            "- pattern: services/mystery/*\n  component: NOT_A_REAL_COMPONENT\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            load_mapping(mapping_file)


def test_load_mapping_rejects_a_rule_missing_the_component_key():
    with tempfile.TemporaryDirectory(prefix="review_agent_mapping_test_") as tmp_dir:
        mapping_file = Path(tmp_dir) / "component_mapping.yaml"
        mapping_file.write_text("- pattern: services/mystery/*\n", encoding="utf-8")

        with pytest.raises(ValidationError):
            load_mapping(mapping_file)


def test_load_mapping_rejects_a_rule_missing_the_pattern_key():
    with tempfile.TemporaryDirectory(prefix="review_agent_mapping_test_") as tmp_dir:
        mapping_file = Path(tmp_dir) / "component_mapping.yaml"
        mapping_file.write_text("- component: AGENT_HARNESS\n", encoding="utf-8")

        with pytest.raises(ValidationError):
            load_mapping(mapping_file)
