"""Validates the real config/component_mapping.yaml, not a test fixture."""

from pathlib import Path

from review_agent.models.finding import Component
from review_agent.tools.component_classifier import classify_file, load_mapping

_MAPPING_PATH = Path("config/component_mapping.yaml")


def test_component_mapping_parses_without_error():
    rules = load_mapping(_MAPPING_PATH)

    assert len(rules) > 0


def test_component_mapping_has_no_duplicate_patterns():
    rules = load_mapping(_MAPPING_PATH)

    patterns = [rule.pattern for rule in rules]

    assert len(patterns) == len(set(patterns))


def test_component_mapping_rules_resolve_to_valid_component_members():
    rules = load_mapping(_MAPPING_PATH)

    for rule in rules:
        assert isinstance(rule.component, Component)
        assert rule.component in Component


def test_component_mapping_classifies_real_datanet_directories():
    """The real example-org/example-repo layout, not just the services/* fixture."""
    rules = load_mapping(_MAPPING_PATH)

    assert classify_file("agent-runtime/src/api.py", rules) == Component.AGENT_HARNESS
    assert classify_file("agent-runtime/auth/token.py", rules) == Component.AGENT_HARNESS
    assert classify_file("rag-retrieval-tool/src/search.py", rules) == Component.SEARCH_SERVICE
    assert classify_file("frontend/src/App.tsx", rules) == Component.FRONTEND
    # Unclear top-level directories stay Unknown rather than guessed at.
    assert classify_file("shared/datanet_logging/log.py", rules) == Component.UNKNOWN
    assert classify_file("teams/serving/x.py", rules) == Component.UNKNOWN
