import tempfile
from pathlib import Path

from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component
from review_agent.nodes.classify import make_classify_node
from review_agent.tools.component_classifier import load_component_mapping

MAPPING_PATH = Path("config/component_mapping.yaml")


def test_classify_node_dedupes_same_component():
    classify_node = make_classify_node(load_component_mapping(MAPPING_PATH))
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        changed_files=[
            "services/mock_agent/search_helper.py",
            "services/mock_agent/other.py",
        ],
    )

    result = classify_node(state)

    assert result == {"components": [Component.AGENT_HARNESS]}


def test_classify_node_multiple_components_no_duplicates():
    classify_node = make_classify_node(load_component_mapping(MAPPING_PATH))
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        changed_files=[
            "services/mock_agent/search_helper.py",
            "services/frontend/app.js",
            "services/mock_agent/other.py",
        ],
    )

    result = classify_node(state)

    assert result == {"components": [Component.AGENT_HARNESS, Component.FRONTEND]}


def test_classify_node_no_changed_files_returns_error():
    classify_node = make_classify_node(load_component_mapping(MAPPING_PATH))
    state = ReviewState(repo="example-org/example-repo", pr_number=1)

    result = classify_node(state)

    assert result["components"] == []
    assert result["errors"] == ["classify_node: no changed_files in state"]


def test_classify_node_returns_error_for_a_malformed_mapping_file():
    with tempfile.TemporaryDirectory(prefix="review_agent_classify_test_") as tmp_dir:
        mapping_path = Path(tmp_dir) / "component_mapping.yaml"
        mapping_path.write_text(
            "- pattern: services/mystery/*\n  component: NOT_A_REAL_COMPONENT\n",
            encoding="utf-8",
        )
        classify_node = make_classify_node(load_component_mapping(mapping_path))
        state = ReviewState(
            repo="example-org/example-repo",
            pr_number=1,
            changed_files=["services/mystery/thing.py"],
        )

        result = classify_node(state)

        assert result["components"] == []
        assert len(result["errors"]) == 1
        assert result["errors"][0].startswith("classify_node: invalid component mapping:")
