from review_agent.config import Settings
from review_agent.graph.build_graph import (
    DEFAULT_DOCS_ROOT,
    DEFAULT_MAPPING_PATH,
    _REVIEWER_NODE_NAMES,
    build_graph,
)
from review_agent.graph.state import ReviewState
from review_agent.models.model_output import ModelReviewOutput
from review_agent.nodes import load_pr as load_pr_module
from review_agent.nodes import review_architecture as review_architecture_module
from review_agent.nodes import review_agent_harness as review_agent_harness_module
from review_agent.nodes import review_performance as review_performance_module
from review_agent.nodes import review_quality as review_quality_module
from review_agent.nodes import review_rag_search as review_rag_search_module
from review_agent.nodes import review_security as review_security_module
from review_agent.tools.github_tool import PRData


def _fake_model_settings() -> Settings:
    return Settings(
        _env_file=None,
        github_app_id=1,
        github_app_private_key_path="secrets/test-private-key.pem",
        github_webhook_secret="fake-webhook-secret",
    )


def test_build_graph_uses_paths_independent_of_current_directory(monkeypatch):
    monkeypatch.chdir(DEFAULT_DOCS_ROOT)

    graph = build_graph("fake-token", _fake_model_settings())

    assert type(graph).__name__ == "CompiledStateGraph"
    assert DEFAULT_MAPPING_PATH.is_file()
    assert DEFAULT_DOCS_ROOT.is_dir()


def test_build_graph_wires_all_six_reviewers_as_a_parallel_fan_out_fan_in():
    """The six reviewers must run in parallel, not a sequential chain.

    Asserts the graph's actual edge structure, not just an end-to-end
    result - a future edit that accidentally reverts to chaining reviewer
    nodes one after another would still produce a correct ReviewResult, so
    only inspecting the wiring itself catches that regression.
    """
    graph = build_graph("fake-token", _fake_model_settings())
    edges = graph.get_graph().edges

    fanned_out_from_gather_context = {
        edge.target for edge in edges if edge.source == "gather_context"
    }
    fanned_in_to_deduplicate_findings = {
        edge.source for edge in edges if edge.target == "deduplicate_findings"
    }

    assert fanned_out_from_gather_context == set(_REVIEWER_NODE_NAMES)
    assert fanned_in_to_deduplicate_findings == set(_REVIEWER_NODE_NAMES)

    # No reviewer may have an edge to or from another reviewer - that would
    # be a sequential dependency hiding alongside the parallel wiring.
    reviewer_to_reviewer_edges = [
        edge
        for edge in edges
        if edge.source in _REVIEWER_NODE_NAMES and edge.target in _REVIEWER_NODE_NAMES
    ]
    assert reviewer_to_reviewer_edges == []


async def test_build_graph_produces_a_review_result_end_to_end(monkeypatch):
    """A full run - GitHub and Model Service faked, everything else real - must reach finalize_review."""

    async def fake_get_pr(repo, pr_number, token):
        return PRData(
            head_sha="abc123",
            diff=(
                "diff --git a/a.py b/a.py\n"
                "index 0000000..1111111 100644\n"
                "--- a/a.py\n"
                "+++ b/a.py\n"
                "@@ -1,3 +1,3 @@\n"
                "+print('hello')\n"
            ),
            changed_files=["a.py"],
            title="A test PR",
        )

    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(load_pr_module, "get_pr", fake_get_pr)
    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)
    monkeypatch.setattr(review_architecture_module, "call_model", fake_call_model)
    monkeypatch.setattr(review_quality_module, "call_model", fake_call_model)
    monkeypatch.setattr(review_rag_search_module, "call_model", fake_call_model)
    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)
    monkeypatch.setattr(review_agent_harness_module, "call_model", fake_call_model)

    graph = build_graph("fake-token", _fake_model_settings())
    result = await graph.ainvoke(ReviewState(repo="example-org/example-repo", pr_number=1))

    review_result = result["review_result"]
    assert review_result is not None
    assert review_result.recommendation == "APPROVE"
    assert review_result.architecture_score == 100
    assert review_result.code_quality_score == 100
    assert review_result.production_readiness_score == 100
    assert review_result.reviewed_head_sha == "abc123"


def test_build_graph_reads_the_component_mapping_exactly_once(monkeypatch):
    """One owner loads the mapping; nothing downstream reaches for the file again.

    It used to be read seven times per review run - once by classify, and once
    inside validate_findings for each of the six reviewers - for a file that
    cannot change mid-run. Counting the reads is the only way to catch a
    regression: the findings are identical either way, so no assertion about
    output would notice a reviewer quietly loading it again.
    """
    from review_agent.tools import component_classifier

    read_count = {"value": 0}
    real_load_mapping = component_classifier.load_mapping

    def counting_load_mapping(path):
        read_count["value"] += 1
        return real_load_mapping(path)

    monkeypatch.setattr(component_classifier, "load_mapping", counting_load_mapping)

    build_graph("fake-token", _fake_model_settings())

    assert read_count["value"] == 1, (
        f"the mapping file should be read once per graph build, was read "
        f"{read_count['value']} times"
    )


def test_a_broken_component_mapping_does_not_prevent_building_the_graph(tmp_path):
    """A malformed mapping degrades classification to Unknown; it must not stop the run.

    The opposite of a missing reviewer prompt, which does fail graph
    construction - a reviewer without its instructions cannot work at all,
    while an unusable mapping only costs the component label on each finding.
    """
    broken_mapping = tmp_path / "component_mapping.yaml"
    broken_mapping.write_text("- pattern: 'x'\n  component: NOT_A_REAL_COMPONENT\n", encoding="utf-8")

    graph = build_graph("fake-token", _fake_model_settings(), mapping_path=broken_mapping)

    assert type(graph).__name__ == "CompiledStateGraph"
