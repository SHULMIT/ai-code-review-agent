from pathlib import Path

from langgraph.graph import END, START, StateGraph

from review_agent.config import Settings
from review_agent.graph.state import ReplacedFindings, ReviewState
from review_agent.models.finding import Finding
from review_agent.nodes.deduplicate_findings import make_deduplicate_findings_node


def _settings() -> Settings:
    """Build node settings without reading a local environment file."""

    return Settings(
        _env_file=None,
        github_app_id=1,
        github_app_private_key_path=Path("secrets/test-private-key.pem"),
        github_webhook_secret="test-webhook-secret",
    )


def _finding(**overrides) -> Finding:
    values = {
        "severity": "WARNING",
        "component": "Agent Harness / LangGraph",
        "review_category": "security",
        "file": "a.py",
        "line": 7,
        "problem": "Hardcoded secret.",
        "why_it_matters": "Exposed to anyone with source access.",
        "suggested_fix": "Read it from configuration instead.",
    }
    values.update(overrides)
    return Finding(**values)


def _field(result, name: str):
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


async def test_deduplicate_findings_node_returns_a_replaced_findings_wrapper():
    node = make_deduplicate_findings_node(_settings())
    state = ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        findings=[_finding(severity="BLOCKING"), _finding(severity="WARNING")],
    )

    result = await node(state)

    assert isinstance(result["findings"], ReplacedFindings)
    assert len(result["findings"]) == 1
    assert result["findings"][0].severity.value == "BLOCKING"


async def test_deduplicate_findings_node_replaces_rather_than_appends_end_to_end():
    """A tiny real graph proves the reducer actually replaces, not just that the node intends to."""

    def seed_findings_node(state: ReviewState) -> dict:
        return {"findings": [_finding(severity="BLOCKING"), _finding(severity="WARNING")]}

    graph = StateGraph(ReviewState)
    graph.add_node("seed_findings", seed_findings_node)
    graph.add_node("deduplicate_findings", make_deduplicate_findings_node(_settings()))
    graph.add_edge(START, "seed_findings")
    graph.add_edge("seed_findings", "deduplicate_findings")
    graph.add_edge("deduplicate_findings", END)
    compiled = graph.compile()

    result = await compiled.ainvoke(ReviewState(repo="example-org/example-repo", pr_number=1))

    findings = _field(result, "findings")
    # Without replacement, the reducer would add the deduplicated list to the
    # two seeded findings, leaving 3 (or more) instead of the merged 1.
    assert len(findings) == 1
    assert findings[0].severity.value == "BLOCKING"
