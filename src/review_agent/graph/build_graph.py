from pathlib import Path

import httpx
from langgraph.graph import END, START, StateGraph

from review_agent.config import Settings
from review_agent.graph.state import ReviewState
from review_agent.nodes.all_zero_verification import make_all_zero_verification_node
from review_agent.nodes.classify import make_classify_node
from review_agent.nodes.deduplicate_findings import make_deduplicate_findings_node
from review_agent.nodes.finalize_review import make_finalize_review_node
from review_agent.nodes.gather_context import make_gather_context_node
from review_agent.nodes.load_pr import make_load_pr_node
from review_agent.nodes.review_agent_harness import make_agent_harness_reviewer_node
from review_agent.nodes.review_architecture import make_architecture_reviewer_node
from review_agent.nodes.review_performance import make_performance_reviewer_node
from review_agent.nodes.review_quality import make_quality_reviewer_node
from review_agent.nodes.review_rag_search import make_rag_search_reviewer_node
from review_agent.nodes.review_security import make_security_reviewer_node
from review_agent.nodes.static_checks import make_static_checks_node
from review_agent.tools.component_classifier import (
    DEFAULT_MAPPING_PATH,
    load_component_mapping,
)

_SERVICE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOCS_ROOT = _SERVICE_ROOT / "docs"

# The six reviewer nodes read only state.review_input and state.components
# (both set upstream, before any of them run) and write only to
# state.findings/state.errors/state.failed_categories, whose reducers
# (_merge_findings and operator.add - see graph/state.py) are commutative and
# order-independent. No reviewer node reads a field only another reviewer
# node would have set (re-verified directly against all six node bodies).
# They therefore have no ordering dependency on each other and are wired as
# parallel fan-out/fan-in branches below, not a sequential chain.
_REVIEWER_NODE_NAMES = (
    "review_security",
    "review_architecture",
    "review_quality",
    "review_rag_search",
    "review_performance",
    "review_agent_harness",
)


def build_graph(
    token: str,
    model_settings: Settings,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    local_docs_root: Path = DEFAULT_DOCS_ROOT,
    client: httpx.AsyncClient | None = None,
):
    """Assemble and compile the review graph.

    client is an optional pooled httpx.AsyncClient shared by every node in
    this graph that makes an HTTP call (load_pr, all six reviewers,
    deduplicate_findings) instead of each one opening its own short-lived
    client - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5. review_launcher.py
    is the real caller that constructs and passes one; every existing caller
    that omits it keeps today's per-call-client behavior unchanged.

    START -> load_pr -> classify -> static_checks -> gather_context, which
    then fans out to all six reviewers (review_security, review_architecture,
    review_quality, review_rag_search, review_performance,
    review_agent_harness) as parallel branches - see _REVIEWER_NODE_NAMES's
    comment for why this is safe - all six fan back in to
    deduplicate_findings, then all_zero_verification, then finalize_review ->
    END. all_zero_verification is a no-op unless all six reviewers completed
    and still found nothing on a non-trivial diff - see
    nodes/all_zero_verification.py. All six review categories from SCOPE.md
    are wired. static_checks runs before gather_context and makes no Model
    Service call, so its findings survive a failure in context retrieval or
    in every reviewer after it.

    This closes DECISION-LOG.md's "Cumulative reviewer-call timeout" entry:
    the six reviewers no longer make their Model Service calls one after
    another, so a review run's wall-clock time no longer grows linearly with
    reviewer count. deduplicate_findings runs once, after every reviewer has
    completed (LangGraph will not start it until all six incoming edges are
    satisfied), and merges findings that likely describe the same underlying
    issue regardless of which reviewer produced them - see
    finding_deduplication.py and DECISION-LOG.md's "Reviewer prompt scope
    boundaries are not enforced" entry. finalize_review runs last,
    deterministically computing scores and a recommendation from the
    deduplicated findings - see DECISION-LOG.md's "Scoring formula and
    category-to-score mapping interpreted" entry.
    """
    # The component mapping is read here, once, and handed to every node that
    # needs it. It used to be re-read from disk seven times per review run -
    # once by classify and once inside validate_findings for each of the six
    # reviewers - for a file that cannot change mid-run. Loading it in one
    # owner also means every reviewer in a run classifies against the same
    # snapshot, and a broken file is reported once rather than silently
    # swallowed by six reviewers.
    component_mapping = load_component_mapping(mapping_path)

    graph = StateGraph(ReviewState)
    graph.add_node("load_pr", make_load_pr_node(token, client=client))
    graph.add_node("classify", make_classify_node(component_mapping))
    graph.add_node("static_checks", make_static_checks_node())
    graph.add_node(
        "gather_context",
        make_gather_context_node(
            token,
            local_docs_root,
            max_repository_context_characters=model_settings.max_repository_context_characters,
            excluded_diff_paths=model_settings.excluded_diff_path_list(),
        ),
    )
    graph.add_node(
        "review_architecture", make_architecture_reviewer_node(
            model_settings, component_mapping, client=client
        )
    )
    graph.add_node(
        "review_security", make_security_reviewer_node(
            model_settings, component_mapping, client=client
        )
    )
    graph.add_node(
        "review_quality", make_quality_reviewer_node(
            model_settings, component_mapping, client=client
        )
    )
    graph.add_node(
        "review_rag_search", make_rag_search_reviewer_node(
            model_settings, component_mapping, client=client
        )
    )
    graph.add_node(
        "review_performance", make_performance_reviewer_node(
            model_settings, component_mapping, client=client
        )
    )
    graph.add_node(
        "review_agent_harness", make_agent_harness_reviewer_node(
            model_settings, component_mapping, client=client
        )
    )
    graph.add_node(
        "deduplicate_findings", make_deduplicate_findings_node(model_settings, client=client)
    )
    graph.add_node(
        "all_zero_verification", make_all_zero_verification_node(
            model_settings, component_mapping, client=client
        )
    )
    graph.add_node("finalize_review", make_finalize_review_node())
    graph.add_edge(START, "load_pr")
    graph.add_edge("load_pr", "classify")
    graph.add_edge("classify", "static_checks")
    graph.add_edge("static_checks", "gather_context")
    for reviewer_node_name in _REVIEWER_NODE_NAMES:
        graph.add_edge("gather_context", reviewer_node_name)
        graph.add_edge(reviewer_node_name, "deduplicate_findings")
    graph.add_edge("deduplicate_findings", "all_zero_verification")
    graph.add_edge("all_zero_verification", "finalize_review")
    graph.add_edge("finalize_review", END)
    return graph.compile()
