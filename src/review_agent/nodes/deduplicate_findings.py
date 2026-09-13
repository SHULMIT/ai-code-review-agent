"""LangGraph node: consolidate every reviewer's findings into one deduplicated list."""

import httpx

from review_agent.config import Settings
from review_agent.graph.state import ReplacedFindings, ReviewState
from review_agent.services.finding_deduplication import deduplicate_findings


# LangGraph nodes must accept only `state` - the Model Service settings are bound here via closure before the node is registered.
def make_deduplicate_findings_node(
    model_settings: Settings, client: httpx.AsyncClient | None = None
):
    """Build the deduplicate_findings node.

    Takes model_settings now that deduplication's similarity check can make
    a real embedding-service call (see finding_deduplication.py) - it was
    previously the one node in this graph needing no external dependency.
    client is the same optional shared httpx.AsyncClient every other node
    in this graph can take - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5.
    """

    async def deduplicate_findings_node(state: ReviewState) -> dict:
        """Consolidate every reviewer's accumulated findings into one merged list.

        Unlike every other node writing to `findings`, this one REPLACES the
        accumulated list instead of adding to it - it consolidates what
        reviewers already produced rather than contributing new findings of
        its own. Wrapping the result in ReplacedFindings tells
        ReviewState's reducer to overwrite rather than concatenate;
        returning a plain list here would double every finding instead of
        merging duplicates.
        """
        deduplicated = await deduplicate_findings(state.findings, model_settings, client=client)
        return {"findings": ReplacedFindings(deduplicated)}

    return deduplicate_findings_node
