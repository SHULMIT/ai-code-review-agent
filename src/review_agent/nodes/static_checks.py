"""LangGraph node: deterministic pull-request checks, run without any Model Service call."""

from review_agent.graph.state import ReviewState
from review_agent.services.static_checks import run_static_checks


# LangGraph nodes must accept only `state` - kept as a factory for consistency with every other node in this graph, even though this one binds no external dependency.
def make_static_checks_node():
    """Build the static_checks node.

    Unlike every reviewer node, this one makes no Model Service call, so it
    adds no measurable time to a review run and cannot be affected by model
    latency, model failure, or malformed model output. It runs early, before
    gather_context, so its findings exist even when context retrieval or
    every reviewer afterwards fails.

    It deliberately does not mark failed_categories. That field means "a
    reviewer category could not run"; these checks are not a category, and
    the categories they stamp their findings with (security, code quality)
    are owned by reviewer nodes that report their own failures.
    """

    def static_checks_node(state: ReviewState) -> dict:
        """Run the deterministic checks against the raw diff and changed-file list.

        Depends only on load_pr's output, not on review_input, which is why
        it can run before gather_context. Any unexpected failure is recorded
        into state.errors rather than raised, so a bug in a check can never
        take down a review run - the same rule every other node follows.
        """
        if state.diff is None or state.changed_files is None:
            return {"errors": ["static checks: no diff available (upstream failure)"]}

        try:
            findings = run_static_checks(state.diff, state.changed_files)
        except Exception as exc:  # noqa: BLE001 - a check bug must not end the run
            return {"errors": [f"static checks: {type(exc).__name__}: {exc}"]}

        return {"findings": findings}

    return static_checks_node
