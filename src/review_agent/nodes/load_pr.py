from typing import Callable, Coroutine

import httpx

from review_agent.graph.state import ReviewState
from review_agent.tools.github_tool import GitHubToolError, get_pr


# LangGraph nodes must accept only `state` - external dependencies like tokens are bound here via closure before the node is registered.
def make_load_pr_node(
    token: str, client: httpx.AsyncClient | None = None
) -> Callable[[ReviewState], Coroutine[None, None, dict]]:
    """Build the load_pr node with the GitHub token bound via closure.

    client is an optional pooled httpx.AsyncClient shared across a whole
    review run - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5. Only
    passed on to get_pr when actually given, so every existing caller that
    omits it keeps get_pr's original per-call-client behavior unchanged.
    """

    async def load_pr_node(state: ReviewState) -> dict:
        """Fetch the PR's head SHA, diff, and changed files from GitHub.

        On a GitHubToolError, does not raise - returns the failure into the
        state's errors field instead, so a GitHub outage doesn't crash the review.
        """
        call_kwargs = {} if client is None else {"client": client}
        try:
            pr_data = await get_pr(state.repo, state.pr_number, token, **call_kwargs)
        except GitHubToolError as exc:
            return {"errors": [str(exc)]}

        return {
            "head_sha": pr_data.head_sha,
            "diff": pr_data.diff,
            "changed_files": pr_data.changed_files,
        }

    return load_pr_node
