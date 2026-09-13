"""Manual verification script: run the review graph against a real GitHub PR.

Not part of the automated test suite - exercises the graph end to end
against the real GitHub API with the available GitHub credential. It is a
dry run by default; --publish explicitly submits the finished review.
"""

import asyncio
import argparse

from pydantic import ValidationError

from review_agent.config import ManualRunSettings, Settings
from review_agent.graph.build_graph import build_graph
from review_agent.graph.state import ReviewState
from review_agent.services.review_launcher import build_publish_block_reason
from review_agent.tools.github_auth import InstallationTokenProvider
from review_agent.tools.github_publisher import (
    GitHubPublisherError,
    _github_event_for,
    publish_review,
)
from review_agent.tools.github_tool import GitHubToolError, get_pr_head_sha


def _field(result, name: str):
    """Read a field off the graph result, whether it's a dict or a ReviewState."""
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


def _parse_args() -> argparse.Namespace:
    """Parse the target PR and the explicit opt-in required for publishing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_number", nargs="?", type=int, default=105)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="submit the review after verifying the PR head SHA; dry run is the default",
    )
    return parser.parse_args()


async def _get_github_token(settings: ManualRunSettings) -> str:
    """Return an App token when configured, otherwise use the local PAT fallback."""

    if settings.github_app_installation_id is not None:
        if (
            settings.github_app_id is None
            or settings.github_app_private_key_path is None
        ):
            raise RuntimeError(
                "GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY_PATH are required "
                "when GITHUB_APP_INSTALLATION_ID is set."
            )
        token_provider = InstallationTokenProvider(
            app_id=settings.github_app_id,
            private_key_path=settings.github_app_private_key_path,
        )
        return await token_provider.get_token(settings.github_app_installation_id)

    if settings.github_token is None:
        raise RuntimeError(
            "Set GITHUB_TOKEN or GITHUB_APP_INSTALLATION_ID "
            "for the manual verification script."
        )

    # Temporary: the personal App installation cannot access example-org/example-repo yet.
    return settings.github_token.get_secret_value()


async def _publish_if_requested(
    should_publish: bool,
    repo: str,
    pr_number: int,
    result: ReviewState | dict,
    github_token: str,
) -> str:
    """Publish only an intact review whose current GitHub head matches its analyzed SHA."""

    if not should_publish:
        return "DRY RUN: review was not published (pass --publish to submit it)."

    review_result = _field(result, "review_result")
    if review_result is None:
        return "PUBLISH BLOCKED: review_result was not computed."

    reviewed_head_sha = _field(result, "head_sha")
    block_reason = build_publish_block_reason(
        _field(result, "review_input"), reviewed_head_sha, review_result
    )
    if block_reason is not None:
        return f"PUBLISH BLOCKED: {block_reason}"
    if review_result.reviewed_head_sha != reviewed_head_sha:
        return (
            "PUBLISH BLOCKED: final result head SHA does not match the SHA "
            "reviewed by the graph."
        )

    try:
        current_head_sha = await get_pr_head_sha(repo, pr_number, github_token)
    except GitHubToolError as exc:
        return f"PUBLISH BLOCKED: unable to verify the current PR head SHA: {exc}"

    if current_head_sha != reviewed_head_sha:
        return (
            "PUBLISH BLOCKED: the PR head changed during review "
            f"(reviewed {reviewed_head_sha}, current {current_head_sha})."
        )

    try:
        await publish_review(
            repo, pr_number, reviewed_head_sha, review_result, github_token
        )
    except GitHubPublisherError as exc:
        return f"PUBLISH FAILED: {exc}"

    return (
        "PUBLISHED: GitHub Pull Request Review submitted for "
        f"{repo}#{pr_number} at {reviewed_head_sha}."
    )


async def main() -> None:
    try:
        settings = ManualRunSettings()
    except ValidationError:
        raise RuntimeError(
            "Manual-run configuration is invalid. Check the local .env file."
        ) from None

    args = _parse_args()
    github_token = await _get_github_token(settings)
    pr_number = args.pr_number
    initial_state = ReviewState(repo="example-org/example-repo", pr_number=pr_number)
    # ManualRunSettings has no App-only GitHub fields required, but build_graph's
    # security reviewer needs a full Settings for Model Service configuration -
    # this script uses its own token above, so these placeholders are unused.
    model_settings = Settings(
        github_app_id=1,
        github_app_private_key_path="secrets/unused-for-manual-run.pem",
        github_webhook_secret="unused-for-manual-run",
    )
    graph = build_graph(github_token, model_settings)

    # load_pr_node is async (it awaits httpx calls), so the graph needs
    # ainvoke - the sync invoke() cannot run an async-only node.
    result = await graph.ainvoke(initial_state)

    print("diff length:", len(_field(result, "diff") or ""))
    print()
    print("changed_files:", _field(result, "changed_files"))
    review_input = _field(result, "review_input")
    if review_input is not None:
        print("diff budget included_files:", review_input.included_files)
        print("diff budget omitted_files:", review_input.omitted_files)
        print("diff budget coverage_notes:", review_input.coverage_notes)
    print("head_sha:", _field(result, "head_sha"))
    print("components:", _field(result, "components"))
    print()
    print("errors:")
    for error in _field(result, "errors") or []:
        print(f"  - {error}")
    print()
    repo_context = _field(result, "repository_context") or {}
    doc_context = _field(result, "documentation_context") or {}
    print("repository_context results:", len(repo_context.get("results", [])))
    print("documentation_context results:", len(doc_context.get("results", [])))
    print()
    findings = _field(result, "findings") or []
    print(f"findings ({len(findings)}):")
    for finding in findings:
        print(f"  - [{finding.severity.value}] {finding.component.value} {finding.file}:{finding.line}")
        print(f"    problem: {finding.problem}")
        print(f"    why it matters: {finding.why_it_matters}")
        print(f"    suggested fix: {finding.suggested_fix}")
        print(f"    review_category: {finding.review_category}")
        print(f"    source_document: {finding.source_document}")
        print(f"    source_snippet: {finding.source_snippet}")
        print(f"    merge_note: {finding.merge_note}")

    print()
    review_result = _field(result, "review_result")
    print("review_result:")
    if review_result is None:
        print("  None")
    else:
        print(f"  pr_summary: {review_result.pr_summary}")
        print(f"  architecture_score: {review_result.architecture_score}")
        print(f"  code_quality_score: {review_result.code_quality_score}")
        print(f"  production_readiness_score: {review_result.production_readiness_score}")
        print(f"  recommendation: {review_result.recommendation}")
        print(f"  incomplete_categories: {review_result.incomplete_categories}")
        print(f"  reviewed_head_sha: {review_result.reviewed_head_sha}")
        print(f"  errors: {review_result.errors}")
        print(f"  blocking_findings ({len(review_result.blocking_findings)}):")
        for finding in review_result.blocking_findings:
            print(f"    - {finding.file}:{finding.line} - {finding.problem}")
        print(f"  warnings ({len(review_result.warnings)}):")
        for finding in review_result.warnings:
            print(f"    - {finding.file}:{finding.line} - {finding.problem}")
        print(f"  suggestions ({len(review_result.suggestions)}):")
        for finding in review_result.suggestions:
            print(f"    - {finding.file}:{finding.line} - {finding.problem}")

        print()
        print("publish decision:")
        block_reason = build_publish_block_reason(
            _field(result, "review_input"), _field(result, "head_sha"), review_result
        )
        if block_reason is not None:
            print(f"  BLOCKED: {block_reason}")
        else:
            print("  not blocked")
            print(f"  GitHub event: {_github_event_for(review_result.recommendation)}")

    print()
    print("publish result:")
    print(
        "  "
        + await _publish_if_requested(
            args.publish,
            "example-org/example-repo",
            pr_number,
            result,
            github_token,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
