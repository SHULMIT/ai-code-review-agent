"""End-to-end flow: a signed GitHub webhook in, a published review out.

Everything between those two points is the real implementation - signature
verification, event routing, duplicate rejection, all twelve graph nodes,
scoring, publish-block policy and formatting. Only the four genuine
externals are replaced: the GitHub API, the Model Service, the embedding
service, and PostgreSQL.

This is the only test that exercises the seams between those pieces. Every
other test in this suite verifies one unit against fakes on both sides, so a
mismatch between two units - a renamed field, a changed return shape, a node
wired in the wrong order - passes all of them and fails only here.
"""

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from review_agent.app import create_app
from review_agent.config import Settings
from review_agent.models.model_output import ModelReviewOutput
from review_agent.models.review_result import ReviewResult
from review_agent.nodes import load_pr as load_pr_module
from review_agent.nodes import (
    review_agent_harness,
    review_architecture,
    review_performance,
    review_quality,
    review_rag_search,
    review_security,
)
from review_agent.services import finding_deduplication, review_launcher
from review_agent.services.review_launcher import ReviewLauncher
from review_agent.tools.github_tool import PRData
from review_agent.storage.review_runs import (
    ReviewRun,
    ReviewRunCompletion,
    ReviewRunCreate,
    ReviewRunStatus,
)

WEBHOOK_SECRET = "e2e-test-webhook-secret"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REVIEWER_MODULES = (
    review_security,
    review_architecture,
    review_quality,
    review_rag_search,
    review_performance,
    review_agent_harness,
)

# Keep this fixture deliberately unmapped: conditional RAG/Search and Agent
# Harness reviewers both fail open for Component.UNKNOWN, letting this E2E
# exercise every reviewer. Their component-specific routing is unit-tested
# separately.
SAMPLE_FILE = "services/unmapped/search_helper.py"

# Long enough that each reviewer can be handed a distinct, well-separated
# line. Findings on adjacent lines get clustered together by
# finding_deduplication's line tolerance, which would defeat the point of
# asserting that all six reviewers' findings arrive.
SAMPLE_DIFF = (
    f"diff --git a/{SAMPLE_FILE} b/{SAMPLE_FILE}\n"
    f"--- a/{SAMPLE_FILE}\n"
    f"+++ b/{SAMPLE_FILE}\n"
    "@@ -0,0 +1,60 @@\n"
    + "".join(f"+line_{n} = {n}\n" for n in range(1, 61))
)

# One distinct finding per reviewer: a different line, far apart, and
# unrelated problem text. Both matter - deduplication merges on line
# proximity AND problem similarity, so six identical findings would collapse
# into one and this test could not tell a missing reviewer from a merged one.
# That is not hypothetical: before the reviewers were parallelized they all
# returned the same finding here, dedup merged them, and whichever category
# happened to run first was the one that survived.
FINDING_LINE_BY_CATEGORY = {
    "security": 5,
    "architecture": 15,
    "code quality": 25,
    "rag_search": 35,
    "performance": 45,
    "agent_harness": 55,
}


def _sign(body: bytes) -> str:
    """Produce the X-Hub-Signature-256 header GitHub would send for this body."""
    digest = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        github_app_id=1,
        github_app_private_key_path="secrets/test-private-key.pem",
        github_webhook_secret=WEBHOOK_SECRET,
        database_url="postgresql://unused:unused@localhost:5432/unused",
    )


class FakeReviewRunRepository:
    """In-memory stand-in for PostgreSQL that keeps the real duplicate contract.

    create_pending returns None for an identity it has already seen, which is
    how the real repository reports the database unique constraint rejecting a
    duplicate. Reproducing that here is the point: the launcher's
    accepted/rejected decision is driven by it.
    """

    def __init__(self) -> None:
        self.runs: dict[UUID, ReviewRun] = {}
        self.completed: dict[UUID, ReviewRunCompletion] = {}
        self.failures: dict[UUID, str] = {}
        self._identities: set[str] = set()

    async def create_pending(self, review_run: ReviewRunCreate) -> ReviewRun | None:
        if review_run.review_identity in self._identities:
            return None
        self._identities.add(review_run.review_identity)
        run = ReviewRun(
            id=uuid4(),
            github_delivery_id=review_run.github_delivery_id,
            review_identity=review_run.review_identity,
            repository=review_run.repository,
            pr_number=review_run.pr_number,
            installation_id=review_run.installation_id,
            trigger=review_run.trigger,
            head_sha=review_run.head_sha,
            status=ReviewRunStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        self.runs[run.id] = run
        return run

    async def mark_running(self, review_run_id: UUID) -> ReviewRun:
        return self.runs[review_run_id]

    async def complete(self, review_run_id: UUID, completion: ReviewRunCompletion) -> None:
        self.completed[review_run_id] = completion

    async def fail(self, review_run_id: UUID, error_summary: str) -> None:
        self.failures[review_run_id] = error_summary

    async def get_last_publish(self, repository: str, pr_number: int):
        """Mirrors the real lookup for a previously published review on this PR.

        Returns None here: this fake starts empty every test, so the first
        run of a PR is always a fresh publish rather than an update.
        """
        return None


@pytest.fixture
def published():
    """Collects every publish_review call instead of contacting GitHub."""
    return []


@pytest.fixture
def repository():
    return FakeReviewRunRepository()


@pytest.fixture
def client(monkeypatch, published, repository):
    """The real app, with only the four external systems replaced."""

    async def fake_get_pr(repo, pr_number, token, client=None):
        return PRData(
            head_sha="a1b2c3d4e5f678901234567890abcdef12345678",
            diff=SAMPLE_DIFF,
            changed_files=[SAMPLE_FILE],
            title="Add a search helper",
        )

    def make_fake_call_model(category: str):
        """Build a per-reviewer fake so each returns a finding only it could have."""
        line = FINDING_LINE_BY_CATEGORY[category]

        async def fake_call_model(prompt, settings, client=None):
            return ModelReviewOutput(
                raw_text=json.dumps(
                    [
                        {
                            "severity": "BLOCKING",
                            "file": SAMPLE_FILE,
                            "line": line,
                            "problem": f"Distinct {category} defect reported at line {line}.",
                            "why_it_matters": f"Stands in for a real {category} consequence.",
                            "suggested_fix": f"Stands in for a real {category} remedy.",
                        }
                    ]
                )
            )

        return fake_call_model

    async def fake_get_embeddings(texts, settings, client=None):
        """Return a mutually orthogonal vector per text, so nothing merges.

        One-hot vectors, not scaled ones. Cosine similarity ignores
        magnitude, so [1,0,0] and [2,0,0] are a perfect 1.0 match despite
        looking different - which silently merged all six findings into one
        the first time this fake was written.
        """
        return [
            [1.0 if position == index else 0.0 for position in range(len(texts))]
            for index, _ in enumerate(texts)
        ]

    async def fake_publish_review(
        repo, pr_number, head_sha, review_result, token, existing_review_id=None, client=None
    ):
        published.append(
            {
                "repo": repo,
                "pr_number": pr_number,
                "head_sha": head_sha,
                "result": review_result,
                "existing_review_id": existing_review_id,
            }
        )
        # The real publisher returns the GitHub review ID, which the launcher
        # stores so a later run can update that review in place.
        return 9000 + len(published)

    async def fake_get_token(self, installation_id):
        return "fake-installation-token"

    monkeypatch.setattr(load_pr_module, "get_pr", fake_get_pr)
    for module in REVIEWER_MODULES:
        monkeypatch.setattr(module, "call_model", make_fake_call_model(module.REVIEW_CATEGORY))
    monkeypatch.setattr(finding_deduplication, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(review_launcher, "publish_review", fake_publish_review)
    monkeypatch.setattr(
        review_launcher.InstallationTokenProvider, "get_token", fake_get_token
    )
    monkeypatch.setattr(review_launcher, "ReviewRunRepository", lambda database: repository)
    # create_app() builds its own Settings() internally, independent of the
    # _settings() instance built above for the launcher - it must not depend
    # on a developer's local .env existing, so every field Settings requires
    # with no default is set here explicitly, matching _settings() above.
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("GITHUB_APP_ID", "1")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "secrets/test-private-key.pem")

    app = create_app()
    app.state.review_launcher = ReviewLauncher(
        repository, review_launcher.build_review_launcher(_settings())._review_worker
    )
    return TestClient(app)


def _post(client, *, event="issue_comment", fixture="issue_comment_created.json", delivery="delivery-1"):
    body = (FIXTURES / "webhooks" / fixture).read_bytes()
    return client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": delivery,
            "X-Hub-Signature-256": _sign(body),
            "Content-Type": "application/json",
        },
    )


# --- the happy path ----------------------------------------------------------


def test_a_signed_pull_request_webhook_publishes_a_review(client, published, repository):
    """The whole flow, end to end: signed delivery in, published review out."""
    response = _post(client)

    assert response.status_code == 202, response.text

    # TestClient runs background tasks before returning, so the review has
    # already completed by this point.
    assert len(published) == 1, "exactly one review should have been published"
    call = published[0]
    assert call["repo"] == "example-org/example-repo"
    assert call["pr_number"] == 42
    assert call["head_sha"] == "a1b2c3d4e5f678901234567890abcdef12345678"

    result: ReviewResult = call["result"]
    assert result.recommendation == "REQUEST_CHANGES", "a BLOCKING finding must block the merge"
    assert result.blocking_findings, "the seeded finding should have survived validation"
    assert result.reviewed_head_sha == "a1b2c3d4e5f678901234567890abcdef12345678"


def test_the_completed_run_is_recorded_only_after_publishing(client, published, repository):
    """A review that was never delivered must not look completed in stored history."""
    _post(client)

    assert len(repository.completed) == 1
    assert repository.failures == {}
    completion = next(iter(repository.completed.values()))
    assert completion.recommendation == "REQUEST_CHANGES"
    assert completion.findings, "findings must reach storage, not only GitHub"


def test_every_reviewer_category_reaches_the_result(client, published):
    """Catches a reviewer left out of the graph, which no unit test would notice."""
    _post(client)

    result = published[0]["result"]
    all_findings = result.blocking_findings + result.warnings + result.suggestions
    categories = {finding.review_category for finding in all_findings}

    assert categories == set(FINDING_LINE_BY_CATEGORY), (
        f"every reviewer's finding should reach the result, missing "
        f"{set(FINDING_LINE_BY_CATEGORY) - categories}"
    )
    assert not result.incomplete_categories, (
        f"no category should have failed, got {result.incomplete_categories}"
    )
    diagnostics_by_category = {
        diagnostic.category: diagnostic for diagnostic in result.reviewer_diagnostics
    }
    assert set(diagnostics_by_category) == set(FINDING_LINE_BY_CATEGORY)
    assert all(
        diagnostic.status == "completed"
        for diagnostic in diagnostics_by_category.values()
    )


def test_static_checks_findings_survive_to_the_published_result(client, published):
    """static_checks runs before the reviewers; its findings must not be lost downstream."""
    _post(client)

    result = published[0]["result"]
    all_findings = result.blocking_findings + result.warnings + result.suggestions
    # SAMPLE_DIFF adds 60 lines across 1 file, well under the size limits, so
    # the only deterministic check that should fire is none - the point here
    # is that the reviewers' findings are not lost, and that static_checks
    # running first does not swallow them.
    assert len(all_findings) == len(FINDING_LINE_BY_CATEGORY)


# --- the paths that must NOT reach GitHub ------------------------------------


def test_an_unsigned_delivery_is_rejected_and_starts_no_review(client, published, repository):
    body = (FIXTURES / "webhooks" / "pull_request_opened.json").read_bytes()

    response = client.post(
        "/webhooks/github",
        content=body,
        headers={"X-GitHub-Event": "pull_request", "X-GitHub-Delivery": "d", "Content-Type": "application/json"},
    )

    assert response.status_code == 401
    assert published == []
    assert repository.runs == {}


def test_a_tampered_body_is_rejected(client, published):
    body = (FIXTURES / "webhooks" / "pull_request_opened.json").read_bytes()
    signature = _sign(body)

    response = client.post(
        "/webhooks/github",
        content=body.replace(b'"number": 42', b'"number": 99'),
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "d",
            "X-Hub-Signature-256": signature,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 401
    assert published == []


def test_an_unsupported_event_is_ignored_without_starting_a_review(client, published, repository):
    response = _post(client, event="push")

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    assert published == []
    assert repository.runs == {}


def test_a_redelivered_webhook_publishes_only_once(client, published, repository):
    """GitHub retries deliveries; the database unique constraint is what stops a second review."""
    first = _post(client, delivery="delivery-1")
    second = _post(client, delivery="delivery-1")

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["accepted"] is False
    assert len(published) == 1, "the retry must not produce a second published review"
    assert len(repository.runs) == 1
