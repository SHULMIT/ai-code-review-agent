"""Unit and opt-in integration tests for PostgreSQL review-run storage."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import BackgroundTasks
from psycopg.types.json import Jsonb

from review_agent.models.finding import Component, Finding, Severity
from review_agent.services.event_router import ReviewRequest, ReviewTrigger
from review_agent.services.review_launcher import ReviewLauncher
from review_agent.storage.database import Database
from review_agent.storage.review_runs import (
    PreviousPublish,
    ReviewRunCompletion,
    ReviewRunCreate,
    ReviewRunNotFoundError,
    ReviewRunRepository,
    ReviewRunStatus,
)


_MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


def _run_row(
    *,
    review_run_id: UUID | None = None,
    status: ReviewRunStatus = ReviewRunStatus.PENDING,
) -> dict[str, Any]:
    """Build a PostgreSQL-shaped row that crosses the repository boundary."""

    return {
        "id": review_run_id or uuid4(),
        "github_delivery_id": "delivery-1",
        "review_identity": "example-org/example-repo:42:abc:pull_request.opened",
        "repository": "example-org/example-repo",
        "pr_number": 42,
        "installation_id": 123,
        "trigger": ReviewTrigger.PULL_REQUEST_OPENED.value,
        "head_sha": "abc",
        "check_run_id": None,
        "status": status.value,
        "created_at": datetime(2030, 1, 1, tzinfo=UTC),
        "started_at": None,
        "completed_at": None,
        "findings": None,
        "architecture_score": None,
        "code_quality_score": None,
        "production_readiness_score": None,
        "recommendation": None,
        "errors": [],
        "incomplete_categories": [],
        "last_review_id": None,
    }


class FakeCursor:
    """Capture repository SQL while returning preselected PostgreSQL rows."""

    def __init__(self, database: "FakeDatabase") -> None:
        self._database = database

    async def __aenter__(self) -> "FakeCursor":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, query: str, parameters: tuple[object, ...]) -> None:
        self._database.queries.append((query, parameters))

    async def fetchone(self) -> dict[str, Any] | None:
        return self._database.rows.pop(0)


class FakeConnection:
    """Expose the cursor shape consumed by ReviewRunRepository."""

    def __init__(self, database: "FakeDatabase") -> None:
        self._database = database

    def cursor(self, **_: object) -> FakeCursor:
        return FakeCursor(self._database)


class FakeDatabase:
    """Provide deterministic rows without replacing repository SQL behavior."""

    def __init__(self, rows: list[dict[str, Any] | None]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, tuple[object, ...]]] = []

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[FakeConnection]:
        yield FakeConnection(self)


def _review_run_create() -> ReviewRunCreate:
    """Return one valid atomic review-run insert request."""

    return ReviewRunCreate(
        github_delivery_id="delivery-1",
        review_identity="example-org/example-repo:42:abc:pull_request.opened",
        repository="example-org/example-repo",
        pr_number=42,
        installation_id=123,
        trigger=ReviewTrigger.PULL_REQUEST_OPENED,
        head_sha="abc",
    )


async def test_create_pending_uses_database_atomic_conflict_handling() -> None:
    """The repository inserts once and lets PostgreSQL decide duplicate conflicts."""

    database = FakeDatabase([_run_row()])
    repository = ReviewRunRepository(database)  # type: ignore[arg-type]

    review_run = await repository.create_pending(_review_run_create())

    assert review_run is not None
    assert review_run.status == ReviewRunStatus.PENDING
    assert "ON CONFLICT DO NOTHING" in database.queries[0][0]
    assert "SELECT" not in database.queries[0][0].split("INSERT", 1)[0]


async def test_create_pending_stores_the_check_run_id_when_given() -> None:
    """A Check Run-triggered review's check_run_id is persisted with the rest of its identity."""

    row = _run_row()
    row["check_run_id"] = 987654
    database = FakeDatabase([row])
    repository = ReviewRunRepository(database)  # type: ignore[arg-type]
    create_request = _review_run_create()
    create_request = create_request.model_copy(update={"check_run_id": 987654})

    review_run = await repository.create_pending(create_request)

    assert review_run is not None
    assert review_run.check_run_id == 987654
    assert database.queries[0][1][8] == 987654


async def test_create_pending_returns_none_for_a_database_duplicate() -> None:
    """A unique-key conflict avoids scheduling a second review run."""

    database = FakeDatabase([None])
    repository = ReviewRunRepository(database)  # type: ignore[arg-type]

    assert await repository.create_pending(_review_run_create()) is None


async def test_repository_reads_and_transitions_review_runs() -> None:
    """Read, running, complete, fail, and skip operations return typed records."""

    review_run_id = uuid4()
    completed_row = _run_row(review_run_id=review_run_id, status=ReviewRunStatus.COMPLETED)
    completed_row.update(
        {
            "findings": [
                {
                    "severity": "WARNING",
                    "component": "Frontend",
                    "review_category": "code quality",
                    "file": "app.py",
                    "line": 1,
                    "problem": "p",
                    "why_it_matters": "w",
                    "suggested_fix": "f",
                }
            ],
            "architecture_score": 80,
            "code_quality_score": 81,
            "production_readiness_score": 82,
            "recommendation": "APPROVE",
            "errors": ["repository search truncated: scanned 500/1200 files due to configured limit"],
            "incomplete_categories": ["rag_search"],
            "last_review_id": 555,
        }
    )
    database = FakeDatabase(
        [
            _run_row(review_run_id=review_run_id),
            _run_row(review_run_id=review_run_id, status=ReviewRunStatus.RUNNING),
            completed_row,
            _run_row(review_run_id=review_run_id, status=ReviewRunStatus.FAILED),
            _run_row(review_run_id=review_run_id, status=ReviewRunStatus.SKIPPED),
        ]
    )
    repository = ReviewRunRepository(database)  # type: ignore[arg-type]
    completion = ReviewRunCompletion(
        findings=[
            Finding(
                severity=Severity.WARNING,
                component=Component.FRONTEND,
                review_category="code quality",
                file="app.py",
                line=1,
                problem="p",
                why_it_matters="w",
                suggested_fix="f",
            )
        ],
        architecture_score=80,
        code_quality_score=81,
        production_readiness_score=82,
        recommendation="APPROVE",
        errors=["repository search truncated: scanned 500/1200 files due to configured limit"],
        incomplete_categories=["rag_search"],
        last_review_id=555,
    )

    assert (await repository.get(review_run_id)) is not None
    assert (await repository.mark_running(review_run_id)).status == ReviewRunStatus.RUNNING
    completed = await repository.complete(review_run_id, completion)
    assert completed.findings is not None
    assert completed.errors == completion.errors
    assert completed.incomplete_categories == completion.incomplete_categories
    assert completed.last_review_id == 555
    assert (await repository.fail(review_run_id, "safe error")).status == ReviewRunStatus.FAILED
    assert (await repository.skip(review_run_id, "safe skip")).status == ReviewRunStatus.SKIPPED
    assert isinstance(database.queries[2][1][1], Jsonb)
    assert database.queries[2][1][6].obj == completion.errors
    assert database.queries[2][1][7].obj == completion.incomplete_categories
    assert database.queries[2][1][8] == 555


async def test_review_run_completion_accepts_a_null_score_for_an_incomplete_category() -> None:
    """A category finalize_review.py could not score must not crash storage on completion."""

    review_run_id = uuid4()
    completed_row = _run_row(review_run_id=review_run_id, status=ReviewRunStatus.COMPLETED)
    completed_row.update(
        {
            "findings": [],
            "architecture_score": None,
            "code_quality_score": 90,
            "production_readiness_score": None,
            "recommendation": "INCOMPLETE",
        }
    )
    database = FakeDatabase([completed_row])
    repository = ReviewRunRepository(database)  # type: ignore[arg-type]
    completion = ReviewRunCompletion(
        findings=[],
        architecture_score=None,
        code_quality_score=90,
        production_readiness_score=None,
        recommendation="INCOMPLETE",
    )

    completed = await repository.complete(review_run_id, completion)

    assert completed.architecture_score is None
    assert completed.code_quality_score == 90
    assert completed.recommendation == "INCOMPLETE"


async def test_get_last_publish_returns_the_most_recent_stored_review() -> None:
    """The lookup review_launcher.py uses before publishing to decide reuse vs. a new review."""

    database = FakeDatabase([{"review_id": 555, "recommendation": "APPROVE"}])
    repository = ReviewRunRepository(database)  # type: ignore[arg-type]

    result = await repository.get_last_publish("example-org/example-repo", 42)

    assert result == PreviousPublish(review_id=555, recommendation="APPROVE")
    query, parameters = database.queries[0]
    assert "last_review_id IS NOT NULL" in query
    assert parameters == ("example-org/example-repo", 42)


async def test_get_last_publish_returns_none_when_nothing_published_yet() -> None:
    """A PR with no completed, published run yet looks like a first-ever publish."""

    database = FakeDatabase([None])
    repository = ReviewRunRepository(database)  # type: ignore[arg-type]

    assert await repository.get_last_publish("example-org/example-repo", 42) is None


@pytest.mark.parametrize("operation", ["complete", "fail", "skip"])
async def test_terminal_updates_reject_missing_or_invalid_runs(operation: str) -> None:
    """Lifecycle writes cannot silently claim success for a missing review run."""

    repository = ReviewRunRepository(FakeDatabase([None]))  # type: ignore[arg-type]
    review_run_id = uuid4()
    completion = ReviewRunCompletion(
        findings=[],
        architecture_score=80,
        code_quality_score=80,
        production_readiness_score=80,
        recommendation="APPROVE",
    )

    with pytest.raises(ReviewRunNotFoundError):
        if operation == "complete":
            await repository.complete(review_run_id, completion)
        elif operation == "fail":
            await repository.fail(review_run_id, "safe error")
        else:
            await repository.skip(review_run_id, "safe skip")


async def test_review_runs_postgres_integration() -> None:
    """Exercise schema, serialization, atomic duplicates, and launch concurrency."""

    database_url = os.getenv("TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("set TEST_DATABASE_URL to run PostgreSQL integration tests")

    database = Database(database_url)
    async with database.connection() as connection:
        await connection.execute("DROP TABLE IF EXISTS review_runs")
    await database.apply_migration(_MIGRATIONS_DIR / "001_create_review_runs.sql")
    await database.apply_migration(_MIGRATIONS_DIR / "002_add_incomplete_categories.sql")
    await database.apply_migration(_MIGRATIONS_DIR / "003_add_last_review_id.sql")
    await database.apply_migration(_MIGRATIONS_DIR / "004_add_check_run_trigger.sql")
    await database.apply_migration(_MIGRATIONS_DIR / "005_allow_retry_after_terminal_review_run.sql")
    repository = ReviewRunRepository(database)
    candidate = _review_run_create()

    try:
        inserted, duplicate = await asyncio.gather(
            repository.create_pending(candidate),
            repository.create_pending(candidate),
        )
        review_run = inserted or duplicate

        assert review_run is not None
        assert sum(result is not None for result in (inserted, duplicate)) == 1
        assert (await repository.get(review_run.id)) is not None

        assert (await repository.mark_running(review_run.id)) is not None
        completed = await repository.complete(
            review_run.id,
            ReviewRunCompletion(
                findings=[
                    Finding(
                        severity=Severity.WARNING,
                        component=Component.FRONTEND,
                        review_category="code quality",
                        file="app.py",
                        line=1,
                        problem="p",
                        why_it_matters="w",
                        suggested_fix="f",
                    )
                ],
                architecture_score=80,
                code_quality_score=81,
                production_readiness_score=82,
                recommendation="APPROVE",
            ),
        )
        assert completed.findings is not None
        assert completed.findings[0].component == Component.FRONTEND

        worker_invocations: list[UUID] = []

        async def review_worker(review_run):
            worker_invocations.append(review_run.id)

        launcher = ReviewLauncher(repository, review_worker)
        review_request = ReviewRequest(
            repository="example-org/example-repo",
            pr_number=42,
            head_sha="def",
            installation_id=123,
            trigger=ReviewTrigger.PULL_REQUEST_OPENED,
        )
        first_tasks = BackgroundTasks()
        second_tasks = BackgroundTasks()
        first_launch, second_launch = await asyncio.gather(
            launcher.launch(review_request, "delivery-2", first_tasks),
            launcher.launch(review_request, "delivery-2", second_tasks),
        )

        assert sum(result.accepted for result in (first_launch, second_launch)) == 1
        assert sum(len(tasks.tasks) for tasks in (first_tasks, second_tasks)) == 1
        accepted_tasks = first_tasks if first_launch.accepted else second_tasks
        await accepted_tasks()
        assert len(worker_invocations) == 1
    finally:
        async with database.connection() as connection:
            await connection.execute("DROP TABLE IF EXISTS review_runs")
