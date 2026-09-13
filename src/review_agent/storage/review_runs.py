"""Typed, atomic persistence for durable Code Review Agent attempts."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, PositiveInt

from review_agent.models.finding import Finding
from review_agent.services.event_router import ReviewTrigger
from review_agent.storage.database import Database


class ReviewRunStatus(str, Enum):
    """The minimal lifecycle states recorded for every review attempt."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReviewRunCreate(BaseModel):
    """The identity and trigger data inserted atomically for a pending review."""

    github_delivery_id: str = Field(min_length=1)
    review_identity: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    pr_number: PositiveInt
    installation_id: PositiveInt
    trigger: ReviewTrigger
    head_sha: str | None = Field(default=None, min_length=1)
    # Set only for a review launched from a Check Run's "Run Code Review"
    # action - lets the launcher update that same Check Run when the run
    # finishes, without every other trigger needing to know Check Runs exist.
    check_run_id: int | None = None


class ReviewRunCompletion(BaseModel):
    """The validated review result persisted after review analysis is complete.

    Each score is nullable because ReviewResult's score is: an incomplete
    review category (see finalize_review.py) has no score to persist, and
    the review_runs table's score columns already allow NULL for exactly
    this reason.
    """

    findings: list[Finding]
    architecture_score: int | None = Field(ge=0, le=100)
    code_quality_score: int | None = Field(ge=0, le=100)
    production_readiness_score: int | None = Field(ge=0, le=100)
    recommendation: str = Field(min_length=1)
    errors: list[str] = Field(default_factory=list)
    incomplete_categories: list[str] = Field(default_factory=list)
    last_review_id: int | None = None


class ReviewRun(BaseModel):
    """One durable review attempt, including its lifecycle state and result data."""

    id: UUID
    github_delivery_id: str
    review_identity: str
    repository: str
    pr_number: PositiveInt
    installation_id: PositiveInt
    trigger: ReviewTrigger
    head_sha: str | None = None
    check_run_id: int | None = None
    status: ReviewRunStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    findings: list[Finding] | None = None
    architecture_score: int | None = None
    code_quality_score: int | None = None
    production_readiness_score: int | None = None
    recommendation: str | None = None
    errors: list[str] = Field(default_factory=list)
    incomplete_categories: list[str] = Field(default_factory=list)
    last_review_id: int | None = None


class ReviewRunNotFoundError(Exception):
    """Raised when an expected review-run record no longer exists."""


class PreviousPublish(BaseModel):
    """The most recently published GitHub review's ID and recommendation for one PR.

    Looked up by review_launcher.py before publishing a new result, to decide
    whether a re-run's recommendation still matches what that review already
    shows. GitHub's review-update endpoint cannot change an already-submitted
    review's event (its approve/request-changes decision), only its body
    text - a changed recommendation must always become a new review instead
    of updating this one in place, so the caller needs both fields, not just
    the ID.
    """

    review_id: int
    recommendation: str


_REVIEW_RUN_COLUMNS = """
    id, github_delivery_id, review_identity, repository, pr_number,
    installation_id, trigger, head_sha, check_run_id, status, created_at,
    started_at, completed_at, findings, architecture_score, code_quality_score,
    production_readiness_score, recommendation, errors, incomplete_categories,
    last_review_id
"""


class ReviewRunRepository:
    """Store review runs through one PostgreSQL interface and atomic inserts."""

    def __init__(self, database: Database) -> None:
        """Bind the repository to the application's PostgreSQL connection boundary."""

        self._database = database

    async def create_pending(self, review_run: ReviewRunCreate) -> ReviewRun | None:
        """Insert one pending run or return None when PostgreSQL finds a duplicate."""

        query = f"""
            INSERT INTO review_runs (
                id, github_delivery_id, review_identity, repository, pr_number,
                installation_id, trigger, head_sha, check_run_id, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING {_REVIEW_RUN_COLUMNS}
        """
        parameters = (
            uuid4(),
            review_run.github_delivery_id,
            review_run.review_identity,
            review_run.repository,
            review_run.pr_number,
            review_run.installation_id,
            review_run.trigger.value,
            review_run.head_sha,
            review_run.check_run_id,
            ReviewRunStatus.PENDING.value,
        )

        row = await self._fetch_one(query, parameters)
        return self._parse_row(row) if row is not None else None

    async def get(self, review_run_id: UUID) -> ReviewRun | None:
        """Return a stored review run by ID without exposing driver row objects."""

        query = f"SELECT {_REVIEW_RUN_COLUMNS} FROM review_runs WHERE id = %s"
        row = await self._fetch_one(query, (review_run_id,))
        return self._parse_row(row) if row is not None else None

    async def mark_running(self, review_run_id: UUID) -> ReviewRun | None:
        """Move one pending run to running, preserving concurrent safety."""

        return await self._update_status(
            review_run_id,
            "status = %s, started_at = NOW()",
            (ReviewRunStatus.RUNNING.value,),
            expected_statuses=(ReviewRunStatus.PENDING,),
        )

    async def complete(
        self, review_run_id: UUID, completion: ReviewRunCompletion
    ) -> ReviewRun:
        """Persist one finished review result and mark its run completed."""

        findings = [finding.model_dump(mode="json") for finding in completion.findings]
        updated = await self._update_status(
            review_run_id,
            """
                status = %s, completed_at = NOW(), findings = %s,
                architecture_score = %s, code_quality_score = %s,
                production_readiness_score = %s, recommendation = %s,
                errors = %s, incomplete_categories = %s, last_review_id = %s
            """,
            (
                ReviewRunStatus.COMPLETED.value,
                Jsonb(findings),
                completion.architecture_score,
                completion.code_quality_score,
                completion.production_readiness_score,
                completion.recommendation,
                Jsonb(completion.errors),
                Jsonb(completion.incomplete_categories),
                completion.last_review_id,
            ),
            expected_statuses=(ReviewRunStatus.RUNNING,),
        )
        if updated is None:
            raise ReviewRunNotFoundError("Review run cannot be completed.")
        return updated

    async def fail(self, review_run_id: UUID, error_summary: str) -> ReviewRun:
        """Persist one safe failure summary without retaining raw exception details."""

        updated = await self._update_status(
            review_run_id,
            "status = %s, completed_at = NOW(), errors = %s",
            (ReviewRunStatus.FAILED.value, Jsonb([error_summary[:500]])),
            expected_statuses=(ReviewRunStatus.PENDING, ReviewRunStatus.RUNNING),
        )
        if updated is None:
            raise ReviewRunNotFoundError("Review run cannot be failed.")
        return updated

    async def skip(self, review_run_id: UUID, reason: str) -> ReviewRun:
        """Record a deliberately skipped review run with a concise safe reason."""

        updated = await self._update_status(
            review_run_id,
            "status = %s, completed_at = NOW(), errors = %s",
            (ReviewRunStatus.SKIPPED.value, Jsonb([reason[:500]])),
            expected_statuses=(ReviewRunStatus.PENDING,),
        )
        if updated is None:
            raise ReviewRunNotFoundError("Review run cannot be skipped.")
        return updated

    async def get_last_publish(
        self, repository: str, pr_number: int
    ) -> PreviousPublish | None:
        """Return the most recently published GitHub review's ID and recommendation for this PR.

        Ordered by created_at so a re-run always compares against the latest
        publish, not an older one. Only rows with a stored last_review_id can
        match - a run that failed before publishing, or that predates this
        column, correctly looks like "nothing published yet".
        """

        query = """
            SELECT last_review_id AS review_id, recommendation
            FROM review_runs
            WHERE repository = %s AND pr_number = %s AND last_review_id IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        """
        row = await self._fetch_one(query, (repository, pr_number))
        return PreviousPublish.model_validate(row) if row is not None else None

    async def _update_status(
        self,
        review_run_id: UUID,
        assignments: str,
        assignment_parameters: tuple[object, ...],
        expected_statuses: tuple[ReviewRunStatus, ...],
    ) -> ReviewRun | None:
        placeholders = ", ".join("%s" for _ in expected_statuses)
        query = f"""
            UPDATE review_runs
            SET {assignments}
            WHERE id = %s AND status IN ({placeholders})
            RETURNING {_REVIEW_RUN_COLUMNS}
        """
        parameters = assignment_parameters + (review_run_id,) + tuple(
            status.value for status in expected_statuses
        )
        row = await self._fetch_one(query, parameters)
        return self._parse_row(row) if row is not None else None

    async def _fetch_one(
        self, query: str, parameters: tuple[object, ...]
    ) -> dict[str, Any] | None:
        async with self._database.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(query, parameters)
                return await cursor.fetchone()

    def _parse_row(self, row: dict[str, Any]) -> ReviewRun:
        """Validate one PostgreSQL row immediately at the storage boundary."""

        return ReviewRun.model_validate(row)
