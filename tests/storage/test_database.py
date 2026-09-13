"""Tests for the PostgreSQL connection boundary."""

from pathlib import Path
from unittest.mock import AsyncMock

import psycopg
import pytest

from review_agent.storage import database as database_module
from review_agent.storage.database import Database, DatabaseError


class FakeConnection:
    """Record transaction cleanup performed by the database boundary."""

    def __init__(self) -> None:
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.close = AsyncMock()


async def test_connection_commits_and_closes_after_a_successful_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful operations use one transaction and always close its connection."""

    connection = FakeConnection()
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(database_module.psycopg.AsyncConnection, "connect", connect)
    database = Database("postgresql://example")

    async with database.connection() as opened_connection:
        assert opened_connection is connection

    connection.commit.assert_awaited_once()
    connection.rollback.assert_not_awaited()
    connection.close.assert_awaited_once()


async def test_connection_wraps_postgres_errors_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driver failures cannot leak through the storage boundary to a request."""

    connection = FakeConnection()
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(database_module.psycopg.AsyncConnection, "connect", connect)
    database = Database("postgresql://example")

    with pytest.raises(DatabaseError, match="PostgreSQL operation failed"):
        async with database.connection():
            raise psycopg.DatabaseError("driver detail")

    connection.rollback.assert_awaited_once()
    connection.close.assert_awaited_once()


async def test_apply_migration_wraps_an_unreadable_file() -> None:
    """Filesystem details from a missing migration are not exposed to callers."""

    database = Database("postgresql://example")

    with pytest.raises(DatabaseError, match="Unable to read PostgreSQL migration"):
        await database.apply_migration(Path("missing-migration.sql"))
