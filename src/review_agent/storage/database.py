"""Small asynchronous PostgreSQL connection boundary for review-run storage."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from psycopg import AsyncConnection


class DatabaseError(Exception):
    """Raised when a PostgreSQL operation cannot be completed safely."""


class Database:
    """Open short-lived PostgreSQL connections without exposing driver failures."""

    def __init__(self, database_url: str) -> None:
        """Store the PostgreSQL URL used for each storage operation."""

        self._database_url = database_url

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection]:
        """Yield one transactional connection and wrap PostgreSQL driver errors."""

        try:
            connection = await psycopg.AsyncConnection.connect(self._database_url)
        except psycopg.Error as exc:
            raise DatabaseError("Unable to connect to PostgreSQL.") from exc

        try:
            yield connection
            await connection.commit()
        except psycopg.Error as exc:
            await connection.rollback()
            raise DatabaseError("PostgreSQL operation failed.") from exc
        finally:
            await connection.close()

    async def ping(self) -> None:
        """Run a trivial query to confirm PostgreSQL is actually reachable.

        Used by the health endpoint - a single "SELECT 1" is enough to prove
        a real connection can be opened and a query executed, without the
        cost of a heavier query against real tables.
        """

        async with self.connection() as connection:
            await connection.execute("SELECT 1")

    async def apply_migration(self, migration_path: Path) -> None:
        """Apply one SQL migration file through this database boundary."""

        try:
            migration_sql = migration_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatabaseError("Unable to read PostgreSQL migration.") from exc

        async with self.connection() as connection:
            await connection.execute(migration_sql)
