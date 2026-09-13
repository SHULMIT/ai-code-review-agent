"""Tests for the health endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

from review_agent.app import create_app
from review_agent.config import Settings
from review_agent.routes import health as health_module
from review_agent.storage.database import DatabaseError


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "github_app_id": 1,
        "github_app_private_key_path": Path("secrets/test-private-key.pem"),
        "github_webhook_secret": "fake-webhook-secret",
        "database_url": "postgresql://fake:fake@localhost:5432/fake",
    }
    values.update(overrides)
    return Settings(**values)


class _FakeDatabase:
    """Stands in for Database, without opening a real PostgreSQL connection."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def ping(self) -> None:
        return None


class _FailingDatabase:
    """Stands in for Database when PostgreSQL is unreachable."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def ping(self) -> None:
        raise DatabaseError("Unable to connect to PostgreSQL.")


def test_health_returns_stable_service_response_when_the_database_is_reachable(
    monkeypatch,
) -> None:
    """The liveness endpoint reports ok once the database is confirmed reachable."""

    monkeypatch.setattr(health_module, "Settings", lambda: _settings())
    monkeypatch.setattr(health_module, "Database", _FakeDatabase)

    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "code-review-agent",
    }
    assert len(response.headers["X-Request-ID"]) == 32


def test_health_returns_503_when_the_database_is_unreachable(monkeypatch) -> None:
    """A database outage is reported as 503, matching the existing misconfiguration convention."""

    monkeypatch.setattr(health_module, "Settings", lambda: _settings())
    monkeypatch.setattr(health_module, "Database", _FailingDatabase)

    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 503
    assert "request_id" in response.json()


def test_health_returns_503_when_database_url_is_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(health_module, "Settings", lambda: _settings(database_url=None))

    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 503


def test_health_returns_503_when_settings_are_invalid(monkeypatch) -> None:
    def _raise_invalid_settings():
        # Missing required fields (github_app_id, etc.) - mirrors a genuinely
        # misconfigured deployment, not just a database outage.
        return Settings(_env_file=None)

    monkeypatch.setattr(health_module, "Settings", _raise_invalid_settings)

    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 503
