"""Tests for the safe HTTP error handlers."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from review_agent.app import create_app


def _app_with_test_routes() -> FastAPI:
    """Build an app with private routes that exercise error handling."""

    app = create_app()

    @app.get("/_test/validation")
    async def validation_route(required_value: int) -> dict[str, int]:
        """Require a query value so FastAPI produces a validation error."""

        return {"required_value": required_value}

    @app.get("/_test/unexpected-error")
    async def unexpected_error_route() -> None:
        """Raise an exception containing a representative secret for the test."""

        raise RuntimeError("GITHUB_TOKEN=example-secret-value")

    return app


def test_validation_error_has_a_safe_stable_response() -> None:
    """Request validation details are not returned to the client."""

    client = TestClient(_app_with_test_routes())

    response = client.get("/_test/validation")

    assert response.status_code == 422
    assert response.json()["error"] == "Request validation failed."
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_not_found_error_has_a_safe_stable_response() -> None:
    """Framework-generated HTTP errors use the same public error contract."""

    client = TestClient(create_app())

    response = client.get("/_test/not-found")

    assert response.status_code == 404
    assert response.json()["error"] == "Not found."
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_unexpected_error_does_not_expose_the_exception_or_secret() -> None:
    """Internal failures have a generic public response and a request ID."""

    client = TestClient(_app_with_test_routes(), raise_server_exceptions=False)

    response = client.get("/_test/unexpected-error")

    assert response.status_code == 500
    assert response.json() == {
        "error": "Internal server error.",
        "request_id": response.headers["X-Request-ID"],
    }
    assert "example-secret-value" not in response.text
