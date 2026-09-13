"""Tests for the Ollama-compatible embedding client, all through mocked HTTP."""

import json
from pathlib import Path

import httpx
import pytest

from review_agent.config import Settings
from review_agent.tools.embedding_tool import (
    EmbeddingServiceError,
    get_embedding,
    get_embeddings,
)


def _settings(**overrides) -> Settings:
    """Build valid settings without touching the local .env file."""

    values = {
        "_env_file": None,
        "github_app_id": 123,
        "github_app_private_key_path": Path("secrets/test-private-key.pem"),
        "github_webhook_secret": "development-webhook-secret",
        "model_service_url": "http://192.0.2.10:11434",
        "model_service_timeout_seconds": 5,
        "embedding_service_timeout_seconds": 5,
    }
    values.update(overrides)
    return Settings(**values)


def _patch_async_client(
    monkeypatch, transport: httpx.MockTransport, client_options: dict | None = None
) -> None:
    """Make embedding_tool's HTTP calls use a local mock transport."""

    class MockedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            if client_options is not None:
                client_options["timeout"] = kwargs.get("timeout")
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockedAsyncClient)


async def test_get_embeddings_returns_one_vector_per_input_in_one_request(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler), captured)

    result = await get_embeddings(
        ["hardcoded API key", "missing tests"],
        _settings(model_service_timeout_seconds=1, embedding_service_timeout_seconds=7),
    )

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "http://192.0.2.10:11434/api/embed"
    assert captured["body"] == {
        "model": "embeddinggemma:latest",
        "input": ["hardcoded API key", "missing tests"],
    }
    assert captured["timeout"] == 7


async def test_get_embeddings_uses_a_provided_client_instead_of_opening_its_own(monkeypatch):
    """When a shared client is passed in, get_embeddings must reuse it, never construct a new one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as shared_client:

        def _fail_if_constructed(*args, **kwargs):
            raise AssertionError("a fresh AsyncClient must not be constructed when one is provided")

        monkeypatch.setattr(httpx, "AsyncClient", _fail_if_constructed)

        result = await get_embeddings(["text"], _settings(), client=shared_client)

    assert result == [[0.1, 0.2]]


async def test_get_embedding_preserves_the_single_text_interface(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    assert await get_embedding("hardcoded API key", _settings()) == [0.1, 0.2, 0.3]


async def test_get_embeddings_uses_the_independently_configured_embedding_url(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"embeddings": [[0.1]]})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    await get_embeddings(
        ["text"],
        _settings(
            model_service_url="http://192.0.2.10:11434",
            embedding_service_url="http://a-different-embedding-host:11434",
        ),
    )

    assert captured["url"] == "http://a-different-embedding-host:11434/api/embed"


async def test_get_embeddings_returns_without_an_http_call_for_empty_input(monkeypatch):
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    assert await get_embeddings([], _settings()) == []
    assert called is False


@pytest.mark.parametrize(
    "error",
    [
        httpx.TimeoutException("timed out"),
        httpx.ConnectError("connection refused"),
    ],
    ids=["timeout", "connection-error"],
)
async def test_get_embeddings_wraps_request_failures(monkeypatch, error):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(EmbeddingServiceError):
        await get_embeddings(["text"], _settings())


async def test_get_embeddings_wraps_a_non_200_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(EmbeddingServiceError, match="500"):
        await get_embeddings(["text"], _settings())


@pytest.mark.parametrize(
    "response_body",
    [
        {"unexpected": "shape"},
        {"embeddings": []},
        {"embeddings": [[]]},
        b'{"embeddings": [[NaN]]}',
        {"embeddings": [[0.1], [0.2]]},
    ],
    ids=["wrong-shape", "empty-list", "empty-vector", "non-finite", "count-mismatch"],
)
async def test_get_embeddings_wraps_malformed_response_vectors(monkeypatch, response_body):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(response_body, bytes):
            return httpx.Response(
                200,
                content=response_body,
                headers={"Content-Type": "application/json"},
            )
        return httpx.Response(200, json=response_body)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(EmbeddingServiceError, match="unparseable"):
        await get_embeddings(["text"], _settings())
