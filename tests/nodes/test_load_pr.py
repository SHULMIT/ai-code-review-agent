import httpx
import pytest

from review_agent.graph.state import ReviewState
from review_agent.nodes.load_pr import make_load_pr_node


def _patch_async_client(monkeypatch, transport: httpx.MockTransport) -> None:
    class _MockedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _MockedAsyncClient)


async def test_load_pr_node_success(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json=[{"filename": "a.py"}])
        if request.headers.get("accept") == "application/vnd.github.v3.diff":
            return httpx.Response(200, text="diff --git a/a.py b/a.py\n")
        return httpx.Response(
            200, json={"head": {"sha": "abc123"}, "title": "t", "body": None}
        )

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))
    state = ReviewState(repo="example-org/example-repo", pr_number=42)
    load_pr_node = make_load_pr_node(token="fake-token")

    result = await load_pr_node(state)

    assert result == {
        "head_sha": "abc123",
        "diff": "diff --git a/a.py b/a.py\n",
        "changed_files": ["a.py"],
    }


async def test_load_pr_node_returns_error_on_http_failure(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))
    state = ReviewState(repo="example-org/example-repo", pr_number=999)
    load_pr_node = make_load_pr_node(token="fake-token")

    result = await load_pr_node(state)

    assert "errors" in result
    assert len(result["errors"]) == 1
    assert "404" in result["errors"][0]


async def test_load_pr_node_returns_error_on_network_failure(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))
    state = ReviewState(repo="example-org/example-repo", pr_number=42)
    load_pr_node = make_load_pr_node(token="fake-token")

    result = await load_pr_node(state)

    assert "errors" in result
    assert len(result["errors"]) == 1
