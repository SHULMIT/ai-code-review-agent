import httpx
import pytest

from review_agent.tools.github_tool import GitHubToolError, PRData, get_pr, get_pr_head_sha


def _patch_async_client(monkeypatch, transport: httpx.MockTransport) -> None:
    class _MockedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _MockedAsyncClient)


def _success_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/files"):
        return httpx.Response(
            200, json=[{"filename": "a.py"}, {"filename": "b.py"}]
        )
    if request.headers.get("accept") == "application/vnd.github.v3.diff":
        return httpx.Response(200, text="diff --git a/a.py b/a.py\n+print('hi')\n")
    return httpx.Response(
        200,
        json={
            "head": {"sha": "abc123"},
            "title": "Add feature",
            "body": "A description",
        },
    )


async def test_get_pr_success(monkeypatch):
    _patch_async_client(monkeypatch, httpx.MockTransport(_success_handler))

    result = await get_pr("example-org/example-repo", 42, "fake-token")

    assert isinstance(result, PRData)
    assert result.head_sha == "abc123"
    assert result.changed_files == ["a.py", "b.py"]
    assert result.title == "Add feature"
    assert result.description == "A description"
    assert "diff --git" in result.diff


async def test_get_pr_uses_a_provided_client_instead_of_opening_its_own(monkeypatch):
    """When a shared client is passed in, get_pr must reuse it, never construct a new one."""

    async with httpx.AsyncClient(transport=httpx.MockTransport(_success_handler)) as shared_client:

        def _fail_if_constructed(*args, **kwargs):
            raise AssertionError("a fresh AsyncClient must not be constructed when one is provided")

        monkeypatch.setattr(httpx, "AsyncClient", _fail_if_constructed)

        result = await get_pr("example-org/example-repo", 42, "fake-token", client=shared_client)

    assert result.head_sha == "abc123"


async def test_get_pr_head_sha_fetches_only_the_current_head(monkeypatch):
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(200, json={"head": {"sha": "def456"}})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    head_sha = await get_pr_head_sha("example-org/example-repo", 42, "fake-token")

    assert head_sha == "def456"
    assert requested_paths == ["/repos/example-org/example-repo/pulls/42"]


async def test_get_pr_head_sha_raises_on_an_unexpected_response_shape(monkeypatch):
    _patch_async_client(monkeypatch, httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    with pytest.raises(GitHubToolError):
        await get_pr_head_sha("example-org/example-repo", 42, "fake-token")


async def test_get_pr_head_sha_raises_when_the_sha_is_empty(monkeypatch):
    """"head" and "sha" are both present, but the value itself is empty - a
    different unexpected shape than a missing key entirely."""
    _patch_async_client(
        monkeypatch,
        httpx.MockTransport(lambda request: httpx.Response(200, json={"head": {"sha": ""}})),
    )

    with pytest.raises(GitHubToolError):
        await get_pr_head_sha("example-org/example-repo", 42, "fake-token")


async def test_get_pr_head_sha_raises_on_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubToolError):
        await get_pr_head_sha("example-org/example-repo", 999, "fake-token")


async def test_get_pr_head_sha_raises_on_network_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubToolError):
        await get_pr_head_sha("example-org/example-repo", 42, "fake-token")


async def test_get_pr_fetches_all_changed_file_pages(monkeypatch):
    requested_pages: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            requested_pages.append(request.url.params.get("page"))
            if request.url.params.get("page") == "1":
                return httpx.Response(
                    200, json=[{"filename": f"file-{index}.py"} for index in range(100)]
                )
            return httpx.Response(200, json=[{"filename": "file-100.py"}])
        if request.headers.get("accept") == "application/vnd.github.v3.diff":
            return httpx.Response(200, text="diff --git a/a.py b/a.py\n")
        return httpx.Response(
            200, json={"head": {"sha": "abc123"}, "title": "Add feature", "body": None}
        )

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    result = await get_pr("example-org/example-repo", 42, "fake-token")

    assert requested_pages == ["1", "2"]
    assert len(result.changed_files) == 101
    assert result.changed_files[-1] == "file-100.py"


async def test_get_pr_raises_on_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubToolError):
        await get_pr("example-org/example-repo", 999, "fake-token")


async def test_get_pr_raises_on_network_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubToolError):
        await get_pr("example-org/example-repo", 42, "fake-token")


async def test_get_pr_raises_on_unexpected_response_shape(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json=[])
        if request.headers.get("accept") == "application/vnd.github.v3.diff":
            return httpx.Response(200, text="")
        # Missing "head" entirely - triggers the KeyError path.
        return httpx.Response(200, json={"title": "Add feature", "body": None})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubToolError):
        await get_pr("example-org/example-repo", 42, "fake-token")


async def test_get_pr_raises_when_the_files_response_is_not_a_list(monkeypatch):
    """A malformed /files response (e.g. an error object) must not be silently iterated."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"message": "unexpected shape"})
        if request.headers.get("accept") == "application/vnd.github.v3.diff":
            return httpx.Response(200, text="diff --git a/a.py b/a.py\n")
        return httpx.Response(
            200, json={"head": {"sha": "abc123"}, "title": "Add feature", "body": None}
        )

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(GitHubToolError):
        await get_pr("example-org/example-repo", 42, "fake-token")
