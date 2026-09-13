"""Tests for GitHub App JWT creation and installation-token caching."""

import asyncio
from datetime import UTC, datetime, timedelta
import shutil
import tempfile
from pathlib import Path

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from review_agent.tools import github_auth
from review_agent.tools.github_auth import (
    GitHubAuthError,
    InstallationAccessToken,
    InstallationTokenProvider,
    create_app_jwt,
)


@pytest.fixture
def private_key_path() -> Path:
    """Create an ephemeral RSA private key without storing a real credential."""

    directory = Path(tempfile.mkdtemp(prefix="review_agent_github_auth_test_"))
    path = directory / "github-app-private-key.pem"
    key = rsa.generate_private_key(public_exponent=65_537, key_size=2_048)
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    yield path
    shutil.rmtree(directory, ignore_errors=True)


def _patch_async_client(monkeypatch, transport: httpx.MockTransport) -> None:
    """Make provider HTTP calls use a local mock transport."""

    class MockedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockedAsyncClient)


def _token_response(token: str, expires_at: datetime) -> dict[str, str]:
    """Build the minimal valid GitHub installation-token response."""

    return {"token": token, "expires_at": expires_at.isoformat().replace("+00:00", "Z")}


def test_create_app_jwt_signs_expected_github_claims(private_key_path: Path) -> None:
    """The JWT has a verifiable RSA signature and GitHub's bounded claims."""

    now = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
    private_key = serialization.load_pem_private_key(
        private_key_path.read_bytes(), password=None
    )

    signed_jwt = create_app_jwt("123", private_key_path, now=now)
    claims = jwt.decode(
        signed_jwt,
        private_key.public_key(),
        algorithms=["RS256"],
        options={"verify_exp": False, "verify_iat": False},
    )

    assert claims["iss"] == "123"
    assert claims["iat"] == int((now - timedelta(seconds=60)).timestamp())
    assert claims["exp"] - claims["iat"] == 600


def test_create_app_jwt_wraps_private_key_failures() -> None:
    """An unreadable private key never leaks a raw filesystem exception."""

    with pytest.raises(GitHubAuthError, match="Unable to create"):
        create_app_jwt("123", Path("missing-private-key.pem"))


async def test_provider_fetches_an_installation_token(monkeypatch, private_key_path: Path) -> None:
    """A GitHub token response is parsed and returned without exposing its cache entry."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/app/installations/12/access_tokens"
        assert request.headers["authorization"] == "Bearer test-app-jwt"
        return httpx.Response(
            201,
            json=_token_response("installation-token", datetime(2030, 1, 1, 13, tzinfo=UTC)),
        )

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr(github_auth, "create_app_jwt", lambda *args: "test-app-jwt")
    provider = InstallationTokenProvider("123", private_key_path)

    assert await provider.get_token(12) == "installation-token"


async def test_provider_reuses_an_unexpired_token(monkeypatch, private_key_path: Path) -> None:
    """A second request for one unexpired installation avoids a GitHub API call."""

    requests = 0
    now = datetime(2030, 1, 1, 12, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(201, json=_token_response("cached-token", now + timedelta(hours=1)))

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr(github_auth, "create_app_jwt", lambda *args: "test-app-jwt")
    provider = InstallationTokenProvider("123", private_key_path, clock=lambda: now)

    assert await provider.get_token(12) == "cached-token"
    assert await provider.get_token(12) == "cached-token"
    assert requests == 1


async def test_provider_refreshes_an_expired_token(monkeypatch, private_key_path: Path) -> None:
    """An expired cache entry is replaced with a new GitHub installation token."""

    current_time = datetime(2030, 1, 1, 12, tzinfo=UTC)
    response_tokens = iter(["first-token", "refreshed-token"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json=_token_response(next(response_tokens), current_time + timedelta(seconds=1)),
        )

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr(github_auth, "create_app_jwt", lambda *args: "test-app-jwt")
    provider = InstallationTokenProvider(
        "123", private_key_path, clock=lambda: current_time
    )

    assert await provider.get_token(12) == "first-token"
    current_time += timedelta(seconds=2)
    assert await provider.get_token(12) == "refreshed-token"


async def test_provider_caches_installations_independently(
    monkeypatch, private_key_path: Path
) -> None:
    """Different installation IDs retain distinct cache entries and API requests."""

    requested_installations: list[str] = []
    now = datetime(2030, 1, 1, 12, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        installation_id = request.url.path.split("/")[3]
        requested_installations.append(installation_id)
        return httpx.Response(
            201,
            json=_token_response(f"token-{installation_id}", now + timedelta(hours=1)),
        )

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr(github_auth, "create_app_jwt", lambda *args: "test-app-jwt")
    provider = InstallationTokenProvider("123", private_key_path, clock=lambda: now)

    assert await provider.get_token(12) == "token-12"
    assert await provider.get_token(34) == "token-34"
    assert await provider.get_token(12) == "token-12"
    assert requested_installations == ["12", "34"]


async def test_provider_wraps_github_api_errors(monkeypatch, private_key_path: Path) -> None:
    """GitHub HTTP errors become safe authentication failures without token details."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr(github_auth, "create_app_jwt", lambda *args: "test-app-jwt")
    provider = InstallationTokenProvider("123", private_key_path)

    with pytest.raises(GitHubAuthError, match="401"):
        await provider.get_token(12)


async def test_provider_serializes_concurrent_fetches_for_the_same_installation(
    monkeypatch, private_key_path: Path
) -> None:
    """Two concurrent callers for one installation ID trigger a single GitHub fetch."""

    call_count = 0
    request_started = asyncio.Event()
    release_response = asyncio.Event()

    async def fake_request_installation_token(installation_id: int) -> InstallationAccessToken:
        nonlocal call_count
        call_count += 1
        request_started.set()
        await release_response.wait()
        return InstallationAccessToken(
            token="shared-token", expires_at=datetime(2030, 1, 1, 13, tzinfo=UTC)
        )

    provider = InstallationTokenProvider(
        "123", private_key_path, clock=lambda: datetime(2030, 1, 1, 12, tzinfo=UTC)
    )
    monkeypatch.setattr(provider, "_request_installation_token", fake_request_installation_token)

    async def release_once_the_first_fetch_is_in_flight() -> None:
        await request_started.wait()
        release_response.set()

    first_token, second_token, _ = await asyncio.gather(
        provider.get_token(12),
        provider.get_token(12),
        release_once_the_first_fetch_is_in_flight(),
    )

    assert first_token == second_token == "shared-token"
    assert call_count == 1


async def test_provider_does_not_block_other_installations_while_one_is_fetching(
    monkeypatch, private_key_path: Path
) -> None:
    """A slow fetch for one installation ID never blocks a concurrent, different ID."""

    release_installation_12 = asyncio.Event()

    async def fake_request_installation_token(installation_id: int) -> InstallationAccessToken:
        if installation_id == 12:
            await release_installation_12.wait()
        return InstallationAccessToken(
            token=f"token-{installation_id}", expires_at=datetime(2030, 1, 1, 13, tzinfo=UTC)
        )

    provider = InstallationTokenProvider(
        "123", private_key_path, clock=lambda: datetime(2030, 1, 1, 12, tzinfo=UTC)
    )
    monkeypatch.setattr(provider, "_request_installation_token", fake_request_installation_token)

    async def fetch_other_installation_then_unblock_the_first() -> str:
        await asyncio.sleep(0)  # let installation 12's fetch start and block first
        other_token = await provider.get_token(34)
        release_installation_12.set()
        return other_token

    installation_12_token, other_token = await asyncio.gather(
        provider.get_token(12), fetch_other_installation_then_unblock_the_first()
    )

    assert installation_12_token == "token-12"
    assert other_token == "token-34"
