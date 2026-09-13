"""GitHub App authentication and short-lived installation-token retrieval."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import jwt
from pydantic import BaseModel, SecretStr, ValidationError, field_validator


_GITHUB_API_BASE = "https://api.github.com"
_JWT_CLOCK_SKEW = timedelta(seconds=60)
_JWT_LIFETIME = timedelta(minutes=10)


class GitHubAuthError(Exception):
    """Raised when GitHub App authentication cannot safely produce a token."""


class InstallationAccessToken(BaseModel):
    """A GitHub installation token and the instant at which it expires."""

    token: SecretStr
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def require_timezone_aware_expiry(cls, value: datetime) -> datetime:
        """Reject ambiguous GitHub expiry timestamps before they enter the cache."""

        if value.tzinfo is None:
            raise ValueError("expires_at must include a timezone")
        return value.astimezone(UTC)


def _utc_now() -> datetime:
    """Return the current timezone-aware instant for JWT and token-cache checks."""

    return datetime.now(UTC)


def create_app_jwt(
    app_id: str | int, private_key_path: Path, now: datetime | None = None
) -> str:
    """Create a GitHub App JWT signed by the RSA key stored on local disk."""

    issued_at = (now or _utc_now()).astimezone(UTC) - _JWT_CLOCK_SKEW
    expires_at = issued_at + _JWT_LIFETIME
    claims = {
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
        "iss": str(app_id),
    }

    try:
        private_key = private_key_path.read_text(encoding="utf-8")
        return jwt.encode(claims, private_key, algorithm="RS256")
    except Exception as exc:
        raise GitHubAuthError("Unable to create a GitHub App JWT.") from exc


class InstallationTokenProvider:
    """Fetch and cache GitHub App tokens separately for each installation ID."""

    def __init__(
        self,
        app_id: str | int,
        private_key_path: Path,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Bind the GitHub App identity and clock used for this provider instance."""

        self._app_id = app_id
        self._private_key_path = private_key_path
        self._clock = clock
        self._token_cache: dict[int, InstallationAccessToken] = {}
        self._token_locks: dict[int, asyncio.Lock] = {}

    async def get_token(self, installation_id: int) -> str:
        """Return a valid installation token, refreshing it only after expiry.

        Concurrent calls for the same installation ID serialize on a
        per-installation lock so only the first caller fetches a new token
        from GitHub - the rest wait for it and then read the now-populated
        cache. Calls for different installation IDs never block each other.
        """

        if installation_id <= 0:
            raise GitHubAuthError("GitHub installation ID must be positive.")

        cached_token = self._unexpired_cached_token(installation_id)
        if cached_token is not None:
            return cached_token.token.get_secret_value()

        lock = self._token_locks.setdefault(installation_id, asyncio.Lock())
        async with lock:
            cached_token = self._unexpired_cached_token(installation_id)
            if cached_token is not None:
                return cached_token.token.get_secret_value()

            token = await self._request_installation_token(installation_id)
            self._token_cache[installation_id] = token
            return token.token.get_secret_value()

    def _unexpired_cached_token(self, installation_id: int) -> InstallationAccessToken | None:
        """Return the cached token for this installation, or None if absent or near expiry.

        Reuses _JWT_CLOCK_SKEW as a safety margin so a token about to expire
        is treated as expired rather than handed out and failing mid-flight.
        """

        cached_token = self._token_cache.get(installation_id)
        if cached_token is None or cached_token.expires_at <= self._clock() + _JWT_CLOCK_SKEW:
            return None
        return cached_token

    async def _request_installation_token(
        self, installation_id: int
    ) -> InstallationAccessToken:
        app_jwt = create_app_jwt(self._app_id, self._private_key_path)
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        url = f"{_GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise GitHubAuthError(
                "GitHub API returned "
                f"{exc.response.status_code} while creating an installation token."
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubAuthError("GitHub API could not create an installation token.") from exc

        try:
            return InstallationAccessToken.model_validate(response.json())
        except (ValidationError, ValueError, TypeError) as exc:
            raise GitHubAuthError(
                "GitHub API returned an invalid installation-token response."
            ) from exc
