"""Tests for GitHub Webhook HMAC verification."""

import hashlib
import hmac

from review_agent.security.webhook_signature import verify_signature


_SECRET = "test-webhook-secret"
_BODY = b'{"action":"opened","number":42}'


def _signature(body: bytes, secret: str = _SECRET) -> str:
    """Create a GitHub-compatible SHA-256 signature for a test payload."""

    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_verify_signature_accepts_a_valid_signature() -> None:
    """An untouched body signed with the configured secret is accepted."""

    assert verify_signature(_BODY, _signature(_BODY), _SECRET) is True


def test_verify_signature_rejects_a_wrong_signature() -> None:
    """A signature generated with another secret is rejected."""

    assert verify_signature(_BODY, _signature(_BODY, "wrong-secret"), _SECRET) is False


def test_verify_signature_rejects_a_tampered_body() -> None:
    """A valid signature cannot authenticate modified bytes."""

    assert verify_signature(_BODY + b" ", _signature(_BODY), _SECRET) is False
