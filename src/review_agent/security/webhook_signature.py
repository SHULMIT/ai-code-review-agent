"""GitHub Webhook HMAC signature verification."""

import hashlib
import hmac


_SIGNATURE_PREFIX = "sha256="
_SHA256_HEX_LENGTH = 64


def verify_signature(payload_body: bytes, signature_header: str, secret: str) -> bool:
    """Verify a GitHub SHA-256 HMAC header without parsing the untrusted body."""

    if not signature_header.startswith(_SIGNATURE_PREFIX):
        return False

    received_digest = signature_header.removeprefix(_SIGNATURE_PREFIX)
    if len(received_digest) != _SHA256_HEX_LENGTH:
        return False
    try:
        int(received_digest, 16)
    except ValueError:
        return False

    expected_digest = hmac.new(
        secret.encode("utf-8"), payload_body, hashlib.sha256
    ).hexdigest()
    expected_header = f"{_SIGNATURE_PREFIX}{expected_digest}"
    return hmac.compare_digest(expected_header, signature_header)
