"""Tests for structured, safe service logging."""

import json
import logging
from unittest.mock import Mock

from review_agent.logging import SafeJsonFormatter, log_event


def test_log_event_drops_sensitive_and_untrusted_context() -> None:
    """Only allowlisted identifiers are passed to the Python logger."""

    logger = Mock(spec=logging.Logger)

    log_event(
        logger,
        logging.INFO,
        "webhook_received",
        delivery_id="delivery-123",
        github_token="example-secret-value",
        payload={"pull_request": {"title": "untrusted"}},
        diff="untrusted diff text",
    )

    extra = logger.log.call_args.kwargs["extra"]

    assert extra == {"event": "webhook_received", "delivery_id": "delivery-123"}
    assert "example-secret-value" not in str(extra)


def test_safe_json_formatter_omits_unstructured_message_content() -> None:
    """A direct logger call cannot write a representative secret to JSON output."""

    record = logging.makeLogRecord(
        {
            "name": "review_agent",
            "levelno": logging.ERROR,
            "levelname": "ERROR",
            "msg": "GITHUB_TOKEN=example-secret-value",
            "event": "request_failed",
            "status_code": 500,
            "payload": {"untrusted": "content"},
        }
    )

    rendered = json.loads(SafeJsonFormatter().format(record))

    assert rendered["event"] == "request_failed"
    assert rendered["status_code"] == 500
    assert "example-secret-value" not in json.dumps(rendered)
    assert "payload" not in rendered
