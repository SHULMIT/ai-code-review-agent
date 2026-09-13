"""Structured logging that deliberately excludes untrusted request content."""

import json
import logging
from contextvars import ContextVar, Token
from typing import Final
from uuid import uuid4


_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
_HANDLER_NAME: Final = "review-agent-json"
# TEMPORARY (PR #9 model-response investigation) - the four fields below
# this comment are local-only reviewer diagnostics, never sent to GitHub.
# Remove them, and the log_event() calls in the six review_*.py nodes that
# use them, once the investigation is closed.
_SAFE_LOG_CONTEXT_FIELDS: Final = frozenset(
    {
        "check_run_id",
        "delivery_id",
        "diff_total_chars",  # TEMPORARY
        "error_type",
        "event",
        "head_sha",
        "included_changed_files",  # TEMPORARY
        "installation_id",
        "method",
        "parsed_findings_count",  # TEMPORARY
        "pr_number",
        "prompt_length",  # TEMPORARY
        "rag_client_diff_chars",  # TEMPORARY
        "rag_client_in_prompt",  # TEMPORARY
        "raw_response",  # TEMPORARY
        "reason_code",
        "request_id",
        "repository",
        "reviewer_category",  # TEMPORARY
        "stale_head_sha",
        "status_code",
        "triggered",
        "validation_result",  # TEMPORARY
    }
)


def create_request_id() -> str:
    """Create an opaque request identifier for correlation within this service."""

    return uuid4().hex


def get_request_id() -> str | None:
    """Return the request identifier active in the current execution context."""

    return _REQUEST_ID.get()


def set_request_id(request_id: str) -> Token[str | None]:
    """Set a request identifier and return the token needed to reset it."""

    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous request identifier after request processing ends."""

    _REQUEST_ID.reset(token)


class SafeJsonFormatter(logging.Formatter):
    """Render only allowlisted observability fields as a single JSON log record."""

    def format(self, record: logging.LogRecord) -> str:
        """Exclude free-form messages and unapproved fields from the log output."""

        payload: dict[str, object] = {
            "event": getattr(record, "event", "unstructured_log"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", get_request_id()),
        }
        for field_name in _SAFE_LOG_CONTEXT_FIELDS - {"event"}:
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value

        return json.dumps(payload, sort_keys=True)


def configure_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure the service logger with one safe JSON handler."""

    service_logger = logging.getLogger("review_agent")
    service_logger.setLevel(log_level)
    service_logger.propagate = False

    handler = next(
        (
            existing_handler
            for existing_handler in service_logger.handlers
            if existing_handler.name == _HANDLER_NAME
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.name = _HANDLER_NAME
        handler.setFormatter(SafeJsonFormatter())
        service_logger.addHandler(handler)

    handler.setLevel(log_level)
    return service_logger


def log_event(
    logger: logging.Logger, level: int, event: str, **context: object
) -> None:
    """Log an internal event while dropping all context not approved for logging."""

    safe_context = {
        field_name: value
        for field_name, value in context.items()
        if field_name in _SAFE_LOG_CONTEXT_FIELDS
    }
    logger.log(level, event, extra={"event": event, **safe_context})
