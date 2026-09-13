"""Safe, consistent HTTP error responses."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from review_agent.logging import get_request_id, log_event


class ErrorResponse(BaseModel):
    """The stable error shape returned by every HTTP endpoint."""

    error: str
    request_id: str


_LOGGER = logging.getLogger("review_agent")
_CLIENT_ERROR_MESSAGES = {
    400: "Bad request.",
    401: "Unauthorized.",
    403: "Forbidden.",
    404: "Not found.",
    405: "Method not allowed.",
    409: "Conflict.",
    422: "Request validation failed.",
}


def _request_id_for_response(request: Request) -> str:
    """Read the request ID from context or request state after middleware unwinds."""

    return get_request_id() or getattr(request.state, "request_id", "unavailable")


def _error_response(request: Request, status_code: int, error: str) -> JSONResponse:
    """Build a response that never exposes request data or exception details."""

    request_id = _request_id_for_response(request)
    response = ErrorResponse(error=error, request_id=request_id)
    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
        headers={"X-Request-ID": request_id},
    )


def _client_error_message(status_code: int) -> str:
    """Return a safe public message for an expected client error."""

    return _CLIENT_ERROR_MESSAGES.get(status_code, "Request failed.")


def register_error_handlers(app: FastAPI) -> None:
    """Register error handlers that expose only stable, non-sensitive details."""

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        request_id = _request_id_for_response(request)
        log_event(
            _LOGGER,
            logging.WARNING,
            "request_validation_failed",
            method=request.method,
            request_id=request_id,
            status_code=422,
        )
        return _error_response(request, 422, "Request validation failed.")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        request_id = _request_id_for_response(request)
        log_event(
            _LOGGER,
            logging.INFO,
            "request_rejected",
            method=request.method,
            request_id=request_id,
            status_code=error.status_code,
        )
        return _error_response(
            request,
            error.status_code,
            _client_error_message(error.status_code),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, error: Exception
    ) -> JSONResponse:
        request_id = _request_id_for_response(request)
        log_event(
            _LOGGER,
            logging.ERROR,
            "request_failed",
            method=request.method,
            request_id=request_id,
            status_code=500,
            error_type=type(error).__name__,
        )
        return _error_response(request, 500, "Internal server error.")
