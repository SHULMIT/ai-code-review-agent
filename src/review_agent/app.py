"""FastAPI application construction."""

from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from review_agent.api.errors import register_error_handlers
from review_agent.logging import (
    configure_logging,
    create_request_id,
    reset_request_id,
    set_request_id,
)
from review_agent.routes.health import router as health_router
from review_agent.routes.webhooks import router as webhooks_router


def create_app(
    review_launcher: object | None = None, check_run_launcher: object | None = None
) -> FastAPI:
    """Build and return the HTTP application without starting a process."""

    configure_logging()
    app = FastAPI(title="DataNet Code Review Agent", version="0.1.0")
    app.state.review_launcher = review_launcher
    app.state.check_run_launcher = check_run_launcher
    register_error_handlers(app)

    @app.middleware("http")
    async def add_request_id(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Attach a generated correlation ID without trusting request-provided IDs."""

        request_id = create_request_id()
        request.state.request_id = request_id
        request_token = set_request_id(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            reset_request_id(request_token)

    app.include_router(health_router)
    app.include_router(webhooks_router)
    return app
