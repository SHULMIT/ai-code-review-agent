"""Health endpoint for the Code Review Agent."""

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from review_agent.config import Settings
from review_agent.schemas.health import HealthResponse
from review_agent.storage.database import Database, DatabaseError


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service liveness only once the database is confirmed reachable.

    Matches the same 503-on-unavailability convention already used by
    webhooks.py's _get_review_launcher for a misconfigured or unreachable
    dependency - not a 200 with a "degraded" body. A caller (or an
    orchestrator's readiness probe) gets one clear signal: 200 means the
    service can actually do its job, 503 means it currently cannot.
    """

    try:
        settings = Settings()
    except ValidationError as exc:
        raise HTTPException(status_code=503, detail="Service configuration is invalid.") from exc

    if settings.database_url is None:
        raise HTTPException(status_code=503, detail="Database is not configured.")

    try:
        await Database(str(settings.database_url)).ping()
    except DatabaseError as exc:
        raise HTTPException(status_code=503, detail="Database is unreachable.") from exc

    return HealthResponse()
