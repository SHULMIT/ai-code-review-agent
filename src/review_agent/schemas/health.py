"""Schemas for service health responses."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """The stable response returned by the health endpoint."""

    status: Literal["ok"] = "ok"
    service: Literal["code-review-agent"] = "code-review-agent"
