"""The Model Service's response shape for one review call.

Deliberately minimal for now - Task 14 only proves the request/response
plumbing works end to end. The real findings schema (Severity, Component,
file, line, etc.) arrives with the reviewer prompts in a later task, and
will replace or extend this model then.
"""

from pydantic import BaseModel


class ModelReviewOutput(BaseModel):
    """The Model Service's raw text response, not yet parsed into findings."""

    raw_text: str
