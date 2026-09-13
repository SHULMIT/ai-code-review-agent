"""Authenticated GitHub Webhook endpoint."""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from review_agent.config import Settings
from review_agent.schemas.webhooks import (
    CheckRunWebhookPayload,
    IssueCommentWebhookPayload,
    PullRequestWebhookPayload,
)
from review_agent.security.webhook_signature import verify_signature
from review_agent.services.check_run_launcher import (
    CheckRunLauncher,
    CheckRunLauncherError,
    build_check_run_launcher,
)
from review_agent.services.event_router import (
    route_check_run_action,
    route_check_run_creation,
    route_github_event,
)
from review_agent.services.review_launcher import (
    ReviewLauncher,
    ReviewLauncherConfigurationError,
    build_review_launcher,
)

_SUPPORTED_EVENTS = {"pull_request", "issue_comment", "check_run"}

router = APIRouter(tags=["webhooks"])


def parse_github_webhook_payload(
    event_name: str | None, payload_body: bytes
) -> PullRequestWebhookPayload | IssueCommentWebhookPayload | CheckRunWebhookPayload:
    """Validate a signed delivery with the model selected by event name."""

    try:
        if event_name == "pull_request":
            return PullRequestWebhookPayload.model_validate_json(payload_body)
        if event_name == "issue_comment":
            return IssueCommentWebhookPayload.model_validate_json(payload_body)
        if event_name == "check_run":
            return CheckRunWebhookPayload.model_validate_json(payload_body)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    raise HTTPException(status_code=400, detail="Unsupported GitHub event.")


@router.post("/webhooks/github")
async def receive_github_webhook(
    request: Request, background_tasks: BackgroundTasks
) -> JSONResponse:
    """Verify a raw GitHub delivery before validating its JSON payload and dispatching it.

    pull_request no longer launches a review directly: opened/synchronize
    only create or reset the on-demand "DataNet Code Review" Check Run.
    check_run only ever launches a review for its exact "Run Code Review"
    requested action. issue_comment's "/review" command is unchanged.
    """

    payload_body = await request.body()
    signature_header = request.headers.get("X-Hub-Signature-256")
    webhook_secret = Settings().github_webhook_secret.get_secret_value()

    if signature_header is None or not verify_signature(
        payload_body, signature_header, webhook_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid GitHub Webhook signature.")

    event_name = request.headers.get("X-GitHub-Event")
    if event_name not in _SUPPORTED_EVENTS:
        return JSONResponse(status_code=200, content={"status": "ignored"})

    payload = parse_github_webhook_payload(event_name, payload_body)

    if isinstance(payload, PullRequestWebhookPayload):
        return await _handle_pull_request_event(request, event_name, payload)
    if isinstance(payload, CheckRunWebhookPayload):
        return await _handle_check_run_event(
            request, event_name, payload, background_tasks
        )
    return await _handle_issue_comment_event(
        request, event_name, payload, background_tasks
    )


async def _handle_pull_request_event(
    request: Request, event_name: str, payload: PullRequestWebhookPayload
) -> JSONResponse:
    """Create or reset the Check Run for a safe opened/synchronize push; never launches a review."""

    creation_request = route_check_run_creation(event_name, payload)
    if creation_request is None:
        return JSONResponse(status_code=200, content={"status": "ignored"})

    check_run_launcher = _get_check_run_launcher(request)
    try:
        await check_run_launcher.handle_pull_request_event(creation_request)
    except CheckRunLauncherError as exc:
        raise HTTPException(
            status_code=502, detail="Unable to create the Code Review Check Run."
        ) from exc
    return JSONResponse(status_code=200, content={"status": "check_run_created"})


async def _handle_check_run_event(
    request: Request,
    event_name: str,
    payload: CheckRunWebhookPayload,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Launch a review only for the exact "Run Code Review" requested action."""

    action_request = route_check_run_action(event_name, payload)
    if action_request is None:
        return JSONResponse(status_code=200, content={"status": "ignored"})

    github_delivery_id = _require_delivery_id(request)
    check_run_launcher = _get_check_run_launcher(request)
    launch_result = await check_run_launcher.handle_requested_action(
        action_request, github_delivery_id, background_tasks
    )
    status_code = 202 if launch_result.accepted else 200
    return JSONResponse(status_code=status_code, content=launch_result.model_dump())


async def _handle_issue_comment_event(
    request: Request,
    event_name: str,
    payload: IssueCommentWebhookPayload,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Launch a review directly for the "/review" comment command - unchanged by this feature."""

    review_request = route_github_event(event_name, payload)
    if review_request is None:
        return JSONResponse(status_code=200, content={"status": "ignored"})

    github_delivery_id = _require_delivery_id(request)
    launcher = _get_review_launcher(request)
    launch_result = await launcher.launch(review_request, github_delivery_id, background_tasks)
    status_code = 202 if launch_result.accepted else 200
    return JSONResponse(status_code=status_code, content=launch_result.model_dump())


def _require_delivery_id(request: Request) -> str:
    """Return GitHub's delivery ID, required for atomic review-launch deduplication."""

    github_delivery_id = request.headers.get("X-GitHub-Delivery")
    if not github_delivery_id:
        raise HTTPException(status_code=400, detail="Missing GitHub delivery ID.")
    return github_delivery_id


def _get_review_launcher(request: Request) -> ReviewLauncher:
    """Return the configured launcher or lazily build it from validated settings."""

    launcher = getattr(request.app.state, "review_launcher", None)
    if launcher is not None:
        return launcher

    try:
        launcher = build_review_launcher(Settings())
    except (ReviewLauncherConfigurationError, ValidationError) as exc:
        raise HTTPException(status_code=503, detail="Review launcher is unavailable.") from exc

    request.app.state.review_launcher = launcher
    return launcher


def _get_check_run_launcher(request: Request) -> CheckRunLauncher:
    """Return the configured Check Run launcher or lazily build it, reusing the review launcher."""

    check_run_launcher = getattr(request.app.state, "check_run_launcher", None)
    if check_run_launcher is not None:
        return check_run_launcher

    check_run_launcher = build_check_run_launcher(Settings(), _get_review_launcher(request))
    request.app.state.check_run_launcher = check_run_launcher
    return check_run_launcher
