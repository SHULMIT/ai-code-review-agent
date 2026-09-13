# AI Code Review Agent

An autonomous code-review agent that reviews GitHub pull requests automatically. It listens for GitHub Webhooks, runs a six-category LangGraph review pipeline against the PR's diff and repository context, validates every finding against the actual code before it ships, and publishes one structured review back to GitHub as a PR Review — with live progress on a GitHub Check Run along the way.

It was built as a real service (Python 3.12, FastAPI, LangGraph, PostgreSQL), not a script: signed Webhooks, idempotent review runs, structured logging, a 96%+ tested codebase with an enforced 85% coverage gate, and a Docker image designed to run as a background service.

## What it does

1. A GitHub App receives `pull_request`, `check_run`, and `issue_comment` Webhook events, each verified with a constant-time HMAC signature check (`X-Hub-Signature-256`).
2. Opening or updating a PR creates/resets a **Check Run** carrying a `Run Code Review` action — review is on-demand (click the action, or comment `/review`), never automatic on every push.
3. Once triggered, the agent:
   - Fetches the PR's diff and changed files from the GitHub REST API.
   - Classifies which components the change touches (`component_mapping.yaml`) so only relevant review categories run.
   - Gathers extra context: a repository search tool (`git`-backed) for related definitions/usages, and a documentation-retrieval tool that cites only an explicit allowlist of approved docs (`document_sources.yaml`) — never a free scan of arbitrary files.
   - Runs up to six specialist reviewers **in parallel** as LangGraph nodes: Security, Architecture, Code Quality, Performance, RAG/Search Grounding, and Agent Harness (LangGraph-specific state/tooling review).
   - Deduplicates findings reported by more than one reviewer (embedding similarity + shared code-identifier matching), then validates every surviving finding is actually grounded in the diff or repository context before it can be published.
4. Publishes one PR Review with per-finding severity (`BLOCKING` / `WARNING` / `SUGGESTION`), component, file/line, problem, why it matters, a suggested fix, three scores (Architecture / Code Quality / Production Readiness), and a final `APPROVE` / `REQUEST_CHANGES` recommendation — while the Check Run shows live ✅/⏳/⚠️ per-category progress and links straight to the published review.
5. Every review run (repository, PR, head SHA, trigger, status, findings, scores) is recorded in PostgreSQL, so duplicate Webhook deliveries and duplicate `Run Code Review` clicks on the same PR/head SHA are rejected rather than starting a second run.

## Architecture

```mermaid
flowchart TD
    GH["GitHub Pull Request"] -- signed webhook --> API["FastAPI /webhooks/github"]
    API --> Launcher["Review Launcher\n(idempotent run tracking)"]
    Launcher --> Graph["LangGraph Review Pipeline"]

    subgraph Graph[" "]
        direction TB
        Load["load_pr / classify"] --> Gather["gather_context\n(repo search + doc retrieval)"]
        Gather --> Fan{"fan-out"}
        Fan --> R1["Security"]
        Fan --> R2["Architecture"]
        Fan --> R3["Code Quality"]
        Fan --> R4["Performance"]
        Fan --> R5["RAG / Search"]
        Fan --> R6["Agent Harness"]
        R1 & R2 & R3 & R4 & R5 & R6 --> Dedup["deduplicate_findings"]
        Dedup --> Verify["all_zero_verification"]
        Verify --> Finalize["finalize_review"]
    end

    Finalize --> Publish["Publish PR Review + Check Run"]
    Publish --> GH
    Launcher --> DB[("PostgreSQL\nreview_runs")]
    R1 & R2 & R3 & R4 & R5 & R6 -. calls .-> Model[("Model Service\n(Ollama-compatible)")]
```

Each reviewer node is a thin LangGraph node that calls one prompt-driven model request (`prompts/*.md`) and returns typed `Finding` objects — no business logic lives in the node itself (see [AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md) for the coding rules this codebase follows).

## Tech stack

| Layer | Choice |
| --- | --- |
| Language / runtime | Python 3.12 |
| HTTP framework | FastAPI |
| Agent workflow | LangGraph |
| Data validation | Pydantic v2 |
| GitHub / Model Service calls | httpx (async) |
| Review history | PostgreSQL |
| LLM access | Any Ollama-compatible endpoint (local Ollama by default), or a directly configured LLM provider |
| Local code search | `git` (shallow clone + text search) |
| Dependency management | [uv](https://docs.astral.sh/uv/) |
| Tests | pytest, pytest-asyncio, pytest-cov (85% coverage gate) |
| Packaging | Docker (multi-stage, non-root runtime user) |

## Project structure

```
src/review_agent/
  api/          shared error handling
  graph/        LangGraph state + graph assembly
  nodes/        one LangGraph node per pipeline step (thin, testable in isolation)
  tools/        GitHub API, repo search, doc retrieval, embeddings, Model Service client
  services/     business logic the nodes call: dedup, validation, formatting, launch orchestration
  routes/       FastAPI route handlers (webhooks, health) — thin, delegate to services
  storage/      PostgreSQL access for review-run history
  security/     webhook signature verification
  models/       Pydantic domain models (Finding, ReviewInput, ReviewResult, ...)
config/         component_mapping.yaml, document_sources.yaml (allowlists, not scans)
prompts/        one Markdown prompt per reviewer category
migrations/     PostgreSQL schema, applied in order
docs/           architecture baseline, scope, tech-stack, and prompt-design notes
scripts/        manual end-to-end run script, local dev startup/shutdown scripts
tests/          mirrors src/ 1:1, plus an end-to-end webhook-to-published-review test
```

## Setup

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/), Docker (for PostgreSQL), and [Ollama](https://ollama.com/) for local model inference (or any Ollama-compatible endpoint / LLM provider).

```bash
uv sync --extra dev
cp .env.example .env
```

Fill in `.env` — at minimum `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY_PATH`, `GITHUB_WEBHOOK_SECRET`, and `DATABASE_URL` to run the real Webhook flow. `MODEL_SERVICE_URL` defaults to a local Ollama instance (`http://localhost:11434`) if left unset. See the comments in [.env.example](.env.example) for every setting.

### Database

```bash
docker compose up -d
psql "$DATABASE_URL" -f migrations/001_create_review_runs.sql
psql "$DATABASE_URL" -f migrations/002_add_incomplete_categories.sql
psql "$DATABASE_URL" -f migrations/003_add_last_review_id.sql
psql "$DATABASE_URL" -f migrations/004_add_check_run_trigger.sql
psql "$DATABASE_URL" -f migrations/005_allow_retry_after_terminal_review_run.sql
```

The bundled `docker-compose.yml` starts Postgres on `localhost:5432` with `review_agent`/`review_agent` as both user and database, matching:

```
DATABASE_URL=postgresql://review_agent:review_agent@localhost:5432/review_agent
```

## Running it

```bash
uv run python -m review_agent.main
```

`GET /health` confirms the process is up and the database is reachable.

**Testing the real Webhook path locally** (Windows): after the setup above, `.\scripts\start_local_review.ps1` starts Postgres, the service, and a [Smee](https://smee.io/) relay so GitHub can reach `localhost`, then prints the URL to paste into the GitHub App's Webhook URL field. Stop everything with `.\scripts\stop_local_review.ps1`.

**Verifying the pipeline without a live Webhook:** `scripts/manual_run.py` runs the same LangGraph graph directly against a real PR and prints the full result (dry run by default; pass `--publish` to actually submit the review).

### Docker

```bash
docker build -t code-review-agent .
docker run --env-file .env -v "$(pwd)/secrets:/app/secrets:ro" -p 8000:8000 code-review-agent
```

## Tests

```bash
uv run pytest
```

Runs the full suite with coverage (`pytest-cov`), enforcing a minimum of 85% coverage on `src/review_agent` — the project currently sits at ~96%. Tests never call a real external service: GitHub, the Model Service, and PostgreSQL are all faked or mocked, including a full end-to-end test (`tests/e2e/test_sample_pr_flow.py`) that drives a signed Webhook payload all the way through the graph to a published review.

## Example

A signed `pull_request.opened` (or `synchronize`) Webhook delivery creates/resets the PR's Check Run without starting a review. Clicking that Check Run's **Run Code Review** action (or commenting `/review` on the PR) starts one:

```
POST /webhooks/github
X-GitHub-Event: check_run
X-Hub-Signature-256: sha256=<hmac>
X-GitHub-Delivery: <uuid>

{ "action": "requested_action", "requested_action": { "identifier": "run_code_review" }, ... }
```

The agent verifies the signature, checks the Check Run's head SHA still matches the PR's current head (refusing a stale click), and returns `202` while the review runs as a background task. Minutes later, the PR gets one Review comment structured like:

```markdown
# Code Review

## Recommendation: REQUEST_CHANGES

## Review Summary
Files reviewed: 4 · Blocking: 1 · Warnings: 2 · Suggestions: 1
Reviewers completed: 6/6 · Review completeness: Complete

## Findings

#### 1. Webhook signature compared with `==` instead of a constant-time check
**Category:** Security · **File:** `src/review_agent/security/webhook_signature.py` · **Line:** 22
**Problem:** ...
**Why it matters:** ...
**Suggested fix:** ...

## Reviewer Status
| Reviewer | Status | Findings | Reason |
| --- | --- | ---: | --- |
| Security | ✅ Completed | 1 | — |
| ... | | | |
```

## Documentation

- [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) — coding rules an AI agent (or a human) working in this codebase should follow: module boundaries, typing, testing, and security requirements.
- [`docs/SCOPE.md`](docs/SCOPE.md) — what this project's MVP does and deliberately does not do.
- [`docs/TECH-STACK.md`](docs/TECH-STACK.md) — the approved technology choices and why.
- [`docs/architecture-baseline.md`](docs/architecture-baseline.md) — the architecture rules the reviewers themselves check PRs against.
- [`docs/PROMPTS.md`](docs/PROMPTS.md) — how the reviewer prompts are structured and versioned.

## License

No license file is included yet — all rights reserved by default until one is added.
