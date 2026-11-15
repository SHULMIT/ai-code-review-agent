<div align="center">

# 🔍 AI Code Review Agent

### Six specialist AI reviewers. One grounded verdict. Straight inside GitHub.

*An autonomous pull-request reviewer built on LangGraph: it runs six review
categories in parallel, verifies every finding against the real code, and
publishes one structured PR Review with live progress on a GitHub Check Run.*

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic_v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-compatible-000000?style=for-the-badge&logo=ollama&logoColor=white)
![GitHub App](https://img.shields.io/badge/GitHub_App-181717?style=for-the-badge&logo=github&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-~96%25-2EA44F?style=for-the-badge&logo=pytest&logoColor=white)

<br/>

[Demo](#demo) •
[How It Works](#how-it-works) •
[Architecture](#architecture) •
[Quick Start](#quick-start) •
[Example Output](#example-output) •
[Testing](#testing) •
[Docs](#documentation)

</div>

---

<a id="demo"></a>

## Demo

<div align="center">

<img src="docs/assets/review-demo.gif" alt="AI Code Review Agent running on a GitHub pull request" width="880"/>

<sub>Click <b>Run Code Review</b> on the Check Run → six reviewers run in parallel →
a grounded review with blocking issues, warnings and suggested fixes is published to the PR.</sub>

</div>

---

## Why this agent?

| | |
| --- | --- |
| **Grounded, not guessed** | Every finding is verified against the diff or repository context before it can be published — no hallucinated issues. |
| **Specialist reviewers** | Six focused LangGraph nodes, each with its own prompt: Security, Architecture, Code Quality, Performance, RAG/Search, Agent Harness. |
| **Native GitHub UX** | Lives where developers already are: a Check Run with a **Run Code Review** button, live per-category progress, and a real PR Review. |
| **On-demand, idempotent** | Runs only when asked (button or `/review`), and duplicate deliveries or clicks on the same head SHA never start a second run. |
| **Built as a service** | Signed Webhooks, structured logging, PostgreSQL run history, Docker image, ~96% test coverage with an enforced 85% gate. |

---

<a id="how-it-works"></a>

## How It Works

```mermaid
sequenceDiagram
    autonumber
    participant Dev as Developer
    participant GH as GitHub
    participant API as FastAPI Webhook
    participant LG as LangGraph Pipeline
    participant LLM as Model Service
    participant DB as PostgreSQL

    Dev->>GH: Open / update pull request
    GH->>API: pull_request webhook (HMAC-signed)
    API->>GH: Create Check Run with "Run Code Review"
    Dev->>GH: Click Run Code Review (or comment /review)
    GH->>API: check_run.requested_action
    API->>DB: Record run (reject duplicate head SHA)
    API-->>GH: 202 Accepted
    API->>LG: Start review in background
    LG->>GH: Fetch diff & changed files
    par Six reviewers in parallel
        LG->>LLM: Security · Architecture · Code Quality
        LG->>LLM: Performance · RAG/Search · Agent Harness
    end
    LG->>LG: Deduplicate + verify every finding
    LG->>GH: Publish PR Review + update Check Run
    LG->>DB: Save findings, scores, status
```

1. **Trigger** — a GitHub App receives `pull_request`, `check_run` and `issue_comment` events, each verified with a constant-time HMAC check (`X-Hub-Signature-256`). Opening or updating a PR creates/resets a **Check Run** carrying a `Run Code Review` action; review never runs automatically on every push.
2. **Load & classify** — fetches the PR diff and changed files, then maps them to components (`component_mapping.yaml`) so only relevant reviewers run.
3. **Gather context** — a `git`-backed repository search finds related definitions and usages, and a documentation tool cites **only** an explicit allowlist (`document_sources.yaml`), never a free scan of arbitrary files.
4. **Review in parallel** — up to six specialist reviewers run as LangGraph nodes.
5. **Deduplicate & verify** — merges overlapping findings (embedding similarity + shared code identifiers), then checks that each surviving finding is grounded in the diff or repository context.
6. **Publish** — one PR Review with severities, file/line, problem, impact, suggested fix, three scores and a final `APPROVE` / `REQUEST_CHANGES`, while the Check Run shows live ✅ / ⏳ / ⚠️ progress per category.
7. **Record** — every run (repository, PR, head SHA, trigger, status, findings, scores) is stored in PostgreSQL.

---

<a id="architecture"></a>

## Architecture

```mermaid
flowchart TD
    GH["GitHub Pull Request"] -- signed webhook --> API["FastAPI<br/>/webhooks/github"]
    API --> Launcher["Review Launcher<br/>(idempotent run tracking)"]
    Launcher --> Load

    subgraph Graph["LangGraph Review Pipeline"]
        direction TB
        Load["load_pr / classify"] --> Gather["gather_context<br/>(repo search + doc retrieval)"]
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

    Finalize --> Publish["Publish PR Review<br/>+ Check Run"]
    Publish --> GH
    Launcher --> DB[("PostgreSQL<br/>review_runs")]
    R1 & R2 & R3 & R4 & R5 & R6 -. prompts .-> Model[("Model Service<br/>(Ollama-compatible)")]

    style GH fill:#24292F,stroke:#24292F,color:#fff
    style API fill:#009688,stroke:#00796B,color:#fff
    style Launcher fill:#0984E3,stroke:#0652DD,color:#fff
    style Graph fill:#F3E8FD,stroke:#6C5CE7,stroke-width:2px
    style Fan fill:#6C5CE7,stroke:#4834d4,color:#fff
    style R1 fill:#D63031,stroke:#b71540,color:#fff
    style R2 fill:#E17055,stroke:#c0392b,color:#fff
    style R3 fill:#00B894,stroke:#009432,color:#fff
    style R4 fill:#FDCB6E,stroke:#F79F1F,color:#2d3436
    style R5 fill:#74B9FF,stroke:#0984E3,color:#2d3436
    style R6 fill:#A29BFE,stroke:#6C5CE7,color:#fff
    style Verify fill:#2D3436,stroke:#000,color:#fff
    style Publish fill:#2EA44F,stroke:#22863A,color:#fff
    style DB fill:#4169E1,stroke:#2F4FB5,color:#fff
    style Model fill:#000,stroke:#000,color:#fff
```

Each reviewer is a **thin** LangGraph node that makes one prompt-driven model
request (`prompts/*.md`) and returns typed `Finding` objects. Business logic
lives in `services/`, never inside a node — see [AGENTS.md](AGENTS.md) /
[CLAUDE.md](CLAUDE.md) for the coding rules this codebase follows.

### The six reviewers

| Reviewer | Focus |
| --- | --- |
| **Security** | Secrets exposure, missing input validation, missing authorization checks |
| **Architecture** | Separation of responsibilities per the [architecture baseline](docs/architecture-baseline.md) |
| **Code Quality** | Naming, duplication, error handling, logging, type safety, tests, documentation |
| **Performance** | Whether work is bounded in memory, time and concurrency at terabyte scale |
| **RAG / Search** | Provenance, validation and grounding of retrieved content and model answers |
| **Agent Harness** | Graph state, nodes, tools, failure handling, and treating model output as untrusted |

### Tech stack

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

<details>
<summary><b>Project structure</b></summary>

<br/>

```text
src/review_agent/
├── api/          Shared error handling
├── graph/        LangGraph state + graph assembly
├── nodes/        One LangGraph node per pipeline step (thin, testable in isolation)
├── tools/        GitHub API, repo search, doc retrieval, embeddings, Model Service client
├── services/     Business logic: dedup, validation, formatting, launch orchestration
├── routes/       FastAPI route handlers (webhooks, health), delegating to services
├── storage/      PostgreSQL access for review-run history
├── security/     Webhook signature verification
└── models/       Pydantic domain models (Finding, ReviewInput, ReviewResult, ...)
config/           component_mapping.yaml, document_sources.yaml (allowlists, not scans)
prompts/          One Markdown prompt per reviewer category
migrations/       PostgreSQL schema, applied in order
docs/             Architecture baseline, scope, tech stack, prompt-design notes
scripts/          Manual end-to-end run, local dev startup/shutdown scripts
tests/            Mirrors src/ 1:1, plus an end-to-end webhook-to-published-review test
```
</details>

---

<a id="quick-start"></a>

## Quick Start

> [!NOTE]
> **Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/), Docker (for
> PostgreSQL), and [Ollama](https://ollama.com/) or any Ollama-compatible
> endpoint / LLM provider.

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Configure
cp .env.example .env

# 3. Start PostgreSQL
docker compose up -d

# 4. Run the service
uv run python -m review_agent.main
```

`GET /health` confirms the process is up and the database is reachable.

<details>
<summary><b>Environment variables</b></summary>

<br/>

To run the real Webhook flow, set at least:

| Variable | Purpose |
| --- | --- |
| `GITHUB_APP_ID` | Your GitHub App's ID |
| `GITHUB_APP_PRIVATE_KEY_PATH` | Path to the App's private key |
| `GITHUB_WEBHOOK_SECRET` | Secret used to verify `X-Hub-Signature-256` |
| `DATABASE_URL` | PostgreSQL connection string |
| `MODEL_SERVICE_URL` | Model endpoint; defaults to local Ollama (`http://localhost:11434`) |

Every setting is documented in [.env.example](.env.example).
</details>

<details>
<summary><b>Database migrations</b></summary>

<br/>

The bundled `docker-compose.yml` starts Postgres on `localhost:5432` with
`review_agent` / `review_agent` as user and database:

```
DATABASE_URL=postgresql://review_agent:review_agent@localhost:5432/review_agent
```

Apply the schema in order:

```bash
psql "$DATABASE_URL" -f migrations/001_create_review_runs.sql
psql "$DATABASE_URL" -f migrations/002_add_incomplete_categories.sql
psql "$DATABASE_URL" -f migrations/003_add_last_review_id.sql
psql "$DATABASE_URL" -f migrations/004_add_check_run_trigger.sql
psql "$DATABASE_URL" -f migrations/005_allow_retry_after_terminal_review_run.sql
```
</details>

<details>
<summary><b>Run with Docker</b></summary>

<br/>

```bash
docker build -t code-review-agent .
docker run --env-file .env -v "$(pwd)/secrets:/app/secrets:ro" -p 8000:8000 code-review-agent
```
</details>

<details>
<summary><b>Test the real Webhook locally (Windows)</b></summary>

<br/>

```powershell
.\scripts\start_local_review.ps1   # Postgres + service + Smee relay
.\scripts\stop_local_review.ps1    # stop everything
```

The start script prints a [Smee](https://smee.io/) URL to paste into the GitHub
App's Webhook URL field so GitHub can reach `localhost`.
</details>

<details>
<summary><b>Run the pipeline without a Webhook</b></summary>

<br/>

`scripts/manual_run.py` runs the same LangGraph graph directly against a real
PR and prints the full result. It is a dry run by default; pass `--publish` to
actually submit the review.
</details>

---

<a id="example-output"></a>

## Example Output

A signed `check_run` delivery for the **Run Code Review** action:

```http
POST /webhooks/github
X-GitHub-Event: check_run
X-Hub-Signature-256: sha256=<hmac>
X-GitHub-Delivery: <uuid>

{ "action": "requested_action", "requested_action": { "identifier": "run_code_review" }, ... }
```

The agent verifies the signature, refuses a stale click whose head SHA no
longer matches the PR, and returns `202` while the review runs in the
background. The PR then receives one Review:

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

| Severity | Meaning |
| --- | --- |
| 🔴 **BLOCKING** | Must be fixed before merging |
| 🟡 **WARNING** | Should be addressed; real risk or debt |
| 🔵 **SUGGESTION** | Optional improvement |

Each review also carries three scores (**Architecture**, **Code Quality**,
**Production Readiness**) and a final `APPROVE` / `REQUEST_CHANGES`.

---

<a id="testing"></a>

## Testing

```bash
uv run pytest
```

Runs the full suite with coverage and **fails below 85%** on
`src/review_agent`; the project currently sits at **~96%**.

- No real external services in tests: GitHub, the Model Service and PostgreSQL
  are faked or mocked.
- A full end-to-end test (`tests/e2e/test_sample_pr_flow.py`) drives a signed
  Webhook payload all the way through the graph to a published review.

---

## Security

- Constant-time HMAC verification of every Webhook (`X-Hub-Signature-256`)
- No secrets in code; tokens, full payloads and full diffs are never logged
- PR contents are treated as **untrusted input**: read as text, never executed, `eval`'d or imported
- Documentation retrieval limited to an explicit allowlist
- Idempotent runs: duplicate deliveries and repeated clicks on the same head SHA are rejected

---

<a id="documentation"></a>

## Documentation

| Document | Contents |
| --- | --- |
| [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) | Coding rules for AI agents and humans: module boundaries, typing, testing, security |
| [`docs/SCOPE.md`](docs/SCOPE.md) | What the MVP does, and deliberately does not do |
| [`docs/TECH-STACK.md`](docs/TECH-STACK.md) | Approved technology choices and why |
| [`docs/architecture-baseline.md`](docs/architecture-baseline.md) | Architecture rules the reviewers check PRs against |
| [`docs/PROMPTS.md`](docs/PROMPTS.md) | How reviewer prompts are structured and versioned |

---

## License

No license has been specified yet. Until one is added, all rights are reserved
by the author.

---

<div align="center">

**Built by [SHULMIT](https://github.com/SHULMIT)**

If you find this project useful, give it a ⭐ on GitHub!

</div>
