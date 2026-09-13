# CLAUDE.md - DataNet Code Review Agent

This file guides any AI coding agent (Claude Code or similar) working in this repository. Read it before writing or editing any code.

## Project context

This is a single Python service that reviews DataNet pull requests automatically, using LangGraph. Full scope and technology decisions live in:

- [`docs/SCOPE.md`](docs/SCOPE.md) - what the MVP does and does not do.
- [`docs/TECH-STACK.md`](docs/TECH-STACK.md) - approved technologies and architecture.

If a task seems to require something listed under "Out of Scope" or "Postponed" in those files (a second service, a job queue, line-level comments, a vector database), stop and ask before implementing it. Do not silently expand scope. (The GitHub Checks API on-demand trigger - a `DataNet Code Review` Check Run with a `Run Code Review` requested action - is no longer on this list: it shipped as a deliberate scope change: see SCOPE.md Section 3 and TECH-STACK.md Section 6.)

## Tech stack (do not introduce alternatives without asking)

Python 3.12, FastAPI, Pydantic, LangGraph, httpx, pytest, PostgreSQL. No TypeScript, no Octokit, no Drizzle in this MVP.

## Code organization rules

- **One responsibility per module.** A file under `nodes/` contains one LangGraph node. A file under `tools/` contains one tool. Do not merge unrelated logic into one file because it's convenient.
- **One responsibility per function.** If a function does more than one thing (e.g. fetch data AND format it AND save it), split it. A good test: you should be able to describe the function in one sentence without using "and".
- **Nodes stay thin.** A LangGraph node function should read the state, call one or two tool functions, and return the changed fields. Business logic (parsing, scoring, classification rules) belongs in a plain helper function the node calls - not inline in the node body. This keeps nodes testable without spinning up the graph.
- **No global mutable state.** Pass data through the LangGraph state object, not module-level variables.
- **Errors never crash a node.** Every node that calls an external system (GitHub, the Model Service, the file system) catches its own exceptions and returns them into the `errors` field of the state - it does not let the exception propagate and stop the whole review.
- **New documentation lives under `docs/`.** Any new Markdown documentation file for this service is added there - never loose at the repository root or anywhere else.

## Designed for extension

Several capabilities are intentionally postponed for the MVP (see "Out of Scope" in SCOPE.md and "What Is Explicitly Postponed" in TECH-STACK.md): a second service, a job queue, line-level PR comments, the GitHub Checks API, and vector-database-based documentation retrieval. Write the MVP so these can be added later without rewriting existing code:

- Publishing a review result to GitHub goes through one function (e.g. `publish_review(result)`), never through inline API calls scattered across nodes. Swapping "one summary comment" for "Checks plus line comments" later means changing this one function's internals, not every caller.
- Saving review history goes through one function (e.g. `save_review_run(...)`), not direct database calls from multiple places. A future job-queue system replaces this function's internals, not its callers.
- The Finding model always includes `file` and `line` fields, even though the MVP only publishes a single summary comment. The data is ready for line-level comments before that feature exists.
- Documentation retrieval goes through one function (e.g. `retrieve_docs(query)`). Its current implementation is keyword search; a future vector-database implementation replaces only what's inside this function.
- A new review category (e.g. a future Breaking API reviewer) is added as a new node plus a new edge - existing nodes are not modified to accommodate it.

Do not build any of the postponed capabilities now. This section only asks that today's interfaces have a natural extension point, not that today's code implements tomorrow's feature.

## Typing and validation

- Every public function has type hints on parameters and return value.
- Every data structure that crosses a boundary (API request/response, LangGraph state, a tool's return value) is a Pydantic model or a `TypedDict` - not a raw `dict`.
- Validate at the edges: parse and validate GitHub payloads and LLM responses immediately when they arrive, not several steps later.

## Comments and docstrings

- Every public function and class gets a one-to-three-line docstring: what it does, and anything non-obvious about its inputs or side effects. Do not restate the function name in prose.
- Inline comments are for **why**, not **what**. `# retry once - GitHub occasionally 500s on first call after token refresh` is useful. `# loop over files` above a for-loop is not - delete it.
- If a piece of logic exists because of a specific requirement (e.g. "line comments limited to 10 per run"), reference it in a comment so a future reader knows it's intentional, not an arbitrary number.
- No commented-out code left in commits. Delete it; git history keeps it if it's ever needed again.

## Naming

- Descriptive names over short ones: `changed_files`, not `cf`. `architecture_reviewer_node`, not `arch_node`.
- Boolean variables and functions read as a yes/no question: `has_source`, `is_grounded`, `requires_review`.
- Match the vocabulary already used in `docs/SCOPE.md` and the original task document (`Component`, `Finding`, `Severity`, `Blocking`/`Warning`/`Suggestion`) - do not invent synonyms for concepts that already have a name in the spec.

## Testing

- Every new node, tool, or scoring function ships with a pytest test in the same PR that adds it - not "added later".
- Tests for nodes use fake tools (a fake GitHub client, a fake Model Service client) - never call real external services in tests.
- At least one test per module covers a failure path (tool raises, LLM returns invalid JSON, file not found) - not only the happy path.
- Do not skip or comment out a failing test to make CI pass. Fix it or flag it explicitly and ask.
- This project enforces a minimum of 85% test coverage on `src/review_agent`, checked automatically on every `pytest` run. New code that drops coverage below this threshold must not be merged - add the missing tests rather than lowering the threshold.

## Efficient use of the AI coding session

- Always use the remote Model Service (MODEL_SERVICE_URL pointing at the remote Ollama instance, not localhost) for any real end-to-end run against a live model, unless explicitly testing local-CPU behavior itself. A single local /api/generate call can take 5-160+ seconds versus a few seconds remotely - defaulting to local wastes session time for no benefit once a faster option exists.
- When reporting pytest results, show only pass/fail count and total coverage percentage by default. Only show the full per-file coverage breakdown when something failed or coverage dropped below the required threshold - the full table is rarely needed and costs unnecessary output.
- Do not re-run a full manual_run.py against a real PR after every small change. Reserve that for changes that could plausibly affect end-to-end behavior (a node's logic, a prompt's content, the graph's wiring). A documentation-only change, a comment fix, or a test-only addition only needs `pytest` to pass - not a full live run.
- When a task is low-risk (a documentation update, a config default, a prompt wording fix) and an investigation step confirms the assumption behind it, proceed directly to implementing the fix in the same turn rather than stopping to report findings and waiting for a separate go-ahead - reserve the report-then-wait pattern for genuinely uncertain or higher-risk changes (anything touching finding_deduplication.py, review_launcher.py, the publish/scoring logic, or shared state).
- Default to `git diff --stat` (file names and line counts) rather than a full `git diff` when reporting what changed. Only show the full diff content when specifically asked, or when the change is to a critical shared file (graph/state.py, finding_deduplication.py, config.py) where the exact logic needs review.
- Finish and commit a task before ending a session whenever the work is in a stable, tested state - do not leave functional work uncommitted "for later" if it already passes its own tests, since a future session has to re-read and re-verify uncommitted context from scratch, which costs time neither of us gets back.
- **Never run the full test suite by default.** Run only the specific test file(s) directly covering the code you changed (e.g. changed `finding_deduplication.py` → run only `test_finding_deduplication.py`). This applies even mid-task, after every small edit.
- **Run the full suite in exactly two situations, and no others:** (1) immediately before a commit, as the final check, or (2) when the change touches a file imported by many others (graph/state.py, config.py, build_graph.py, finding_deduplication.py, review_launcher.py) - and even then, only once, not after every intermediate edit to those files.
- If you're unsure whether a change is "widely shared" enough to warrant a full run, default to NOT running it - ask me first instead of running the full suite "just in case."
- A full run costs real session budget every time - treat it as an expensive operation to be requested deliberately, not a routine sanity check.
- When waiting on a long-running background process (a live model run, a slow command), check its status at most once per minute, not continuously/immediately - polling too frequently wastes session overhead without providing useful information sooner.

## Security

- Never hardcode a secret, token, or API key, even temporarily "to test something."
- Never log a full webhook payload, a full diff, or any GitHub token - log identifiers (PR number, repo, head SHA) instead.
- Treat PR file contents as untrusted input: read them as text for analysis, never execute, `eval`, or import them.

## Before opening a PR

- **Work on a dedicated feature branch, then open a PR.** Direct pushes to `main` are not permitted - branch off it, do the work there, and submit a Pull Request into `main` when done.
- Run `pytest` and confirm it passes.
- Re-read the function/file you changed once as if you were reviewing someone else's code - if you'd flag it as a code reviewer, fix it before asking a human to look.
- Keep commits small and scoped to one change - do not bundle unrelated changes.

## What to do when unsure

If a requirement in the task conflicts with something already implemented, or a design decision isn't covered by `docs/SCOPE.md` / `docs/TECH-STACK.md`, stop and ask rather than guessing. State the conflict and the options - do not pick one silently.
