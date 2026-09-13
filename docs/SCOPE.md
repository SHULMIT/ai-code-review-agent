# DataNet Code Review Agent - MVP Scope (Reduced)

**Status:** Proposed reduction of v1.0, per team lead feedback
**Version:** 2.0
**Project:** DataNet

## 1. Why this version is smaller

The team lead reviewed the original plan and asked the team to first build one working end-to-end flow: the Agent connects to GitHub, receives a Pull Request, reads the diff and relevant context from the repository and from DataNet's documentation, and returns an organized Code Review with issues and recommendations based on the DataNet architecture.

Advanced production mechanisms (complex history management, advanced protections, a metrics dashboard, and full production layers) are postponed until the basic flow is proven to work and to produce good findings.

## 2. Goal

Build an automated Code Review Agent for DataNet pull requests. The agent inspects proposed code changes, retrieves relevant repository and architecture context, applies DataNet-specific review rules, and publishes one clear, actionable review comment on the PR.

## 3. MVP Triggers

**Revised (see Definition of Done for Scope Task): the review is now on-demand, not automatic on every push.**

- A pull request is opened, or updated with a new commit (`synchronize`): this only creates or resets the **`DataNet Code Review` GitHub Check Run** into a completed/neutral "ready to run" state carrying a **`Run Code Review`** requested action. It does not, by itself, start a review or call the Model Service.
- Clicking that Check Run's **`Run Code Review`** action (`check_run.requested_action` with `requested_action.identifier == "run_code_review"`) is what actually starts a review - verified against the PR's current head SHA first, so a Check Run superseded by a newer push is never used to review a stale commit.
- A manual rerun through a `/review` comment on the PR still starts a review directly, unchanged.

`pull_request.reopened` is still postponed (see Out of Scope). GitHub's own Check **Re-run** button convenience (automatic on certain conclusions in Actions-integrated contexts) does not apply to a third-party App's Check Run; the `Run Code Review` requested action is this App's equivalent, and it is always attached to a completed Check Run's output so the same Check Run can be triggered again.

## 4. GitHub Integration

- A GitHub App with signed Webhooks receives PR, issue comment, and Check Run events.
- Every incoming Webhook is authenticated using its signature (`X-Hub-Signature-256`).
- The Agent reads PR metadata, the diff, and changed files through the GitHub REST API.
- The Agent uses the **GitHub Checks API** to create/reset the `DataNet Code Review` Check Run, move it to `in_progress` when a review actually starts, and complete it (`success` / `action_required` / `neutral` / `failure`) once the run finishes - this is the on-demand trigger and status surface described in Section 3.
- The Agent publishes **one summary comment**, as an official GitHub Pull Request Review, per review run. This is unchanged by the Check Run: the Check Run starts and reports on a run, it never replaces this review output.

Duplicate GitHub deliveries (same delivery ID) must not create duplicate review runs. The same review-identity uniqueness check also prevents a duplicate "Run Code Review" click (same PR, same head SHA) from starting a second run while one is already pending or running. This is a simple check-and-skip, not a transactional job queue (see Out of Scope).

## 5. Review Output

A single PR comment containing:

- PR Summary.
- Architecture Score.
- Code Quality Score.
- Production Readiness Score.
- Blocking Issues.
- Warnings.
- Suggestions.
- Final recommendation: `APPROVE` or `REQUEST_CHANGES`.

Each issue includes: Severity, Component, File, Line, Problem, Why it matters, Suggested Fix.

**Line-level inline comments are postponed** (see Out of Scope). All findings, including blocking ones, appear in the single summary comment for the MVP.

## 6. Review Categories

Only categories relevant to the changed components run:

- DataNet architecture and separation of responsibilities.
- Agent Harness / LangGraph state, nodes, tools, failure handling, logging, hallucination prevention.
- RAG and Search grounding, sources, metadata, validation.
- Performance, async processing, batching, scalability risks.
- Code quality: naming, duplication, error handling, logging, type safety, tests, documentation.
- Security: secrets, API keys, permissions, input validation.

## 7. Required Agent Tools

- **GitHub PR Tool** - reads PR metadata, diffs, and changed files (via the GitHub REST API).
- **Repository Search Tool** - finds relevant definitions and usages in the existing repository.
- **Documentation Retrieval Tool** - retrieves DataNet architecture documentation, coding guidelines, API contracts, and service documentation, using simple document chunking and keyword search (no vector database for the MVP).

## 8. Review History and Reruns

The system stores basic review-run history:

- Repository and PR number.
- Reviewed head commit SHA.
- Trigger type.
- Execution status and timestamp.
- Scores, findings, recommendation, errors.

A new commit or a `/review` comment creates a new review attempt tied to the current head SHA. This is a simple append-only history table, not a job-processing system with locking, leases, or retries (see Out of Scope).

## 9. Reliability and Safety Boundaries

- Return quickly from the Webhook endpoint and process the review as a background task.
- Avoid publishing findings that are not grounded in the diff, repository context, or approved documentation.
- Treat PR code as untrusted input; never execute it during context retrieval.
- Never store secrets in code or logs.
- Record tool and model failures without inventing missing information.

## 10. Out of Scope for this MVP

Postponed until the basic end-to-end flow works reliably:

- A second, separate GitHub Integration service (TypeScript/Fastify/Octokit). The MVP uses **one Python service** for everything.
- A durable job queue with transactional claiming, row locking, leases, and bounded retries.
- Line-by-line inline PR comments (only the single summary comment ships in this MVP).
- `pull_request.reopened` handling.
- Stale-result protection for a newer commit arriving **while a review is already running** (a mid-flight race). A narrower, click-time check does exist (Section 3): a `Run Code Review` click is refused, not queued, when its Check Run's head SHA no longer matches the PR's current head SHA.
- Automatic code modification or generated fix commits.
- Advanced breaking-API detection.
- Automatic comparison against an architecture diagram.
- A metrics dashboard and long-term analytics.
- Vector database / embeddings-based documentation retrieval.

## 11. MVP Acceptance Criteria

The MVP is complete when:

- Opening or updating a PR creates or resets its `DataNet Code Review` Check Run, without starting a review by itself; clicking that Check Run's `Run Code Review` action (or commenting `/review`) starts one authenticated review run.
- The run reviews the latest head commit and records its SHA.
- A single, well-formatted PR comment is published with the required format.
- Duplicate Webhook deliveries, and a duplicate `Run Code Review` click on the same PR/head SHA, do not create duplicate runs.
- A sample PR containing known DataNet violations is detected correctly.
- Automated tests cover Webhook validation, event routing, component classification, result formatting, and reruns.
- Basic architecture, workflow, tools, and prompt documentation are available.

## 12. Definition of Done for Scope Task

This reduced scope is approved when the team agrees these are the first deliverable, and that everything under **Out of Scope** will not delay this MVP.
