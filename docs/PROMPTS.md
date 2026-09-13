# Reviewer Prompts

Every reviewer node's system instructions live as a plain-text file under
[`prompts/`](../prompts/), not as a Python string inside the node module.
This keeps prompt wording reviewable and editable on its own, separately
from the code that loads and uses it.

## Naming convention

One file per review category, named `<category>_review.md`, matching the
`review_category` value the node passes to `build_review_prompt()`:

- `security_review.md` - the Security reviewer.
- `architecture_review.md` - the Architecture reviewer.

A Quality reviewer will add `quality_review.md` here when it is built.

## How a prompt file is loaded

A reviewer node's factory function (e.g. `make_security_reviewer_node()` in
`src/review_agent/nodes/review_security.py`) reads its prompt file once, at
node-construction time - not on every review run. The loaded text becomes
`system_instructions` for that reviewer's calls to
`services/prompt_builder.py`'s `build_review_prompt()`, which wraps it
together with the PR's untrusted content and the output-format instructions.
A missing or unreadable prompt file raises `PromptTemplateError` immediately
when the node is built, rather than failing silently or only surfacing much
later on the first request against real PR content.

## What each prompt file contains

- **`security_review.md`** - Instructs the model to check for hardcoded
  secrets or credentials, exposed API keys, missing permission/authorization
  checks, and missing input validation - the four checks from
  `DataNet-Code-Review-Agent-SCOPE.md`'s Security review category. Ends with
  the required JSON output shape and a worked example, since security review
  applies to every PR regardless of which component changed.
- **`architecture_review.md`** - Instructs the model to check for missing
  separation of responsibility between components, a direct dependency that
  should instead go through an API (e.g. an Agent querying a Vector DB
  directly instead of the Search Service), and business logic leaking into
  the wrong component - drawn from [`architecture-baseline.md`](../../../DOCS/code-review-agent/architecture-baseline.md).
  Uses the same severity/component value lists and output-format instructions
  as `security_review.md`, since both validate against the same `Finding`
  model.
