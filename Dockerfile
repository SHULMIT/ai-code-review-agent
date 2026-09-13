# syntax=docker/dockerfile:1
#
# Build from the repository root:
#
#   docker build -t code-review-agent .
#
# graph/build_graph.py resolves its documentation root relative to this
# file's location:
#
#   _SERVICE_ROOT     = <repository root> (parents[3] of build_graph.py)
#   DEFAULT_DOCS_ROOT  = _SERVICE_ROOT / "docs"
#
# config/document_sources.yaml currently approves "architecture-baseline.md"
# under it; without that file the documentation retrieval step silently
# returns nothing.


# ---------------------------------------------------------------- builder ---
# Pinned to a specific patch version, never a floating tag: prompts/security_review.md
# flags "a base image referenced by a mutable tag such as latest", and this
# image has to satisfy the checks this service itself enforces.
FROM python:3.12.13-slim-bookworm AS builder

# pyproject.toml sets [tool.uv] package = true and the repository ships a
# uv.lock, so uv is the dependency manager. Version matches the one this
# project is developed against (uv 0.10.9).
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the
# source, so editing a node does not reinstall the dependency tree.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# --frozen fails if uv.lock disagrees with pyproject.toml rather than silently
# re-resolving, which is the reproducibility the lockfile exists for.
# --no-dev keeps pytest and coverage out of the runtime image.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---------------------------------------------------------------- runtime ---
FROM python:3.12.13-slim-bookworm AS runtime

# git is a hard runtime requirement, not a build tool: tools/repo_search_tool.py
# calls asyncio.create_subprocess_exec("git", ...) to shallow-clone a pull
# request's commit. Without it every repository-context lookup fails.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# A fixed non-root UID. prompts/security_review.md flags "a container left
# running as root - a process that escapes the application then owns the
# container". The UID is explicit so bind-mounted file ownership is predictable.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin reviewer

WORKDIR /app

COPY --from=builder --chown=reviewer:reviewer /app/.venv ./.venv

# Only what the service reads at runtime. Everything here was confirmed by
# grepping for paths the code actually resolves:
#   src/        - the package itself
#   prompts/    - each reviewer node reads _SERVICE_ROOT / "prompts" at factory time
#   config/     - component_mapping.yaml and document_sources.yaml
#   migrations/ - the review_runs schema, for applying it against the database
#   docs/       - the approved documentation sources retrieve_documentation() cites
COPY --chown=reviewer:reviewer src ./src
COPY --chown=reviewer:reviewer prompts ./prompts
COPY --chown=reviewer:reviewer config ./config
COPY --chown=reviewer:reviewer migrations ./migrations
COPY --chown=reviewer:reviewer docs ./docs

# NOT copied, deliberately:
#   .env       - configuration is supplied as environment variables at run time
#   secrets/   - the GitHub App private key is mounted read-only at run time.
#                config.py resolves a relative github_app_private_key_path
#                against _SERVICE_ROOT, so mount it at /app/secrets.
#                prompts/security_review.md flags "a secret supplied as a build
#                argument, copied into the image, or written into a compose
#                file" - an image layer is readable by anyone who can pull it.
#   tests/     - not needed to run the service.

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER reviewer

# config.py defaults: host "0.0.0.0", port 8000.
EXPOSE 8000

# routes/health.py registers GET /health with no prefix, and its handler
# deliberately touches no external dependency - so this reports whether the
# process is serving, not whether GitHub or the Model Service are reachable.
# Uses the standard library rather than adding curl to the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# main.py guards run() behind `if __name__ == "__main__"`, which -m satisfies.
# run() calls Settings().validate_runtime_configuration() before starting, so
# a container with missing or production-invalid configuration exits at
# startup instead of serving in a broken state.
CMD ["python", "-m", "review_agent.main"]
