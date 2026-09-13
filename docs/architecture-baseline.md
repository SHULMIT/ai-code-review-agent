# DataNet Architecture Baseline (Interim)

> This is an interim document, sourced from the original DataNet Code Review Agent task specification, until the DataNet project's own teams publish real architecture documentation on `main`.

## Architecture

An Agent must not query a Vector DB directly - it must go through the Search Service.

The Model Service must not contain Agent business logic.

## Agent Harness / LangGraph

State must be managed correctly.

There must be a clear separation between Nodes and Tools.

Tool failures must be handled, not left to crash the Agent.

Logging and tracing must be present.

Hallucination prevention: an Agent must not return an answer without a source that came from Retrieval.

## RAG and Search

Search results must contain a source.

Metadata must be preserved.

Grounding must be checked.

No direct information pass-through without validation.
