"""Documentation retrieval tool: plain keyword search over approved local Markdown docs.

No vector DB, no embeddings, per TECH-STACK.md - retrieve_documentation() is
the one place a future vector-based implementation would replace.
"""

import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ValidationError

from review_agent.models.finding import Component

_SNIPPET_MAX_CHARS = 300
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SERVICE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES_PATH = _SERVICE_ROOT / "config" / "document_sources.yaml"

COMPONENT_KEYWORDS: dict[Component, list[str]] = {
    Component.AGENT_HARNESS: ["Agent Harness", "LangGraph", "Vector DB"],
    Component.SEARCH_SERVICE: ["Search Service", "Search API"],
    Component.INDEXING_PIPELINE: ["Indexing Pipeline", "Document Loader"],
    Component.MODEL_SERVICE: ["Model Service", "Model Node"],
    Component.INFRASTRUCTURE: ["Infrastructure", "Docker", "Docker Compose"],
    Component.FRONTEND: ["Frontend"],
    Component.UNKNOWN: [],
}


class DocumentChunk(BaseModel):
    """One piece of approved documentation a Finding can cite as its source."""

    source_path: str
    heading: Optional[str] = None
    snippet: str


class DocumentSourceRole(BaseModel):
    """One documentation role's approval state in document_sources.yaml."""

    status: Literal["planned", "interim", "active"]
    approved_paths: list[str] = []


class DocumentSources(BaseModel):
    """The four documentation roles document_sources.yaml declares."""

    architecture: DocumentSourceRole
    api_contracts: DocumentSourceRole
    coding_guidelines: DocumentSourceRole
    service_docs: DocumentSourceRole

    def all_approved_paths(self) -> list[str]:
        """Flatten every role's approved_paths into one deduped, order-preserving list."""
        paths: list[str] = []
        for role in (self.architecture, self.api_contracts, self.coding_guidelines, self.service_docs):
            for path in role.approved_paths:
                if path not in paths:
                    paths.append(path)
        return paths


class DocRetrievalToolError(Exception):
    """Raised when reading the document-sources config or an approved documentation file fails, for any reason."""


def load_document_sources(path: Path) -> DocumentSources:
    """Read and validate document_sources.yaml into its four documentation roles."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return DocumentSources.model_validate(raw)


def retrieve_documentation(
    components: list[Component],
    docs_root: Path,
    sources_path: Path = DEFAULT_SOURCES_PATH,
) -> list[DocumentChunk]:
    """Return documentation chunks from approved sources that mention a keyword for any given component.

    Only files listed in document_sources.yaml's approved_paths (resolved
    relative to docs_root) are ever read - this replaces freely scanning
    every .md file under docs_root. A role with no approved_paths yet
    contributes nothing; that is a legitimate "no approved documentation
    exists yet" outcome, not a failure, so it never raises on its own.
    """
    try:
        sources = load_document_sources(sources_path)
    except OSError as exc:
        raise DocRetrievalToolError(
            f"failed to read document sources config at {sources_path}: {exc}"
        ) from exc
    except (yaml.YAMLError, ValidationError) as exc:
        raise DocRetrievalToolError(
            f"invalid document sources config at {sources_path}: {exc}"
        ) from exc

    results: list[DocumentChunk] = []
    for relative_path in sources.all_approved_paths():
        doc_path = docs_root / relative_path
        try:
            text = doc_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise DocRetrievalToolError(f"failed to read approved document {doc_path}: {exc}") from exc

        for heading, chunk_text in _extract_chunks(text):
            if _matches_any_component(chunk_text, components):
                results.append(
                    DocumentChunk(
                        source_path=doc_path.as_posix(),
                        heading=heading,
                        snippet=chunk_text.strip()[:_SNIPPET_MAX_CHARS],
                    )
                )
    return results


def _extract_chunks(text: str) -> list[tuple[Optional[str], str]]:
    """Split Markdown text into paragraph chunks paired with the nearest heading above each one.

    Only ATX-style headings (# through ######) are recognized; a heading
    line updates the current heading and is not itself returned as a chunk.
    """
    chunks: list[tuple[Optional[str], str]] = []
    current_heading: Optional[str] = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            chunks.append((current_heading, "\n".join(current_lines)))
            current_lines.clear()

    for line in text.splitlines():
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            current_heading = heading_match.group(2).strip()
            continue
        if line.strip() == "":
            flush()
            continue
        current_lines.append(line)
    flush()

    return chunks


def _matches_any_component(text: str, components: list[Component]) -> bool:
    """Return whether the chunk text mentions a keyword for any of the given components."""
    for component in components:
        keywords = COMPONENT_KEYWORDS.get(component, [])
        if keywords and _contains_any_keyword(text, keywords):
            return True
    return False


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)
