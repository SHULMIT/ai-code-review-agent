import shutil
import tempfile
from pathlib import Path

import pytest

from review_agent.models.finding import Component
from review_agent.tools.doc_retrieval_tool import (
    DocumentChunk,
    DocRetrievalToolError,
    load_document_sources,
    retrieve_documentation,
)


@pytest.fixture
def tmp_path():
    # Overrides pytest's built-in tmp_path: its pytest-of-<user> retention
    # directory needs symlink permissions that this machine denies
    # (WinError 5). A plain mkdtemp sidesteps that entirely.
    path = Path(tempfile.mkdtemp(prefix="review_agent_doc_test_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _write_sources_yaml(path: Path, **approved_paths_by_role: list[str]) -> None:
    """Write a document_sources.yaml fixture; roles not passed default to empty/planned."""
    roles = {"architecture": [], "api_contracts": [], "coding_guidelines": [], "service_docs": []}
    roles.update(approved_paths_by_role)
    lines: list[str] = []
    for role, paths in roles.items():
        lines.append(f"{role}:")
        lines.append(f"  status: {'active' if paths else 'planned'}")
        lines.append(f"  approved_paths: {paths!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_retrieve_documentation_returns_empty_when_nothing_is_approved(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "architecture.md").write_text(
        "# Architecture\n\nThe Agent Harness orchestrates the review using LangGraph.\n",
        encoding="utf-8",
    )
    sources_path = tmp_path / "document_sources.yaml"
    _write_sources_yaml(sources_path)

    results = retrieve_documentation([Component.AGENT_HARNESS], docs_root, sources_path)

    assert results == []


def test_retrieve_documentation_returns_a_chunk_from_an_approved_path(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "architecture.md").write_text(
        "# Architecture\n\nThe Agent Harness orchestrates the review using LangGraph nodes.\n",
        encoding="utf-8",
    )
    sources_path = tmp_path / "document_sources.yaml"
    _write_sources_yaml(sources_path, architecture=["architecture.md"])

    results = retrieve_documentation([Component.AGENT_HARNESS], docs_root, sources_path)

    assert len(results) == 1
    chunk = results[0]
    assert isinstance(chunk, DocumentChunk)
    assert chunk.source_path.endswith("architecture.md")
    assert chunk.heading == "Architecture"
    assert "Agent Harness" in chunk.snippet


def test_retrieve_documentation_excludes_a_present_but_unapproved_file(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "approved.md").write_text(
        "# Approved\n\nThe Agent Harness is documented here.\n", encoding="utf-8"
    )
    (docs_root / "unapproved.md").write_text(
        "# Unapproved\n\nThe Agent Harness is mentioned here too, but this file is not approved.\n",
        encoding="utf-8",
    )
    sources_path = tmp_path / "document_sources.yaml"
    _write_sources_yaml(sources_path, architecture=["approved.md"])

    results = retrieve_documentation([Component.AGENT_HARNESS], docs_root, sources_path)

    assert len(results) == 1
    assert results[0].source_path.endswith("approved.md")
    assert all("unapproved.md" not in chunk.source_path for chunk in results)


def test_retrieve_documentation_no_matching_keyword_yields_nothing(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "architecture.md").write_text(
        "# Architecture\n\nNothing about any of our components here.\n", encoding="utf-8"
    )
    sources_path = tmp_path / "document_sources.yaml"
    _write_sources_yaml(sources_path, architecture=["architecture.md"])

    results = retrieve_documentation([Component.AGENT_HARNESS], docs_root, sources_path)

    assert results == []


def test_retrieve_documentation_unknown_component_yields_nothing(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "architecture.md").write_text(
        "# Architecture\n\nSome paragraph with lots of words in it.\n", encoding="utf-8"
    )
    sources_path = tmp_path / "document_sources.yaml"
    _write_sources_yaml(sources_path, architecture=["architecture.md"])

    results = retrieve_documentation([Component.UNKNOWN], docs_root, sources_path)

    assert results == []


def test_retrieve_documentation_chunk_without_a_preceding_heading_has_no_heading(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "no_heading.md").write_text(
        "The Agent Harness is mentioned before any heading exists in this file.\n",
        encoding="utf-8",
    )
    sources_path = tmp_path / "document_sources.yaml"
    _write_sources_yaml(sources_path, architecture=["no_heading.md"])

    results = retrieve_documentation([Component.AGENT_HARNESS], docs_root, sources_path)

    assert len(results) == 1
    assert results[0].heading is None


def test_retrieve_documentation_wraps_a_missing_sources_config(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()

    with pytest.raises(DocRetrievalToolError):
        retrieve_documentation([Component.AGENT_HARNESS], docs_root, tmp_path / "missing.yaml")


def test_retrieve_documentation_wraps_an_invalid_sources_config(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    sources_path = tmp_path / "document_sources.yaml"
    # Missing api_contracts/coding_guidelines/service_docs entirely.
    sources_path.write_text("architecture:\n  status: planned\n  approved_paths: []\n", encoding="utf-8")

    with pytest.raises(DocRetrievalToolError):
        retrieve_documentation([Component.AGENT_HARNESS], docs_root, sources_path)


def test_retrieve_documentation_wraps_a_read_failure_on_an_approved_file(tmp_path, monkeypatch):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "broken.md").write_text("# Broken\n\ncontent\n", encoding="utf-8")
    sources_path = tmp_path / "document_sources.yaml"
    _write_sources_yaml(sources_path, architecture=["broken.md"])

    def broken_read_text(self, *args, **kwargs):
        raise OSError("disk exploded")

    monkeypatch.setattr(Path, "read_text", broken_read_text)

    with pytest.raises(DocRetrievalToolError):
        retrieve_documentation([Component.AGENT_HARNESS], docs_root, sources_path)


def test_load_document_sources_reads_the_real_config_file():
    sources = load_document_sources(Path("config/document_sources.yaml"))

    assert sources.architecture.status == "interim"
    assert sources.architecture.approved_paths == ["architecture-baseline.md"]
    assert sources.api_contracts.status == "planned"
    assert sources.api_contracts.approved_paths == []
    assert sources.coding_guidelines.status == "planned"
    assert sources.coding_guidelines.approved_paths == []
    assert sources.service_docs.status == "planned"
    assert sources.service_docs.approved_paths == []
    assert sources.all_approved_paths() == ["architecture-baseline.md"]
