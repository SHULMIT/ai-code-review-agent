import math
from pathlib import Path

import pytest

from review_agent.config import Settings
from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component, Finding
from review_agent.models.review_input import ReviewInput
from review_agent.nodes.finalize_review import make_finalize_review_node
from review_agent.services import finding_deduplication as deduplication_module
from review_agent.services.finding_validation import validate_findings
from review_agent.tools.component_classifier import (
    DEFAULT_MAPPING_PATH,
    load_component_mapping,
)
from review_agent.tools.embedding_tool import EmbeddingServiceError


def _settings() -> Settings:
    """Build settings without loading a local environment file."""

    return Settings(
        _env_file=None,
        github_app_id=1,
        github_app_private_key_path=Path("secrets/test-private-key.pem"),
        github_webhook_secret="test-webhook-secret",
    )


def _finding(**overrides) -> Finding:
    values = {
        "severity": "WARNING",
        "component": "Agent Harness / LangGraph",
        "review_category": "security",
        "file": "services/mock_agent/search_helper.py",
        "line": 7,
        "problem": "Hardcoded API key used directly in the code.",
        "why_it_matters": "A hardcoded API key is exposed to anyone with source access.",
        "suggested_fix": "Read the API key from a configuration file or environment variable.",
    }
    values.update(overrides)
    return Finding(**values)


def _vector_with_similarity(score: float) -> list[float]:
    """Return a unit vector whose cosine against [1, 0] is score."""

    return [score, math.sqrt(1 - score**2)]


def _embedding_text(finding: Finding) -> str:
    """Use the production representation when stubbing a batched embedding response."""

    return deduplication_module._finding_embedding_text(finding)


def _mock_embeddings(monkeypatch, vectors_by_text: dict[str, list[float]]) -> list[list[str]]:
    """Mock the batched tool boundary; no deduplication test contacts Ollama."""

    calls: list[list[str]] = []

    async def fake_get_embeddings(texts: list[str], settings: Settings) -> list[list[float]]:
        calls.append(texts)
        return [vectors_by_text.get(text, [1.0, 0.0]) for text in texts]

    monkeypatch.setattr(deduplication_module, "get_embeddings", fake_get_embeddings)
    return calls


async def test_known_grounding_paraphrases_merge_at_the_calibrated_similarity(monkeypatch):
    """0.5661 is the lowest observed same-issue score and remains above the threshold."""

    first = _finding(
        line=18,
        problem="Missing RAG grounding.",
        why_it_matters=(
            "The return value from VectorDBClient.query() lacks important metadata, "
            "making it impossible to ground or cite the results downstream."
        ),
        suggested_fix=(
            "Modify query() to return a structured result including source, URL, "
            "and document ID."
        ),
    )
    second = _finding(
        line=18,
        component="Unknown",
        review_category="architecture",
        problem=(
            "The VectorDBClient.query() returns a plain list[str] of document text "
            "without source, URL, or document ID."
        ),
        why_it_matters=(
            "This lack of context prevents downstream answers from being grounded "
            "or cited."
        ),
        suggested_fix=(
            "Modify VectorDBClient.query() to return structured data including "
            "source, URL, or document ID."
        ),
    )
    _mock_embeddings(
        monkeypatch,
        {
            _embedding_text(first): [1.0, 0.0],
            _embedding_text(second): _vector_with_similarity(0.5661),
        },
    )

    result = await deduplication_module.deduplicate_findings([first, second], _settings())

    assert len(result) == 1
    assert result[0].merge_note is not None


async def test_known_different_findings_do_not_merge_below_the_calibrated_threshold(monkeypatch):
    """0.4062 is an observed different-issue score and must stay below the threshold."""

    api_key = _finding(
        line=7,
        problem="Hardcoded API key is committed in the application code.",
    )
    input_validation = _finding(
        line=9,
        problem="Incoming webhook data has no input validation before processing.",
    )
    _mock_embeddings(
        monkeypatch,
        {
            _embedding_text(api_key): [1.0, 0.0],
            _embedding_text(input_validation): _vector_with_similarity(0.4062),
        },
    )

    result = await deduplication_module.deduplicate_findings(
        [api_key, input_validation], _settings()
    )

    assert len(result) == 2


async def test_known_no_tests_findings_merge_across_distant_lines(monkeypatch):
    """The calibrated 0.9563 no-tests pair must merge even at lines 1 and 31."""

    first = _finding(
        line=1,
        problem="No tests for the search_helper module.",
        why_it_matters=(
            "Without search_helper tests, regressions in the search workflow "
            "can reach users."
        ),
        suggested_fix="Add tests for the search_helper functions.",
    )
    second = _finding(
        line=31,
        component="Unknown",
        review_category="architecture",
        problem="Missing tests were added for search_helper.py.",
        why_it_matters=(
            "The search_helper behavior needs test coverage to catch "
            "regressions."
        ),
        suggested_fix="Create search_helper unit tests.",
    )
    _mock_embeddings(
        monkeypatch,
        {
            _embedding_text(first): [1.0, 0.0],
            _embedding_text(second): _vector_with_similarity(0.9563),
            first.problem: [1.0, 0.0],
            second.problem: _vector_with_similarity(0.9563),
        },
    )

    result = await deduplication_module.deduplicate_findings([first, second], _settings())

    assert len(result) == 1
    assert result[0].merge_note is not None


async def test_vector_db_and_error_handling_do_not_merge_across_distant_lines(monkeypatch):
    """A broad full-content match cannot override a low problem-only match."""

    vector_db = _finding(
        line=10,
        problem="The agent queries the Vector DB directly instead of Search Service.",
        why_it_matters="Direct Vector DB access bypasses the governed search boundary.",
        suggested_fix="Use the Search Service API for Vector DB retrieval.",
    )
    error_handling = _finding(
        line=23,
        component="Unknown",
        review_category="architecture",
        problem="No error handling around the external Vector DB query call.",
        why_it_matters="An external query failure can crash the caller.",
        suggested_fix="Wrap the Vector DB query call in try/except handling.",
    )
    _mock_embeddings(
        monkeypatch,
        {
            _embedding_text(vector_db): [1.0, 0.0],
            _embedding_text(error_handling): _vector_with_similarity(0.8),
            vector_db.problem: [1.0, 0.0],
            error_handling.problem: _vector_with_similarity(0.5348),
        },
    )

    result = await deduplication_module.deduplicate_findings(
        [vector_db, error_handling], _settings()
    )

    assert len(result) == 2


@pytest.mark.parametrize("score", [0.5661, 0.8502, 0.9563])
def test_each_calibrated_same_issue_score_remains_above_the_threshold(score):
    first = _finding(
        problem="First parser problem",
        why_it_matters="The parser behavior requires dedicated coverage.",
        suggested_fix="Add parser tests.",
    )
    second = _finding(
        problem="Second parser concern",
        why_it_matters="The parser behavior needs dedicated coverage.",
        suggested_fix="Add parser tests.",
    )
    cache = {
        _embedding_text(first): [1.0, 0.0],
        _embedding_text(second): _vector_with_similarity(score),
    }

    assert deduplication_module._findings_are_similar(first, second, cache)


@pytest.mark.parametrize("score", [0.4062, 0.2973, 0.4321])
def test_each_calibrated_different_issue_score_remains_below_the_threshold(score):
    first = _finding(
        problem="First parser problem",
        why_it_matters="The parser behavior requires dedicated coverage.",
        suggested_fix="Add parser tests.",
    )
    second = _finding(
        problem="Second parser concern",
        why_it_matters="The parser behavior needs dedicated coverage.",
        suggested_fix="Add parser tests.",
    )
    cache = {
        _embedding_text(first): [1.0, 0.0],
        _embedding_text(second): _vector_with_similarity(score),
    }

    assert not deduplication_module._findings_are_similar(first, second, cache)


@pytest.mark.parametrize(
    "error",
    [
        EmbeddingServiceError("Embedding service request failed."),
        EmbeddingServiceError("Embedding service timed out."),
        EmbeddingServiceError("Embedding service returned an unparseable response."),
    ],
    ids=["connection-error", "timeout", "malformed-response"],
)
async def test_embedding_service_failures_fall_back_to_word_overlap(monkeypatch, error):
    """Connection, timeout, and malformed-response failures never escape deduplication."""

    first = _finding(line=7, problem="No tests are provided for the new parser.")
    second = _finding(
        line=9,
        problem="No unit tests exist for the parser changes in this pull request.",
    )

    async def failing_get_embeddings(texts: list[str], settings: Settings) -> list[list[float]]:
        raise error

    monkeypatch.setattr(deduplication_module, "get_embeddings", failing_get_embeddings)

    result = await deduplication_module.deduplicate_findings([first, second], _settings())

    assert len(result) == 1


async def test_embedding_failure_preserves_distant_no_tests_word_overlap_merge(monkeypatch):
    """The legacy fallback deliberately does not impose the distant embedding guards."""

    first = _finding(
        line=1,
        problem="No tests for the search_helper module.",
        why_it_matters="Without search_helper tests, regressions in the search workflow can reach users.",
        suggested_fix="Add tests for the search_helper functions.",
    )
    second = _finding(
        line=31,
        problem="Missing tests were added for search_helper.py.",
        why_it_matters="The search_helper behavior needs test coverage to catch regressions.",
        suggested_fix="Create search_helper unit tests.",
    )

    async def failing_get_embeddings(texts: list[str], settings: Settings) -> list[list[float]]:
        raise EmbeddingServiceError("Embedding service request failed.")

    monkeypatch.setattr(deduplication_module, "get_embeddings", failing_get_embeddings)

    result = await deduplication_module.deduplicate_findings([first, second], _settings())

    assert len(result) == 1


async def test_validated_findings_keep_derived_components_through_deduplication(monkeypatch):
    """Validation, not the model, supplies the component values deduplication reports."""

    review_input = ReviewInput(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        included_diff=(
            "diff --git a/services/mock_agent/search_helper.py b/services/mock_agent/search_helper.py\n"
            "@@ -1 +1 @@\n"
            "+first change\n"
            "@@ -31 +31 @@\n"
            "+second change\n"
        ),
        diff_truncated=False,
        included_files=["services/mock_agent/search_helper.py"],
        omitted_files=[],
        repository_context={"results": []},
        documentation_context=[],
        coverage_notes=[],
    )
    raw_findings = [
        {
            "severity": "WARNING",
            "review_category": "security",
            "file": "services/mock_agent/search_helper.py",
            "line": 1,
            "problem": "No tests cover the search helper.",
            "why_it_matters": "Search helper regressions can reach users.",
            "suggested_fix": "Add search helper tests.",
        },
        {
            "severity": "WARNING",
            "review_category": "architecture",
            "file": "services/mock_agent/search_helper.py",
            "line": 31,
            "problem": "No tests cover the search helper.",
            "why_it_matters": "Search helper regressions can reach users.",
            "suggested_fix": "Add search helper tests.",
        },
    ]
    validated, rejections = validate_findings(
        raw_findings, review_input, load_component_mapping(DEFAULT_MAPPING_PATH).rules
    )
    static_check = Finding(
        severity="WARNING",
        component=Component.UNKNOWN,
        review_category="static_checks",
        file="services/mock_agent/search_helper.py",
        line=2,
        problem="No tests cover the search helper.",
        why_it_matters="Search helper regressions can reach users.",
        suggested_fix="Add search helper tests.",
    )

    _mock_embeddings(monkeypatch, {})
    result = await deduplication_module.deduplicate_findings(
        [*validated, static_check], _settings()
    )

    assert rejections == []
    assert {finding.component for finding in validated} == {Component.AGENT_HARNESS}
    assert len(result) == 1
    assert result[0].merge_note is not None
    assert "component values disagreed: Agent Harness / LangGraph, Unknown" in result[0].merge_note


@pytest.mark.parametrize(
    "vectors",
    [
        [[], []],
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [1.0, 0.0, 0.0]],
        [[float("nan"), 0.0], [1.0, 0.0]],
        [["not-a-number"], [1.0]],
    ],
    ids=["empty", "zero-magnitude", "dimension-mismatch", "non-finite", "non-numeric"],
)
async def test_unusable_vectors_fall_back_without_crashing(monkeypatch, vectors):
    """Malformed vectors are equivalent to an unavailable embedding comparison."""

    first = _finding(line=7, problem="No tests are provided for the new parser.")
    second = _finding(
        line=9,
        problem="No unit tests exist for the parser changes in this pull request.",
    )

    async def fake_get_embeddings(texts: list[str], settings: Settings) -> list[list[float]]:
        return [vectors[index % len(vectors)] for index in range(len(texts))]

    monkeypatch.setattr(deduplication_module, "get_embeddings", fake_get_embeddings)

    result = await deduplication_module.deduplicate_findings([first, second], _settings())

    assert len(result) == 1


async def test_fallback_word_overlap_preserves_non_merge_for_insignificant_words(monkeypatch):
    """The prior heuristic still refuses descriptions with no significant shared words."""

    first = _finding(file="", line=None, problem="It is a it of a to.")
    second = _finding(file="", line=None, problem="A to is it of a.")

    async def failing_get_embeddings(texts: list[str], settings: Settings) -> list[list[float]]:
        raise EmbeddingServiceError("Embedding service request failed.")

    monkeypatch.setattr(deduplication_module, "get_embeddings", failing_get_embeddings)

    result = await deduplication_module.deduplicate_findings([first, second], _settings())

    assert len(result) == 2


async def test_high_embedding_score_without_two_corroborating_terms_does_not_merge(monkeypatch):
    """Broad generic code-quality prose must not discard a distinct finding."""

    documentation = _finding(
        line=1,
        problem="Lack of documentation for the file and its functions.",
        why_it_matters=(
            "Documentation is essential for understanding and maintaining the "
            "code, especially for new contributors."
        ),
        suggested_fix="Add docstrings that describe each function's purpose.",
    )
    type_hints = _finding(
        line=1,
        problem="Lack of type hints for function parameters and return values.",
        why_it_matters=(
            "Type hints improve readability and help catch incorrect values "
            "before runtime."
        ),
        suggested_fix="Add type hints to parameters and return values.",
    )
    _mock_embeddings(
        monkeypatch,
        {
            _embedding_text(documentation): [1.0, 0.0],
            _embedding_text(type_hints): _vector_with_similarity(0.6040),
        },
    )

    result = await deduplication_module.deduplicate_findings(
        [documentation, type_hints], _settings()
    )

    assert len(result) == 2


async def test_high_embedding_score_with_only_boilerplate_overlap_does_not_merge(monkeypatch):
    """A terse no-tests concern must not merge into an unrelated RAG concern."""

    grounding = _finding(
        line=28,
        problem="No RAG grounding.",
        why_it_matters=(
            "The external call lacks necessary context such as a source URL or "
            "document ID for citations."
        ),
        suggested_fix="Modify VectorDBClient.query() to return source information.",
    )
    no_tests = _finding(
        line=31,
        problem="No tests.",
        why_it_matters=(
            "Tests are necessary to ensure expected behavior and prevent "
            "regressions."
        ),
        suggested_fix="Modify the test suite to provide comprehensive coverage.",
    )
    _mock_embeddings(
        monkeypatch,
        {
            _embedding_text(grounding): [1.0, 0.0],
            _embedding_text(no_tests): _vector_with_similarity(0.8),
        },
    )

    result = await deduplication_module.deduplicate_findings(
        [grounding, no_tests], _settings()
    )

    assert len(result) == 2


async def test_deduplication_batches_each_unique_problem_once_per_run(monkeypatch):
    """Pairwise clustering reuses one batched embedding lookup instead of repeated calls."""

    findings = [
        _finding(line=7, problem="One duplicate concern."),
        _finding(line=8, problem="A paraphrase of the first concern."),
        _finding(line=9, problem="Another paraphrase of the same concern."),
    ]
    calls = _mock_embeddings(
        monkeypatch,
        {
            _embedding_text(findings[0]): [1.0, 0.0],
            _embedding_text(findings[1]): _vector_with_similarity(0.8502),
            _embedding_text(findings[2]): _vector_with_similarity(0.9563),
        },
    )

    result = await deduplication_module.deduplicate_findings(findings, _settings())

    assert len(result) == 1
    assert calls == [[
        *[_embedding_text(finding) for finding in findings],
        *[finding.problem for finding in findings],
    ]]


async def test_deduplication_keeps_highest_severity_and_merge_metadata(monkeypatch):
    security_finding = _finding(severity="BLOCKING", component="Agent Harness / LangGraph")
    architecture_finding = _finding(
        severity="WARNING", component="Unknown", review_category="architecture"
    )
    _mock_embeddings(
        monkeypatch,
        {
            _embedding_text(security_finding): [1.0, 0.0],
            _embedding_text(architecture_finding): [1.0, 0.0],
        },
    )

    result = await deduplication_module.deduplicate_findings(
        [security_finding, architecture_finding], _settings()
    )

    assert len(result) == 1
    assert result[0].severity.value == "BLOCKING"
    assert result[0].merge_note is not None
    assert "Agent Harness / LangGraph" in result[0].merge_note
    assert "Unknown" in result[0].merge_note
    assert "security" in result[0].merge_note
    assert "architecture" in result[0].merge_note


async def test_deduplication_returns_an_empty_list_without_embedding_calls(monkeypatch):
    called = False

    async def fake_get_embeddings(texts: list[str], settings: Settings) -> list[list[float]]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(deduplication_module, "get_embeddings", fake_get_embeddings)

    result = await deduplication_module.deduplicate_findings([], _settings())

    assert result == []
    assert called is False


async def test_cache_symptoms_with_the_same_explicit_identifier_merge_into_rag_search(monkeypatch):
    """Different cache symptoms are one defect when both cite the same code anchor."""

    unbounded_cache = _finding(
        severity="BLOCKING",
        review_category="performance",
        line=10,
        problem="The _QUERY_CACHE grows without any size bound.",
        why_it_matters="Long-running workers can exhaust memory as unique queries accumulate.",
        suggested_fix="Bound _QUERY_CACHE and expire entries after a TTL.",
    )
    stale_cache = _finding(
        severity="WARNING",
        review_category="code quality",
        line=80,
        problem="The _QUERY_CACHE is never invalidated after a document update.",
        why_it_matters="Search can return stale results after the source changes.",
        suggested_fix="Invalidate _QUERY_CACHE entries whenever indexed documents change.",
    )
    _mock_embeddings(
        monkeypatch,
        {
            _embedding_text(unbounded_cache): [1.0, 0.0],
            _embedding_text(stale_cache): _vector_with_similarity(0.8),
            unbounded_cache.problem: [1.0, 0.0],
            stale_cache.problem: _vector_with_similarity(0.3),
        },
    )

    result = await deduplication_module.deduplicate_findings(
        [unbounded_cache, stale_cache], _settings()
    )

    assert len(result) == 1
    assert result[0].severity.value == "BLOCKING"
    assert result[0].review_category == "rag_search"
    assert result[0].suggested_fix == unbounded_cache.suggested_fix
    assert result[0].merge_note is not None

    review_result = make_finalize_review_node()(
        ReviewState(repo="example-org/example-repo", pr_number=1, findings=result)
    )["review_result"]
    assert review_result.production_readiness_score == 70
    assert review_result.code_quality_score == 100
    assert review_result.recommendation == "REQUEST_CHANGES"
