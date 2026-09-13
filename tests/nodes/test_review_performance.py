import json
from pathlib import Path

import pytest

from review_agent.config import Settings
from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component
from review_agent.models.model_output import ModelReviewOutput
from review_agent.models.review_input import ReviewInput
from review_agent.nodes import review_performance as review_performance_module
from review_agent.nodes.review_performance import (
    DEFAULT_PERFORMANCE_PROMPT_PATH,
    PromptTemplateError,
    load_prompt_template,
    make_performance_reviewer_node,
)
from review_agent.tools.model_service_tool import ModelServiceError


def _model_settings() -> Settings:
    return Settings(
        _env_file=None,
        github_app_id=1,
        github_app_private_key_path="secrets/test-private-key.pem",
        github_webhook_secret="fake-webhook-secret",
    )


def _review_input(**overrides) -> ReviewInput:
    values = {
        "repo": "example-org/example-repo",
        "pr_number": 1,
        "head_sha": "abc123",
        "included_diff": (
            "diff --git a/a.py b/a.py\n"
            "index 0000000..1111111 100644\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,3 +1,5 @@\n"
            "+def query(self, text) -> list[str]:\n"
            "+    return [hit.text for hit in self._index.search(text)]\n"
        ),
        "diff_truncated": False,
        "included_files": ["a.py"],
        "omitted_files": [],
        "repository_context": {"results": []},
        "documentation_context": [],
        "coverage_notes": [],
    }
    values.update(overrides)
    return ReviewInput(**values)


def _valid_raw_finding(**overrides) -> dict:
    values = {
        "severity": "BLOCKING",
        "file": "a.py",
        "line": 2,
        "problem": "load_document reads the whole file into memory with no size limit.",
        "why_it_matters": "A multi-gigabyte document exhausts the worker rather than failing on its own.",
        "suggested_fix": "Stream the file in fixed-size chunks, or enforce a maximum size before reading."
    }
    values.update(overrides)
    return values


def _state(review_input: ReviewInput | None) -> ReviewState:
    return ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        components=[Component.INDEXING_PIPELINE],
        review_input=review_input,
    )


async def test_performance_reviewer_node_returns_validated_findings_on_success(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text=json.dumps([_valid_raw_finding()]))

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert len(result["findings"]) == 1
    assert result["findings"][0].file == "a.py"
    assert "errors" not in result
    diagnostic = result["reviewer_diagnostics"][0]
    assert diagnostic.category == "performance"
    assert diagnostic.status == "completed"
    assert diagnostic.findings_count == 1
    assert diagnostic.rejected_count == 0
    assert diagnostic.error is None


async def test_performance_reviewer_node_derives_component_deterministically(monkeypatch):
    """component is never read from the model - the fake output above never includes it,
    and "a.py" matches no rule in the real component_mapping.yaml, so it derives to Unknown."""

    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text=json.dumps([_valid_raw_finding()]))

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["findings"][0].component == Component.UNKNOWN


async def test_performance_reviewer_node_stamps_its_own_review_category_onto_findings(monkeypatch):
    """review_category is stamped by the node in code - the fake model output never includes it."""

    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text=json.dumps([_valid_raw_finding()]))

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["findings"][0].review_category == "performance"


def test_review_category_matches_the_string_finalize_review_scores_against():
    """finalize_review.py keys the production-readiness score off this exact string."""
    from review_agent.nodes.finalize_review import _PRODUCTION_READINESS_CATEGORIES

    assert review_performance_module.REVIEW_CATEGORY in _PRODUCTION_READINESS_CATEGORIES


async def test_performance_reviewer_node_parses_findings_wrapped_in_a_markdown_code_fence(
    monkeypatch,
):
    async def fake_call_model(prompt, settings):
        fenced = "```json\n" + json.dumps([_valid_raw_finding()]) + "\n```"
        return ModelReviewOutput(raw_text=fenced)

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert len(result["findings"]) == 1
    assert result["findings"][0].file == "a.py"
    assert "errors" not in result


async def test_performance_reviewer_node_unwraps_a_findings_object_response(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text=json.dumps({"findings": [_valid_raw_finding()]}))

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert len(result["findings"]) == 1
    assert result["findings"][0].file == "a.py"
    assert "errors" not in result


async def test_performance_reviewer_node_records_error_for_an_unrecognized_object_shape(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(
            raw_text=json.dumps({"findings": [_valid_raw_finding()], "summary": "ok"})
        )

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert "findings" not in result
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("performance reviewer: model output is a JSON object")


async def test_performance_reviewer_node_records_error_for_invalid_json(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text="not valid json")

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert "findings" not in result
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("performance reviewer: model output is not valid JSON")


async def test_performance_reviewer_node_records_error_when_model_returns_a_bare_scalar(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text="42")

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert "findings" not in result
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("performance reviewer: raw_findings must be a list")


async def test_performance_reviewer_node_records_rejection_reason_for_an_invalid_finding(
    monkeypatch,
):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(
            raw_text=json.dumps([_valid_raw_finding(file="never/touched.py", line=None)])
        )

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["findings"] == []
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("performance reviewer: finding[0]:")
    assert "never/touched.py" in result["errors"][0]


async def test_performance_reviewer_node_records_error_when_review_input_is_none(monkeypatch):
    async def fake_call_model(prompt, settings):
        raise AssertionError("call_model should never be called without a review_input")

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(None))

    assert result["errors"] == ["performance reviewer: no review_input available (upstream failure)"]
    assert result["failed_categories"] == ["performance"]
    assert len(result["reviewer_diagnostics"]) == 1
    diagnostic = result["reviewer_diagnostics"][0]
    assert diagnostic.category == "performance"
    assert diagnostic.status == "failed"


async def test_performance_reviewer_node_records_error_for_an_empty_diff(monkeypatch):
    async def fake_call_model(prompt, settings):
        raise AssertionError("call_model should never be called with an empty diff")

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input(included_diff="", included_files=[])))

    assert result["errors"] == ["performance reviewer: included_diff is empty, nothing to review"]
    assert result["failed_categories"] == ["performance"]
    assert len(result["reviewer_diagnostics"]) == 1
    diagnostic = result["reviewer_diagnostics"][0]
    assert diagnostic.category == "performance"
    assert diagnostic.status == "failed"


async def test_performance_reviewer_node_records_model_service_error(monkeypatch):
    async def fake_call_model(prompt, settings):
        raise ModelServiceError("Model Service request failed.")

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["errors"] == ["performance reviewer: Model Service request failed."]
    assert result["failed_categories"] == ["performance"]
    assert len(result["reviewer_diagnostics"]) == 1
    diagnostic = result["reviewer_diagnostics"][0]
    assert diagnostic.category == "performance"
    assert diagnostic.status == "failed"


def test_load_prompt_template_reads_the_default_rag_search_prompt_file():
    instructions = load_prompt_template(DEFAULT_PERFORMANCE_PROMPT_PATH)

    assert instructions.strip() != ""
    assert "bounded" in instructions.lower()


_MISSING_PROMPT_PATH = Path(__file__).resolve().parent / "does-not-exist-performance-prompt.md"


def test_load_prompt_template_raises_a_clear_error_for_a_missing_file():
    with pytest.raises(PromptTemplateError) as exc_info:
        load_prompt_template(_MISSING_PROMPT_PATH)

    assert str(_MISSING_PROMPT_PATH) in str(exc_info.value)


def test_make_performance_reviewer_node_raises_a_clear_error_for_a_missing_prompt_file():
    with pytest.raises(PromptTemplateError):
        make_performance_reviewer_node(_model_settings(), prompt_path=_MISSING_PROMPT_PATH)


async def test_a_parse_failure_also_marks_the_category_as_failed(monkeypatch):
    """An unreadable model response means this category produced no result, like a dead call."""

    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text="not valid json")

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["failed_categories"] == ["performance"]


async def test_a_rejected_finding_does_not_mark_the_category_as_failed(monkeypatch):
    """This reviewer ran and answered - it just answered partly wrong, which is not a failure."""

    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(
            raw_text=json.dumps([_valid_raw_finding(file="never/touched.py", line=None)])
        )

    monkeypatch.setattr(review_performance_module, "call_model", fake_call_model)

    node = make_performance_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["errors"]
    assert "failed_categories" not in result
