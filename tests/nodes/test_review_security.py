import json
from pathlib import Path

import pytest

from review_agent.config import Settings
from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component
from review_agent.models.model_output import ModelReviewOutput
from review_agent.models.review_input import ReviewInput
from review_agent.nodes import review_security as review_security_module
from review_agent.nodes.review_security import (
    DEFAULT_SECURITY_PROMPT_PATH,
    PromptTemplateError,
    load_prompt_template,
    make_security_reviewer_node,
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
            "+import os\n"
            "+os.system(cmd)\n"
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
        "problem": "Command injection via os.system.",
        "why_it_matters": "Untrusted input reaches a shell command.",
        "suggested_fix": "Use subprocess with a list of args, never shell=True.",
    }
    values.update(overrides)
    return values


def _state(review_input: ReviewInput | None) -> ReviewState:
    return ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        components=[Component.AGENT_HARNESS],
        review_input=review_input,
    )


async def test_security_reviewer_node_returns_validated_findings_on_success(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text=json.dumps([_valid_raw_finding()]))

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert len(result["findings"]) == 1
    assert result["findings"][0].file == "a.py"
    assert "errors" not in result


async def test_security_reviewer_receives_logged_plaintext_credentials_in_the_grounded_diff(
    monkeypatch,
):
    captured: dict[str, str] = {}

    async def fake_call_model(prompt, settings):
        captured["prompt"] = prompt
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)
    review_input = _review_input(
        included_diff=(
            "diff --git a/a.py b/a.py\n"
            "@@ -1,1 +1,2 @@\n"
            '+logger.info("using API key %s", "plaintext-secret")\n'
        ),
        included_files=["a.py"],
    )

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(review_input))

    assert result["findings"] == []
    assert 'logger.info("using API key %s", "plaintext-secret")' in captured["prompt"]
    assert "Plaintext credentials or API keys interpolated into a log call" in captured["prompt"]


async def test_security_reviewer_node_records_a_completed_diagnostic_on_success(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text=json.dumps([_valid_raw_finding()]))

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert len(result["reviewer_diagnostics"]) == 1
    diagnostic = result["reviewer_diagnostics"][0]
    assert diagnostic.category == "security"
    assert diagnostic.status == "completed"
    assert diagnostic.raw_findings_count == 1
    assert diagnostic.findings_count == 1
    assert diagnostic.rejected_count == 0
    assert diagnostic.rejection_reasons == []
    assert diagnostic.duration_seconds >= 0
    assert diagnostic.error is None


async def test_security_reviewer_node_logs_temporary_diff_evidence_when_rag_client_present(
    monkeypatch,
):
    """Temporary PR #9 investigation diagnostics: which files reached the prompt, total
    diff size, and rag_client.py's own diff size when it's one of the changed files."""
    captured: dict = {}

    def fake_log_event(logger, level, event, **fields):
        if event == "reviewer_model_response_diagnostic":
            captured.update(fields)

    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)
    monkeypatch.setattr(review_security_module, "log_event", fake_log_event)

    base_hunk = (
        "diff --git a/a.py b/a.py\n"
        "index 0000000..1111111 100644\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,3 +1,5 @@\n"
        "+import os\n"
        "+os.system(cmd)\n"
    )
    rag_client_hunk = (
        "diff --git a/datanet_review_recording/rag_client.py b/datanet_review_recording/rag_client.py\n"
        "index 0000000..1111111 100644\n"
        "--- a/datanet_review_recording/rag_client.py\n"
        "+++ b/datanet_review_recording/rag_client.py\n"
        "@@ -1,1 +1,2 @@\n"
        "+import requests\n"
    )
    review_input = _review_input(
        included_diff=base_hunk + rag_client_hunk,
        included_files=["a.py", "datanet_review_recording/rag_client.py"],
    )

    node = make_security_reviewer_node(_model_settings())
    await node(_state(review_input))

    assert captured["reviewer_category"] == "security"
    assert captured["included_changed_files"] == ["a.py", "datanet_review_recording/rag_client.py"]
    assert captured["rag_client_in_prompt"] is True
    assert captured["diff_total_chars"] == len(review_input.included_diff)
    # split_diff_by_file() rejoins each hunk's lines with "\n" (no trailing
    # newline), matching split_diff_by_file's own behavior rather than the
    # raw literal's trailing "\n".
    assert captured["rag_client_diff_chars"] == len("\n".join(rag_client_hunk.splitlines()))
    # Never the full prompt or raw repository source, only sizes/flags/text the model itself wrote.
    assert "prompt" not in captured


async def test_security_reviewer_node_logs_rag_client_absent_when_not_a_changed_file(
    monkeypatch,
):
    captured: dict = {}

    def fake_log_event(logger, level, event, **fields):
        if event == "reviewer_model_response_diagnostic":
            captured.update(fields)

    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)
    monkeypatch.setattr(review_security_module, "log_event", fake_log_event)

    node = make_security_reviewer_node(_model_settings())
    await node(_state(_review_input()))

    assert captured["rag_client_in_prompt"] is False
    assert captured["rag_client_diff_chars"] is None


async def test_security_reviewer_node_records_raw_count_and_rejection_reasons(monkeypatch):
    """A finding for a file outside the PR is rejected, but still counted in raw_findings_count
    and its reason kept on the diagnostic - not just folded into a generic error string."""

    async def fake_call_model(prompt, settings):
        raw_finding = _valid_raw_finding()
        raw_finding["file"] = "not_in_the_diff.py"
        return ModelReviewOutput(raw_text=json.dumps([raw_finding]))

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    diagnostic = result["reviewer_diagnostics"][0]
    assert diagnostic.raw_findings_count == 1
    assert diagnostic.findings_count == 0
    assert diagnostic.rejected_count == 1
    assert len(diagnostic.rejection_reasons) == 1
    assert "does not appear among this PR's changed files" in diagnostic.rejection_reasons[0]


async def test_security_reviewer_node_derives_component_deterministically(monkeypatch):
    """component is never read from the model - the fake output above never includes it,
    and "a.py" matches no rule in the real component_mapping.yaml, so it derives to Unknown."""

    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text=json.dumps([_valid_raw_finding()]))

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["findings"][0].component == Component.UNKNOWN


async def test_security_reviewer_node_stamps_its_own_review_category_onto_findings(monkeypatch):
    """review_category is stamped by the node in code - the fake model output never includes it."""

    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text=json.dumps([_valid_raw_finding()]))

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["findings"][0].review_category == "security"


async def test_security_reviewer_node_parses_findings_wrapped_in_a_markdown_code_fence(monkeypatch):
    async def fake_call_model(prompt, settings):
        fenced = "```json\n" + json.dumps([_valid_raw_finding()]) + "\n```"
        return ModelReviewOutput(raw_text=fenced)

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert len(result["findings"]) == 1
    assert result["findings"][0].file == "a.py"
    assert "errors" not in result


async def test_security_reviewer_node_records_error_for_invalid_json(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text="not valid json")

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert "findings" not in result
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("security reviewer: model output is not valid JSON")


async def test_security_reviewer_node_records_rejection_reason_for_an_invalid_finding(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(
            raw_text=json.dumps([_valid_raw_finding(file="never/touched.py", line=None)])
        )

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["findings"] == []
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("security reviewer: finding[0]:")
    assert "never/touched.py" in result["errors"][0]


async def test_security_reviewer_node_records_error_when_review_input_is_none(monkeypatch):
    async def fake_call_model(prompt, settings):
        raise AssertionError("call_model should never be called without a review_input")

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(None))

    assert result["errors"] == ["security reviewer: no review_input available (upstream failure)"]
    assert result["failed_categories"] == ["security"]
    assert len(result["reviewer_diagnostics"]) == 1
    diagnostic = result["reviewer_diagnostics"][0]
    assert diagnostic.category == "security"
    assert diagnostic.status == "failed"
    assert diagnostic.error == "no review_input available (upstream failure)"


async def test_security_reviewer_node_records_error_for_an_empty_diff(monkeypatch):
    async def fake_call_model(prompt, settings):
        raise AssertionError("call_model should never be called with an empty diff")

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input(included_diff="", included_files=[])))

    assert result["errors"] == ["security reviewer: included_diff is empty, nothing to review"]
    assert result["failed_categories"] == ["security"]
    assert result["reviewer_diagnostics"][0].status == "failed"


async def test_security_reviewer_node_records_model_service_error(monkeypatch):
    async def fake_call_model(prompt, settings):
        raise ModelServiceError("Model Service request failed.")

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["errors"] == ["security reviewer: Model Service request failed."]
    assert result["failed_categories"] == ["security"]
    assert result["reviewer_diagnostics"][0].error == "Model Service request failed."


async def test_security_reviewer_node_unwraps_a_findings_object_response(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text=json.dumps({"findings": [_valid_raw_finding()]}))

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert len(result["findings"]) == 1
    assert result["findings"][0].file == "a.py"
    assert "errors" not in result


async def test_security_reviewer_node_records_error_for_an_unrecognized_object_shape(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(
            raw_text=json.dumps({"findings": [_valid_raw_finding()], "summary": "ok"})
        )

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert "findings" not in result
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("security reviewer: model output is a JSON object")


async def test_security_reviewer_node_records_error_when_model_returns_a_bare_scalar(monkeypatch):
    async def fake_call_model(prompt, settings):
        return ModelReviewOutput(raw_text="42")

    monkeypatch.setattr(review_security_module, "call_model", fake_call_model)

    node = make_security_reviewer_node(_model_settings())
    result = await node(_state(_review_input()))

    assert "findings" not in result
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("security reviewer: raw_findings must be a list")


def test_load_prompt_template_reads_the_default_security_prompt_file():
    instructions = load_prompt_template(DEFAULT_SECURITY_PROMPT_PATH)

    assert instructions.strip() != ""
    assert "input validation" in instructions.lower()


_MISSING_PROMPT_PATH = Path(__file__).resolve().parent / "does-not-exist-security-prompt.md"


def test_load_prompt_template_raises_a_clear_error_for_a_missing_file():
    with pytest.raises(PromptTemplateError) as exc_info:
        load_prompt_template(_MISSING_PROMPT_PATH)

    assert str(_MISSING_PROMPT_PATH) in str(exc_info.value)


def test_make_security_reviewer_node_raises_a_clear_error_for_a_missing_prompt_file():
    missing_path = _MISSING_PROMPT_PATH

    with pytest.raises(PromptTemplateError):
        make_security_reviewer_node(_model_settings(), prompt_path=missing_path)
