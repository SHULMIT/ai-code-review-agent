import json

import pytest

from review_agent.config import Settings
from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component, Finding, Severity
from review_agent.models.model_output import ModelReviewOutput
from review_agent.models.review_input import ReviewInput
from review_agent.models.reviewer_diagnostic import ReviewerDiagnostic
from review_agent.nodes import all_zero_verification as all_zero_verification_module
from review_agent.nodes.all_zero_verification import make_all_zero_verification_node
from review_agent.tools.model_service_tool import ModelServiceError

_NON_TRIVIAL_DIFF = (
    "diff --git a/a.py b/a.py\n"
    "index 0000000..1111111 100644\n"
    "--- a/a.py\n"
    "+++ b/a.py\n"
    "@@ -1,3 +1,6 @@\n"
    "+import subprocess\n"
    "+def run(cmd):\n"
    "+    subprocess.run(cmd, shell=True)\n"
)
_TRIVIAL_DIFF = (
    "diff --git a/a.py b/a.py\n"
    "index 0000000..1111111 100644\n"
    "--- a/a.py\n"
    "+++ b/a.py\n"
    "@@ -1,1 +1,1 @@\n"
    "+x = 1\n"
)


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
        "included_diff": _NON_TRIVIAL_DIFF,
        "diff_truncated": False,
        "included_files": ["a.py"],
        "omitted_files": [],
        "repository_context": {"results": []},
        "documentation_context": [],
        "coverage_notes": [],
    }
    values.update(overrides)
    return ReviewInput(**values)


def _completed_diagnostic(category: str) -> ReviewerDiagnostic:
    return ReviewerDiagnostic(category=category, status="completed", duration_seconds=0.1)


def _six_completed_diagnostics() -> list[ReviewerDiagnostic]:
    return [
        _completed_diagnostic(category)
        for category in (
            "security", "architecture", "code quality",
            "rag_search", "performance", "agent_harness",
        )
    ]


def _state(
    review_input: ReviewInput | None,
    *,
    findings: list[Finding] | None = None,
    reviewer_diagnostics: list[ReviewerDiagnostic] | None = None,
) -> ReviewState:
    return ReviewState(
        repo="example-org/example-repo",
        pr_number=1,
        head_sha="abc123",
        review_input=review_input,
        findings=findings or [],
        reviewer_diagnostics=reviewer_diagnostics if reviewer_diagnostics is not None else _six_completed_diagnostics(),
    )


def _valid_raw_finding(**overrides) -> dict:
    values = {
        "severity": "WARNING",
        "file": "a.py",
        "line": 3,
        "problem": "subprocess.run is called with shell=True and an externally supplied command.",
        "why_it_matters": "Untrusted input reaching a shell command allows command injection.",
        "suggested_fix": "Pass the command as a list of args and drop shell=True.",
    }
    values.update(overrides)
    return values


async def test_does_not_trigger_when_a_reviewer_did_not_complete(monkeypatch):
    called = {"count": 0}

    async def fake_call_model(prompt, settings, **kwargs):
        called["count"] += 1
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(all_zero_verification_module, "call_model", fake_call_model)

    diagnostics = _six_completed_diagnostics()
    diagnostics[0] = ReviewerDiagnostic(category="security", status="failed", duration_seconds=0.1)

    node = make_all_zero_verification_node(_model_settings())
    result = await node(_state(_review_input(), reviewer_diagnostics=diagnostics))

    assert result == {}
    assert called["count"] == 0


async def test_does_not_trigger_when_findings_already_exist(monkeypatch):
    called = {"count": 0}

    async def fake_call_model(prompt, settings, **kwargs):
        called["count"] += 1
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(all_zero_verification_module, "call_model", fake_call_model)

    existing_finding = Finding(
        severity=Severity.WARNING, component=Component.UNKNOWN, review_category="security",
        file="a.py", line=1, problem="p", why_it_matters="w", suggested_fix="f",
    )

    node = make_all_zero_verification_node(_model_settings())
    result = await node(_state(_review_input(), findings=[existing_finding]))

    assert result == {}
    assert called["count"] == 0


async def test_does_not_trigger_on_a_trivial_diff(monkeypatch):
    called = {"count": 0}

    async def fake_call_model(prompt, settings, **kwargs):
        called["count"] += 1
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(all_zero_verification_module, "call_model", fake_call_model)

    node = make_all_zero_verification_node(_model_settings())
    result = await node(_state(_review_input(included_diff=_TRIVIAL_DIFF)))

    assert result == {}
    assert called["count"] == 0


async def test_triggers_exactly_once_on_all_zero_non_trivial_diff(monkeypatch):
    called = {"count": 0}

    async def fake_call_model(prompt, settings, **kwargs):
        called["count"] += 1
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(all_zero_verification_module, "call_model", fake_call_model)

    node = make_all_zero_verification_node(_model_settings())
    result = await node(_state(_review_input()))

    assert called["count"] == 1
    assert result["findings"] == []
    assert result["reviewer_diagnostics"][0].category == "all_zero_verification"
    assert result["reviewer_diagnostics"][0].status == "completed"
    assert "failed_categories" not in result


async def test_non_zero_findings_do_not_trigger_verification(monkeypatch):
    """Covers the 'non-zero findings do not trigger it' requirement explicitly,
    distinct from the already-exist check above which asserts the same via call count."""
    called = {"count": 0}

    async def fake_call_model(prompt, settings, **kwargs):
        called["count"] += 1
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(all_zero_verification_module, "call_model", fake_call_model)

    one_finding = [
        Finding(
            severity=Severity.SUGGESTION, component=Component.UNKNOWN, review_category="code quality",
            file="a.py", line=1, problem="p", why_it_matters="w", suggested_fix="f",
        )
    ]
    node = make_all_zero_verification_node(_model_settings())
    result = await node(_state(_review_input(), findings=one_finding))

    assert called["count"] == 0
    assert result == {}


async def test_verification_findings_flow_through_validation_and_dedup(monkeypatch):
    dedup_calls: list[list[Finding]] = []

    async def fake_call_model(prompt, settings, **kwargs):
        return ModelReviewOutput(raw_text=json.dumps([_valid_raw_finding()]))

    async def fake_deduplicate_findings(findings, settings, **kwargs):
        dedup_calls.append(findings)
        return findings  # nothing to merge for a single finding

    monkeypatch.setattr(all_zero_verification_module, "call_model", fake_call_model)
    monkeypatch.setattr(
        all_zero_verification_module, "deduplicate_findings", fake_deduplicate_findings
    )

    node = make_all_zero_verification_node(_model_settings())
    result = await node(_state(_review_input()))

    assert len(dedup_calls) == 1
    findings = result["findings"]
    assert len(findings) == 1
    assert findings[0].file == "a.py"
    # review_category is inferred from the finding's own text (never model-supplied),
    # and "shell"/"subprocess"/"injection" wording here should land on security.
    assert findings[0].review_category == "security"
    diagnostic = result["reviewer_diagnostics"][0]
    assert diagnostic.status == "completed"
    assert diagnostic.findings_count == 1
    assert diagnostic.rejected_count == 0


async def test_verification_returning_empty_allows_clean_approval(monkeypatch):
    async def fake_call_model(prompt, settings, **kwargs):
        return ModelReviewOutput(raw_text="[]")

    monkeypatch.setattr(all_zero_verification_module, "call_model", fake_call_model)

    node = make_all_zero_verification_node(_model_settings())
    result = await node(_state(_review_input()))

    assert result["findings"] == []
    assert "failed_categories" not in result
    diagnostic = result["reviewer_diagnostics"][0]
    assert diagnostic.status == "completed"
    assert diagnostic.findings_count == 0


async def test_a_model_service_failure_never_produces_failed_categories(monkeypatch):
    """A transient failure in this optional extra check must never downgrade an
    otherwise-clean, all-zero result to INCOMPLETE (see finalize_review.py:
    any non-empty failed_categories flips the recommendation away from APPROVE)."""

    async def failing_call_model(prompt, settings, **kwargs):
        raise ModelServiceError("Model Service request failed.")

    monkeypatch.setattr(all_zero_verification_module, "call_model", failing_call_model)

    node = make_all_zero_verification_node(_model_settings())
    result = await node(_state(_review_input()))

    assert "failed_categories" not in result
    assert result["reviewer_diagnostics"][0].status == "failed"
