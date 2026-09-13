import pytest

from review_agent.services.model_output_parsing import (
    ModelOutputParsingError,
    extract_json_from_model_output,
)


def test_extract_json_from_model_output_parses_plain_json_with_no_fence():
    result = extract_json_from_model_output('[{"a": 1}]')

    assert result == [{"a": 1}]


def test_extract_json_from_model_output_strips_a_json_language_fence():
    raw_text = '```json\n[{"a": 1}]\n```'

    assert extract_json_from_model_output(raw_text) == [{"a": 1}]


def test_extract_json_from_model_output_strips_a_bare_fence_with_no_language_tag():
    raw_text = '```\n[{"a": 1}]\n```'

    assert extract_json_from_model_output(raw_text) == [{"a": 1}]


def test_extract_json_from_model_output_strips_an_uppercase_json_fence():
    raw_text = '```JSON\n[{"a": 1}]\n```'

    assert extract_json_from_model_output(raw_text) == [{"a": 1}]


def test_extract_json_from_model_output_ignores_whitespace_around_the_fence():
    raw_text = '  \n\n```json\n[{"a": 1}]\n```\n\n  '

    assert extract_json_from_model_output(raw_text) == [{"a": 1}]


def test_extract_json_from_model_output_raises_a_clear_error_for_invalid_content():
    with pytest.raises(ModelOutputParsingError) as exc_info:
        extract_json_from_model_output("this is not json at all")

    assert "not valid JSON" in str(exc_info.value)


def test_extract_json_from_model_output_raises_a_clear_error_when_still_invalid_after_fence_stripping():
    raw_text = "```json\nthis is not json even after stripping the fence\n```"

    with pytest.raises(ModelOutputParsingError) as exc_info:
        extract_json_from_model_output(raw_text)

    assert "not valid JSON" in str(exc_info.value)


def test_extract_json_from_model_output_returns_a_bare_array_unchanged():
    result = extract_json_from_model_output('[{"a": 1}, {"a": 2}]')

    assert result == [{"a": 1}, {"a": 2}]


def test_extract_json_from_model_output_unwraps_a_findings_object():
    raw_text = '{"findings": [{"a": 1}]}'

    assert extract_json_from_model_output(raw_text) == [{"a": 1}]


def test_extract_json_from_model_output_unwraps_a_differently_named_single_key():
    raw_text = '{"results": [{"a": 1}]}'

    assert extract_json_from_model_output(raw_text) == [{"a": 1}]

    assert extract_json_from_model_output('{"issues": [{"a": 1}]}') == [{"a": 1}]


def test_extract_json_from_model_output_unwraps_a_fenced_findings_object():
    raw_text = '```json\n{"findings": [{"a": 1}]}\n```'

    assert extract_json_from_model_output(raw_text) == [{"a": 1}]


def test_extract_json_from_model_output_rejects_a_dict_with_multiple_keys():
    with pytest.raises(ModelOutputParsingError) as exc_info:
        extract_json_from_model_output('{"findings": [{"a": 1}], "summary": "ok"}')

    assert "findings" in str(exc_info.value)
    assert "summary" in str(exc_info.value)


def test_extract_json_from_model_output_rejects_a_single_key_whose_value_is_not_a_list():
    with pytest.raises(ModelOutputParsingError) as exc_info:
        extract_json_from_model_output('{"findings": "not a list"}')

    assert "findings" in str(exc_info.value)
