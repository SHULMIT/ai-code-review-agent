"""Every reviewer's category must reach a score, and its prompt must exist.

A reviewer whose REVIEW_CATEGORY is missing from finalize_review.py's
category sets still runs, still produces findings, and still passes all of
its own tests - its findings just never affect any score. Nothing raises,
nothing logs, and the reviewer looks like it is working. This file exists to
turn that silent gap into a failing test.

Reviewer modules are discovered by globbing nodes/review_*.py rather than
being listed, so a newly added reviewer is checked the moment it lands
instead of when someone remembers to extend this file. That is the whole
point: the failure mode being guarded against is forgetting a step.
"""

import importlib
from pathlib import Path

import pytest

from review_agent.nodes.finalize_review import SCORED_CATEGORIES

NODES_DIR = Path(__file__).resolve().parents[2] / "src" / "review_agent" / "nodes"
REVIEWER_MODULE_NAMES = sorted(path.stem for path in NODES_DIR.glob("review_*.py"))


def test_reviewer_modules_are_discovered():
    """Guard the glob - every other test here passes vacuously on an empty list."""
    assert REVIEWER_MODULE_NAMES, f"no review_*.py modules found in {NODES_DIR}"


@pytest.fixture(params=REVIEWER_MODULE_NAMES, ids=lambda name: name)
def reviewer_module(request):
    return importlib.import_module(f"review_agent.nodes.{request.param}")


def test_reviewer_declares_a_review_category(reviewer_module):
    category = getattr(reviewer_module, "REVIEW_CATEGORY", None)
    assert isinstance(category, str) and category, (
        f"{reviewer_module.__name__} must define a non-empty REVIEW_CATEGORY"
    )


def test_reviewer_category_reaches_a_score(reviewer_module):
    """An unmapped category produces findings that silently affect no score at all."""
    category = reviewer_module.REVIEW_CATEGORY
    assert category in SCORED_CATEGORIES, (
        f"{reviewer_module.__name__}'s category {category!r} is in none of "
        f"finalize_review.py's category sets, so its findings would never be "
        f"scored. Add it to the set it belongs to. Currently scored: "
        f"{sorted(SCORED_CATEGORIES)}"
    )


def test_reviewer_categories_are_unique():
    """Two reviewers sharing a category would have their scores silently combined."""
    categories = [
        importlib.import_module(f"review_agent.nodes.{name}").REVIEW_CATEGORY
        for name in REVIEWER_MODULE_NAMES
    ]
    duplicates = {value for value in categories if categories.count(value) > 1}
    assert not duplicates, f"review categories must be unique, found duplicates: {duplicates}"


def test_reviewer_prompt_file_exists(reviewer_module):
    """The factory reads its prompt at graph-build time, so a missing file breaks startup."""
    prompt_paths = [
        value
        for name, value in vars(reviewer_module).items()
        if name.startswith("DEFAULT_") and name.endswith("_PROMPT_PATH")
    ]
    assert len(prompt_paths) == 1, (
        f"{reviewer_module.__name__} must define exactly one DEFAULT_*_PROMPT_PATH"
    )
    assert prompt_paths[0].is_file(), f"missing prompt file: {prompt_paths[0]}"
