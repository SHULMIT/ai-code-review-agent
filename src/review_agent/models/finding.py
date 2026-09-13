from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Severity(str, Enum):
    BLOCKING = "BLOCKING"
    WARNING = "WARNING"
    SUGGESTION = "SUGGESTION"


class Component(str, Enum):
    AGENT_HARNESS = "Agent Harness / LangGraph"
    SEARCH_SERVICE = "Search Service"
    INDEXING_PIPELINE = "Indexing Pipeline"
    MODEL_SERVICE = "Model Service"
    INFRASTRUCTURE = "Infrastructure / Docker"
    FRONTEND = "Frontend"
    UNKNOWN = "Unknown"


class Finding(BaseModel):
    """A single review comment, matching the Output format from the task spec
    (Severity / Component / File / Line / Problem / Why it matters / Suggested Fix).

    source_document and source_snippet are optional: they cite the approved
    DocumentChunk (see doc_retrieval_tool.py) that grounds this finding, when
    one exists. Most findings will have neither until real documentation is
    approved in document_sources.yaml.

    review_category records which reviewer category produced this finding
    (e.g. "security", "architecture", "code quality" - the same string
    already passed to build_review_prompt's review_category parameter).
    It is always stamped by the reviewer node in code, right before
    validation, never something the Model Service is asked to report about
    itself - finalize_review.py relies on it to compute per-category scores.

    component is likewise never reported by the Model Service: models were
    observed repeatedly failing to match one of Component's exact display
    values (see DECISION-LOG.md's "Component derived deterministically"
    entry), so finding_validation.py always derives it from this finding's
    own file field instead, via component_classifier.py's classify_file() -
    the same function nodes/classify.py uses to classify the PR's changed
    files.

    merge_note is optional: finding_deduplication.py sets it whenever this
    finding was merged from more than one reviewer's findings, listing a
    brief summary of each original finding (and, when they disagreed, the
    differing component and/or review_category values) - a merge must
    never be silent, even when every merged finding happened to share the
    same component and category.
    """

    severity: Severity
    component: Component
    review_category: str
    file: str
    line: Optional[int] = None
    problem: str
    why_it_matters: str
    suggested_fix: str
    source_document: Optional[str] = None
    source_snippet: Optional[str] = None
    merge_note: Optional[str] = None
