"""Thin consumer of config/component_mapping.yaml - the same mapping shape a
future contributor's repo-mapping tool may provide. This module should stay
a plain reader of that file, not grow its own classification rules.
"""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError, field_validator

from review_agent.models.finding import Component


class ComponentRule(BaseModel):
    """One component-mapping entry: a glob pattern and the component it maps to."""

    pattern: str
    component: Component

    @field_validator("component", mode="before")
    @classmethod
    def resolve_component_name(cls, value: object) -> object:
        """Resolve the mapping YAML's enum member name (e.g. "AGENT_HARNESS") to its Component.

        The YAML stores Component enum member names, not enum values, so this
        must look the name up via Component[...] instead of relying on
        Pydantic's default by-value enum validation. Component is itself a
        str subclass, so an already-resolved Component must be passed
        through before the plain-string branch, or its value would be
        looked up as if it were a member name and fail.
        """
        if isinstance(value, Component):
            return value
        if isinstance(value, str):
            try:
                return Component[value]
            except KeyError as exc:
                raise ValueError(f"Unknown component name: {value!r}") from exc
        return value


def load_mapping(path: Path) -> list[ComponentRule]:
    """Read the component-mapping YAML file and validate it into typed rules.

    Raises a pydantic ValidationError immediately for an unknown component
    name or a rule missing "pattern"/"component", instead of that surfacing
    as a bare KeyError later inside classify_file.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_rules = yaml.safe_load(f) or []
    return [ComponentRule.model_validate(rule) for rule in raw_rules]


_SERVICE_ROOT = Path(__file__).resolve().parents[3]
# The one definition of where the mapping lives. It used to be spelled out
# separately in build_graph.py and finding_validation.py, each carrying a
# comment asking the other to be kept in sync - two sources of truth for one
# fact. The module that reads the file owns its location.
DEFAULT_MAPPING_PATH = _SERVICE_ROOT / "config" / "component_mapping.yaml"


class ComponentMapping(BaseModel):
    """The parsed mapping, plus why it is empty when loading it failed.

    Carrying the failure alongside the rules rather than raising is what
    lets the mapping be loaded once, at graph-build time, without a broken
    YAML file taking down graph construction. A missing or invalid mapping
    is a degradation - every finding classifies as Component.UNKNOWN - not
    a reason the review cannot run at all, which is the opposite of a
    missing reviewer prompt.
    """

    rules: list[ComponentRule] = []
    load_error: str | None = None


def load_component_mapping(path: Path) -> ComponentMapping:
    """Load the mapping once for the whole graph, reporting failure instead of raising.

    The single place the mapping file is read. Every consumer - classify and
    all six reviewer nodes, through validate_findings - receives the parsed
    rules rather than a path, so nothing downstream of graph construction
    touches the filesystem or needs to know where the file lives.
    """
    try:
        return ComponentMapping(rules=load_mapping(path))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        return ComponentMapping(rules=[], load_error=f"invalid component mapping: {exc}")


def _pattern_to_regex(pattern: str) -> re.Pattern:
    """Translate a glob pattern into a path-aware regex.

    '*' matches within a single path segment (never crosses '/');
    '**' matches across segments, including zero segments.
    """
    regex_parts = []
    i = 0
    n = len(pattern)
    while i < n:
        if pattern[i : i + 2] == "**":
            if i + 2 < n and pattern[i + 2] == "/":
                regex_parts.append("(?:.*/)?")
                i += 3
            else:
                regex_parts.append(".*")
                i += 2
        elif pattern[i] == "*":
            regex_parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            regex_parts.append("[^/]")
            i += 1
        else:
            regex_parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(regex_parts) + "$")


def classify_file(path: str, rules: list[ComponentRule]) -> Component:
    """Match a file path against the rules in order; the first match wins."""
    normalized = Path(path).as_posix()
    for rule in rules:
        if _pattern_to_regex(rule.pattern).match(normalized):
            return rule.component
    return Component.UNKNOWN
