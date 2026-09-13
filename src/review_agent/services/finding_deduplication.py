"""Consolidates overlapping findings from multiple reviewers into one merged list.

Reviewer prompts (security_review.md, architecture_review.md) now ask the
model to stay within its own review category, but this session's own
evidence - see DECISION-LOG.md's "Reviewer prompt scope boundaries are not
enforced" entry - shows prompt compliance cannot be relied on alone: the
Architecture reviewer duplicated a finding Security had already reported.
This module is the actual enforcement mechanism, not the prompt wording. It
is meant to run once, after every reviewer node completes and before
scoring, merging findings that are almost certainly reports of the same
underlying issue regardless of which reviewer produced them.

Similarity is decided by _findings_are_similar, which now makes a real
network call (an embedding lookup) the first time this module ever has -
see DECISION-LOG.md's "finding_deduplication.py's similarity check
upgraded to embeddings" entry for why, and why a word-overlap fallback is
required rather than optional whenever that call fails.
"""

import math
import re

import httpx

from review_agent.config import Settings
from review_agent.models.finding import Finding, Severity
from review_agent.services.finding_validation import LINE_TOLERANCE
from review_agent.tools.embedding_tool import EmbeddingServiceError, get_embeddings

_SEVERITY_RANK = {Severity.SUGGESTION: 0, Severity.WARNING: 1, Severity.BLOCKING: 2}
_SIMILARITY_THRESHOLD = 0.5
_MIN_SIGNIFICANT_WORD_LENGTH = 4
_MIN_CORROBORATING_WORD_LENGTH = 5
_MIN_CORROBORATING_TERMS = 2
_WORD_RE = re.compile(r"[a-z0-9]+")
_CODE_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])_?[A-Z][A-Z0-9_]{2,}(?![A-Za-z0-9])")
_NON_CORROBORATING_WORDS = {
    "additional",
    "about",
    "after",
    "before",
    "change",
    "changes",
    "code",
    "comprehensive",
    "context",
    "could",
    "ensure",
    "file",
    "files",
    "function",
    "functions",
    "have",
    "important",
    "include",
    "includes",
    "information",
    "lack",
    "lacks",
    "missing",
    "modify",
    "necessary",
    "needs",
    "problem",
    "provide",
    "provided",
    "result",
    "results",
    "return",
    "returns",
    "should",
    "suggested",
    "using",
    "without",
}

# Calibrated against 6 real embedding calls (3 confirmed same-issue pairs
# scoring 0.5661-0.9563, 3 confirmed different-issue pairs scoring
# 0.2973-0.4321) - see DECISION-LOG.md. A separate constant from
# _SIMILARITY_THRESHOLD above: same numeral, different metric, derived
# independently rather than carried over from the word-overlap heuristic.
_EMBEDDING_SIMILARITY_THRESHOLD = 0.5
# Only a very strong semantic match may cross a large line-distance boundary.
# Calibrated from the known distant no-tests duplicate (0.9563) and the
# Vector DB/error-handling non-duplicate (0.4321): midpoint 0.6942.
_DISTANT_EMBEDDING_SIMILARITY_THRESHOLD = 0.7
# Distant findings must also identify the same problem, not merely share a
# semantically similar impact or fix. The known distant no-tests duplicate
# scores 0.9563 problem-only; the observed Vector DB/error-handling false
# positive scores 0.5348. This threshold is separate from the full-content
# thresholds because it is an identity gate used only across distant lines.
_DISTANT_PROBLEM_EMBEDDING_SIMILARITY_THRESHOLD = 0.7
_REVIEW_CATEGORY_KEYWORDS = {
    "security": {"secret", "credential", "token", "apikey", "api", "authorization", "injection"},
    "architecture": {"boundary", "service", "layer", "dependency", "separation"},
    "code quality": {"test", "documentation", "readme", "constant", "magic", "duplicate", "naming"},
    "rag_search": {"cache", "search", "retrieval", "ranking", "relevance", "embedding", "grounding", "source"},
    "performance": {"unbounded", "memory", "latency", "throughput", "parallel", "timeout"},
    "agent_harness": {"langgraph", "graph", "state", "reviewer", "model output", "node"},
}
_REVIEW_CATEGORY_ORDER = tuple(_REVIEW_CATEGORY_KEYWORDS)


async def deduplicate_findings(
    findings: list[Finding], settings: Settings, client: httpx.AsyncClient | None = None
) -> list[Finding]:
    """Merge findings that different reviewers likely reported for the same underlying issue.

    Findings with both a file and a line are first clustered by line number
    using LINE_TOLERANCE - the same best-effort tolerance
    finding_validation.py already uses - so two findings a few lines apart
    still merge. A second same-file semantic pass then catches duplicates
    reported at distant lines (for example, two whole-file "no tests"
    concerns assigned to lines 1 and 31 by different reviewers). Findings
    missing a file or a line are grouped by file plus semantic similarity
    directly. Always returns a valid list[Finding], possibly shorter than
    the input; never raises on a valid list[Finding] input, since anything
    it receives has already passed Finding's own validation, and a failed
    similarity lookup falls back to a heuristic rather than propagating.

    client lets a caller pass in one pooled httpx.AsyncClient shared across
    an entire review run instead of opening a fresh TCP connection per call
    - see RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 5.
    """
    embedding_cache = await _build_embedding_cache(findings, settings, client)
    with_location = [finding for finding in findings if finding.file and finding.line is not None]
    without_location = [finding for finding in findings if not (finding.file and finding.line is not None)]

    merged: list[Finding] = []
    for file_findings in _group_by_file(with_location).values():
        merged.extend(_cluster_by_line(file_findings, embedding_cache))
    for file_findings in _group_by_file(without_location).values():
        merged.extend(_cluster_by_problem_similarity(file_findings, embedding_cache))

    return merged


async def _build_embedding_cache(
    findings: list[Finding], settings: Settings, client: httpx.AsyncClient | None = None
) -> dict[str, list[float]]:
    """Fetch each full and problem-only embedding once, or return fallback cache.

    An empty cache deliberately means "use word overlap". This includes an
    unavailable or malformed embedding response: deduplication is a best-
    effort refinement and must never make the review graph fail.
    """

    full_texts = list(dict.fromkeys(_finding_embedding_text(finding) for finding in findings))
    if len(full_texts) < 2:
        return {}
    problem_texts = list(dict.fromkeys(finding.problem for finding in findings))
    embedding_texts = list(dict.fromkeys(full_texts + problem_texts))

    try:
        embed_kwargs = {} if client is None else {"client": client}
        embeddings = await get_embeddings(embedding_texts, settings, **embed_kwargs)
        if len(embeddings) != len(embedding_texts):
            return {}
    except (EmbeddingServiceError, TypeError, ValueError):
        return {}

    return dict(zip(embedding_texts, embeddings, strict=True))


def _finding_embedding_text(finding: Finding) -> str:
    """Return the complete reviewer-supplied semantic content for embedding.

    Some reviewers provide a short title-like problem (for example,
    ``"Missing RAG grounding."``) and explain the concrete issue only in
    the impact or fix. Embedding all three fields lets deduplication compare
    the actual finding while the fallback continues to use the historical
    problem-text word-overlap heuristic unchanged.
    """

    return "\n".join(
        (
            f"Problem: {finding.problem}",
            f"Impact: {finding.why_it_matters}",
            f"Suggested fix: {finding.suggested_fix}",
        )
    )


def _group_by_file(findings: list[Finding]) -> dict[str, list[Finding]]:
    groups: dict[str, list[Finding]] = {}
    for finding in findings:
        groups.setdefault(finding.file, []).append(finding)
    return groups


def _cluster_by_line(
    findings: list[Finding], embedding_cache: dict[str, list[float]]
) -> list[Finding]:
    """Cluster nearby same-file findings, then merge distant semantic duplicates.

    Line proximity alone is not sufficient: two genuinely unrelated findings
    (e.g. a hardcoded secret and an unrelated direct Vector DB dependency)
    can land a few lines apart by pure coincidence and must not be merged
    just because of that - doing so silently discards one of them. This
    reuses _findings_are_similar as an additional requirement alongside
    proximity.
    """
    ordered = sorted(findings, key=lambda finding: finding.line)
    nearby_clusters: list[list[Finding]] = []
    for finding in ordered:
        if (
            nearby_clusters
            and finding.line - nearby_clusters[-1][-1].line <= LINE_TOLERANCE
            and _findings_are_similar(
                nearby_clusters[-1][-1], finding, embedding_cache
            )
        ):
            nearby_clusters[-1].append(finding)
        else:
            nearby_clusters.append([finding])

    semantic_clusters: list[list[Finding]] = []
    for nearby_cluster in nearby_clusters:
        match = next(
            (
                cluster
                for cluster in semantic_clusters
                if _clusters_are_similar(cluster, nearby_cluster, embedding_cache)
            ),
            None,
        )
        if match is None:
            semantic_clusters.append(nearby_cluster)
        else:
            match.extend(nearby_cluster)

    return [_merge_group(cluster) for cluster in semantic_clusters]


def _clusters_are_similar(
    first: list[Finding], second: list[Finding], embedding_cache: dict[str, list[float]]
) -> bool:
    """Return whether any pair across two line clusters describes one issue."""

    return any(
        _distant_findings_are_similar(
            first_finding,
            second_finding,
            embedding_cache,
        )
        for first_finding in first
        for second_finding in second
    )


def _distant_findings_are_similar(
    first: Finding, second: Finding, embedding_cache: dict[str, list[float]]
) -> bool:
    """Require full-content and problem-only agreement across distant lines.

    A nearby finding may have a terse title whose meaningful detail lives in
    its impact or fix. At distant locations, however, matching only that
    broad context can merge distinct concerns about the same dependency.
    Both embedding views must therefore meet their separate conservative
    thresholds. If either vector cannot be used, preserve the existing
    word-overlap fallback exactly.
    """

    full_score = _embedding_similarity(
        _finding_embedding_text(first), _finding_embedding_text(second), embedding_cache
    )
    problem_score = _embedding_similarity(first.problem, second.problem, embedding_cache)
    if full_score is None or problem_score is None:
        return _word_overlap_similar(first.problem, second.problem)
    if full_score < _DISTANT_EMBEDDING_SIMILARITY_THRESHOLD:
        return False
    if problem_score >= _DISTANT_PROBLEM_EMBEDDING_SIMILARITY_THRESHOLD:
        return _findings_have_corroborating_term(first, second)
    # A precise source-code identifier is stronger evidence than generic
    # prose overlap: two reviewers can describe distinct symptoms (growth
    # and invalidation) of the same _QUERY_CACHE defect with little shared
    # natural language. It is intentionally limited to explicit code-like
    # identifiers, never ordinary terms such as "cache".
    return bool(_concrete_identifiers(first) & _concrete_identifiers(second))


def _cluster_by_problem_similarity(
    findings: list[Finding], embedding_cache: dict[str, list[float]]
) -> list[Finding]:
    """Cluster same-file findings whose `problem` text is similar."""
    clusters: list[list[Finding]] = []
    for finding in findings:
        match = None
        for cluster in clusters:
            if _findings_are_similar(cluster[0], finding, embedding_cache):
                match = cluster
                break
        if match is not None:
            match.append(finding)
        else:
            clusters.append([finding])
    return [_merge_group(cluster) for cluster in clusters]


def _findings_are_similar(
    first: Finding,
    second: Finding,
    embedding_cache: dict[str, list[float]],
    *,
    embedding_similarity_threshold: float = _EMBEDDING_SIMILARITY_THRESHOLD,
) -> bool:
    """Decide whether two findings describe the same underlying issue.

    Tries the embedding-based similarity first - it correctly recognizes
    paraphrases in different vocabulary that the word-overlap heuristic
    misses (see DECISION-LOG.md). Falls back to the word-overlap heuristic
    when no usable embedding comparison is available - not when a valid
    comparison simply scores low - so an outage or malformed response
    degrades dedup to a less precise method instead of ever crashing it.
    """
    embedding_score = _embedding_similarity(
        _finding_embedding_text(first), _finding_embedding_text(second), embedding_cache
    )
    if embedding_score is not None:
        return (
            embedding_score >= embedding_similarity_threshold
            and _findings_have_corroborating_term(first, second)
        )
    return _word_overlap_similar(first.problem, second.problem)


def _findings_have_corroborating_term(first: Finding, second: Finding) -> bool:
    """Require two shared content-specific terms before an embedding match can merge.

    Embeddings can place generic code-quality prose close together even when
    the actionable concerns differ (for example, documentation and type
    hints). Two shared non-generic terms keep a 0.5+ cosine score a
    candidate, but make a destructive merge conservative. It is
    intentionally not applied to the network-failure fallback, which
    preserves the prior word-overlap behavior exactly.
    """

    shared_terms = _corroborating_terms(first) & _corroborating_terms(second)
    return len(shared_terms) >= _MIN_CORROBORATING_TERMS


def _corroborating_terms(finding: Finding) -> set[str]:
    """Return non-generic words that substantiate a semantic match."""

    text = " ".join(
        (finding.problem, finding.why_it_matters, finding.suggested_fix)
    )
    return {
        word
        for word in _WORD_RE.findall(text.lower())
        if len(word) >= _MIN_CORROBORATING_WORD_LENGTH
        and word not in _NON_CORROBORATING_WORDS
    }


def _concrete_identifiers(finding: Finding) -> set[str]:
    """Return explicit all-caps/underscore code identifiers cited by a finding."""

    text = " ".join((finding.problem, finding.why_it_matters, finding.suggested_fix))
    return {identifier.upper() for identifier in _CODE_IDENTIFIER_RE.findall(text)}


def _embedding_similarity(
    text_a: str, text_b: str, embedding_cache: dict[str, list[float]]
) -> float | None:
    """Return cached cosine similarity, or None when vectors cannot be used safely.

    The cache is empty when the batched service call failed. Directly
    supplied malformed vectors are also treated as unavailable here, which
    keeps fallback behavior safe even if a future client changes.
    """
    vector_a = embedding_cache.get(text_a)
    vector_b = embedding_cache.get(text_b)
    if vector_a is None or vector_b is None:
        return None
    return _cosine_similarity(vector_a, vector_b)


def _cosine_similarity(a: list[float], b: list[float]) -> float | None:
    """Return a finite cosine score, or None when either vector is unusable."""

    if not a or not b or len(a) != len(b):
        return None
    try:
        if any(not math.isfinite(value) for value in a + b):
            return None

        dot_product = sum(x * y for x, y in zip(a, b, strict=True))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(y * y for y in b))
        if magnitude_a == 0 or magnitude_b == 0:
            return None

        similarity = dot_product / (magnitude_a * magnitude_b)
        return similarity if math.isfinite(similarity) else None
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def _word_overlap_similar(a: str, b: str) -> bool:
    """A simple word-overlap heuristic - no embeddings, just how much vocabulary two problems share.

    Uses the overlap coefficient (shared words / the smaller word set), not
    Jaccard similarity, since two paraphrases of the same issue are often
    different lengths - Jaccard's union-sized denominator would unfairly
    punish that instead of measuring whether the shorter one is "contained"
    in the longer one's vocabulary. Kept as the fallback for when the
    embedding service is unreachable - see _problems_are_similar.
    """
    words_a = _significant_words(a)
    words_b = _significant_words(b)
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b)) >= _SIMILARITY_THRESHOLD


def _significant_words(text: str) -> set[str]:
    """Lowercase words of at least _MIN_SIGNIFICANT_WORD_LENGTH characters, skipping short/common ones."""
    return {word for word in _WORD_RE.findall(text.lower()) if len(word) >= _MIN_SIGNIFICANT_WORD_LENGTH}


def _merge_group(group: list[Finding]) -> Finding:
    """Merge one cluster of likely-duplicate findings into a single Finding.

    Keeps the highest-severity finding, using its most detailed variant as a
    tie-breaker so a breaking defect is never softened by a weaker report. A
    merge is never silent: merge_note always records how many findings were
    merged and a brief summary of each one's problem, regardless of whether
    the group's component and review_category values agree - the fact that
    a merge happened at all is what must stay visible, not just a
    disagreement. When component and/or review_category values do differ
    across the group (e.g. Security and Architecture both flagged the same
    spot), that is called out too. The merged Finding keeps one category,
    selected deterministically from concrete defect vocabulary rather than
    whichever reviewer happened to return first.
    """
    if len(group) == 1:
        return group[0]

    primary = max(
        group,
        key=lambda finding: (
            _SEVERITY_RANK[finding.severity],
            len(finding.problem) + len(finding.why_it_matters) + len(finding.suggested_fix),
        ),
    )
    severity = max((finding.severity for finding in group), key=lambda severity: _SEVERITY_RANK[severity])
    review_category = _select_review_category(group)

    components = sorted({finding.component.value for finding in group})
    review_categories = sorted({finding.review_category for finding in group})
    problem_summaries = "; ".join(finding.problem for finding in group)
    merge_note = f"Merged {len(group)} independently reported findings: {problem_summaries}"

    disagreements = []
    if len(components) > 1:
        disagreements.append(f"component values disagreed: {', '.join(components)}")
    if len(review_categories) > 1:
        disagreements.append(f"review categories disagreed: {', '.join(review_categories)}")
    if disagreements:
        merge_note += f" ({'; '.join(disagreements)})"

    return primary.model_copy(
        update={
            "severity": severity,
            "review_category": review_category,
            "merge_note": merge_note,
        }
    )


def _select_review_category(group: list[Finding]) -> str:
    """Choose a stable final category from concrete defect vocabulary, not input order."""

    text = " ".join(
        " ".join((finding.problem, finding.why_it_matters, finding.suggested_fix)).lower()
        for finding in group
    )
    reported_counts = {
        category: sum(finding.review_category == category for finding in group)
        for category in _REVIEW_CATEGORY_ORDER
    }
    scores = {
        category: sum(keyword in text for keyword in keywords)
        for category, keywords in _REVIEW_CATEGORY_KEYWORDS.items()
    }
    return max(
        _REVIEW_CATEGORY_ORDER,
        key=lambda category: (
            scores[category],
            reported_counts[category],
            -_REVIEW_CATEGORY_ORDER.index(category),
        ),
    )
