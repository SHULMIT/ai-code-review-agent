"""Deterministic pull-request checks that need no Model Service call.

Every check here answers a question with a definite answer: a file either is
committed or is not, a lockfile either changed alongside its manifest or did
not. Nothing is inferred, so nothing can be hallucinated - the file and line
on each finding come from the diff itself rather than from a model's account
of it.

That also means severity is decided here, in code, per check. The
model-backed reviewers currently let the model choose severity, which has
been observed producing different severities for the same line across
consecutive runs. These checks cannot drift: a committed .env file is
BLOCKING every time.

Checks are deliberately conservative. A false positive from a deterministic
check is worse than one from a model, because a reader reasonably assumes a
mechanical check is right.
"""

import re
from pathlib import PurePosixPath

from review_agent.models.finding import Component, Finding, Severity

# Filenames and directories that should never be committed. Matched against
# path segments, so a nested services/x/.env is caught the same as a root one.
_FORBIDDEN_NAMES = {".env", "__pycache__", ".DS_Store", ".coverage"}
_FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".log", ".sqlite3")
# .env.example and friends are documentation, not secrets - they are the
# file this module elsewhere asks people to keep updated.
_ALLOWED_ENV_FILES = re.compile(r"^\.env\.(example|sample|template)$")

_MANIFEST_TO_LOCKFILE = {
    "pyproject.toml": "uv.lock",
    "requirements.txt": "requirements.lock",
    "package.json": "package-lock.json",
}

# Conservative on purpose: each pattern requires an assignment to a
# secret-shaped name AND a quoted literal of real length. A value read from
# the environment or from settings never matches, because it is not a
# literal.
_SECRET_PATTERNS = (
    re.compile(
        r"""(?ix)
        \b(?:api[_-]?key|secret|password|passwd|token|access[_-]?key|private[_-]?key)\b
        \s*[:=]\s*
        ['"][^'"\s${}]{12,}['"]
        """
    ),
    re.compile(r"""(?i)\b(?:sk-|ghp_|github_pat_|xox[baprs]-|AKIA)[A-Za-z0-9_\-]{10,}"""),
)
# Obvious non-secrets that match the shape above.
_SECRET_PLACEHOLDERS = re.compile(
    r"(?i)(your[_-]?|example|placeholder|changeme|xxxx|<.*>|\.\.\.|fake|dummy|redacted)"
)

DEFAULT_MAX_CHANGED_FILES = 40
DEFAULT_MAX_ADDED_LINES = 800


def _finding(
    severity: Severity,
    review_category: str,
    file: str,
    line: int | None,
    problem: str,
    why_it_matters: str,
    suggested_fix: str,
) -> Finding:
    """Build a Finding with the fields every static check shares.

    component is always Unknown: these checks run before any per-file
    component classification and describe the change itself rather than a
    service's behaviour. The reviewer prompts use Unknown the same way.
    """
    return Finding(
        severity=severity,
        component=Component.UNKNOWN,
        review_category=review_category,
        file=file,
        line=line,
        problem=problem,
        why_it_matters=why_it_matters,
        suggested_fix=suggested_fix,
    )


def parse_added_lines(diff: str) -> list[tuple[str, int, str]]:
    """Return (path, new-file line number, text) for every added line in a unified diff.

    Line numbers come from the hunk headers, so a finding can point at the
    line as it appears in the file after the change - the same numbering the
    reviewers' findings use, and the only one a reader can act on.
    """
    added: list[tuple[str, int, str]] = []
    path = ""
    line_number = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            path = target[2:] if target.startswith("b/") else target
            continue
        if raw.startswith("@@"):
            match = re.search(r"\+(\d+)", raw)
            line_number = int(match.group(1)) if match else 0
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            if path and path != "/dev/null":
                added.append((path, line_number, raw[1:]))
            line_number += 1
        elif not raw.startswith("-") and not raw.startswith("\\"):
            line_number += 1
    return added


def check_forbidden_files(changed_files: list[str]) -> list[Finding]:
    """Flag files that should never reach version control."""
    findings: list[Finding] = []
    for path in changed_files:
        parts = PurePosixPath(path).parts
        name = parts[-1] if parts else path
        if _ALLOWED_ENV_FILES.match(name):
            continue
        hit = (
            name in _FORBIDDEN_NAMES
            or any(part in _FORBIDDEN_NAMES for part in parts[:-1])
            or name.endswith(_FORBIDDEN_SUFFIXES)
        )
        if not hit:
            continue
        is_secret_file = name == ".env"
        findings.append(
            _finding(
                Severity.BLOCKING if is_secret_file else Severity.WARNING,
                "security" if is_secret_file else "code quality",
                path,
                None,
                f"{path} is committed in this pull request, and files of this kind should not be.",
                (
                    "A committed .env exposes every credential it holds to anyone with "
                    "repository access, and it stays in git history after deletion."
                    if is_secret_file
                    else "Generated and local-only files create noise in every future diff "
                    "and conflict between machines."
                ),
                f"Remove {path} from the change and add its pattern to .gitignore.",
            )
        )
    return findings


def check_hardcoded_secrets(diff: str) -> list[Finding]:
    """Flag added lines that assign a secret-shaped name to a long quoted literal."""
    findings: list[Finding] = []
    for path, line_number, text in parse_added_lines(diff):
        if not any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            continue
        if _SECRET_PLACEHOLDERS.search(text):
            continue
        findings.append(
            _finding(
                Severity.BLOCKING,
                "security",
                path,
                line_number,
                "This added line assigns what looks like a real credential to a literal value.",
                (
                    "A credential in source is readable by everyone with repository access, "
                    "remains in git history after it is deleted, and cannot be rotated "
                    "without changing code."
                ),
                "Read the value from configuration or a secret store, and rotate the exposed credential.",
            )
        )
    return findings


def check_lockfile_kept_in_sync(changed_files: list[str]) -> list[Finding]:
    """Flag a dependency manifest changed without its lockfile."""
    findings: list[Finding] = []
    names = {PurePosixPath(path).name for path in changed_files}
    for path in changed_files:
        manifest = PurePosixPath(path).name
        lockfile = _MANIFEST_TO_LOCKFILE.get(manifest)
        if lockfile is None or lockfile in names:
            continue
        findings.append(
            _finding(
                Severity.WARNING,
                "code quality",
                path,
                None,
                f"{manifest} changed but {lockfile} did not.",
                (
                    "Without the matching lockfile update the install is no longer "
                    "reproducible: two machines can resolve different versions from the "
                    "same manifest."
                ),
                f"Regenerate {lockfile} and include it in this change.",
            )
        )
    return findings


def check_undocumented_settings(diff: str, changed_files: list[str]) -> list[Finding]:
    """Flag an environment variable newly read in code but absent from .env.example."""
    example_changed = any(
        _ALLOWED_ENV_FILES.match(PurePosixPath(path).name) for path in changed_files
    )
    if example_changed:
        return []

    pattern = re.compile(r"""(?:os\.environ(?:\.get)?[\[(]|getenv\()\s*['"]([A-Z][A-Z0-9_]{2,})['"]""")
    seen: dict[str, tuple[str, int]] = {}
    for path, line_number, text in parse_added_lines(diff):
        for name in pattern.findall(text):
            seen.setdefault(name, (path, line_number))

    return [
        _finding(
            Severity.WARNING,
            "code quality",
            path,
            line_number,
            f"{name} is read here but no .env.example entry was added for it.",
            (
                "Whoever runs this service next has no way to know the variable exists. "
                "The service starts and then fails at the point the value is first needed."
            ),
            f"Add {name} to .env.example with a comment describing what it configures.",
        )
        for name, (path, line_number) in sorted(seen.items())
    ]


def check_change_size(
    diff: str,
    changed_files: list[str],
    max_changed_files: int = DEFAULT_MAX_CHANGED_FILES,
    max_added_lines: int = DEFAULT_MAX_ADDED_LINES,
) -> list[Finding]:
    """Flag a change too large to be reviewed meaningfully, by a human or by this agent."""
    added_lines = len(parse_added_lines(diff))
    file_count = len(changed_files)
    if file_count <= max_changed_files and added_lines <= max_added_lines:
        return []
    return [
        _finding(
            Severity.WARNING,
            "code quality",
            changed_files[0] if changed_files else "",
            None,
            (
                f"This pull request changes {file_count} files and adds {added_lines} "
                f"lines, past the reviewable limit of {max_changed_files} files or "
                f"{max_added_lines} added lines."
            ),
            (
                "A change this size is approved without being read. It also exceeds the "
                "diff budget this agent sends to the Model Service, so parts of it were "
                "reviewed by nobody - neither a person nor a reviewer node."
            ),
            "Split the change into smaller pull requests that can each be reviewed on their own.",
        )
    ]


def run_static_checks(diff: str, changed_files: list[str]) -> list[Finding]:
    """Run every deterministic check and return the combined findings."""
    return [
        *check_forbidden_files(changed_files),
        *check_hardcoded_secrets(diff),
        *check_lockfile_kept_in_sync(changed_files),
        *check_undocumented_settings(diff, changed_files),
        *check_change_size(diff, changed_files),
    ]
