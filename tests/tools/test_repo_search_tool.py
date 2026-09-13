import asyncio
from pathlib import Path

import pytest

from review_agent.tools import repo_search_tool
from review_agent.tools.repo_search_tool import (
    RepoSearchOutcome,
    RepoSearchToolError,
    SearchResult,
    search_repository,
)


async def test_run_git_keeps_authorization_out_of_process_arguments(monkeypatch):
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "unsafe-global-config")
    monkeypatch.setattr(
        repo_search_tool.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )

    auth_header = "AUTHORIZATION: basic secret-token"
    await repo_search_tool._run_git(
        ["fetch", "origin", "deadbeef"], Path("repo"), timeout_seconds=5, auth_header=auth_header
    )

    assert "secret-token" not in " ".join(captured["args"])
    environment = captured["kwargs"]["env"]
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == repo_search_tool.os.devnull
    assert environment["GIT_CONFIG_KEY_0"] == "http.extraHeader"
    assert environment["GIT_CONFIG_VALUE_0"] == auth_header


async def test_run_git_kills_the_process_and_raises_on_timeout(monkeypatch):
    killed = {"called": False}

    class FakeProcess:
        returncode = None

        async def communicate(self):
            await asyncio.sleep(10)  # never resolves before the test's short timeout

        def kill(self):
            killed["called"] = True

        async def wait(self):
            return None

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(
        repo_search_tool.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )

    with pytest.raises(RepoSearchToolError, match="timed out after"):
        await repo_search_tool._run_git(
            ["fetch", "origin", "deadbeef"], Path("repo"), timeout_seconds=0.05
        )

    assert killed["called"] is True


async def test_search_repository_finds_matches(monkeypatch):
    captured = {}

    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        captured["path"] = dest
        (dest / "example.py").write_text(
            "import os\nresult = do_search(query)\n", encoding="utf-8"
        )
        (dest / "notes.txt").write_text("nothing relevant here\n", encoding="utf-8")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository("example-org/example-repo", "deadbeef", ["do_search"], "fake-token")

    assert isinstance(outcome, RepoSearchOutcome)
    assert len(outcome.results) == 1
    assert isinstance(outcome.results[0], SearchResult)
    assert outcome.results[0].file == "example.py"
    assert outcome.results[0].line == 2
    assert "do_search" in outcome.results[0].snippet
    assert outcome.truncated is False
    assert outcome.results_capped is False
    assert outcome.skipped_large_files == []
    assert outcome.skipped_binary_files == []
    assert not captured["path"].exists()  # temp dir cleaned up afterward


async def test_search_repository_no_matches(monkeypatch):
    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        (dest / "example.py").write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository(
        "example-org/example-repo", "deadbeef", ["nonexistent_term"], "fake-token"
    )

    assert outcome.results == []


async def test_search_repository_uses_word_boundaries_not_substring_matches(monkeypatch):
    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        (dest / "matches.py").write_text("value = os.getenv('HOME')\n", encoding="utf-8")
        (dest / "incidental.py").write_text("connection.close()\nhostname = 'x'\n", encoding="utf-8")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository("example-org/example-repo", "deadbeef", ["os"], "fake-token")

    assert [(result.file, result.line) for result in outcome.results] == [("matches.py", 1)]


async def test_search_repository_wraps_clone_failure(monkeypatch):
    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    with pytest.raises(RepoSearchToolError):
        await search_repository("example-org/example-repo", "deadbeef", ["query"], "fake-token")


async def test_search_repository_cleans_up_temp_dir_after_a_clone_timeout(monkeypatch):
    captured = {}

    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        captured["path"] = dest
        raise RepoSearchToolError("git fetch timed out after 1s")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    with pytest.raises(RepoSearchToolError, match="timed out"):
        await search_repository(
            "example-org/example-repo", "deadbeef", ["query"], "fake-token", timeout_seconds=1
        )

    assert not captured["path"].exists()


async def test_search_repository_truncates_scan_at_max_scan_files(monkeypatch):
    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        for index in range(5):
            (dest / f"file_{index}.py").write_text(f"needle {index}\n", encoding="utf-8")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository(
        "example-org/example-repo", "deadbeef", ["needle"], "fake-token", max_scan_files=3
    )

    assert outcome.truncated is True
    assert outcome.total_file_count == 5
    assert outcome.scanned_file_count == 3
    assert len(outcome.results) == 3  # one match per scanned file, in sorted order


async def test_search_repository_scan_not_truncated_within_the_limit(monkeypatch):
    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        (dest / "a.py").write_text("needle\n", encoding="utf-8")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository(
        "example-org/example-repo", "deadbeef", ["needle"], "fake-token", max_scan_files=10
    )

    assert outcome.truncated is False
    assert outcome.total_file_count == 1
    assert outcome.scanned_file_count == 1


async def test_search_repository_skips_a_file_exceeding_max_file_bytes(monkeypatch):
    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        (dest / "huge.py").write_text("needle " * 10, encoding="utf-8")  # 70 bytes
        (dest / "small.py").write_text("needle\n", encoding="utf-8")  # 7 bytes

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    original_read_text = Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        assert self.name != "huge.py", "a file over the size limit must never be read"
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    outcome = await search_repository(
        "example-org/example-repo", "deadbeef", ["needle"], "fake-token", max_file_bytes=20
    )

    assert outcome.skipped_large_files == ["huge.py"]
    assert all(result.file != "huge.py" for result in outcome.results)
    assert any(result.file == "small.py" for result in outcome.results)


async def test_search_repository_skips_a_binary_file(monkeypatch):
    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        (dest / "binary.dat").write_bytes(b"needle\x00binary junk")
        (dest / "text.py").write_text("needle\n", encoding="utf-8")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository("example-org/example-repo", "deadbeef", ["needle"], "fake-token")

    assert outcome.skipped_binary_files == ["binary.dat"]
    assert all(result.file != "binary.dat" for result in outcome.results)
    assert any(result.file == "text.py" for result in outcome.results)


async def test_search_repository_caps_results_at_max_results(monkeypatch):
    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        lines = "\n".join(f"needle {index}" for index in range(10))
        (dest / "many_matches.py").write_text(lines + "\n", encoding="utf-8")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository(
        "example-org/example-repo", "deadbeef", ["needle"], "fake-token", max_results=3
    )

    assert len(outcome.results) == 3
    assert outcome.results_capped is True


# --- one clone for many terms -------------------------------------------------


async def test_search_repository_clones_once_for_many_terms(monkeypatch):
    """The whole point of taking a list: N terms must not mean N clones.

    Each term used to trigger its own shallow clone of the identical repo at
    the identical commit - a real network fetch and a full tree scan per
    term. This asserts the count directly, because the results alone look
    the same either way and would not catch a regression to per-term cloning.
    """
    clone_count = {"value": 0}

    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        clone_count["value"] += 1
        (dest / "app.py").write_text(
            "import httpx\nfrom pathlib import Path\nclient = VectorDBClient()\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository(
        "example-org/example-repo",
        "deadbeef",
        ["httpx", "Path", "VectorDBClient"],
        "fake-token",
    )

    assert clone_count["value"] == 1, "three terms must still produce exactly one clone"
    assert {result.line for result in outcome.results} == {1, 2, 3}


async def test_search_repository_reports_a_line_matching_two_terms_once(monkeypatch):
    """A line is relevant or it is not - matching twice does not make it two findings."""

    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        (dest / "app.py").write_text("from httpx import Client\n", encoding="utf-8")

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository(
        "example-org/example-repo", "deadbeef", ["httpx", "Client"], "fake-token"
    )

    assert len(outcome.results) == 1


async def test_search_repository_with_no_terms_does_not_clone(monkeypatch):
    """Nothing to search for means no reason to pay for a network fetch."""
    clone_count = {"value": 0}

    async def fake_clone(repo, head_sha, token, dest, timeout_seconds):
        clone_count["value"] += 1

    monkeypatch.setattr(repo_search_tool, "_shallow_clone", fake_clone)

    outcome = await search_repository("example-org/example-repo", "deadbeef", [], "fake-token")

    assert clone_count["value"] == 0
    assert outcome.results == []
    assert outcome.truncated is False
    assert outcome.results_capped is False
