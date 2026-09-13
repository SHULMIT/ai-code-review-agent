"""Tests for the process startup entry point."""

from review_agent import main
from review_agent.config import Settings


def test_run_starts_uvicorn_with_configured_network_settings(monkeypatch) -> None:
    """The startup entry point builds a Uvicorn Server from configured network settings.

    main.run() drives uvicorn.Server.serve() directly (see its own
    docstring - never uvicorn.run()), so this fakes uvicorn.Config and
    uvicorn.Server themselves rather than a run() call that no longer
    happens; the fake Server's serve() is a no-op coroutine so this test
    never opens a real socket.
    """

    settings = Settings(
        _env_file=None,
        host="127.0.0.1",
        port=8100,
        github_app_id=123,
        github_app_private_key_path="secrets/test-private-key.pem",
        github_webhook_secret="test-webhook-secret",
    )
    captured: dict[str, object] = {}

    def fake_config(*args: object, **kwargs: object):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "fake-config"

    class _FakeServer:
        def __init__(self, config: object) -> None:
            captured["server_config"] = config

        async def serve(self) -> None:
            return None

    monkeypatch.setattr(main, "Settings", lambda: settings)
    monkeypatch.setattr(main.uvicorn, "Config", fake_config)
    monkeypatch.setattr(main.uvicorn, "Server", _FakeServer)

    main.run()

    assert captured["args"] == (main.app,)
    assert captured["kwargs"] == {
        "host": "127.0.0.1",
        "port": 8100,
        "log_level": "info",
    }
    assert captured["server_config"] == "fake-config"
