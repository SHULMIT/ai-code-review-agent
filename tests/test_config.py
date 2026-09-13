from pathlib import Path

import pytest
from pydantic import ValidationError

from review_agent.config import ConfigurationError, ManualRunSettings, Settings


def _clear_settings_environment(monkeypatch) -> None:
    names = [
        "ENVIRONMENT",
        "LOG_LEVEL",
        "HOST",
        "PORT",
        "GITHUB_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        "GITHUB_WEBHOOK_SECRET",
        "DATABASE_URL",
        "MODEL_SERVICE_URL",
        "MODEL_SERVICE_MODEL_NAME",
        "MODEL_SERVICE_TIMEOUT_SECONDS",
        "EMBEDDING_SERVICE_TIMEOUT_SECONDS",
        "LLM_API_KEY",
        "LLM_MODEL",
        "GIT_COMMAND_TIMEOUT_SECONDS",
        "MAX_REPOSITORY_SCAN_FILES",
        "MAX_REPOSITORY_FILE_BYTES",
        "MAX_REPOSITORY_SEARCH_RESULTS",
        "MAX_REPOSITORY_CONTEXT_CHARACTERS",
        "MAX_DIFF_CHARACTERS",
        "MAX_MODEL_INPUT_TOKENS",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)


def _set_required_github_app_environment(monkeypatch) -> None:
    """Set non-secret GitHub App settings required by every Settings instance."""

    monkeypatch.setenv("GITHUB_APP_ID", "123")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "secrets/test-private-key.pem")


def test_settings_loads_development_token_from_env(monkeypatch):
    _clear_settings_environment(monkeypatch)
    _set_required_github_app_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "development-webhook-secret")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.github_token is not None
    assert settings.github_token.get_secret_value() == "fake-token"
    assert settings.log_level == "DEBUG"


def test_settings_requires_a_webhook_secret_in_every_environment(monkeypatch):
    _clear_settings_environment(monkeypatch)
    _set_required_github_app_environment(monkeypatch)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert exc_info.value.errors(include_url=False)[0]["type"] == "missing"
    assert exc_info.value.errors(include_url=False)[0]["loc"] == (
        "github_webhook_secret",
    )
    assert exc_info.value.errors(include_url=False)[0]["msg"] == "Field required"


def test_settings_uses_safe_development_defaults_with_a_webhook_secret(monkeypatch):
    _clear_settings_environment(monkeypatch)
    _set_required_github_app_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "development-webhook-secret")

    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.github_token is None
    assert settings.github_webhook_secret.get_secret_value() == "development-webhook-secret"
    assert settings.git_command_timeout_seconds == 60
    assert settings.max_diff_characters == 45_000
    assert settings.max_repository_context_characters == 15_000
    assert settings.model_service_url == "http://localhost:11434"
    assert settings.model_service_model_name == "qwen2.5:7b"
    assert settings.model_service_timeout_seconds == 180
    assert settings.embedding_service_timeout_seconds == 180
    assert settings.excluded_diff_paths == ""
    assert settings.excluded_diff_path_list() == ()
    settings.validate_runtime_configuration()


def test_excluded_diff_path_list_parses_comma_separated_patterns(monkeypatch):
    _clear_settings_environment(monkeypatch)
    _set_required_github_app_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "development-webhook-secret")
    monkeypatch.setenv("EXCLUDED_DIFF_PATHS", "all_classes_code.txt, , generated/*")

    settings = Settings(_env_file=None)

    assert settings.excluded_diff_path_list() == ("all_classes_code.txt", "generated/*")


def test_settings_rejects_invalid_resource_limits(monkeypatch):
    _clear_settings_environment(monkeypatch)
    _set_required_github_app_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "development-webhook-secret")
    monkeypatch.setenv("GIT_COMMAND_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_rejects_an_unsupported_log_level(monkeypatch):
    _clear_settings_environment(monkeypatch)
    _set_required_github_app_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "development-webhook-secret")
    monkeypatch.setenv("LOG_LEVEL", "TRACE")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_requires_github_app_credentials(monkeypatch):
    _clear_settings_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "development-webhook-secret")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert {error["loc"] for error in exc_info.value.errors(include_url=False)} == {
        ("github_app_id",),
        ("github_app_private_key_path",),
    }


def test_manual_run_settings_allow_the_pat_fallback_without_app_configuration(monkeypatch):
    """The manual PR check can temporarily use its existing personal token."""

    _clear_settings_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")

    settings = ManualRunSettings(_env_file=None)

    assert settings.github_token is not None
    assert settings.github_token.get_secret_value() == "fake-token"
    assert settings.github_app_installation_id is None


def test_manual_run_settings_resolve_an_optional_app_key_path(monkeypatch):
    """The future App path uses the same stable service-root resolution."""

    _clear_settings_environment(monkeypatch)
    _set_required_github_app_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "456")

    settings = ManualRunSettings(_env_file=None)

    assert settings.github_app_installation_id == 456
    assert settings.github_app_private_key_path == (
        Path(__file__).parents[1] / "secrets" / "test-private-key.pem"
    )


def test_production_settings_require_database_and_model_service(monkeypatch):
    _clear_settings_environment(monkeypatch)
    _set_required_github_app_environment(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "development-webhook-secret")

    settings = Settings(_env_file=None)

    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        settings.validate_runtime_configuration()


def test_production_settings_validate_complete_configuration(monkeypatch):
    _clear_settings_environment(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("GITHUB_APP_ID", "123")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "secrets/test-private-key.pem")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "development-webhook-secret")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://reviewer:password@localhost:5432/reviews"
    )
    monkeypatch.setenv("MODEL_SERVICE_URL", "https://model-service.example.internal")

    settings = Settings(_env_file=None)

    settings.validate_runtime_configuration()
    credentials = settings.github_app_credentials()

    assert credentials.app_id == 123
    assert credentials.private_key_path == (
        Path(__file__).parents[1] / "secrets" / "test-private-key.pem"
    )


def test_production_settings_require_database_and_model_service(monkeypatch):
    _clear_settings_environment(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("GITHUB_APP_ID", "123")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "secrets/test-private-key.pem")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "development-webhook-secret")

    settings = Settings(_env_file=None)

    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        settings.validate_runtime_configuration()

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://reviewer:password@localhost:5432/reviews"
    )
    settings = Settings(_env_file=None)

    with pytest.raises(ConfigurationError, match="MODEL_SERVICE_URL"):
        settings.validate_runtime_configuration()
