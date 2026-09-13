"""The single place environment variables are read for this service.

Every other module receives configuration through a Settings instance (or
its fields) rather than calling os.environ directly.
"""

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    PostgresDsn,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_CONFIG = SettingsConfigDict(
    # .env.local is loaded second, so any value it sets (e.g. a locally
    # pasted GEMINI_API_KEY) overrides .env without ever needing to be
    # committed - see .gitignore, which excludes .env.local specifically.
    env_file=(_SERVICE_ROOT / ".env", _SERVICE_ROOT / ".env.local"),
    env_file_encoding="utf-8",
    extra="ignore",
    # llm_api_key's validation_alias ("GEMINI_API_KEY", for the env var name
    # the task requires) would otherwise also become the ONLY accepted
    # keyword when constructing Settings directly - populate_by_name lets
    # tests (and any other direct construction) keep using the field's own
    # name, llm_api_key=..., same as every other field.
    populate_by_name=True,
)

# The default points at a local Ollama instance for development. Production
# must override MODEL_SERVICE_URL with a real endpoint - see
# validate_runtime_configuration, which rejects this default in production.
DEFAULT_MODEL_SERVICE_URL = "http://localhost:11434"


class ConfigurationError(ValueError):
    """Raised when settings required for an enabled runtime mode are missing."""


class GitHubAppCredentials(BaseModel):
    """The GitHub App credentials used for Webhook and API authentication."""

    app_id: PositiveInt
    private_key_path: Path
    webhook_secret: SecretStr


class Settings(BaseSettings):
    """Typed environment configuration for the Review Agent service."""

    model_config = _SETTINGS_CONFIG

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: str = "0.0.0.0"
    port: PositiveInt = 8000

    github_token: SecretStr | None = None
    github_app_id: PositiveInt
    github_app_private_key_path: Path
    github_app_installation_id: PositiveInt | None = None
    github_webhook_secret: SecretStr

    database_url: PostgresDsn | None = None
    # "ollama" (default, Version A / production) sends every reviewer call to
    # model_service_url as today. "gemini" routes the exact same call_model()
    # call through the Gemini API instead - see model_service_tool.py. Every
    # node, the parser, validator, and dedup are unaware of which backend is
    # selected; they only ever see a ModelReviewOutput.
    model_backend: Literal["ollama", "gemini"] = "ollama"
    model_service_url: str = DEFAULT_MODEL_SERVICE_URL
    model_service_model_name: str = "qwen2.5:7b"
    model_service_timeout_seconds: PositiveInt = 180
    # Gemini-backend settings only - unused when model_backend is "ollama".
    # Reads the GEMINI_API_KEY env var name specifically (matching
    # .env.local's documented key), never logged or printed - see
    # model_service_tool.py's _call_gemini().
    llm_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    llm_model: str = "gemini-3.6-flash"
    # Passed as Ollama's per-request "keep_alive" on every /api/generate call,
    # overriding Ollama's own 5-minute-idle-unload default - see
    # RUNTIME-OPTIMIZATION-OPPORTUNITIES.md item 6. Keeps the model resident
    # in memory across the gaps between review triggers so a run never pays
    # a cold-start reload, regardless of how long those gaps are.
    model_service_keep_alive: str = "30m"
    # Independently configurable in case the embedding model ever moves to a
    # different server than text generation - falls back to
    # model_service_url when unset, see resolved_embedding_service_url().
    embedding_service_url: str | None = None
    embedding_service_timeout_seconds: PositiveInt = 180

    git_command_timeout_seconds: PositiveInt = 60
    max_repository_scan_files: PositiveInt = 10_000
    max_repository_file_bytes: PositiveInt = 1_000_000
    max_repository_search_results: PositiveInt = 500
    # Both budgets are sized to fit under max_model_input_tokens together with
    # room for system instructions and documentation context: at
    # model_service_tool.py's calibrated ~2.5 chars/token, 45_000 + 15_000
    # characters is ~24,000 tokens, leaving ~4,000 of the 28_000-token budget
    # for everything else in the prompt. Previously 200_000 and 50_000 - a
    # diff alone could exceed the entire token budget and be rejected outright
    # by call_model rather than fitting, since nothing kept the two in sync.
    max_repository_context_characters: PositiveInt = 15_000
    max_diff_characters: PositiveInt = 45_000
    # Sent to the Model Service as num_ctx on every generate call. Ollama
    # otherwise defaults to 2048 and silently truncates the beginning of a
    # prompt, including reviewer instructions.
    model_context_tokens: PositiveInt = 32_768
    # Sent as Ollama's "seed" alongside temperature=0 (hardcoded in
    # model_service_tool.py) so identical prompts produce identical output -
    # a review run is not reproducible otherwise.
    model_service_seed: int = 42
    # Keep the input budget below the model context window so responses have
    # room and oversized prompts are rejected rather than silently truncated.
    max_model_input_tokens: PositiveInt = 28_000
    # Comma-separated glob patterns (e.g. "all_classes_code.txt,*.lock") for
    # files to drop entirely from a PR's diff before it ever reaches a
    # reviewer prompt. Which files are "generated noise" for a given target
    # repository is a deployment property, not something this service can
    # know in general, so this defaults to empty - see excluded_diff_path_list().
    excluded_diff_paths: str = ""

    def excluded_diff_path_list(self) -> tuple[str, ...]:
        """Parse excluded_diff_paths into individual glob patterns."""

        return tuple(
            pattern.strip() for pattern in self.excluded_diff_paths.split(",") if pattern.strip()
        )
    # How many reviewer calls may be in flight against the Model Service at
    # once. The graph fans all six reviewers out in parallel, which is the
    # right shape and is not changed by this - the limit applies only to the
    # HTTP call itself, so prompt building, parsing and validation still
    # overlap. Each in-flight request needs its own KV cache sized for
    # model_context_tokens, and the current Ollama box has no room for six.
    # Measured against it with six concurrent 50,000-character prompts: two
    # returned (313s and 466s) and the server closed the other four itself
    # after 684s. Two is what it actually served, so three is the first value
    # worth trying above that. Raise this toward six once the Model Service is
    # the shared GPU-backed one, where concurrent sequences are batched and
    # the parallelism is a real gain rather than a queue.
    max_concurrent_model_calls: PositiveInt = 3

    @field_validator("github_app_private_key_path")
    @classmethod
    def resolve_private_key_path(cls, value: Path) -> Path:
        """Resolve a relative private-key path from the service root."""

        if value.is_absolute():
            return value
        return _SERVICE_ROOT / value

    def resolved_embedding_service_url(self) -> str:
        """Return embedding_service_url if set, otherwise fall back to model_service_url.

        Both are Ollama-compatible endpoints today; kept as separate
        settings only so the embedding model can move to a different server
        later without also having to reconfigure text generation.
        """

        return self.embedding_service_url or self.model_service_url

    def github_app_credentials(self) -> GitHubAppCredentials:
        """Return the required GitHub App credentials as one typed object."""
        return GitHubAppCredentials(
            app_id=self.github_app_id,
            private_key_path=self.github_app_private_key_path,
            webhook_secret=self.github_webhook_secret,
        )

    def validate_runtime_configuration(self) -> None:
        """Validate settings that are mandatory when the service runs in production."""
        if self.environment != "production":
            return

        self.github_app_credentials()

        required_fields = {
            "DATABASE_URL": self.database_url,
        }
        missing = [name for name, value in required_fields.items() if value is None]
        if missing:
            raise ConfigurationError(
                "Missing production settings: " + ", ".join(missing)
            )

        if self.model_service_url == DEFAULT_MODEL_SERVICE_URL:
            raise ConfigurationError(
                "MODEL_SERVICE_URL is still the local Ollama default - "
                "set it to the real Model Service endpoint in production."
            )


class ManualRunSettings(BaseSettings):
    """Minimal local configuration for the manual GitHub PR verification script."""

    model_config = _SETTINGS_CONFIG

    github_token: SecretStr | None = None
    github_app_id: PositiveInt | None = None
    github_app_private_key_path: Path | None = None
    github_app_installation_id: PositiveInt | None = None

    @field_validator("github_app_private_key_path")
    @classmethod
    def resolve_optional_private_key_path(cls, value: Path | None) -> Path | None:
        """Resolve the App key path when the optional App flow is configured."""

        if value is None or value.is_absolute():
            return value
        return _SERVICE_ROOT / value
