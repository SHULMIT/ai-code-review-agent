"""Manual verification script: call the real local Ollama instance.

Not part of the automated test suite - exercises model_service_tool.call_model
against a real, running Ollama server to confirm the local setup works
end to end. Requires Ollama running locally with the configured model pulled.
"""

import asyncio

from review_agent.config import Settings
from review_agent.tools.model_service_tool import ModelServiceError, call_model


async def main() -> None:
    settings = Settings(
        _env_file=None,
        github_app_id=1,
        github_app_private_key_path="secrets/unused-for-this-script.pem",
        github_webhook_secret="unused-for-this-script",
    )

    prompt = (
        "You are reviewing a one-line code change. Reply in a single short "
        "sentence: `print('hello')` was added to a Python file. Is this safe?"
    )

    print("model_service_url:", settings.model_service_url)
    print("model_service_model_name:", settings.model_service_model_name)
    print()

    try:
        result = await call_model(prompt, settings)
    except ModelServiceError as exc:
        raise RuntimeError(f"Model Service call failed: {exc}") from exc

    print("raw_text:", result.raw_text)


if __name__ == "__main__":
    asyncio.run(main())
