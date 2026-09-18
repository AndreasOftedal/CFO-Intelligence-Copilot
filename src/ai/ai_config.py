from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AIConfig:
    """
    Central configuration for the CFO Intelligence Copilot AI layer.

    Core financial calculations are performed outside the LLM.
    The model is used only for interpretation and management commentary
    based on deterministic facts and approved evidence.
    """

    model: str
    reasoning_effort: str
    max_output_tokens: int
    store_responses: bool
    timeout_seconds: float
    max_retries: int


def load_ai_config() -> AIConfig:
    """
    Load and validate AI configuration.

    The OpenAI API key must be provided through the
    OPENAI_API_KEY environment variable. The secret key is never
    stored in source code or configuration files.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set the environment variable before running the AI layer."
        )

    return AIConfig(
        model=os.getenv("CFO_AI_MODEL", "gpt-5.6-luna"),
        reasoning_effort=os.getenv("CFO_AI_REASONING_EFFORT", "low"),
        max_output_tokens=int(
            os.getenv("CFO_AI_MAX_OUTPUT_TOKENS", "3000")
        ),
        store_responses=False,
        timeout_seconds=float(
            os.getenv("CFO_AI_TIMEOUT_SECONDS", "45")
        ),
        max_retries=int(
            os.getenv("CFO_AI_MAX_RETRIES", "0")
        ),
    )


if __name__ == "__main__":
    config = load_ai_config()

    print("AI configuration loaded successfully")
    print(f"Model: {config.model}")
    print(f"Reasoning effort: {config.reasoning_effort}")
    print(f"Max output tokens: {config.max_output_tokens}")
    print(f"Store responses: {config.store_responses}")
    print(f"Timeout: {config.timeout_seconds} seconds")
    print(f"Automatic retries: {config.max_retries}")