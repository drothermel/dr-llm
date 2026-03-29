from dr_llm.providers.headless.claude import (
    ANTHROPIC_BASE_URL_ENV,
    ClaudeHeadlessAdapter,
)


MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"
KIMI_CODING_BASE_URL = "https://api.kimi.com/coding/"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"
KIMI_API_KEY_ENV = "KIMI_API_KEY"


class ClaudeHeadlessMiniMaxAdapter(ClaudeHeadlessAdapter):
    def __init__(self, command: list[str] | None = None) -> None:
        super().__init__(
            command=command,
            name="claude-code-minimax",
            env_overrides={ANTHROPIC_BASE_URL_ENV: MINIMAX_ANTHROPIC_BASE_URL},
            api_key_env=MINIMAX_API_KEY_ENV,
        )


class ClaudeHeadlessKimiAdapter(ClaudeHeadlessAdapter):
    def __init__(self, command: list[str] | None = None) -> None:
        super().__init__(
            command=command,
            name="claude-code-kimi",
            env_overrides={ANTHROPIC_BASE_URL_ENV: KIMI_CODING_BASE_URL},
            api_key_env=KIMI_API_KEY_ENV,
        )
