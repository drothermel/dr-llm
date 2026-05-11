from pydantic import field_validator

from dr_llm.llm.providers.api_config import APIProviderConfig


class OpenAICompatConfig(APIProviderConfig):
    api_key_env: str = "OPENAI_API_KEY"
    chat_path: str = "/chat/completions"

    @field_validator("chat_path")
    @classmethod
    def _ensure_leading_slash(cls, v: str) -> str:
        if not v.startswith("/"):
            return f"/{v}"
        return v
