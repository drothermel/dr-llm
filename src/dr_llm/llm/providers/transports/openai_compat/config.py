from pydantic import ConfigDict, field_validator

from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.concepts.model_family import ModelFamily
from dr_llm.llm.providers.transports.api_config import APIProviderConfig


class OpenAICompatConfig(APIProviderConfig):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    api_key_env: str = ApiKeyNames.OPENAI
    chat_path: str = "/chat/completions"
    max_completion_token_model_families: tuple[ModelFamily, ...] = ()

    @field_validator("chat_path")
    @classmethod
    def _ensure_leading_slash(cls, v: str) -> str:
        if not v.startswith("/"):
            return f"/{v}"
        return v
