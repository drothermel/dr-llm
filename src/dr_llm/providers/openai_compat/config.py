from dr_llm.providers.api_provider_config import APIProviderConfig


class OpenAICompatConfig(APIProviderConfig):
    api_key_env: str = "OPENAI_API_KEY"
    chat_path: str = "/chat/completions"
