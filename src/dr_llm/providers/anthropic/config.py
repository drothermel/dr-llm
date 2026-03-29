from dr_llm.providers.api_provider_config import APIProviderConfig


class AnthropicConfig(APIProviderConfig):
    name: str = "anthropic"
    base_url: str = "https://api.anthropic.com/v1/messages"
    api_key_env: str = "ANTHROPIC_API_KEY"
    anthropic_version: str = "2023-06-01"
