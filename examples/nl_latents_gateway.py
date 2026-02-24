from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from llm_pool import LlmClient, LlmRequest, Message, PostgresRepository, StorageConfig


class NlLatentsQueryInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    prompt: str
    task_id: str
    pool_name: str
    run_id: str | None = None


class NlLatentsGateway:
    def __init__(self, *, repository: PostgresRepository, client: LlmClient) -> None:
        self._repository = repository
        self._client = client

    @classmethod
    def from_dsn(cls, dsn: str) -> NlLatentsGateway:
        repository = PostgresRepository(StorageConfig(dsn=dsn))
        client = LlmClient(repository=repository)
        return cls(repository=repository, client=client)

    def query(self, query_input: NlLatentsQueryInput) -> str:
        request = LlmRequest(
            provider=query_input.provider,
            model=query_input.model,
            messages=[Message(role="user", content=query_input.prompt)],
            metadata={
                "consumer": "nl_latents",
                "task_id": query_input.task_id,
                "latent_pool": query_input.pool_name,
            },
        )
        response = self._client.query(request, run_id=query_input.run_id)
        return response.text

    def close(self) -> None:
        close_client = getattr(self._client, "close", None)
        if callable(close_client):
            close_client()
        self._repository.close()
