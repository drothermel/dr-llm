from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from llm_pool import LlmClient, LlmRequest, Message, PostgresRepository, StorageConfig


class UnitBenchCaseInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    suite: str
    case_id: str
    prompt: str
    run_id: str | None = None


class UnitBenchGateway:
    def __init__(self, *, repository: PostgresRepository, client: LlmClient) -> None:
        self._repository = repository
        self._client = client

    @classmethod
    def from_dsn(cls, dsn: str) -> UnitBenchGateway:
        repository = PostgresRepository(StorageConfig(dsn=dsn))
        client = LlmClient(repository=repository)
        return cls(repository=repository, client=client)

    def run_case(self, case_input: UnitBenchCaseInput) -> str:
        request = LlmRequest(
            provider=case_input.provider,
            model=case_input.model,
            messages=[Message(role="user", content=case_input.prompt)],
            metadata={
                "consumer": "unitbench",
                "suite": case_input.suite,
                "case_id": case_input.case_id,
            },
        )
        response = self._client.query(request, run_id=case_input.run_id)
        return response.text

    def close(self) -> None:
        close_client = getattr(self._client, "close", None)
        if callable(close_client):
            close_client()
        self._repository.close()
