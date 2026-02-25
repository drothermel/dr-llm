# Compatibility Contract

`dr-llm` is a shared infrastructure package. Domain logic must remain outside this repo.

## Stable Contracts (Post-M1)
1. Core call API:
- `LlmClient.query(request: LlmRequest, ...) -> LlmResponse`
2. Session API:
- `SessionClient.start_session(...)`
- `SessionClient.step_session(...)`
- `SessionClient.resume_session(...)`
- `SessionClient.cancel_session(...)`
3. Tool worker API:
- `run_tool_worker(...)`
4. Storage interface:
- `PostgresRepository` public methods and canonical tables.

## Additive Policy
1. New fields must be additive and optional by default.
2. Existing enums may gain values; consumers must handle unknown values defensively.
3. Existing JSON payload shapes may include additional keys.

## Breaking Change Policy
1. Any removal/rename of public methods or non-additive schema changes requires:
- migration SQL and rollback notes
- release note with impacted consumer paths
- downstream compatibility validation (`nl_latents`, `unitbench`)

## Consumer Guidance
1. Depend on public types in `dr_llm.types`.
2. Do not read private storage modules (`_runtime.py`, `_runs_calls_store.py`) directly.
3. Use repository/query APIs instead of handwritten SQL in consuming repos.
