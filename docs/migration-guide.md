# Migration Guide: Direct Providers to llm-pool

## 1) Replace provider SDK calls
Before:
1. direct OpenAI/Anthropic/Google client calls in app code

After:
1. construct `LlmRequest`
2. call `LlmClient.query(...)`
3. consume normalized `LlmResponse`

## 2) Enable canonical recording
1. Configure `LLM_POOL_DATABASE_URL`
2. instantiate `PostgresRepository`
3. pass repository to `LlmClient`

## 3) Move multistep/tool loops into SessionClient
1. Replace custom loop state with `SessionClient.start/step/resume/cancel`
2. Register tools with `ToolRegistry`
3. Run `llm-pool tool worker run ...` for queued brokered tools

## 4) Preserve compatibility
1. Keep benchmark/task scoring in consumer repo.
2. Keep only shared llm/storage/session runtime concerns in `llm-pool`.
3. Store consumer metadata via request/session metadata fields.

## 5) Validation
1. replay sample sessions and compare outputs
2. verify call counts, failure handling, and idempotency under retries
3. run `llm-pool run benchmark` before cutover
