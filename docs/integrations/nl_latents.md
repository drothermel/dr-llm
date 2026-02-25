# Integration Example: nl_latents

## Goal
Use `dr-llm` for model calls + recording while keeping latent/task semantics in `nl_latents`.

## Pattern
1. `nl_latents` builds domain prompt/messages.
2. `nl_latents` calls shared gateway backed by `LlmClient`.
3. `nl_latents` stores domain-specific outputs in its own tables/files.
4. `dr-llm` stores canonical call/session/tool traces.

## Minimal shape
```python
request = LlmRequest(
    provider=provider,
    model=model,
    messages=messages,
    metadata={
        "consumer": "nl_latents",
        "task_id": task_id,
        "latent_pool": pool_name,
    },
)
response = llm_client.query(request, run_id=run_id)
```

## Notes
1. Keep `nl_latents` scoring/ranking logic outside `dr-llm`.
2. Use `run_id` to group experiment cohorts.
3. Use session APIs for multistep/tooling workflows.
