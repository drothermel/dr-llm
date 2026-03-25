# Integration Contract: nl_latents

## Goal
Use `dr-llm` for model calls + recording while keeping latent/task semantics in `nl_latents`.

## Pattern
1. `nl_latents` builds domain prompt/messages.
2. `nl_latents` calls shared gateway backed by `LlmClient`.
3. `nl_latents` stores domain-specific outputs in its own tables/files.
4. `dr-llm` stores canonical call/session/tool traces.

## Environment contract

1. `DR_LLM_DATABASE_URL` is required for `nl_latents` experiment run tracking.
2. `NL_LATENTS_DATABASE_URL` remains the `nl_latents` domain catalog database.

## Reasoning contract

1. `nl_latents` now resolves reasoning explicitly per stage:
   - stage override
   - shared override
   - `minimal` (default)
2. `none` is treated as explicit reasoning disable.

## Minimal shape
```python
request = LlmRequest(
    provider=provider,
    model=model,
    messages=messages,
    reasoning=ReasoningConfig(effort="minimal"),
    metadata={
        "consumer": "nl_latents",
        "run_id": run_id,
        "stage": stage,
        "task_id": task_id,
        "task_family": family,
        "budget": budget,
    },
)
response = llm_client.query(request, run_id=run_id)
```

## Notes
1. Keep `nl_latents` scoring/ranking logic outside `dr-llm`.
2. Use `run_id` to group experiment cohorts and artifacts.
3. Use session APIs for multistep/tooling workflows.
