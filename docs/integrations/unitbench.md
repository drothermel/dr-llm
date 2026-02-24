# Integration Example: unitbench

## Goal
Keep UnitBench case generation/scoring in `unitbench`, while using `llm-pool` for provider abstraction, recording, and session/tool runtime.

## Pattern
1. `unitbench` assembles benchmark case inputs.
2. `unitbench` calls shared `llm-pool` client/session APIs.
3. `unitbench` computes benchmark metrics from normalized response fields.
4. `llm-pool` remains domain-neutral.

## Minimal shape
```python
request = LlmRequest(
    provider=provider,
    model=model,
    messages=[Message(role="user", content=case_prompt)],
    metadata={
        "consumer": "unitbench",
        "suite": suite_name,
        "case_id": case_id,
    },
)
response = llm_client.query(request, run_id=run_id)
```

## Notes
1. Persist benchmark scores in `unitbench` storage.
2. Use model catalog APIs for cost-aware model selection.
3. Use transcript logs for debugging parse/provider failures.
