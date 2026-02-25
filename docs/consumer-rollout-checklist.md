# Consumer Rollout Checklist

Use this checklist when rolling `llm-pool` into consumer repos (`nl_latents`, `unitbench`, or new repos).

## 1) Prerequisites
1. Pin a released `llm-pool` version/commit in the consumer repo.
2. Set required env vars:
- `LLM_POOL_DATABASE_URL`
- provider keys used by that consumer workflow
3. Confirm DB connectivity from the consumer runtime environment.

## 2) Integration Scope
1. Replace direct provider SDK calls with `LlmClient.query(...)` for one narrow path first.
2. Pass consumer metadata (`consumer`, `suite/task identifiers`, `experiment/run context`) in request/session metadata.
3. Keep domain logic in the consumer repo; do not add it to `llm-pool`.

## 3) Session/Tool Adoption (if applicable)
1. Switch multistep flows to `SessionClient.start/step/resume/cancel`.
2. Register consumer tools via `ToolRegistry` and run workers using `llm-pool tool worker run`.
3. Validate retries, leases, and dead-letter behavior under forced tool failures.

## 4) Validation Gates
1. Functional parity:
- sampled prompts/cases produce equivalent outputs vs previous implementation
2. Storage parity:
- expected runs/calls/sessions/events are persisted and queryable
3. Replay parity:
- sampled sessions replay correctly from stored events
4. Error-path visibility:
- provider/parse failures appear in generation transcript logs

## 5) Performance Gate
1. Run benchmark in staging-like environment:
```bash
llm-pool run benchmark --workers 128 --total-operations 200
```
2. Capture baseline: throughput, p50/p95 latency, failure counts.
3. Block rollout if throughput regresses materially or failure rate increases.

## 6) Cutover Strategy
1. Deploy behind a feature flag/call-path switch in the consumer repo.
2. Start with low traffic cohort.
3. Expand gradually while monitoring:
- error rates
- tool queue depth/dead-letter counts
- DB saturation and latency

## 7) Rollback Strategy
1. Keep old provider path available until rollout stabilizes.
2. Define explicit rollback triggers (error thresholds, latency SLO breach, dead-letter spike).
3. Roll back by switching the feature flag and preserving collected diagnostics.

## 8) Completion Criteria
1. Consumer path fully uses `llm-pool` APIs for call/session/tool/storage concerns.
2. Benchmarks and validation gates pass.
3. On-call/runbook links are added in the consumer repo.
