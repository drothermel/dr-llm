# M2b Throughput and Operations Checklist

## 1) Benchmark Baseline
Run on staging-like infrastructure before release:

```bash
llm-pool run benchmark \
  --workers 128 \
  --operations-per-worker 200 \
  --min-pool-size 8 \
  --max-pool-size 128
```

Record and compare:
1. `operations_per_second`
2. `p50_latency_ms`
3. `p95_latency_ms`
4. `failed_operations`
5. failure distribution by operation type

## 2) Database Tuning
1. Set `max_pool_size` based on DB `max_connections` budget.
2. Keep transactions short; never execute tools inside DB transactions.
3. Verify index usage for:
- session replay (`session_events(session_id, event_seq)`)
- queue claims (`tool_calls(status, created_at)`)
- idempotency (`tool_calls(idempotency_key)` unique)
4. Validate lock behavior under load (`FOR UPDATE SKIP LOCKED` claims).

## 3) Worker Reliability
1. Ensure lease values are short and renewed by active workers.
2. Confirm retries return failed claims to `pending` until dead-letter threshold.
3. Validate dead-letter volume alarms.
4. Track `attempt_count` distributions to detect systemic tool instability.

## 4) Logging and Auditability
1. Keep generation transcript logging enabled in staging/prod.
2. Verify redaction behavior for secrets.
3. Confirm parse-failure and provider-error paths produce transcript events.
4. Retain logs by size/rotation policy and centralize shipping if needed.

## 5) Partition-Ready Posture
1. Keep time-based keys (`created_at`) populated for all high-volume tables.
2. Prepare partition plan for:
- `llm_calls`
- `session_events`
- `tool_calls`
3. Establish archive/retention windows before data growth forces emergency migration.

## 6) Release Gate
Block rollout if any of the following are true:
1. Throughput regresses >10% from last accepted baseline.
2. `failed_operations` increases materially without explanation.
3. Queue retry/dead-letter behavior diverges from expected policy.
4. Replay consistency checks fail for sampled sessions.
