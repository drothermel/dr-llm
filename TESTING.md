# Testing

## Metadata Projection End-to-End Demo

Run:

```bash
uv run python scripts/demo-metadata-projection-e2e.py
```

The demo uses live Postgres, NATS JetStream, artifact storage, metadata
projection, and a real provider. A passing run must show all of the following:

- A provider candidate is tried and then reported as a successful lifecycle.
- The streaming event sequence includes:
  - `work_submitted`
  - `attempt_started`
  - `provider_request_prepared`
  - `provider_response_received`
  - `attempt_succeeded`
  - `work_completed`
- The output must not report `attempt_failed` for the selected work.
- The `work_completed` event must have status `succeeded`; a clean failure
  lifecycle is not a successful demo.
- The response artifact check must print `Response artifact verified` with an
  artifact ID and a non-empty text preview.
- Metadata verification must report `errors=0`.
- Replay idempotency and rebuild determinism must both be verified.

If the first provider candidate has billing, quota, credential, or model
availability problems, the demo should print that candidate's failure details
and try the next available provider. If no provider produces a successful
response, the demo must fail rather than treating projection of failure events
as success.
