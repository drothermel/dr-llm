# Testing

## Review Skill Loading

When reviewing a chunk of code, load the review skills that match the scope
before giving feedback. Prefer a small, relevant set over every installed
skill.

For general structural review, load:

- `danielle-code-quality` for Danielle's review, implementation, refactoring,
  and code-quality preferences.
- `python-design-patterns` when looking for duplicated flows, misplaced
  abstractions, long functions, or split-out opportunities.

For Python-specific code, also load:

- `python-type-safety` when reviewing public APIs, Pydantic models, typed
  protocols, or `ty` failures.
- `python-testing-patterns` when reviewing tests, fixtures, integration
  boundaries, or missing regression coverage.
- `python-error-handling` when reviewing validation, CLI exits, partial
  failures, persistence errors, or external-service boundaries.
- `async-python-patterns` when reviewing async projectors, workers, clients,
  or concurrent/background flows.
- `python-configuration` when reviewing environment parsing, settings models,
  or config defaults.
- `python-resilience` when reviewing retries, timeouts, transient failures, or
  long-running services.

Use the specialized skills only when the code touches that area:

- `marimo-pair` is required before reading, editing, verifying, or reviewing
  notebooks under `nbs/`.
- `web-design-guidelines`, `frontend-design`, or React-specific skills apply
  only to UI/frontend review.
- GitHub plugin skills such as `github:gh-address-comments`,
  `github:gh-fix-ci`, and `github:github` apply when reviewing PR comments,
  GitHub Actions failures, or repository metadata through GitHub.

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

## Streaming Log Worker Demo

Run:

```bash
uv run python scripts/demo-streaming-log-worker.py
```

The demo uses live NATS JetStream and a real provider. A passing run must show:

- A selected provider/model and submitted `work_id`.
- Lifecycle verification for:
  - `work_submitted`
  - `attempt_started`
  - `provider_request_prepared`
  - `provider_response_received`
  - `attempt_succeeded`
  - `work_completed`
  - `producer_started`
  - `producer_stopped`
- `work_completed` status `succeeded` for the selected work.
- Verified payload references.
- `Response payload verified` with a non-empty text preview.

Provider billing, quota, credential, or runtime failures should either fall
back to another provider or make the demo fail. A clean failed lifecycle is not
a successful worker demo.

## Artifact Projection Demo

Run:

```bash
uv run python scripts/demo-artifact-projection.py
```

The demo uses live NATS JetStream and synthetic artifact-bearing payload refs.
A passing run must show:

- One published `provider_response_received` event with duplicate
  `response_json` payload refs.
- The artifact projector processed exactly one event.
- Artifact verification reports one finalized artifact.
- Open artifact references are `0`.
- Projection errors are `0`.
- The projection checkpoint points at the published event.
- JSON readback matches the expected synthetic payload.

This demo intentionally does not call a live provider and does not verify
metadata projection.

## Streaming Log Pool Import Demo

Run:

```bash
uv run python scripts/demo-streaming-log-pool-import.py \
  --dsn <postgres-dsn> \
  --pool-name <existing-pool>
```

The demo uses an existing Postgres-backed pool and live NATS JetStream. A
passing run must show:

- Source pool counts for the requested pool.
- Imported sample count matching the expected source/sample-limit count.
- Exactly one `pool_import_started` event.
- Exactly one `pool_sample_imported` event per imported sample.
- Exactly one `pool_import_completed` event.
- No `pool_import_failed` events.
- Replayed import event IDs matching the importer result.
- Sample events include non-empty `sample_id` and `row_state_hash`.
- Sample events include `pool_schema`, `request_json`, and `metadata_json`
  payload refs.
- Payload references are verified for hash and size.

This demo verifies pool snapshot import into the streaming log. It does not
verify live provider execution, artifact projection, or metadata projection.
