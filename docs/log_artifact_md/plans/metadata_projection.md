# Metadata Projection Implementation Plan

## Purpose

Build the v1 metadata projection layer on top of the streaming log and
artifact projection work. The metadata projection is the high-contact catalog
for filtering, inspection, coordination, provenance, and analysis. It should
store queryable facts in Postgres while referencing low-contact payload bytes
through finalized artifact projection references.

This implementation starts from a new branch off `artifact_projection`:

```text
git worktree add .worktrees/metadata_projection \
  -b metadata_projection artifact_projection
```

The first implementation step is this plan document. Pause for review after
this document is written and before adding dependencies, schema code, projector
code, CLI commands, migrations, or tests.

## Upstream Contracts

The metadata projection consumes the streaming log and artifact projection
contracts that already exist on the `artifact_projection` branch.

- Projection input is `DRLLM_EVENTS`. Do not read pool tables, JSONL generation
  logs, or `DRLLM_WORK` for projection correctness.
- Event payloads are validated with `dr_llm.streaming_log.events.EventEnvelope`.
  The metadata projector should preserve `event_id`, `event_type`,
  `schema_version`, `occurred_at`, `producer`, `idempotency_key`, `run_id`,
  `work_id`, `attempt_id`, `causation_id`, `correlation_id`, `source`, and
  event `metadata`.
- Large payloads are represented by `PayloadRef` values. Keep payload reference
  facts queryable, but do not dereference `DRLLM_PAYLOADS` directly in the
  metadata projection.
- Artifact payloads are represented by
  `dr_llm.artifact_projection.models.ArtifactReference`. The metadata
  projection should load finalized artifact references from
  `ArtifactIndex.list_references()` or finalized manifest JSONL files and
  attach them to the source event, work, attempt, run, and payload role. Do not
  consume `open_artifact_references`; open references are writer recovery state,
  not reader-visible metadata input.
- Projection delivery is at least once. Metadata writes must be idempotent
  across redelivery and replay.

## V1 Data Model

Use Postgres as the primary metadata projection store. Model the catalog as a
small graph/fact store using entities, assertions, and assertion roles.

Tables:

- `metadata_entities`
  - `entity_id text primary key`
  - `entity_type text not null`
  - `identity_key text not null`
  - `content_hash text`
  - `display_name text`
  - `metadata_json jsonb not null default '{}'::jsonb`
  - `created_at timestamptz not null default now()`
  - unique `(entity_type, identity_key)`
- `metadata_assertions`
  - `assertion_id text primary key`
  - `assertion_type text not null`
  - `projection_version text not null`
  - `source_event_id text not null`
  - `source_event_type text not null`
  - `source_schema_version integer not null`
  - `source_idempotency_key text not null`
  - `occurred_at timestamptz not null`
  - `status text`
  - `metadata_json jsonb not null default '{}'::jsonb`
  - `created_at timestamptz not null default now()`
  - unique `(projection_version, assertion_type, source_idempotency_key)`
- `metadata_assertion_roles`
  - `assertion_id text not null references metadata_assertions`
  - `role_name text not null`
  - `entity_id text not null references metadata_entities`
  - primary key `(assertion_id, role_name, entity_id)`
- `metadata_projection_checkpoints`
  - `projection_version text not null`
  - `durable_consumer text not null`
  - `stream_sequence bigint not null`
  - `event_id text`
  - `updated_at timestamptz not null default now()`
  - primary key `(projection_version, durable_consumer)`
- `metadata_projection_errors`
  - `error_id bigserial primary key`
  - `projection_version text not null`
  - `source_event_id text not null`
  - `source_idempotency_key text not null`
  - `source_event_type text`
  - `error_kind text not null`
  - `message text not null`
  - `metadata_json jsonb not null default '{}'::jsonb`
  - `stream_sequence bigint`
  - `created_at timestamptz not null default now()`

Core entity types for v1:

- `run`: identity key `run_id`
- `work`: identity key `work_id`
- `attempt`: identity key `attempt_id` when present, otherwise
  `work_id:attempt`
- `producer`: identity key from producer name, version, and instance ID
- `provider`: identity key provider name
- `model`: identity key provider plus model
- `model_config`: content-addressed normalized request model/config fields
- `prompt_instance`: content-addressed prompt/request summary when inline
  fields are available
- `output_result`: content-addressed response summary when inline fields are
  available
- `artifact`: identity key `artifact_id`
- `pool`: identity key imported pool name
- `pool_sample`: identity key imported source ID, pool name, and sample ID

Core assertion types for v1:

- `pool_import_started`
- `pool_sample_imported`
- `pool_import_completed`
- `pool_import_failed`
- `work_submitted`
- `attempt_started`
- `provider_request_prepared`
- `provider_response_received`
- `attempt_succeeded`
- `attempt_failed`
- `work_retry_scheduled`
- `work_completed`
- `work_cancelled`
- `artifact_attached`
- `producer_started`
- `producer_stopped`
- `streaming_log_error`

## Event Mapping

Each accepted event creates one source assertion with matching
`assertion_type`, plus roles for entities present in the event context and
payload. Every assertion stores the original inline event payload in
`metadata_json`, after removing or summarizing fields that are represented as
entities.

Mapping defaults:

- All events create or link `producer`, `run`, `work`, and `attempt` entities
  when their identifiers are present.
- `pool_sample_imported` creates `pool`, `pool_sample`, and optional `run`
  entities. It stores `sample_id`, `sample_idx`, `key_values`,
  `finish_reason`, `attempt_count`, `completion_state`, `row_state_hash`, and
  `reconstructed` in assertion metadata.
- `work_submitted` creates `work`, optional `run`, and a `model_config` entity
  if provider/model/config fields are present in the event payload.
- `attempt_started` creates an `attempt` entity and links it to `work` and
  `run` when present.
- `provider_request_prepared` creates or links `provider`, `model`,
  `model_config`, and `prompt_instance` entities when inline fields are
  present.
- `provider_response_received` creates or links `provider`, `model`, and
  `output_result` entities from inline response summary fields.
- `attempt_succeeded`, `attempt_failed`, `work_retry_scheduled`,
  `work_completed`, and `work_cancelled` update no previous rows. They create
  new immutable assertions whose status field is copied from the event payload
  when present.
- `producer_started`, `producer_stopped`, and `streaming_log_error` are stored
  as operational assertions. They should be queryable but should not be used as
  generation-attempt facts.

Artifact attachment:

- For every finalized `ArtifactReference`, create or upsert an `artifact`
  entity with identity key `artifact_id`.
- Create an `artifact_attached` assertion keyed by
  `artifact-v1:artifact_attached:<artifact_id>`.
- Link roles: `artifact`, plus any available `run`, `work`, `attempt`,
  `source_event`, `producer`, `provider`, or `model` entities known from
  `reference.source_ref`, `reference.event_context`, and existing source
  assertions.
- Store the full `ArtifactReference` JSON in assertion metadata and keep the
  queryable fields duplicated on the `artifact` entity metadata:
  `source_ref.event_id`, `source_ref.event_type`,
  `source_ref.idempotency_key`, `source_ref.payload_role`,
  `source_ref.object_key`, `source_ref.sha256`, `source_ref.size_bytes`,
  `source_ref.content_type`, `source_ref.encoding`,
  `source_ref.compression`, `event_context.run_id`,
  `event_context.work_id`, `event_context.attempt_id`, `logical_sha256`,
  `size_bytes`, `lane`, `shard_id`, and `shard_uri`.

## Identity and Hashing

Use `dr_llm.streaming_log.serialization.canonical_json_bytes` for all
metadata projection content hashes. Add metadata projection helpers rather than
reimplementing ad hoc JSON serialization.

Identifier rules:

- Entity IDs use prefix `ent_` plus the SHA-256 of canonical JSON containing
  `entity_type` and `identity_key`.
- Assertion IDs use prefix `asrt_` plus the SHA-256 of canonical JSON
  containing `projection_version`, `assertion_type`, and
  `source_idempotency_key`.
- Content hashes use SHA-256 of normalized semantic content, not database row
  JSON. Keep `content_hash` nullable for identity-only entities such as runs.
- The v1 projection version is `metadata-v1`.
- The default durable consumer is `drllm_metadata_projection_v1`.

Duplicate handling:

- If an entity with the same `(entity_type, identity_key)` already exists and
  has identical stable fields, projection is a no-op.
- If an entity with the same `(entity_type, identity_key)` has conflicting
  stable fields, record a `duplicate_entity_conflict` projection error.
- If an assertion with the same unique logical key already exists and has
  identical stable fields and roles, projection is a no-op.
- If an assertion conflicts, record a `duplicate_assertion_conflict` projection
  error.
- Redelivery of the same event must not create duplicate rows.

## Implementation Package

Add `src/dr_llm/metadata_projection/` only after this plan is reviewed.

Recommended modules:

```text
src/dr_llm/metadata_projection/
  __init__.py
  config.py
  models.py
  identity.py
  schema.py
  store.py
  mapper.py
  artifact_links.py
  projector.py
  cli.py
```

Responsibilities:

- `config.py`: Pydantic settings with env prefix
  `DR_LLM_METADATA_PROJECTION_`. Include database DSN, projection version,
  durable consumer, fetch batch size, artifact index path, and application
  name.
- `models.py`: Pydantic models for entities, assertions, roles, checkpoints,
  errors, and projection summaries. Use `extra="forbid"` for durable models.
- `identity.py`: canonical hash helpers, entity ID helpers, assertion ID
  helpers, and normalized content extraction.
- `schema.py`: SQLAlchemy table definitions and idempotent schema creation.
  Keep runtime-owned projection tables separate from existing pool tables.
- `store.py`: transactional Postgres store for upserting entities,
  assertions, roles, checkpoints, and errors.
- `mapper.py`: event-to-fact mapper from `EventEnvelope` to projection write
  plans. It should contain no database or NATS code.
- `artifact_links.py`: load `ArtifactReference` values from the artifact
  projection index or finalized manifests and create artifact attachment write
  plans. Use `ArtifactIndex.list_references()` for finalized index rows and
  `load_manifest_references()` for manifest-based rebuilds.
- `projector.py`: consume `DRLLM_EVENTS`, apply mappings transactionally,
  checkpoint after durable writes, and acknowledge messages only after commit.
- `cli.py`: Typer commands for initialization, projection runs, inspection,
  verification, and rebuild.

Do not add a new database dependency. Reuse existing SQLAlchemy, psycopg,
Pydantic, and NATS dependencies.

## Implementation Discipline

Follow the `clean-code` and `karpathy-guidelines` skills explicitly while
implementing this layer.

- Keep changes surgical. Add only the metadata projection package, CLI
  registration, and tests needed for this plan. Do not refactor pool,
  streaming-log, or artifact-projection internals unless a failing test proves
  the metadata projection cannot bind to their current public APIs.
- Prefer the simplest working shape. Do not add generic graph engines,
  repository frameworks, plugin systems, background schedulers, or additional
  configuration knobs in v1.
- Use intention-revealing names. Avoid vague class names such as `Manager`,
  `Handler`, `Data`, or `Info` when a domain name like `MetadataProjector`,
  `MetadataStore`, `EventFactMapper`, or `ArtifactAttachmentPlan` is more
  precise.
- Keep modules at one abstraction level. `mapper.py` turns events into write
  plans; `store.py` owns database transactions; `projector.py` owns delivery,
  checkpoint, and acknowledgment flow. Do not mix SQL, NATS acknowledgments,
  and event interpretation in the same function.
- Keep functions small and single-purpose. If a mapper or store function needs
  more than two or three inputs, group related values in Pydantic request or
  plan models rather than passing long argument lists.
- Prefer explicit Pydantic models over dictionaries at module boundaries.
  Raw event payload dictionaries may enter the mapper, but durable write plans,
  entities, assertions, roles, checkpoints, and errors should be typed models.
- Use exceptions for impossible internal states and durable error rows for
  recoverable source/projection failures. Do not encode control flow with
  ambiguous boolean return values.
- Let code explain itself. Add comments only for non-obvious external
  constraints, such as JetStream acknowledgment ordering or artifact finalized
  versus open-reference semantics.
- State assumptions in the implementation PR or module docstring when an event
  payload shape is not yet strongly typed upstream. If an event cannot be
  mapped without guessing, record a projection error rather than inventing
  missing semantics.
- Define a verification target before each implementation slice: model
  identity, schema creation, event mapping, store idempotency, projector
  acknowledgment behavior, artifact attachment, CLI dispatch, then integration.

## CLI Surface

Register a new Typer sub-app as `dr-llm metadata-projection`.

Commands:

- `init`: create or validate metadata projection tables.
- `run`: consume `DRLLM_EVENTS`; support `--batch-size`, `--max-messages`,
  `--from-start`, and `--artifact-index-path`.
- `attach-artifacts`: read finalized artifact references from the artifact
  index and create `artifact_attached` assertions.
- `inspect`: print projection version, row counts, latest checkpoint, and
  error counts.
- `verify`: validate uniqueness, dangling roles, checkpoint presence, and
  finalized artifact references that have no matching source event assertion.
- `rebuild`: clear rebuildable metadata rows for a projection version and
  replay from the durable event stream plus finalized artifact references.

The CLI should use `DR_LLM_DATABASE_URL` as a fallback DSN when
`DR_LLM_METADATA_PROJECTION_DATABASE_DSN` is unset, matching the existing
project/pool database convention.

## Checkpointing and Recovery

The event projector and artifact attachment step have separate checkpoints.

- Event projection checkpoint key:
  `(metadata-v1, drllm_metadata_projection_v1)`.
- Artifact attachment checkpoint key:
  `(metadata-v1, drllm_metadata_artifact_attach_v1)`.

Rules:

- A message may be acknowledged only after projection writes and checkpoint or
  durable error rows are committed.
- Malformed events, unsupported event payloads, duplicate conflicts, and store
  errors that can be recorded durably should become
  `metadata_projection_errors`.
- Do not block later events on a durable projection error unless the database
  transaction itself cannot commit.
- Rebuild should be deterministic for the same source event stream and artifact
  reference set.
- Artifact attachment should only use finalized references from
  `artifact_references` or finalized shard manifests. Open references are
  ignored until the artifact projection finalizes their shard and promotes them
  into `artifact_references`.
- Imported pool facts are reconstructed snapshot facts. Do not infer missing
  lifecycle events from them.

## Testing Plan

Unit tests:

- Entity and assertion IDs are stable and change on their defining identity
  fields.
- Content hashes are stable for canonical JSON and independent of key order.
- Event mapper creates expected entities, assertions, roles, and metadata for
  each v1 event type.
- Artifact reference mapping creates an `artifact` entity and
  `artifact_attached` assertion from nested `source_ref` and `event_context`
  fields.
- Open artifact references are ignored until they are promoted to finalized
  `artifact_references`.
- Duplicate identical entities and assertions are no-ops.
- Conflicting duplicates produce projection errors.
- Malformed or unsupported event payloads produce projection errors.

Database tests:

- Schema creation is idempotent.
- Store transactions insert entities, assertions, roles, checkpoints, and
  errors atomically.
- Unique constraints reject conflicting logical facts.
- Roles cannot reference missing entities or assertions.
- `verify` catches dangling or inconsistent projection state.

Integration tests:

- With Docker NATS and Postgres, publish representative lifecycle events, run
  the metadata projector, and query the resulting graph rows.
- Replay the same events and confirm row counts do not change.
- Project an imported pool sample event and verify reconstructed provenance.
- Project finalized artifact references and verify they attach to source
  events, work, attempts, and runs.
- Rebuild from the event stream and artifact index and verify the same logical
  graph facts are present.

Quality gate for library-code changes:

1. `uv run ruff format`
2. `uv run ruff check --fix .`
3. Manually fix any remaining lint issues in the repository.
4. `uv run ty check`
5. `uv run pytest tests/ -v -m "not integration"`
6. `./scripts/run-tests-local.sh`

For this documentation-only checkpoint, inspect `git diff` and pause for
review. Do not run the full library quality gate until code is added.

## Non-Goals for V1

- Do not replace the existing pool system.
- Do not read current pool tables directly during projection.
- Do not store artifact bytes in Postgres.
- Do not query Zarr shards from metadata projection code.
- Do not implement canonicalization or equivalence assertions beyond reserving
  the graph/fact shape for future derived assertions.
- Do not add Neo4j, RDF, or Datalog projections in v1.
- Do not mutate or compact artifact projection shards.

## Review Checkpoint

Stop after this plan document is created. The next implementation pass should
begin only after review confirms:

- The entity and assertion table shape is acceptable.
- The event-to-fact mapping is sufficient for v1.
- Artifact attachment should be a separate projector step.
- The CLI surface and rebuild behavior match expected workflows.
