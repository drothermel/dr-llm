# Goal

`dr-llm` currently centers large-scale model interaction workflows around
typed, Postgres-backed pools. A pool is both an experiment boundary and a
storage/coordination unit: it has its own sample and lease tables, key columns,
request payloads, response payloads, completion state, and progress queries.
This works well for bounded experiments, but it couples ingestion, storage
layout, worker coordination, and later analysis to the physical pool design.

The next architecture should separate those concerns. Instead of treating each
pool as a separate storage object, the system should collect model-interaction
facts into one aggregate corpus and expose pool-like distinctions through
projections, tags, assertions, and queryable metadata. Experiments, runs,
prompt families, datasets, model configurations, sampling settings, output
types, and derived equivalence classes should be represented as metadata over
shared records rather than as separate physical tables.

The high-level design has three stages:

1. A streaming-system style event log records durable ingestion events.
2. An immutable artifact projection stores large low-contact payloads.
3. A metadata projection exposes high-contact facts for filtering,
   inspection, coordination, and analysis.

The event log is the source of truth. Artifact and metadata stores are
projections derived from that log. This should make raw collection durable
while allowing storage formats, metadata schemas, indexing strategies, graph
views, and analysis-specific projections to evolve.

The target workload includes many parallel workers producing heterogeneous
records. A record may contain structured metadata, prompt and request payloads,
model responses, system metrics, logs, errors, generated code, validation
results, and other variable-length artifacts. The architecture should support
high-throughput local collection while preserving a path to remote cataloging,
object storage, and richer graph-style analysis.
