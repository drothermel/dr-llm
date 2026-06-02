# Artifact Store

The artifact store is an immutable projection for large payloads. It stores
the low-contact data that should not live directly in the metadata database:
raw requests, raw responses, reasoning traces, logs, generated code, metrics
payloads, error details, validation artifacts, and other variable-length data.

The current design target is Zarr v3 as a modern HDF5-like storage substrate.
The goal is not to use the artifact store as a query engine. Its job is
efficient physical storage and retrieval after metadata queries have identified
which records are relevant.

The artifact projection consumes events from the streaming log and writes
finalized immutable shards. Those shards can be verified, checksummed, and
referenced from the metadata projection. Since the shards are immutable, the
system avoids many multi-writer mutable blob-store problems. If the physical
layout changes, the artifact projection can be rebuilt from the log.

The SQLite sidecar is the projection's idempotency authority during normal
writes. It records open references as soon as the store accepts an artifact, then
promotes those references when the shard finalizes. Readers still consume only
finalized shard references. Finalized manifests remain the rebuild source of
truth, so a missing or stale sidecar can be reconstructed from finalized marker
files.

Zarr is attractive because it supports chunked storage, compression,
structured grouping, and efficient partial reads. It is also better aligned
with a local-to-object-store path than traditional HDF5, which is strongest in
POSIX-style local file contexts. This fits the near-term goal of local
operation while keeping open a later move to object storage such as R2.

Because the data includes variable-length text, the artifact layer needs a
vartext-style wrapper. Conceptually, text payloads are stored as byte arrays,
with a pointer layer mapping logical record IDs to shard identifiers, byte
offsets, lengths, encodings, checksums, and related physical-location data.
This keeps retrieval simple while allowing the artifact store to remain a
durable chunked blob substrate rather than a semantic query engine.
