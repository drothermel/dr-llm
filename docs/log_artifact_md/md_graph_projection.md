# Metadata Graph Projection

The metadata projection stores the high-contact facts needed to filter,
inspect, coordinate, and analyze the corpus. It should include experiment
identifiers, run metadata, prompt and model settings, timestamps, statuses,
failure indicators, system metrics, event IDs, artifact references, checksums,
projection versions, and processing state.

Although the original plan described this as a Postgres/Neon projection, the
data is naturally graph-shaped. A prompt type, data element, model
configuration, and hyperparameters jointly produce an output result. Later,
many raw outputs may be reduced into one canonical output, such as a
standardized implementation representing multiple equivalent generated code
variants.

The first metadata projection should still use Postgres, but it should be
modeled as a graph/fact store from the beginning. The core shape is closer to
n-ary relations than simple binary edges:

```text
GenerationAttempt(
  prompt_template,
  prompt_instance,
  data_element,
  hparams,
  model_config,
  provider,
  run,
  output_result,
  artifacts,
  metrics,
  status
)
```

In relational form, this can be represented with entities, assertions, and
assertion roles:

```text
entities
  entity_id
  entity_type
  content_hash
  display_name
  metadata_json

assertions
  assertion_id
  assertion_type
  event_id
  projection_version
  status
  metadata_json

assertion_roles
  assertion_id
  role_name
  entity_id
```

This gives the metadata layer hypergraph-like semantics while preserving
Postgres strengths: indexed queries, constraints, joins, transactions, JSONB,
schema migration, and local or Neon-backed operation.

Content addressing should be part of the metadata design early. Stable hashes
should identify normalized content where identity should be content-based:
prompt templates, prompt instances, data elements, model configs,
hyperparameters, raw responses, normalized code, canonical implementations,
and artifact payloads. Raw outputs should remain immutable. Canonicalization
should create derived assertions rather than overwrite raw facts.

Example derived assertion:

```text
EquivalentImplementation(
  candidate_implementation,
  canonical_implementation,
  equivalence_method,
  evidence_artifact,
  confidence,
  projection_version
)
```

This lets the system preserve provenance while supporting reductions from
many raw output nodes to fewer canonical nodes.

A dedicated graph system such as Neo4j can be useful as a secondary analysis
projection, especially for interactive traversal questions:

- Which outputs derive from this prompt family and dataset slice?
- Which failures are near this model family, hparam regime, and output type?
- Which raw implementations collapse into this canonical implementation?
- What is the provenance path from a canonical result back to raw generations,
  validation artifacts, and source events?

Neo4j-style labeled property graphs are useful for nodes, relationships,
paths, and visual exploration. RDF or quad stores may be useful if formal
ontologies, named graphs, or interoperability become important. Datalog-style
systems are especially relevant for rule-based derivations such as
equivalence, reachability, transitive closure, and canonicalization. These
systems should be treated as projections or derivation engines, not as the
first operational metadata catalog.

The working recommendation is therefore:

1. Use Postgres/Neon as the primary metadata projection.
2. Model it as a graph/fact/hyperedge store from the start.
3. Use content hashes for deduplication and stable identity.
4. Represent canonicalization as derived assertions with provenance.
5. Add Neo4j, RDF, or Datalog projections when concrete analysis workflows
   need them.
