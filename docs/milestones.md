# Milestone Status

## M1: Core Primitive
Status: complete.

## M2a: Session + Tool Runtime
Status: complete.

## M2b: Throughput / Ops Hardening
Status: complete with benchmark + runbook artifacts.

Implemented artifacts:
1. `dr-llm run benchmark`
2. `src/dr_llm/benchmark.py`
3. `docs/ops/m2b-hardening-checklist.md`

## M3: Ecosystem Integration
Status: complete with compatibility + migration + integration examples.

Implemented artifacts:
1. `docs/compatibility-contract.md`
2. `docs/migration-guide.md`
3. `docs/integrations/nl_latents.md`
4. `docs/integrations/unitbench.md`
5. `examples/nl_latents_gateway.py`
6. `examples/unitbench_gateway.py`

## M4: Sample Pools
Status: complete.

Generic typed, schema-driven sample storage for benchmarks with no-replacement
acquisition, pending sample lifecycle, and top-up orchestration.

Implemented artifacts:
1. `src/dr_llm/pool/` subpackage (schema, store, service, DDL, models, errors)
2. `tests/test_pool_ddl.py`, `tests/test_pool_models.py` (unit tests)
3. `tests/integration/test_pool_store.py` (integration tests)
