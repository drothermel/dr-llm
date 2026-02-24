# M2b and M3 Closeout Design

## Scope
Close the remaining milestones by adding:
1. A repeatable, DB-backed throughput benchmark path for mixed read/write/session operations.
2. Explicit operational guidance for scaling PostgreSQL and worker fleets.
3. Integration contracts and examples for `nl_latents`, `unitbench`, and future consumers.

## Recommended Approach
1. Add a benchmark primitive in core (`run_repository_benchmark`) that executes a deterministic operation mix under configurable thread count.
2. Expose that primitive through CLI as `llm-pool run benchmark` so performance checks are easy in CI or local staging.
3. Define M2b runbook expectations (pool sizing, retries, leases, indexes, partition readiness) in a single ops checklist.
4. Define an M3 compatibility contract that locks stable API/storage expectations for downstream repos.
5. Add focused integration example docs and minimal code skeletons for `nl_latents` and `unitbench`.

## Success Criteria
1. Team can execute a standard benchmark command and compare throughput/latency before releases.
2. Ops checklist is explicit enough to run production readiness reviews without tribal knowledge.
3. Downstream repos can adopt `llm-pool` using stable interfaces without introducing domain logic into this repo.
4. Migration steps from direct provider usage to `llm-pool` are documented and reusable.

## Tradeoffs
1. Benchmark focuses on storage and orchestration paths, not provider network latency realism.
2. Integration examples remain lightweight and intentionally non-domain-authoritative.
3. Compatibility contract is additive-forward and may require explicit migration scripts for breaking storage changes.
