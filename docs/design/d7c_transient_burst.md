# D-7c — G-TIER one-write-per-batch + DURABLE-UNAFFECTED-BY-BURST

LOCKED design (workflow `wf_364d34ba`: ground → 2 designs → 4 adversarial skeptics default-refute →
synthesize). Builds on D-7a/D-7b (thru HEAD `891d99b`). The skeptics caught 3 blockers (the G-TIER
false-green, the cap-split broken under derived ids, a private-fn cross-crate dep) + majors; all fixed.

## The crown insight (G-TIER must measure WRITES, not map size)

`batch_goes.len()==K` is FALSE-GREEN: `commit_result` does `batch_goes.entry(BatchId(ctx.transfer))
.or_insert(...)`, idempotent on the SAME key, so a per-item-write regression (1000 same-key writes)
collapses to `len==1` and the test passes vacuously. The ONLY observable that turns the regression RED
is a monotonic **`batch_go_writes: u64`** counter, incremented once per `BatchGo` in `commit_result`'s
loop BEFORE the `or_insert`. G-TIER asserts `batch_go_writes == distinct-batch-count` (== 1 for the
1000-item single batch). Under a per-item regression `batch_go_writes -> 1000` while `len` stays 1.

## 7c.1 — the Tier-A observable (lands FIRST, fully covered)

- vd-node `saga_runtime.rs`: `SagaRuntimeRes.batch_go_writes: u64` + `runtime.batch_go_writes += 1`
  in `commit_result`'s `batch_gos` loop BEFORE `or_insert` (straight-line, no branch) + `pub fn
  batch_go_writes() -> u64` reader. Coverage: an `assert_eq!(rt.batch_go_writes(), 1)` added to the
  EXISTING `transient_subject_commits_via_the_go_token_not_a_cas` test.
- vd-harness `topology.rs`: `InspectReport.batch_go_writes: u64` + `report.batch_go_writes =
  rt.batch_go_writes()` inside the already-covered `Some(rt)` arm (straight-line, no branch).

## 7c.2 — G-TIER (vd-tests, coverage-ignored consumer)

`seed_transient_burst(topo, batch, BURST_SIZE, src_fence, dst_fence)` (in `tests/src/lib.rs`) seeds
`BURST_SIZE=1000` distinct Debris (`EntityId::pack(Debris, SHARD.0 as u32, i as u64, 0)`) ALL carrying
ONE shared `batch` via the committed 6-arg `seed_transient_crossing`. ONE `trigger_transfer(Transient,
Realm(DST_REALM))`. `step_asserting_conservation(topo, 24)`. Assertions: (A) PROBE `sum(batch_go_writes)
== 1`; (B) SIZE `batch_goes == [(BatchId(batch), dst_fence)]`; (C) ZERO directory rows for any burst
entity; (D) positive guards `held_at_dest == BURST_SIZE` AND `transient_dropped_total == 0` (close the
drop-axis vacuity — all 1000 settled singly, nothing lost). K>1 discriminator
(`p3_gtier_write_rate_scales_with_batch_count`): K=3 DISTINCT batch ids via THREE `seed_transient_burst`
+ THREE `trigger_transfer` (NOT the cap-split — broken), assert `batch_go_writes==3 && batch_goes.len()
==3 && transients_emitted==3` while `held_at_dest==1000` (per-batch ack cost is O(batches), not items).

## 7c.3 — DURABLE-UNAFFECTED-BY-BURST (vd-tests, coverage-ignored consumer)

`durable_subset(reports, subject, session) -> Vec<(NodeId, DurableProjection)>` (`#[derive(PartialEq,
Eq,Debug)]`) projects ONLY the durable-relevant fields filtered to the subject/session: the
`DirectoryKey::Entity(subject)` row, `held_entities`/`held_poses`/`pending`/`departing` filtered, the
`ghost_dots` membership, the durable `active_transfers` + `OwnerRecord.in_transfer`/`.fence` + `live_sagas`.
EXCLUDED (legitimately move under a burst): `owned_transients`, `held_transient_poses`, `batch_goes`,
`batch_go_writes`, `transient_loss`, `held_realms`, and **`trace_bytes`** (the burst's real
TransientBatch/ack frames make per-tick sent/drained differ → diffing it would FALSE-RED).
Test `p3_durable_transfer_byte_identical_under_concurrent_burst`: same `FaultFabric::new(909,2)` seed,
same durable warmup (via PUBLIC `p1_client`/`walk_forward`/`trigger_transfer`/`read_subject`/`realm_fence`
— NOT the private `run_cut_transfer`), BASELINE (durable only) vs BURST (durable + a concurrent
1000-`seed_transient_burst` + a second Transient `trigger_transfer`, both driven to quiescence on the
ONE `SagaRuntimeRes`); `assert_eq!(durable_subset(base), durable_subset(burst))`. WHY CORRECT: transients
take NO directory lock + write ZERO directory rows, so the durable saga and the burst share NO mutable
orchestrator state except `batch_goes` (excluded) + the wire fabric (why `trace_bytes` moves, the subset
doesn't) — any subset diff is a burst leak = an HR1/HR2 violation, exactly what the gate catches.
Two-sided gates: SENSITIVITY `p3_durable_subset_moves_when_durable_saga_perturbed` (perturb the durable
path → `durable_subset(base) != durable_subset(perturbed)`, rejects an over-filtered inert subset);
SPECIFICITY (main-test negative control) `with_burst owned_transients@DEST==1000 && batch_go_writes==1`
vs `no_burst batch_go_writes==0` (the burst genuinely ran). PRECONDITION (doc-comment): byte-identity is
valid ONLY under the zero `LinkPolicy` (`p2_cluster` installs no faults); a faulted variant would need
invariant-equality (same terminal directory record + applied-input multiset), not raw `assert_eq!`.

## Multithreading verdict: SEQUENTIAL (rayon NOT adopted)

`readvance_transients` stays sequential; NO new library. Justification (all 4 axes, skeptic-confirmed):
(1) 1000 closed-form advances ≈ low-µs L1-resident FMAs — a rayon dispatch (wakeup + work-steal + join)
would itself cost µs, plausibly EXCEEDING the work. (2) The sim thread is synchronous single-threaded
(`ExecutorKind::SingleThreaded`); the byte-identical replay gate is a STRUCTURAL guarantee that
thread-scheduling + a future shared accumulator (P5 seed-gravity / any `.reduce`) would silently break —
don't trade it for µs. (3) Parallelism is ALREADY at the right granularity: each shard is its own
process/World/`step_tick` with per-node tokio reader/writer tasks; the cross-shard fan-out IS the
parallelism (HR1). (4) `rayon` is NOT a dep (absent from Cargo.toml + Cargo.lock) — adopting it is a USER
decision (investigate-libs-together), FLAGGED not adopted. The `par_iter_mut`-ready note is forward-only.

## DEFERRED-to-D-7d (deliberately NOT in D-7c)

- `StubConfig.max_items_per_batch` (the source-side wire-frame cap): the design proposed shipping it now
  unit-test-only, but a cap-split is LOSSY-if-triggered (the dest's `(transfer, BATCH_STEP)` journal
  dedups overflow envelopes → items dropped) and `MsgClass::Saga` is reliable (no MTU cap actually needed
  for a 1000-item batch). Shipping the half is a latent landmine — DEFER the WHOLE thing to D-7d, where
  the sub-batch→distinct-go-token wiring makes it correct. ("burst params → StubConfig" moves to D-7d.)
- batch_goes bounded GC: re-filed from D-7c/D-6 to **D-7d** (GC needs the drop-completion signal D-7d
  co-designs; landing it in D-7c would make `batch_goes.len()` time-dependent + break the quiescence count).
- MIXED-KIND / MULTI-DEST-REALM burst: the committed `seed_transient_crossing` hardcodes `dest=DEST`/
  `to_realm=System(8)` + `p2_cluster` has ONE DEST → a true mixed multi-realm burst is structurally
  unreachable through the current API. Owed D-7d (a 3rd shard + a `seed_transient_crossing_to(dest,realm)`
  parameterization). D-7c's single-kind single-realm headline is honestly scoped, not the full claim.
