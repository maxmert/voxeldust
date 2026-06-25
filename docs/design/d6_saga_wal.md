# D-6 — durable saga WAL + orchestrator-kill resilience

LOCKED design (workflow `wf_0f8321dc-32c`: ground → 2 designs → 4 adversarial skeptics default-refute →
synthesize; design-B agent died on an API Overload but A + all 4 critiques + synthesis completed). Chosen
#1 by the next-step assessment (`wf_b965c810`): the kill-9 headline NAMES the orchestrator as a victim but
has ZERO coverage — the saga set + `batch_goes` are in-memory, so an orchestrator restart vaporizes every
in-flight transfer. Pre-cut seam (the FSM already emits `PersistCheckpoint`), max leverage (unblocks D-37
re-home + D-7d-S2 orchestrator-kill cell + the D-7c gate).

## Resolved decisions (recommended defaults, mandate-aligned — no genuine fork)

1. **NO new dependency.** The `Store` seam + `MemStore` use only `postcard` (already a vd-sim/vd-node dep)
   + `std::sync::Arc<Mutex<BTreeMap>>`. **redb stays deferred to io-prod** (where redb-vs-sled-vs-custom is
   jointly investigated — investigate-libraries-together). D-6 lands ONLY the trait + the mem impl.
2. **Two-tier (staged/committed) `MemStore`.** `put`/`delete` mutate `staged`; `commit()` merges
   staged→committed; a crash (drop the World, NOT the `MemStore` handle) drops `staged` only — modeling the
   prod fsync window. LOAD-BEARING: a put-is-instantly-durable store would make the LOSE-side commit-window
   cell (C4) untestable. Do not optimize the split away.
3. **Directory persists INDEPENDENTLY of the saga WAL** (separate `Store` families, separate-file-
   compatible per PLAN.md:126); the `CommittingCas-may-have-won` case is resolved by **re-reading the
   durable directory head**, NOT by atomic cross-file co-commit (which contradicts PLAN.md:126, is
   unachievable in redb, and paints D-32 directory-partitioning into a corner — for zero correctness gain,
   since the head-re-read already tolerates disagreement, as the FSM's `(CommittingCas, Timeout)` arm
   mandates). The directory MUST persist (it is the commit point); it just must not be welded to the saga
   WAL transaction.

## The crown insight (persist QUIESCENT state at the write-back, NOT the emit sites)

`IssueCommitCas` is a same-tick DIRECT call (`saga_runtime.rs:466`), so `CommittingCas`/`Freezing`/
`Demoting`/`Aborting`/advanced-`BatchHandoff` are NEVER quiescent — a WAL anchored only to the two
`PersistCheckpoint` emit sites could ONLY ever hold `Cutting` or `Swapping`. Re-driving a real
`CommittingCas` crash from a stale `Cutting` snapshot → `abort_from_pre_freeze` → split-brain abort of a
possibly-committed transfer. FIX: persist the `SagaSnapshot` from **`commit_result`'s write-back**
(`saga_runtime.rs:670`, where `live.state = final_state`) — the SINGLE point that knows the true quiescent
phase. The two `A::PersistCheckpoint` arms become DURABILITY-FORCE markers (set a `force_commit` flag so
the end-of-tick barrier flushes BEFORE the checkpoint's effect leaves — persist-before-effect).

A second BLOCKER the skeptics caught: the **clock ceiling** must persist + `CeilingClock::recover` on
rebuild (not `genesis`, which resets `current=0`) — else the rebuilt clock restarts near 0,
`scan_deadlines`' `now.saturating_sub(since)` saturates to 0 forever, the re-drive engine is silently dead,
and the cells FALSE-GREEN on World-rebuild while recovery is broken.

## Steps (S0–S5)

- **S0 — the `Store` seam + `MemStore`** (`sim/io/mod.rs`, `sim/io/mem.rs`). `Store` is OBJECT-SAFE
  (`dyn Store` — no per-monomorphization HR5 gotcha). Minimal v1: `put(&[u8], &Bytes)`, `delete(&[u8])`,
  `scan(prefix:&[u8]) -> Vec<(Vec<u8>, Bytes)>`, `commit()` (the durability barrier). NO `get` (no flow
  point-reads — dead Tier-A surface). `StoreKey` enum (TYPED, not stringly-keyed): `Saga(TransferId)` |
  `BatchGo(BatchId)` | `Directory(DirectoryKey)` | `Tombstone(TransferId)` | `Clock`, each a 1-byte prefix
  tag so `scan(prefix)` recovers one family; encode via postcard at the MONOMORPHIC call sites (HR5
  generic-shim). `MemStore` = `Arc<Mutex<StoreInner>>` (clone = same backing log, like `MemHub`); `StoreInner`
  holds `committed` + `staged` BTreeMaps; `commit()` merges; drop-the-World keeps the handle's `committed`.
  SPIKE-0a contract tests at 100%: put/scan/delete/commit + staged-dropped-on-crash + commit-then-survive.
- **S1 — persist the quiescent SagaSnapshot at the write-back** (`saga_runtime.rs`, `orchestrator.rs`).
  `SagaSnapshot{ctx, state: final_state, gateway, since, flushed_pose}` (all fields Serialize-verified).
  In `commit_result`: every transition stages `put(Saga(t), snapshot)`; a tombstone stages `delete(Saga(t))`
  + `put(Tombstone(t))`. `flushed_pose` captured AFTER write-back (a Swapping snapshot has `Some(pose)`).
  Per-batch `put(BatchGo(b))` ONCE in the batch_gos loop (idempotent overwrite — preserves the G-TIER
  `batch_go_writes` decoupling; do NOT re-funnel through the `+= 1`). `StoreRes` = a bevy Resource (analog
  of `DirectoryRes`) injected by `register_orchestrator`; non-orchestrator nodes never get it.
- **S2 — group-commit barrier + durable directory + durable clock ceiling** (`saga_runtime.rs`,
  `orchestrator.rs`). ONE `Store::commit()` at the END of `drive_sagas` (after starts + ack loop +
  scan_deadlines) — a tick advancing N sagas fsyncs ONCE. At `commit_cas` + `abort_clear`, stage
  `put(Directory(subject), OwnerRecord)` (INDEPENDENT, not co-committed). On `ReserveCeiling`, persist the
  ceiling (`StoreKey::Clock`); on rebuild, `CeilingClock::recover(persisted_ceiling)` not `genesis`.
  `register_orchestrator` is Store-aware: non-empty Store → recover; empty → genesis.
- **S3 — `rehydrate`** (`saga_runtime.rs`, `orchestrator.rs`). Once on rebuild, BEFORE the first tick:
  scan Directory→`DirectoryCore`, BatchGo→`batch_goes` (set `batch_go_writes := len` via a monomorphic
  helper, NO re-drive), Tombstone→tombstone set, Saga→`sagas` with `since` armed so the next
  `scan_deadlines` fires each immediately and the EXISTING Slice-2a per-phase Timeout arms re-drive (NO new
  path). FORWARD-ONLY PAST COMMITTED + the TOTAL 3-way head-re-read (won → forward; not-committed → safe
  re-CAS; window-closed → clean terminal). Orphan-lock sweep: a Directory `in_transfer` lock with no live
  saga is `abort_clear`ed. The rehydrate match is EXHAUSTIVE (no `_`, like `deadline_for`).
- **S4 — harness `KillRebuild`** (`harness/fabric.rs`, `sim/io/mem.rs`, `harness/topology.rs`,
  `tests/src/lib.rs`). The TRUE kill-9 analog: `Topology::replace_node` deregisters the node + builds a
  FRESH one against the SAME retained `MemStore` handle (the World's RAM dies; the store's `committed`
  survives). `Fault::KillRebuild`. Anti-theater guard: a no-op-stub Store variant goes RED (proves the
  rebuild truly loses RAM, not free in-process World survival).
- **S5 — the orchestrator-victim crash cells** (`tests/tests/p3_crash_matrix.rs`). C1–C9 (below), reusing
  `EndState`/`assert_end_state`.

## Crash cells (C1–C9)

C1 Preparing (pre-cp-1): no Saga snapshot + maybe a Directory lock → orphan-lock sweep clears it, avatar on
live source. C2 Cutting (post-cp-1): Cutting-Timeout → clean abort, avatar on live source. C3
Freezing→Aborting (the abort-side hole the original's two-Cutting cells hid): re-hydrate at Aborting →
ThawSource re-emits → source THAWED (never wedged-frozen). C4 commit-window BEFORE `commit()` (LOSE side):
staged flip + Swapping snapshot dropped → re-hydrate at Freezing + directory source-owned → head-re-read
"not committed" → re-CAS wins → Done at DEST (or clean abort). C5 commit-window AFTER `commit()` (WON side,
THE central case): directory durably flipped + Swapping snapshot durable → forward-only tail re-drives →
DEST, NOT split-brain. C6 Demoting: idempotent Demote re-emit (journaled) → Done at DEST, no vanish. C7
BatchHandoff AwaitPromote (transient): re-hydrate at the QUIESCENT advanced phase → EmitTransientPromote
re-emit (NOT re-driving from AwaitAdopt) → Held at DEST, zero loss, `batch_go_writes` preserved. C8 terminal
edge then redeliver PREPARE: tombstone drops it (no zombie re-run). C9 byte-identity: two runs under one
seed → identical inspect + trace (the WAL perturbs no observable frame).

## Correctness (recovery is complete, no double-commit, no split-brain, no zombie)

`commit_cas` is the SOLE durable `OwnerRecord` mutation, runs at most once durably (staged/committed
barrier). A post-cp-2 saga only re-emits idempotent `(transfer, step_id)`-journaled commands. The directory
+ saga MAY land independently (separate families, D-32-safe); recovery NEVER assumes they agree — it READS
the durable head (the total 3-way re-read). `CeilingClock::recover` keeps the clock forward (re-drive
engine alive). Terminal sagas tombstone crash-atomically with the terminal directory mutation; a redelivered
PREPARE is dropped. Forward progress is bounded by the proven Slice-2a producer.

## Deferred (HONEST, each with WHEN)

- **redb `Store` impl** (real WAL-buffered, fsync-off-tick) → io-prod (PLAN.md:130; the backend is a joint
  decision). **Off-tick fsync thread** → io-prod. **`batch_goes` WAL GC** → D-7d Slice 2 (co-gated with the
  drop-completion signal; a delete HOOK co-located now so the WAL twin can't outlive its in-memory twin, but
  the retire trigger is NOT decided here). **WAL compaction / retention / schema-versioning** → P6/P7.
  **Replicated/partitioned WAL** → D-32 (single-orchestrator + single WAL by construction now). **Re-home of
  a permanently-dead participant** → D-37 (+ D-3) — D-6 makes the orchestrator-crash ROW recoverable; a
  permanent participant kill still parks/orphans honestly. **3rd pre-CAS persist (durable CommittingCas)** —
  not needed now (Freezing pre-CAS + Swapping post-CAS are the natural durable phases; revisit at io-prod's
  off-tick barrier). **LOSE-side re-CAS vs a future lease reaper (D-3/XSI-1)** — documented cross-design dep
  (a reaper would require distinguishing "reaped during downtime" → re-home from "still locked" → re-CAS).

## Multithreading verdict: SEQUENTIAL

No new parallelism. The WAL is the orchestrator's private durable medium (HR1); group-commit is one
synchronous `commit()` per tick (the mem tier); the io-prod off-tick fsync thread is the only future
concurrency, deferred. rayon remains un-adopted.
