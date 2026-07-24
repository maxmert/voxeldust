# RLM Step 3 — The Orchestrator Realm-Lifecycle Reconciler: DEFINITIVE SPEC

Status: VETTED. Synthesized from four independent designs (robustness-first, determinism-first,
simplicity-dry, cloud-seam-first) + their adversarial critiques, arbitrated against the live tree
(HEAD on `worktree-new-system`, Step 2 committed `d961563`). Every disputed fact below was
re-verified against source; file:line anchors are load-bearing.

The four designs AGREE on the spine (level-triggered re-derivation, Directory-as-truth, pure
DECIDE / `Box<dyn RealmSpawner>` EXECUTE, affirmative-Empty NO-STRAND). They DISAGREE on three
things that the critiques proved are **correctness bugs, not preferences**. This spec adopts the
spine and resolves the three bugs with the fixes the critiques converged on, each grounded:

- **BUG-A (crashed-shard immortality / leak)** — verified: the reaper's `Realm(_) => {}` arm
  (`saga_runtime.rs:2196`, comment `:2144` "revoking a player's ship realm would freeze the ship
  forever") leaves a dead realm's Directory head in place forever. Every design that models
  `is_running := head-present` keeps a crashed occupied realm "desired-alive-by-default" forever
  (arm B) → leaked pod, never re-spun. **FIX: `running` means LIVE.** The reconciler consults the
  existing `LivenessTracker::is_latched_dead` (`saga_runtime.rs:361`) — the exact disambiguator —
  and force-revokes + respawns a dead-latched realm. This is NOT deferrable to Step 4.
- **BUG-B (spawned-but-not-leased double-spawn)** — verified: `MemSpawner::spawn_realm`
  (`mem.rs:484-507`) mints a fresh monotone `NodeId` and plants a hub transport but writes NO
  Directory head; the head appears only after the booted shard self-grants at `Fence::GENESIS.next`
  (`stub.rs:1285`, gated on `has_synced`) — a MANY-tick window. A reconciler that keys `actual` on
  the head alone re-spawns every tick of that window → N pods per realm. **FIX: the spawner's OWN
  `live()` map (`mem.rs:478`, already built "for RLM") is the intermediate source of truth.**
  `actual := head-present OR spawner.live() holds a node for this coord`. Plus a per-coord launch
  ledger (not a single overwriting ticket) so re-issues never orphan.
- **BUG-C (login-into-draining rescue race)** — verified via the tick trace: a single-tick
  synchronous `kill_realm` fired the tick BEFORE a rescuing demand lands cannot be vetoed. **FIX:
  two-phase teardown with a DRAIN latch** — the reconciler never kills in the same tick it decides;
  a `Draining` reservation holds for `teardown_drain_ticks` during which ANY demand (AoI or spawn)
  aborts the kill, and the realm is never even marked closing until the drain confirms.

All three fixes reuse existing machinery (`is_latched_dead`, `live()`, the demand ledger). No new
inter-shard wire arm is required (BUG-C's drain is orchestrator-local: the reconciler already knows
emptiness from the realm's own `Empty` self-report; the drain is a reconciler-side timer, not a new
child handshake). HR1 is preserved.

---

## PART 1 — THE realm-lifecycle STATE MACHINE (the ONE chosen machine)

### 1.0 Design stance (why derived, not stored)

There is **no stored per-realm state enum that transitions on events**. The lifecycle state is
**re-derived every tick** (level-triggered) as a pure function of four observable facts per realm
`RealmPath`:

1. **demand facts** (RAM ledger): `last_demand_tick`, `last_empty_tick`, `spawn_watermark`,
   `teardown_watermark`, `draining_since` — re-accrued from `ReDriven` demands.
2. **head fact** (Directory): is there a `DirectoryKey::Realm(coord.lowered())` record?
   (`directory.rs:644` `head`, `:650` `entries`.)
3. **liveness fact** (LivenessTracker): `is_latched_dead(node)` for the head's owner node
   (`saga_runtime.rs:361`) — the crash disambiguator.
4. **launch fact** (spawner): `spawner.live()` holds a minted-not-yet-leased node for this coord
   (`mem.rs:478`) — the spawned-but-not-leased bridge.

A stored FSM would be a second source of truth that can desync from the Directory; the
level-triggered contract forbids it. The named states below exist ONLY as a derived classification
(for the admin view, the reconcile diff, and this spec's proofs).

### 1.1 The derived states

Let `running_live(P)` be the composite "actual" predicate (§1.4). Then per realm path P each tick:

| State | Derivation |
|---|---|
| `Absent` | not desired-alive, no head, not in `spawner.live()` |
| `Launching` | desired-alive, no head yet, BUT `spawner.live()` holds a node for P (spawn accepted, shard booting/self-granting) |
| `Running` | desired-alive AND `running_live(P)` (head present, owner not dead-latched) |
| `Draining` | `running_live(P)`, `teardown_ready(P)` held continuously since `draining_since` (kill DECIDED, not yet executed — the BUG-C veto window) |
| `Reaping` | `Draining` AND `now - draining_since >= teardown_drain_ticks` AND still `teardown_ready(P)` (kill EXECUTES this tick) |
| `Zombie` | head present BUT owner `is_latched_dead` (crashed shard — the BUG-A path: force-revoke + respawn-if-demanded) |
| `Retired` | no head, not desired, GC the ledger cell |

### 1.2 Transitions (ASCII)

```
                        demand (SpinUp/KeepAlive/spawn) within TTL, or ancestor-closure
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                     ▼
[Absent] ──spawn intent, spawner.live gains node──► [Launching] ──head self-granted──► [Running]
  ▲                                                      │  (many-tick boot window)         │
  │                                                launch_ttl expired &                     │  demand drops past TTL
  │                                                node dead in live() ──re-spawn (same P)   │  AND realm said Empty past grace
  │                                                                                          ▼
  │                                                                            teardown_ready(P) first true
  │                                                                                          │ set draining_since=now
  │                                                                                          ▼
  │                              any demand for P (rescue) ─────────────────────────►  [Draining]  (VETO WINDOW)
  │                              clears draining_since, back to Running                        │
  │                                                                            drain elapsed & still ready
  │                                                                                          ▼
  └──────────────── head revoked, spawner killed, ledger retired ◄──────────────────── [Reaping]

  [Running]/[Launching] with head owner is_latched_dead ──────────────────────────────► [Zombie]
       force-revoke stale head + kill_realm(corpse) ; if still demanded → [Absent]→re-spawn ; else → [Retired]
```

### 1.3 Timers (ONE tuning struct `RlmTuning`, no inline literals; INERT by default)

| Timer | Meaning | Prevents |
|---|---|---|
| `demand_ttl_ticks` | a demand keeps P desired for this long after its last SpinUp/KeepAlive/spawn | one dropped `KeepAlive` from un-desiring a live realm (self-heals via `ReDriven`) |
| `empty_grace_ticks` | after P's FIRST continuous `Empty`, hold before P counts as confirmed-empty | an `Empty` racing a just-arrived occupant |
| `teardown_cooldown_ticks` | continuous `teardown_ready` dwell before entering `Draining` | flap: empty→refill within cooldown never even proposes a kill |
| `teardown_drain_ticks` | the BUG-C veto window: `Draining`→`Reaping` delay during which any demand aborts | login/crossing-into-draining strand (stress #2, #9) |
| `spinup_cooldown_ticks` | after a spawn intent for P, suppress re-spawn (ambient churn only; a hard demand bypasses) | double-spawn churn during the boot window (belt to BUG-B's `live()` suspenders) |
| `launch_ttl_ticks` | how long a `Launching` node is trusted before a re-spawn is considered | a silently-failed launch wedging `Launching` forever |
| `reconcile_interval_ticks` | sweep cadence; `0` ⇒ INERT (the dev/test + legacy default) | per-tick full sweep at scale; byte-identical legacy rigs |

`RlmTuning::default()` = **all zero = INERT**: `reconcile_interval_ticks == 0` makes the whole
system a no-op, so every existing orchestrator rig (which emits no `RealmDemand` and inserts a
default tuning) is byte-identical. `RlmTuning::cloud(tick_hz)` derives the active set off `tick_hz`,
mirroring `DirectoryTuning::cloud` (`directory.rs:164`). `RlmTuning::validate()` (mirrors
`DirectoryTuning::validate`, `directory.rs:223`) enforces, fail-loud, the anti-thrash ordering
invariant (§ RISK item 3):

```
demand_ttl_ticks        >  shard_grace_ticks + launch_ttl_ticks     // orch outlasts the shard AoI band + boot
teardown_cooldown_ticks >= empty_grace_ticks                        // can't be eligible before confirmed-empty
teardown_cooldown_ticks >= demand_ttl_ticks                         // can't kill faster than a demand re-desires
teardown_drain_ticks    >  0  (cloud)                               // the veto window must exist
demand_ttl_ticks        >= 2 * reconcile_interval_ticks             // a demand survives to the next sweep
```

`shard_grace_ticks` is re-derived by the SAME `tick_hz` formula the shard's `AoiConfig` uses (a
shared const module), NOT by importing `AoiConfig` into `vd-node` (keeps `node → sim` DAG clean).

### 1.4 THE EXACT `running_live` (composite "actual") predicate

Every earlier design's fatal simplification was `is_running := head-present`. The correct actual set
distinguishes three sub-cases, all monomorphic:

```rust
// P = realm path; rid = P.lowered(); node = head owner if any.
fn head_present(dir, rid) -> Option<(NodeId, Fence)>        // dir.head(Realm(rid)) → owner+fence
fn launch_present(live, P) -> Option<NodeId>                // spawner.live() reverse-lookup by coord.path()
fn owner_dead(liveness, node) -> bool                       // liveness.is_latched_dead(node)

running_live(P) :=  head_present(P).is_some_and(|(n,_)| !owner_dead(n))   // a LIVE head
zombie(P)       :=  head_present(P).is_some_and(|(n,_)|  owner_dead(n))   // BUG-A: dead head lingering
launching(P)    :=  head_present(P).is_none() AND launch_present(P).is_some()
```

`launch_present` is the BUG-B bridge: `spawner.live()` (`mem.rs:478`, returns
`BTreeMap<NodeId,(ProfileKind,UniverseTick)>`) is enumerated and a node is attributed to P by the
coord the reconciler recorded in its launch ledger when it minted the node (§2.3). This closes the
"spawned but head not yet self-granted" gap deterministically — no guessed timeout.

### 1.5 THE EXACT desired-alive predicate

```
desired_alive(P, now) :=
      demanded_recently(P, now)                          // (A) demand-driven (AoI OR spawn — source-agnostic)
   OR ( running_live(P) AND NOT empty_confirmed(P, now) )// (B) LIVE and hasn't affirmatively gone empty
   // NOTE arm (B) uses running_LIVE, not head-present — a Zombie is NOT desired-alive (BUG-A fix).

where
  demanded_recently(P, now) :=
        ledger.get(P).is_some_and(|e| now - e.last_demand_tick <= demand_ttl_ticks)
        // last_demand_tick is refreshed by SpinUp | KeepAlive | player-spawn (ancestor-closure sets it
        // on the whole parent chain, §3.2). Empty does NOT refresh it.

  empty_confirmed(P, now) :=
        ledger.get(P).is_some_and(|e| match e.last_empty_tick {
            Some(t) => now - t <= empty_grace_ticks       // an Empty within the grace window ("recently")
                       AND e.last_demand_tick < e.first_empty_tick,  // (C) no non-Empty demand AFTER the Empty
            None => false,                                 // NEVER said Empty ⇒ never confirmed-empty
        })
```

Then the FINAL desired set is the **ancestor-closure** of `{P : desired_alive(P)}` (§3.2) — every
desired realm pulls its whole `Universe→…→P` parent chain in.

**Arm (B) rationale (NO-STRAND core):** a LIVE realm stays desired-alive by default; it can only
LEAVE desired-alive by affirmatively saying `Empty` (and staying empty past grace with no newer
demand). Demand-absence alone never un-desires a live realm (a partitioned parent stops
`KeepAlive`-ing, but arm B holds). Verified against the producer: a realm with occupants emits NO
`Empty` (`stub.rs:4356` only pushes `Empty` when `occupants.is_empty()`), so `last_empty_tick`
stays stale ⇒ `empty_confirmed` false ⇒ desired-alive holds. The occupancy authority and the
Empty-report authority are the SAME sealed shard — no cross-shard race in the emptiness fact.

**Arm (C) rationale (the Empty/warp-in race, hardened per critique C-2):** the determinism-first
critique proved a `>`-against-`now` tie-break can invert on a same-tick Empty+SpinUp. The correct
comparison is between the two competing FACTS, not against `now`: `empty_confirmed` requires
`last_demand_tick < first_empty_tick` — i.e. NO non-Empty demand at-or-after the Empty. A same-tick
or newer SpinUp (`last_demand_tick >= first_empty_tick`) provably keeps the realm alive. We track
`first_empty_tick` (the tick the CURRENT empty streak started, reset when any demand clears it) so
grace measures from the streak start (a genuinely-empty realm ages out; a re-asserted Empty does
NOT advance the clock).

### 1.6 THE EXACT teardown-ready + two-phase teardown predicates

`teardown_ready` is necessary-but-not-sufficient to kill; the kill only fires after the drain.

```
teardown_ready(P, now) :=
      running_live(P)                                    // (1) a LIVE lease to reclaim (never a Zombie here)
   AND NOT desired_alive_closure(P, now)                 // (2) out of ALL AoI (incl. as an ancestor) past TTL
   AND empty_confirmed(P, now)                           // (3) it AFFIRMATIVELY said Empty (arm C hardened)
   AND (now - ledger.get(P).spawn_watermark) >= min_dwell // (4) not inside its own boot+settle window
   AND (now - ledger.get(P).teardown_watermark) >= teardown_cooldown_ticks // (5) past teardown cooldown
   AND NOT has_desired_descendant(P, closure)            // (6) never orphan a live child (ancestor-closure guard)
   AND head_present(P).1.in_transfer.is_none()           // (7) not mid re-home saga (reuse directory lock)
   AND now >= rlm_quiesced_until                          // (8) orchestrator-crash quiesce (§4.5)
```

`min_dwell = spinup_cooldown_ticks + launch_ttl_ticks` (clause 4, the thrash-heed floor from the
determinism-first critique H-2/#3): a just-spun realm cannot be reaped inside its own boot+settle
window even if it reports Empty immediately (a booted-empty realm must survive at least one
demand-cadence so a warping-in occupant's demand can land).

**Two-phase execution (BUG-C fix):**

```
tick T:   teardown_ready(P) first becomes continuously true ⇒ set draining_since = T. EMIT NO KILL.
                                                              (state: Draining — the veto window)
tick T..T+drain:  each tick, if ANY of:
                    - a demand for P arrived (demanded_recently flips true), OR
                    - empty_confirmed flips false (a new occupant → new non-Empty), OR
                    - a crossing/spawn targets P (in_transfer set, or a spawn demand)
                  ⇒ clear draining_since, ABORT the teardown, back to Running. THE REALM IS RESCUED.
tick T+drain (>= teardown_drain_ticks):  if teardown_ready(P) STILL holds continuously
                  ⇒ state Reaping: emit Kill (execute §3.3).
```

This closes the stress-#2 race: the kill is never synchronous with the decision; a rescuing demand
that lands anywhere in `[T, T+drain]` aborts it. The reconciler is orchestrator-local — no new wire
arm; the "am I still empty?" fact is the realm's own continuous `Empty` self-report, which STOPS the
instant an occupant arrives (the shard emits a non-Empty child demand / ceases `Empty`), flipping
`empty_confirmed` false within one tick. (See RISK item 2 for the full tick-by-tick proof.)

**BUG-A execution (Zombie path):** independently of teardown-ready, a `zombie(P)` (head present,
owner `is_latched_dead`) is handled every sweep: force-revoke the stale head
(`dir.revoke(Realm(rid), fence)`) + `kill_realm(node)` (idempotent — likely already gone,
`AlreadyKilled` ⇒ success), then if `demanded_recently(P)` the next sweep re-spawns it (self-heal),
else retire. A Zombie is NEVER desired-alive (arm B uses `running_live`, not head-present), so it
cannot pin itself alive; and it is NEVER on the teardown-ready path (clause 1 requires `running_live`,
which excludes a Zombie), so it is not mistaken for an affirmative-Empty teardown either. This is
the exact dead-vs-Empty disambiguation stress #6 demands, and it requires the Step-3 force-revoke
(pulled forward from Step 4 — see RISK item 6).

---

## PART 2 — DATA STRUCTURES + WHERE THEY HOOK (file:line)

New pure decision code lives in **`crates/sim/src/rlm.rs`** (NEW, `vd-sim`, next to `directory.rs`
— pure, unit-testable without a `World`, per-monomorphization coverable in its own crate). The Bevy
runtime lives in **`crates/node/src/rlm_runtime.rs`** (NEW, `vd-node`, peer of `saga_runtime.rs`).
Wire types are UNTOUCHED (HR1): `RealmDemand`/`DemandVerb` already exist frozen
(`wire/src/intershard.rs:517-545`).

### 2.1 `RlmTuning` — `crates/sim/src/rlm.rs`
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RlmTuning {
    pub demand_ttl_ticks: u64,
    pub empty_grace_ticks: u64,
    pub teardown_cooldown_ticks: u64,
    pub teardown_drain_ticks: u64,       // BUG-C veto window
    pub spinup_cooldown_ticks: u64,
    pub launch_ttl_ticks: u64,
    pub reconcile_interval_ticks: u64,   // 0 = INERT
}
impl Default for RlmTuning { /* all zero = INERT */ }
impl RlmTuning {
    pub fn cloud(tick_hz: u32) -> Self { /* mirrors DirectoryTuning::cloud, directory.rs:164 */ }
    pub fn validate(&self) -> Result<(), RlmTuningError> { /* the §1.3 ordering invariant, fail-loud */ }
}
```

### 2.2 `DemandLedger` — `crates/sim/src/rlm.rs`
```rust
#[derive(Default)]
pub struct DemandLedger {
    cells: BTreeMap<RealmPath, LedgerCell>,   // keyed by the collision-free lineage PATH (never lowered())
}
pub struct LedgerCell {
    pub coord: RealmCoord,                     // full lineage — for spawn (profile_kind) + ancestor walk
    pub last_demand_tick: UniverseTick,        // SpinUp|KeepAlive|spawn (ancestor-closure refreshes chain)
    pub last_empty_tick: Option<UniverseTick>, // most recent Empty
    pub first_empty_tick: Option<UniverseTick>,// start of the CURRENT empty streak (grace anchor; arm C)
    pub spawn_watermark: UniverseTick,         // last tick a Spawn intent was emitted for P (min_dwell + cooldown)
    pub teardown_watermark: UniverseTick,      // last tick a Kill was executed for P
    pub draining_since: Option<UniverseTick>,  // BUG-C: continuous teardown-ready start; None = not draining
    pub last_fence: Fence,                     // freshest demand's parent_fence (audit; Step-4 auth)
}
```
Bounded: `gc_ledger` drops cells that are `Retired` (no head, not launching, not demanded, past all
cooldowns) each sweep — prevents the unbounded-ledger footgun.

Fold rules (all monomorphic in `record_demand`):
- `SpinUp`/`KeepAlive`/spawn → `last_demand_tick = max(last_demand_tick, tick)`; **clear the empty
  streak IFF `tick >= first_empty_tick`** (BUG determinism #8: a stale reordered SpinUp must NOT
  clear a fresher Empty). Clearing sets `first_empty_tick = None`, `last_empty_tick` retained for
  audit but `empty_confirmed` reads the streak.
- `Empty` → `last_empty_tick = max(last_empty_tick, tick)`; if `first_empty_tick.is_none()` set
  `first_empty_tick = tick` (streak start; a re-asserted Empty does NOT advance it).
- `TearDown` verb (Step 2 never sends it — `stub.rs:4384` debug_assert) → no-op refresh (does not
  refresh `last_demand_tick`; the reconciler is the SOLE kill authority).

### 2.3 `LaunchLedger` + `LifecycleAction` — the decide/execute boundary

```rust
// crates/sim/src/rlm.rs — the PURE decision output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifecycleAction {
    SpinUp   { coord: RealmCoord },
    Kill     { path: RealmPath, node: NodeId, fence: Fence },  // node+fence from the head
    ForceReap{ path: RealmPath, node: NodeId, fence: Fence },  // BUG-A Zombie: revoke + kill corpse
}
```
```rust
// crates/node/src/rlm_runtime.rs — the launch bookkeeping (BUG-B: per-COORD, a SET, never one ticket).
struct LaunchLedger {
    // every minted node per coord, so a re-issue never orphans; drained when the head appears or the
    // node dies in spawner.live(). Deterministic BTreeMap.
    minted: BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>>,
    fail_streak: BTreeMap<RealmPath, u32>,   // launch-failure backoff (§3.4)
}
```

### 2.4 `RlmReconcilerRes` — `crates/node/src/rlm_runtime.rs`
```rust
#[derive(Resource)]
pub struct RlmReconcilerRes {
    ledger: DemandLedger,
    launches: LaunchLedger,
    tuning: RlmTuning,
    spawner: Box<dyn RealmSpawner + Send + Sync>,   // THE execute seam (mem twin ↔ k8s), like StoreRes
    rlm_quiesced_until: UniverseTick,                // crash-recovery freeze (mirror liveness_quiesced_until)
    // honesty counters (surfaced in admin_snapshot; 0 on a quiescent run)
    pub spins_requested: u64, pub spins_failed: u64,
    pub teardowns_reaped: u64, pub force_reaps: u64,
    pub undecodable_demands: u64, pub desired_gauge: u64, pub running_gauge: u64,
}
```
`spawner: Box<dyn RealmSpawner + Send + Sync>` mirrors `StoreRes(Box<dyn Store + Send + Sync>)`
(`saga_runtime.rs:449`-region, inserted at `orchestrator.rs:122`). The trait is object-safe with NO
per-monomorphization coverage debt (`io/mod.rs:444-447` says so explicitly). `RealmSpawner`:
`spawn_realm(&coord, at_tick) -> Result<NodeId, SpawnError>` (`io/mod.rs:452`),
`kill_realm(node) -> Result<(), SpawnError>` (`:455`); `SpawnError::{UnknownNode, AlreadyKilled}`
(`:433-439`) — both are `kill_realm` outcomes (the doc at `:430-431` confirms `spawn_realm`'s
`Result` is the future-placement seam).

### 2.5 Persistence key family (RESERVED, INERT in Step 3)
Extend `StoreKey` at `crates/node/src/saga_runtime.rs:88-95` with `Rlm(RealmPath)` + `const RLM: u8
= 6` (next free tag after `ABORT_REPLY = 5`; append-only frozen discriminant). **Step 3 does NOT
write it.** The demand ledger is RAM-only and self-heals: on restart the shards re-assert every
demand (`ReDriven`, `intershard.rs:512`) and the durable Directory is re-read; the CAP freeze (§4.5)
blocks premature teardown until demands re-accrue. Reserving tag 6 lets Step 4 add a durable ledger
snapshot with zero wire/key migration if soak testing finds a restart strand.

### 2.6 Registration + chain + admin hooks (file:line)

| Piece | File:line | Change |
|---|---|---|
| Config field | `orchestrator.rs:23-43` (`OrchestratorConfig`) | add `pub rlm: RlmTuning` |
| Spawner param | `orchestrator.rs:84-89` (`register_orchestrator_with_store`) | add `spawner: Box<dyn RealmSpawner + Send + Sync>`, mirroring the `store` param; `register_orchestrator` (`:71`) defaults it to `MemSpawner` |
| Resource inserts | `orchestrator.rs:117-122` block | `world.insert_resource(RlmReconcilerRes::new(cfg.rlm, spawner))` |
| Demand ingest + reconcile system | `orchestrator.rs:127-133` chain | insert (§2.7) |
| Actual-set reads | `directory.rs:644` `head`, `:650` `entries` | read-only `Res<DirectoryRes>`; filter `DirectoryKey::Realm(_)` (same scan as `reap_lapsed_leases`, `saga_runtime.rs:2196`) |
| Liveness read | `saga_runtime.rs:361` `is_latched_dead` via `SagaRuntimeRes.liveness` | BUG-A |
| Grant/revoke | `directory.rs:450` `grant`, `:516` `revoke` | staged into `dir.0` dirty set |
| Admin view | `orchestrator.rs:280-311` `admin_snapshot` | parallel `RlmReconcilerRes` block: desired/running gauges, launching set, draining set, force-reap + fail counters (the 2am-curl) |
| StoreKey reserve | `saga_runtime.rs:88-103` | `Rlm(RealmPath)` + tag 6 (INERT) |

### 2.7 Chain placement (the ONE ordering decision)

The barrier is the **tail of `drive_sagas`** (`saga_runtime.rs:2415-2449`: drains `pending_writes`
then `dir.0.take_dirty()` then ONE `store.0.commit()`). The reconciler must (a) read this tick's
just-applied directory heads, (b) read the liveness tracker (in `SagaRuntimeRes`), and (c) stage its
grant/revoke into the SAME barrier for persist-before-effect.

**DECISION — split the barrier (the robustness-first + determinism-first recommendation, over the
simpler "run before drive_sagas" option):** extract the barrier body (`saga_runtime.rs:2406-2449`,
which today runs `reap_lapsed_leases` → `process_rehome_starts` → drain+commit) into its own final
system `commit_barrier`. New chain at `orchestrator.rs:127-133`:

```
(advance_and_broadcast_clock,
 record_realm_demands,          // NEW: inbox → ledger (before serve so this-tick demands are fresh)
 serve_directory,
 drive_sagas_core,              // drive_sagas minus its barrier tail (reaper + rehome-arm STAY here, unchanged order)
 reconcile_realm_lifecycle,     // reads post-CAS heads + liveness; stages grant/revoke + calls spawner
 commit_barrier).chain()        // the ONE fsync: pending_writes + dir.take_dirty() + clock ceiling
```

The reaper (`reap_lapsed_leases`) and rehome-arm (`process_rehome_starts`) STAY at the tail of
`drive_sagas_core` in their existing order (they must run before the reconcile so a reaper revoke /
armed saga is visible), and their writes accumulate in `dir.0.dirty` + `runtime.pending_writes`
which `commit_barrier` drains — so the split is a MOVE of the drain/commit only, not a reorder of the
mutation phases. This preserves the D-6 persist-before-effect and COMP-2 anti-zombie guarantees
(the barrier tests cover the moved drain unchanged). **The barrier split is its own slice (3e) with
its own review** — the determinism-first critique correctly flagged it as touching the
most-safety-critical function; it is NOT a one-line "mechanical" change.

**Where the spawner call happens:** inside `reconcile_realm_lifecycle` (the `RealmSpawner` methods
are `&self` interior-mutable — no `&mut` resource contention). The spawner call must be
non-blocking (the real k8s launcher fires an async "ensure pod" and returns a provisional `NodeId`
immediately — Step 5 contract, §3.4). The grant/revoke it stages ride `commit_barrier`, so the
directory delta is durable in the same fsync; the pod side-effect is (unavoidably)
effect-before-persist, which is why BUG-A liveness + BUG-B `live()` are load-bearing for restart
safety (§ RISK item 7).

---

## PART 3 — THE DECIDE/EXECUTE SEAM + ANCESTOR-CLOSURE + SOURCE-AGNOSTIC + LAUNCH-FAILURE

### 3.1 DECIDE — pure, in `vd-sim` (`reconcile`)

```rust
// crates/sim/src/rlm.rs — NO I/O, NO World, NO RealmSpawner, NO wall clock. Deterministic.
pub fn reconcile(
    ledger: &DemandLedger,
    dir: &DirectoryCore,                                  // read-only head()/entries()
    liveness_dead: &dyn Fn(NodeId) -> bool,              // is_latched_dead, passed as a fn (no vd-node dep)
    launch_live: &BTreeMap<RealmPath, BTreeMap<NodeId, UniverseTick>>, // spawner.live() attributed by coord
    tuning: &RlmTuning,
    now: UniverseTick,
    quiesced_until: UniverseTick,
) -> (Vec<LifecycleAction>, LedgerDelta)                 // actions + draining_since mutations (applied by caller)
```
1. Build the raw desired set `{P : desired_alive(P)}` (§1.5), then **ancestor-close** it (§3.2).
2. For each `zombie(P)` → `ForceReap` (BUG-A), regardless of desire.
3. For each desired-closed P with no head and no live launch, past `spinup_cooldown` (or hard
   demand): `SpinUp`.
4. For each `running_live(P)`: run the two-phase teardown (§1.6) — set/clear `draining_since` in the
   delta; emit `Kill` only when the drain has elapsed and readiness still holds.
5. Return actions in `RealmPath`-sorted order (deterministic); `SpinUp` ordered ancestor-first,
   `Kill` ordered child-first.

All branching is in monomorphic helpers (`desired_alive`, `empty_confirmed`, `teardown_ready`,
`running_live`, `zombie`, `launching`) — the ECS wrapper is a branchless shim. Predicates use
bitwise `&`/`|` (not short-circuit) so no uncoverable false-arm (mirrors `should_reap`
`saga_runtime.rs:2113`, `reap_lapsed_leases` `:2164`).

### 3.2 Ancestor-closure (hardened per critique #1/#4)

A demand for P pulls P's entire `Universe→…→P` chain alive (each parent is its children's frame
authority). Mechanism: after ingest AND after computing the arm-(B)-desired set, walk `coord.parent()`
(`realm_coord.rs:69`, total, `None` at root) for **every desired path** and refresh the implied
demand on each ancestor. **CRITICAL correction over the simplicity-dry design:** closure is
recomputed EVERY tick from the FULL desired set (including arm-B running-occupied-but-undemanded
realms), NOT only on demand ingest. The simplicity-dry design declared the live descendant-scan
"redundant"; critique #1 proved that strands a parent under a live-but-undemanded child. So:

```rust
fn ancestor_close(desired: &BTreeMap<RealmPath,RealmCoord>) -> BTreeMap<RealmPath,RealmCoord> {
    let mut closed = desired.clone();
    for coord in desired.values() {
        let mut cur = coord.clone();
        while let Some(p) = cur.parent() {               // bounded by lineage depth (<= 6), no cycle
            closed.entry(p.path().clone()).or_insert_with(|| p.clone());
            cur = p;
        }
    }
    closed
}
```
- Termination: `parent()` → `None` at path len 1 (`realm_coord.rs:69-76`).
- Teardown guard: `teardown_ready` clause (6) `!has_desired_descendant(P, closed)` — a `System`
  parent of a demanded `Planet` is in the closure ⇒ not ready. Bottom-up teardown for free.
- Spin-up ordering: `SpinUp` intents sorted ancestor-first (a shorter path prefix sorts before its
  extension) — the building boots before the apartment; the admission handshake already tolerates a
  not-yet-ready dest (`stub.rs:1630` realm-lease gate defers+re-drives).
- **Mid-teardown ancestor race (critique #4/H-3a):** because closure is recomputed every tick and a
  parent is only ever `Draining` (never synchronously killed — BUG-C), a demand for a deep child
  arriving while the parent is `Draining` re-adds the parent to the closure ⇒ `has_desired_descendant`
  true ⇒ `draining_since` cleared ⇒ the kill is aborted before it ever executed. The two-phase drain
  and the every-tick closure recompute together dissolve this race. Assumption: demand coords carry
  the full Universe-rooted lineage (producer contract `realm_coord.rs:26`; the AoI producer builds via
  `own_coord.child(level)` `stub.rs:4374`, `Empty` via `own_coord` `stub.rs:4360` — both full).

### 3.3 EXECUTE — `drive` in `rlm_runtime.rs` (the ONLY impure step)

```rust
for action in actions {
  match action {
    SpinUp{coord} => {
        spins_requested += 1;
        // backoff: skip if fail_streak(coord) says we're within a backoff window (§3.4)
        match spawner.spawn_realm(&coord, now) {
            Ok(node) => { launches.minted[coord.path()].insert(node, now);   // per-coord SET (BUG-B)
                          ledger.cell(coord.path()).spawn_watermark = now;
                          fail_streak.remove(coord.path()); }
            Err(_)   => { spins_failed += 1; *fail_streak.entry(coord.path()).or_default() += 1;
                          // LEVEL-TRIGGERED: no state staged; re-derived next sweep with backoff. }
        }
    }
    Kill{path,node,fence} => {
        match spawner.kill_realm(node) {
            Ok(())                                     => { revoke(path,fence); reaped(path); }
            Err(UnknownNode|AlreadyKilled)             => { revoke(path,fence); reaped(path); } // idempotent-converged
        }
    }
    ForceReap{path,node,fence} => {                    // BUG-A
        force_reaps += 1;
        let _ = spawner.kill_realm(node);              // corpse; AlreadyKilled ⇒ fine
        revoke(path,fence);                            // force-revoke the stale head so it stops reading "running"
        // ledger cell kept if still demanded ⇒ next sweep re-spawns; else gc.
    }
  }
}
// launch reconcile: drain launches.minted entries whose head appeared (→ Running) OR whose node
// vanished from spawner.live() past launch_ttl (→ dead launch; leave it removed so SpinUp re-fires).
```
`revoke` uses the exact current fence from the head; a `RevokeOutcome::Refused` (fence race / a
concurrent re-home lock) is handled exhaustively — **leave the head + retry next sweep** (do NOT
retire the ledger cell), so a refused revoke self-heals rather than desyncing (critique H-1). A
`Kill` whose realm is mid-`in_transfer` is blocked at the DECIDE layer by clause (7), so revoke sees
a stable key.

### 3.4 Launch-failure policy (critique M-2/#10/HIGH-5 — resolved, not hand-waved)

- **Synchronous refusal** (`spawn_realm → Err`): count, increment `fail_streak`, stage nothing.
  Re-derived next sweep. Retry is throttled by an **exponential backoff on `fail_streak`**:
  skip re-spawn until `now - spawn_watermark >= spinup_cooldown_ticks * 2^min(fail_streak, cap)`.
  This is the give-up-gradient: it never hard-stops (a transient cluster-full self-heals when
  capacity returns) but it never hammers either. `fail_streak` past a `stuck_threshold` flips a loud
  `spins_failed`-driven admin alarm (visible, never silent) so a permanently-unschedulable realm
  (bad profile, image-pull) surfaces to ops — and Step 6's login path can surface "spawn failed,
  retry later" to the player instead of an infinite hang.
- **Silent async failure** (`Ok(node)` but the pod never self-grants a head): the `launches.minted`
  entry for that node ages out of `spawner.live()` (a real k8s launcher's `live()` lists pods; a
  crash-looped pod leaves the set). Once no live node remains for P past `launch_ttl_ticks`, P is no
  longer `launching` ⇒ `SpinUp` re-fires (a genuinely fresh incarnation — F2 monotone ids mean the
  dead id is never resurrected, `mem.rs:288`). Because we track a **set** of minted nodes per coord
  (not one overwriting ticket), a re-issue never orphans the prior node — the launch reconcile
  reaps any minted node that both (a) fell out of `live()` and (b) never produced a head.
- **Real k8s launcher contract (Step 5, called out to not foreclose):** `spawn_realm` fires a
  non-blocking "ensure pod" (StatefulSet/pod parameterized per-coord via the existing
  `realm_shard_env` helper, `bins/lib.rs`), returns a provisional `NodeId` immediately, and MUST be
  idempotent per `coord.path()` (a second `spawn_realm` for a coord already launching returns the
  existing node). `live()` MUST derive from the cluster (list pods by label) so a rebuilt
  orchestrator recognizes its pre-crash pods (the BUG-A/#7 restart bridge). The mem twin is
  synchronous + idempotent-via-`live()`.

### 3.5 Source-agnostic demands (AoI + player-spawn) — structural, zero-rework

The reconciler consumes `RealmDemand` verbs with NO source discriminator. Both producers fold into
the SAME `record_demand`:
- **AoI (Step 2, LIVE):** `evaluate_realm_aoi`/`aoi_decide` → `push_demand`
  (`stub.rs:4280-4388`) emits `SpinUp`/`KeepAlive`/`Empty` over `InterShardFlow::RealmDemand`.
- **Player-spawn (Step 6, seam planted now):** login looks up a stored `SpawnPoint { realm: RealmCoord,
  position }` (a generic per-player location — space port today, purchased apartment later, placeable
  in ANY area of ANY realm) and emits the IDENTICAL `RealmDemand { child: spawn.realm, verb: SpinUp,
  parent_fence: <login authority>, universe_tick: now }`. Indistinguishable from an AoI `SpinUp` to
  `record_demand` — same `last_demand_tick` refresh, same ancestor-closure, same union.

The reconciler keeps the UNION alive (arm A is true if ANY demand is fresh). Step 6 adds only a
`RealmDemand` PRODUCER (login) + a `SpawnPoint` store + the admission handoff — it touches ZERO
reconciler code. The dead `needs_provision`/`AwaitProvision`/`ProvisionReady` saga gate
(`saga.rs`) is the Step-6 wiring point: the reconciler is the `ProvisionReady` CAUSE (it makes the
realm `Running`); the login saga is the consumer; they meet at the Directory head (`is_running`).
Step 3 leaves `needs_provision` inert (unchanged), foreclosing nothing.

---

## PART 4 — SLICE PLAN 3a..3f

Each slice: INERT-by-default so it lands byte-identical on the walk/canonical gate.

### 3a — The pure decision kernel
- **WHAT:** `RlmTuning` (Default-inert, `cloud`, `validate`), `DemandLedger`/`LedgerCell`,
  `LifecycleAction`, and the pure predicates `desired_alive`, `empty_confirmed`, `running_live`,
  `zombie`, `launching`, `teardown_ready`, plus `record_demand` (the arm-C-hardened fold) — no ECS,
  no I/O.
- **WHERE:** `crates/sim/src/rlm.rs` (NEW); `pub mod rlm;` in `crates/sim/src/lib.rs`.
- **TESTS:** truth tables for each predicate (every `Option` arm × fresh/stale/cooldown/dead-latched
  combo, `assert_eq!` on the bool, bitwise `&`/`|`); the arm-C same-tick Empty+SpinUp inversion case
  (SpinUp at `>= first_empty_tick` ⇒ NOT empty_confirmed); the reordered stale-SpinUp-after-Empty
  case (does NOT clear the streak); `min_dwell` floor (a booted-empty realm is not ready inside its
  window); `validate` fail-loud on each ordering violation.
- **HR5 note:** all monomorphic, both INERT and cloud tunings exercised in-slice (the INERT-zero
  trap: with `demand_ttl==0` a `<=0` comparison's other arm is dead — so cover a non-zero tuning
  here, NOT deferred to 3f). Nested `Option` unwraps get explicit `None`-arm tests. Split every
  multi-clause `&&` into named monomorphic helpers with `&` (the `should_reap`/`reap_lapsed_leases`
  discipline, `saga_runtime.rs:2113`/`:2164`).

### 3b — Ancestor-closure + `reconcile` + two-phase teardown delta
- **WHAT:** `ancestor_close` (every-tick, full-desired-set), `has_desired_descendant`, the two-phase
  `draining_since` set/clear delta, and `reconcile()` returning `(Vec<LifecycleAction>, LedgerDelta)`.
- **WHERE:** `crates/sim/src/rlm.rs`.
- **TESTS:** depth-6 closure (all prefixes desired, idempotent shared parent, root termination);
  child-first Kill / ancestor-first SpinUp ordering (`assert_eq!` on the ordered `Vec`); a live
  descendant keeps its ancestor un-ready; the two-phase drain: ready→Draining (no Kill), a demand in
  the window clears `draining_since` (rescue), drain-elapsed→Kill; a mid-drain descendant demand
  re-adds the ancestor and aborts (critique #4); Zombie → `ForceReap` regardless of desire; run twice
  → identical output (determinism).
- **HR5 note:** the `has_desired_descendant` live-scan branches (empty-descendant / some-desired /
  self-is-leaf) each covered per realm depth; `reconcile` is a straight-line iteration shim, all
  branching in the predicates.

### 3c — Demand ingest + honesty counters
- **WHAT:** `record_realm_demands` (inbox → ledger: decode `InterShardFlow::RealmDemand` from
  `MsgClass::Saga`, per-verb fold, `undecodable_demands` on decode-fail), `RlmReconcilerRes` +
  `RlmTuning` threaded through `OrchestratorConfig`.
- **WHERE:** `crates/node/src/rlm_runtime.rs` (NEW); `orchestrator.rs:23-43,117-122`.
- **TESTS:** synthetic `InboundBox` (the `serve_directory` test pattern) of each verb: SpinUp
  refreshes + clears streak; Empty latches `first_empty_tick`; KeepAlive refreshes; corrupt Saga
  bumps `undecodable_demands` (parity with `orchestrator.rs:191`); a `Directory` op and a
  `RealmDemand` on the same `MsgClass::Saga` don't collide; freshest-wins/out-of-order convergence.
- **HR5 note:** the arm that ignores non-RealmDemand `InterShardFlow` variants (`_ =>`) is covered by
  feeding a `SagaAck`; per-verb coverage.

### 3d — EXECUTE against the mem twin (BUG-B `live()` bridge + launch ledger)
- **WHAT:** `drive` (the impure loop §3.3), `LaunchLedger` (per-coord node SET), the `spawner.live()`
  attribution + launch reconcile, the fail-streak backoff. `SpawnerRes`/spawner param + `MemSpawner`
  default (`register_orchestrator`). `drive_lifecycle` runs after `commit_barrier`.
- **WHERE:** `crates/node/src/rlm_runtime.rs`; `orchestrator.rs:71-89,117-122`.
- **TESTS:** in-process — a SpinUp demand → `MemSpawner::live()` gains the node (Launching, no head
  yet) → the reconciler does NOT re-spawn while the node is live (BUG-B: the many-tick window emits
  ONE spawn, asserted via `MemHub::node_count`, `mem.rs:296`); a rigged `FailingSpawner` → `spins_failed`
  bumps, backoff throttles retry (spawn count over N ticks bounded); Kill on an already-killed id →
  `AlreadyKilled` converged, no panic; monotone non-reuse (spawn→kill→spawn ≠ same id, `mem.rs:288`);
  Zombie `ForceReap` (a head whose owner we mark `is_latched_dead` → force-revoke + kill corpse).
- **HR5 note:** each `SpawnError` arm hit by a rigged twin; the launch-reconcile branches (head-appeared
  vs node-vanished vs still-launching) each covered.

### 3e — Barrier split + chain wire-up + reconcile system LIVE (its own review)
- **WHAT:** extract `commit_barrier` out of `drive_sagas` (`saga_runtime.rs:2406-2449`; reaper +
  rehome-arm stay in `drive_sagas_core` in order); insert `record_realm_demands`,
  `reconcile_realm_lifecycle`, `drive_lifecycle` into the chain (§2.7); reserve `StoreKey::Rlm` tag 6.
- **WHERE:** `saga_runtime.rs:88-103,2406-2449`; `orchestrator.rs:127-133`; `rlm_runtime.rs`.
- **TESTS:** golden — the split `drive_sagas_core` + `commit_barrier` is byte-identical to the old
  combined system on every existing saga/transfer scenario (no regression); ONE `commit()` per tick;
  a SpinUp's grant + a paired revoke commit in the same fsync (persist-before-effect preserved); the
  reaper's `Realm(_)` no-op arm still holds (the reconciler, not the reaper, is now the realm-revoke
  authority via `ForceReap`/`Kill`).
- **HR5 note:** the barrier move introduces no new branch (pure relocation); the D-6 barrier tests
  cover it. Review this slice in isolation (most-safety-critical function).

### 3f — Adversarial capstone: the two shutdown edges + thrash + crash + determinism, E2E
- **WHAT:** a harness fixture wiring a real orchestrator + `MemSpawner` + a real Step-2-emitting stub
  shard through the FULL loop, plus the crash/restart fail-safe and cloud tuning.
- **WHERE:** `crates/tests/` (new `realm_lifecycle_e2e.rs`), `crates/harness/` (spawner-twin plant
  via `MemHub::transport_for`, `mem.rs:300`), `docs/design/DEFERRED.md`.
- **TESTS:**
  - **No-strand (edge #1):** a player-bearing realm whose parent partitions is NEVER killed (arm B);
    a genuinely-empty realm IS reclaimed after exactly `empty_grace + teardown_cooldown + drain`.
  - **Rescue (edge #2):** a login demand landing anywhere in the drain window aborts the teardown
    (tick-by-tick), and the realm is NEVER killed then re-spun (no fresh-id thrash).
  - **Dead-vs-Empty (stress #6):** a crashed occupied shard → `is_latched_dead` → `ForceReap` +
    re-spawn (demanded) / retire (not) — NOT pinned alive, NOT torn down as Empty.
  - **Thrash-heed:** a demand flapping every tick within the bands → ZERO spawn/kill churn (assert
    `MemHub::node_count` flat + spawn/kill counts flat over a long run).
  - **Crash-recovery (stress #7):** kill-9 the orchestrator → empty ledger + surviving durable heads;
    first post-reboot tick tears down NOTHING (quiesce + arm B); demands re-accrue.
  - **Ancestor-closure:** demand a deep Area → whole chain spins up ancestor-first; drop → children
    reap child-first, System survives while any descendant demanded.
  - **Determinism:** whole suite twice in separate processes → byte-identical action + minted-id
    sequences (the `collections.rs` replay-twice gate).
- **HR5 note:** `just coverage-fast` green on `vd-sim::rlm` + `vd-node::rlm_runtime`, both tunings
  exercised; no `coverage-exemptions.toml` entry expected (the only `dyn` surface is the spawner —
  no per-instantiation debt, `io/mod.rs:444`).

---

## PART 5 — RISK REGISTER (all 12 stress items)

| # | Stress item | Resolution |
|---|---|---|
| 1 | Fall out of AoI → TTL reclaim, no parent kill | **RESOLVED.** Step 2 provably never emits `TearDown` (`stub.rs:4384` debug_assert; `(true,false)` past grace → `(None,None)`). Reclaim = `demanded_recently` false (arm A off) + `empty_confirmed` (continuous `Empty`, `stub.rs:4356`) + cooldown + drain. Critique gap (a closure-only ancestor with no occupancy loop never emits Empty) resolved by arm-B closure: an ancestor is kept alive by its live descendant's every-tick closure (§3.2), and a childless empty ancestor IS itself a running shard emitting `Empty` (every realm runs `aoi_decide`), so it reclaims. |
| 2 | Load INTO a queued-for-teardown realm → rescue, no race | **RESOLVED (BUG-C fix).** Two-phase teardown (§1.6): the kill is NEVER synchronous with the decision; `Draining` holds for `teardown_drain_ticks` during which any demand (AoI/spawn/crossing) clears `draining_since` and aborts. Tick-by-tick: T sets Draining (no kill); a rescuing demand at T..T+drain flips `demanded_recently` true (arm A) → `teardown_ready` false → `draining_since` cleared → Running. The realm is provably never killed-then-respun. |
| 3 | Thrash / limit-cycle | **RESOLVED.** Double hysteresis: `teardown_cooldown` requires continuous readiness (any disturbance resets the candidate), `min_dwell` (§1.6 clause 4) floors a just-spun realm alive through boot+settle, and the `validate` ordering invariant `demand_ttl > shard_grace + launch_ttl` composes with the shard's own `AoiConfig` grace band (`stub.rs:4382` `grace_ticks`) so the orchestrator can't tear down faster than the shard re-warms — the explicit inequality the co-hosting limit-cycle needed (`project_node_per_realm_rehome`). |
| 4 | Ancestor-closure correctness + mid-teardown races | **RESOLVED.** Every-tick full-desired closure (§3.2, NOT ingest-only) + two-phase drain: a deep demand arriving while an ancestor is Draining re-adds it to the closure → `has_desired_descendant` → abort. Spin-up ancestor-first, teardown child-first. Termination bounded by depth ≤ 6. |
| 5 | Admission during spin-up (pod starting) | **RESOLVED (Step-3 half) + deferred handshake.** The `Launching` state is first-class (BUG-B `live()`); admission waits on the head via the existing realm-lease gate (`stub.rs:1630` defers+re-drives, no drop). The client-facing wait/timeout + "spawn failed" surfacing is Step 6 (the login saga), but the failure SIGNAL originates here (§3.4 fail-streak alarm). Not foreclosed. |
| 6 | Dead-shard vs affirmatively-Empty | **RESOLVED (BUG-A fix).** A crash = ABSENCE of emission; the reaper leaves the `Realm` head (`saga_runtime.rs:2196`). The reconciler consults `is_latched_dead` (`:361`): a `zombie(P)` (head + dead owner) is NEVER arm-B-desired (arm B uses `running_live`) and NEVER teardown-ready (clause 1), so it's neither pinned-alive-forever NOR mistaken for Empty — it takes the `ForceReap` path (revoke stale head + kill corpse + respawn if demanded). Requires pulling the realm force-revoke into Step 3 (not Step 4) — justified: without it the crash leaks. |
| 7 | Orchestrator crash/restart mid-reconcile | **RESOLVED.** Level-triggered re-derivation + RAM ledger re-accrues (`ReDriven`) + durable heads survive + `rlm_quiesced_until` (§4.5, mirror `liveness_quiesced_until` `saga_runtime.rs:1394`) blocks teardown until demands return (arm B keeps running realms alive during the freeze; a fresh Empty is required to kill, so no mass-reap). No spin-up lost (re-derived); the per-coord launch SET + `spawner.live()` (crash-durable in the real launcher, Step 5) prevents double-spawn of a pre-crash pod. |
| 8 | Determinism under reorder/dup/multi-parent | **RESOLVED.** `BTreeMap<RealmPath,_>` keys (collision-free, `Ord`), `max`-fold on `universe_tick`, streak-clear only on `tick >= first_empty_tick` (the reorder hole from the critique), closure via order-free `BTreeSet` union. Two parents → one cell, union kept alive. Same-tick KeepAlive(R)+Empty(R) on one path resolved by "refresh-after-latch, streak-clear conditional on tick" (deterministic regardless of inbox order). Multi-parent `last_fence` tie broken by `max` on `(tick, fence)` lexicographic. |
| 9 | Interaction with the re-home saga (crossing INTO a spinning-up realm) | **RESOLVED (composition) + one Step-6 note.** Crossing needs a dest head (`saga_runtime.rs:1485` `dir.head(Realm(to_realm))`); a crossing into a dormant realm must ALSO drive spin-up. **Resolution:** the crossing-request producer, on finding no dest head, emits/implies a `RealmDemand::SpinUp` for the target (the source-agnostic path, §3.5) so the reconciler spins it; the saga re-drives (source-authority-retained) until the head appears. `in_transfer` (clause 7) protects a landing realm from teardown. The crossing→SpinUp emit is a Step-6 producer addition (zero reconciler change). |
| 10 | Decide/execute seam shape + launch failure | **RESOLVED.** Pure `reconcile` → `Vec<LifecycleAction>` + thin `dyn RealmSpawner` drive; identical decisions mem↔k8s. Launch failure: exponential backoff on `fail_streak` (never hammer, never hard-stop), loud stuck-alarm past threshold, silent-async-fail caught by `live()`-ages-out + re-spawn. The synchronous `spawn_realm` signature is a "future-placement seam" (`io/mod.rs:430`); the real launcher returns a provisional id + reconciles against the head (Step-5 contract, stated now so the seam isn't foreclosed). |
| 11 | "Actual = Directory lease" sufficiency (spawned-not-leased gap) | **RESOLVED (BUG-B fix).** `actual := running_live OR launching` where `launching` reads `spawner.live()` (`mem.rs:478`, built for exactly this) — the intermediate source of truth between minted and leased. Per-coord node SET (not one ticket) means a re-issue never orphans; the launch reconcile reaps minted nodes that fell out of `live()` without producing a head. No guessed timeout. |
| 12 | HR compliance + 100% coverage | **RESOLVED.** See PART 6. Kind-blind (`profile_kind()` only inside the spawner), sealed-shard wire seam (frozen `RealmDemand`, no new arm), pure monomorphic predicates with `&`/`|`, both tunings covered, `dyn` spawner = no per-monomorphization debt. |

---

## PART 6 — HR1 / HR3 / HR5 + DETERMINISM SIGN-OFF

**HR1 (sealed shards).** No new inter-shard bytes. Demands ride the existing frozen
`InterShardFlow::RealmDemand` arm (`intershard.rs:257,517-545`). The BUG-C drain is
orchestrator-LOCAL (a reconciler-side timer over the realm's OWN `Empty` self-report — no new
child handshake). The `RealmSpawner` port lives OUTSIDE the sealed shard (`io/mod.rs:444`, "a shard
cannot spin up a sibling"). BUG-A's `is_latched_dead` and BUG-B's `live()` are orchestrator-local
reads. **PASS.**

**HR3 (one tooling, kind-blind).** `reconcile`/`drive` NEVER match on realm kind. A `System` spinning
up a `Planet` and a `Planet` spinning up an `Area` traverse identical code; the only kind-derived
value is `coord.profile_kind()`, resolved INSIDE `spawn_realm` (`mem.rs`/Step-5 launcher), never in
the decision. The `match` in `drive` is on `LifecycleAction` variant, not kind. ONE reconciler, ONE
`RealmSpawner` seam, ONE demand type. **PASS.**

**HR5 (100% region+branch).** The pure kernel (`rlm.rs`) is monomorphic — no per-monomorphization
multiplication; the only generic surface is `Box<dyn RealmSpawner>` (`dyn`, not monomorphized, so no
per-instantiation region debt, exactly the `Store` precedent `io/mod.rs:444`). All predicates use
bitwise `&`/`|` (no short-circuit false-arm), tested with `assert_eq!`/`expect_err` equality (not
`matches!`). The INERT-zero trap is closed by covering BOTH inert and cloud tunings in 3a/3c/3d (not
deferred to 3f). Nested `Option` unwraps get explicit `None`-arm tests. The `drive` error arms
(`UnknownNode`/`AlreadyKilled`/revoke-`Refused`) are each reachable via a rigged twin — reachable
BECAUSE BUG-A/BUG-B give us a real killable/leasable node (the determinism-first critique's M-4 was
contingent on C-1, now fixed). **PASS (feasible; the barrier-split slice 3e carries the only
non-kernel coverage, and it's a pure relocation covered by existing D-6 tests).**

**Determinism.** `reconcile` is pure over `(BTreeMap ledger, DirectoryCore, is_latched_dead fn,
live() BTreeMap, RlmTuning, now)` → no wall clock (uses `ClockSample.universe_tick`), no RNG, no
default-hasher map (all `BTreeMap<RealmPath,_>` / `BTreeSet`). `MemSpawner` mints monotone ids
(`mem.rs:488`) and uses `BTreeMap` internally. Folds are commutative (`max` on tick; conditional
streak-clear). Output `Vec` is `RealmPath`-sorted. → identical inputs, identical actions + minted
ids, across processes; the chaos-replay-twice gate guards it. **PASS.**

---

## PART 7 — DRIFT CHECK against the full end-goal

Does anything here paint us into a corner vs the SC+Minecraft end-goal
(seamless warp, player-built apartments as spawn points, cloud/K8s, hundreds-in-one-location,
signal-heavy cross-shard blocks)?

- **Seamless warp (no toggles/loading/teleport, physical fly-by).** RLM IS the warp mechanism: B
  enters a system's AoI → `SpinUp` → ancestor chain warms BEFORE reach (the F7 predictive-distance
  term in `aoi_min_dist`, `stub.rs:4405`, already spins up ahead of arrival) → cross SOI → the
  existing re-home saga. The two-phase drain guarantees a system you're flying toward is never torn
  down under you (rescue), and the closure guarantees the whole ancestor frame is authored before
  you arrive. **No corner** — RLM realizes warp, it doesn't foreclose it.
- **Player-built apartments as spawn points.** `SpawnPoint { realm: RealmCoord, position }` is a
  generic per-player stored location, placeable in ANY area of ANY realm (space port today, purchased
  apartment later) — because the reconciler spins up an ARBITRARY demanded `RealmCoord`'s closure
  with zero kind-special-casing (HR3). Step 6 adds only the store + login producer + admission
  handoff, all against the already-built source-agnostic seam. **No corner.**
- **Cloud/K8s.** The DECIDE/EXECUTE split is the exact seam: `reconcile` is pure; `Box<dyn
  RealmSpawner>` is `MemSpawner` in test and the k8s launcher in prod (Step 5), byte-identical
  decisions. The launch-failure backoff + `live()`-from-cluster contract + provisional-id async model
  are all stated so Step 5 plugs in without reworking Step 3. The cluster "adds a node when out of
  room" is the autoscaler's job, transparent to the launcher. **No corner** — this is the whole point
  of the split.
- **Hundreds-in-one-location.** The reconciler is O(desired ∪ running) per sweep, keyed by
  `RealmPath`, sweep-throttled by `reconcile_interval_ticks`. It operates at the REALM granularity
  (systems/planets/areas), NOT per-player — hundreds of players in one area is ONE desired realm, one
  head, one lease. The density wall (D-9, byte-VOLUME) is orthogonal and already ledgered to P6. RLM
  does not add per-player state. **No corner.**
- **Signal-heavy cross-shard blocks (P6/P9).** Signals ride a DISTINCT reliability plane
  (`InterShardFlow::Signal`, reserved) over the SAME hierarchy routing fabric RLM's ancestor-closure
  walks. RLM keeps the realm shards ALIVE that signals route between; it does not touch the signal
  plane. A demanded realm's ancestor chain being alive is exactly what a signal routed up-to-LCA
  needs. **No corner** — RLM is the substrate signals ride on.
- **Multi-galaxy `lowered()` aliasing (MEDIUM-3, verified real WITHIN a galaxy).** The one honest
  constraint: `is_running` reads `dir.head(Realm(coord.lowered()))`, and `lowered()` aliases
  Universe→System(0), Galaxy→System(1) even in a single galaxy (`realm_coord.rs:230` pinned test).
  The reconciler's LEDGER is correctly path-keyed everywhere; only the Directory bridge is lossy —
  and it's lossy BECAUSE `DirectoryKey::Realm(RealmId)` is a pre-existing lossy key, NOT introduced
  by RLM. **Mitigation shipped in Step 3:** a boot/sweep tripwire that fails loud if two distinct
  live desired `RealmPath`s ever collide under `lowered()` (path-cardinality ≠ lowered-cardinality)
  — so the reconciler NEVER silently strands the wrong realm; a collision is a loud panic, not a
  data-loss. The true cure (a `DirectoryKey::Realm(RealmPath)` arm) is ledgered to P4 (the standing
  `realm_path.rs` deferral), and the reconciler is built path-keyed so flipping the Directory key
  later is a localized change. **No NEW corner** (the aliasing predates RLM), and the tripwire
  ensures Step 3's kill authority never acts on an aliased head.

**DRIFT VERDICT: NO CORNER.** RLM Step 3 as specified is the substrate for warp, spawn-points,
cloud, and signals — each downstream pillar plugs into an already-built seam (source-agnostic
demands, DECIDE/EXECUTE split, ancestor-closure, path-keyed ledger). The only constraint (lossy
Directory key) predates RLM, is guarded by a loud tripwire, and has a ledgered P4 cure. Build it.

---

## Deferred registry (DEFERRED.md entries this spec creates)

- **D-RLM-1** — real k8s `RealmSpawner` with cluster-derived `live()` + per-coord idempotent
  `spawn_realm` + provisional-id async model. Owner: Step 5. Bridged now by the per-coord launch SET
  + `spawner.live()` (mem twin).
- **D-RLM-2** — durable ledger snapshot (`StoreKey::Rlm` tag 6, reserved-inert now). Owner: Step 4,
  only if soak testing finds a restart-window strand the CAP-freeze + arm-B don't cover.
- **D-RLM-3** — `DirectoryKey::Realm(RealmPath)` (the `lowered()` aliasing cure). Owner: P4. Guarded
  now by the loud collision tripwire.
- **D-RLM-4** — demand `parent_fence` authentication (reject a forged SpinUp from a non-parent).
  Owner: Step 4. Inert tripwire now (`last_fence` recorded, not enforced).
- **D-RLM-5** — login/spawn wiring: `SpawnPoint` store + login `RealmDemand` producer +
  `needs_provision`/`ProvisionReady` admission handoff + crossing-into-dormant→SpinUp. Owner: Step 6.
  Seam built now (source-agnostic ingest), zero reconciler rework.
