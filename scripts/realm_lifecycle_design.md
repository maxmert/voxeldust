All findings confirmed against the code: F1 (crossing CAS on `ctx.subject`, disjoint from `Realm(child)`), F8 (`revoke` returns `Refused` on lock/fence, no branch in emit path), the reap-Realm-left `RealmHeldNowhere` hole (line ~2143), the liveness latch keyed by `NodeId` cleared only by `record_ack` (F2), and the clock-ungated `emit_realm_frames` author path (Finding 1 determinism). I have everything needed to write the final plan.

---

# FINAL DESIGN — Realm Lifecycle Machinery (RLM)

> ## REVISION 1 — user review (2026-07-23). These deltas OVERRIDE the sections below where they conflict.
>
> **R1 — the CHILD proposes its own shutdown (not the parent).** A parent cannot see inside a sealed
> child (HR1), so it cannot know the child is empty. The child (which hosts its own occupants) is the
> occupancy authority: when its last occupant leaves it self-reports **Empty** upward. The parent/orchestrator
> combines that report with AoI to decide teardown. This SUPERSEDES §2.4's "occupied set at commit" as the
> *primary* occupancy signal — the commit-synchronous flip (F4) remains the tight-race guard, but ground
> truth is the child's own report. `DemandVerb` gains **`Empty`** (child→orchestrator, level-triggered).
>
> **R2 — the mechanism is a RECONCILER, not an event queue (user: "extremely robust, nobody forgets").**
> The orchestrator holds a DESIRED live-set = ancestor-closure of (realms-with-an-occupant ∪ realms-in-any-AoI)
> and each supervisor tick drives ACTUAL→DESIRED: spin up anything desired-but-down, tear down anything
> up-but-undesired past grace+cooldown. ALL inputs (AoI demands, Empty reports, occupancy) are LEVEL-TRIGGERED
> (re-asserted every tick) → a dropped message self-heals next tick; no realm is ever orphaned up or down.
> - **EDGE 1 (empty but still in someone's AoI, later not):** stays DESIRED via the AoI demand → kept; when
>   AoI *also* clears it drops out of DESIRED → torn down after grace+cooldown. Re-evaluated every tick, so
>   the teardown fires whenever both conditions finally hold — a parent that once declined cannot "forget."
> - **EDGE 2 (someone loads INTO a realm queued for teardown):** occupancy flips synchronously at the
>   admit/crossing commit the orchestrator serializes → realm re-enters DESIRED → not killed. The actual kill
>   is gated on a fresh DESIRED re-check + the F1 saga-guard + the F8 revoke-refused-cancel, and the
>   orchestrator is the SINGLE serialization point for admit and kill, so they can never both win.
>
> **R3 — SEQUENCING: generic-and-right, NOT smallest-first.** Part 3's smallest-first slices are SUPERSEDED
> by 7 complete-generic layers, reconciler-central, bottom-up:
> 1. **Foundation** — realm addressing that names any level (`RealmCoord`) + the ONE lifecycle wire channel
>    (`RealmDemand` incl. `Empty`) + the sealed spawn/kill port with its in-process twin.
> 2. **Generic decision** — seed-declared per-realm wake/sleep radii (uniform factor × own extent) + the
>    per-realm AoI loop + the child's Empty self-report; proven identical on ≥2 realm kinds (HR4).
> 3. **The reconciler** — desired-vs-actual, single kill authority, ALL race guards (F1/F2/F8) + both edge
>    cases + keep-alive-from-any / teardown-only-when-empty-and-unapproached + grace + cooldown.
> 4. **Determinism + persistence** — woken == never-slept (closed-form + epoch-tagged checkpoint + clock-gate).
> 5. **Real-process spawner** — real shards on demand w/ seed-derived profile; retire the static forest.
> 6. **Bootstrap** — login → containing realm → spin the chain top-down → admit at the leaf.
> 7. **Warp E2E** — the VU acceptance test; same saga + same channel, zero warp code.
>
> **R4 — TESTING CADENCE (user):** after each step run ONLY that step's tests, at 100% region+branch on that
> step's code. Run the FULL `just gate` only ONCE, after all 7 steps land. Global requirements HOLD every
> step (HR1-6, performance/load, robustness/determinism, no-magic-numbers, sealed shards — no drift).

**One-line thesis.** The universe is a lazily-instantiated forest of sealed shards. A realm runs iff a user's Area-of-Interest reaches it. The whole mechanism is **two generic primitives bolted onto the proven saga + directory + generator + frame-authority substrate**: a *decentralized per-realm AoI loop* (policy — "I need child X live") and a *single orchestrator spawn/kill chokepoint* (mechanism). The **live fabric = the ancestor-closure of every Active leaf**, O(Σleaves × depth). Warp, bootstrap, Signals, and content-streaming are consequences, not bespoke code.

The synthesized design (D3 scale-thesis + D2 correctness-spine + D1 minimal-posture) is architecturally sound, but three of its headline proofs did **not** survive contact with the actual key/lock/liveness/latency semantics of the code. The final design below **rewrites those three proofs** and folds every CRITICAL/HIGH finding into the machinery.

---

## PART 1 — VET FINDINGS: HOW EACH IS RESOLVED

### The single root change that dissolves four findings: `RealmCoord`, not bare `RealmId`

Findings **C2, H1, M3** (and the interim-`RealmId` strain) all trace to one defect: `RealmId` cannot name Galaxy/Universe, cannot distinguish two galaxies, and discards the `RealmKindTag` the profile selector needs. **Resolution: the lifecycle/bootstrap/demand vocabulary is `RealmCoord`, a thin newtype wrapping `RealmLevel { kind: RealmKindTag, seed: u64 }` plus a `RealmPath`-derived parent chain, NOT bare `RealmId`.**

```rust
// crates/core/src/realm_path.rs — extends the existing RealmPath/RealmLevel vocabulary
pub struct RealmCoord {
    pub level: RealmLevel,          // { kind: RealmKindTag, seed: u64 } — CAN name Galaxy/Universe today
    pub path:  RealmPath,           // globally-unique lineage from root (fixes M3 aliasing)
}
impl RealmCoord {
    pub fn lowered(&self) -> RealmId;         // System(seed)/Planet(..)/GALAXY_STANDIN — only where one exists
    pub fn profile_kind(&self) -> ProfileKind; // → profile_for → NodeKind::Shard(profile)  (fixes C2)
}
```

- `RealmDemand.child` and `.parent` become `RealmCoord`, not `RealmId`. The orchestrator selects the spawned profile from `child.profile_kind()` → `profile_for()` → `NodeKind::Shard(profile)` (**C2 fixed** — the profile is carried, not lost at the `RealmId::System` collapse).
- `LiveFabric` is keyed by `RealmPath` (globally unique by construction), not `RealmId` (**M3 fixed** — no cross-galaxy aliasing; also add a proptest asserting `RealmPath` uniqueness over generated multi-galaxy forests).
- The bootstrap resolver already produces a `RealmPath` (`containing_path_for_position`), so lifecycle and bootstrap speak **one vocabulary** (**H1 fixed**).

This is a one-vocabulary change threaded through the demand arm and closure. It is the highest-leverage fix and lands in Slice 1.

---

### F1 (CRITICAL) — tear-down/crossing key-disjointness — the §6 proof was WRONG. **RESOLVED: realm un-killable while any saga names it as source/dest.**

Confirmed against code: a crossing's `commit_cas` fires on `ctx.subject` (the entity/session key, `saga_runtime.rs:953/966`), which is **disjoint** from `Realm(child)`. The design's "revoke's fence no longer matches → no-op" interlock never engages. A tear-down could revoke `Realm(child)` and `kill_realm(node)` while a crossing then commits an occupant onto the dead shard — landing in the exact `RealmHeldNowhere` hole `reap_lapsed_leases` (`:2143`) documents it cannot yet fix.

**Resolution — a realm is un-killable while it is referenced by any in-flight saga.** The tear-down path gains a guard mirroring the proven `subject_has_live_saga` (`saga_runtime.rs:2158`):

```rust
fn realm_has_inbound_or_hosted_saga(sagas: &BTreeMap<TransferId, LiveSaga>, coord: &RealmCoord) -> bool {
    let rid = coord.lowered();
    sagas.values().any(|s|
        s.ctx.to_realm == rid                      // inbound crossing (entity or transient-batch dest)
        || s.ctx.subject == DirectoryKey::Realm(rid) // the realm key itself is a saga subject
        || hosts_subject(&s.ctx, rid))              // occupant currently owned-by / crossing-from this realm
}
```

`kill_realm` for `Realm(child)` is **gated on `!realm_has_inbound_or_hosted_saga`**. A realm with an inbound crossing lock is un-killable; the crossing's CAS commits onto a live shard, and only after it settles (no saga names the realm, occupant re-homed away) can tear-down proceed. §6's proof is rewritten around the *entity-key saga guard*, not the `Realm`-key fence.

---

### F2 (CRITICAL) — NodeId reuse collides with the RAM-only `is_latched_dead` monotone latch. **RESOLVED: never reuse a NodeId; mint fresh per spawn.**

Confirmed: `confirmed_dead_latched` is a monotone RAM latch keyed by `NodeId`, cleared **only** by `record_ack` (`saga_runtime.rs:283, 336`), and `select_rehome_target` filters on it (`:1832`). A tear-down kill accrues the latch for `node_C`; reusing that id for a respawn makes the fresh realm invisible-as-rehome-target and reapable-on-first-lapse.

**Resolution — the orchestrator NEVER reuses a NodeId.** A monotone `NodeId` allocator in the orchestrator mints a fresh id per `spawn_realm` (the spawn mints a fresh sealed shard — the natural semantics). A corpse's latch can never shadow a new incarnation. The `RealmSpawner` port returns the freshly-minted `NodeId`; the placement pool tracks (node, RealmPath, incarnation) but never recycles an id whose latch is set. This also subsumes the store-side per-`RealmId` boot-counter requirement (§7) — spawn identity is `(NodeId monotone) × (RealmId boot-counter on persistent volume)`, both monotone, both crash-durable.

---

### F3 (HIGH) — bootstrap `AwaitProvision` timeout aborts the only admission path into nothing. **RESOLVED: bootstrap provisioning is retry-with-backoff (idempotent re-drive), NOT deadline-abort.**

The bootstrap chain spawn is idempotent (`LeaseGrant` for a live realm is a no-op). Resolution:

- **Provisioning re-drives, it does not abort.** The bootstrap saga's `AwaitProvision` uses a **heartbeat park**, not a hard deadline. Every re-drive interval it re-issues the (idempotent) chain spawn and re-reads `HeadRead(Realm(leaf))`. This mirrors the `ReDriven` demand class, not the destructive transfer abort.
- **`ProvisionTimeout` is reserved for a genuine `ProvisionFailed` from the spawner** — no capacity / placement impossible — which surfaces to the client as a typed "server full," distinct from "still booting." That is the ONE terminal, and it fires only on an explicit spawner failure, never on slow-boot.
- The §5 statement "session sits in `AwaitingAttach` until the leaf resolves" is now the *only* behavior for slow boot; the deadline-abort statement is deleted. The client's admission wait is **unbounded-with-heartbeat** (a keep-alive on the connection), so a mass-reconnect cold-start storm re-drives rather than thrashing retries.

### F4 (HIGH) — occupancy hand-off window re-creates the frozen-player limit cycle. **RESOLVED: `occupied` asserted SYNCHRONOUSLY at the commit the orchestrator already serializes.**

During a System→Planet crossing, the occupant leaves the System's `Dots` at the CAS; if the Planet's `occupied=true` propagates lazily via heartbeat (ttl/4), a window exists where `child_demand[C]==0 && !occupied[C]` and the closure gate permits `TearDown{C}` — killing the realm the player just entered.

**Resolution — occupancy is tied to the directory commit, not the async renew.** The orchestrator *drives* the crossing CAS (or observes it via the directory head it owns), so at the exact commit point that flips authority to `C`, it sets `LiveFabric.occupied[C.path] = true` synchronously — before any tear-down evaluation in that tick. Additionally, **`child_demand` is handed across with the crossing** (source-retained-until-CAS semantics): the parent keeps `child_demand[C] > 0` for a departing occupant until `C` confirms adoption via the commit. There is no lazy-heartbeat window. (The heartbeat `occupied` bit remains as a level-triggered *reconciliation* backstop, immune to lost messages, but it is not the primary signal.)

### F5 (HIGH) — straddler opposite-demand: single-declared-parent authority kills a child under another system's occupant. **RESOLVED: KeepAlive from ANY authorized parent; TearDown ONLY from the declared parent.**

The clean asymmetry (KeepAlive is monotone-safe — it can only extend life):

- **`RealmDemand{SpinUp|KeepAlive}`** — accepted from **any realm whose current occupant's AoI reaches the child** (authority gate = "you authentically hold an occupant near it," proven by the parent fence + the occupant's authored position). A straddler child at a shared SOI boundary is kept alive by *either* neighboring system.
- **`RealmDemand{TearDown}`** — accepted **only** from the child's generator-declared parent (`RealmPath::parent_realm`), AND honored **only** through the orchestrator's own `LiveFabric` closure check (F5 folds into F/H3 below): zero occupants AND zero live descendants AND no live KeepAliver AND the declared parent itself Live.

This makes the straddler case (the memory-mandated "fly into an orbiting planet's SOI" that must work NOW) correct: U2 in System B keeps the shared child alive even though B is not its declared parent; only the declared parent may *propose* death, and death is gated on the closure that sees U2's KeepAlive.

### H2 (HIGH) — the loop is blind to static and signal-authored children. **RESOLVED: ONE `child_placements(tick)` accessor.**

`authored_realm_snaps` iterates only `self.moving` (orbital movers); static children (`region.center`) and P6/P9 signal-authored children (thruster-driven ships) produce zero rows.

**Resolution — introduce ONE accessor** `child_placements(tick) -> Vec<(RealmId, StampedPose)>` in `stub.rs` that folds movers + static centers + (P6/P9) signal-authored children into a single list. **Both** the observer feed (`authored_realm_snaps` becomes a thin filter over it) and the new AoI loop consume it. The "one loop at every altitude" claim now genuinely includes the walk-inside-a-flying-ship case — the project headline — with no third position code-path.

### H3 (HIGH) — parent-decides-child-death is a new authority edge with no analog in the proven saga. **RESOLVED: TearDown is advisory input to the orchestrator's closure; the closure is the sole kill authority.**

`TearDown` never revokes a lease directly. It only decrements/advises the orchestrator's `LiveFabric` computation. The orchestrator **independently** confirms from `LiveFabric`: child has zero occupants AND zero live descendants AND no live KeepAliver AND the **declared parent is itself Live** (a TearDown from a parent whose own `keep_alive_until` has lapsed is dropped — a partitioned/stale parent cannot kill its child). The `LeaseRevoke` is issued by the orchestrator off its own closure state, never off the demand path.

### F8 (from the tear-down-slice batch, promoted because it's a silent-corruption one-liner) — `revoke` returns `Refused` and the emit path doesn't branch. **RESOLVED: tear-down is a saga with an explicit `RevokeRefused → cancel` arm.**

Confirmed: `revoke` returns `RevokeOutcome::Refused` on `in_transfer.is_some() || fence != fence` (`directory.rs:520`). **`kill_realm` is gated on `RevokeOutcome::Revoked`.** A `Refused` (a transient-batch inbound holds `Realm(C)`'s lock, or the fence moved) **aborts the tear-down** and restores `keep_alive_until` for re-evaluation next tick. This is one branch, but the §6 straight-line sequence omitted it. Combined with F1's saga guard, tear-down is now a proper little saga: `evaluate closure → drain+fsync → revoke → {Revoked: kill | Refused: cancel, restore}`.

---

### Determinism/persistence findings

**D-Finding-1 (HIGH) — clock gate is promised but absent on the author paths.** Confirmed: `emit_realm_frames` reads `authored_realm_snaps(.., clock.universe_tick)` with **no `clock.is_some()` gate** (`stub.rs:4076`); `evaluate_realm_boundaries` gates only on authority. A freshly-spun shard authors children at `universe_tick=0` until its first `ClockSync`. **RESOLVED: a shared `has_synced` run-condition** gates `evaluate_realm_boundaries`, `emit_realm_frames`, AND the new `evaluate_realm_aoi`, ordered `.after(observe_clock_syncs)`. An unsynced shard authors NOTHING and takes no containment/AoI decision. Slice 1's DET-1 gate drives the *live schedule* of a fresh shard through its first-sync boundary (not just the pure-function equality) and asserts zero authored snaps pre-sync.

**D-Finding-2 (CRITICAL) — cross-epoch checkpoint load silently corrupts Cat-C position (the R6 km-error class).** Confirmed: `mean_anomaly_epoch` anchors to absolute `universe_tick=0`; re-genesis resets `universe_tick` under a new `EpochId`; the network legs guard epoch (`stub.rs:2186`) but the durable checkpoint load has **no epoch tag**. **RESOLVED: every Cat-C checkpoint carries `(EpochId, universe_tick)` in its durable header; `rehydrate`-on-spin-up fails LOUD with typed `CheckpointEpochMismatch` when the checkpoint's `EpochId` ≠ the shard's current clock epoch** (mirroring `crossings_epoch_mismatch`). Slice 4's kill-9 chaos gains an explicit re-genesis cell.

**D-Finding-3 (HIGH) — `secs_since_epoch` bakes per-shard `VD_TICK_HZ`, so DET-1 assumes an unenforced cross-spawn invariant.** Confirmed: `tick_dt_s`/`tick_hz` are independent env vars; `secs_since_epoch` bakes `tick_hz` into the seconds conversion. **RESOLVED: universe seconds-per-tick becomes a `UniverseConfig`/`ClockSync`-carried constant** (the canonical `tick_hz` the ephemeris samples), decoupled from a shard's *pacer* rate. Every shard converts `universe_tick→seconds` identically regardless of local pacing. Slice 5's `process_parity` gains a cross-shard invariant test: an on-demand spawn handed a mismatched `VD_TICK_DT` is either rejected loud or produces identical authored snaps.

**D-Finding-4 (MEDIUM) — "byte-identical" is only same-build-true until SPIKE-6a.** Cross-host libm/glam drift is ledgered to SPIKE-6a (P4). **RESOLVED as a wording/scoping fix:** the DET-1 byte-equality gate is strictly **same-binary/same-host** (both twins in one process, as `MemSpawner` does); the design text says "byte-identical to an always-on one *on the same build*." No cross-host byte-diff is ever wired; the retired seed-replica oracle stays retired.

**D-Finding-9/F9 (MED→scoped honestly) — the DEMAND SET is not `(seed,tick)`-deterministic across respawn (hysteresis is history-dependent).** **RESOLVED as scope + a safety proof:** DET-1 (byte-identity) holds for **pose** (Cat A closed-form) and **checkpoint-carried Cat-C occupant state** — NOT for the demand set. AOI-2 is restated: *the demand set is a pure function of (Dots, tick, seed, AoiMembership-history)*. Across a respawn boundary the `was_live` history resets, so a spun-up parent may diverge by ≤1 demand transition from an always-on twin. **This is proven harmless:** a spurious extra `SpinUp` is idempotent (no-op if live); a missing `SpinUp` on a hovering occupant self-heals next tick (level-triggered `ReDriven`); and critically **it can never cause a kill** — only KeepAlive is emitted from the acquire edge on respawn, and TearDown requires `grace_ticks` of fresh dwell. If demand determinism across respawn is ever *required*, the `AoiMembership` bits ride the parent's own Cat-C checkpoint (seam noted, not built).

### Findings deferred with justification

- **C1 (CRITICAL — AoI loop is O(children); IS the deferred spatial index; `MAX_REGIONS=64` is boot-blocking for a galaxy).** *Partially resolved, partially deferred with a hard seam.* The loop cannot claim to scale as written. **In-scope now:** (a) the child-enumeration goes through a `children_within(parent_coord, occupant_pos, radius) -> impl Iterator<RealmCoord>` accessor on the generator, NOT a linear `realm_neighbourhood_for` scan — the accessor is spatial-index-shaped from day one (Slice 2 lands the trait with a linear-scan impl behind it, correct under sparse occupancy); (b) `MAX_REGIONS` is raised and the per-shard region bitset is documented as capping *simultaneously-live* direct children, not *generatable* children. **Deferred to D-9/D-45 (P6 spatial index):** the real spatial-index impl behind `children_within` for a galaxy of 10⁴ systems. The design is reframed honestly: *"the AoI loop is O(live children), correct under sparse occupancy; the `children_within` query is the seam to the P6 spatial index"* — it is a Slice-2 **interface** dependency, not a post-P7 afterthought, but the scaling impl is legitimately P6.
- **F6 (MED — orchestrator crash between spawn and persist-intent orphans a shard).** *Resolved by design, landed in Slice 5:* **persist-intent-then-spawn**, with the spawn **idempotent at the OS layer** — `ProcSpawner` tags every node with `(RealmPath, incarnation)` as a queryable label and is a no-op if that tuple is already running; on rehydrate the orchestrator *discovers* running shards by label and reconciles `LiveFabric` against discovered-running + directory-heads, killing an unjustified shard only after a **post-rehydrate quiesce** (extend the existing `liveness_quiesced_until` concept to lifecycle: no tear-downs for `quiesce` ticks after restart).
- **F7 (MED — cold-boot latency defeats AOI-1 `spin_up_r_m ≥ v_max·boot_ticks`; visible warp-arrival hitch).** *Mitigation in-scope for Slice 7; full fix ledgered.* AOI-1 is downgraded from "invariant" to "generator sizing constraint under bounded `boot_ticks`." **In-scope for the warp arc:** a **predictive spin-up term** — the parent emits `SpinUp` when the occupant's *velocity vector* projects entry within `boot_ticks_p99` (`pos + vel·boot_ticks_p99`), still fully generic (velocity is in the pose, no up-tree pre-warm). **Deferred:** the warm-spare-pool (bounds `boot_ticks` to peer-book + self-grant, not process cold-start) — this is the "Dormant-as-cheaper-capability" optimization, now flagged as *load-bearing for warp latency*, not a pure optimization. Slice 7 asserts the predictive term keeps the boundary stall below a named budget; the warm pool lands when warp latency demands it.
- **M1 (MED — single-root `frame_context` mid-chain-spawn).** *Judged sound, add an assertion:* each shard's region neighbourhood is closed-form-complete from seed at boot independent of which ancestors are *running*. Slice 6 adds a test: boot a leaf Area shard with NO ancestor shard running, assert its frame resolves.
- **M2 (MED — AoI `v_rel` must be the two-mover closing speed).** *Resolved in Slice 2:* `AoiConfig`'s velocity-safe ctor uses `v_rel = v_occupant_max + v_child_orbital_max` (both seed-derivable; child orbital speed is closed-form from its `OrbitalElements`). AT-1 gains a variant: a *stationary* occupant near a *fast-orbiting* child produces ≤1 spawn across a full orbital period.
- **F10, L3 (LOW).** F10 → a warp-sweep load-test cell (peak live-shard count within pool). L3 → the post-spawn cooldown is a **new per-realm timer** in `LiveFabric` reusing the `k_dwell` *constant* (not the per-entity `should_rehome` *mechanism*); documented.

---

## PART 2 — FINAL ARCHITECTURE

### (2.1) AoI-config declaration

New `Copy` field `aoi: AoiConfig` on `RealmRegion` (`crates/core/src/geometry.rs`, beside `band: ContainmentBand`). It is neither the containment band nor the SOI shell — a strictly larger *approach* sphere with its own create/destroy hysteresis, built on the verified `OverlapBand` contract:

```rust
pub struct AoiConfig {
    spin_up_r_m:   f64,  // create (outer) edge
    tear_down_r_m: f64,  // destroy (inner) edge
    grace_ticks:   u32,  // dwell below tear_down before TearDown is emitted
}
```

- **Fallible velocity-safe constructor** mirroring `OverlapBand::for_soi_velocity_safe`: rejects `spin_up ≤ tear_down` LOUD; widens `spin_up − tear_down ≥ |v_rel|·dt·K_SAFETY` with `v_rel = v_occupant_max + v_child_orbital_max` (M2). Makes anti-thrash an *unconstructible-if-violated* type invariant. `member(was_live, min_dist)` reuses the covered `OverlapBand` hysteresis verbatim.
- **Declared BY the generator, per-instance, uniform-factor** (HR3): one `InterestConfig` on `UniverseConfig` (beside `BandConfig`) with a *single uniform* `spin_up_factor`, `tear_down_factor`, `grace_ticks`, `k_safety`. `to_regions` computes `aoi = factor × Boundary::finite_extent()`, seed-perturbed. A bright system reaches farther *because its `finite_extent` is larger* — no per-kind branch. `finite_extent` is the sanctioned geometry-owned KIND match (L1). `walk_scale()` sets factors to 0 → AoI inert → today's byte-identity preserved.

### (2.2) The per-realm generic AoI LOOP

New pure-sim system `evaluate_realm_aoi` in `stub.rs`, scheduled after `evaluate_realm_boundaries`, gated on `RealmAuthority` **and** the shared `has_synced` run-condition, `.after(observe_clock_syncs)` (D-Finding-1). A read-only sibling of the containment detector: same frame algebra, same hysteresis, larger radius, output = demand not a crossing. **Occupants-optional falls out** — steps 1-2 always run; step 3 (`integrate`) already no-ops on empty `Dots`. **No mode machine.**

Branchless generic shim over a monomorphic `aoi_decide` helper (per `tlv.rs`, HR5):

```
for child_coord in children_within(my_coord, occupant_bbox, max_spin_up_r):   # C1 spatial-index seam
    p_c   = child_placements(tick)[child]                    # STEP 1: H2 unified accessor, my frame
    min_d = min over occupants of distance(occ, p_c)         # ∞ if no occupants
    live  = child.aoi.member(was_live[child], min_d)         # STEP 2: OverlapBand hysteresis
    predict = child.aoi.member(was_live[child], dist(occ + vel·boot_p99, p_c))  # F7 predictive term
    if (live||predict) && !was_live[child]: emit RealmDemand{SpinUp,   child_coord}
    if (live||predict):                     emit RealmDemand{KeepAlive, child_coord}   # ReDriven, level-triggered
    if !live && !predict:
        grace[child]+=1
        if grace[child] >= aoi.grace_ticks: emit RealmDemand{TearDown, child_coord}    # declared-parent only
    else: grace[child] = 0
# STEP 3 (occupants-optional): integrate() runs only when Dots non-empty
```

Per-child hysteresis + grace live in a new `AoiMembership(BTreeMap<RealmPath, AoiBits>)` resource, the twin of `RegionMembership`, keyed by child path, bounded by direct-children, lazily evicted. **Only DIRECT children** — lifecycle propagates DOWN one level/tick.

### (2.3) POLICY → MECHANISM seam — precisely named: **`RealmDemand` arm (wire) → `RealmSpawner` port (`sim::io`)**

**Wire seam — ONE grouped arm** appended to `crates/wire/src/intershard.rs` (the one reviewed HR1 file):

```rust
RealmDemand(RealmDemand),   // SHARD → orchestrator
pub struct RealmDemand {
    pub parent: RealmCoord,        // demanding authority (RealmPath-identified — H1/M3)
    pub child:  RealmCoord,        // carries RealmKindTag → profile_kind (C2)
    pub parent_fence: Fence,       // FencedKey idempotency + authority proof
    pub verb:   DemandVerb,        // SpinUp | KeepAlive | TearDown
    pub universe_tick: UniverseTick,
}
enum DemandVerb { SpinUp, KeepAlive, TearDown }
```

- **Classification (compiler-forced, exhaustive — G-SEALED):** `effect_class → SideEffecting{FencedKey{parent_fence + child.path}}`; `durability_class → ReDriven`. **Level-triggered, not edge-triggered** — the parent re-emits every tick the demand holds, so a dropped verb self-heals next tick. Mirrors `CrossingRequest`; the grouped-arm-shares-one-classification mirrors the five `TransientHandoff` arms.
- **Authority gate (F5/H3):** SpinUp/KeepAlive accepted from any authority whose occupant's AoI reaches the child; TearDown accepted only from `child.path.parent_realm`, and honored only through the closure check.

**Mechanism — the orchestrator chokepoint.** `crates/node/src/orchestrator.rs` gains a **Spawn Resolver** beside `drive_sagas`, calling a **frozen `sim::io` port**:

```rust
pub trait RealmSpawner {                        // frozen seam, like Store/Transport
    fn spawn_realm(&self, coord: &RealmCoord, at_tick: UniverseTick) -> Result<NodeId, SpawnError>; // mints FRESH NodeId (F2)
    fn kill_realm(&self, node: NodeId) -> Result<(), SpawnError>;
}
```

- **`MemSpawner`** (`sim::io::mem`) — plants an in-proc shard into the harness `Topology`; deterministic, virtual-clock; all proptests/chaos run against it (parallels `MemStore`).
- **`ProcSpawner`** (`vd-bins`) — wraps `spawn_node` + a **kind→ProfileKind derivation** (`coord.profile_kind()` → `profile_for` → `NodeKind::Shard(profile)`, deleting the `NodeKind::StubShard` hardcode — C2) + `realm_shard_env` + `MeshControl::update_peer_addr`. Tags each node with `(RealmPath, incarnation)` label; spawn is OS-idempotent (F6).

**Handling:** SpinUp/KeepAlive → `HeadRead(Realm(child.lowered()))`. Resolved → refresh `keep_alive_until`. Unresolved → **persist `ProvisionIntent` (durable) → place via `select_rehome_target`-style capability match (spawn fresh node if none idle) → spawn_realm → book peer at runtime**. The new shard **self-grants** its realm on first tick via `request_pending_grants` (pull-boot, no push). TearDown → advisory into closure; never a direct revoke.

### (2.4) Ancestor-closure / live-fabric

**Invariant LF-1:** a realm is at-least-Dormant iff it is an ancestor of (or equal to) some Active leaf; killable only if zero occupants AND zero live descendants AND no live KeepAliver AND declared-parent-live.

`LiveFabric` — orchestrator-private, **non-durable derived cache**, keyed by `RealmPath` (M3):

```rust
struct RealmLiveness { keep_alive_until: UniverseTick, child_demand: u32, occupied: bool, node: NodeId, incarnation: u64 }
struct LiveFabric { live: BTreeMap<RealmPath, RealmLiveness> }
```

- **Level-triggered recompute** (immune to lost messages): each supervisor tick, `child_demand` is recomputed by walking the live subtree via `RealmPath::parent_realm`. `occupied` is set **synchronously at the crossing/bootstrap commit the orchestrator serializes** (F4), with the heartbeat bit as a reconciliation backstop.
- **Tear-down gate:** honor TearDown only if `!occupied && child_demand==0 && no_live_keepaliver && declared_parent_live`, past grace + post-spawn cooldown.
- **Rebuildable on restart** from directory head + re-driven KeepAlives + discovered-running labels (F6) → stays out of the durable barrier.
- Root/relay have infinite `keep_alive_until` (the permanent floor).

**Invariant LF-2 (no orphaned authority):** no Active realm ever lacks a running parent chain to root. Proven by a Release-storm chaos cell.

### (2.5) Bootstrap admission

1. Gateway `Hello` → validate login → read stored pose (Cat-C `player_home` redb key). None → generator spawn-point default.
2. **Position → PATH:** `containing_path_for_position(seed, StampedPose) -> RealmPath` descends the seed forest level-by-level (reuses `container`/`region_signed_distance` per level), never materializing the universe. P4-owed general inversion; **P3 stand-in = the fixed `path_for_realm` book** (flagged deferral).
3. **Spawn the chain parents-first** (frame-formation ordering; each ancestor self-grants).
4. **Admit at the LEAF via `AwaitProvision`** — parks (heartbeat, **retry-not-abort** — F3) until `HeadRead(Realm(leaf))` resolves → `ProvisionReady` → `AttachSession{session, fence}` to the **leaf** → leaf takes physics authority → `Active`. Per-session `home_shard = leaf` deletes the `config.shard` hardwire (D-34). `ProvisionFailed`/`ProvisionTimeout` fires ONLY on a real spawner failure → typed "server full."
5. THEN per-realm AoI takes over for neighbors.
- Crash during bootstrap: `ProvisionIntent`s persisted; rehydrated orchestrator re-drives idempotently; session sits in `AwaitingAttach`; leaf attach is the commit.

### (2.6) Anti-thrash — unconstructible at three layers + F1/F8 race-safe tear-down

1. **Geometric hysteresis** (§2.1): `spin_up > tear_down`, velocity-safe-widened with two-mover `v_rel` (M2). Unconstructible if violated.
2. **Grace timer:** TearDown only after `grace_ticks` consecutive ticks below `tear_down_r` — a fly-through never tears down a skimmed system.
3. **Ancestor-closure + post-spawn `k_dwell` cooldown** (a new per-realm `LiveFabric` timer reusing the `k_dwell` constant — L3): kill only past the cooldown, killing the spawn→kill limit cycle at the mechanism.

**Fence-safe kill (F1/F8-corrected):** `evaluate closure → drain: signal flush + await DurabilityHandle watermark → revoke(Realm(child), held_fence)` **guarded by `!realm_has_inbound_or_hosted_saga`** (F1) and **branching on the outcome** (F8): `Revoked → kill_realm(node)` (fresh id never reused — F2); `Refused → cancel tear-down, restore keep_alive_until`. Respects the tuned split-brain window (`THETA_MAX·grace < ttl+max`). **THRASH lesson honored:** node-per-realm (never co-hosts), reuse the proven hysteresis, every spin-up/tear-down is the ONE orchestrator saga.

### (2.7) Persistence + determinism

- **Category A (`f(seed, tick)`)** — celestial/ambient. A spun-up shard computes geometry from the generator and child poses from `child_placements(tick)` at the current `universe_tick` (first `ClockSync` → `FollowerClock`; **must be synced before authoring** — D-Finding-1). Byte-identical by construction; no state load. Universe seconds-per-tick is `ClockSync`-carried (D-Finding-3), so pacing-independent.
- **Category C (checkpoint-carried)** — occupant/rapier/blocks (P4+). Load-on-up = per-realm Store B (`RedbStore` at `path = RealmId` on a persistent volume), `rehydrate`; **checkpoint header carries `(EpochId, universe_tick)`; load fails LOUD on epoch mismatch** (D-Finding-2). Checkpoint-on-down = the drain-then-kill flush. Reuses `ReHomeState::PoseOnly` (P7 adds `Snapshot(Vec<u8>)`).

**Ordering (crash-safety linchpin):** `Release honored → checkpoint written → fsync durable → LeaseRevoke honored → kill_realm`. Killed before fsync → lease NOT revoked → reaper re-homes from last durable checkpoint.

**Invariant DET-1 (byte-identical spun-up, SAME BUILD — D-Finding-4):** for any tick T and realm R, R-spun-up-at-T is byte-identical to R-run-continuously-to-T, on the same binary. Proof (Cat A): pure function of (seed, T) — proptest serializes `child_placements` of an always-on vs fresh-spun R at T, `assert_eq!` on bytes, *both twins in one process*. Proof (Cat C): checkpoint-at-T′ reconciled forward, epoch-matched; kill-9-mid-checkpoint chaos asserts respawn == last durable. **DET-1 does NOT extend to the demand set** (F9 — hysteresis is history-dependent; proven kill-safe).

### (2.8) HR compliance

HR1 ✅ (realms emit only `RealmDemand`, one reviewed arm; `RealmSpawner` outside the sealed shard; `LiveFabric` orchestrator-private). HR2 ✅ (no new transfer machinery; spin-up is a *precondition* of the one-class saga). HR3 ✅ (ONE shard binary; `profile_kind()`→`profile_for` selects the realm — never a feature match-on-kind; AoI radius = uniform factor × instance extent; `children_within`/`containing_path_for_position` descend generically). HR4 ✅ (`evaluate_realm_aoi` runs on every `ShardProfile`; G-IDENTICAL "occupant approaches child → child spins up" passes on System→Planet AND Planet→Area). HR5 ✅ (branchless shim over `aoi_decide`; `AoiConfig::member` reuses covered `OverlapBand`; fallible ctor with `expect_err` per arm; `MemSpawner` twin; equality-over-`matches!`). HR6 ✅ (VU arc drives the real shipped path).

---

## PART 3 — SLICED, GATED IMPLEMENTATION PLAN

All slices run against `MemSpawner` (virtual clock, no sockets) until Slice 5. Each is independently gate-able. Theme: **activate, don't invent.**

### Slice 1 — Smallest end-to-end demand-driven spin-up + tear-down (minimal inversion + DET-1 + clock-gate from day one)
- **GOAL:** a running parent, on an occupant entering a (hardcoded) child's AoI, causes the orchestrator to spawn that child (fresh NodeId), which self-grants its lease; `HeadRead` flips `None→resolved`. When the occupant leaves past grace, the child is torn down (revoke branched on outcome, saga-guarded). The spun-up child's tick-0 `child_placements` is byte-equal to an always-on twin, AND a fresh unsynced shard authors NOTHING pre-`ClockSync`.
- **Crates/files:** `core` (`AoiConfig` + fallible ctor + `RealmCoord` on `geometry.rs`/`realm_path.rs`), `wire` (`RealmDemand` arm + classification + conformance in `intershard.rs`), `sim` (`evaluate_realm_aoi` spin-up+teardown edges, `AoiMembership`, `child_placements` accessor unifying movers/static — H2, `has_synced` run-condition gating the author paths — D-Finding-1, `MemSpawner` twin), `node` (Spawn Resolver + `LiveFabric` skeleton + `RealmSpawner` port + monotone NodeId allocator — F2 + F1 saga-guard + F8 outcome-branch).
- **Wire/seam:** `InterShardFlow::RealmDemand`; `RealmSpawner` (`sim::io`).
- **GATE:** 1-parent-1-child harness (child OFF → head resolves within N ticks → occupant leaves → child killed via `Revoked`); DET-1 Cat-A byte-identity *both twins one process*; **fresh-shard-authors-nothing-pre-sync** test; F1 (inbound-saga → realm un-killable); F8 (`Refused` → tear-down cancels); F2 (killed NodeId never reused, latch can't shadow); G-SEALED exhaustive classification; 100% region+branch on the new sim/core code.
- **DEMONSTRATES:** the inversion of the static forest — a realm exists because a user needs it, and stops when nobody does — with the three CRITICAL races closed and determinism pinned.
- **REUSE:** `CrossingRequest` pattern, `spawn_node`/`realm_shard_env`, `request_pending_grants` self-boot, `OverlapBand` hysteresis, `select_rehome_target`/`profile_for`, `subject_has_live_saga` (guard template), `RevokeOutcome`.

### Slice 2 — Generic AoI loop + per-instance generator config + spatial-index seam (HR4)
- **GOAL:** AoI radius is generator-authored per-instance (uniform factor × `finite_extent`, seed-jittered, two-mover `v_rel` — M2); the loop scans children via `children_within` (spatial-index-shaped accessor, linear impl — C1 seam); hysteresis + grace + predictive term (F7) live.
- **Crates/files:** `core` (`InterestConfig` on `UniverseConfig`, `aoi` in `to_regions`, `children_within` accessor with linear impl + doc-flagged P6 index dependency, `v_rel` from `OrbitalElements`), `sim` (loop generalized over `children_within`, grace dwell, predictive term), `tests`.
- **Wire/seam:** `RealmRegion.aoi`; `children_within` generator query (the C1/D-9 seam).
- **GATE:** G-IDENTICAL fixture on System→Planet AND Planet→Area (differ only in seed radii); walk-scale byte-identity preserved (factors 0 → inert); AOI-1 (radius-nesting proptest), AOI-2 (demand determinism modulo hysteresis-history), AT-1 + AT-1-orbital-variant (stationary occupant vs fast-orbiting child → ≤1 spawn/period — M2).
- **DEMONSTRATES:** one generic loop, radii declared by the generator, running identically on ≥2 shard kinds; a fast approach spins up ahead of arrival.
- **REUSE:** `realm_neighbourhood_for` (behind `children_within`), `child_placements`, `BandConfig`/`OverlapBand` shape, `finite_extent` (L1).

### Slice 3 — Tear-down + ancestor-closure + anti-thrash (full)
- **GOAL:** an unoccupied child past grace with no live descendant AND no live KeepAliver AND live declared-parent is killed + lease-revoked; a child with a live grandchild is not; the straddler child kept alive by either neighbor (F5); `LiveFabric` `child_demand` level-recompute live; `occupied` set synchronously at commit (F4); no boundary flap.
- **Crates/files:** `node` (`LiveFabric` full level-triggered recompute keyed by `RealmPath`, declared-parent-only TearDown authority — H3, synchronous `occupied` at commit — F4, KeepAlive-from-any / TearDown-from-declared asymmetry — F5, post-spawn `k_dwell` per-realm timer — L3, drain-then-kill saga — F1/F8), `sim` (occupancy hand-off with the crossing — F4), `bins` (`kill_realm`), `tests`.
- **Wire/seam:** `RealmDemand{TearDown/KeepAlive}` authority gate + `LeaseRevoke` fence discipline (no new arm for occupancy).
- **GATE:** LF-1/LF-2 + AT-1 chaos (Release-storm; hover-at-boundary no-flap; grandchild pins parent; straddler kept alive by the non-declared neighbor — F5; stale/partitioned declared-parent TearDown dropped — H3; fence-safe revoke no split-brain; F4 hand-off no frozen-player window).
- **DEMONSTRATES:** the fabric self-prunes without thrash; the memory's frozen-player limit cycle cannot recur at the lifecycle layer; the straddler case works.
- **REUSE:** `should_rehome` `k_dwell` (constant), `self_fence_lapsed_realm`, `DirectoryTuning::cloud()`, `reap_lapsed_leases` fence discipline, `subject_has_live_saga`.

### Slice 4 — Determinism gate + Cat-C checkpoint (load-on-up / checkpoint-on-down) + epoch guard
- **GOAL:** spun-up byte-identical to always-on (Cat A proven, Cat C `PoseOnly` proven, `Snapshot` seam planted); per-realm Store B open/rehydrate with `(EpochId, universe_tick)` header + LOUD `CheckpointEpochMismatch` (D-Finding-2); checkpoint-durable-before-revoke wired; per-`RealmId` boot-counter on persistent volume.
- **Crates/files:** `io-prod` (Store B `path=RealmId`, epoch-tagged checkpoint header, RealmId-keyed boot-counter), `sim`/`node` (shard-side `rehydrate` with epoch check, checkpoint-on-drain, persist-before-revoke gate), `core` (universe seconds-per-tick on `ClockSync`/`UniverseConfig` — D-Finding-3), `tests`.
- **Wire/seam:** durable-checkpoint-ack gate on `LeaseRevoke`; `ReHomeState::Snapshot` (P7 additive); `ClockSync` carries canonical `tick_hz`.
- **GATE:** DET-1 (Cat A, same-build) proptest; kill-9-mid-checkpoint chaos (respawn == last durable, no vanished occupant) **+ a re-genesis cell** (new `EpochId` → checkpoint load fails LOUD, not silent teleport — D-Finding-2); CrashLoopBackOff incarnation-monotonicity; cross-`tick_hz` invariant (mismatched `VD_TICK_DT` rejected loud OR identical snaps — D-Finding-3).
- **DEMONSTRATES:** a realm never needs to have been always-running; a re-genesis can't silently corrupt a player's position (the R6 km-error class killed at the seam).
- **REUSE:** `RedbStore`/`DurabilityHandle`, `rehydrate`, `CeilingClock::recover`, `FollowerClock`, `BootCounter`, `crossings_epoch_mismatch` guard pattern.

### Slice 5 — Prod process spawner + refactor the static launcher + profile selection
- **GOAL:** `ProcSpawner` wraps `spawn_node`, selects the profile from `coord.profile_kind()` (deleting `NodeKind::StubShard` hardcode — C2), tags nodes with `(RealmPath, incarnation)` (OS-idempotent — F6); the orchestrator spawns real `vd-shard` processes on demand; `vd-devcluster` boots only the root chain and lets AoI grow the rest (`--static-forest` compat retained).
- **Crates/files:** `bins` (`ProcSpawner`, kind→ProfileKind derivation, label-tagged idempotent spawn, runtime peer-book via `MeshControl::update_peer_addr`, incremental `set_roster`/clock-peers, persist-intent-then-spawn + post-rehydrate quiesce — F6).
- **Wire/seam:** `ProcSpawner` impl of `RealmSpawner`; runtime peer-book re-plumb; `NodeKind::Shard(profile)` in the boot path.
- **GATE:** `process_parity.rs` extended — an on-demand-spawned shard byte-identical to a statically-spawned one; a spawned Galaxy carries the `signal_relay` capability (C2); kill-9 the orchestrator mid-spawn → rehydrate discovers running shards by label → re-drive, no orphan/double (F6).
- **DEMONSTRATES:** real OS processes appear/vanish on demand with the correct capability profile; the Signal corner is uncornered (intermediate realms are live *with the relay capability*).
- **REUSE:** `spawn_node`, `realm_shard_env`, `Cluster`/`kill_and_reap`, `MeshControl::update_peer_addr`, `profile_for`, `liveness_quiesced_until` concept.

### Slice 6 — Bootstrap admission (deep-place → path spin-up → leaf admit)
- **GOAL:** connect with a stored deep pose → orchestrator computes the path → spins the chain parents-first → `AwaitProvision` (retry-not-abort — F3) → admits at the leaf; per-session `home_shard`; `Resume` works; `config.shard` hardwire deleted.
- **Crates/files:** `core` (`containing_path_for_position` — P4 general inversion; P3 `path_for_realm` book stand-in), `connection-plane`/`gateway` (Bootstrap Resolver, `home_shard`, un-refuse `Resume`), `node` (`AwaitProvision` live with heartbeat-park + `ProvisionFailed`-only-on-real-failure — F3, `provision_deadline_ticks`), `io-prod` (`player_home` key family).
- **Wire/seam:** the orchestrator Bootstrap Resolver (D-34); `AwaitProvision`/`ProvisionReady`/`ProvisionFailed` lit for bootstrap only.
- **GATE:** connect-into-a-deep-Area E2E (chain spins root→leaf, admit at leaf, AoI takes over); crash-during-bootstrap rehydrate + re-drive (no half-admitted player); slow-cold-chain re-drives, does NOT abort (F3); leaf-Area-frame-resolves-with-no-ancestor-shard-running (M1); D-34 session-leak fix verified.
- **DEMONSTRATES:** you connect deep into the universe and the exact ancestor chain lazily materializes to admit you — the bootstrap consequence of the lifecycle mechanism.
- **REUSE:** `RealmPath`/`ancestor_realms`/`lineage_seeds`, `path_for_realm` (P3 stand-in), the whole `AwaitProvision` FSM, `AttachSession`, `RedbStore` checkpoint.

### Slice 7 — VU-arc warp E2E (HR6 acceptance)
- **GOAL:** `vdctl` flies an agent avatar out of System A across the Galaxy into System B; B spawned-on-approach (predictive term keeps the boundary stall under a named budget — F7), A torn-down; wgpu readback captures dot-shrink/dot-grow; `runs/` manifest records spawn/kill/re-home; assert the SAME saga + SAME `RealmDemand` arm carry it (no bespoke warp code).
- **Crates/files:** `client` (pure renderer, RealmRegistry feed — unchanged), `bins`/`harness` (bootstrap-spawn replaces static Forest boot), `tests`.
- **GATE:** composed warp E2E over live QUIC; seamless spawn/despawn (no toggle/loading/teleport); per-`RealmId` high-water prevents the warp freeze bug; F7 boundary-stall-under-budget; F10 warp-sweep load cell (peak live-shard count within pool).
- **DEMONSTRATES:** warp is lifecycle + transfer composing — fly past a system that pops in on approach and winks out behind you, seamlessly, with zero warp-specific code.
- **REUSE:** the entire re-home saga + RealmSnap observer pipeline + `RealmView` + VU arc plan.

---

## PART 4 — EXPLICIT DEFERRALS (each with its seam)

1. **Dormant-as-a-cheaper-capability optimization** (v1 runs a full server per realm even when unoccupied). *Now partly load-bearing (F7):* the warm-spare-pool that bounds `boot_ticks` to peer-book + self-grant is the concrete first form. **Seam:** the `RealmSpawner` port already abstracts spawn cost — a warm-pool impl slots behind it with no caller change; `LiveFabric.node/incarnation` already tracks the reusable slot.
2. **Two-levels-ahead pre-warm** (report occupant-near-my-edge up to parent). *Mitigated in-scope by F7's predictive spin-up term* (one level, velocity-projected). **Seam:** `RealmDemand` already carries a `parent` chain; an up-tree "edge-approach" verb is additive on the same arm; the loop already computes the predictive projection.
3. **Real-astronomy scale + floating-origin renderer + the P6 spatial index behind `children_within` (C1) + `containing_path_for_position` general inversion (Slice-6 P3 stand-in).** **Seams:** `children_within(parent, pos, radius)` is spatial-index-shaped from Slice 2 (linear impl swaps for the D-9/D-45 index with no caller change); `containing_path_for_position` has the interface at Slice 6 (P3 book → P4 seed-descent); `LatticePos`/tiered-i64 (D-41) already planted for floating-origin.
4. **`RealmId` naming Galaxy/Universe as first-class arms** (P4). **Seam:** `RealmCoord`/`RealmKindTag` names them today; lifecycle keys on `RealmPath`, never `RealmId::Galaxy`; the dedicated `RealmId` arms are additive.
5. **`AoiMembership` bits in the parent's Cat-C checkpoint** (only if demand-set determinism across respawn is ever required — F9). **Seam:** `ReHomeState::Snapshot` (Slice 4) is the carrier; proven harmless not to, today.
6. **Multi-orchestrator `LiveFabric` sharding by subtree** (the single-orchestrator lifecycle ceiling, D-32). **Seam:** `LiveFabric` keyed by `RealmPath` shards cleanly by subtree; the demand arm is already routable per-subtree.

---

## PART 5 — HOW WARP AND THE VU ARC REBUILD ON TOP

**Warp = lifecycle + transfer composing, zero warp code.** Fly out of System A's SOI → containment `should_rehome` → the proven saga re-homes you UP to the already-live (ancestor-closure-pinned) Galaxy → the Galaxy's `evaluate_realm_aoi` sees you approach System B's `spin_up_r_m` (predictive term fires ahead of arrival — F7) → `RealmDemand{SpinUp, B}` → orchestrator spawns B with the correct profile (C2) → you cross B's SOI → the *existing* crossing saga hands you DOWN into B → System A, now occupant-less with `child_demand==0` past grace and no live descendant, is torn down (F1/F8-safe). The system shrinking-to-a-dot / growing-from-a-dot is a pure-client consequence of the authored-pose feed. **Spin-up-vs-in-flight-crossing race:** the directory head is the rendezvous — while B is `Provisioning` the head is absent, the crossing counts `crossing_unresolved` and re-drives (source-retained, B2 safe); when B self-grants, the next re-drive starts the saga. F1's saga-guard ensures A stays un-killable while your crossing-out names it.

**Signals ride the same live fabric** (P9, uncornered by C2's fix): ancestor-closure guarantees the LCA and every realm on the up-path is at-least-Dormant *with the `signal_relay` capability* (because the Galaxy is now spawned with `profiles::galaxy()`), so leaf→LCA→leaf routing always finds every intermediate realm running — on reserved `InterShardFlow::Signal`.

**Content-streaming** (P4 terrain → P11): the same per-realm AoI loop generalizes from "spin up child shards" to "stream child content" — a live Area runs `evaluate_realm_aoi` over terrain chunks with the identical `AoiConfig`/`children_within` shape (LOD-free: in-range → real mesh, out → nothing). ONE mechanism at every altitude — and, via H2's `child_placements`, that includes the walk-inside-a-flying-ship case.

**The VU visual arc becomes the acceptance test for RLM itself.** VU-0's static Forest boot becomes a single bootstrap spawn (root + connected leaf, Slice 6). Flying toward another system *demonstrates* lifecycle (B pops in on approach, A winks out on departure — seamless physical fly-by, no toggles/loading/teleport). The client stays a pure agnostic renderer (RealmRegistry/RealmSnap feed; per-`RealmId` high-water prevents the warp freeze bug) — it observes spawned realms appearing in its AoI feed and knows nothing of lifecycle. Slice 7 is that arc, driving the real shipped path (real spawn, real transport, real seed; no override hacks — the "test exactly production" mandate).

---

**Relevant files (all absolute):**
- `crates/core/src/geometry.rs` — `AoiConfig` (beside `ContainmentBand`/`OverlapBand`), `finite_extent` (L1), `should_rehome` `k_dwell`.
- `crates/core/src/realm_path.rs` — `RealmCoord`/`RealmLevel`/`RealmKindTag`, `containing_path_for_position` (Slice 6).
- `crates/core/src/worldgen.rs` — `InterestConfig` on `UniverseConfig`, `to_regions` AoI computation, `children_within` (C1 seam), `realm_neighbourhood_for`.
- `crates/core/src/celestial.rs` — `secs_since_epoch` (D-Finding-3), `mean_anomaly_epoch` (D-Finding-2), SPIKE-6a note (D-Finding-4).
- `crates/wire/src/intershard.rs` — the ONE `RealmDemand` arm + G-SEALED classification.
- `crates/sim/src/stub.rs` — `evaluate_realm_aoi`, `child_placements` (H2), `AoiMembership`, `has_synced` gate on `emit_realm_frames:4076`/`evaluate_realm_boundaries:2670` (D-Finding-1).
- `crates/sim/src/directory.rs` — `revoke:516` (F8 `Refused` branch), `commit_cas:552` on `ctx.subject` (F1 disjointness).
- `crates/node/src/saga_runtime.rs` — `LivenessTracker:283/336` (F2 latch), `subject_has_live_saga:2158` (F1 guard template), `reap_lapsed_leases:2143` (Realm-left hole), `liveness_quiesced_until` (F6).
- `crates/node/src/orchestrator.rs` — Spawn Resolver + `LiveFabric` + `RealmSpawner` port + monotone NodeId allocator (F2).
- `crates/io-prod/src/store.rs` — Store B `path=RealmId`, epoch-tagged header (D-Finding-2).
- `crates/bins/src/bin/shard.rs:44` — the `NodeKind::StubShard` hardcode deleted for `profile_kind()` selection (C2).
