## Charter B — Sealed Shards, Signal Couplings, Feature-Anywhere (REVISED FINAL, hardened)

This document is self-contained. It sits on the binding architecture (do not re-litigate): ONE `Fence(u64)`; orchestrator-hosted Ownership Directory keyed `Session(id) | Entity(EntityId) | Realm(RealmId) | Ship(ShipId)` with CAS commit point; durable Transfer Saga (pure FSM `vd-saga`, redb WAL, group-commit, ONE u128 `TransferId == correlation_id`, `(TransferId, step_id)` idempotency in `applied_steps` written in the same txn as the effect, fabric is AT-LEAST-ONCE); ghosts are continuous kinematic colliders fed by `GhostDelta` datagrams (never independent integration), reaching the client only as normal entities in the neighbor shard's snapshot; per-RealmId single-writer redb; deterministic harness FIRST (`ShardIo` traits, `VirtualClock`, `FaultFabric`, `ControlOracle`, `WireMonitor`); `node::build_app(NodeKind, cfg, io)` over `ShardIo::Transport::send(to, class, bytes)`; postcard v1 everywhere; no global sim tick (`universe_tick` analytic clock + `source_tick` per message); `StampedPose` source-computed, dest-sanity-bounded.

Charter B adds the **sealing**, **coupling**, **signal**, and **capability** layers and resolves every adversarial finding below. All paths are NEW greenfield; `OLD:` is reference-only (re-verified against branch `ecs-system`).

---

### 0. The two orthogonal closed taxonomies (resolves Finding #4 — InterShardFlow vs Transport seam collision)

**There are TWO closed type families, on TWO orthogonal planes. Neither has an escape hatch. Conflating them was the original error.**

1. **`InterShardFlow` — the SHARD↔SHARD and SHARD↔ORCHESTRATOR control+replication plane.** Every byte that crosses a *shard-to-shard* or *shard-to-orchestrator* boundary is exactly one of these. This is Charter B's domain.
2. **The connection-plane client taxonomy — the SHARD→GATEWAY→CLIENT world-state plane** (owned by `connection_plane.md`): `ControlMsg | BulkMsg | EventMsg | SnapshotDatagram` (+ `InputDatagram` C→G). The largest byte volume crossing a shard boundary — the snapshot/BULK stream to the gateway — lives HERE, not in `InterShardFlow`.

These planes were already declared orthogonal by the integration spec's ghost-transport resolution ("two planes... shard↔shard replication vs shard→gateway→client snapshot"). Charter B's HR1 guarantee is therefore the accurate, two-part statement:

> **HR1 (precise):** Shard-to-shard data flows ONLY through the closed `InterShardFlow` set; shard-to-client world state flows ONLY through the four connection-plane families. Two closed sets, each conformance-tested, neither with a `Raw(Vec<u8>)`/`EcsSync`/`Other` escape hatch.

**GhostFlow's relationship to the client is explicit:** a ghost crosses shard→shard as `InterShardFlow::Ghost`; on the neighbor shard it becomes a *normal entity* in that shard's authoritative snapshot, and reaches the client as an ordinary `SnapshotDatagram` entity — the client NEVER receives a `GhostFlow`. (Matches transfer_protocol §9 "client never knows some entities are cross-shard.")

```
crates/wire/src/intershard.rs   // THE closed shard↔shard taxonomy. One reviewed file.

/// Every byte crossing a SHARD-TO-SHARD or SHARD-TO-ORCHESTRATOR boundary is exactly
/// one of these. No Other, no Raw, no EcsSync. Client-facing world state is NOT here.
pub enum InterShardFlow {
    Ghost(GhostFlow),               // (1) replication into a neighbor's overlap band
    Transfer(TransferEnvelope),     // (2) authority handoff — Class-A saga AND Class-B transient (§5)
    Directory(DirectoryOp),         // (3) orchestrator-authoritative control ONLY (no spatial — §6)
    BlockEdit(BlockEditForward),    // (4) a block mutation forwarded to the RealmId owner
    Coupling(ContinuousCoupling),   // (5) EFFECT-FREE continuous sim input ONLY (§2) — no discrete state
    Signal(SignalFrame),            // (6) gameplay signal pub/sub (§3)
}
```

**Note there is no separate discrete-coupling arm.** Discrete couplings that gate a saga or authority change are NOT couplings — they are `Transfer` (Finding #1). This is enforced at the type level in §2.

---

### 1. The sealed-shard principle and its REAL enforcement (resolves Finding #11 — drop the grep)

**Statement.** A shard's `bevy_ecs::World`, its `rapier3d` context, its per-RealmId redb store are PRIVATE. The only way data leaves a shard is by being serialized into an `InterShardFlow` variant (shard↔shard) or a connection-plane family (shard→client) and handed to `ShardIo::Transport::send`. No "sync my ECS," no shared memory, no RPC into another shard's storage.

**Enforcement — three SOUND layers (the brittle grep is deleted, per test_harness §A2-8):**

1. **The type system + the single egress.** `Transport::send(to: NodeId, class: MsgClass, bytes: Bytes)` is the ONLY egress. Code in `sim`/`node` can only *produce* the `bytes` for a shard↔shard send by going through the `InterShardFlow` encoder (the encoder is the sole `pub fn` that yields a shard-bound payload). To express a new cross-shard kind you MUST add a variant to `InterShardFlow` in `crates/wire` — a small, line-by-line-reviewed file. There is no type in which to put an un-enumerated flow.
2. **Crate-graph isolation (the structural core, already sound in the original layer-3).** There is NO crate dependency edge by which one shard's `World` is reachable from another. A shard is a `ShardNode<IO>` whose `World` is private; its only public surface is `step_tick()` + `ShardIo`. Nodes are instances of ONE `node` lib (not separate crates exposing component types for remote reads), so a shard *cannot* `use` another shard's components.
3. **Clippy `disallowed-methods` (consistent with test_harness §9), NOT a grep.** `sim`/`node` forbid any direct `quinn`/socket/`tokio::net` call; the only sanctioned network egress is `Transport::send`. `io-prod` is exempt (it IS the prod transport). This is standard, maintained, precise-path tooling.

**The conformance test asserts the REAL invariant (resolves Finding #1's "conformance test checks the wrong thing"):**

```
crates/wire/tests/intershard_closed.rs
// Exhaustive (compiler-forced) match over InterShardFlow. For each arm:
//   - SIDE-EFFECTING arms (Transfer Class-A, BlockEdit, Directory CAS ops):
//       MUST carry (TransferId/correlation_id, step_id) AND be ack-driven (at-least-once).
//   - FIRE-AND-FORGET arms (Ghost, ContinuousCoupling, Transfer Class-B, Signal):
//       payload MUST be provably effect-free OR idempotently re-derivable — it may NOT
//       contain a transfer trigger / authority-gating discrete state (a marker trait
//       `EffectFree` the payload type must implement; ContinuousCoupling::Output: EffectFree).
```

The old assertion ("each arm has a fence + a source_tick") is replaced because it passed while a discrete `DockState.clamped` rode a lossy datagram. The new test makes that *uncompilable*: a discrete authority-gating payload cannot implement `EffectFree`, so it cannot be a `ContinuousCoupling`, so it must be a `Transfer`.

**Explicitly FORBIDDEN (stated loud, tested):** no shard reads another's ECS; no shared memory; no `EcsSync`/`Raw(Vec<u8>)`/`Any` flow; no shard writes another realm's redb directly (cross-realm mutations go through `BlockEdit(4)` to the single-writer owner); ghosts carry no authoritative state (no field to put it in).

---

### 2. Simulation couplings as EFFECT-FREE typed ports (resolves Findings #1 fatal, #2 major)

A *coupling* is a directed, typed, **provably effect-free** sim-input dependency: the SINK applies the value as a force/sample/input only; it NEVER mutates durable authority, never triggers a saga, never flips a directory record. The canonical case: a ship shard produces aggregate thrust; the hull rigid body is owned by the exterior host (system/planet/station).

```
crates/sim/src/coupling.rs

pub trait CouplingPort {
    type Output:   Encode + Decode + EffectFree;   // SOURCE produces each tick — MUST be EffectFree
    type Feedback: Encode + Decode + EffectFree;    // SINK returns — MUST be EffectFree
    const ID: CouplingPortId;
    const RATE: PortRate;            // Datagram20Hz | ReliableLatestOnChange  (both latest-wins, effect-free)
    const DEGRADED: DegradedMode;    // declared, tested as a permanent invariant
}

/// Marker: a type carrying NO discrete state that gates a saga/authority change and
/// NO durable side effect. Implemented only for force/sample/input value types.
/// CANNOT be derived for a type containing a transfer trigger (enforced by the
/// conformance test's negative case + a sealed sub-trait).
pub unsafe trait EffectFree {}
```

**Why `EffectFree` is the fix (Finding #1):** the original `Coupling` arm carried discrete `DockState.clamped` (a saga trigger) on a lossy `ReliableOnChange` datagram with no `(TransferId, step_id)` and no redeliver-until-acked. Under the evil scheduler, a dropped "clamp-engaged" then a superseding "undock" left hull-host and station disagreeing on whether a transfer fires — the exact `try_send` class the architecture exists to kill. **Resolution: split by effect class at the TYPE level, not by a `const RATE`.** Continuous latest-wins couplings (ShipThrust, WeatherDrag) stay datagram/no-idempotency and are FORBIDDEN (compile-time) from carrying authority-gating state. **DockClamp is NOT a coupling** — docking IS an exterior-authority transfer (Charter B already routed it "through a standard transfer on `Ship(ShipId)`"), so it goes through the `Transfer` arm + saga and inherits `(TransferId, step_id)` idempotency + at-least-once for free. The clamp *transform*, once docked, is an effect-free `CouplingFrame` feedback; the *decision to dock* is a saga.

**The thrust port, concretely:**

```
struct ShipThrustPort;
impl CouplingPort for ShipThrustPort {
    type Output   = ShipOutputs;   // { thrust_vector: DVec3, torque: DVec3, power_state: PowerState,
                                   //   rcs_trim: DVec3, mass_kg: f64, com_offset: DVec3 }  : EffectFree
    // ⚠️ REVISED 2026-08-05 (owner's reversal — see DEFERRED.md realm-unification): the feedback MUST
    // NOT be a POSE. A realm is never told where it is; a hull's pose inside its own frame never changes.
    // What the interior legitimately needs is FELT ACCELERATION + ORIENTATION — which way is "down" for
    // someone walking inside at 3g or landed on a planet — plus contact state. That is an ambient-physics
    // input for the child's own interior, not a coordinate. Re-name and re-scope it before building:
    type Feedback = HullFelt;      // { linear_accel: DVec3, ang_vel, orientation, contact_flags } : EffectFree
    // (the struck original: `HullPose { StampedPose, ang_vel, contact_flags }` — the StampedPose is the
    //  part that violates the reversal; whether the interior needs the rest is still an OPEN question.)
    const ID = CouplingPortId::ShipThrust;
    const RATE = PortRate::Datagram20Hz;    // forces continuous, latest-wins, loss-tolerant
    const DEGRADED = DegradedMode::CoastBallistic;
}
```

**Wire frame** (one shape for ALL couplings — HR3): `CouplingFrame { port: CouplingPortId, fence: Fence, source_tick: u64, payload: P /* P: EffectFree */ }`. No `step_id` because it is provably effect-free; the conformance test verifies the payload implements `EffectFree`.

**Dataflow (one tick):** (1) ship shard runs its `SignalGraph`+`FunctionalBlocks` capability systems, aggregates thruster outputs into `ShipOutputs` (the SAME aggregation as `OLD: core/src/block/aggregation.rs`). (2) emits `InterShardFlow::Coupling(CouplingFrame{port:ShipThrust, fence:Ship(ShipId).fence, source_tick, payload:ShipOutputs})` to the current hull-host (directory `Ship(ShipId)` owner). (3) host integrates the hull rapier body with those forces. (4) host returns `HullFelt` feedback to the ship — the interior "down" reference ONLY (felt acceleration + orientation + contact, **never a position**: 2026-08-05 reversal) — AND emits the hull as a normal entity in its own snapshot stream (so observers + the piloting client see it move — no special path; the client never sees a `CouplingFrame`).

**Fencing + the NORMAL-TRANSFER hold-last-thrust fix (Finding #2 major).** Every `CouplingFrame` carries the `Ship(ShipId)` directory fence the ship shard believes current; the sink rejects `fence < highest_seen`. The flaw the original missed: during a NORMAL hull-host transfer A→B (saga bumps Ship fence E→E+1, ordered Demote-before-Promote), there is a control-RTT window where A is demoted (won't apply thrust), B rejects the ship's still-E-fenced frames, and the ship hasn't yet re-targeted — so thrust is applied by NOBODY for the whole window. At 3g burn crossing a SOI boundary that is a visible, repeatable thrust dropout on EVERY transfer.

> **Fix:** the hull-host handoff state carries the LAST RECEIVED `ShipOutputs`. The saga's `FlushComplete`/dest-pose payload (which already ships authoritative state source→dest) gains a `last_ship_outputs: Option<ShipOutputs>` field. B integrates the hull with the last known thrust **from tick 0 of owning it** (held, not zeroed), and the first fresh E+1-fenced `CouplingFrame` from the re-targeted ship supersedes it. The normal transfer becomes **hold-last-thrust** (physically continuous), not coast-to-zero.

**Degraded modes (each declared; tested permanently — now INCLUDING the normal-transfer row):**

| Failure | Port | Declared behavior | Mechanism |
|---|---|---|---|
| **Authority-transfer-in-progress (NORMAL path)** | ShipThrust | Sink B holds last `ShipOutputs` (from saga handoff state) until first fresh-fence frame; hull acceleration is CONTINUOUS across the transfer | `last_ship_outputs` in FlushComplete; superseded by E+1 frame |
| Hull host crashed (sink dead) | ShipThrust | Hull coasts ballistic (Category-A closed-form, never a re-stepped rapier) on last `HullPose` until saga re-homes the hull; ship buffers latest-wins `ShipOutputs` | `DegradedMode::CoastBallistic` — ship never falls; coasting is physically correct |
| Ship shard crashed (source dead) | ShipThrust | Hull holds last forces N ticks then decays to zero → ballistic; passengers frozen-realm (interior is the ship's private realm) until recovery | `DegradedMode::HoldLastInput` on sink |
| Link partitioned (both alive) | any | each side applies its declared mode independently; on heal latest-wins resyncs within one tick (forces are absolute, not deltas) | FaultFabric partition tests; LIVENESS on heal |
| Stale fence (post-transfer) | any | sink drops the frame; source re-targets from directory | fence rule |

**Generalization (proves HR3/HR4 — same machinery, zero forks):**
- **WeatherDragPort** (planet→hull): `Output = AtmosphereSample{density, wind_vector, pressure}` (EffectFree), sink integrates drag. `DEGRADED = VacuumFallback` (no atmosphere = no drag = safe). Weather is also closed-form from the seed, so this port is an authority-pin/optimization; the client renders clouds from the seed directly.
- **DockClamp is NOT here** — it is a `Transfer` (see above). After docking, the station is the hull host (standard exterior-authority transfer); the clamp transform rides as an EffectFree `CouplingFrame` feedback.

**Client compositing — version-matched, NOT a raw cross-sub parent_ref (resolves Finding #9 major).** The original made the ship-interior render frame's parent a *separately-streamed* hull entity pose on a different sub (`ReferenceFrameDef.parent_ref: SubEntityRef`). That re-introduces transfer_protocol §8.1's "passenger jumps relative to hull" bug CLIENT-side: hull pose arrives on the host sub's lossy datagrams, interior entities on the ship sub's independent lossy datagrams (different `sub_id`, possibly different `source_tick`); a lost hull-T datagram nests interior-T under hull-(T-1). A parent that is "another sub's sampled droppable pose" is NOT a closed-form `f(universe_tick)`, which the connection-plane frame model requires.

> **Fix (a connection-plane amendment with a correctness obligation):** the client composites interior-tick-T through hull-tick-T only — **version-matched buffering** (the client-side analog of transfer_protocol §8.1). The ship sub's `ReferenceFrameDef` references the hull via `parent_ref: SubEntityRef{sub_id, entity_id}` AND a `parent_version`; the client holds the newer stream until both the hull pose and the interior entities are present at a matching `(source_tick, version)` within the interp buffer, then nests. Bound `ε` and `max_hold` explicitly and assert `freeze_gap + InputGap + stagger_skew + parent_match_hold < interp_buffer_headroom` (ties into the integration spec's orphaned "interpolation-buffer ownership"). Add a multi-sub skew+loss test (connection_plane §10 extended with the parent_ref case + injected hull-datagram loss) asserting passenger-relative-to-hull continuity within ε under loss.

---

### 3. Gameplay cross-shard data = the Signal system (HR1 "only through signals")

Couplings (§2) are engine-level effect-free sim inputs. *Signals* are the gameplay pub/sub (antenna/radio/terminal) from `OLD: signal/src/`, riding the fenced reliable control plane as arm (6).

**Model:** a `SignalChannel(ChannelId)` has `SignalScope ∈ {Local, ShortRange{range_m}, LongRange, Radio{frequency}}`, `AccessPolicy ∈ {OwnerOnly, AllowList, Public}`, `SignalValue ∈ {Bool, Float, State}`. `Local` = zero network (stays in one realm's `SignalGraph`). The rest cross realms.

**Wire:** `SignalFrame { channel: ChannelId, fence: Fence, source_tick: u64, scope: SignalScope, value: SignalValue, auth: SignalAuth }`. A `SignalFrame` IS side-effecting at the gameplay level (it can flip a functional block), so where it drives a durable config change it carries `(correlation_id, step_id)` and is idempotent on apply — it is NOT in the effect-free class. Pure transient signals (a momentary radio tone) are latest-wins and idempotent by absoluteness. (The conformance test classifies SignalFrame as side-effecting-when-durable.)

**Routing (directory-mediated only for genuinely central hops; spatial is shard-local — see §6):**
- `ShortRange` → resolved from the **shard-local** band/ghost set (the SAME locally-maintained membership that drives ghosts — NOT a directory query; Finding #5). Sent directly to neighbor shards in proximity.
- `LongRange` → sent to the system-host shard, which fans out to in-system registered interest.
- `Radio{freq}` → sent to the **Galaxy Relay** (a `NodeKind` with the `signal_relay` capability) holding a `RadioSubscribers` registry (`OLD: signal/src/radio_subscribers.rs`): `freq → {subscriber_shard → lease_deadline}` + wildcard, fanned out only to leased subscribers.

A subscriber block (Terminal) consumes V identically to a same-realm signal — the cross-realm hop is invisible to block logic (HR4: the block code is the same on ship and planet).

**Rate limits / auth / grants / leases (all ported, all fenced):** per-client `TokenBucket` (`OLD: signal/src/rate_limit.rs`) at ingress (back-pressure → drop-with-reason, never a silent tick stall); `AccessPolicy` checked at publish AND relay fan-out; held-grant publishes carry `SignalAuth = HMAC(grant_key, frame)` (`OLD: signal/src/grants.rs`), persisted per-realm in the `grants` redb table, revocable; `SignalFrame.fence` is the publishing realm's fence so stale-fence frames (mid-transfer) are dropped; radio subscriptions are leased (crashed subscriber silently drops — no orphaned fan-out).

**Anti-pattern fixed (HR3 at the signal layer):** old `signal_broadcast_remote` "lived outside the plugin because it depends on each shard's QUIC sender" (`OLD: shard-common/src/signal_pipeline.rs`). Here cross-realm egress is `InterShardFlow::Signal` over `ShardIo::Transport` — identical for every NodeKind. No shard-specific sender.

---

### 4. Feature-anywhere — capability DAG, not boolean flags (resolves Finding #6 major)

**Core idea.** A "shard type" is a `NodeKind` value + a **validated capability set**. ALL gameplay features are `sim`-crate systems written against capability *traits*; a feature is registered for every NodeKind whose capability set satisfies the feature's trait bounds. Implement once → runs everywhere identically.

**The capability traits have a dependency lattice** — `FunctionalBlocksCap: VoxelRealmCap + SignalGraphCap`, `BlockEditCap: VoxelRealmCap`, `SurfaceCap`/`SeatCap` are independent. The original `ShardProfile` was a struct of independent `bool`s, which permits the type-INCOHERENT `functional_blocks:true, signal_graph:false` — it compiles, then panics at runtime when the functional-block system queries an uninitialized `SignalChannelTable`. The IDENTICAL-FIXTURE gate misses it because both proven profiles (planet, ship) set all co-dependent flags.

> **Fix: `ShardProfile` is a validated DAG, FAIL-LOUD at config load, never a runtime explosion.**

```
crates/sim/src/capability.rs

pub enum NodeKind { Gateway, Orchestrator, GalaxyRelay, Shard(ShardProfile) }

/// Capabilities form a DAG. The constructor ENFORCES the lattice and DERIVES dependents.
pub struct ShardProfile { caps: CapSet, voxel: Option<VoxelRealm> }   // fields private

impl ShardProfile {
    /// The ONLY constructor. Rejects incoherent sets with a typed error at parse time.
    pub fn build(req: CapRequest) -> Result<ShardProfile, ProfileError> {
        let mut caps = req.caps;
        if caps.functional_blocks { caps.signal_graph = true; caps.voxel_required(); } // derive deps
        if caps.signal_graph      { /* ok standalone */ }
        if caps.block_edit        { caps.voxel_required(); }
        if caps.surfaces || caps.seats { caps.voxel_required(); }
        if caps.voxel_required_flag && req.voxel.is_none() {
            return Err(ProfileError::Incoherent("voxel-dependent cap without a VoxelRealm"));
        }
        Ok(ShardProfile { caps, voxel: req.voxel })
    }
}
```

`build_app` registers by capability, and the DAG guarantees every dependent table exists:

```
// node/src/build.rs — the ONE place features attach
pub fn build_app(kind: NodeKind, cfg, io) -> Result<ShardNode<IO>> {
    let mut app = ShardNode::new(kind, io);
    if let NodeKind::Shard(p) = &kind {           // p is ALREADY coherent (build() validated it)
        if p.voxel().is_some()       { feature::voxel::register(&mut app, p); }
        if p.signal_graph()          { feature::signal::register(&mut app); }
        if p.functional_blocks()     { feature::functional_blocks::register(&mut app); } // signal_graph guaranteed on
        if p.block_edit()            { feature::block_edit::register(&mut app); }
        if p.surfaces()              { feature::surfaces::register(&mut app); }
        if p.seats()                 { feature::seats::register(&mut app); }
        if p.signal_relay()          { feature::relay::register(&mut app); }
    }
    Ok(app)
}
```

**Capability trait surface** (each write-once): `VoxelRealmCap` (`grid() -> &dyn BlockGridView` from `OLD: core/src/block/block_grid.rs`; `frame_space() -> &mut dyn FrameSpace`), `BlockEditCap: VoxelRealmCap` (`OLD: edit_pipeline.rs apply_edits_to_grid()`, pure), `FunctionalBlocksCap: VoxelRealmCap + SignalGraphCap` (`OLD: functional_block_kind.rs signal_schema()`), `SignalGraphCap`, `SurfaceCap`, `SeatCap`.

**Profiles are DATA, built through `build()`:** `galaxy{signal_relay}`, `system{hull_host}`, `planet{voxel:Spherical, functional_blocks(→signal_graph), surfaces, seats, block_edit, hull_host}`, `ship{voxel:Cartesian, functional_blocks, surfaces, seats, block_edit}`, `station{... hull_host}`, `asteroid{voxel:Cartesian, block_edit, no functional_blocks}`.

**The spherical-vs-flat geometry seam — STATEFUL FrameSpace with anchor-generation fencing (resolves Finding #3 major).** The original `FrameSpace` was a pure stateless `f(position)` (`to_local_flat(world)->Vec3`). Verified against `OLD: planet-shard/src/main.rs:145-255`: the real planet projection is a SINGLE mutable `SurfaceAnchor` (one `TangentFrame` for the ENTIRE planet shard's one rapier world), valid only for "a clustered group (one planet shard owns one density region)," using f32 flat coords with a precision budget that REQUIRES `reanchor_surface()` to recenter on the cluster centroid when drift threatens the budget, "preserving every player's world position + velocity." A stateless map cannot express the anchor, the reanchor event, per-cluster validity, or velocity-preservation-on-reanchor.

> **Fix: `FrameSpace` is STATEFUL and owns the anchor lifecycle; staleness is fenced with the SAME generation pattern as authority.**

```
crates/sim/src/feature/voxel/frame_space.rs

pub trait FrameSpace {
    fn anchor_generation(&self) -> AnchorGen;              // monotonic, bumped on reanchor
    fn chunk_addr(&self, world: DVec3) -> ChunkAddress;    // the ONLY geometry-aware step
    fn to_local_flat(&self, world: DVec3) -> (Vec3, AnchorGen);   // anchor-relative; carries the gen
    fn from_local_flat(&self, flat: Vec3, gen: AnchorGen) -> Result<DVec3, StaleAnchor>;
    fn gravity_dir(&self, world: DVec3) -> DVec3;          // "down": constant ship; toward center planet
    /// Re-center onto the cluster centroid; returns the frame-shift transform so callers
    /// re-express every body pose+velocity. Bumps anchor_generation. (planet only)
    fn reanchor(&mut self, centroid: DVec3) -> FrameShift;
}
struct CartesianSpace;                  // ship/station/asteroid: chunk_addr = world/edge; to_local_flat = identity;
                                        //   reanchor is a NO-OP, anchor_generation never advances.
struct SphericalSpace{ radius, anchor: SurfaceAnchor };  // planet: tangent LOG/EXP map + center gravity + reanchor
```

Every block address and pose carries the `AnchorGen` it was computed under (the SAME fence pattern, reused). A flat coordinate computed under gen N is rejected/retransformed under gen N+1 (`from_local_flat` returns `StaleAnchor` → caller re-derives from the world pose, which is gen-invariant). Everything ABOVE `FrameSpace` is shared write-once: `ChunkAddress` (flat `IVec3` chunk + intra-chunk `(u8,u8,u8)` in `[0,61]`, `OLD: ship_grid.rs` 62³), binary-greedy-meshing on the flat grid (`OLD: chunk_mesher.rs`), `gravity_dir` (the one "up" seam), and `apply_edits_to_grid()` (pure). Block code never branches on shard type.

**Binding single-cluster constraint (consistent with identity_persistence's deferred `(RealmId, region-range)` sharding):** ONE planet shard = ONE player cluster = ONE anchor. A second cluster on one planet realm is a **HARD ERROR, not silent f32 corruption** — asserted at the interest/spawn layer. Multi-cluster-per-planet (multiple anchors, or region sharding of `RealmId`) is explicitly deferred.

**Enforcement — the IDENTICAL-FIXTURE gate, now exercising reanchor (Finding #3's teeth):**

```
crates/harness/src/identical_fixture.rs
pub fn assert_feature_anywhere<F: FeatureFixture>(profiles: &[ShardProfile]) {
    assert!(profiles.len() >= 2, "a feature must be proven on >=2 shard kinds");
    let results = profiles.iter().map(|p| run_fixture::<F>(p)).collect::<Vec<_>>();
    assert_all_equal_control_plane(&results);    // delivered-bytes invariants; physics on tolerance
}
```

CI rules (permanent gates): (a) every gameplay feature lands with `assert_feature_anywhere` on a Spherical (planet) AND Cartesian (ship) profile; (b) **at least one fixture FORCES a reanchor on the Spherical profile** (roam the cluster past the f32 budget) and asserts block-edit/pose continuity through it, while the Cartesian profile asserts `reanchor` is a no-op (`anchor_generation` never advances); (c) a NEGATIVE test asserts an incoherent `CapRequest` is rejected by `ShardProfile::build()` and never reaches `register()`. This prevents both the old drift ("Seat+Thruster on a planet did nothing") AND the new false-confidence (single-cluster fixtures hiding reanchor divergence) AND the incoherent-profile runtime explosion.

---

### 5. Generic transfer (HR2) — tiered by durability class, ONE mechanism (resolves Finding #7 fatal)

HR2 demands ANY object (rockets, debris, blocks, constructions, BULLETS) transfer through the same machinery. The established saga (per-EntityId lock, redb WAL, PREPARE/FLUSH/FENCE_DEMOTE/PROMOTE, group-commit fsync, `(TransferId, step_id)` idempotency, ControlOracle AUTHORITY-UNIQUE `len==1`) is correct and necessary for a PLAYER. It is **physically impossible at 20Hz for a firefight** (hundreds of bullets/sec crossing a hull↔system boundary, each needing an orchestrator saga + directory record + fence). The likely solo-dev failure is a special-case "lightweight bullet path" that bypasses the directory — which is BOTH the per-kind fork HR3 forbids AND a class of cross-shard entity with no AUTHORITY-UNIQUE guarantee (the oracle can't assert `len==1` on an entity with no OwnerRecord).

> **Fix: tier the ONE generic mechanism by an `EntityTransferClass` carried on `EntityId.kind`. The class is a parameter of the single mechanism (HR3-compliant), not a fork. Both classes use the SAME `Transfer` wire arm and the SAME `StampedPose`.**

```
// On EntityId(u128){ kind:u8, mint_shard, seq, rand } — kind encodes the class:
enum EntityTransferClass {
    Durable,     // (A) players, ships, persistent constructions
    Transient,   // (B) bullets, debris, particles, short-lived rockets
}
```

**Class A — Durable/authoritative.** Full directory OwnerRecord + saga + WAL + `Fence(u64)`; ControlOracle AUTHORITY-UNIQUE `len==1` ENFORCED. Unchanged from the binding transfer protocol. These can be ghosts requiring AUTHORITY-UNIQUE.

**Class B — Transient/ballistic.** NO directory record, NO saga, NO per-entity fence. Handed off as **authoritative-state-in-the-message**:
- The source computes the dest `StampedPose` + ballistic params (Category-A closed-form, per identity_persistence §7.1) and emits a SINGLE fire-and-forget `Transfer` envelope (it spawns a transient; it does not mutate durable authority).
- The dest spawns and integrates ballistically (closed-form, never a re-stepped rapier for the handoff advance).
- The source despawns on send.
- Exactly-once is replaced by **at-most-once with deterministic ballistic re-derivation**: a lost bullet-handoff just despawns the bullet (ACCEPTABLE — bullets are not durable); a duplicated one is deduped by a **content hash** `(source_mint, source_seq, source_tick, quantized_pose)` so a redelivered duplicate spawns at most one bullet.
- **Class-B entities are NEVER ghosts requiring AUTHORITY-UNIQUE.** They may appear as collision/render mirrors but carry no OwnerRecord; the ControlOracle AUTHORITY-UNIQUE invariant applies ONLY to Class A.

**Harness — a separate, explicitly weaker invariant for Class B.** A `TRANSIENT-CONSERVATION` WireMonitor invariant: a Class-B entity delivered to a client may vanish on a lost handoff (weaker than Class-A NO-VANISH) but may NOT *duplicate* (content-hash dedup asserted) and may NOT exceed a bounded count (datagram/spawn budget). `G-TIER` CI gate: a firefight fixture spawns N bullets/sec across a boundary and asserts (a) no Class-B entity ever acquires a directory OwnerRecord, (b) no Class-B duplicate survives dedup, (c) the saga/orchestrator write rate is unaffected by bullet density.

This makes the cost model HONEST (10000 bullets/sec do not touch the orchestrator) while keeping ONE mechanism with a class parameter (HR3) and reusing `StampedPose` + the `Transfer` arm. The §1 conformance test classifies `Transfer` by class: Class-A is side-effecting (`(TransferId, step_id)`, ack-driven); Class-B is fire-and-forget (content-hash dedup, effect-free spawn).

---

### 6. DirectoryOp — orchestrator-authoritative control ONLY; spatial is shard-local (resolves Finding #5 major)

The original folded `InterestSetDiff{add, remove}` into `DirectoryOp`, putting per-tick spatial proximity fan-out (for ghosts AND ShortRange signals) on the orchestrator control stream at gameplay frequency — the orchestrator is the single-writer SPOF the entire architecture works to keep OFF the 20Hz data path (connection_plane lock-free hot path; transfer_protocol §3.5 "lease renewals never touch redb"). It also CONFLATED three different interest concepts and contradicted connection_plane §4.1 (interest is computed by the AUTHORITATIVE SHARD, not the orchestrator).

> **Fix: separate the three interest concepts; remove `InterestSetDiff` from `DirectoryOp`.**

1. **Client render-subscription diffs** = a connection_plane SHARD→GATEWAY type (computed by the player's authoritative shard, per connection_plane §4.1). NOT an `InterShardFlow` at all.
2. **Ghost-band membership** = computed LOCALLY by each shard from velocity-scaled band geometry (transfer_protocol §1.3) against the ghost poses it already receives — no orchestrator round-trip. `GhostSpawn/Despawn` (already in `GhostFlow`) are the only directory-adjacent events and they are shard↔shard.
3. **ShortRange signal proximity** resolves from the SAME locally-maintained ghost/band set (§3), not a directory query.

`DirectoryOp` therefore contains ONLY genuine orchestrator-authoritative control:

```
enum DirectoryOp {                  // orchestrator <-> shard, reliable+acked, fence-stamped
  LeaseGrant{key, fence}, LeaseRenew, LeaseRevoke,
  CommitCas{key, expected_fence}, AbortCas,
  RevokeSession{session_id},
  EphemerisPublish(EphemerisFrame),     // authority-decision body poses (rate per integration spec)
  ClockSync(UniverseTickSlew),
}
```

This keeps the SPOF off the spatial hot path and matches the established "directory cache is a hint; authority decisions pull-through; spatial is shard-local" model.

**Other shard↔shard arm shapes (restated):**
- `enum GhostFlow { Spawn(GhostSpawn), Delta(GhostDelta), Despawn(GhostDespawn) }` — `GhostDelta` on datagrams; Spawn/Despawn reliable+acked. Carries pose/vel/orient + small replicated blob (health/anim/pilot-flags). NO authoritative state.
- `TransferEnvelope { transfer_id, universe_epoch, schema_version, fence, step_id, entity, class: EntityTransferClass, payload: TransitionClass }` — Class-A and Class-B (§5).
- `struct BlockEditForward { realm: RealmId, fence: Fence, edits: Vec<BlockEdit>, origin: EditSource, corr: (TransferId, StepId) }` — a player whose INPUT is routed to shard A but whose target block lives in realm R owned by shard B is forwarded to B (single-writer redb), idempotent on `corr`. This is the integration spec's orphaned "cross-shard block-edit forwarding" responsibility's owner.

---

### 7. What a "shard type" is after this

**Before:** separate `ship-shard`/`planet-shard`/`system-shard`/`galaxy-shard` bins, copy-pasted-then-diverged signal/block/handoff code (`OLD:` 4 `main.rs` files, `ShardType::` branching).

**After:** ONE `shard` binary + a `ShardProfile` capability config built through `ShardProfile::build()`. `galaxy`/`system`/`planet`/`ship`/`station`/`asteroid` are *coherent configurations*, not codebases:

```
// node/src/bin/shard.rs — the ENTIRE bin
fn main() -> anyhow::Result<()> {
    let cfg     = NodeConfig::parse();
    let io      = prodio::ProdIo::bind(&cfg)?;
    let profile = ShardProfile::build(cfg.cap_request())?;   // FAIL LOUD here if incoherent
    let kind    = NodeKind::Shard(profile);
    node::build_app(kind, cfg, io)?.run_forever()
}
```

**Reconciliation with the integration spec** ("Shard (Ship/Planet/System/Galaxy): One bin, ShardKind param; lib+thin-bin"): Charter B refines `ShardKind` from an *enum tag code branches on* into a `ShardProfile` *validated capability struct features dispatch by*. Code must NOT `match shard_kind { Planet => …, Ship => … }` (a fork); it must dispatch by capability. **`G-NO-SHARD-FORK` lint:** forbid `match` on a shard-kind discriminant in `sim::feature::*`; geometry differences are confined to `FrameSpace` impls selected by `profile.voxel().geometry`. Gateway and Orchestrator remain distinct NodeKinds (not shards). A new shard type ("moon", "derelict") is a new `ShardProfile` value (a new coherent `CapRequest`) — zero new feature code; every existing feature works the moment its (validated) capability set permits it, proven by the IDENTICAL-FIXTURE gate.

---

### 8. Harness + roadmap amendments (incremental freeze — resolves Finding #8 major)

The original front-loaded `InterShardFlow` (all 6 arms), `ShardProfile`, and `CouplingPort` into the P0 frozen-contract set. But Coupling has NO consumer until P8 (ships) and Signal/functional-blocks none until P9; the established roadmap freezes at P0 ONLY what the walking skeleton needs (test_harness §0 "4-crate start, grow on pressure"; transfer_protocol §0.6 Milestone 1 "one transition class, in-memory directory"). Freezing `CouplingPort` before its first consumer is the canonical way to freeze the WRONG shape (P8 discovers it needs the held-thrust field of Finding #2 and the Class-split of Finding #1 — and must break a "frozen" contract).

> **Fix: freeze incrementally; the "closed set" is a CI gate that the set is closed AT EACH RELEASE, not that all arms exist on day one.**

- **`crates/wire/src/intershard.rs` exists from P0** as the one reviewed file, but `InterShardFlow` carries only the arms with a P0/P1 consumer: `Ghost`, `Transfer` (Class-A only at P0), `Directory`. These power the stub-shard "two dots crossing" milestone and ARE in the established P0.
- `BlockEdit` arm + Class-B `Transfer` freeze at **P6** (block edits / first transient entities).
- `Coupling` (with the Finding-#1 EffectFree split AND the Finding-#2 hold-last-thrust field, both informed by the real ships consumer) freezes at **P8**.
- `Signal` arm freezes at **P9**.
- `ShardProfile` freezes incrementally: P0 needs only the stub profile (`voxel: None`); voxel/block caps at P4-P6; functional/signal/surface/seat caps at P9.
- During development the enum is `#[non_exhaustive]` INTERNALLY; the closed-by-construction guarantee is asserted by the conformance test at the milestone each arm lands. This keeps the HR1 "one reviewed file, closed taxonomy" property (the file exists from P0; arms are added under review) while NOT freezing unconsumed shapes.

**Permanent gates (added to the P0→P11 spine):**
- **G-SEALED (from P0):** the `intershard_closed` conformance test (effect-class invariant of §1) + clippy crate-graph/disallowed-methods isolation. NO grep. Permanent.
- **G-IDENTICAL (from P6 block-edit, P9 functional-blocks/signals):** `assert_feature_anywhere` on Spherical + Cartesian, INCLUDING the forced-reanchor fixture (§4). Plus the incoherent-profile NEGATIVE test. Permanent; no feature lands single-kind.
- **G-COUPLING-DEGRADED (from P8 ships):** for every `impl CouplingPort`, a FaultFabric test kills source, kills sink, partitions the link, AND drives a NORMAL hull-host transfer, asserting each declared `DegradedMode` AND **hull acceleration continuity across a hull-host transfer** (the Finding-#2 row), AND a dropped-discrete-frame test confirming discrete state can't ride a coupling (Finding-#1). Permanent.
- **G-TIER (from P6/P11 combat):** the §5 firefight fixture (no Class-B OwnerRecord, content-hash dedup, orchestrator write-rate independent of bullet density). Permanent.
- **G-NO-SHARD-FORK (from P9):** lint forbidding `match` on a shard-kind discriminant in `sim::feature::*`.

**Phase deltas:** P0 adds the `intershard.rs` file (Ghost/Transfer-A/Directory arms) + `capability.rs` (`NodeKind`, stub `ShardProfile`, `ShardProfile::build`) + `coupling.rs`'s `EffectFree` marker (the trait, no ports yet). P4/P5 introduce `FrameSpace` (`CartesianSpace` + `SphericalSpace` with `reanchor`); add the `chunk_addr` + anchor-gen seam tests. P6 lands `BlockEditCap` + first `assert_feature_anywhere` + Class-B `Transfer` + `BlockEditForward`. P8 lands couplings (`ShipThrustPort`, `WeatherDragPort`) + the `last_ship_outputs` saga-field amendment + the version-matched `parent_ref` connection-plane amendment; G-COUPLING-DEGRADED permanent. P9 lands signals + functional-blocks; G-IDENTICAL for both; G-NO-SHARD-FORK on. P10/P11: Radio `signal_relay` on galaxy; combat damage rides EVENTS or `Signal` (NOT a coupling, NOT a new flow).

**No new processes.** Charter B adds zero deployable units — it removes them (4 shard bins → 1). It adds shared types frozen incrementally and one CI gate family. Net-simplifying for the solo dev: one feature impl serves all kinds; "where can data leak?" is answerable by reading one file.

---

### Attack resolutions (each finding, how fixed)

- **[fatal #1] Coupling lacks idempotency but mutates the hull / discrete DockClamp on lossy datagram** → §1+§2: split couplings by EFFECT CLASS at the type level (`EffectFree` marker), not by `const RATE`. Continuous couplings are compile-time forbidden from carrying authority-gating discrete state; DockClamp is DELETED as a coupling and routed through the `Transfer` arm + saga (inheriting `(TransferId, step_id)` + at-least-once). The §1 conformance test asserts the REAL invariant (effect-free fire-and-forget vs `(TransferId, step_id)` ack-driven side effect), making the old "fence + source_tick" check that passed the bug uncompilable. G-COUPLING-DEGRADED gains a dropped-discrete-frame test.
- **[fatal #7] Generic transfer (HR2) for bullets breaks AUTHORITY-UNIQUE / can't scale to 10000/s** → §5: `EntityTransferClass` on `EntityId.kind`. Class-A = full saga+directory+fence (AUTHORITY-UNIQUE enforced); Class-B = fenced, content-hash-deduped, fire-and-forget, deterministically re-derivable, NO directory record, NEVER an AUTHORITY-UNIQUE ghost. ONE mechanism with a class parameter (HR3), same `Transfer` arm + `StampedPose`. New `TRANSIENT-CONSERVATION` harness invariant + G-TIER gate.
- **[major #2] Coupling fencing drops ALL thrust for a control-RTT on every NORMAL transfer** → §2: the saga handoff state carries `last_ship_outputs`; sink B holds last thrust from tick 0 of owning the hull until the first fresh-fence frame supersedes it (hold-last-thrust, physically continuous). New degraded-table row + G-COUPLING-DEGRADED asserts hull acceleration continuity across a transfer, not just across a crash.
- **[major #3] FrameSpace can't be stateless f(position) — real planet is a stateful single anchor with f32 reanchor** → §4: `FrameSpace` is STATEFUL, owns the anchor lifecycle (`reanchor() -> FrameShift`), carries `anchor_generation` on every address/pose (the fence pattern reused; stale gen rejected/retransformed). IDENTICAL-FIXTURE gate FORCES a reanchor on Spherical and asserts continuity; Cartesian asserts no-op. Single-cluster-per-planet is a binding constraint asserted as a HARD ERROR (consistent with identity_persistence's deferred region-sharding).
- **[major #4] InterShardFlow collides with the Transport seam / snapshots aren't in the 6 variants** → §0: TWO orthogonal closed taxonomies — `InterShardFlow` (shard↔shard) and the connection-plane families (shard→client). HR1 reworded to "shard-to-shard data ONLY through these flows AND shard-to-client world state ONLY through the four connection-plane families." Ghost→client relationship made explicit (a ghost reaches the client as a normal snapshot entity, never as `GhostFlow`).
- **[major #5] DirectoryOp folds spatial InterestSetDiff onto the SPOF at gameplay frequency / contradicts shard-computed interest** → §6: `InterestSetDiff` REMOVED from `DirectoryOp`. Three interest concepts separated: client subs (connection-plane shard→gateway), ghost-band membership (shard-local geometry), ShortRange signal proximity (shard-local band set). `DirectoryOp` keeps only orchestrator-authoritative control (lease/CAS/revoke/ephemeris/clock). SPOF stays off the spatial hot path.
- **[major #6] Boolean ShardProfile flags don't encode the trait DAG → incoherent profile compiles, panics at runtime** → §4: `ShardProfile` is a validated DAG built ONLY via `ShardProfile::build(CapRequest)`, which enforces the lattice + derives dependents + FAILS LOUD at config load. NEGATIVE CI test: an incoherent request is rejected and never reaches `register()`.
- **[major #8] Freezing all 3 contract types (Coupling/Signal unconsumed until P8/P9) at P0 freezes the wrong shape** → §8: incremental freeze. P0 freezes only Ghost/Transfer-A/Directory; BlockEdit+Transfer-B at P6; Coupling at P8 (with the #1/#2 corrections from a real consumer); Signal at P9. The closed-set guarantee is a per-release CI gate, not a day-one empty freeze; the one reviewed file exists from P0.
- **[major #9] parent_ref makes the client render frame depend on a lossy cross-sub datagram → ghost-reorder bug client-side** → §2: NOT a raw cross-sub `parent_ref`. Version-matched buffering (client-side analog of transfer_protocol §8.1): composite interior-T through hull-T only, hold the newer until both present within the interp buffer at matching `(source_tick, version)`. Explicit ε + max_hold tied to interp-buffer headroom; multi-sub skew+loss test with injected hull-datagram loss asserts passenger-relative-to-hull continuity.
- **[minor #11] Grep no-escape-hatch test is brittle (the bespoke-grep anti-pattern test_harness already rejected)** → §1: grep DROPPED. HR1 enforced by (1) the type system + the single `Transport::send` egress (callers can only produce shard-bound bytes via the `InterShardFlow` encoder), (2) crate-graph isolation (no edge reaches another shard's `World`), (3) human review of one small file + clippy `disallowed-methods`. Stated plainly as such, not dressed as a CI guarantee.

---

### Explicit trade-offs (genuine tensions, with recommendation)

1. **Class-B transient transfer trades exactly-once for at-most-once.** A lost bullet-handoff despawns the bullet; a Class-B entity can client-side vanish on loss (weaker than Class-A NO-VANISH). **Recommendation: accept it.** Bullets/debris are not durable; the cost of a saga per projectile is 2-3 orders of magnitude over budget and would put projectile density on the orchestrator. Content-hash dedup prevents the dangerous direction (duplication); vanishing a transient is invisible at gameplay scale. The class boundary is the honest cost model.

2. **Single-cluster-per-planet is a hard limit, not a soft degrade.** The stateful single `SurfaceAnchor` means one planet realm supports one player cluster. **Recommendation: keep the hard-error assertion now**, deferring multi-cluster (multiple anchors per realm, or `(RealmId, region-range)` store sharding) until player density on a single planet justifies it — consistent with identity_persistence's already-deferred region sharding. A silent f32-corruption second cluster is far worse than a loud "this planet is full here."

3. **Couplings are NOT a trust path, but a buggy ship shard can ship a bad `ShipOutputs`.** **Recommendation: the hull host sanity-bounds received `ShipOutputs`** (thrust/torque/mass within a generous physical band derived from the ship's design envelope) and clamps rather than trusts — the same source-authoritative-but-dest-sanity-bounded pattern the transfer protocol uses for `StampedPose`. Out-of-band outputs are clamped + metered, never applied raw.