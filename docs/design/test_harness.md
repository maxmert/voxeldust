## Deterministic Test Harness — REVISED FINAL DESIGN (hardened)

This is the first code written in the greenfield workspace. It is normative for how every later subsystem (transfer saga, authority directory, gateway multiplexing, shard sim) is *shaped* so it is testable. The cardinal rule, restated precisely after the adversarial review: **production simulation code obtains time, randomness, persistence, and message I/O only through injected `ShardIo` traits; it never calls `tokio::time`, `quinn`, `redb`, `SystemTime`, or `rand::thread_rng` directly.** Enforcement is a scoped clippy `disallowed_methods` rule (not a bespoke source grep) — see §9.

The defining correction from the review: **invariants about what a player sees are checked against the bytes the player actually received after the FaultFabric mangled the link — never against the node's internal ECS state.** The old repo's `connect_tx` synthetic-enqueue hack (`shard-common/src/harness.rs:48-56`) is the canonical bug where ECS state and wire state diverged; a monitor that taps internal state would green-light it. This design makes that structurally impossible.

---

### 0. Crate layout — START SMALL, grow on real pressure (fixes Attack-2 "12 crates up front")

The original 12-crate layout front-loaded architecture before a single block moved. We **start at 4 crates** and split only when a second impl or compile pain demands it.

```
PHASE-0 (foundation, what we build first):
  sim/        # pure ECS world + systems + saga/authority FSMs + io traits as a MODULE (sim::io)
              #   sim::io defines Transport/Clock/Store/DetRng/Provisioner traits
              #   sim::io::mem defines the test impls (MeshFabric handle, VirtualClock, MemStore...)
  node/       # build_app(cfg, io) for ALL node types behind a `NodeKind` param; bins are 4-line wrappers
  harness/    # Topology, FaultFabric, ScriptedClient, ControlOracle, WireMonitor, ChaosRunner
  tests/      # in-process scenarios (harness + sim::io::mem)

GROW LATER, only when justified by a real second impl or real-binary need:
  io-prod/         # split out of sim::io when quinn/tokio/redb impls exist (a SECOND impl forces the seam)
  tests-process/   # real-binary localhost tier — added when ProdIo exists
```

Rule (enforced by review now, by clippy later): **every node type is a LIB; its `bin` is a 4-line wrapper** that builds the production `ShardIo` and calls `node::build_app`. The old repo's shards were bin-only — untestable (R5). Here the bin depends on the lib, never the reverse.

```rust
// node/src/bin/system-shard.rs  (the ENTIRE bin)
fn main() -> anyhow::Result<()> {
    let cfg = NodeConfig::parse();                  // clap
    let io  = prodio::ProdIo::bind(&cfg)?;          // quinn+tokio+redb (queue-bridged, see §2)
    let mut node = node::build_app(NodeKind::SystemShard, cfg, io)?;
    node.run_forever()                              // prod loop: ProdClock drives ticks
}
```

```rust
// node/src/lib.rs
pub struct ShardNode<IO: ShardIo> { world: bevy_ecs::World, schedule: Schedule, io: IO, tick: TickId }
pub fn build_app<IO: ShardIo>(kind: NodeKind, cfg: NodeConfig, io: IO) -> Result<ShardNode<IO>>;
impl<IO: ShardIo> ShardNode<IO> {
    pub fn step_tick(&mut self) -> TickReport;   // THE deterministic unit of progress (§2)
    pub fn run_forever(self) -> Result<()>;      // prod-only: ProdClock loop calling step_tick
}
```

**Vertical-slice build order (fixes Attack-2 "test before subsystem", "16-cell wall of red"):**
1. Queue-bridge tracer bullet: a stub node, ProdIo reader/writer tasks, `step_tick` over in-memory queues, no gameplay. Proves the threading model (§2) before anything depends on it.
2. ONE transition end-to-end in-process: `BoardShip` happy path + saga FSM + ControlOracle's AUTHORITY-UNIQUE + WireMonitor's POSE-CONTINUITY/NO-VANISH. **Zero faults.**
3. Add fault classes incrementally — each flips a few tests green: `drop`/`dup`/`reorder` (idempotency) → `send-reject` → `delay`/`partition` (timeouts) → `crash`/`resurrect` (durability). Crash-recovery is the LAST capability, started with ONE hand-written cell, then macro-generated once the saga API is stable.

---

### 1. The `ShardIo` abstraction (transport, clock, persistence, rng, provisioner)

`ShardIo` is a bundle of traits, generalizing the one thing the old repo got right: the object-safe `ShardProvisioner` trait (`orchestrator/src/provisioner.rs`) with `Local`/`K8s` swap.

```rust
pub trait ShardIo: Send + 'static {
    type Transport: Transport; type Clock: Clock; type Store: Store; type Rng: DetRng;
    fn transport(&mut self) -> &mut Self::Transport;
    fn clock(&self) -> &Self::Clock;
    fn store(&mut self) -> &mut Self::Store;
    fn rng(&mut self) -> &mut Self::Rng;
}
```

**Transport — enqueue-only, never claims synchronous send-success (fixes Attack-1 "test never returns SendError", Attack-2 "synchronous step_tick incompatible with async QUIC").**

The review correctly demolished the original claim that `send` "returns Err on hard failure so the saga can react." QUIC send completion is async; a non-blocking enqueue *cannot* surface a hard failure synchronously without lying or buffering unboundedly. Corrected contract:

```rust
pub trait Transport {
    /// Enqueue an outbound message. Returns Ok if it was ACCEPTED into the outbound
    /// queue (back-pressure: Err(QueueFull) if the bounded queue is saturated — a real,
    /// testable, synchronous condition). It does NOT and CANNOT report whether the peer
    /// received it. Hard delivery failures surface LATER as an inbound NodeUnreachable event.
    fn send(&mut self, to: NodeId, class: MsgClass, bytes: Bytes) -> Result<(), SendError>;
    /// Drain everything DURABLY DELIVERED to us as of the current tick. Includes both
    /// peer messages AND synthetic control events (NodeUnreachable, ProvisionResult, ...).
    fn drain_inbound(&mut self) -> Vec<Inbound>;
    fn local_id(&self) -> NodeId;
}
pub enum MsgClass { Control, Snapshot, Input, Saga, Membership }
pub enum Inbound {
    Wire { from: NodeId, class: MsgClass, bytes: Bytes },
    NodeUnreachable { to: NodeId, class: MsgClass, undelivered: CorrelationId }, // async failure, arrives next tick(s)
    ProvisionResult { intent: ProvisionIntentId, node: NodeId },
}
pub enum SendError { QueueFull }   // the ONLY synchronous error; everything else is async
```

This means the saga's error-handling branch is reachable in two distinct, separately tested ways: (a) synchronous `Err(QueueFull)` (back-pressure), and (b) asynchronous `Inbound::NodeUnreachable` (peer/link death). The FaultFabric can deterministically trigger BOTH (§3), so the R9 cure is demonstrable, not aspirational.

The production transport reads a **cryptographically verified source identity header** off each QUIC stream (generalizing the old `QueuedShardMsg.source_shard_id`), so `from: NodeId` is never an ephemeral-port heuristic — kills the "UDP-to-session IP-matching" race (R2).

**Clock — logical, closed-form celestial time (fixes Attack-1/Attack-2 "shared clock assumes a global barrier production lacks").**

```rust
pub trait Clock {
    fn now(&self) -> LogicalInstant;
    fn tick(&self) -> TickId;            // THIS node's current committed tick
    fn universe_epoch(&self) -> EpochId; // persisted, identical on every node (§5)
}
```

The old repo computed `celestial_time_from_epoch` from `SystemTime::now()` on each shard independently (`shard-common/src/harness.rs:176-182`); at 110 km/s, ms of skew = km of error (R6). The correction has two layers:

1. **All celestial/frame math is CLOSED-FORM in `(tick, epoch, seed)` with zero per-tick incremental accumulation.** A node that joins at tick T is instantly correct because position is `f(T)`, not `Σ f(0..T)`. No incremental mean-anomaly integrator is permitted in committed state (a clippy lint flags float accumulators tagged `#[celestial]`).
2. **Production tick numbers are NOT globally synchronized, and the design says so explicitly.** Real shards are separate pods with independent tick loops. Therefore **every cross-shard message carries the sender's `source_tick`** (the old `PlayerHandoff.source_tick`, `core/src/handoff.rs:46`, already did this), and a receiver reconstructs the sender's frame at the SENDER's tick, never its own. The harness has a **staggered-stepping mode** (§2) that runs nodes at deliberately different tick offsets to exercise tick-skew in the fast tier — the bug class that previously only appeared in the slow real-binary tier.

**Store — write-ahead, fsync OFF the tick thread (fixes Attack-2 "redb checkpoint() has no prod semantics" + "fsync stalls the 20Hz loop").**

The original `checkpoint()/restore()` "clone the maps" had no faithful redb implementation (redb has no O(1) snapshot) and `Store::put` doing a synchronous redb `begin_write/commit` (as in `orchestrator/src/persistence.rs:14-19`) would fsync inside the tick, blowing the 50ms budget. Both removed.

```rust
pub trait Store {
    /// Append to an in-memory write-ahead buffer. NO fsync, NO disk I/O. Returns immediately.
    fn put(&mut self, table: Table, key: &[u8], val: &[u8]);
    /// Reads see the buffer + last durable state (read-your-writes within a node).
    fn get(&self, table: Table, key: &[u8]) -> Option<Vec<u8>>;
    fn scan_prefix(&self, table: Table, prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)>;
    /// Drain the WAL buffer for the persistence task to commit. Returns the durable-up-to tick.
    fn take_pending(&mut self) -> WalBatch;
    fn durable_through(&self) -> TickId; // last tick whose writes are fsynced
}
```

A **separate persistence task** (prod: a tokio task; test: a harness-driven pseudo-node) drains `WalBatch` and commits to redb with batched, durability-relaxed transactions, then feeds `Inbound`-style "durable through tick N" back. **Crash modeling no longer clones maps**: a crash drops the volatile node and *re-opens the same redb file* (prod) or *retains the same MemStore handle* (test); recovery replays persisted records — exactly what production does. The harness adds a fault that **stalls the persistence task** to assert the tick loop never blocks on it and that a crash loses only un-fsynced (post-`durable_through`) writes.

**DetRng** — every stochastic decision draws from `DetRng` seeded from `(universe_seed, node_id, purpose_tag)`. Never `thread_rng`.

**Provisioner — deterministic, persist-intent-before-spawn (fixes Attack-1 "provisioner determinism gap" + "crash between spawn and persist").** Old `LocalProvisioner` used `AtomicU64::new(1000).fetch_add` (`orchestrator/src/provisioner.rs:52,64`) — spawn-order-dependent, non-durable. Both prod and test now:
1. Orchestrator **persists** `ProvisionIntent { id, node_id = derive(cid, epoch) }` to the saga store FIRST (a deterministic NodeId derived from the correlation id + epoch).
2. THEN provisions. Spawn is **idempotent**: if a node with that derived id already registered, adopt it.
3. On orchestrator resume, it re-reads the intent and re-uses the same NodeId — no orphan-plus-new.

The test `InProcProvisioner` has the SAME persist-then-spawn ordering and a **controllable async gap** so the crash-between-intent-and-spawn window is testable (the prod async provisioner has this window wide open; the original "synchronous" test provisioner hid it).

---

### 2. Virtual time, threading, and ordering determinism

**Threading model (fixes the Attack-2 fatal): determinism is decoupled from the async net stack.**

```
PRODUCTION node process:
  ┌─ tokio reader task ─┐   crossbeam        ┌─ SIM THREAD (no tokio runtime) ─┐   crossbeam   ┌─ tokio writer task ─┐
  │ quinn recv_stream → │ ─inbound queue──►  │ step_tick(): drain inbound,     │ ─outbound──►  │ drains, quinn send  │
  │ verify id header    │                    │ run schedule, emit outbound     │   queue       │ on failure → enqueue│
  └─────────────────────┘                    │ NO .await on wall time          │               │ NodeUnreachable into│
                                             └─────────────────────────────────┘               │ the INBOUND queue   │
                                                                                                └─────────────────────┘
```

`step_tick` is synchronous over **in-memory queue snapshots**. The tick thread never awaits. QUIC's irreducibly-async nature lives entirely in the two bridge tasks. A hard send failure is detected by the writer task and **re-enters as `Inbound::NodeUnreachable` next tick** — which is *why* Transport::send is enqueue-only. This is the tracer-bullet built first (§0). Fire-and-forget broadcasts the old repo did via `tokio::spawn` inside the tick (`hud_delta.rs`, `signal_pipeline.rs`) are re-expressed as outbound enqueues drained by the writer; "a slow client never stalls the tick" is preserved because enqueue is O(1) and bounded (back-pressure → `Err(QueueFull)` → explicit drop-with-reason, never a silent stall).

**Decision: explicit `step_tick`, NOT `tokio::time::pause`.** `pause/advance` controls timers but not task scheduling order, and bevy_ecs multi-threaded executors are nondeterministic. The whole in-process topology runs on **one thread** advanced by a driver:

```rust
pub struct Topology {
    nodes:   BTreeMap<NodeId, AnyNode>,   // deterministic iteration
    fabric:  FaultFabric,
    clock:   VirtualClock,
    oracle:  ControlOracle,               // ground-truth invariants (§5a)
    wire:    WireMonitor,                 // client-received invariants (§5b)
    sched_rng: SeededRng,
    stagger: StaggerPlan,                 // per-node tick offset (default 0 = lockstep)
}
impl Topology {
    pub fn step(&mut self) {
        self.clock.advance_one_tick();
        let t = self.clock.tick();
        // CRASH PHASE A (pre-inject): fire any CrashOp scheduled at inject-time for tick t.
        self.fabric.apply_crashes(CrashWhen::PreInject, t, &mut self.nodes);
        // 1. Release messages whose deliver_tick == t that the receiver will DURABLY accept.
        //    Delivery order is a stable shuffle keyed by (deliver_tick, sender_seq) — seq is a
        //    per-sender monotonic counter, NEVER collection-iteration order (fixes Attack-1 #8).
        let due = self.fabric.release_due(t, &mut self.sched_rng);
        for d in due { if self.nodes.contains_key(&d.to) { self.nodes[&d.to].inject(d); } }
        // CRASH PHASE B (post-inject, pre-step): models a node dying with the COMMIT sitting
        //    in its inbound queue. Because the fabric is AT-LEAST-ONCE (§3), this message is
        //    NOT acked, so it is REDELIVERED after resurrect — not silently lost (fixes Attack-1 #3).
        self.fabric.apply_crashes(CrashWhen::PostInject, t, &mut self.nodes);
        // 2. Step each node ONCE, in NodeId order, but ONLY if its staggered local tick fires.
        for (id, node) in self.nodes.iter_mut() {
            if self.stagger.fires(*id, t) { let report = node.step_tick(); self.oracle.observe(*id, &report); }
        }
        // CRASH PHASE C (post-step, pre-collect): models a node that PROCESSED the COMMIT and
        //    mutated durable Store, but dies before its ACK leaves. The ACK is lost; the source
        //    times out and the at-least-once contract forces redelivery on the source's retry.
        self.fabric.apply_crashes(CrashWhen::PostStep, t, &mut self.nodes);
        // 3. Collect outbound, hand to fabric (applies drop/dup/reorder/delay/send-reject).
        for (id, node) in self.nodes.iter_mut() {
            for out in node.take_outbound() { self.fabric.enqueue(*id, out, t); }
        }
        // 4. Capture each ScriptedClient's ACTUALLY-DELIVERED bytes this tick, then assert.
        self.wire.observe_delivered(&self.fabric, &self.nodes, t);   // see §5b
        self.oracle.assert_tick_invariants(t, &self.nodes, &self.fabric);
        self.wire.assert_tick_invariants(t);
    }
}
```

**Crash timing is now a first-class third dimension (fixes Attack-1 #3 "phantom timeline").** `CrashWhen ∈ {PreInject, PostInject, PostStep}` crosses with the saga phase and the victim. The recovery matrix (§3) iterates all three. Reading (b) — "message removed from inflight but not durably processed = silently lost" — is structurally impossible because **the fabric is at-least-once: a message is removed from the in-flight set only when the receiver emits a durable ack** (§3).

**Staggered stepping (fixes Attack-2 tick-skew in the fast tier).** `StaggerPlan` lets node B fire its local tick one (or more) topology-ticks after node A. Cross-shard frame/transfer invariants must pass under stagger, not just lockstep — catching the R6-reincarnation (tick-skew) cheaply. POSE-CONTINUITY's ε accounts for the stagger offset (it compares sender-tick-stamped poses, not topology-tick-stamped, since every message carries `source_tick`).

Determinism sources and pins:
- **bevy_ecs system order**: `ExecutorKind::SingleThreaded` + explicit `.chain()`/sets → total order. `par_iter` only in read-only meshing math that never touches committed state.
- **Message delivery order**: a single `sched_rng` stable-shuffles co-arriving messages; `seq` is a **per-sender monotonic counter**, never a collection-iteration artifact.
- **Collections feeding observable order**: `HashMap` with default `RandomState` is **banned** in `sim`/`node` via clippy `disallowed_types`; allowed only as `HashMap<_,_,FixedSeedHasher>` or `BTreeMap`. The chaos harness runs each failing seed **twice in separate processes** and asserts byte-identical logical traces — turning "reproducible" from a claim into a gate (fixes Attack-1 #8 and the physics-determinism flake in #11).

One `step()` == one global 20Hz tick (lockstep) or one driver tick (stagger). A 10-minute scenario is 12 000 `step()` calls in milliseconds, zero real sleeping.

---

### 3. Fault injection — the `FaultFabric` (at-least-once, send-reject, crash-timed)

```rust
pub struct LinkPolicy {
    drop_p: f64, dup_p: f64, reorder_p: f64, delay_ticks: Range<u32>,
    partitioned: bool, send_reject_p: f64,   // NEW: makes Transport::send return Err(QueueFull)-class
}
pub struct FaultFabric {
    policies: BTreeMap<(NodeId, NodeId, MsgClass), LinkPolicy>,
    inflight: BinaryHeap<Scheduled>,          // keyed by (deliver_tick, per_sender_seq)
    unacked:  BTreeMap<MsgId, Scheduled>,     // AT-LEAST-ONCE: redeliver until durable ack
    rng:      SeededRng,
    crashes:  BTreeMap<(TickId, CrashWhen), Vec<CrashOp>>,
}
```

**At-least-once contract (fixes Attack-1 #3 fatal mechanics).** "Delivered" means **"the receiver durably acked"**. A message stays in `unacked` and is redelivered (after a retry delay) until its ack is observed. A crash that drops a message sitting in a node's inbound queue (PostInject) therefore forces redelivery, not loss. This is the mechanical guard against R1/R9: a lost ack can never wedge a player because the sender's redelivery + the receiver's idempotency (correlation-id tombstone, below) converge.

Capabilities (each a deterministic draw from the fabric RNG → a seed reproduces the exact fault tape):
- **drop** — survive via redelivery (LIVENESS).
- **duplicate** — receivers idempotent via single-use-terminal correlation ids (below).
- **reorder** — perturbed deliver_tick; control invariants must not rely on order (kills R2 "ordering enforced only by comments").
- **delay** — bounded; tests forced-resolution timeout (P4).
- **send-reject** — `Transport::send` returns `Err(SendError::QueueFull)` deterministically; the saga must retry-with-backoff or move toward ABORT within T_max (the synchronous half of the R9 cure).
- **node-unreachable** — link death surfaces as `Inbound::NodeUnreachable` next tick (the asynchronous half).
- **partition** — 100% drop on a link/direction; **ALWAYS bounded in chaos runs** so it eventually heals (fixes Attack-1 #12, below).
- **crash(node, when, at_tick)** — drop volatile state at the specified `CrashWhen` phase; retain the redb file / MemStore handle (no map-clone).
- **resurrect(node, at_tick)** — `build_app` fresh node, re-open the same store, replay persisted records, adopt the topology's current tick, re-register, resume persisted saga.

**Correlation-id tombstones: single-use-terminal (fixes Attack-1 #5 "dup PREPARE after ABORT recreates a zombie ghost").** Idempotency keyed by `TransferCorrelationId` is **single-use-terminal**: when a saga reaches `Committed`/`Aborted`, a tombstone `(cid → terminal_state)` is persisted and **retained for ≥ max_delay + max_retry ticks**. Every message handler consults the tombstone BEFORE treating a message as a new saga step. A delayed duplicate `PREPARE(cid=X)` arriving after `ABORT(cid=X)` is a **no-op** (tombstone says terminal), so no zombie ghost. A dedicated invariant (ORPHAN-GHOST, §5a) asserts no entity has a ghost on a node lacking a corresponding non-terminal saga record.

**Authority leases are fencing-token guarded (fixes Attack-1 #2 fatal split-brain-on-resurrect).** Authority is **durable, version-stamped `(epoch, tick, fencing_token)` state**, not volatile ECS state. When a stale node resurrects from a checkpoint that predates a COMMIT, its restored lease carries an OLD fencing token; the directory **rejects** it (provably-old, fenced out) rather than admitting a competing Owner. Restore therefore yields a node that *knows it is stale* and must re-sync, never a second Owner and never zero owners. The DURABILITY test explicitly **resurrects a node whose checkpoint predates a COMMIT and asserts the harness FAILS first** (proving it can see the bug) before asserting the production fencing prevents it.

**Crash-recovery matrix (saga-phase × victim × crash-when).** The transfer's two-phase handover (P4: prepare → freeze+flush → commit → demote, plus ABORT) crosses {before-prepare, after-prepare, after-freeze, after-commit} × {source, dest, orchestrator, gateway} × {PreInject, PostInject, PostStep}. Every cell asserts the saga reaches COMMITTED or ABORTED, exactly one Owner, no stranded player.

| Saga phase \ victim | source | dest | orchestrator | gateway |
|---|---|---|---|---|
| before PREPARE | resume→retry prepare | no ghost yet; no-op | reload saga; re-drive | conn held; re-route |
| after PREPARE, before FREEZE | retry from persisted step | ghost idempotent; dedup re-prepare via tombstone | reload lease; re-drive | transparent |
| after FREEZE, before COMMIT | frozen; timeout→fenced ABORT or resume→COMMIT | hold ghost to deadline | arbitrate via fencing token | transparent |
| after COMMIT, before DEMOTE | still ghosting; idempotent fenced demote | already Owner; ignore fenced-old source | directory flipped; converge | route swap; client blind |

Built **last**, one hand-written cell first, then macro-generated once the saga API stabilizes — preserving a feedback gradient (Attack-2).

---

### 4. Scripted client model — the WIRE source of truth

```rust
pub trait ClientScript {
    /// Called with the DECODED WorldState the gateway actually DELIVERED this tick
    /// (post-fabric: may be stale, dropped, reordered). Returns input to send, or None.
    fn on_tick(&mut self, ctx: &mut ClientCtx, world: &DeliveredWorldView) -> Option<PlayerInput>;
}
```

A `ScriptedClient` holds exactly ONE logical connection to a gateway `NodeId` (binding #2). It never observes shard transfers; the harness asserts the client's route target NodeId never changes during a transfer (server-side route swap). Built-in scripts compose the seamless-transition matrix (binding #3): `WalkToBoundary`, `BoardShip`/`ExitToEva`, `LaunchToSpace`/`LandOnPlanet`, `WarpToStar` (incl. on-demand dest provisioning).

**`DeliveredWorldView` is decoded from the exact bytes the FaultFabric delivered** to this client's connection — NOT a tap of any node's internal ECS state. This is the central fix for Attack-1 #1 (fatal). Each script asserts on delivered state ("after BoardShip my delivered pose is inside the ship hull AABB"). The same `ScriptedClient`, instantiated N times with seeded scripts, is the load driver (no separate tool); a load run is a `Topology` with N client nodes and a longer `run_until`.

**Quantized physics→control boundaries (fixes Attack-2 "replay broken when physics gates control flow").** Every physics-derived value that gates a control decision (sector-edge crossing, SOI transition, land/launch threshold, re-entry) is **quantized to a fixed integer grid before comparison**. `WalkToBoundary` triggers on `cell_index(pos) != cell_index(prev)`, not on a raw `f64` comparison. Control flow thus depends only on quantized integers; a 1e-12 rapier jitter cannot shift which tick the saga starts. The reproducibility guarantee is explicitly **control-plane only**, achieved by quantization at every physics→control boundary.

---

### 5. Invariant assertion — TWO monitors (fixes Attack-1 #1 fatal, Attack-2 "Inspect mirror divergence")

The original single monitor read internal ECS state via an `Inspect` hook and validated state the network never delivered. Split into two, with a strict rule about which data source each may use.

#### 5a. `ControlOracle` — GROUND-TRUTH invariants (may read authoritative central state)

The control plane is *legitimately* centralized (the authority directory, P5). The oracle may read it and saga records directly, because these ARE the authoritative source of truth (not a mirror of it).

```rust
pub trait Inspect {
    fn directory_view(&self) -> DirectoryView;   // the authority directory IS authoritative
    fn saga_records(&self) -> Vec<SagaRecord>;   // saga state IS authoritative
    fn ghost_set(&self) -> Vec<(EntityId, NodeId)>; // for ORPHAN-GHOST
}
```

| Invariant | Check each committed tick |
|---|---|
| **AUTHORITY-UNIQUE** | Build `BTreeMap<EntityId, SmallVec<(NodeId, FencingToken)>>` from the directory, filtered to `Owner`. Assert **`len == 1` EXACTLY** — catches both split-brain (`len>1`) AND zero-owner (`len==0`, the orphan case the original `<=1` missed, Attack-1 #2). Fencing tokens make a stale resurrected claim non-competing. |
| **LIVENESS (eventual delivery)** | For every saga first seen at `t0`, assert it reaches `Committed`/`Aborted` by `t0 + T_max`. **Only asserted on links that are not permanently partitioned** (Attack-1 #12, below). |
| **ORPHAN-GHOST** | No entity may have a ghost on a node with no corresponding non-terminal saga record. Catches zombie ghosts from dup-after-abort (Attack-1 #5). |
| **NO-UNILATERAL-COMMIT (safety under partition)** | A node may transition to `Owner` for an entity ONLY if a fencing-token go-decision was durably recorded by the orchestrator. Forbids the partitioned-peer split-brain (Attack-1 #12). |
| **DURABILITY** | After crash/resurrect, run to quiescence: assert exactly one Owner per player, no stuck saga, every persisted block-edit delta present on the owning shard (sampled via `scan_prefix` at quiescence ONLY, never per-tick), reconstructed poses within ε. |

#### 5b. `WireMonitor` — CLIENT-OBSERVED invariants (consumes ONLY delivered bytes)

This monitor sees exactly what `WireMonitor::observe_delivered` captured from the fabric for each client: the decoded `DeliveredWorldView`. It NEVER reads a node's ECS state. **`what the monitor checks == what a real client sees` by construction.**

| Invariant | Check from delivered bytes only |
|---|---|
| **NO-VANISH** | Per client C, per entity E ever delivered to C: track last delivered tick. If E is absent for > K ticks **while a transfer involving E or C is in-flight**, fail. K is derived (below), not magic. Because the source is *delivered* bytes, a fabric that delays the dest's first ghost-snapshot by D>K produces a real client-visible vanish the monitor DOES see (the original mirror-tap would not). |
| **POSE-CONTINUITY** | When an entity's delivered pose source flips owner A→B, compare the last delivered owned pose (sender-tick-stamped) to the first from B. Assert `‖Δpos‖ < ε_pos`, `angle(Δrot) < ε_rot`. ε = max_velocity × dt × (1 + stagger_offset) + coordinate-conversion slack (derived, not zero). |
| **INPUT-CONSERVATION** | Each `PlayerInput` carries `input_seq` AND a target `(cid, authority_epoch)` fencing stamp. It must appear in exactly one node's applied-log OR a discarded-log with reason. **Ghost-node logs ARE inspected** (not filtered to Owner) so a ghost mis-applying buffered input is caught (Attack-1 #10). A node applies an input ONLY if it is the current Owner for that authority-epoch; otherwise it logs `discarded: stale-authority`. Buffered inputs handed to dest are re-stamped to dest's epoch exactly once; source NEVER applies buffered input after FREEZE. |

**Meta-test (proves 5b actually catches the divergence class):** a test deliberately diverges a node's ECS state from its delivered snapshots (re-creating the `connect_tx` bug) and asserts the `WireMonitor` FAILS. This is the regression guard that the harness checks wire reality, not internal hope.

**Derived overlap-band / K / demote inequality (fixes Attack-1 #7, kills the banned magic 15).** The old `DEFAULT_SOURCE_DEMOTE_TICKS = 15` (`shard-common/src/handoff_pipeline.rs:21`) is a banned magic number. Replaced by a **lease-based demotion**: the source demotes its ghost ONLY upon a destination-acked "ghost delivered to all observers" signal (P5/P8 lease), never a fixed count. The harness asserts the inequality chain:
`overlap_band_ticks > max_fabric_delay_on_saga_class ≥ (implicit demote latency) AND K < (band − delay)`.
`ChaosRunner` specifically searches for `delay_ticks` ranges exceeding the band and asserts the saga **ABORTs** (fenced) rather than popping. NO-VANISH is asserted on delivered state only, so an over-delay is visible as either an abort or a caught vanish, never silently "within K".

---

### 6. Test taxonomy

1. **Unit (pure FSM + property)**: saga and authority FSMs are pure `fn step(State, Event) -> (State, Vec<Action>)`. `proptest` over random event sequences asserts no illegal state and guaranteed termination. R4 dies here: the transfer is a sum type `enum Transfer { ShipToSurface(..), SurfaceToEva(..), PlanetToSpace(..), SystemWarp(..) }` — each variant carries ONLY its fields; the old 12-`Option` `PlayerHandoff` (`core/src/handoff.rs:50-74`) is unrepresentable.
2. **Integration (in-process multi-node)**: `harness` + `sim::io::mem`. Topologies of {orchestrator, gateway, system, planet, ship} + scripted clients. Seamless-transition and crash-recovery matrices live here. Both monitors armed. Includes **staggered-stepping** runs.
3. **Process-level (real binaries, real QUIC localhost)**: added once `ProdIo` exists. Validates quinn handshakes, redb files, identity headers, and the queue-bridge match in-memory semantics. Fewer, slower, pre-push.
4. **Chaos (seed-reproducible)**: `ChaosRunner(seed)` derives link policies + crash schedule (across all three `CrashWhen` phases) + scripted programs, runs N ticks with both monitors armed. **Partitions are always bounded < T_max − ε** so LIVENESS is a fair eventual-delivery test; permanent-partition SAFETY is a separate test asserting NO-UNILATERAL-COMMIT, not termination (Attack-1 #12). On failure it prints the **exact seed**; replay `CHAOS_SEED=<seed>` runs **twice in separate processes** and asserts byte-identical logical traces (the reproducibility gate). Auto-shrink bisects tick range + disables fault subsets → emits a named regression `#[test]`.

Observability (R10 cure): every message/saga step carries a `TransferCorrelationId` (P8); the harness records a structured per-transfer trace dumped as a timeline on failure — no log archaeology.

---

### 7. Determinism boundary honesty

**Guaranteed: same-binary, cross-process logical determinism of the CONTROL PLANE.** Given a seed, the topology produces an identical sequence of saga transitions, message delivery order, directory contents, input application order, and entity *logical control* state — reproducibly across separate processes (enforced by the twice-in-separate-processes gate + the `FixedSeedHasher`/`BTreeMap` rule). This holds even though physics is float-jittery because **every physics→control boundary is quantized** (§4): control flow branches on integer cell indices, never raw `f64`.

**NOT guaranteed: cross-platform bit-identical floating point.** Rapier3d and f64 orbital math are not bit-reproducible across architectures. Therefore:
- **Physical assertions use tolerances** (ε_pos/ε_rot, "within X m of analytic brachistochrone", energy-drift bounds) — derived, never magic.
- **Control-plane assertions use exact equality** (FSM states, cids, authority sets, fencing tokens, input seqs).
- `physics_determinism` is a **SOFT diagnostic (warn, not fail)**, run as TWO separate processes (fork/re-exec) so per-process hasher reseeding is actually exercised; the spawn/handoff path uses the same `FixedSeedHasher` collections so rapier insertion order is deterministic. Chaos-replay correctness NEVER depends on rapier being bitwise stable (Attack-1 #11, Attack-2). It catches accidental nondeterminism leaks as a warning, not a flaky gate.

This is itself the R6 cure: refusing wall-clock-derived frame reconstruction and deriving frames from closed-form `(tick, epoch, seed)` lets the harness assert frame continuity with a *tight* tolerance, surfacing any reintroduced skew (including tick-skew under stagger) immediately.

---

### 8. Attack resolutions

| # | Attack (severity) | Resolution |
|---|---|---|
| 1 | Monitor bypasses the seam, validates undelivered state (FATAL) | Split into ControlOracle (ground-truth: directory/saga only) and WireMonitor (decodes ONLY fabric-delivered bytes). NO-VANISH/POSE-CONTINUITY/INPUT-CONSERVATION asserted exclusively from delivered bytes. Meta-test deliberately diverges ECS from snapshots and proves WireMonitor fails. (§5, §4) |
| 2 | Resurrect-from-stale-checkpoint split-brain invisible to per-tick check (FATAL) | (a) AUTHORITY-UNIQUE asserts `len==1` exactly (catches zero-owner too). (b) Authority is durable, `(epoch,tick,fencing_token)`-stamped; stale restore is fenced out, non-competing. (c) Explicit test resurrects a pre-COMMIT checkpoint and asserts the harness FAILS first, then prevents. (§3, §5a) |
| 3 | Crash-vs-deliver intra-step ordering is a phantom timeline (MAJOR) | `CrashWhen ∈ {PreInject, PostInject, PostStep}` is a first-class third matrix dimension. Fabric is AT-LEAST-ONCE: "delivered" == "durably acked", so a message in an inbound queue at crash is redelivered, never silently lost. (§2, §3) |
| 4 | Test transport never returns SendError → R9 branch is dead code (MAJOR) | `send_reject_p` makes `Transport::send` return `Err(QueueFull)` deterministically (synchronous failure); link death surfaces as `Inbound::NodeUnreachable` (async failure). Both tested; monitor asserts retry-or-ABORT within T_max. (§1, §3) |
| 5 | Dup+reorder defeats cid idempotency across abort boundary → zombie ghost (MAJOR) | Cids are single-use-terminal: persisted tombstone `(cid→terminal)` retained ≥ max_delay+max_retry; handlers consult it before treating a message as new. ORPHAN-GHOST invariant catches zombies. Explicit dup-PREPARE-after-ABORT-is-no-op test. (§3, §5a) |
| 6 | Single shared clock breaks under provisioning/resurrect; tick-skew = R6 reborn (MAJOR) | Celestial math is closed-form in `(tick,epoch,seed)` — zero per-tick accumulation, so a node joining at T is instantly correct; oracle spawns a node at a random tick and checks its frame == oracle. Resurrected/provisioned nodes adopt topology tick; `|node.tick − topology.tick|` checked. Messages carry `source_tick`; receivers reconstruct sender frames. (§1) |
| 7 | NO-VANISH K-tolerance unfalsifiable; overlap band has no lower bound; magic 15 (MAJOR) | NO-VANISH on delivered bytes (so over-delay is visible). Demotion is lease-based on a dest-acked "delivered to all observers" signal, NOT a fixed count — magic 15 deleted. Inequality `band > delay ≥ demote, K < band−delay` asserted; ChaosRunner searches for band-exceeding delays and asserts fenced ABORT. (§5b) |
| 8 | HashMap iteration feeds wire order → replay not reproducible (MAJOR) | Default-`RandomState` HashMap banned in sim/node via clippy `disallowed_types`; only `BTreeMap` or `FixedSeedHasher`. Fabric `seq` is a per-sender monotonic counter, never iteration order. Chaos replay runs twice in separate processes and gates on byte-identical traces. (§2) |
| 9 | INPUT-CONSERVATION duplicate-during-freeze hole; ghost mis-applies (MAJOR) | Inputs carry `(cid, authority_epoch)` fencing stamp; a node applies only if current Owner for that epoch, else `discarded: stale-authority`. Buffered inputs re-stamped to dest epoch exactly once; source never applies after FREEZE. WireMonitor inspects ghost-node logs too. (§5b) |
| 10 | Provisioner determinism gap; crash between spawn and persist (MAJOR) | NodeId = `derive(cid, epoch)`, persisted as ProvisionIntent BEFORE spawn in both impls; spawn idempotent (adopt if already registered). Test provisioner has same ordering + controllable async gap. Matrix asserts orchestrator-crash-during-provision yields exactly one dest. (§1) |
| 11 | LIVENESS T_max circular; masks stranded sagas under permanent partition (MAJOR) | Chaos partitions always heal < T_max−ε (fair eventual-delivery LIVENESS). Permanent-partition is a separate SAFETY test asserting NO-UNILATERAL-COMMIT (no fenced-less commit/abort), not termination. Dest cannot COMMIT before orchestrator durably records a fencing-token go-decision. (§5a, §6) |
| 12 | physics_determinism flaky / unfalsifiable (MINOR) | Run as two separate processes; spawn/handoff uses FixedSeedHasher collections; treated as a SOFT warn-only diagnostic. Chaos-replay correctness depends on quantization, not rapier bit-stability. (§7) |
| A2-1 | Synchronous step_tick incompatible with async QUIC (FATAL) | Production runs a reader/writer tokio task pair bridging crossbeam queues to a synchronous step_tick; Transport::send is enqueue-only; hard failures surface next tick as NodeUnreachable. Built FIRST as a tracer bullet. (§2) |
| A2-2 | Shared clock assumes a global barrier prod lacks (MAJOR) | Stated explicitly: prod ticks are NOT synchronized; all cross-shard math is `f(source_tick, epoch, seed)` carried in-message. Harness has staggered-stepping mode exercising tick-skew in the fast tier. (§1, §2) |
| A2-3 | 12 crates / 5 traits before any gameplay (MAJOR) | Start at 4 crates (io traits as a module); split io-prod/tests-process only when a second impl exists. Build vertically: tracer bullet → one BoardShip transition + 3 invariants → faults incrementally. (§0) |
| A2-4 | 16-cell matrix before the saga exists = wall of red (MAJOR) | Crash-recovery is the LAST capability; one hand-written cell first, macro-generated once the saga API stabilizes. Each fault class flips a few tests green — preserving a feedback gradient. (§0, §3) |
| A2-5 | redb checkpoint() unfaithful; fsync stalls the tick (MAJOR) | Dropped checkpoint/restore; crashes re-open the same redb file / retain MemStore handle and replay. Store::put writes a WAL buffer (no fsync); a separate persistence task commits off-thread; a fault stalls that task to prove the tick never blocks. (§1) |
| A2-6 | Chaos replay broken when physics gates control flow (MAJOR) | Every physics→control boundary quantized to an integer grid; control flow branches on cell indices, never raw f64. Reproducibility guarantee scoped to control plane. (§4, §7) |
| A2-7 | BTreeMap-everywhere + single-thread + full scans won't scale (MINOR) | BTree only where iteration order is observable in committed state; internal hot maps use FixedSeedHasher HashMap (O(1) + cache locality, still deterministic). Monitor is incremental: nodes emit authority-CHANGE diffs checked against a maintained index (O(changes)); whole-world scans only at quiescence. Thousands-of-players gated as a later concern. (§2, §5) |
| A2-8 | grep-lint fights the ecosystem, incomplete protection (MINOR) | Replaced bespoke scanner with scoped clippy `disallowed_methods`/`disallowed_types` (std::time::*::now, rand::thread_rng, default-RandomState HashMap) on sim/node, io-prod exempt. Dependency-behavior determinism is enforced by VirtualClock injection at the seam, not source scanning. (§9) |
| A2-9 | Inspect mirror doubles API and rots (MINOR) | Spatial/visibility invariants derive from delivered wire bytes (no mirror). Inspect retained ONLY for the authority directory (legitimately central) and saga records — both authoritative, not mirrors. (§5) |

---

### 9. Lint enforcement (replaces the bespoke grep scanner)

`clippy.toml` in `sim` and `node`:
```toml
disallowed-methods = [
  "std::time::SystemTime::now", "std::time::Instant::now", "rand::thread_rng",
  # tokio::time::* reachable only behind the io-prod seam
]
disallowed-types = [ "std::collections::HashMap" ]   # use BTreeMap or HashMap<_,_,FixedSeedHasher>
```
`io-prod` carries crate-level `#![allow(clippy::disallowed_methods)]`. This is standard, maintained, precise-path tooling with zero custom maintenance. Dependency wall-clock reads are neutralized by the fact that dependencies get time/rng ONLY through the injected `Clock`/`DetRng` — not by scanning their source.

---

### 10. Unsolvable-tension trade-offs surfaced

- **CAP under permanent partition (Attack-1 #12).** Liveness and safety cannot both hold under a never-healing partition (CAP). **Recommendation: choose SAFETY.** The harness tests them separately: LIVENESS only on eventually-healing links; under permanent partition the assertion is NO-UNILATERAL-COMMIT (no fenced-less progress), accepting that the saga correctly *waits*. A waiting saga is not "stranded" — it converges on heal via fencing tokens. This is a deliberate, documented choice, not a gap.
- **Cross-platform float determinism (Attack-1 #11/#12).** Genuinely unsolvable for rapier/orbital f64 across architectures. **Recommendation: do not pursue it.** Quantize every physics→control boundary so the *control plane* is fully deterministic and reproducible; leave physical state to tolerances. The cost is that two architectures may diverge in sub-quantum physical detail — acceptable because no control decision and no invariant depends on it.