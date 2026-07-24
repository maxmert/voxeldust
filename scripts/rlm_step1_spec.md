This confirms F-B precisely: `MemHub::kill` (line 273-277) sets `alive = false` but leaves the entry in `nodes`; `register` (line 222) panics on duplicate; `reregister` (line 252) reclaims a same id. There is no `deregister` that removes an entry. F2 monotone means killed ids never re-register, so the entry accretes. F-B's fix (add `MemHub::deregister` that removes) is correct and confirmed necessary.

I have verified every load-bearing claim against source. Now I resolve the one live disagreement (F-A) and produce the final spec.

**F-A adjudication (decisive, from source):** `EffectClass::FireAndForget` doc at `intershard.rs:259-261` states it "May NEVER carry a transfer trigger or **authority-gating discrete state**." A `SpinUp`/`TearDown` verb IS authority-gating discrete state (it drives a spawn/kill and carries `parent_fence` as the authority proof). Reviewer "carries-later" argues it's re-derivable so `FireAndForget` — but re-derivability is the `durability_class` (`ReDriven`) axis, which is orthogonal (`intershard.rs:286`). Level-triggered self-heal → `ReDriven` durability; authority-gating discrete state → `SideEffecting` effect. Both reviewers correctly reject widening `IdempotencyKey`. The real fix for F-A's dedup-collision concern is **not** reclassification but (1) the per-child dedup key being `child.path()` not `child.lowered()` (F3/F-D), and (2) making the wire carry only what's needed. I keep `SideEffecting{FencedKey{parent_fence}}` (correct effect class) and fold F-A's collision fix into the F4/F-B/F3 changes below.

The final spec follows.

---

# STEP 1 FINAL BUILD SPEC — Realm Lifecycle Machinery foundation

Source-verified. Every CRITICAL/HIGH finding folded in. Resolutions of the six adversarial findings, stated up front so the frozen skeleton is decided NOW (not in Step 2/3):

- **F-A (HIGH):** `RealmDemand` stays `SideEffecting{ FencedKey{ parent_fence } }` (authority-gating discrete state cannot be `FireAndForget` per `intershard.rs:259-261`) but the **dedup-collision** F-A raises is killed by F3+F4: the wire carries `child: RealmCoord` (globally-unique path) and the Step-2 consumer keys dedup on `child.path()`, never `lowered()`. Effect-class idempotency (`FencedKey`) gates *authority* (is this parent live), consumer dedup gates *per-child* — two different keys, both correct.
- **F-B (HIGH):** add `MemHub::deregister(id)` (removes the entry; F2-monotone makes re-`register` impossible so removal is safe); `MemSpawner::kill_realm` calls it. No hub-table accretion under Step-3 churn.
- **F-C (HIGH, gate-blocker):** **drop `SpawnError::BadProfile`** — `profile_for` is infallible for every kind `profile_kind_of` produces (verified `capability.rs:250-265`; `ShardProfile::build` only errs on voxel-less-voxel requests no canonical profile makes). `SpawnError = { UnknownNode, AlreadyKilled }` — both reachable, zero uncoverable region.
- **F-D (MEDIUM):** `RealmCoord::lowered()` ships with a loud doc + a `lowered_aliases_across_levels` collision test so no future author keys a directory/saga on it.
- **F-E (MEDIUM):** add `RealmCoord::child(level)` — the extend constructor Step 2's AoI loop needs (parent path + child level → child coord).
- **F-F (LOW):** `MemSpawner.live` value is `(ProfileKind, UniverseTick)` — the `at_tick` witness for Step-3 proofs, zero later struct edit.
- **F4 (MEDIUM, perf):** `RealmDemand` drops the redundant `parent: RealmCoord` field (parent path is a prefix of child path; `parent_fence` is the only thing needed from the parent). Halves the per-tick allocation on the level-triggered hot path; removes a representable-but-meaningless `parent != child.parent()` state.

---

## 1. Exact new/changed types

### 1a. `crates/core/src/realm_path.rs` — serde + freeze prerequisites (CHANGED)

Add `Serialize, Deserialize` to the derive lists of `RealmKindTag` (`:27`), `RealmLevel` (`:41`), `RealmPath` (`:70`). Add `#[repr(u8)]` with explicit discriminants to `RealmKindTag` (F7 nit — makes reorder loud at the definition, matching `ProfileKind`/`MsgClass` convention):

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RealmKindTag {
    Universe = 0, Galaxy = 1, System = 2, Planet = 3, Station = 4, Area = 5,
}
```

Add the drift-proof registry const + accessor (mirrors `ProfileKind::ALL`):

```rust
impl RealmKindTag {
    /// Every kind, declaration order = frozen postcard discriminant order (APPEND-only).
    pub const ALL: [RealmKindTag; 6] = [
        RealmKindTag::Universe, RealmKindTag::Galaxy, RealmKindTag::System,
        RealmKindTag::Planet, RealmKindTag::Station, RealmKindTag::Area,
    ];
}
```

Update the module header (`:5-11`): keep the derivation-layer framing, add a line — *"`RealmKindTag`/`RealmLevel`/`RealmPath` are now serde-derived because [`crate::realm_coord::RealmCoord`] lifts them onto the wire behind the frozen `InterShardFlow::RealmDemand` arm; their postcard discriminants are frozen append-only surfaces (see `realm_coord.rs` + the discriminant-freeze test)."* This stops the "strictly OFF the wire" header from becoming a lie (HR1 self-audit).

### 1b. `crates/core/src/realm_coord.rs` — `RealmCoord` (NEW FILE)

Register in `crates/core/src/lib.rs`: `pub mod realm_coord;` near `pub mod realm_path;`.

File header: *"THIS TYPE IS ON THE WIRE (rides `InterShardFlow::RealmDemand`). Globally unique by construction: two same-seed Systems in different Galaxies get distinct `path`s (different Galaxy level), fixing the cross-galaxy aliasing bare `RealmId` has. The lowered `RealmId` is LOSSY (Universe/Galaxy collapse) — the `path` is the disambiguator, never `lowered()` (see `lowered` doc)."*

```rust
use crate::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use crate::pose::RealmId;
use crate::taxonomy::ProfileKind;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealmCoord {
    level: RealmLevel,   // own kind+seed (the leaf's identity)
    path: RealmPath,     // root -> leaf lineage; INVARIANT path.levels().last() == level
}

impl RealmCoord {
    /// Build from a full lineage path. The leaf level IS the coord's own level.
    /// `None` if the path is empty (the sole failure branch). PRODUCER CONTRACT (not
    /// runtime-validated, to stay branchless): the path SHOULD be Universe-rooted +
    /// contiguous; a truncated path aliases (F3) — feed the generator's full lineage.
    #[must_use]
    pub fn from_path(path: RealmPath) -> Option<RealmCoord> {
        path.levels().last().copied().map(|level| RealmCoord { level, path })
    }

    #[must_use]
    pub fn level(&self) -> RealmLevel { self.level }

    #[must_use]
    pub fn path(&self) -> &RealmPath { &self.path }

    /// The `RealmId` this coord LOWERS to — total, via the leaf's kind. LOSSY: Universe->
    /// System(0), Galaxy->System(1), and every System(seed) across every galaxy collapses
    /// by seed alone. NEVER use as a directory/saga/dedup key at multi-galaxy scale — the
    /// `path` is the collision-free key (F-D). Provided only where a real `RealmId` is owed
    /// (keyed kinds today). One-line delegate to `RealmLevel::to_realm_id`.
    #[must_use]
    pub fn lowered(&self) -> RealmId { self.level.to_realm_id() }

    /// The `ShardProfile` selector for this realm's kind (HR3: capability config, never a
    /// feature match-on-kind). Universe/Galaxy collapse to `Galaxy` is INTENTIONAL-for-now
    /// and mirrors `profile_for` (a Universe root and a Galaxy relay want the same no-voxel/
    /// signal-relay caps today); a future split is a deliberate edit of `profile_kind_of`.
    #[must_use]
    pub fn profile_kind(&self) -> ProfileKind { profile_kind_of(self.level.kind) }

    /// The parent coord (path truncated by one leaf), or `None` at the root (len <= 1).
    #[must_use]
    pub fn parent(&self) -> Option<RealmCoord> {
        let levels = self.path.levels();
        if levels.len() >= 2 {
            let parent_levels = levels[..levels.len() - 1].to_vec();
            RealmCoord::from_path(RealmPath::from_levels(parent_levels))
        } else {
            None
        }
    }

    /// The child coord: this coord's path EXTENDED by one level (F-E). The extend direction
    /// Step 2's AoI loop needs to name a child it hasn't spawned — builds child paths by
    /// `parent.child(level)`, NOT by inverting a `RealmId` (the general inversion is P4-owed).
    /// Total: appends `level` to a clone of `path`; the new leaf IS `level`. Never fails
    /// (a non-empty extended path always has a leaf).
    #[must_use]
    pub fn child(&self, level: RealmLevel) -> RealmCoord {
        let mut levels = self.path.levels().to_vec();
        levels.push(level);
        RealmCoord { level, path: RealmPath::from_levels(levels) }
    }
}

/// The ONE `RealmKindTag -> ProfileKind` bridge (branchless-shim helper: `profile_kind()` is a
/// one-line delegate, ALL branching here — the canonical monomorphic-helper split, mirroring
/// `RealmLevel::to_realm_id`). Total over the 6 tags ONLY (never add a Ship/Asteroid/Stub arm —
/// no `RealmKindTag` source exists, so it would be an uncoverable region). Consistent with
/// `profile_for`'s Universe/Galaxy->galaxy collapse.
#[must_use]
fn profile_kind_of(kind: RealmKindTag) -> ProfileKind {
    match kind {
        RealmKindTag::Universe => ProfileKind::Galaxy,
        RealmKindTag::Galaxy   => ProfileKind::Galaxy,
        RealmKindTag::System   => ProfileKind::System,
        RealmKindTag::Planet   => ProfileKind::Planet,
        RealmKindTag::Station  => ProfileKind::Station,
        RealmKindTag::Area     => ProfileKind::Area,
    }
}
```

Note: `child()` sets the leaf directly rather than via `from_path().unwrap()`, so it has NO failure branch to cover (branchless — the extended path is always non-empty). All six methods + the bridge are monomorphic; no generic-monomorphization region trap (HR5).

### 1c. `crates/wire/src/intershard.rs` — `RealmDemand` arm + `DemandVerb` (CHANGED)

Struct + enum in the struct-definition region (near `FlushSource` ~`:504`), importing `RealmCoord` from `vd_core`:

```rust
/// The lifecycle verb a demand carries. LEVEL-TRIGGERED: re-asserted every tick; a dropped
/// verb self-heals on the next re-assertion (-> `ReDriven`). `Empty` is the R1 revision: the
/// CHILD self-reports upward that it holds no occupants (it is the occupancy authority — a
/// sealed parent cannot see inside it). APPEND-only after `TearDown` (frozen discriminants).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DemandVerb {
    SpinUp = 0,    // an AoI now reaches this child: it must be running
    KeepAlive = 1, // still reached: keep it running (steady-state re-assertion)
    Empty = 2,     // child -> parent: I hold no occupants (occupancy self-report, R1)
    TearDown = 3,  // no AoI reaches this child: it may be reclaimed
}

/// A level-triggered realm-lifecycle demand. Carries the CHILD endpoint as a lineage-anchored
/// `RealmCoord` (globally unique — the dedup key at the consumer is `child.path()`, NOT the
/// lossy `child.lowered()`, F-A/F-D), the PARENT's authority fence (the authority proof; the
/// parent COORD is redundant — `child.parent()` derives it — so it is NOT on the wire, F4),
/// the verb, and the universe tick asserted at. Field order frozen once shipped (positional).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealmDemand {
    pub child: RealmCoord,
    pub parent_fence: Fence,
    pub verb: DemandVerb,
    pub universe_tick: UniverseTick,
}
```

Append the arm at the END of `InterShardFlow` (after `CrossingAbortedAck`, `:250`):

```rust
    /// Realm lifecycle (RLM Step 1): a level-triggered spin-up/keep-alive/empty/teardown demand.
    /// Re-asserted every tick, so a dropped verb self-heals -> `ReDriven`. Authority-gating
    /// discrete state keyed on `parent_fence` (the parent's authority proof), like
    /// `CrossingRequest` -> `SideEffecting{FencedKey}`. APPEND-only.
    RealmDemand(RealmDemand),
```

`Fence` (`fence.rs:26`), `UniverseTick` (`ids.rs:137`), `RealmCoord` (§1b) — all serde. `RealmDemand` cannot be `Copy` (embeds a `Vec`-backed `RealmCoord`); the appended `InterShardFlow` arm is already boxed-by-enum so no clippy `large_enum_variant` beyond the existing arms.

### 1d. `crates/sim/src/io/mod.rs` — `RealmSpawner` + `SpawnError` (NEW)

Object-safe (`&dyn` — copy the `Store` object-safety rationale, `io/mod.rs:414-415`; no per-monomorphization region cost, HR5):

```rust
/// The sealed spawn/kill port (RLM Step 1c). Lives OUTSIDE the sealed shard (HR1): a shard
/// cannot spin up a sibling; only the lifecycle authority (orchestrator, or the harness twin)
/// holds a `RealmSpawner`. Object-safe like `Store` — no per-monomorphization region cost.
pub trait RealmSpawner {
    /// Spin up `coord` as a sealed shard at `at_tick`. Mints a FRESH `NodeId` (F2: monotone,
    /// NEVER reused — a torn-down realm's id is retired forever). The profile is derived from
    /// `coord.profile_kind()` (HR3: capability config, never a match-on-kind in features).
    fn spawn_realm(&self, coord: &RealmCoord, at_tick: UniverseTick) -> Result<NodeId, SpawnError>;

    /// Tear down a running realm shard by its minted id.
    fn kill_realm(&self, node: NodeId) -> Result<(), SpawnError>;
}

/// Closed spawn/kill failure taxonomy (thiserror), mirroring `SendError`. NO `BadProfile`:
/// `profile_for` is infallible for every kind `profile_kind_of` produces (F-C) — profile
/// selection cannot fail today, so `spawn_realm`'s Result exists only for FUTURE placement
/// failures; both current variants are `kill_realm` outcomes.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SpawnError {
    /// `kill_realm` named an id that was never spawned by this spawner.
    #[error("unknown realm node: {0}")]
    UnknownNode(NodeId),
    /// `kill_realm` named an id already torn down (F2: never resurrected under the same id).
    #[error("realm node already killed: {0}")]
    AlreadyKilled(NodeId),
}
```

`spawn_realm` keeps `-> Result<NodeId, SpawnError>` (frozen signature; the Result is the future-placement seam), but with `SpawnError` holding no profile arm it cannot currently return `Err` from the spawn path — verified reachable-region-free.

### 1e. `crates/sim/src/io/mem.rs` — `MemSpawner` twin (NEW) + `MemHub::deregister` (CHANGED)

Add `deregister` to `MemHub` (F-B), beside `kill` (`:273`):

```rust
/// REMOVE a node from the hub entirely (RLM: a torn-down realm's transport slot is reclaimed).
/// Distinct from `kill` (flag-only, for the kill-9 crash model where the id may be reregistered):
/// `deregister` is safe to fully remove ONLY because the caller (MemSpawner, F2-monotone) NEVER
/// re-`register`s a removed id — so there is no resurrection. A no-op on an unknown id (mirrors
/// `kill`). Prevents unbounded hub accretion under Step-3 reconciler spawn/kill churn.
pub fn deregister(&self, id: NodeId) {
    self.lock().nodes.remove(&id);
}
```

`MemSpawner` (follows `MemStore`/`MemHub` idiom: `Arc<Mutex>`, `Clone`, poison-tolerant `lock()` `:189`, all state `BTreeMap`/`BTreeSet`, no default-hasher HashMap):

```rust
#[derive(Clone, Debug)]
pub struct MemSpawner {
    inner: Arc<Mutex<SpawnerInner>>,
}
#[derive(Debug)]
struct SpawnerInner {
    hub: MemHub,                                          // to register/deregister transports
    next_node: u64,                                       // F2 monotone; never decremented/reused
    live: BTreeMap<NodeId, (ProfileKind, UniverseTick)>,  // alive ids: kind + spawn tick (F-F witness)
    killed: BTreeSet<NodeId>,                             // retired ids (F2: never re-minted)
    outbound_capacity: usize,                             // ONE config value (no inline literal)
}

impl MemSpawner {
    /// `first_node` starts the mint range ABOVE the scenario's statically-chosen ids so minted
    /// ids never collide with hand-picked ids (Risk 1); both params are config (no magic numbers).
    #[must_use]
    pub fn new(hub: MemHub, first_node: NodeId, outbound_capacity: usize) -> Self { /* NodeId->u64 */ }

    fn lock(&self) -> MutexGuard<'_, SpawnerInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Read back the last-spawned (id, kind, tick) for the harness driver's build_app+add_node
    /// (Option A return-and-plant, F/§3c). Also the Step-3 registry-inspection seam.
    #[must_use]
    pub fn live(&self) -> BTreeMap<NodeId, (ProfileKind, UniverseTick)> {
        self.lock().live.clone()
    }
}

impl RealmSpawner for MemSpawner {
    fn spawn_realm(&self, coord: &RealmCoord, at_tick: UniverseTick) -> Result<NodeId, SpawnError> {
        let kind = coord.profile_kind();
        let mut g = self.lock();
        let id = NodeId(g.next_node);
        g.next_node += 1;                                 // F2 monotone (never decremented)
        g.hub.register(id, g.outbound_capacity);          // plants transport; monotone => never duplicate
        g.live.insert(id, (kind, at_tick));
        Ok(id)
    }

    fn kill_realm(&self, node: NodeId) -> Result<(), SpawnError> {
        let mut g = self.lock();
        if g.killed.contains(&node) {
            Err(SpawnError::AlreadyKilled(node))
        } else if !g.live.contains_key(&node) {
            Err(SpawnError::UnknownNode(node))
        } else {
            g.live.remove(&node);
            g.killed.insert(node);
            g.hub.deregister(node);                       // F-B: remove, not flag (no accretion)
            Ok(())
        }
    }
}
```

`spawn_realm` has NO branch (`profile_for` never called at the spawn path since profile selection is infallible; the `register` never-duplicates by monotone construction) — fully straight-line, no uncoverable region. `kill_realm`'s three arms are each individually reachable (tests 15/16/17). Both trait methods are thin monomorphic wrappers — no `?`/branching in a generic body (there are no generics here; trivially satisfies the branchless-shim rule).

### 1f. Crate-layering (Option A return-and-plant — CONFIRMED, F4-flag resolved)

`MemSpawner` in `crates/sim/src/io/mem.rs`; `Topology` in `crates/harness` (harness → sim, one-directional). `MemSpawner` plants ONLY the sim-visible `MemHub` transport + records `(id, kind, tick)` in `live`. The **harness test driver** reads `spawner.live()`, and for the newly minted id calls `build_app(NodeConfig{ node_id: id, kind: node_kind_for(kind) }, transport)` + `Topology::add_node` (both `[REUSE]` verbatim — `topology.rs:419`, `app.rs:87`). Keeps `sim` free of `harness`. (Option B planting-closure rejected: adds a boxed closure field for no Step-1 benefit.) Real-process spawner (Step 5, out of scope) implements the identical `RealmSpawner` signature launching a process + registering a `MeshTransport`.

---

## 2. Exact wire classification + conformance additions (`crates/wire`)

### 2a. `effect_class` (`intershard.rs:306-433`) — append one arm (mirrors `CrossingRequest`, `:401`):
```rust
    InterShardFlow::RealmDemand(d) => EffectClass::SideEffecting {
        idempotency: IdempotencyKey::FencedKey { fence: d.parent_fence },
    },
```
Do NOT widen `IdempotencyKey::FencedKey` (it holds only `{fence}`, `:276`; widening trips the `!= GENESIS` assert `:353` and every caller). Per-child dedup is a Step-2 consumer map keyed on `child.path()` (F-A/F-D) — not the wire enum.

### 2b. `durability_class` (`intershard.rs:444-497`) — add to the grouped `ReDriven` `|`-chain (ending `:495`):
```rust
            | InterShardFlow::CrossingAbortedAck(_)
            | InterShardFlow::RealmDemand(_) => FlowDurabilityClass::ReDriven,
```
Add a one-line comment on the arm: *"level-triggered re-assertion every tick ⇒ a dropped demand self-heals ⇒ ReDriven, never producer-less."* Keeps the producer-less golden pin at exactly 2.

### 2c. `crates/wire/tests/intershard_closed.rs` — conformance:
1. **`arm_tripwire`** (`:283`): add `| InterShardFlow::RealmDemand(_)` — the compile-forcing enumeration.
2. **NEW `demand_verb_tripwire`** (mirror `payload_tripwire` `:321`): `#[allow(dead_code)] fn demand_verb_tripwire(v: &DemandVerb) { match v { DemandVerb::SpinUp | DemandVerb::KeepAlive | DemandVerb::Empty | DemandVerb::TearDown => {} } }` — a 5th verb fails compilation until enumerated. Required by the same reasoning as `payload_tripwire` (a nested variant slips the golden pin vacuously otherwise).
3. **`every_arm()`** (`:40-276`): append ONE `InterShardFlow::RealmDemand(RealmDemand{ child: <coord from a fixed Universe-rooted 3-level path>, parent_fence: Fence(non-zero), verb: DemandVerb::SpinUp, universe_tick: UniverseTick(..) })`. Use a small local helper `fn demand_child_coord() -> RealmCoord` building a `[Universe, Galaxy, System]` path via `RealmPath::from_levels` so `every_arm` stays a flat list AND the fixture path is Universe-rooted (F3 — the helper must be lineage-rooted, not a bare 1-level). `parent_fence` MUST be non-`GENESIS` (the roundtrip asserts `!= GENESIS` for `FencedKey`, `:353`).
4. **`every_arm_roundtrips_...`** (`:337`): no edit — the loop visits the appended arm; its `FencedKey` branch (`:352`) asserts the real fence.
5. **`durability_class_pins_...`** (`:366`): no edit — the `_ => ReDriven` expectation (`:385`) classifies `RealmDemand`; pin stays `== 2`.

### 2d. `intershard.rs` `mod tests` (near `:945`) — unit tests:
- `realm_demand_arms_roundtrip`: all 4 `DemandVerb` × 2 coord shapes (one leaf lowering to a real `RealmId` e.g. System; one lowering via stand-in e.g. Galaxy), postcard round-trip, `assert_eq!(back, flow)`. Drives every `DemandVerb` monomorphization + both `lowered` outcomes on the wire.
- `realm_demand_effect_and_durability`: `assert_eq!(flow.effect_class(), EffectClass::SideEffecting{ idempotency: IdempotencyKey::FencedKey{ fence }})` and `assert_eq!(flow.durability_class(), FlowDurabilityClass::ReDriven)` (equality, not `matches!`).

---

## 3. Exact TEST LIST for 100% region+branch on ONLY Step-1 code + filters

Every match total over a closed enum with a golden loop; every branch/error-arm individually reached; `assert_eq!`/`expect_err`, never `assert!(matches!)`; `DemandVerb`×4 + `RealmKindTag`×6 each exercised.

### vd-core — filter: `cargo test -p vd-core realm_coord` + `cargo test -p vd-core realm_kind_tag`
In `realm_coord.rs::tests`:
1. `lowered_covers_every_kind` — loop `RealmKindTag::ALL`, build a 1-level `RealmCoord`, `assert_eq!(coord.lowered(), expected)` for all 6 (Universe→System(0), Galaxy→System(1), keyed otherwise). Drives every `to_realm_id` arm through `lowered`.
2. `profile_kind_covers_every_kind` — loop `ALL`, `assert_eq!(profile_kind_of(k), expected)` for all 6 (Universe→Galaxy, Galaxy→Galaxy, System→System, Planet→Planet, Station→Station, Area→Area). Plus `assert!(profile_for(coord.profile_kind()).is_ok())` per kind (proves Galaxy-collapse consistency, F2/Risk 5).
3. `from_path_leaf_and_empty` — `assert_eq!(RealmCoord::from_path(non_empty).map(|c| c.level()), Some(leaf))` AND `assert_eq!(RealmCoord::from_path(RealmPath::from_levels(vec![])), None)` (both `.last().map` branches).
4. `parent_walks_up_and_root_is_none` — from a 3-level path, `parent()` twice yields the 2-then-1-level coord (`assert_eq!` on `level()`+`path().levels().len()`); the 1-level coord's `parent()` is `None` (both sides of `len()>=2`).
5. `child_extends_path` (F-E) — `let c = coord.child(new_level); assert_eq!(c.level(), new_level); assert_eq!(c.path().levels().len(), n+1); assert_eq!(c.parent(), Some(coord.clone()))` (the extend inverts the truncate — covers `child`'s straight-line body).
6. `level_and_path_accessors` — `assert_eq!(coord.level(), leaf)`, `assert_eq!(coord.path().levels().len(), n)` (the two trivial getters).
7. `realm_coord_postcard_roundtrips` — encode/decode, `assert_eq!(back, coord)` (derived serde path in core).
8. `lowered_aliases_across_levels` (F-D tripwire) — `assert_eq!(universe_coord.lowered(), RealmId::System(0))` AND `assert_eq!(galaxy_coord.lowered(), RealmId::System(1))` AND two distinct-galaxy System coords with the same system seed `assert_eq!(a.lowered(), b.lowered())` while `assert_ne!(a.path(), b.path())` — pins the known collisions so no future author keys a directory/saga on `lowered()`.

In `realm_path.rs::tests` (for the new `RealmKindTag::ALL` + freeze):
9. `realm_kind_tag_all_is_exhaustive` — `for k in RealmKindTag::ALL { match k { <all 6 arms> => {} } }` + `assert_eq!(RealmKindTag::ALL.len(), 6)` (drift tripwire; mirrors `profile_kind_registry_cannot_drift`).
10. `realm_kind_tag_serde_discriminants_frozen` — postcard-encode each variant, `assert_eq!` the first byte to its append-only discriminant 0..5 (mirror `msgclass_wire_discriminant_is_frozen_append_only`).

### vd-wire — filter: `cargo test -p vd-wire realm_demand` (unit) + `cargo test -p vd-wire --test intershard_closed`
In `intershard.rs::tests`:
11. `realm_demand_arms_roundtrip` — 4 `DemandVerb` × 2 coord shapes, postcard round-trip, `assert_eq!(back, flow)`.
12. `realm_demand_effect_and_durability` — the two classifier arms by `assert_eq!` (§2d).
In `tests/intershard_closed.rs` (per-release gate, runs on the appended arm once `every_arm` includes it):
13. `every_arm_roundtrips_postcard_and_classifies_coherently` (`:337`) — now visits `RealmDemand`, asserts `FencedKey` fence `!= GENESIS`. No edit; verify reached.
14. `durability_class_pins_the_producer_less_reliable_set` (`:366`) — verifies `RealmDemand == ReDriven`, pin stays 2. No edit.
   (`arm_tripwire` + `demand_verb_tripwire` are compile-time exhaustiveness guards — no runtime test.)

### vd-sim — filter: `cargo test -p vd-sim mem_spawner` + `cargo test -p vd-sim spawn_error` + `cargo test -p vd-sim deregister`
In `mem.rs::tests`:
15. `spawn_mints_fresh_monotone_ids` — spawn 3 coords, `assert_eq!` ids `first, first+1, first+2` (F2), and each transport registered (assert `hub` membership / round-trip a send).
16. `spawn_derives_profile_from_kind` — spawn a System coord, `assert_eq!(spawner.live()[&id].0, ProfileKind::System)` (HR3: profile from coord).
17. `spawn_records_at_tick` (F-F) — spawn at `tick=T`, `assert_eq!(spawner.live()[&id].1, T)` (the tick witness).
18. `kill_unknown_errors` — `assert_eq!(kill_realm(never_spawned).expect_err("unknown"), SpawnError::UnknownNode(id))`.
19. `kill_then_kill_again_errors` — spawn, `assert_eq!(kill_realm(id), Ok(()))`, then `assert_eq!(kill_realm(id).expect_err("twice"), SpawnError::AlreadyKilled(id))` (success arm + already-killed arm).
20. `killed_id_is_never_reminted` — spawn (id=first), kill, spawn again → `assert_ne!(new_id, first)` and `assert_eq!(new_id, first+1)` (F2 monotone across kills — the core invariant).
21. `kill_deregisters_from_hub_no_accretion` (F-B) — spawn/kill in a loop N times, `assert_eq!(hub.node_count(), baseline)` after each cycle (no accretion); assert the killed id's hub entry is GONE (a send toward it or a `contains` check). Add a tiny `MemHub::node_count()`/test accessor if none exists (or assert via existing pump behavior). Covers `deregister`'s remove path.
22. `deregister_unknown_is_noop` — `hub.deregister(never_registered)` does not panic (the `remove` on an absent key branch; mirrors `killing_an_unknown_node_is_a_no_op`).
23. `mem_spawner_recovers_from_poisoned_lock` — poison the mutex in a panicking closure, a subsequent `spawn`/`kill` still works (the `unwrap_or_else(PoisonError::into_inner)` arm; mirror `hub_recovers_from_a_poisoned_lock` `:591`).
24. `spawn_error_display` — `assert_eq!(SpawnError::UnknownNode(NodeId(5)).to_string(), "unknown realm node: node-5")` AND `assert_eq!(SpawnError::AlreadyKilled(NodeId(5)).to_string(), "realm node already killed: node-5")` (each `#[error]` format region; `NodeId` Display is `node-{n}`, verified `ids.rs`).

### harness plant smoke (`crates/harness` or `crates/tests`, NOT Tier-A) — filter: `cargo test -p vd-harness realm_spawn_plant`
25. `mem_spawner_plants_a_steppable_shard` — under a `VirtualClock`: `MemSpawner::spawn_realm(coord, tick)` → read `spawner.live()` → `build_app(NodeConfig{ id, kind }, hub.transport_for(id))` → `Topology::add_node` → `topo.step()` runs deterministically (no panic; node steps in NodeId order). Proves Option A return-and-plant end-to-end under the virtual clock, no sockets. Lives outside Tier-A (need not hit 100%).

**Coverage note:** tests 1–24 are Tier-A and must reach 100% region+branch. The F-C hole is GONE (no `BadProfile` arm exists). `spawn_realm` is branchless (no uncoverable map_err). Every `kill_realm`/`deregister` branch and every `SpawnError`/`DemandVerb`/`RealmKindTag` region has a dedicated equality/`expect_err` assertion.

---

## 4. Reuse map

| New/changed piece | Mirrors (existing, source-verified) |
|---|---|
| `RealmCoord` struct + `from_path`/`level`/`path` | `RealmPath` accessors (`realm_path.rs:75-88`) |
| `RealmCoord::lowered` | delegates to `RealmLevel::to_realm_id` (`realm_path.rs:56`) verbatim |
| `RealmCoord::parent` | `RealmPath::parent_realm` len-guard (`realm_path.rs:94-101`) |
| `RealmCoord::child` (F-E) | inverse of `parent`; branchless push (net-new direction, trivial) |
| `profile_kind_of` bridge | `RealmLevel::to_realm_id` total closed match; consistency w/ `profile_for` (`capability.rs:250-262`) |
| `RealmKindTag::ALL` + `#[repr(u8)]` + drift/freeze tests | `ProfileKind::ALL` + `profile_kind_registry_cannot_drift`; `MsgClass` discriminant-freeze |
| serde on `RealmKindTag`/`RealmLevel`/`RealmPath` | `CapRequest`/`ProfileError` serde derives (`capability.rs:45,58`) |
| `InterShardFlow::RealmDemand` arm + classification | `CrossingRequest` FencedKey/ReDriven (`intershard.rs:401,491`); `ReSolicitBatch` add-lifecycle (CA-1 S3) |
| `DemandVerb` + `demand_verb_tripwire` | `TransitionPayload`/`GhostFlow` + `payload_tripwire`/`ghost_tripwire` (`intershard_closed.rs:321,330`) |
| `RealmSpawner` trait (object-safe) | `Store` object-safe trait + doc rationale (`io/mod.rs:414-416`) |
| `SpawnError` (thiserror closed) | `SendError`/`ProfileError` closed taxonomy (`io/mod.rs:135`, `capability.rs:58`) |
| `MemSpawner` (Arc<Mutex>, poison-recovery, BTreeMap/Set) | `MemStore`/`MemHub` (`mem.rs:182-189`) |
| `MemHub::deregister` (F-B) | sibling of `kill` (`mem.rs:273`); `nodes.remove` (net-new: no removing primitive today) |
| F2 monotone NodeId allocator | NET-NEW (all ids hand-picked today; F2 mandate) |
| harness plant (return-and-plant, Option A) | `Topology::add_node` + `build_app` + `NodeConfig` (`topology.rs:419`, `app.rs:87`) verbatim |

---

## 5. HR1–6 + perf/robustness/determinism sign-off

- **HR1 sealed shards:** only new inter-shard bytes = the `RealmDemand` arm in the ONE reviewed file `intershard.rs`. `RealmSpawner` is a `sim::io` trait held OUTSIDE the sealed shard by the lifecycle authority. `RealmCoord` lifting `RealmKindTag`/`RealmLevel`/`RealmPath` onto the wire is made explicit (new reviewed file + updated `realm_path.rs` header + discriminant-freeze test) — the off-wire header stays honest. **PASS.**
- **HR2 generic transfer:** untouched; `RealmDemand` carries no entity kind, no Durable/Transient fan-out. **Consistent.**
- **HR3 one tooling:** `RealmCoord::profile_kind() → profile_for()` is the ONLY realm→profile path; the kind→profile match lives once in `profile_kind_of` (a capability selector like `profile_for`), zero match-on-shard-kind in features. **PASS.**
- **HR4 features once:** no feature code; port/wire/addressing kind-agnostic (any coord/verb rides the same arm; `MemSpawner` spawns any coord). **Carries forward.**
- **HR5 100% coverage:** all new fns monomorphic (no generic-monomorphization trap); every match total over a closed enum with a golden loop; every branch reached with `assert_eq!`/`expect_err`; `RealmSpawner` object-safe (`&dyn`). **The F-C uncoverable region is ELIMINATED** (`BadProfile` dropped; `spawn_realm` branchless). `child()` is branchless (no failure arm). **PASS by construction.**
- **HR6:** N/A to Step 1; the `MemSpawner`+`Topology` plant is the agent-operable-E2E substrate Steps 2–3 use. **Neutral.**
- **postcard v1 / Fence discipline:** `RealmDemand` carries `parent_fence` (every authoritative demand carries its Fence); appended arm + `#[repr(u8)]`+freeze-tested `DemandVerb`/`RealmKindTag` preserve/pin all discriminants; positional postcard. Receiver-side stale-Fence reject is a Step-2 consumer concern. **PASS (wire layer).**
- **No I/O outside the seam:** `RealmSpawner` IS a new `sim::io` seam; `MemSpawner` is its in-proc twin (no sockets, virtual clock). **PASS.**
- **No magic numbers:** `MemSpawner`'s `first_node`/`outbound_capacity` are config params. **PASS.**
- **Determinism:** monotone `u64` allocator (no wall clock, no RNG), all state `BTreeMap`/`BTreeSet` (no default-hasher HashMap), `MemHub::pump` delivers in NodeId order (`:280`), `VirtualClock` drives time. **PASS.**
- **Performance/robustness:** classification is straight-line (zero branches added to the hot classifiers); `ReDriven` means a dropped level-triggered demand self-heals next tick (robust to loss, no durable outbox). F4 applied — `RealmDemand` carries ONE coord not two, halving the per-tick allocation/bytes on the level-triggered hot path and removing the meaningless `parent != child.parent()` state; F-B applied — `deregister` prevents hub accretion under churn. **Residual perf note (deferred, not a Step-1 defect):** the surviving `child: RealmCoord` still re-emits a `Vec`-backed path every tick per demanded child; ship it (correctness-first), plant the follow-up note for a bounded/inline `[RealmLevel; MAX_REALM_DEPTH]` or interned-path-id representation if Step-2 throughput profiling shows the encode cost matters. **PASS with the note.**

---

## 6. How Steps 2–3 sit on it UNCHANGED

- **Step 2 (AoI decision + `Empty` self-report):** the AoI evaluator builds a child `RealmCoord` by **`parent_coord.child(child_level)`** (F-E — the extend direction; NOT by inverting a `RealmId`, which doesn't exist until P4) and emits `InterShardFlow::RealmDemand{ child, parent_fence, verb: SpinUp/KeepAlive, universe_tick }`. The child shard (occupancy authority) emits `RealmDemand{ verb: Empty }` upward — `Empty` already in the frozen enum, no wire change. The consumer dedups on **`child.path()`** (globally unique, F-A/F-D) — the `FencedKey{parent_fence}` idempotency gates *parent authority*, the path-key gates *per-child*; no enum widening. The orchestrator calls `RealmSpawner::spawn_realm(&coord, at_tick)` — the exact frozen signature. **Nothing in wire/port/addressing changes.**
- **Step 3 (the reconciler):** the level-triggered `ReDriven` classification IS the reconciler's invariant — it re-asserts the desired verb every tick and a dropped verb self-heals (no producer-less outbox owed). It compares desired-set (from AoI demands, keyed `child.path()`) vs live-set (from `MemSpawner::live()` — the `(kind, at_tick)` witness lets it assert *when* a realm was (re)spawned, F-F) and calls `spawn_realm`/`kill_realm`. F2-monotone guarantees a re-spawned realm gets a fresh id (never resurrects a killed id); `MemHub::deregister` (F-B) keeps the twin's transport table bounded under 10³-cycle release-storm chaos. `MemSpawner` under `VirtualClock`+`Topology` is the deterministic substrate its chaos/robustness tests run against. **No rework:** the reconciler is new logic on the frozen port + arm.

Generic-first ordering holds: Step 1 builds the addressing (names ANY level, extends+truncates in both directions), the ONE wire channel (any verb, any kind), and the sealed port (spawn/kill any coord) COMPLETE — Steps 2/3 add decision + reconciliation without touching the frozen skeleton.

---

## 7. Ordered implementation checklist (top-to-bottom)

1. **vd-core `realm_path.rs`:** add `Serialize, Deserialize` to `RealmKindTag`/`RealmLevel`/`RealmPath`; add `#[repr(u8)]`+explicit discriminants to `RealmKindTag`; add `RealmKindTag::ALL`; update the module header (wire-lift note). Add tests 9, 10.
2. **vd-core `realm_coord.rs` (new file)** + register `pub mod realm_coord;` in `lib.rs`: `RealmCoord` struct + `from_path`/`level`/`path`/`lowered`/`profile_kind`/`parent`/`child` + `profile_kind_of` helper, with the F-D `lowered` doc + F-E `child`. Add tests 1–8.
3. **Run** `cargo test -p vd-core realm_coord` + `cargo test -p vd-core realm_kind_tag` — confirm green + (later) 100% on the new core surface. *(Note to the running agent: a coverage gate is live in the shared target dir — the person implementing runs this; the planning task does not.)*
4. **vd-wire `intershard.rs`:** add `DemandVerb` (`#[repr(u8)]`) + `RealmDemand` struct (single `child` coord, F4); append the `InterShardFlow::RealmDemand` arm; add the `effect_class` arm (§2a) and the `durability_class` `|`-chain arm (§2b). Add unit tests 11, 12.
5. **vd-wire `tests/intershard_closed.rs`:** add `RealmDemand` to `arm_tripwire`; add `demand_verb_tripwire`; append the Universe-rooted fixture to `every_arm()` (with the `demand_child_coord` helper, non-`GENESIS` fence). Verify tests 13, 14 pass unchanged.
6. **Run** `cargo test -p vd-wire realm_demand` + `cargo test -p vd-wire --test intershard_closed`.
7. **vd-sim `io/mod.rs`:** add `RealmSpawner` trait + `SpawnError` (UnknownNode, AlreadyKilled — NO BadProfile, F-C).
8. **vd-sim `io/mem.rs`:** add `MemHub::deregister` (F-B); add `MemSpawner` + `SpawnerInner` (`live: BTreeMap<NodeId,(ProfileKind,UniverseTick)>`, F-F) + `impl RealmSpawner`. Add tests 15–24.
9. **Run** `cargo test -p vd-sim mem_spawner` + `cargo test -p vd-sim spawn_error` + `cargo test -p vd-sim deregister`.
10. **vd-harness (or vd-tests):** add the plant smoke test 25 (`mem_spawner_plants_a_steppable_shard`) using return-and-plant (Option A) under `VirtualClock`. **Run** `cargo test -p vd-harness realm_spawn_plant`.
11. **Full Step-1 coverage pass** (the implementer, not this task): `just coverage-fast` after `cargo llvm-cov clean --workspace` (stale-profdata gotcha) — confirm 100% region+branch on the new core/wire/sim surface; then `just gate`.

**Files touched:** `crates/core/src/lib.rs`, `crates/core/src/realm_path.rs`, `crates/core/src/realm_coord.rs` (new), `crates/wire/src/intershard.rs`, `crates/wire/tests/intershard_closed.rs`, `crates/sim/src/io/mod.rs`, `crates/sim/src/io/mem.rs`, and one harness/tests file for the plant smoke.

**Frozen-now decisions (do not revisit in Step 2/3):** `RealmDemand` = single `child` coord + `parent_fence` (F4); `SideEffecting{FencedKey{parent_fence}}` + `ReDriven` (F-A); consumer dedup keys on `child.path()` (F-A/F-D); `SpawnError = {UnknownNode, AlreadyKilled}` (F-C); `kill_realm` removes via `MemHub::deregister` (F-B); `RealmCoord::child` is the child-naming producer (F-E); `MemSpawner.live` carries `(kind, at_tick)` (F-F).
