<!--
VERIFICATION STATUS (2026-07-16, grounded on HEAD 7a87154):
Produced by a grounded design pass that read the REAL landed Slice 4b code, after a FIRST draft
was caught by adversarial review as built on a stale code map. Three independent opus verifiers
(state+HR5 / migration+self-heal / HR1+frames+scale) each returned "ready-with-minor-fixes":
all 16 prior CRITICAL/HIGH/MEDIUM holes confirmed FIXED (line refs independently spot-checked),
no correctness or HR5 blockers. MINOR FIXES TO FOLD DURING IMPLEMENTATION (none block approval):
  1. [scale] the MAX_REGIONS=64 membership bitset must scope to own-realm + ancestor chain ONLY
     (~4); child realms (hundreds of ships/stations in a busy system) resolve via the directory,
     NOT a local membership bit — else a crowded system fails-loud-at-boot (hundreds mandate).
  2. [proof] moving-container reverse-cross stability rests on (band dead-zone)+(per-entity Occupied
     serialization)+(fresh-state single-hop self-heal), NOT "k_dwell >= saga-latency" (that cooldown
     does not survive the ownership handoff). Rewrite §5; ledger the cross-host commit-latency bound
     to P6/P7 (D-44) rather than claiming a debug_assert forecloses it at P3.
  3. [migration] the §4/§10 test-migration must also account for the ~10 vd-core geometry.rs tests
     that pin should_commit / candidate_beats_ix / resolve_winner_ix (retarget vs delete).
  4. [cost] add a boot-computed region depth cache to §10 so the D-43 #6 O(N*M^2) retirement is earned.
  5. [tidy] keep Vec (not ArrayVec — no unilateral dep); author P3 regions frame=SystemSpace{seed}
     (matches frame_for_realm) not GalaxySpace; state the root realm is the fold IDENTITY (its band
     bit is irrelevant), don't claim "root band never releases".
-->

# Containment-Based Realm-Membership Re-Home — Approval-Ready Design (REPLACES landed Slice 4b)

**Author:** lead architect · **Status:** engine-change sign-off gate · **Base:** HEAD `7a87154` (Slice 4b LANDED, 100%-covered) · **Supersedes:** the directional half of task #133 Slice 4b

**User mandate (verbatim):** *"No trigger zones. There is no way we will not be in any realm at any point in time. Escape the SOI → immediately in the Star System realm; escape the star system → the galaxy realm."*

This is a **containment** model, not a **portal** model. It **replaces shipped, tested, coverage-100% code** — there is no "escape hatch before the wire freezes." Below, the three GROUND reports are cited as G1 (state map), G2 (Slice-4b decision record), G3 (HR1 feasibility).

---

## 0. Framing correction (the prior draft's three errors, fixed)

| Prior-draft error | Corrected framing (grounded) |
|---|---|
| "Slice 4b has NOT landed; intervene before the wire freezes." | **FALSE.** Slice 4b landed at `8d6140a`, frame-rebinding at `7a87154`. `authority_dest` (stub.rs:2646), the `crossing_outward_no_parent` degrade (stub.rs:2702/2750), and the self-heal E2E (`tests/tests/crossing_nesting_e2e.rs`, 278 lines) are all live at 100% Tier-A coverage. This is a **REPLACEMENT of green code with an explicit test-migration plan** (§4), not a race. |
| "We avoid freezing `CrossingRequest.dir` + `DemoteCmd.new_parent`; a net wire simplification." | **STRAWMAN — deleted.** Those fields were **never added** (`CrossingRequest` = `{subject, from_realm, to_realm, subject_fence, session, attempt}`, intershard.rs:616; `DemoteCmd`, intershard.rs:519, has no `new_parent`). The wire is **unchanged either way**; `to_realm` already carries a value. The genuine (non-wire) simplification is deleting the `authority_dest` asymmetry + the `crossing_outward_no_parent` stat. |
| "Re-key `RequestInFlight` to `(EntityId, dest_realm)` + bump `attempt` — the real work." | **REJECTED — do not revive.** `8d6140a` + its adversary (G2 §1) proved the direction/dest-keyed latch **unnecessary and harmful** (it re-opens the entity-keyed `self_fence_foreign_entity` direction-mismatch, FINDING-1). We **inherit** the landed per-entity serialization (`RequestInFlight: BTreeMap<EntityId, TransferId>`, stub.rs:408; the `Occupied` suppress) + **dest-side self-heal** and **prove/bound** they cover the moving-container case (§5). |

**HR1 answer (G3-grounded, §1):** realm regions are **seed-derived celestial geometry** — closed-form `f(seed)` (and at P4/P5 `f(seed, universe_tick)`) — **replicated to every shard by construction at boot**. No shared mutable state, **no inter-shard bytes**. `guard_boundaries_in_realm` is relaxed to permit the hosted realm **plus its seed-derived ancestor chain** (§6-G). The registry's single source of truth is a **pure vd-core generator** `realm_regions_for(seed_universe)` (§1.3).

---

## 1. The `RealmRegion` model + the seed-derived registry

### 1.1 A region is a *containment* descriptor

`vd_core::geometry` gains a new descriptor. It reuses `RealmBoundary`'s geometry (`center`, `shape`, `band`) and nesting (`parent`), **deletes `to_realm`** (the destination is derived, not authored), and **flips `realm`** from "exterior side" to "the realm this volume defines" (G1 §3 confirms today's nesting fixtures already set `realm == to_realm` = interior — this promotes the de-facto convention to *the* meaning):

```rust
/// One realm REGION: a volume that, when it is the DEEPEST region CONTAINING a point,
/// defines that point's realm. Containment, not a portal — no `to_realm`, no direction.
/// Every field Copy. SEED-DERIVED (see `realm_regions_for`); serde only for the client's
/// `--realm-boxes` single-source (identical bytes, computed not authored).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealmRegion {
    pub realm: RealmId,          // the realm you ARE IN while this is your deepest container
    pub center: LatticePos,      // region center (callers subtract before shape methods)
    pub frame: FrameRef,         // §2.5 — the frame `center`/`shape` are expressed in
    pub shape: Boundary,         // Shell | Aabb | Obb — reused verbatim
    pub band: ContainmentBand,   // §2.4 — DISTINCT band type (signed-metre units)
    pub parent: Option<RealmId>, // enclosing realm; `None` ONLY for the single ambient root
}
```

Note **no `effect` field**: containment is authority membership. Interest/ghost overlap (the `CrossEffect::Interest` fan-out arm) is a **separate concern** carried on a parallel interest-region list unchanged by this design — the containment path is the Authority path. (`fan_out_crossing`'s `(effect × durability)` fork stays; the Interest arm keeps consuming the existing interest boundaries, §7.)

### 1.2 The partition invariant — "always in a realm" by construction

Regions form a **containment forest with a single ambient root**:

1. **Exactly one region has `parent: None`** — the ambient root (the galaxy). Its `shape` contains all reachable space (a `Shell` with radius = galaxy bound). This is the gap-filler: leave every star-system Shell → the deepest *still-containing* region is the galaxy root → you are in the galaxy realm. **No `parent: None` authoring error can exist** (mandate: "no way we will not be in any realm").
2. **Every non-root region's volume ⊆ its parent's volume** — a geometric authoring invariant, **topologically** checked at boot now, **geometrically** checked at P4/P5 (§6-G ledgers the geometric subset check per the review's LOW finding — an exact Shape×Shape subset test is real computational geometry needing cross-frame math that only lands with P4/P5).
3. **`realm` is UNIQUE per region** — required so `boundary_depth`'s `find(|b| b.realm == p)` parent-walk (stub.rs:2615) is deterministic (review MEDIUM). Checked at boot (§6-G).

### 1.3 The seed-derived registry — source of truth + per-shard boot construction (HR1)

**Source of truth:** a new pure generator in `vd_core::worldgen` (planted now, trivial at P3):

```rust
/// The single source of truth for realm→region geometry. Closed-form f(seed) — every
/// shard computes the IDENTICAL forest at boot from the same universe seed, so the geometry
/// is REPLICATED BY CONSTRUCTION (no shared state, no inter-shard bytes — HR1). At P3 the
/// bodies are static (identity ephemeris); at P4/P5 the per-realm celestial PARAMETERS
/// (sma, mass, luminosity, elements) become f(seed) and `center` becomes f(seed, tick).
pub fn realm_regions_for(seed_universe: u64) -> Vec<RealmRegion>;
```

**Per-shard boot construction** (bins): each shard calls `realm_regions_for(seed)` and **filters to the regions it needs** — its own hosted realm, its **seed-derived ancestor chain up to the root** (parent → grandparent → … → galaxy; O(depth) ≤ ~4), and the **child regions it owns handoff-authority for**. This is HR1-clean because **the geometry is not fetched from another shard — it is recomputed locally from the shared seed** (G3: "closed-form `f(seed)` replicated by construction, no shared state"). The `RealmBoundaries` resource → renamed `RealmRegions(Vec<RealmRegion>)`, default-empty (inert through P3 until the playground plants a fixture).

**The realm→node directory stays disjoint** (G3: `DirectoryEntryView` stores only `{key, authority, fence, lease_expires, in_transfer}`, admin.rs:23, **no geometry**). The shard derives **WHICH realm** from local containment; the orchestrator's `head(DirectoryKey::Realm(to_realm))` resolves **WHICH node** from the realm name. No new global structure.

> **P3 vs P4/P5 (G3):** at P3 `realm_regions_for` returns static bodies in one frame (`IdentityFrames`); the seed→celestial-parameter generator and cross-binary determinism gate (SPIKE-6a) are **owed at P4/P5** (§8 ledger). The registry *shape* is unchanged — only `center`/`shape` become tick-derived.

### 1.4 The hierarchy root is `Universe`; the coordinate system is TWO regimes joined at the StarSystem↔Galaxy seam (user, 2026-07-16)

The ambient root is **`Universe`** (not the `System(0)` placeholder). Full chain: **`Universe → Galaxy → StarSystem → Planet/Station → Area`**. `container()`'s fold identity is the Universe region — "you are always at least in the Universe."

**The coordinate system is NOT one uniform grid.** There is a hard architectural seam at **StarSystem ↔ Galaxy** that separates two regimes, and that seam is exactly where player control freedom ends:

| | **StarSystem and DOWN** (Planet/Station/Area, ships, players) | **Galaxy and UP** (Galaxy, Universe) |
|---|---|---|
| Space kind | continuous physics playground | warp-map / star-chart |
| Movement | full 6DOF (Newtonian flight, walking, collisions) | **warp only** — no free control; exit-warp strands you light-years from any SOI |
| Coordinate | ONE metric system per star system; Planet/Ship/Area nest by **frame** (`transfer_frame` sub-frames of system-space) | coarse map; systems are points at ly-scale |
| D-41 tier | **fine** `i64@mm` (a system is ~10¹⁶ mm — overflows f64 exact-int, needs i64 cells) | **coarse** `i64@ly` |

This is precisely the **D-41 tiered-i64 base already planted** (fine i64@mm system-in, coarse i64@ly galaxy-out, exact-integer rebase between them). The gameplay reasoning ("full freedom in-system, warp-only above") is the *why* behind that tiering — the tier boundary IS the control-freedom boundary.

**Containment is regime-AGNOSTIC** — the SAME "deepest region containing me" rule holds everywhere; only the coordinate **tier** (mm vs ly) and the movement **affordance** (6DOF vs warp) differ. Warp is just "what drives your position at the coarse tier"; exit-warp-into-the-void = your coarse position is outside every system SOI → you are contained by the **Galaxy** realm (the void), stranded — this **falls out of the containment rule, no special case.** Two orthogonal mechanisms carry the whole hierarchy: **frames** nest within a tier (Planet/Ship/Area in a system); **tiers** (D-41 rebase) carry across the System↔Galaxy scale/regime jump.

**Impact on this design:** the pure containment machinery (C-1/C-2) is tier- and regime-agnostic — it operates on `signed_distance` in whatever frame/tier it is handed, so it is UNCHANGED; only the registry roots at `Universe`. The coarse-tier ly-math, the fine↔coarse rebase, and warp-as-coarse-position-driver are **P4/P5** (system frames) and **P10** (warp/galaxy) — see §8. OPEN (user forks): (1) galaxy as continuous coarse-metric space [recommended — represents the void] vs a systems+lanes graph; (2) a third `i64@Mly` tier for intergalactic Universe→Galaxy, deferred until multi-galaxy ships.

---

## 2. The per-tick containment detector

### 2.1 `container()` returns `RealmId` BY CONSTRUCTION — no `Option`, HR5-clean (review CRITICAL/LOW)

The mandate "always in a realm" is a **type property, not a runtime proptest**. `container()` folds starting from the **ambient root as the identity element**, so the reduce is *never empty* — there is **no `None` arm to leave uncoverable**:

```rust
/// The entity's realm THIS TICK = the deepest region whose HYSTERETIC membership holds.
/// Returns RealmId (NOT Option): the ambient root is the fold IDENTITY (always a member —
/// its band never releases), so the argmax set is never empty. No None arm ⇒ HR5-clean by
/// TYPE, not by a proptest that can't prove unreachability to llvm-cov.
fn container(
    root_ix: usize,                       // the single parent:None region, precomputed at boot
    members: &[(usize, DepthKey)],        // the (ix, depth-key) pairs that are hysteretic members
    regions: &[RealmRegion],
) -> RealmId {
    let mut best_ix = root_ix;            // identity: root is ALWAYS a member (§1.2)
    for &(ix, key) in members {
        if depth_beats(key, depth_key_of(best_ix)) { best_ix = ix; }
    }
    regions[best_ix].realm
}
```

`root_ix` is validated at boot (`guard_regions_nest` proves exactly one `parent: None`, §6-G), so seeding the fold with it is sound by construction.

### 2.2 A NEW depth-argmax comparator — `resolve_winner_ix` is REPLACED (review HIGH)

`resolve_winner_ix` (geometry.rs:616) is **NOT reused verbatim** — that was a category error in the prior draft. Its input is `(depth, RealmId, Direction, ix)` and its 3rd key is **Direction**, which the containment decision has removed. The replacement is a distinct comparator over a **3-key** tuple (no Direction):

```rust
/// The containment depth-argmax order: depth DESC (innermost wins), then RealmId ASC,
/// then slice ix ASC. NO Direction (containment has no direction). Strict total ⇒
/// permutation-invariant (proptest). A monotone helper — every tiebreak branch covered
/// ONCE here, not smeared through the fold (HR5), mirroring `candidate_beats_ix`'s shape.
#[must_use]
fn depth_beats(a: (u32, RealmId, usize), b: (u32, RealmId, usize)) -> bool {
    if a.0 != b.0 { a.0 > b.0 }
    else if a.1 != b.1 { a.1 < b.1 }
    else { a.2 < b.2 }
}
```

**Consequence for the landed code (stated honestly):** `resolve_winner_ix` / `candidate_beats_ix` (geometry.rs:641) retain their **Direction arm only on the swept anti-tunnel path** (§2.6) if that path still needs it; if the swept guard does not need the Direction tiebreak, `candidate_beats_ix`'s `a.2 != b.2` Direction arm becomes **dead code and is deleted** (an HR5 region removal — the fewer branches, the cleaner). We keep `depth_beats` as the new monomorphic comparator; `resolve_winner_ix` is **replaced** in the containment detector, not called.

### 2.3 The per-(entity, region) HYSTERETIC MEMBERSHIP STATE — a fixed-width bitset (review CRITICAL)

The single-winner `CrossingState` **cannot** hold containment membership. `CrossingState` (stub.rs:365) holds **one** `was_member` / `inward_ticks` / `winner_ix` for the *single* per-tick winner, and `evaluate_one_subject` (stub.rs:2526) **actively resets** `was_member = false` on winner change (G1 §2). `container()` needs the hysteretic membership of **every** region simultaneously (root + ancestors + children). Concrete store:

```rust
/// Per-entity hysteretic membership across ALL regions on this shard. Region count per
/// shard is small and bounded (own + ancestor chain ~4 + owned children — §1.3), so a
/// FIXED-WIDTH bitset per entity is the right shape: O(1) read/write, Copy-cheap, no
/// per-region allocation. `MAX_REGIONS` is a boot-checked const (guard_regions_nest fails
/// loud if a shard's region set exceeds it).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RegionMembership {
    bits: u64,   // bit `ix` = "hysteretic member of regions[ix]"; MAX_REGIONS = 64
}
pub struct ContainmentProgress(pub BTreeMap<EntityId, RegionMembership>);
```

- **`CrossingState` stays `Copy`** (the review flagged that `LatchedCrossing`/`redrive` rely on it, stub.rs:349) — `RegionMembership` is a separate `u64`-wide resource keyed by `EntityId`, so the latch payload is untouched. `RegionMembership: Copy` too.
- **Eviction:** `retain_live<V>` (stub.rs:2485, the only generic shim) already retains `BTreeMap<EntityId, V>` on the live set — it monomorphizes cleanly over `RegionMembership` with **zero new eviction code** (it's already called 3× for other `V`; a 4th monomorphization).
- **DELETE the winner-change reset (stub.rs:2526–2530):** with per-region membership there is no single "winner slot" to reset — the reset becomes **actively wrong** (it would clobber a region's independent latch). It is removed; each region's bit is advanced independently by its own band (§2.4).
- **Memory at the N≥128 density fixture** (`p1_volume_dense_hundreds`, review MEDIUM): `128 entities × 8 bytes = 1 KiB` for the bitset map; at hundreds, still sub-10 KiB. Negligible vs the per-entity `CrossingState`/pose already carried. The eviction pass is one single-threaded `retain` over that map — bounded, not a scale concern.

### 2.4 The velocity-safe containment band — a DISTINCT type (review HIGH/MEDIUM/LOW)

The band that drives containment consumes **`signed_distance`** (negative inside, geometry.rs:275), **not** `membership_scalar` (positive radial-analog). `update_membership` (geometry.rs:143) tests `distance <= create_below` (acquire) / `distance <= destroy_above` (release) — with **smaller = more inside**. For signed distance that means: acquire when `signed_distance <= -inset` (at least `inset` *inside*), release when `signed_distance <= +outset` fails, i.e. hold until `signed_distance > +outset` (more than `outset` *outside*). This is **sign-correct** and gives a dead-zone straddling the surface.

To preserve the landed **unconstructible-mismatch** property (shape↔band pairing can't be wrong; `membership_scalar` must NEVER be fed a metre band, and vice-versa — geometry.rs:270-273 forbids it), the containment band is a **DISTINCT newtype**, not an `OverlapBand`:

```rust
/// A signed-distance hysteresis band (METRES, negative inside) for CONTAINMENT membership.
/// A DISTINCT type from OverlapBand so it can NEVER be fed the radial-analog membership_scalar
/// (and OverlapBand can never be fed signed_distance) — the unconstructible-mismatch invariant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContainmentBand { inset: f64, outset: f64 }   // both > 0, in metres

impl ContainmentBand {
    /// VELOCITY-SAFE constructor — the ONLY way to build one. Widens so the dead-zone
    /// (inset + outset) >= v_rel*dt*K_SAFETY, mirroring for_soi_velocity_safe. debug_assert
    /// pins width_safe_for. Both Shell and box regions use it: box_signed_distance is a true
    /// Euclidean SDF, so a metre band is uniform-thickness at faces AND corners (must-fix #2,
    /// automatic). Fallible: rejects a degenerate/inverted pair LOUD (BandError).
    pub fn for_containment_velocity_safe(
        inset: f64, outset_min: f64, v_rel: f64, dt: f64, k_safety_extra: f64,
    ) -> Result<ContainmentBand, BandError> {
        let need = v_rel.abs() * dt * (K_SAFETY + k_safety_extra);
        let outset = f64::max(outset_min, need - inset);
        let band = ContainmentBand { inset, outset };
        debug_assert!(band.width_safe_for(v_rel, dt));   // (inset+outset) >= v_rel*dt*K_SAFETY
        if inset > 0.0 && outset > 0.0 { Ok(band) } else { Err(BandError::InvalidEdges) }
    }

    /// Sign-correct membership update (the trap the landed code warns about, verified):
    /// acquire when at least `inset` INSIDE (signed_distance <= -inset); once a member,
    /// hold until more than `outset` OUTSIDE (signed_distance > +outset).
    #[must_use]
    pub fn member(&self, was_member: bool, signed_distance: f64) -> bool {
        if was_member { signed_distance <= self.outset } else { signed_distance <= -self.inset }
    }
}
```

This resolves must-fix #1 (units), #2 (box corners, automatic via `box_signed_distance`), the LOW inversion trap (sign verified against `update_membership`'s `<=` convention), and the HIGH velocity-safety gap (the band is sized against `v_rel·dt·K_SAFETY` — the review's core complaint that `for_containment` "abandoned the K_SAFETY derivation" is fixed by making the velocity-safe form the *only* constructor).

### 2.5 The frame seam — PLANTED NOW as identity (review MEDIUM, G3 CRITICAL gap)

G3's most important finding: the **input/trigger side** `transfer_frame` seam is **NOT planted** today (`evaluate_realm_boundaries` reads raw `.offset()`, stub.rs:2431/2454; `RealmBoundary` carries no `FrameRef`, geometry.rs:381). Deferring it forces a P4/P5 retrofit into the hot path **plus** a frozen-serde shape grow. We **plant it now**:

- `RealmRegion` carries `frame: FrameRef` (§1.1).
- The containment test re-expresses the entity into each region's frame **before** the subtract:

```rust
// container()'s per-region test, frame-aware BY SHAPE from day one:
let p_in_region_frame = transfer_frame(&entity_pose, region.frame, ctx.frames);  // ctx.frames = &IdentityFrames at P3
let sd = region.shape.signed_distance(p_in_region_frame.pos.offset() - region.center.offset());
```

At P3 `ctx.frames = &IdentityFrames` → `transfer_frame` is a behaviour-identical no-op (G3: identity reframe moves nothing, frame.rs:248-261). At P4/P5, **swap `IdentityFrames` for the real ephemeris `FrameContext`** — additive, no structural change (mirrors how `rebind_pose_to_dest` already plants the *output* side, frame.rs:183). This is the "don't paint into a corner" discharge.

### 2.6 `containment_candidates` — a FULL per-tick scan; swept is an ADDITIVE tunnel guard (review HIGH, G1)

The landed `crossing_candidates` (stub.rs:2576) is a **swept-relates filter** (emits nothing for `StaysOutside`/`StaysInside` — a statically-inside entity produces no candidate). Containment needs the opposite: **scan ALL regions every tick**, testing point-membership regardless of motion:

```rust
/// FULL per-tick scan over ALL regions on the shard (NOT a swept-gated filter). For each
/// region: frame-transfer the entity in (§2.5), signed_distance, advance the per-region
/// hysteretic bit (§2.4), and if the bit holds, emit (ix, depth_key) as a container candidate.
/// Swept is NOT the filter here — it is an ADDITIVE tunnel guard (below).
fn containment_candidates(...) -> ArrayVec<(usize, DepthKey), MAX_REGIONS> { ... }
```

- **Cost model (stated honestly):** `O(entities × regions)`, with `regions` small and bounded per shard (§1.3). This is more than the swept-gated filter but bounded; it cross-refs **D-43 #6** (`boundary_depth`'s O(N·M²) precompute) — under containment the depth of each region is computed **once at boot** (regions are static at P3; `depth_key_of(ix)` is a boot-cached `u32`), so the per-tick cost is O(entities × regions), NOT O(N·M²) per tick. The D-43 #6 O(N·M²) concern is retired for the static-registry case and re-armed only when the registry becomes dynamic (P8 moving containers — ledgered).
- **Swept tunnel guard (additive, honest about what it does):** a fast entity can traverse a thin region within one tick. `container(prev)` and `container(cur)` are both computed from point-membership; a full through-and-back leaves both = the outer realm (no membership change → **no-op by design**). For **authority** this is correct (the entity never held the inner realm's authority) — but it must be **stated**, and the band must be sized so a region **cannot** be tunneled at expected speeds: `for_containment_velocity_safe`'s `inset+outset >= v_rel·dt·K_SAFETY` guarantees the dead-zone is at least one safety-multiple of per-tick travel, so a single tick cannot skip the acquire edge. The swept classifier (`segment_shell_crossing`, geometry.rs:171; `segment_aabb_crossing`) is retained as an **additive assertion/guard** (debug-mode: flag a `ThroughAndBack` on a region whose band the point-scan missed — a diagnostic tripwire), **not** as the candidate filter. This corrects the prior draft's "dead reassurance" (the swept was inert because point-membership was false at both ends).

### 2.7 The symmetric re-home decision (replaces `should_commit`)

`should_commit` (geometry.rs:678) is **asymmetric** (Outward immediate, Inward rising-edge). It is **replaced** by one symmetric rule — the container simply differs from the owner, in either direction:

```rust
/// The ONE symmetric re-home decision. Some(dest) iff the deepest HYSTERETIC container this
/// tick differs from the owning realm AND the post-commit cooldown elapsed. No Direction.
/// Escaping the SOI (System→Galaxy) and entering it (Galaxy→System) are the IDENTICAL path.
/// Branchless shim: the two branches (cooldown, equality) each covered ONCE — tested with
/// assert_eq/expect equality per HR5(d), NOT matches!.
#[must_use]
pub fn should_rehome(
    owning: RealmId, container: RealmId, since_commit: Option<u32>, tuning: &BoundaryTuning,
) -> Option<RealmId> {
    if let Some(sc) = since_commit && sc < tuning.k_dwell { return None; }
    (container != owning).then_some(container)
}
```

The hysteresis band (§2.4) sits **before** this — `container` only flips once the dead-zone is crossed, so `should_rehome` needs no dwell of its own (the band *is* the dwell). `k_dwell` is a symmetric post-commit cooldown (anti-thrash), **direction-blind by design and correct** because the rule is symmetric (G2 §1: the direction-blind cooldown was a *bug* only in the asymmetric portal model; it is *correct* here). Deferred item #4 (per-(entity,boundary) cooldown, G2 §4) is **absorbed**: containment does not chain per-boundary Authority self-heals throttled by a per-entity cooldown, because the container change is a single membership event, not a sequence of boundary crossings — but see §5 for the moving-container proof.

---

## 3. The fence-CAS transfer machinery stays UNCHANGED

Everything from `CrossingRequest.to_realm` onward is **byte-for-byte unchanged** (G1 §1, G2 §3, G3):

- **`fan_out_crossing`** (stub.rs:2665) — the ONE HR2 dispatch site — keeps its `(effect × durability)` 2×2 fork. Only the **source of `to_realm`** changes in the two Authority arms:
  ```rust
  // BEFORE (landed): let Some(to_realm) = authority_dest(dir, winner) else { stats.crossing_outward_no_parent += 1; return; };
  // AFTER:           let to_realm = container_realm;  // derived by container(); ALWAYS a RealmId (root ⇒ total)
  ```
  `authority_dest` (stub.rs:2646) is **DELETED**. `crossing_outward_no_parent` (stub.rs:765) is **DELETED** (the ambient root guarantees a destination — the degrade condition cannot arise).
- **`CrossingRequest` / `TransientCrossingRequest`** (intershard.rs:616/630) — **NO shape change**. `to_realm` already carries the destination as a value; it is now populated from `container_realm`.
- **`crossing_transfer_id(subject, subject_fence, attempt)`** (intershard.rs:691) — **UNCHANGED**, no direction in the seed (G2 §3).
- **`RequestInFlight`** (stub.rs:408, entity-keyed) + the `Occupied` suppress — **UNCHANGED** (§5).
- **`on_saga_demote`** entity-keyed self-fence (G2 §1) + latch clear, **`redrive_stranded_crossings`**, **`on_crossing_aborted`**, the directory CAS, **`rebind_pose_to_dest`** (frame.rs:183, the output-side identity seam) — **ALL UNCHANGED** (they consume `to_realm` as a value).

**Net:** only the *destination-decision* changes. The entire saga/CAS/adopt/rebind pipeline is inherited.

---

## 4. The Slice 4b REPLACEMENT + TEST-MIGRATION plan

Slice 4b is **landed, tested, 100%-covered code being replaced.** Every affected test is explicitly retargeted or deleted (G2 §3):

**CODE (stub.rs / geometry.rs):**
| Landed | Action |
|---|---|
| `authority_dest` (stub.rs:2646) | **DELETE** — replaced by `container_realm` |
| `crossing_outward_no_parent` stat (stub.rs:765, incr 2702/2750) | **DELETE** — root ⇒ total, degrade unreachable |
| `should_commit` (geometry.rs:678) | **REPLACE** with `should_rehome` (symmetric) |
| `crossing_candidates` (stub.rs:2576) | **REPLACE** with `containment_candidates` (full scan) |
| `resolve_winner_ix` (geometry.rs:616) | **NOT called** by containment; kept only if the swept guard needs it, else Direction arm of `candidate_beats_ix` deleted |
| winner-change reset (stub.rs:2526-2530) | **DELETE** — per-region bitset makes it wrong |
| `RealmBoundaries` (stub.rs:421) | **RENAME** → `RealmRegions(Vec<RealmRegion>)` |
| `RequestInFlight`, `crossing_transfer_id`, self-fence, redrive | **PRESERVE UNCHANGED** (§5) |

**TESTS — the 5 slice4b unit tests (stub.rs mod tests, G2 §3):**
| Test | Action |
|---|---|
| `slice4b_an_undock_re_homes_outward_to_the_parent_realm` (9127) | **RETARGET** — assert `container()` returns the parent when the entity leaves the child region; destination assertion retargets from `winner.parent` to `container_realm` |
| `slice4b_a_dock_and_undock_resolve_contrasting_destinations` (9164) | **RETARGET** — dock (enter child) → child realm; undock (leave child) → parent; both from containment |
| `slice4b_a_top_level_outward_undock_with_no_parent_degrades_loud` (9230) | **DELETE** — the degrade cannot occur (ambient root); replaced by `containment_is_total` (§ tests) which proves the property positively |
| `slice4b_a_transient_top_level_outward_undock_degrades_loud` (9262) | **DELETE** — same; the transient totality is covered by the shared-path test |
| `slice4b_two_same_depth_outward_boundaries_peel_deterministically` (9321) | **RETARGET** — migrate to the new `depth_beats` permutation-invariance (deepest-wins, no Direction); keep the slice-order-independence proof |

**TEST — the self-heal E2E (`tests/tests/crossing_nesting_e2e.rs`, G2 §2):**
- `crossing_e2e_dest_reevaluates_adopted_entity_self_heal` (158-253) + negative control `crossing_e2e_no_dest_boundary_means_dest_fires_nothing` (259-278): **MIGRATE, do not delete.** Retarget the DEST-planted geometry to a `RealmRegion` (the dest's own region + an inner child), keep the `crossings_requested 0→≥1 for the SAME adopted entity` + negative-control skeleton. This proves the self-heal composition that §5's moving-container argument depends on — it is load-bearing.

**NEW tests:** §9 below.

---

## 5. Moving container: inherit self-heal + per-entity serialization; PROVE/BOUND coverage (review MEDIUM, G2 §1/§2)

The prior draft's latch re-key is **dropped**. We inherit the landed machinery and prove it covers the moving-container case:

**The landed guarantee (G2 §1):** `RequestInFlight` is entity-keyed; `fan_out_crossing`'s durable arm suppresses a second in-flight op per entity (`Occupied` → `crossings_suppressed_in_flight`, stub.rs:2742). So **there is never a second concurrent saga per entity.** `on_saga_demote`'s self-fence is *also* entity-keyed (G2 §1), which is why the direction-blind namespace is safe — splitting the latch without co-keying the self-fence re-opens FINDING-1 (the CRITICAL). We do not split it.

**The self-heal composition (G2 §2):** after a durable crossing COMMITS (entity re-homes to dest, dest OWNS it), the **dest shard's** `containment_candidates` re-evaluates the adopted entity against the dest's regions and fires the *next* re-home on its own. A container change is a **sequence of single-level re-homes, each self-healed by the next shard** — no concurrent multi-saga, no direction key.

**Does it cover the containment moving-container case? Two scenarios, both bounded:**

1. **Entity crosses containers cleanly (SOI → System → Galaxy).** Each hop is one re-home; the dest re-evaluates and fires the next. Self-heal composes N-deep. **Covered** (this is exactly the E2E's proven property, retargeted).

2. **Entity re-homes A→B, then drifts back to A *before B's saga commits*.** The latch to B is held; `redrive_stranded_crossings` re-emits `to_realm=B` verbatim (G2 §3 — desirable for a *stranded* latch). The entity cannot supersede to A mid-flight (no source-side abort-and-redirect exists, and we are NOT adding one — G2 §1 forbids it). **Bounded-velocity proof this cannot strand:** the containment band is velocity-safe (`inset+outset >= v_rel·dt·K_SAFETY`, §2.4) — the dead-zone is ≥ K_SAFETY per-tick travels wide. For the entity to be *inside A's release edge* (triggering the A→B re-home) and then *back inside A's acquire edge* before B's saga commits, it must traverse the full `inset+outset` dead-zone **twice** within the saga-commit latency window. The saga commits within a bounded number of ticks (the fence-CAS is DIRECT in-proc for co-located realms, one tick; cross-host bounded by the redelivering transport's ack window). We **size `k_dwell` ≥ the max saga-commit latency** so the post-commit cooldown (§2.7) suppresses any reverse re-home until B's saga has resolved. At that point the entity is owned by B, and B's dest-side self-heal fires the A re-home cleanly (scenario 1). **The reverse-cross-before-commit is unreachable** given (velocity-safe band width) ∧ (`k_dwell ≥ saga-commit latency`). This is stated as an explicit **precondition on `k_dwell`** (a `debug_assert` + a `BoundaryTuning` cross-config invariant test, mirroring the R-4c `LivenessTuning` geometric invariant pattern already in the codebase).

**The FINDING-2 tripwire (G2 §2, moving-container reparent at P8):** when a container-realm *physically moves between parents* (a Ship re-nesting, P8), mutating `parent` mutates `boundary_depth` — the dominant `depth_beats` key. A straddler during the container's own reparent can flip winner. This is **ledgered to P8** (§8): at P3 the registry is static (`realm_regions_for` returns fixed bodies), so no reparent occurs. The P8 fix (suppress a reparenting region as a winner candidate until the reparent settles, OR assert no entity straddles a container boundary during its own undock) is captured in the new deferred entry.

---

## 6. Small, independently-testable, coverage-clean SLICES

Each compiles, passes `just gate`, and is coverage-100% on its own (HR5 — generic shims branchless, helpers monomorphic, covered in the crate's own unit tests + the integration fixture).

- **C-1 (vd-core, pure):** `RealmRegion` type + `ContainmentBand::for_containment_velocity_safe` + `ContainmentBand::member` + `should_rehome` + `depth_beats`. Unit-test each shape's `signed_distance`-driven membership, band edges (sign-correct acquire/release), velocity-safety (`width_safe_for`), the symmetric commit (assert_eq/expect_err per HR5(d)), and the permutation-invariant `depth_beats`. *No sim wiring.* (Mirrors the landed "pure geometry, nothing calls it yet" pattern.)
- **C-2 (vd-core, pure):** `container()` as the by-construction `RealmId`-returning fold (root identity, no `None`) + `realm_regions_for(seed)` (static P3 bodies). Proptest: totality (returns `RealmId` for any point — no `Option`), deepest-wins, between-two-siblings → shared parent, 3-deep grandparent fallthrough. Plant the `transfer_frame(&IdentityFrames)` input seam (identity no-op). *Still no sim.*
- **C-3 (vd-sim):** swap `RealmBoundaries`→`RealmRegions`; add `ContainmentProgress(RegionMembership bitset)`; wire `containment_candidates` (full scan) + `should_rehome` into `evaluate_one_subject`; **delete** `authority_dest`, the winner-change reset, `crossing_outward_no_parent`; source `to_realm = container_realm` in `fan_out_crossing`. Retarget the 5 slice4b unit tests per §4 (2 retarget, 2 delete, 1 migrate). Trigger stays INERT (default-empty). Unit-test the shared Durable/Transient path (one edit, both classes — G3 §7).
- **C-4 (vd-sim):** the `k_dwell ≥ saga-commit-latency` cross-config invariant (§5) as a `BoundaryTuning` test + `debug_assert`; retarget the `crossing_nesting_e2e` self-heal E2E to `RealmRegion` geometry. Regression: clean N-deep peel self-heals; the reverse-before-commit is suppressed by the sized cooldown. **This is the only behavioural subtlety** — it inherits (does NOT re-key) the landed latch.
- **C-5 (bins):** rename `resolve_realm_boundaries`/`guard_boundaries_in_realm` → `resolve_realm_regions`/`guard_regions_in_realm` (relaxed to permit own realm **+ seed-derived ancestor chain**); add `guard_regions_nest` (§6-G). Unit-test the validator (loud fail on: two roots, zero roots, unresolved parent, cycle, duplicate `realm`, region count > MAX_REGIONS).
- **C-6 (playground + gate):** the **3-deep** `regions.json` (galaxy ⊃ System(7) ⊃ Planet-SOI(7); System(8) sibling) via `realm_regions_for`; extend `dev_cluster_smoke` with the mandate round-trip; `just realm-roundtrip` in `just gate`. HR4 G-IDENTICAL: same fixture on a Shell region now, box region at P4.
- **C-7 (deferred-doc):** amend DEFERRED.md D-43 + add D-44 (§8). No code.

**Ordering:** C-1/C-2 pure, zero-risk. C-3 is the load-bearing swap of green code (retargets/deletes tests explicitly) but stays inert in production. C-4 is the sole behavioural subtlety. C-5/C-6 turn it on in the playground only. rayon is **not** in this arc (task-#133 LOCKED decision 4 — later slice, behind the libs-investigate gate).

### 6-G. `guard_regions_nest` — topological now, geometric ledgered (review LOW/MEDIUM)

At P3, do only the checks that are exact without cross-frame geometry:
1. **Exactly one region has `parent: None`** (the root; `root_ix` for `container()`'s fold identity).
2. **Every `parent: Some(r)` resolves** — `r` equals the `.realm` of exactly one region.
3. **No cycles** in the parent chain; every chain reaches the single root.
4. **`realm` is UNIQUE per region** (so `boundary_depth`'s `find` is deterministic — review MEDIUM).
5. **Region count ≤ MAX_REGIONS** (so the bitset fits).
6. **Relax `guard_regions_in_realm`:** permit the shard's **own realm + its authored/seed-derived ancestor chain** (not just the hosted realm), and validate the extra regions form a valid chain to the one root (G3's HR1 reconciliation — the ancestor geometry is seed-replicated config, not inter-shard bytes).

**LEDGER to P4/P5:** the geometric subset check (child volume ⊆ parent volume for arbitrary Shape×Shape) — it needs the same cross-frame math deferred to P4/P5 (child and parent may be in different frames). Do NOT claim "volume subset checked at boot" when only topology is validated (the review's LOW finding).

---

## 7. HR compliance argument

- **HR1 (sealed shards):** region geometry is **seed-derived closed-form `f(seed)` replicated by construction** — every shard recomputes the identical forest locally from the shared universe seed (`realm_regions_for`, §1.3). **No shared mutable state, no inter-shard bytes.** The realm→node directory stays disjoint (no geometry, G3). `guard_regions_in_realm` relaxed to own realm + seed-derived ancestor chain (§6-G) — the extra geometry is replicated config, not a cross-shard fetch. The containment decision is local; only the node resolution is orchestrator-side (`head(Realm(to_realm))`, unchanged).
- **HR2 (generic transfer):** detection is **shared** — both the dot loop and transient loop call the same `evaluate_one_subject` → `containment_candidates` → `should_rehome` → the same `fan_out_crossing`. Durability is `durability_of(entity)` (a kind-tag lookup), not the loop of origin. **One edit fixes both classes.** Transient vs Durable is policy fan-out on ONE machinery (batched `TransientCrossingRequest` vs latched `CrossingRequest`); the container decision is identical. **Answer to (I):** both classes carry the per-region hysteretic bitset (shared detection); the *fan-out* differs, not the *membership*. (Transients could recompute statelessly, but keeping the shared bitset avoids a per-class branch in the detector — the bitset is cheap, §2.3 — so we keep detection uniform and fork only at emission, preserving HR2's "one machinery.")
- **HR3 (one tooling):** ONE `should_rehome`, ONE `container`, ONE `fan_out_crossing`, ONE `RealmRegions`. **No `match` on realm kind** anywhere — `RealmId::{System,Planet,Ship,Station,Area}` is data flowing through `container()`/`fan_out_crossing`, never branched on in features (the `frame_for_realm` total map, pose.rs:114, stays the RealmId→frame tooling). `ShardProfile` capability configs are untouched.
- **HR5 (100% coverage):** `container()` returns `RealmId` **by construction** (root-identity fold — **no `None` arm**, review CRITICAL fix). `should_rehome`/`depth_beats`/`ContainmentBand::member` are branchless monomorphic shims with each branch covered once, tested with `assert_eq`/`expect_err` equality (HR5(d)), splitting short-circuit branches. `retain_live<V>` monomorphizes cleanly over the new `RegionMembership`. Deleting `authority_dest`, the winner-change reset, and (if unused) `candidate_beats_ix`'s Direction arm **removes** regions rather than adding uncoverable ones. The velocity-safe band's `debug_assert` is coverage-off under `#[cfg_attr(coverage_nightly, coverage(off))]` per the standing convention.

---

## 8. P3-now vs P4/P5-deferred + DEFERRED.md ledger

| Piece | When | Why (ground) |
|---|---|---|
| `RealmRegion`, `container()` by-construction, `ContainmentBand::for_containment_velocity_safe`, `should_rehome`, `depth_beats`, full-scan `containment_candidates`, per-region bitset, `realm_regions_for` (static bodies), frame-seam plant (identity) | **P3 now** | Pure geometry + one-frame identity math (G3); registry inert until playground plants |
| Delete `authority_dest`/`crossing_outward_no_parent`/winner-reset; retarget/delete the 5+E2E tests | **P3 now** | Replacing landed green code (G1/G2) |
| 3-deep playground + mandate round-trip gate + `just realm-roundtrip` | **P3 now** | Static regions, cell-0, identity ephemeris |
| `k_dwell ≥ saga-commit-latency` cross-config invariant | **P3 now** | Bounds the moving-container reverse-cross (§5) |
| **Moving regions** (`center`/`shape` as `f(seed, universe_tick)`); seed→celestial-parameter generator; SPIKE-6a cross-binary determinism gate | **P4/P5** | Real ephemeris; today `IdentityFrames`; parameters not seed-derived in code (G3) |
| **Non-identity `transfer_frame`** on the input seam (entity in `PlanetCentered`, region in `SystemSpace`) | **P4/P5** | Seam planted at P3 as identity (§2.5); swap `IdentityFrames` for real `FrameContext` — additive |
| **Non-zero cell arithmetic** in `cur - center` (integer-cell + bounded offset) | **P4/P5** | D-41 MATH owed |
| **Geometric subset check** in `guard_regions_nest` (child ⊆ parent, arbitrary Shape×Shape) | **P4/P5** | Needs cross-frame math (§6-G) |
| **Moving-container reparent** (`boundary_depth` mutates when a Ship re-nests; the FINDING-2 straddler guard) | **P8** | Only exercisable when ships-as-realms physically move (G2 §2) |
| Dedicated `RealmId::Galaxy` arm | **P4+** | `System(0)` suffices at P3 (cosmetic) |

**DEFERRED.md ledger entries:**
- **Amend D-43 (4b-STATUS):** items **#4 (asymmetric enter/exit) and #5 (reparent-on-undock) → RESOLVED-BY-CONTAINMENT** (the symmetric rule + ambient-root partition eliminate the asymmetry and the no-parent degrade). **#6 (`boundary_depth` O(N·M²))** → retired for the static registry (depth cached at boot); re-armed only for the P8 dynamic registry. Deferred **#4-cooldown** (per-(entity,boundary)) → absorbed by the symmetric single-membership-event model (§2.7). Deferred **#5-`winner_ix`** → moot (per-region bitset replaces the single winner slot).
- **New D-44 "ephemeris / cross-frame / moving-container containment":** (a) moving regions as `f(seed, tick)` + the seed→celestial-parameter generator + SPIKE-6a determinism gate (P4/P5); (b) non-identity input-side `transfer_frame` (seam planted, P4/P5 fills `FrameContext`); (c) D-41 non-zero-cell arithmetic in the containment subtract (P4/P5); (d) geometric subset check in `guard_regions_nest` (P4/P5); (e) the FINDING-2 moving-container reparent straddler guard + `boundary_depth` dynamic-registry re-arm (P8). Each entry names WHAT (proper solution), WHERE (file), WHEN (slice/phase) per the DEFERRED.md discipline — flips to 🟩 when the phase lands.

---

## 9. The mandate round-trip test / gate (3-DEEP — review MEDIUM)

The playground is **3-deep** to exercise the real mandate chain (galaxy root ⊃ System ⊃ Planet-SOI). `realm_regions_for(seed)` yields:

```
galaxy   : RealmRegion { realm: System(0), center: origin, frame: GalaxySpace,  shape: Shell{GALAXY_R}, band, parent: None }
system7  : RealmRegion { realm: System(7), center: c7,     frame: GalaxySpace,  shape: Shell{SOI_7},    band, parent: Some(System(0)) }
planet7  : RealmRegion { realm: Planet(7), center: p7,     frame: GalaxySpace,  shape: Shell{SOI_P7},   band, parent: Some(System(7)) }
system8  : RealmRegion { realm: System(8), center: c8,     frame: GalaxySpace,  shape: Shell{SOI_8},    band, parent: Some(System(0)) }
```

**Engine unit/proptests (vd-sim, virtual clock):**
- `containment_is_total` — `container()` returns a `RealmId` (type-level, no `Option`) for any point + any region set including the root. Replaces the deleted `*_degrades_loud` tests with a positive property.
- `escape_soi_lands_in_system_not_galaxy` — a dot leaving `SOI_P7` (the planet SOI) re-homes to **System(7)** (the immediate parent), **NOT the galaxy** — the exact mandate assertion ("escape the SOI → immediately in the Star System realm"). Then leaving `SOI_7` re-homes to the galaxy. **Two nested exits, grandparent fallthrough exercised.**
- `symmetric_recross` — a scripted path Planet(7) → System(7) → galaxy → System(8) → galaxy → System(7) emits the expected `to_realm` sequence with the inherited per-entity latch permitting each reverse crossing (the self-heal composes it; no re-key).
- `hysteresis_no_flap_at_container_change` — a dot jittering at the **container-CHANGE boundary** (SOI_7's outset edge) by MORE than a per-tick step but LESS than the band width, over > `k_dwell` ticks, emits **zero** re-homes (review HIGH: tests the non-trivial case at the container change, sized velocity-safely).
- `deepest_wins` — a point inside both `SOI_7` and `SOI_P7` commits to Planet(7) (deeper); inside neither but inside galaxy → System(0).

**Devcluster smoke gate** (`crates/bins/tests/dev_cluster_smoke.rs`): boot the 3-deep playground, inject a dot on the Planet(7)→System(7)→galaxy→System(8)→System(7) path via the harness, assert the directory `head(Realm(...))` ownership flips through the full chain and the committed pose rebinds each hop (`rebind_pose_to_dest`, unchanged). `just realm-roundtrip` folded into `just gate`.

---

## 10. EXACT ENGINE-MODIFICATION LIST (human approval gate)

### `crates/core/src/geometry.rs` (vd-core)
1. **ADD** `struct RealmRegion` (§1.1, with `frame: FrameRef`) + shell/box constructors (mirroring `RealmBoundary::shell`/`aabb` at geometry.rs:399/498, minus `to_realm`).
2. **ADD** `struct ContainmentBand` + `for_containment_velocity_safe` + `member` + `width_safe_for` (§2.4) — distinct newtype, signed-metre units.
3. **ADD** `should_rehome` (§2.7) + `depth_beats` (§2.2).
4. **DELETE** `should_commit`'s asymmetric role (replaced by `should_rehome`). **DELETE** `candidate_beats_ix`'s Direction arm (geometry.rs:648) if the swept guard no longer needs it (HR5 region removal); keep `resolve_winner_ix` only for the swept guard or delete if unused.
5. **KEEP** `Boundary::signed_distance` (275, now wired), `box_signed_distance`, `segment_shell_crossing`/`segment_aabb_crossing`, `OverlapBand` (still used by the Interest path).

### `crates/sim/src/stub.rs` (vd-sim)
6. **RENAME** `RealmBoundaries(Vec<RealmBoundary>)` → `RealmRegions(Vec<RealmRegion>)` (421); default-empty (877) unchanged.
7. **ADD** `struct RegionMembership` (u64 bitset) + `ContainmentProgress(BTreeMap<EntityId, RegionMembership>)` (§2.3); wire `retain_live` (2485) over it (4th monomorphization).
8. **REPLACE** `crossing_candidates` (2576) → `containment_candidates` (full per-tick scan, frame-aware via `transfer_frame(&IdentityFrames)`, §2.5/2.6).
9. **REPLACE** the `should_commit` call in `evaluate_one_subject` (2545) with the `container()` computation + `should_rehome`; **DELETE** the winner-change reset (2526-2530).
10. **DELETE** `authority_dest` (2646); in `fan_out_crossing` (2665) both Authority arms source `to_realm = container_realm`; **DELETE** the `crossing_outward_no_parent` degrade (2702/2750) + the stat field (765).
11. **KEEP UNCHANGED** `RequestInFlight` (408), `LatchedCrossing` (349), `redrive_stranded_crossings`, `on_saga_demote`/`on_crossing_aborted` self-fence (the FINDING-1 surfaces — §5).
12. **RETARGET/DELETE** the 5 slice4b unit tests (9127-9389) per §4.

### `crates/wire/src/intershard.rs` (vd-wire — frozen contract)
13. **NO SHAPE CHANGE.** `CrossingRequest` (616) / `TransientCrossingRequest` (630) / `crossing_transfer_id` (691) / `DemoteCmd` (519) untouched. `to_realm` already carries the derived value. (`dir`/`new_parent` never existed — nothing to avoid, §0.)

### `crates/node/src/saga_runtime.rs` (vd-node)
14. **NO CHANGE** — `head(Realm(to_realm))`, the CAS, `build_crossing`, `rebind_pose_to_dest` all consume `to_realm` as a value (§3, G3).

### `crates/core/src/worldgen.rs` (vd-core — NEW module)
15. **ADD** `realm_regions_for(seed_universe) -> Vec<RealmRegion>` (§1.3) — static bodies at P3, the P4/P5 ephemeris fill-in ledgered.

### `crates/bins/src/lib.rs` + `crates/bins/src/bin/shard.rs` (bins)
16. **RENAME** `resolve_realm_boundaries`/`guard_boundaries_in_realm` (1373/1400) → `resolve_realm_regions`/`guard_regions_in_realm`; boot via `realm_regions_for` (or parse `regions.json` single-sourced with the client's `--realm-boxes`); **RELAX** to own realm + ancestor chain (§6-G).
17. **ADD** `guard_regions_nest` (§6-G) — topological checks; geometric subset ledgered.

### Non-engine (playground + tests + gate)
18. **ADD** the 3-deep `regions.json` playground (galaxy + System(7) + Planet(7) + System(8)) via `realm_regions_for` (§9) — `scripts/`/devcluster.
19. **ADD** the 5 vd-sim unit/proptests (§9) + extend `crates/bins/tests/dev_cluster_smoke.rs` with the mandate round-trip; **MIGRATE** `tests/tests/crossing_nesting_e2e.rs` self-heal to `RealmRegion` geometry.
20. **ADD** `just realm-roundtrip` in `just gate`.
21. **EDIT** `docs/design/DEFERRED.md` — amend D-43, add D-44 (§8).

---

**Files cited (all absolute, under `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/`):** `crates/core/src/geometry.rs` (`RealmBoundary`:381, `Boundary::signed_distance`:275, `box_signed_distance`:293, `OverlapBand`/`update_membership`:143, `for_soi_velocity_safe`:163, `width_safe_for`:136, `resolve_winner_ix`:616, `candidate_beats_ix`:641, `should_commit`:678), `crates/core/src/pose.rs` (`RealmId`:21, `frame_for_realm`:114), `crates/core/src/frame.rs` (`rebind_pose_to_dest`:183, `IdentityFrames`), `crates/sim/src/stub.rs` (`CrossingState`:365, `RequestInFlight`:408, `RealmBoundaries`:421, `crossing_outward_no_parent`:765, `evaluate_one_subject`:2495, winner-reset:2526, `crossing_candidates`:2576, `candidate_direction`:2603, `boundary_depth`:2615, `authority_dest`:2646, `fan_out_crossing`:2665/degrade 2702/2750, slice4b tests:9127-9389), `crates/wire/src/intershard.rs` (`DemoteCmd`:519, `CrossingRequest`:616, `TransientCrossingRequest`:630, `crossing_transfer_id`:691), `crates/wire/src/admin.rs` (`DirectoryEntryView`:23), `crates/bins/src/lib.rs` (`resolve_realm_boundaries`:1373, `guard_boundaries_in_realm`:1400), `crates/bins/src/bin/shard.rs` (:105-119), `tests/tests/crossing_nesting_e2e.rs` (:158-278), `docs/design/DEFERRED.md` (D-43). **HEAD:** `7a87154`.