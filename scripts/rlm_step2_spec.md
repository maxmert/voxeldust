I have full verification of every load-bearing seam. `region_signed_distance` is confirmed (referenced at stub.rs:2860, test at worldgen.rs:951 — frame-aware, identity at P3). `realm_neighbourhood_for` at worldgen.rs:449 is exactly the O(all realms) `realm_regions_for` scan the C1 seam must avoid at the AoI call site. All findings are grounded. Here is the final spec.

---

# STEP 2 BUILD SPEC (FINAL) — Realm Lifecycle Machinery: the generic per-realm AoI DECISION

**Verification basis:** every seam Read at HEAD (`worktree-new-system`). Line numbers current. This folds C-1 (serde default), H-1 (frame-algebra), H-2 (permuted determinism), M-1/M-2/M-3, and L1/L2/L3 into an implementable spec. Files touched: `crates/core/src/{geometry.rs,worldgen.rs,celestial.rs}`, `crates/sim/src/{stub.rs,runtime.rs}`, `crates/node/src/follower.rs`, plus enumerated test-rig updates.

**Frozen seams (EMIT only, no change):** `InterShardFlow::RealmDemand{child:RealmCoord, parent_fence:Fence, verb:DemandVerb, universe_tick}` (intershard.rs:540), classified `SideEffecting{FencedKey{parent_fence}}` (intershard.rs:441) + `ReDriven` (intershard.rs:512). `DemandVerb::{SpinUp=0,KeepAlive=1,Empty=2,TearDown=3}`. `RealmCoord::child/parent/path` (realm_coord.rs:83/69/45).

---

## 1. NEW/CHANGED TYPES

### 1a. `AoiConfig` + ctor — `crates/core/src/geometry.rs`

Place after `ContainmentBand` (ends `:766`), before `RealmRegion` (`:774`). Mirror `ContainmentBand::for_containment_velocity_safe` (`:715`) arm-for-arm — the fallible template with a reject arm; NOT `OverlapBand::for_soi_velocity_safe` (`:605`, no reject arm). Fields **private** (the invariant is unconstructible-if-violated).

```rust
/// A per-realm Area-of-Interest hysteresis band (METRES) for demand-driven realm lifecycle. DISTINCT
/// from OverlapBand/ContainmentBand: it drives a child SHARD's spin-up/down, not entity membership.
/// `spin_up_r_m` (a child within this range of an occupant is DEMANDED live) and `tear_down_r_m >
/// spin_up_r_m` (released only past this larger radius) straddle a dead-zone so an occupant loitering
/// at the edge cannot flap the child. `grace_ticks` holds a would-be release that many ticks after the
/// last in-range observation (the temporal half of the anti-thrash; the geometric half is the gap).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AoiConfig {
    spin_up_r_m: f64,
    tear_down_r_m: f64,
    grace_ticks: u32,
}

impl AoiConfig {
    /// The VELOCITY-SAFE constructor — the ONLY fallible way to build a LIVE band. `spin_up =
    /// base_extent·spin_up_factor`; `tear_down = max(base_extent·tear_down_factor, spin_up + need)`,
    /// `need = |v_rel|·dt·(K_SAFETY + k_safety_extra)` — the gap WIDENED so a body closing at `v_rel`
    /// (m/s) over `dt`-second ticks can neither skip the band nor thrash the child. `debug_assert!`
    /// pins the width invariant. Fallible LOUD unless `0 < spin_up < tear_down` (the exact flap this
    /// type prevents). NOTE: at zero factor the generator calls [`AoiConfig::inert`], never this.
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] unless `0 < spin_up < tear_down`.
    #[must_use = "the Result carries a BandError that must not be dropped"]
    pub fn for_velocity_safe(
        base_extent: f64, spin_up_factor: f64, tear_down_factor: f64,
        v_rel: f64, dt: f64, grace_ticks: u32, k_safety_extra: f64,
    ) -> Result<AoiConfig, BandError> {
        let spin_up = base_extent * spin_up_factor;
        let need = v_rel.abs() * dt * (K_SAFETY + k_safety_extra);
        let tear_down = f64::max(base_extent * tear_down_factor, spin_up + need);
        if spin_up > 0.0 && tear_down > spin_up {
            let band = AoiConfig { spin_up_r_m: spin_up, tear_down_r_m: tear_down, grace_ticks };
            debug_assert!(band.width_safe_for(v_rel, dt));
            Ok(band)
        } else {
            Err(BandError::InvalidEdges)
        }
    }

    /// The inert AoI (zero factor / walk-scale): `spin_up == 0` ⇒ nothing is ever in range ⇒ no demand.
    /// Distinct from the fallible ctor so the byte-identity path never touches the reject arm. This is
    /// the `#[serde(default)]` for `RealmRegion.aoi` (C-1) — a legacy `regions.json` missing the field
    /// decodes to inert (behaviour-identical).
    #[must_use]
    pub fn inert() -> AoiConfig {
        AoiConfig { spin_up_r_m: 0.0, tear_down_r_m: 0.0, grace_ticks: 0 }
    }

    #[must_use] pub fn spin_up_r_m(&self) -> f64 { self.spin_up_r_m }
    #[must_use] pub fn tear_down_r_m(&self) -> f64 { self.tear_down_r_m }
    #[must_use] pub fn grace_ticks(&self) -> u32 { self.grace_ticks }

    fn width_safe_for(&self, v_rel: f64, dt_s: f64) -> bool {
        (self.tear_down_r_m - self.spin_up_r_m) >= v_rel.abs() * dt_s * K_SAFETY
    }

    /// AoI hysteresis over a scalar distance (branchless; both bounds compile-visible compares). `hold`
    /// and `acquire` are split into locals so no `&&`/`||` short-circuit region is uncovered (HR5).
    /// `min_dist` is the smallest distance from ANY occupant to the child — the caller reduces.
    #[must_use]
    pub fn in_range(&self, was_in: bool, min_dist: f64) -> bool {
        let hold = min_dist <= self.tear_down_r_m;
        let acquire = min_dist <= self.spin_up_r_m;
        was_in && hold || acquire
    }
}
```

Generic-free ⇒ no per-monomorphization gotcha; every branch (reject arm, `debug_assert`, `in_range`'s three compares) is monomorphic and covered by §4 tests.

### 1b. `OrbitalElements::v_peri` — `crates/core/src/celestial.rs` (after `mean_motion` `:166`)

```rust
/// Max orbital speed (periapsis): v_peri = sqrt(mu/p)·(1+e), p = a(1-e²). Closed-form f(elements) —
/// v_rel is seed-derivable. Used to widen a MOVING child's AoI band by its own orbital motion (M2).
#[must_use]
pub fn v_peri(&self) -> f64 {
    let p = self.sma * (1.0 - self.ecc * self.ecc);
    (self.mu() / p).sqrt() * (1.0 + self.ecc)
}
```

### 1c. `RealmRegion.aoi` field (C-1 / L1 — the fix that supersedes the draft's flag #6) — `geometry.rs:774`

`RealmRegion` derives `Serialize, Deserialize` (`:774`, doc `:771` "Serde only for the client's `--realm-boxes`"), fields **all `pub`** (NOT private — the draft's accessor claim is WRONG). It is `serde_json`-round-tripped: bins write `regions.json` (`bins/src/lib.rs:1460 write_regions_json`, `:1438`/`:1454` seed/visual forests), client parses `Vec<RealmRegion>` via `serde_json::from_str` (`client/src/realm_scene.rs:282 parse_regions` → `:213 from_regions_json`). It is NOT on any `InterShardFlow` arm (grep-confirmed), so no frozen-wire violation — but a non-defaulted field makes `serde_json` **reject every legacy `regions.json`** (the `MalformedJson` arm at `realm_scene.rs:90`). Fix: `#[serde(default)]`.

```rust
    pub band: ContainmentBand,
    /// Per-realm AoI hysteresis (RLM Step 2), populated by `worldgen::to_regions` from the seed. Inert
    /// (`spin_up == 0`) at walk-scale ⇒ no demand ⇒ byte-identity of BEHAVIOR. `#[serde(default)]` so a
    /// legacy `regions.json` (written before this field) decodes to inert — the client/devcluster
    /// on-disk contract stays forward-compatible (C-1). Byte-identity of on-disk BYTES does NOT hold
    /// (the field is additive to the JSON); a freshly-written file round-trips, a legacy one parses.
    #[serde(default = "ContainmentBand_default_bridge_replace_me")] // see note
    pub aoi: AoiConfig,
    pub parent: Option<RealmId>,
```

Path-default: `#[serde(default = "AoiConfig::inert")]` — `AoiConfig::inert` is a zero-arg fn returning `AoiConfig`, exactly the serde `default = "path"` signature. (`RealmRegion` derives `Copy`; `AoiConfig` is `Copy`, so `Copy` is preserved.)

### 1d. `InterestConfig` on `UniverseConfig` (M-2 single-source) — `crates/core/src/worldgen.rs`

7th sub-struct beside `BandConfig` (`:639`), added to `UniverseConfig` (`:662`). Mirror `BandConfig` + `::build()` (`:645/648`). **M-2:** the widening needs `v_rel`/`dt`, sim-resident today (`StubConfig.move_speed_mps` `stub.rs:76`, `tick_dt_s` `:78`). To keep `to_regions` a pure `f(UniverseConfig)` AND avoid a two-home DRY split, put `occupant_v_max_mps` + `tick_dt_s` ON `InterestConfig`, and add a BOOT `debug_assert!` cross-check at `register_stub_shard` that `StubConfig.tick_dt_s == config.interest.tick_dt_s` and `StubConfig.move_speed_mps·time_multiplier == config.interest.occupant_v_max_mps` (the `DirectoryTuning::validate` fail-loud pattern, `stub.rs:114`). This single-sources the physical `dt` so the anti-thrash pad is measured against the `dt` the sim integrates at.

```rust
/// Per-realm AoI radii as UNIFORM FACTORS of the realm's own finite extent (HR3: no match-on-kind — a
/// bigger realm reaches proportionally farther). Also carries the widening inputs (`occupant_v_max_mps`,
/// `tick_dt_s`) so `to_regions` is a pure f(UniverseConfig) — a boot debug_assert cross-checks them
/// against StubConfig's live values (M-2, no two-home drift). walk_scale sets `spin_up_factor = 0` ⇒ AoI
/// inert ⇒ byte-identity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterestConfig {
    pub spin_up_factor: f64,
    pub tear_down_factor: f64,
    pub grace_ticks: u32,
    pub k_safety_extra: f64,
    pub occupant_v_max_mps: f64,  // == StubConfig.move_speed_mps · time_multiplier (boot-asserted)
    pub tick_dt_s: f64,           // == StubConfig.tick_dt_s (boot-asserted)
}

impl InterestConfig {
    /// Build the per-realm AoI band from the realm's own extent + the child's own orbital closing speed.
    /// At walk-scale (`spin_up_factor == 0`) returns the inert band BRANCHLESSLY — NEVER through the
    /// fallible ctor (whose reject arm the byte-identity path must not touch). `v_child` is the moving
    /// child's `v_peri` (0 for a static child). Fallible only for the LIVE case, factors ordered by
    /// construction.
    pub fn build(&self, finite_extent: f64, v_child: f64) -> Result<AoiConfig, BandError> {
        if self.spin_up_factor <= 0.0 {
            Ok(AoiConfig::inert())
        } else {
            AoiConfig::for_velocity_safe(
                finite_extent, self.spin_up_factor, self.tear_down_factor,
                self.occupant_v_max_mps + v_child, self.tick_dt_s,
                self.grace_ticks, self.k_safety_extra,
            )
        }
    }
}
```

`spin_up_factor <= 0.0` is a MONOMORPHIC helper branch — inert arm covered by walk test, live arm by visual test.

- `walk_scale()` (`:722`) + `canonical()` (`:779`): add `interest: InterestConfig { spin_up_factor: 0.0, tear_down_factor: 0.0, grace_ticks: 0, k_safety_extra: 0.0, occupant_v_max_mps: 0.0, tick_dt_s: WALK_TICK_DT_S }` — factor 0 ⇒ inert ⇒ byte-identity.
- `visual_scale()` (`:797`, `let mut cfg = walk_scale()`): override `cfg.interest = InterestConfig { spin_up_factor: <named const>, tear_down_factor: <named const>, grace_ticks: <const>, k_safety_extra: <const>, occupant_v_max_mps: <visual occupant speed>, tick_dt_s: <visual dt> }` — the ONE place LIVE AoI turns on. All factors are documented seed-relative named consts (HR3: radius = factor × extent, no magic).

### 1e. `AoiMembership` / `AoiState` — `crates/sim/src/stub.rs` (twin of `ContainmentProgress` `:490`)

```rust
/// Per-CHILD AoI hysteresis + grace, keyed by child RealmPath (globally unique — Step 3 dedups on
/// child.path()). Twin of ContainmentProgress but child-path-keyed (not entity-keyed). BTreeMap
/// (no default-hasher HashMap in sim — determinism).
#[derive(Resource, Debug, Default)]
pub struct AoiMembership(pub BTreeMap<RealmPath, AoiState>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AoiState {
    was_in: bool,
    grace_remaining: u32,
}
```

Eviction: widen `retain_live<V>` (`stub.rs:2785`, currently `BTreeMap<EntityId,V>`) to `retain_live<K: Ord + Clone, V>(map, live: &BTreeSet<K>)` — one covered monomorphization more, DRY. The AoI live-set is the child-path set produced this tick.

---

## 2. `child_placements` + `children_within` + `authored_realm_snaps` REFACTOR

### 2a. `child_placements` — UNIFIED accessor (H2), `RealmRegions` in `stub.rs` (beside `authored_realm_snaps` `:595`)

Folds movers (`self.moving`, orbital) + static-center children into ONE list. **Both the observer feed AND the AoI loop consume it.** Returns the region ref alongside so callers get `finite_extent`, `aoi()`, `parent` without a second lookup.

```rust
/// The UNIFIED per-tick placement of EVERY direct child (H2): movers authored from OrbitalElements,
/// static children at their region `center` (P6/P9: signal-authored later). ONE position code-path both
/// the observer feed AND the AoI loop consume. Direct children only (`parent == Some(own_realm)`);
/// ancestor/self/root regions excluded. Poses stamped in the shard's OWN (ambient-root) frame — the
/// SAME frame `authored_realm_snaps` uses (`own`), so the AoI distance and the observer feed measure the
/// same geometry (H-1). Returns `(&RealmRegion, StampedPose)` so callers read extent/aoi/parent without
/// a re-scan.
#[must_use]
pub fn child_placements(&self, tick_hz: f64, tick: UniverseTick)
    -> Vec<(&RealmRegion, StampedPose)> {
    let own = self.regions.iter().find(|r| r.parent.is_none())
        .map_or(FrameRef::GalaxySpace, |r| r.frame);
    let own_realm = self.regions.iter().find(|r| r.parent.is_none()).map(|r| r.realm);
    let secs = secs_since_epoch(tick.0, tick_hz);
    self.regions.iter()
        .filter(|r| is_direct_child(r.parent, own_realm))          // monomorphic predicate
        .map(|r| (r, place_child(&self.moving, r, own, secs, tick))) // monomorphic placer
        .collect()
}
```

Monomorphic helpers (all branches here, covered once):

```rust
/// A region is a DIRECT child iff its parent IS the shard's own (ambient-root) realm. Monomorphic so
/// both arms (`Some(own)` match / mismatch, `None` own) are covered once (HR5).
fn is_direct_child(region_parent: Option<RealmId>, own_realm: Option<RealmId>) -> bool {
    match (region_parent, own_realm) {
        (Some(p), Some(own)) => p == own,
        _ => false,
    }
}

/// Place ONE child: a mover from its ephemeris (`orbital_state`), a static child at its `center`. The
/// `match` covered once here (mover test + static test), NOT per generic mono.
fn place_child(
    moving: &BTreeMap<RealmId, OrbitalElements>,
    r: &RealmRegion, own: FrameRef, secs: f64, tick: UniverseTick,
) -> StampedPose {
    match moving.get(&r.realm) {
        Some(elements) => {
            let st = orbital_state(elements, secs);
            let mut p = StampedPose::at_rest(own, st.position, tick);
            p.vel = st.velocity;
            p
        }
        None => StampedPose::at_rest(own, r.center.offset(), tick), // static-center; vel = 0
    }
}
```

### 2b. `authored_realm_snaps` becomes a thin MOVERS-ONLY filter (preserves the observer feed byte-identity — grounding risk #4)

```rust
pub fn authored_realm_snaps(&self, tick_hz: f64, tick: UniverseTick) -> Vec<RealmSnap> {
    self.child_placements(tick_hz, tick)
        .into_iter()
        .filter(|(r, _)| self.moving.contains_key(&r.realm))   // MOVERS ONLY — today's rows
        .map(|(r, pose)| RealmSnap { realm: r.realm, pose })
        .collect()
}
```

Walk-scale: `moving` empty ⇒ `child_placements` yields static rows, the filter drops all ⇒ empty ⇒ byte-identical to today (the `emit_realm_frames`/`authored_realm_snaps` gates at `stub.rs:7816/7847` still pass). The AoI loop consumes `child_placements` UNFILTERED (it needs static children too). **H-1 note:** because `authored_realm_snaps` currently authors movers in the `own` frame via `orbital_state`, and `child_placements` does the identical thing, the refactor is a pure extraction — no behavior drift.

### 2c. `children_within` (C1 seam, spatial-index-SHAPED, LINEAR body) — `worldgen.rs` (beside `realm_neighbourhood_for` `:449`)

**Correction to the draft sketch:** `children_within` must NOT wrap `realm_neighbourhood_for` — that calls `realm_regions_for` (a full-forest O(all realms) scan, `:450`) at the call site, the exact C1 anti-pattern. It operates on the ALREADY-bounded direct-child slice the sim hands it. **H-1:** it takes frame-local `DVec3` positions (`.offset()`), documented that occupant and child share a cell through P3 (cross-cell fold is P4/P5-owed, like `LatticePos::at` `pose.rs:174`).

```rust
/// Direct children of `parent` within `radius` of `occupant_pos`, as (child RealmCoord, distance). The
/// SPATIAL-INDEX-SHAPED lifecycle query: today a LINEAR fold over the caller's already-bounded
/// direct-child slice (correct under sparse occupancy — MAX_REGIONS caps live direct children); the
/// P6/D-9 spatial index replaces the linear body WITHOUT changing this signature. Positions are
/// FRAME-LOCAL DVec3 (`.offset()`) in the parent's OWN frame — occupant and child MUST share a cell
/// through P3 (every pose is cell-ZERO; the cross-cell fold is P4/P5-owed). Yields RealmCoord (via
/// parent.child(level)) so a not-yet-spawned child is nameable — NOT the lossy RealmId.
#[must_use]
pub fn children_within<'a>(
    parent: &'a RealmCoord,
    occupant_pos: DVec3,
    radius: f64,
    direct_children: &'a [(RealmLevel, DVec3)],   // (child level from seed, its authored frame-local pos)
) -> impl Iterator<Item = (RealmCoord, f64)> + 'a {
    direct_children.iter().filter_map(move |(level, pos)| {
        aoi_within(*pos, occupant_pos, radius).map(|d| (parent.child(*level), d))
    })
}

/// The monomorphic distance predicate (HR5: the `?`/compare live here, `children_within`'s closure is a
/// branchless map). `Some(d)` iff `d <= radius`.
fn aoi_within(child_pos: DVec3, occupant_pos: DVec3, radius: f64) -> Option<f64> {
    let d = (child_pos - occupant_pos).length();
    (d <= radius).then_some(d)
}
```

### 2d. `direct_child_levels` generator accessor (the child-LEVEL roster, resolves risk #3 / L1) — `worldgen.rs`

The AoI loop needs each direct child's `RealmLevel` to call `parent.child(level)`; `RealmId → RealmLevel` inversion is lossy/P4-owed. The generator KNOWS its children (it created them). **L1:** derive from the IDENTICAL `generate_system_forest` the shard boots, keyed `b.parent == Some(hosted)` — mirror `moving_children`'s `:206` filter — so a region present in `RealmRegions.regions` but absent from the roster (a silent spin-up hole) cannot occur.

```rust
/// The direct-child `RealmLevel` roster for `hosted` — from the SAME seed forest `to_regions`/
/// `moving_children` consume (`b.parent == Some(hosted)`), so the AoI child roster and the containment
/// region roster never diverge (L1). Closed-form f(seed, hosted). At P3 hosted is a `RealmId`; the
/// full lineage that builds the child's RealmCoord is the shard's own_coord (§3, StubConfig.own_coord).
#[must_use]
pub fn direct_child_levels(seed_universe: u64, hosted: RealmId) -> Vec<RealmLevel> {
    let config = UniverseConfig::walk_scale(); // (visual/canonical selected by boot, mirrors moving_children_for)
    generate_system_forest(seed_universe, &config).iter()
        .filter(|b| b.parent == Some(hosted))
        .map(|b| level_of(b))   // monomorphic RealmId→RealmLevel from the GeneratedBody's known kind+seed
        .collect()
}
```

`level_of` reads the `GeneratedBody`'s own kind+seed (the generator has them un-lossily) — a monomorphic helper, NOT a lossy `RealmId` inversion. (Mirror the boot-scale selection `moving_children_for` uses.)

---

## 3. `evaluate_realm_aoi` + `aoi_decide` + EMIT + GATE/ORDER + PREDICTIVE

### 3a. `StubConfig` additions (M-3 build precondition + F7 horizon) — `stub.rs:63`

`StubConfig.realm: RealmId` (`:64`) is LOSSY (`RealmCoord::lowered` doc, realm_coord.rs:49). The loop CANNOT name children or key `child.path()` without full lineage. **These are the FIRST lines the loop needs — build preconditions, not flags:**

```rust
    /// The shard's OWN full lifecycle coord (RLM Step 2). Seed-derived at boot from the hosted realm's
    /// lineage (the generator knows it). REQUIRED: the AoI loop names children via `own_coord.child(level)`
    /// and keys AoiMembership + the emitted demand on `child.path()` — unbuildable from the lossy
    /// `realm: RealmId` (M-3). Default = the single-realm root coord for `realm` (byte-identity: inert AoI
    /// never reads it).
    pub own_coord: RealmCoord,
    /// F7 predictive horizon (ticks): the AoI loop projects `pos + vel·(boot_ticks_p99·tick_dt_s)` so a
    /// fast occupant demands spin-up before it arrives (boot latency masked). A config field, never a
    /// literal. `0` ⇒ no predictive term (default; byte-identity).
    pub boot_ticks_p99: u32,
```

**L3 (co-hosting hygiene):** `held_realms` is `{realm}` in every live path (node-per-realm base; co-hosting KEPT-unused, D-44). `aoi_decide` builds ONE `own_coord`. Add `debug_assert!(config.held_realms.len() == 1, "AoI loop assumes node-per-realm; co-hosting (D-44) needs per-held-realm coords")` at the loop head + the doc note, so the single-`own_coord` assumption is LOUD, not latent. (Not a Step-3 concern — Step 3 dedups on `child.path()` regardless.)

### 3b. `has_synced` seam (D-Finding-1 / M-1 / L2) — `runtime.rs` + `follower.rs`

`ClockSample` (`runtime.rs:152`, derives `Default,PartialEq,Eq`) has no synced bit and defaults `universe_tick=0` (a VALID synced tick — a `!=0` test is WRONG). `observe_clock_syncs` (`follower.rs:37`) writes `sample.universe_tick`/`epoch` ONLY inside the `for msg` loop (`:73-74`), on decodable-ClockSync ticks.

Add `pub synced: bool` to `ClockSample` (Eq-compatible). **L2:** set it `= state.clock.is_some()` **OUTSIDE the loop, at fn exit** — not "alongside universe_tick" (which fires only on delivery ticks → a sync-flap the run-condition inherits). Once `state.clock` is `Some` it never resets, so every subsequent tick reads `synced = true`:

```rust
// follower.rs, after the `for msg in &inbox.0` loop (outside it):
sample.synced = state.clock.is_some();
```

Run-condition in `vd-sim`:
```rust
fn has_synced(clock: Res<ClockSample>) -> bool { clock.synced }
```

**M-1 retrofit (behavior-CHANGING by design, not identity-preserving):** gate `evaluate_realm_aoi`, AND retrofit `emit_realm_frames` (`stub.rs:4062`) + `evaluate_realm_boundaries` (`:2670`) with `.run_if(has_synced)` — both currently gate ONLY on authority (`:2685`/`:4072`), the D-Finding-1 hole (a fresh shard authors at tick 0 pre-sync). This is a real semantics change to shipped systems; both early-return on empty snaps/regions so at walk-scale it is inert, but at visual-scale the first-sync boundary is exactly what must be gated. Because `ClockSample` is a persistent resource and the two systems already sit AFTER `observe_clock_syncs` in the tick (it registers on the shared schedule), reading `synced` is order-correct.

### 3c. `evaluate_realm_aoi` — the branchless system shim (sibling of `evaluate_realm_boundaries`)

Third chained group (group B is at Bevy's 8-`.chain()` arity limit, `stub.rs:1096`), `.after(emit_realm_frames)` (the detector group's tail) + `.run_if(has_synced)`:

```rust
// in register_stub_shard, after the group-B add_systems (:1122):
schedule.add_systems(
    (evaluate_realm_aoi,).chain().after(emit_realm_frames).run_if(has_synced),
);
```

```rust
fn evaluate_realm_aoi(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    dots: Res<Dots>,
    owned_transients: Res<OwnedTransients>,   // occupants = dots ∪ held transients (mirror :2694)
    mut membership: ResMut<AoiMembership>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(realm_fence) = authority.0 else { return; };   // authority gate (verbatim :2685)
    if regions.is_empty() { return; }                        // inert (verbatim :2689)
    debug_assert!(config.held_realms.len() == 1,
        "AoI loop assumes node-per-realm; co-hosting (D-44) needs per-held-realm coords"); // L3
    aoi_decide(&config, &clock, &regions, &dots, &owned_transients,
               realm_fence, &mut membership.0, &mut outbox);   // delegate — ALL branching downstream
}
```

Straight-line iterate→delegate (only the two guard `else`s, both mirroring the covered detector).

### 3d. `aoi_decide` — the monomorphic helper (ALL branches here, HR5)

```rust
#[allow(clippy::too_many_arguments)]
fn aoi_decide(
    config: &StubConfig, clock: &ClockSample, regions: &RealmRegions,
    dots: &Dots, owned: &OwnedTransients, realm_fence: Fence,
    membership: &mut BTreeMap<RealmPath, AoiState>, outbox: &mut OutboundBox,
) {
    let own_coord = &config.own_coord;
    let tick = clock.universe_tick;
    let tick_hz = 1.0 / config.tick_dt_s;
    let horizon_s = config.boot_ticks_p99 as f64 * config.tick_dt_s;   // F7

    // Occupants: owned durable dots ∪ held transients (mirror :2694). Frame-local pos + vel. Reduced to
    // a scalar min BEFORE any emit, so occupant iteration ORDER cannot leak into the demand set (H-2).
    let occupants: Vec<(DVec3, DVec3)> = dots.0.values()
        .filter(|d| d.authority.simulates())
        .map(|d| (d.pose.pos.offset(), d.pose.vel))
        .chain(owned.0.values().filter(|t| t.status.is_held())
            .map(|t| (t.pose.pos.offset(), t.pose.vel)))
        .collect();

    // Zero-occupant self-report: the CHILD-shard reports ITS OWN realm holds nobody (child = own_coord).
    if occupants.is_empty() {
        push_demand(outbox, config.orchestrator, own_coord.clone(), realm_fence, DemandVerb::Empty, tick);
        return;
    }

    // The UNIFIED child list (movers + static), stable Vec order (seed-derived) — no HashSet reorder (H-2).
    let placements = regions.child_placements(tick_hz, tick);   // Vec<(&RealmRegion, StampedPose)>
    let live_paths = &mut BTreeSet::<RealmPath>::new();
    for (region, pose) in &placements {
        let level = region_level(region);                        // monomorphic RealmId→RealmLevel (seed-known)
        let child_coord = own_coord.child(level);
        let key = child_coord.path().clone();
        live_paths.insert(key.clone());
        let child_pos = pose.pos.offset();                       // H-1: frame-local, own frame (== placements' frame)

        // Reduce over occupants to a scalar min (live + F7 predictive), fixed fold order.
        let min_dist_eff = aoi_min_dist(&occupants, child_pos, horizon_s);

        let state = membership.get(&key).copied().unwrap_or_default();
        let now_in = region.aoi().in_range(state.was_in, min_dist_eff);
        let (verb, next) = aoi_transition(state, now_in, region.aoi().grace_ticks()); // ALL hysteresis here
        if let Some(v) = verb {
            debug_assert!(v != DemandVerb::TearDown,
                "Step 2 never emits parent TearDown — Step-3 closure is the sole kill authority (M-1)");
            push_demand(outbox, config.orchestrator, child_coord, realm_fence, v, tick);
        }
        match next { Some(s) => { membership.insert(key, s); } None => { membership.remove(&key); } }
    }
    // Evict any child-path no longer in the roster (no leak) — DRY, keyed by RealmPath.
    retain_live(membership, live_paths);
}

/// The min distance from any occupant to `child_pos`, taking the LESSER of the live distance and the
/// F7 predictive distance (`occ_pos + occ_vel·horizon_s`). Fixed fold order (order-independent for the
/// scalar min value, but folded deterministically). Monomorphic (HR5).
fn aoi_min_dist(occupants: &[(DVec3, DVec3)], child_pos: DVec3, horizon_s: f64) -> f64 {
    let mut min = f64::MAX;
    for (pos, vel) in occupants {
        let live = (child_pos - *pos).length();
        let pred = (child_pos - (*pos + *vel * horizon_s)).length();   // F7: static occ ⇒ vel=0 ⇒ pred==live
        min = min.min(live).min(pred);
    }
    min
}

/// The hysteresis + grace state machine (ALL of it monomorphic, each arm a covered region). Returns
/// `(verb_to_emit, next_state)`. Step 2 NEVER returns TearDown — a child leaving range drops its key and
/// emits nothing; Step-3's closure tears it down (M-1, REVISION 1 R2 supersedes §2.2 pseudocode).
fn aoi_transition(state: AoiState, now_in: bool, grace_ticks: u32)
    -> (Option<DemandVerb>, Option<AoiState>) {
    match (state.was_in, now_in) {
        (false, true)  => (Some(DemandVerb::SpinUp),
                           Some(AoiState { was_in: true, grace_remaining: grace_ticks })),
        (true, true)   => (Some(DemandVerb::KeepAlive),
                           Some(AoiState { was_in: true, grace_remaining: grace_ticks })),
        (true, false)  => if state.grace_remaining > 0 {
                              (Some(DemandVerb::KeepAlive),
                               Some(AoiState { was_in: true, grace_remaining: state.grace_remaining - 1 }))
                          } else {
                              (None, None)   // drop the key; NO TearDown
                          },
        (false, false) => (None, None),
    }
}
```

`region_level(region)` is the monomorphic `RealmId → RealmLevel` for a hosted region — sourced from the seed (paired with `direct_child_levels`; §2d supplies the roster so the pose+level join is exact). In `aoi_decide` the cleanest shape is to fetch `direct_child_levels(config seed, config.realm)` once and zip by realm with `placements` (both derive from the same seed forest in the same order, L1) — so `region_level` is a lookup into that roster, not a lossy inversion.

### 3e. The emit seam — `push_demand`

Rides the verbatim `CrossingRequest` egress path (`push_flow`, `runtime.rs:63`).

```rust
fn push_demand(outbox: &mut OutboundBox, orch: NodeId, child: RealmCoord, fence: Fence,
               verb: DemandVerb, tick: UniverseTick) {
    outbox.push_flow(
        orch,                    // StubConfig.orchestrator (:88) — same dest as every egress
        MsgClass::Saga,          // Reliable (io/mod.rs:123) — SideEffecting MUST ride Reliable (guard :106)
        &InterShardFlow::RealmDemand(RealmDemand { child, parent_fence: fence, verb, universe_tick: tick }),
    );
}
```

- `push_flow` (3-arg) defaults `Durability::Ephemeral` — CORRECT: `RealmDemand` is `ReDriven` (`intershard.rs:512`), NOT `ProducerLessReliable`, so the durability guard (`runtime.rs:92`) passes. Do NOT use `push_flow_durable`/`Retained`.
- `MsgClass::Saga` (Reliable) satisfies the side-effecting guard (`runtime.rs:106`). NEVER a `Snapshot`/Unreliable class.
- `parent_fence = realm_fence = authority.0`: for SpinUp/KeepAlive it proves the PARENT's authority over the child; for Empty it is the CHILD-shard's authority over its own emptiness. **M-3 note:** the wire field is named `parent_fence` but is the EMITTER's authority fence (parent for SpinUp/KeepAlive, child-self for Empty) — annotate at the emit site so Step-3 keys the Empty idempotency on the child's fence (`IdempotencyKey::FencedKey{fence: d.parent_fence}`, `intershard.rs:442`).

---

## 4. TEST LIST — 100% region+branch on ONLY Step-2 code

Co-locate in the owning crate (HR5). Prefix every test `aoi_`/`interest_`/`child_placements_`/`evaluate_realm_aoi_`/`from_regions_json_`. Discipline in EVERY test: `assert_eq!`/`expect_err` (never `assert!(matches!)`); NO expression in an `assert!` message (cold-path uncovered region); split `assert!(a && b)`.

**Run JUST Step 2 (no full gate):**
```
cargo test -p vd-core   aoi_ interest_ from_regions_json_
cargo test -p vd-sim    aoi_ child_placements_ evaluate_realm_aoi_ authored_realm_snaps_
cargo test -p vd-node   observe_clock_syncs_sets_synced
```
(Cargo treats each space-separated token as a substring filter across the crate's tests.)

### `vd-core` geometry.rs — `AoiConfig`
- `aoi_config_for_velocity_safe_ok` — valid factors build; `assert_eq!` both resolved edges + `assert!(spin < tear)` (split). Covers `Ok` arm + `debug_assert`.
- `aoi_config_rejects_inverted_edges` — `expect_err(InvalidEdges)` for `spin_factor >= tear_factor`, zero velocity. Covers the reject arm.
- `aoi_config_rejects_zero_spin_up` — `expect_err` for `spin_factor = 0`. Covers `spin_up > 0.0` false branch.
- `aoi_config_velocity_widens_the_gap` — non-zero `v_rel` makes `tear_down == spin_up + need` exceed `base·tear_factor`; `assert_eq!` the widened edge (twin of `containment_band_velocity_widens_the_outset` `:1057`).
- `aoi_config_inert_is_zero` — `inert()` has `spin_up_r_m() == 0.0`; `in_range(false, x) == false` AND `in_range(true, x) == false` (two `assert_eq!`). Covers `inert()` + `in_range` with a 0 band.
- `aoi_in_range_acquire_and_hold` — live band: `in_range(false, spin−ε)==true`, `in_range(false, spin+ε)==false`, `in_range(true, tear−ε)==true`, `in_range(true, tear+ε)==false` (four `assert_eq!`). Covers all three compares + both `was_in` arms.

### `vd-core` celestial.rs — `v_peri`
- `aoi_v_peri_periapsis_speed` — circular (`ecc=0`) ⇒ `v_peri == sqrt(mu/a)`; eccentric ⇒ `> circular` (two `assert!`, split). Covers `v_peri`.

### `vd-core` worldgen.rs — `InterestConfig` + `to_regions` + `children_within` + `direct_child_levels`
- `interest_config_build_inert_at_zero_factor` — `walk_scale().interest.build(extent, v)` == `inert()` (`assert_eq!`). Covers the inert arm.
- `interest_config_build_live` — `visual_scale().interest.build(extent, v)` has `spin_up == extent·factor` (`assert_eq!`). Covers the live arm.
- `to_regions_stamps_per_realm_aoi` — walk forest: every region's `aoi` inert (`assert_eq!`). Visual forest: a Planet's `aoi.spin_up_r_m() == planet_extent·factor` (`assert_eq!`), bigger than an Area's. Covers the aoi threading + the `orbital_of`/`v_child` fold.
- `children_within_filters_by_radius` — bounded `(RealmLevel, DVec3)` slice + occupant; `assert_eq!` the returned child-coord SET for a radius including some/excluding others (both `d <= radius` arms). `assert_eq!` on an included child's `parent.child(level).path()`.
- `aoi_within_boundary` — direct unit test of `aoi_within`: `Some(0.0)` at coincident, `Some(r)` at exactly `radius`, `None` just past. Covers the predicate's both arms.
- `direct_child_levels_from_seed` — `assert_eq!` the roster for a known parent against the seed forest (L1: derived from `generate_system_forest`, `b.parent == Some(hosted)`).
- `from_regions_json_defaults_missing_aoi_to_inert` (C-1) — a legacy JSON array WITHOUT `aoi` parses via `serde_json::from_str::<Vec<RealmRegion>>` and every region's `aoi == inert()` (`assert_eq!`). Plus `from_regions_json_roundtrips_with_aoi` — a freshly-serialized forest round-trips (`assert_eq!`).
- **byte-identity gate** `walk_regions_behaviour_unchanged_with_aoi` — the existing walk region test still passes with `aoi: inert` (the field does not change `container`/observer decisions).

### `vd-node` follower.rs — has_synced source
- `observe_clock_syncs_sets_synced_after_first_sync` — pre-sync `ClockSample.synced == false`; after one `ClockSync`, `synced == true`; on a subsequent NO-sync tick, `synced` STAYS `true` (L2: set outside the loop, `is_some()`). Three `assert_eq!`.
- `observe_clock_syncs_undecodable_leaves_unsynced` — only undecodable frames ⇒ `synced == false` (`state.clock` still `None`). Covers the "inside decode-success path" contract.

### `vd-sim` stub.rs — `child_placements`, `AoiMembership`, `evaluate_realm_aoi`
- `child_placements_folds_movers_and_static` — one mover + one static child; `assert_eq!` the two-row list (mover pose from `orbital_state`, static from `center`). Covers `place_child` both arms + `is_direct_child`.
- `child_placements_excludes_ancestors_and_self` — an ancestor/self region is NOT returned (`is_direct_child` false arm). `assert_eq!` the filtered set.
- `authored_realm_snaps_filters_to_movers_only` — same rig; `authored_realm_snaps` yields ONLY the mover row. `authored_realm_snaps_walk_empty` — empty `moving` ⇒ empty snaps (observer feed byte-identity).
- `evaluate_realm_aoi_inert_without_authority` — no `RealmAuthority.0` ⇒ `outbox.0.is_empty()`. Covers the authority guard `else`.
- `evaluate_realm_aoi_inert_empty_regions` — authority set, empty regions ⇒ empty. Covers the `is_empty` guard.
- `evaluate_realm_aoi_unsynced_authors_nothing` (D-Finding-1) — `ClockSample.synced = false` ⇒ the system does not run ⇒ empty. Covers `has_synced`.
- `evaluate_realm_boundaries_unsynced_authors_nothing` + `emit_realm_frames_unsynced_ships_nothing` (M-1 twins) — PROVE the retrofit on all three gated systems.
- `evaluate_realm_aoi_spinup_then_keepalive` (AOI-2) — occupant approaches: tick 1 emits exactly one `SpinUp` keyed on `child.path()`; tick 2 emits `KeepAlive`. `assert_eq!` the full demand (`child`, `parent_fence`, `verb`, `universe_tick`). Covers `(false,true)`+`(true,true)`.
- `evaluate_realm_aoi_grace_then_drop` — occupant leaves: while `grace_remaining > 0` emits `KeepAlive` (`(true,false)` grace arm); after grace expires, the key is dropped, no demand (`(true,false)` else + `(false,false)`). **M-1 assertion:** `assert!(!demands.iter().any(|d| d.verb == DemandVerb::TearDown))` (NO parent TearDown ever — locks the REVISION-1 contract against a §2.2-pseudocode regression). Covers the grace countdown both branches.
- `evaluate_realm_aoi_empty_self_report` — zero occupants ⇒ exactly one `RealmDemand { child: own_coord, verb: Empty }` (`assert_eq!`). Covers the empty-occupant branch.
- `evaluate_realm_aoi_predictive_spinup` (F7) — occupant OUTSIDE `spin_up_r_m` but `pos + vel·horizon` inside ⇒ `SpinUp`; a STATIC occupant (`vel=0`) at the same pos ⇒ NO demand. `assert_eq!` both. Covers `aoi_min_dist`'s predictive-hit + predictive-miss.
- `evaluate_realm_aoi_demand_order_is_stable` (H-2) — two occupants at different distances to two children: the emitted `Vec<RealmDemand>` is IDENTICAL (element order included) across two runs AND across PERMUTED `Dots`/`OwnedTransients` insertion order. (Stronger than AOI-2's same-input-twice — catches a hidden reorder.)
- `aoi_membership_evicts_absent_children` — a child leaving the roster has its key dropped by `retain_live` (no leak). Covers the widened `retain_live<K,V>` over `RealmPath`.

### HR4 G-IDENTICAL (`vd-sim` or `vd-tests`)
- **AT-1 / G-IDENTICAL (≥2 realm kinds)** `occupant_approaches_child_demands_it(host_profile, child_level)` — ONE fixture fn, `assert_eq!` on the emitted `RealmDemand` SET, invoked for BOTH: `System deciding about a Planet child` (host = System shard, child = Planet) AND `Planet deciding about an Area child` (host = Planet shard, child = Area). Differ ONLY in seed radii + coords — SAME code path, SAME assertion. The HR4 proof.
- **AOI-1 (radius nesting)** `aoi_radius_scales_with_extent` — `System.aoi().spin_up_r_m() > Planet.aoi().spin_up_r_m()` AND `Planet > Area` (three `assert!`, split — HR3 proportionality).
- **AT-1 orbital variant (M2)** `occupant_approaches_moving_child_demands_it` — the child is a MOVER (`OrbitalElements`); pose from `orbital_state`, band widened by `v_child = v_peri`; the approach still yields deterministic `SpinUp` and the mover's motion cannot make it flap. Proves the M2 velocity-safe path.

---

## 5. REUSE MAP (verbatim, no reinvention)

| Reused | Location | Role |
|---|---|---|
| `InterShardFlow::RealmDemand` + `DemandVerb` + classification | wire/intershard.rs:257,441,512,523,540 | FROZEN — emit only |
| `RealmCoord::child/parent/path` + `RealmPath::from_levels/levels` | core/realm_coord.rs:83,69,45; realm_path.rs:95,100 | name children; `path()` dedup key |
| `Boundary::finite_extent()` | core/geometry.rs:277 | AoI radius base — the ONE sanctioned kind match |
| `ContainmentBand::for_containment_velocity_safe` | core/geometry.rs:715 | the fallible ctor template |
| `ContainmentBand::member` | core/geometry.rs:759 | `AoiConfig::in_range` shape |
| `K_SAFETY`, `BandError::InvalidEdges` | core/geometry.rs:31,64 | width invariant + LOUD reject |
| `BandConfig` + `::build()` | core/worldgen.rs:639,648 | `InterestConfig` template |
| `to_regions` band-set pattern | core/worldgen.rs:173 | per-body `aoi` stamp |
| `moving_children`/`orbital_of`/`orbital_state`/`OrbitalElements`/`mu` | core/worldgen.rs:200,213; celestial.rs:220,157 | mover fold + `v_peri`; `direct_child_levels` roster (`b.parent==Some`) |
| `realm_neighbourhood_for` (REFERENCE only — the O(all realms) anti-pattern) | core/worldgen.rs:449 | documents what `children_within` avoids (C1) |
| `evaluate_realm_boundaries` structure + `CrossingCtx` | sim/stub.rs:2670,2762 | the read-only sibling shape (guards, iterate→delegate) |
| `evaluate_one_subject` / `region_signed_distance` frame algebra | sim/stub.rs:2828,2860 | the monomorphic-helper canonical shape; the SAME `own`-frame distance (H-1) |
| `retain_live<V>` | sim/stub.rs:2785 | widen over `K` for AoiMembership |
| `RegionMembership`/`ContainmentProgress` | sim/stub.rs:466,490 | `AoiMembership`/`AoiState` twin (child-path keyed) |
| `authored_realm_snaps`/`frame_context`/`secs_since_epoch`/`place`-in-`own` | sim/stub.rs:595,571 | `child_placements` fold + observer filter |
| `OutboundBox::push_flow` + `StubConfig.orchestrator` + `MsgClass::Saga` | sim/runtime.rs:63; stub.rs:88; io/mod.rs:123 | the emit seam (verbatim) |
| `RealmAuthority.0` gate + `simulates()`/`is_held()` occupant filters | sim/stub.rs:236,2685,2719,2741 | `parent_fence` source + authority guard + occupant set |
| `StampedPose.pos.offset()` / `.vel` | core/pose.rs:182,218 | frame-local distance + F7 predictive term |
| `observe_clock_syncs`/`FollowerState.clock` | node/follower.rs:37,18 | has_synced source (via new `ClockSample.synced`) |
| `RealmRegion` serde + client `from_regions_json` + bins `write_regions_json` | core/geometry.rs:774; client/realm_scene.rs:213,282; bins/lib.rs:1460 | the `#[serde(default)]` on-disk contract (C-1) |

**New (Step 2):** `AoiConfig`+`for_velocity_safe`/`inert`/`in_range`/`width_safe_for`/accessors; `OrbitalElements::v_peri`; `RealmRegion.aoi` (`#[serde(default)]`); `InterestConfig`+`build` (with `occupant_v_max_mps`/`tick_dt_s`); `RealmRegions::child_placements` + `is_direct_child`/`place_child`; `worldgen::children_within`+`aoi_within`+`direct_child_levels`+`level_of`; `AoiMembership`/`AoiState`; `evaluate_realm_aoi`+`aoi_decide`+`aoi_min_dist`+`aoi_transition`+`push_demand`+`region_level`; `ClockSample.synced`+`has_synced`; `StubConfig.own_coord`+`boot_ticks_p99`; the boot cross-check `debug_assert!`s (M-2/L3).

---

## 6. HR1–6 + PERF/DETERMINISM SIGN-OFF

- **HR1 sealed shards:** only inter-shard bytes are the FROZEN `RealmDemand` arm; no new wire type; `SideEffecting{FencedKey}` on `Saga` (Reliable). No shard reads another's World. `RealmRegion.aoi` is a boot-local resource + a `regions.json` DEV-FIXTURE field (`#[serde(default)]`), NOT an `InterShardFlow` arm — no frozen-wire violation. ✓
- **HR2:** N/A (lifecycle, not entity transfer); the loop is kind-agnostic — ONE machinery demands any realm kind. ✓
- **HR3 no match-on-kind:** AoI radius = `uniform_factor × Boundary::finite_extent()`; the ONLY kind match is inside `finite_extent` (geometry-owned, `:277`). `aoi_decide` is kind-blind. ONE `evaluate_realm_aoi`/`AoiConfig`/`InterestConfig`. ✓
- **HR4 features once, run anywhere:** `evaluate_realm_aoi` on every ShardProfile's schedule; AT-1 passes byte-identically System→Planet AND Planet→Area. H-1 fix (same `own`-frame distance as the detector) guarantees the algebra is identical across kinds. ✓
- **HR5 100% region+branch:** `evaluate_realm_aoi`/`children_within` are branchless shims; ALL branching in monomorphic `aoi_decide`/`aoi_transition`/`aoi_min_dist`/`aoi_within`/`place_child`/`is_direct_child`/`in_range`/`InterestConfig::build`. `AoiConfig` generic-free. Tests use `assert_eq!`/`expect_err`, split conjunctions, no assert-message expressions. ✓
- **DETERMINISM:** `AoiMembership` is `BTreeMap<RealmPath,_>` (no default hasher); occupants reduced to a scalar min BEFORE emit (H-2 order-safety); `child_placements` folds a stable `Vec` (no HashSet reorder); the demand SET is a pure `f(occupants, tick, seed, AoiMembership-history)`; `has_synced` guarantees a fresh shard authors NOTHING pre-`ClockSync`; celestial child poses are closed-form `orbital_state(f(seed,tick))`; walk-scale factor-0 ⇒ zero demands ⇒ byte-identity. H-2 permuted-order test PROVES it. ✓
- **PERF:** O(live direct children × occupants) via `children_within` over the bounded `child_placements` list — NOT the O(all realms) `realm_neighbourhood_for` scan (C1); capped by `MAX_REGIONS=64`. `push_flow` allocation-lean (postcard into `Arc`-`Bytes`). Anti-thrash = geometric (`spin_up<tear_down`, velocity-widened) + grace + (Step-3) cooldown. P6/D-9 spatial-index swap documented at `children_within`. ✓
- **No magic numbers:** `InterestConfig` factors, `grace_ticks`, `k_safety_extra`, `occupant_v_max_mps`, `tick_dt_s`, `boot_ticks_p99` are config fields; the M-2 boot `debug_assert!` single-sources `tick_dt_s`/occupant-speed against `StubConfig`. ✓
- **Fence discipline:** every demand carries the emitter's `authority.0` (parent for SpinUp/KeepAlive, child-self for Empty — annotated). ✓

---

## 7. HOW STEP 3 SITS ON THIS UNCHANGED

Step 3 = the orchestrator holding DESIRED = ancestor-closure of (realms-with-occupants ∪ realms-in-any-AoI), reconciling ACTUAL→DESIRED keyed on `child.path()`. Verified each dependency:
- **Dedup key:** every demand carries `child: RealmCoord`; `AoiMembership` keyed by `child.path()` (globally unique, Step 1). Step 3 dedups directly — no key rework.
- **Level-triggered re-assertion:** `RealmDemand` is `ReDriven` + `push_flow` default Ephemeral — the parent re-emits SpinUp/KeepAlive EVERY tick in range; a dropped verb self-heals next tick. Step 3's reconciler is edge-free.
- **NO parent TearDown (M-1):** Step 2 emits none — a child leaving AoI simply STOPS being demanded (key drops after grace); Step 3's closure is the SOLE kill authority (tears down when the child falls out of the desired set past grace+cooldown). REVISION 1 R2 supersedes the §2.2 pseudocode; the `TearDown` verb stays frozen/unused, guarded by the `evaluate_realm_aoi_grace_then_drop` no-TearDown assertion. **EDGE 1 (empty-but-in-AoI stays alive):** the child emits `Empty{self}` while the parent still emits `KeepAlive{child}` (gated on `in_range`, NOT child-occupancy — a sealed parent cannot see inside) ⇒ DESIRED keeps it via the AoI-half ⇒ correct. **EDGE 2 (load into queued-teardown):** pure Step-3 machinery; Step 2 contributes nothing and correctly need not.
- **Empty self-report:** the CHILD self-reports `RealmDemand{child: own_coord, verb: Empty}` (child = self); Step 3 reads it as the occupancy signal (the child is authority on its own emptiness). The `parent_fence` field carries the CHILD's fence for Empty — annotated so Step-3 keys idempotency correctly.
- **No wire/loop/config change forced:** Step 3 adds ONLY an orchestrator inbound arm decoding `RealmDemand` + a reconcile loop over DESIRED calling the existing `RealmSpawner` port (Step 1). It reads Step-2's demands as-is. C-1/H-1/M-3 are the fixes that PREVENT a Step-2 rewrite during Step 3 (a scene-file break, a frame mismatch, an unbuildable dedup key respectively) — all resolved here.

---

## 8. ORDERED IMPLEMENTATION CHECKLIST (top-to-bottom)

1. **core/celestial.rs:** add `OrbitalElements::v_peri` (§1b) + `aoi_v_peri_periapsis_speed` test.
2. **core/geometry.rs:** add `AoiConfig` + `for_velocity_safe`/`inert`/`in_range`/`width_safe_for`/accessors (§1a) + its 6 tests. Add `aoi: AoiConfig` field to `RealmRegion` with `#[serde(default = "AoiConfig::inert")]` (§1c). `RealmRegion` stays `Copy`.
3. **core/worldgen.rs:** add `InterestConfig` + `build` (§1d) to `UniverseConfig`; set inert in `walk_scale`/`canonical`, live in `visual_scale`; thread `aoi` through `to_regions` (per-body `finite_extent` + `v_child` via `orbital_of`/`v_peri`, using the config `interest`). Add `children_within`+`aoi_within` (§2c) + `direct_child_levels`+`level_of` (§2d). Add the interest/children/direct-child/`from_regions_json` tests (§4) — INCLUDING the C-1 legacy-parse test.
4. **sim/runtime.rs:** add `pub synced: bool` to `ClockSample` (§3b). Fix `boxes_default_empty` (already `..default()`-friendly).
5. **node/follower.rs:** set `sample.synced = state.clock.is_some();` OUTSIDE the `for msg` loop (§3b/L2) + the two follower tests.
6. **sim/stub.rs — types:** add `AoiMembership`/`AoiState` (§1e); widen `retain_live<K,V>` (§1e). Add `StubConfig.own_coord: RealmCoord` + `boot_ticks_p99: u32` (§3a) — update EVERY `StubConfig { .. }` construction site (grep `StubConfig {` — the `:4149` rig + all others) to supply them (own_coord = the single-realm root coord for `realm`, boot_ticks_p99 = 0 default). Register the `AoiMembership` resource in `register_stub_shard`.
7. **sim/stub.rs — placements:** add `RealmRegions::child_placements` + `is_direct_child` + `place_child` (§2a); refactor `authored_realm_snaps` to the movers-only filter (§2b). Verify the `authored_realm_snaps` tests (`:7816`/`:7847`) + `frame_context` tests still pass (byte-identity).
8. **sim/stub.rs — the loop:** add `evaluate_realm_aoi` + `aoi_decide` + `aoi_min_dist` + `aoi_transition` + `push_demand` + `region_level` (§3c/3d/3e). Wire the boot `debug_assert!` cross-checks in `register_stub_shard` (M-2 tick_dt_s/occupant-speed vs `InterestConfig`; L3 `held_realms.len()==1`).
9. **sim/stub.rs — schedule:** add the third chained group `(evaluate_realm_aoi,).chain().after(emit_realm_frames).run_if(has_synced)` (§3c). Add `.run_if(has_synced)` to `emit_realm_frames` + `evaluate_realm_boundaries` (M-1 retrofit).
10. **sim/stub.rs — rig sweep (M-1/L2):** grep every `ClockSample {` literal + `insert_resource(ClockSample` (the ~24 stub sites + `follower.rs` + `runtime.rs:boxes_default_empty`) and set `synced: true` in each rig that expects the newly-gated `evaluate_realm_boundaries`/`emit_realm_frames`/`evaluate_realm_aoi` to run — else they silently no-op (false-green). Enumerate as a named sub-step, NOT "mechanical."
11. **sim/stub.rs — tests:** add all `evaluate_realm_aoi_*`, `child_placements_*`, `aoi_membership_*`, the M-1 twin unsynced-authors-nothing tests, and the H-2 `demand_order_is_stable` permuted test (§4). Add the HR4 AT-1 fixture (System→Planet + Planet→Area) + AOI-1 + the M2 orbital variant.
12. **Verify Step-2-only:** run the §4 cargo filters per crate (no full gate) until 100% region+branch on the new code, THEN `just gate`.

**Every load-bearing seam Read at the cited line.** The two behavior-changing retrofits (M-1 has_synced on the two shipped systems) and the four config/seam additions (own_coord, boot_ticks_p99, InterestConfig widening inputs, ClockSample.synced) are the only places Step 2 touches beyond the primary files; none alters the frozen wire or forces a Step-3 rework. C-1/H-1/H-2/M-1/M-2/M-3/L1/L2/L3 are all folded in above.
