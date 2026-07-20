# Seed Universe Generator — design (D-45(a), pre-P4)

Status: **Slices 0–2 LANDED**; Slices 3–4 + step-2 DESIGNED + adversarially vetted, not yet
implemented (Jul 2026). Slice 0 (seed-tree RNG) `07a2c1d`; Slice 1 (3D Kepler ephemeris math)
`3ae580f`; Slice 2 (taxonomy as data) on this branch — all HR5 100%, full workspace gate green. This is the plan of
record for turning `worldgen::realm_regions_for` from a hardcoded 7-region demo forest into a real,
seed-derived, generic, configurable universe generator — the foundation the later steps (orbits,
dynamic shard spawn, server-authoritative LOD, warp) all plug into.

Built on the vetted survey + design-and-vet workflow (transcript: session subagents/workflows
`wf_43f2aefe-459`). Landed AFTER the node-per-realm re-home (`a42826f`, see
`[[project_node_per_realm_rehome]]`).

## Locked decisions (user, Jul 2026)

1. **Extend the hand-rolled `celestial.rs`, NOT brahe.** brahe is a declared-but-unused workspace dep;
   the codebase deliberately hand-rolls Category-A analytic celestial math for cross-host
   bit-determinism (an iterative solver that isn't bit-identical across OSes would split-brain the
   *containment/ownership* decision, which is cross-host). `celestial.rs` already has the Kepler
   anomaly chain (`mean_anomaly_at` → `solve_kepler` → `eccentric_to_true_anomaly`, `planet_soi`) —
   dead code today; the generator is its first real consumer.
2. **Dedicated `RealmId::Universe` / `RealmId::Galaxy` arms — no aliasing.** Real from day one; today
   they are placeholder `System(0)`/`System(1)` stand-ins. This is an additive frozen-wire change and
   gets its OWN slice (Slice 4), never co-mingled with the generator.
3. **Real universe from day one** — full 3D Kepler (all 6 orbital elements + inclination/RAAN/arg-of-
   periapsis), real astrophysical taxonomy (galaxy types, stellar spectral classes O–M via the IMF,
   planet types via core-accretion). NOT a toy. "Real from day one" is achieved as a **coherent
   sequence of real (non-stub) slices** — see Scope honesty below.
4. **Star map = observer-local, apparent-magnitude visibility.** See Map & travel model.

## The three pillars

### 1. Lazy, hierarchical local generation (the scale foundation)
A real universe cannot be materialized into one `Vec<RealmRegion>`. Each shard derives, from
`(universe_seed, its_realm)`, ONLY its own realm + its ancestor-chain-to-root + its (nearby) direct
children — never the whole tree.

- **`RealmPath`** — a realm's ordered lineage from the Universe root (`[Universe, Galaxy(g),
  System(s), Planet(p), Area(a)]`). This is a NEW **derivation layer**, NOT a wire type — `RealmId` /
  `FrameRef` / `StampedPose` frozen bytes are untouched (HR1). The path is the sole source of "who is
  my parent" (drop the last level) and "who are my children" (extend by one seed-derived level). It
  also closes the `Planet(p)`-can't-name-its-`System` gap (`realm_id()` from the last level,
  `parent_realm()` from the second-last — the exact Area→Planet parent provenance `frame_for_realm`
  already needs, read from the path not a region scan).
- **Per-realm deterministic RNG** — `realm_stream(universe_seed, path)` = a PURE HASH of
  `(universe_seed, lineage)` over the EXISTING `SplitMix64` (integer-only → bit-identical cross-host by
  construction; no wall clock). `child_seed(parent_seed, salt, index)` = a 3-round mix so a parent
  enumerates reproducible child seeds. Both get a known-vector pin like `rng.rs` already has.
- **Spatial children query** — `direct_children` for a low-fan-out realm (a system's few planets)
  enumerates all; for a high-fan-out realm (a galaxy of millions of systems) it is a **spatial query:
  children in a region around the observer** (`f(seed, realm, region)`). Hard-capped in `LazyGenTuning`
  so `guard_regions_nest`'s `MAX_REGIONS` (=64) never trips. The cap MUST be ≥ max legitimate
  direct-child fan-out (fail-loud) or a real re-home target silently vanishes from the neighbourhood.
- **The seam swap** — `realm_neighbourhood_for` / `_for_held` STOP calling the whole-Vec
  `realm_regions_for` and generate path-locally (own + ancestor chain + nearby children). Output stays
  a valid single-root forest the UNCHANGED `container` / `region_depth` / `guard_regions_nest` consume.
  ALSO rewrite the two other whole-Vec boot callers the vet caught: shard `own_frame` (shard.rs:92 —
  becomes `path_for_realm(realm).realize_region().frame`) and the client render single-source
  (`write_seed_regions`, lib.rs:1440 — becomes a BOUNDED renderable-neighbourhood generator, or a
  real-scale galaxy materializes on the client). `realm_regions_for` is retained ONLY as a
  `cfg(test/debug)` small-forest enumerator, off the boot path.

### 2. Real 3D Kepler ephemeris math (`celestial.rs`)
- **`OrbitalElements`** — all 6 elements (`sma`, `ecc`, `inclination`, `raan`, `arg_periapsis`,
  `mean_anomaly_epoch`) + `central_mass`; `mean_motion = sqrt(mu/a^3)`, `mu = G·central_mass`, `period`
  DERIVED. HR3: one function for planet-around-star, moon-around-planet, station-around-planet —
  parent is `central_mass` DATA, not a match.
- **`orbital_state(elements, time_s) -> OrbitalState{position, velocity}`** in the parent inertial
  frame (metres, m/s, fine tier): the existing anomaly chain → perifocal `r`,`v` (from angular
  momentum) → a 3-1-3 Z-X-Z rotation (`Rz(raan)·Rx(incl)·Rz(argp)` via `DQuat::from_axis_angle`). This
  is the MISSING elements→`DVec3` piece (the chain stops at true anomaly today).
- **Time is SECONDS, not ticks** — `secs_since_epoch(tick, tick_hz)`; tick rate is a per-shard knob, a
  global `TICKS_PER_SECOND` would be a magic number splitting a 10 Hz vs 50 Hz shard.
- **`solve_kepler_fixed`** — CONSTANT `KEPLER_FIXED_ITERS=32` Newton steps, no early-out, branchless
  denom floor → (a) bit-equal iteration count across builds, (b) HR5-clean (no data-dependent branch =
  no uncoverable false arm). Adaptive `solve_kepler` retained for adaptive/test callers.

### 3. Real taxonomy as DATA + the ONE config home
- **`UniverseConfig`** (in worldgen.rs, `Serialize`/`Deserialize`, seed-derived) — the ONE home
  replacing the ~15 placeholder consts (child-count ranges, IMF slope + mass bounds, orbital-spacing
  law, ecc/inclination scales, moon/station/area occurrence probs, band inset/outset, render-extent).
  Each a NAMED field with a doc-comment citing its astrophysical basis (no magic numbers). Grouped into
  sub-structs (`Scale`/`Galaxy`/`Stellar`/`Planet`/`Satellite`/`Band`) to avoid a god-struct.
  `canonical()` = the shipped real-scale tuning; `walk_scale()` = the preset reproducing today's EXACT
  P3 coordinates (the hard regression gate); `seed_derived(seed)` may perturb within documented bounds
  so different seeds → different galaxy morphologies.
- **Taxonomy as data** — `GalaxyType` (spiral/elliptical/irregular), `SpectralClass` (O–M from mass +
  luminosity + habitable zone), `PlanetType` (rocky/gas/ice/ocean from formation distance + mass),
  each an enum + `&'static` table with a total-over-`ALL` drift tripwire — the proven `entity_kind.rs`
  `KindDef` pattern. Distributions are real: IMF-like stellar mass (bounded Pareto), physically-motived
  orbital spacing, Rayleigh eccentricity/inclination — all **inverse-CDF closed forms** (no rejection
  loop → bit-reproducible + HR5-clean).
- **Capability link** — `ProfileKind` (in vd-core) tags each generated body; `capability::profile_for`
  (in vd-sim) is the ONE total core→sim map to the existing `profiles::{galaxy,system,planet,...}`
  `ShardProfile`s. New body kind = one data row, zero feature code (HR3/HR4). Dep direction legal.

## Map & travel model (refined with user, Jul 2026)

**The map and the simulation are separate; only the simulation is lazy.**

- **Star map = observer-local, apparent-magnitude visibility** — `f(seed, your-position)`: enumerate
  the systems whose apparent magnitude from your position clears a threshold
  (`m = M + 5·log10(distance/10pc) ≤ threshold`; reuses the generated luminosity). Real night-sky
  physics: luminous O/B beacons visible from thousands of ly, dim M dwarfs next door invisible.
  Inherently BOUNDED (only what's visible around you) — never a whole-galaxy sweep. **Client-derivable**
  (pure `f(seed, position)`, zero bandwidth — the client already derives the static universe geometry).
- **NO siblings in the containment neighbourhood** — a system's neighbourhood stays own + ancestors +
  direct children. The map does NOT come from the containment mechanism; it is the separate static read.
- **Travel = the generic containment re-home you already have.** Fly out of System A's SOI → deepest
  container becomes the **Galaxy** (A's parent) → re-home UP into the galaxy realm. Travel there;
  approach Destination B → cross B's SOI → deepest container becomes System B → re-home DOWN → B's shard
  streams. Same coordinate-driven containment that carries you Planet↔System now carries you
  System↔Galaxy↔System. The galaxy realm's re-home scan uses the **same spatial children query** as the
  map (systems near YOU — a tiny region for re-home, a larger one for the map).
- **Warp = hop through the visible set.** Point at a visible star → warp (fast galaxy traversal) →
  arrive at its SOI → the natural SOI-crossing re-home streams the system; its shard spawns on demand.
  Long journeys = a sequence of hops (the visible set updates as you move) — physically real, scales
  forever without knowing the whole galaxy.
- **Server-authoritative layer** = only the DYNAMIC overlay (which systems have activity/fleets/
  players) — the AoI/LOD layer (D-9). Static star field = client-derived; live activity = streamed.

## Coordinate tiers + frame-seam safety

- **Tiers (D-41):** galaxy/nav in the COARSE ly/AU tier; system-and-inward in the FINE mm tier. The
  tier keys off `FrameRef`; the fine↔coarse flip lands ON the SOI crossing — which IS the re-home
  event. Map/travel/tier boundaries all coincide on the SOI.
- **Step-1 stays IdentityFrames — PROVABLY does not touch the re-home.** The re-home reads frames at
  exactly two seam sites, both hardcoding `IdentityFrames`: input `region_signed_distance` (stub.rs:2750)
  and output `rebind_pose_to_dest` (frame.rs:191). Step-1 supplies ONLY a static `center` and NEVER a
  non-identity `FrameContext`, so the arithmetic (`p.pos.offset() - region.center.offset()`), the depth
  cache, and `root_realm` are byte-unchanged. **Hard constraint:** step-1 `center ==
  LatticePos::local(epoch_offset)` with **cell == ZERO**. Non-zero cells, the moving ephemeris, the D-41
  cross-cell re-base, mm↔ly conversion — all step-2/P4/P5/P10; step-1 touches none.
- Orbital MOTION does NOT live in `center`. `center` is the STATIC tick-0 epoch anchor; step-2's
  ephemeris re-derives the live origin every tick via `FramePlacement{origin, velocity, ...}` WITHOUT
  mutating `center`.

## Determinism (two-tier)

- **Tier 1 (RNG / seed-tree / generator):** SAFE, bit-equal cross-host by construction (`SplitMix64`
  integer-only). Needs known-vector pins. Step-1's proof = same-binary same-seed replay.
- **Tier 2 (Kepler float math):** `solve_kepler_fixed` kills the data-dependent-iteration hole (→
  HR5-clean). libm transcendental divergence across OSes is the ONE residual — closed only by a
  BUILD-AND-DIFF golden-table gate (SPIKE-6a) with exact-pinned `glam =0.30.x` and no FMA/target-cpu
  native (or a vendored pure-Rust libm). **This gate is a STEP-2 precondition, NOT step-1** — step-1
  evaluates Kepler ONCE at boot for a static epoch center; per-tick per-host re-evaluation is step-2.
  Authority is immune either way (source ships the transfer pose); the gate protects only the
  replicated-by-seed oracle consumers (containment membership, dest sanity-bound, client render).
  Note: `Cargo.toml` has `glam = "0.30"` (caret) — tighten to `=0.30.x` with the SPIKE.

## Slice sequence

- **Slice 0** ✅ DONE (`07a2c1d`) — determinism foundation: `child_seed` + `realm_stream` over
  `SplitMix64` + known-vector pins. Pure-additive, no behaviour change.
- **Slice 1** ✅ DONE (this branch) — real 3D Kepler math in `celestial.rs` (`OrbitalElements` +
  derived `mu`/`mean_motion`/`period`, `OrbitalState`, `orbital_state`, `solve_kepler_fixed`,
  `secs_since_epoch`, named `KEPLER_FIXED_ITERS`/`KEPLER_ECC_MAX`/`KEPLER_DENOM_FLOOR`); no worldgen
  consumer yet. Perifocal r + angular-momentum-form v → 3-1-3 `Rz(Ω)·Rx(i)·Rz(ω)` glam rotation on
  BOTH vectors; the fixed solver's denominator floor is BRANCHLESS (`fp.abs().max(FLOOR).copysign(fp)`)
  so no uncoverable arm. Tests: analytic goldens (M=0/π/2/π, inclination+RAAN handedness probes,
  full-period return) + invariants (|r|=a(1−e·cosE), vis-viva, specific energy, |r×v|) +
  fixed-vs-adaptive residual (incl. a negative raw M to light `normalize_angle`'s branch) + 2 proptests
  (residual over `e ≤ KEPLER_ECC_MAX`, rotation norm-preservation) + serde round-trips + same-binary
  bit-replay. HR5 100% region+branch. Adversarial vet folded in three fixes: (a) the frozen
  rotation-convention **byte-pin lives in `crates/core/tests/` (integration, excluded from the coverage
  gate)** — an exact-bits assert inside the Tier-A unit suite would be a cross-toolchain-libm oracle
  that could spuriously RED the HR5 gate; (b) `KEPLER_ECC_MAX = 0.97` is the fail-loud cross-slice
  invariant the Slice-3 eccentricity sampler must respect (32 iters do NOT converge as e→1); (c) the
  negative-M case keeps `normalize_angle` coverage self-contained. Cross-host SPIKE-6a DEFERRED to
  step-2 explicitly (pin is host-local, documented).
- **Slice 2** ✅ DONE (this branch) — taxonomy as data in a new `vd-core` `taxonomy.rs`:
  `GalaxyType`/`SpectralClass`/`PlanetType` (the `entity_kind.rs` `KindDef` idiom — `ALL` + `from_tag`
  fail-loud + `def()` + dual drift tripwires + coherence) with REAL cited boundaries (Pecaut-Mamajek MK
  masses, Duric MLR, Hayashi frost line, Pollack core-accretion, Nair-Abraham galaxy census); closed-form
  inverse-CDF samplers `sample_imf_mass` (bounded Pareto, branchless shim + log-uniform limit),
  `sample_rayleigh`, `orbital_axis_au`, `sample_galaxy_type` + classifiers `classify_spectral` /
  `main_sequence_luminosity` / `habitable_zone_radius_au` / `frost_line_radius_au` / `classify_planet`
  (params-as-args — Slice-3 `UniverseConfig` supplies them, none gathered here); `ProfileKind` tag +
  `vd-sim` `capability::profile_for` (the ONE total wildcard-free core→sim map onto `profiles::*`, plus a
  named `profiles::stub()`). HR5 100% region+branch on BOTH crates; full workspace gate green. Adversarial
  vet folded in: corrected the transposed Duric MLR table (+ a continuity tripwire), fixed the wrong IMF
  midpoint golden, made libm goldens tolerance-based, `requires_frost_line` (IceGiant-only, gas giants form
  both sides), and STRIPPED serde from the `*Def` structs (a `&'static str` field is not `Deserialize`-able
  — the fieldless enums carry the wire discriminant). Off-wire (HR1); no `UniverseConfig`, no wire arm, no
  worldgen consumer. Cross-host SPIKE-6a deferred (transcendentals are boot/seed-time).
- **Slice 3** — THE generator + `UniverseConfig` + the lazy `RealmPath` layer, at **walk-scale
  byte-identical**. `RealmPath`/`RealmLevel`/`RealmKindTag`, the P3 `RealmPathBook`, `walk_scale()` +
  `canonical()` presets, `generate()`/`to_regions()` (pure `GeneratedBody` descriptor → frozen
  `RealmRegion` lowering), `epoch_offset_in_parent` baking a static cell==ZERO center via
  `orbital_state(elements, 0.0)`, the lazy neighbourhood + `own_frame` + client-single-source rewrites,
  the observer-local map query. **HARD GATES:** `node_per_realm_walk.rs` passes byte-identical;
  worldgen containment tests pass byte-identical; a NEW test slices every roster realm's neighbourhood
  and asserts `guard_regions_nest(slice, MAX_REGIONS).is_ok()` + count ≤ 64 + single-root; same-seed
  same-forest replay.
- **Slice 4** — the `RealmId::Universe`/`Galaxy` (+ `FrameRef`) wire arms, its OWN additive frozen-wire
  slice; extend `frame_for_realm`'s one match; flip the `System(0)`/`System(1)` stand-ins; DELIBERATELY
  re-baseline the coordinate-pinned tests (D-17-style, never silently).
- **Step 2 (SEPARATE, LATER — the risky frame-seam slice, gated):** the `SeedEphemeris` / `OrbitalFrames`
  `FrameContext` replacing `IdentityFrames` at BOTH seam sites; the moving center; the D-41 non-zero-cell
  containment math; real-AU-scale bodies going LIVE. **Precondition: SPIKE-6a green.**

## Scope honesty

"Real from day one" = the STRUCTURE + real PARAMETERS + real Kepler + real taxonomy, evaluated at epoch
for STATIC positions, at WALK scale reproducing today's forest byte-identically (a genuine, non-stub
generator — Slices 0–3). What CANNOT be simultaneously true in one first slice and phases later:
1. dedicated `RealmId::Universe`/`Galaxy` arms — a frozen-wire change → Slice 4.
2. `path_for_realm` at REAL scale (the `RealmId`↔`RealmPath` inversion — a flat `RealmId::Planet(p)`
   can't name its `System` without lineage-in-seed-bits or a seed-forest locator) — genuinely owed at
   P4; at P3 the fixed `RealmPathBook` is real-enough. **This is the load-bearing owed piece.**
3. real AU/ly-scale bodies (need non-zero `LatticePos` cells, D-41 math) — `canonical()` is planted as
   a config instance but its bodies can't go LIVE as containment regions until D-41 re-quantization.
4. the moving center (ephemeris) — step-2, gated on SPIKE-6a.

## Top risks (carry into implementation)

1. **Walk-scale byte-identity is a HARD gate, not a nicety** — `node_per_realm_walk` pins exact coords
   (r=40@x0, r=10@x20, half=3@x25, r=180@x0, r=40@x130, ε=2.0) + worldgen probes x=0/20/25/50/130. Any
   body moved by >~1e-9 breaks both. `walk_scale()` radii stay named consts fed as config — `system_soi`
   (luminosity, BASE=100) CANNOT source the walk r=40.
2. **`path_for_realm` at real scale** is the unsolved P4 piece — do not assume the P3 book generalizes
   for free.
3. **Dense-child cap can silently drop a real child** — assert cap ≥ max legit fan-out (fail-loud);
   overflow bodies are the D-45 ambient AoI scan, never a truncated containment neighbourhood.
4. **`realm_regions_for` is still on the boot path** (shard `own_frame`, client single-source) — the
   lazy rewrite must replace both, in-slice.
5. **Cross-host float determinism is a STEP-2 precondition** — do not conflate; shipping step-2 before
   SPIKE-6a green split-brains containment (131 km/ULP).
6. **`UniverseConfig` god-struct + wire-arm creep** — enforce sub-structs; keep Slice 4 (wire arms)
   separate from Slice 3 (generator) so a generator bug and a contract bug never co-mingle.

## Ledger

D-45(a) (seed→celestial-parameter generator) + D-44 (the registry). The moving-regions ephemeris is
D-45. `RealmId::Universe`/`Galaxy` arms are a new additive wire item (Slice 4). SPIKE-6a (cross-host
build-and-diff golden ephemeris) is the step-2 precondition.
