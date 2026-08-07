All load-bearing anchors confirmed against the live tree: `geometry.rs:794` `for_velocity_safe` requires `spin_up > 0 && tear_down > spin_up` strictly (equal-factor panic path is real); `worldgen.rs:194` `.expect("aoi band edges are valid by construction")`; `bins/lib.rs:406` `move_speed: 2.0` / `:412` `boot_ticks_p99: 0`; `stub.rs:4805-4812` edge block, `:4852-4867` dot emit (already `&&`-gated on edge sets), `:4879` proxy reflect `!=` gate, `:5114` `on_proxy_scene_set`. The adversary findings all land where they claim. Here is the final hardened plan.

---

# ONE-UNIFORM-RULE real foundation — FINAL hardened slice plan

## Migration-size headline

**4 substantive re-tunes, ~7 light number-bumps, 1 orthogonal render refactor (~5 tests), and 2 new robustness tripwires the equal-factor choice forces. Everything else — ~100+ tests — is untouched.**

| Bucket | Count | Effort / risk |
|---|---|---|
| **Substantive migrations** (proofs move from toy → compressed-real proportions) | 4 | The moving-walk process smoke at 15 m/s is the one heavy item; de-risked by Slice-0 unit proofs. |
| **Light re-tunes** (frozen f64 literals, factor 14.301, wording) | ~7 | Mechanical. |
| **Render edge→level refactor** (orthogonal to the scale) | ~5 tests + 1 extracted helper | Byte-identical at inert scale by construction — low risk once Finding-1 gate is pinned. |
| **New tripwires** (equal-factor `v_rel=0`, `boot_ticks_p99>0` real-process arm) | 2 | Cheap; both close a *silent* failure. |
| **Class B — untouched** | ~100+ (all crossing/transfer/containment e2e, reconciler verb-injection suite, ~30 synthetic-fixture AoI mechanism tests, the whole client renderer) | Zero change beyond, at most, a scale-selector string. |

**Why Class B is safe (type-verified, not convention):** `RealmRegion.band: ContainmentBand` and `RealmRegion.aoi: AoiConfig` are **distinct types**, each unconstructible from the other's inputs. Crossing/re-home reads `region.band.member(…)` (`stub.rs:3275`); the AoI loop reads `region.aoi.in_range(…)` (`stub.rs:4794`). Changing the AoI *factor* cannot perturb `band`, so no containment/crossing/transfer proof can depend on the AoI value. Every crossing e2e builds its forest from `walk_scale()` (untouched, AoI-inert); the ~30 mechanism tests build their own 1000 m bands (`for_velocity_safe(1000.0, …)`) — preset-independent.

**Files touched (all four slices):** `core/worldgen.rs`, `sim/stub.rs`, `connection-plane/gateway.rs`, `bins/lib.rs`. **No `geometry.rs` field, no wire shape change, no `PROTO_MINOR` bump, no `client-render` / `net.rs` / `realm_scene.rs` change.**

---

## Decision locked by the model (was the plan's open question)

**Option A — one live geometry.** `visual_scale()`'s geometry migrates 40 m/3-planet → **150 m/5-planet** and arms its band to `visibility_factor(θ)=14.301`; it becomes the **static-render expression** of the one game geometry. `visual_demand(v_max, dt)` shares that *exact* geometry with a live occupant-threaded band — the **demand-cluster expression** of the same scale. `walk_demand`'s **live role is retired** (its 1.2 factor was the toy vacuum). `walk_scale` (inert containment forest) and `canonical` (inert planted-real) are untouched.

One live geometry, two drive modes (static vs live-threaded). No 40 m/3-planet live scale survives → no toy-vs-real split. This is what "compressed-real IS the scale, not a demo alongside a toy" means, so it is **mandated by the locked model, not a residual choice.** (Option B — keep `visual_scale` byte-identical, add a parallel `visual_demand` — is explicitly rejected: it leaves a second live geometry.)

---

## Locked numbers (cite these everywhere)

- **The one constant:** `θ_min = 8° = 0.139_626 rad`; `visibility_factor(θ) = cot(θ/2) = 14.301` — a realm is in AoI iff its angular size ≥ θ_min, i.e. visible out to `extent · 14.301 + velocity-lead`. ONE config constant, no magic number, no kind-branch.
- **Compressed-real geometry:** `system_soi = 150`, `n = 5`, `a0 = 0.4 AU`, `ratio = 1.7`, `outer_period = 300 s` → `au→render ≈ 42.45`, `planet_soi ≈ 4.16`, orbits **17 / 29 / 49 / 83 / 142** render-m. `galaxy = 180`, cull `MAX_RENDERABLE_EXTENT_M = 200` — both unchanged.
- **Containment order + headroom flag:** System 150 ⊂ Galaxy 180 < cull 200 — **only 20 m of headroom. FLAG: no one raises `system_soi` past ~180 without also moving the cull.**
- **Visibility reach:** planet `4.16 · 14.301 = 59.5 m`; system `150 · 14.301 = 2145 m`; galaxy `180 · 14.301 = 2574 m` (these are only ever the RHS of `in_range`, never a coordinate — nothing clips). Outer planet @142 subtends `2·atan(4.16/142) = 3.36° < 8°` → **invisible from the star**, streams in as the 15 m/s ship closes to 59.5 m.
- **Ship:** `15 m/s = move_speed(15) · time_multiplier(1.0)`.
- **Hysteresis is velocity-lead only:** with `spin_up_factor == tear_down_factor == 14.301`, `for_velocity_safe` gives `tear_down = spin_up + need`, `need = v_rel · dt · (K_SAFETY 2.0 + extra 0.5)`. At `v_rel≈15, dt=0.02`: `need ≈ 0.75 m ≈ 2–3 ticks`. **The geometric dead-zone is gone — anti-thrash rests entirely on grace** (`~50 ticks = 1 s = 15 m hold at 15 m/s`, reusing the grace-seconds constant). This is intentional per the model, and load-bearing (see Slice 3 reap tail).
- **Over-spin is structurally bounded, not speed-bounded:** the forest is `Universe(root) + Galaxy + System + 5 planets`. Galaxy (reach 2574 m) and System (reach 2145 m) both **exceed any in-system distance (≤180 m) → both permanently up**. So the always-on floor is **2 shards**, not 1. Planets spin up only inside 59.5 m. Worst case all 5 → **max concurrent = 7 shards; typical 3–4.** The ceiling is the realm count — independent of ship speed, `boot_ticks_p99`, and player count (5 planets cap the union). No runaway.
- **Render envelope:** boxes ≤ 180 < star sphere 2000 < far-plane 6000; f32 ULP at ≤200 m ≈ 1e-5 m → no jitter, no clip. **No client-render change.**

---

## Slice 0 — compressed-real geometry + the one constant + the equal-factor tripwire
**(mostly GATED; early visible win)**

**Goal:** land the one game geometry and the one visibility constant; close the panic the equal-factor choice plants.

**Files/functions — `crates/core/src/worldgen.rs` only:**
- Add `VISIBILITY_THETA_MIN_RAD = 0.139_626` (8°, doc-cited) + `fn visibility_factor(theta) -> f64 { 1.0 / (theta / 2.0).tan() }` — **straight-line, HR5-clean** (no branch, one region).
- Parameterize the four derive helpers `visual_au_to_render_m:329` / `visual_planet_soi_r_m:337` / `visual_outer_sma_render_m:342` / `visual_central_mass_kg:347` to take `(system_soi, margin, n, gap_fraction, a0, ratio, outer_period)`.
- Migrate `visual_scale():1108` geometry to 150/5/300 via those helpers; set its band `spin_up_factor = tear_down_factor = visibility_factor(θ)`. Bump `VISUAL_N_PLANETS 3→5`, add `system_soi 150`, `outer_period 300`. `visual_scale` keeps its own authoring occupant speed `VISUAL_OCCUPANT_V_MAX_MPS` (it is the *static* expression — see the Finding-4 reconciliation below).
- Add `UniverseConfig::visual_demand(occupant_v_max_mps, tick_dt_s)` (~:1146): same geometry, band built the `walk_demand` way (speed+tick from args, `grace_ticks_from_seconds`), factor `visibility_factor(θ)`. Reuses `generate_system_forest` + `planet_elements` + `moving_children_for_config` verbatim — **zero new generator control flow.**

**FOLD-IN (adversary — the equal-factor tripwire, ADV-1 #4 / ADV-2 Finding 2):** equal factors collapse `base·tear_down_factor == spin_up`, so `for_velocity_safe` returns `Err(InvalidEdges)` unless `need > 0`, i.e. `v_rel = occupant_v_max + v_child > 0`. `to_regions:194` `.expect(…)` would then **panic at boot**. The shipped presets are safe (`visual_scale` threads its 2.0 authoring speed, `visual_demand` threads 15), but this is a latent trap for any future `v=0` static/idle-occupant preset.
- **Document the precondition** in `visual_demand` / on the equal-factor band construction: *"under a single visibility factor, band validity requires `occupant_v_max + v_child > 0` — a live occupant. A zero-relative-velocity band is `Err`, never a valid inert band."*
- **Add the tripwire test** (Slice 0, on `vd-core`): `equal_visibility_factor_with_zero_v_rel_is_err_not_panic` — asserts `for_velocity_safe(base, 14.301, 14.301, v_rel=0.0, …)` returns `Err(InvalidEdges)` (equality via `expect_err`, not `matches!`).
- **Do NOT add an epsilon to `tear_down_factor`** — that would reintroduce a magic number and break "hysteresis = velocity-lead only." Keep the factors exactly equal; the test + documented precondition closes the immediate hole. (The deeper "a genuinely idle live occupant needs a band" question is flagged DEFERRED below — it needs a user decision, not a silent epsilon.)

**Migration in this slice:**
- **Inert → byte-identical:** `walk_scale` / `canonical` and their goldens (`realm_regions_for_matches_the_frozen_pre_generator_golden:1616`, `walk_scale_equals_the_named_geometry_consts:1530`) **unchanged** — `visual_scale` mutates a `walk_scale()` *clone*, and walk/canonical set `au_to_render_m` directly (never call the parameterized helpers). No `geometry.rs` `RealmRegion` field touched.
- **EXPLICIT re-tune (7 light):** `visual_scale_preset…:2087` + `visual_geometry_respects_the_far_plane_and_soi_non_overlap:2246` + `walk_path_is_untouched_and_visual_planet_ids_are_distinct:2287` → 150 m/5-planet frozen f64 literals; `interest_config_build_live:1778` + `to_regions_stamps_per_realm_aoi:1787` + band tests `1783/1800` → factor 14.301; `walk_and_canonical_keep_aoi_inert_while_visual_stays_live:1888` → wording.

**Gate (100% Tier-A on `vd-core`):**
- `visibility_factor_is_cot_half_theta` (frozen 14.301).
- `visual_demand_geometry_is_compressed_real` — **frozen, non-self-referential** literals (au→render 42.45 / SOI 4.16 / orbits 17,29,49,83,142 / mass) so the assertion can't drift with the code (ADV-FIX-FP).
- `visual_demand_band_is_crossable_for_the_outer_planet` (`spin_up_r 59.5 < outer orbit gap` — **non-vacuous now**, unlike the toy where 1.2×extent swallowed everything).
- `visual_demand_is_releasable_at_the_star` (`tear_down_r < 142`).
- `equal_visibility_factor_with_zero_v_rel_is_err_not_panic` (the new tripwire).
- the 7 re-tuned tests above.

**What the user sees flying it:** launch the client against a **static** compressed-real shard (existing edge-render + full-registry path — no mechanism change yet): the **5-planet Kepler system at real proportions** — inner planets close, outer far, orbits sweeping (moving boxes), and the **angular-size rule already visible from the star** (inner 3 drawn, outer 2 correctly *not* drawn at 3.36° < 8°), inside translucent System-150 / Galaxy-180 shells. First proof the one rule culls by angular size.

---

## Slice 1 — login-registry shrink to the ancestor chain (GATED; flyable-equivalent)
**(`PROTO_MINOR` is already 6 — no bump owed)**

**Goal:** stop the login dump from drawing phantom far planets once render becomes a level set.

**Why required, not optional:** once render is a level set (Slice 2), a login `RealmRegistry` that still ships all children (`realm_registry_for_home:1734` via `realm_neighbourhood_for_held_config`) draws boxes the shard's empty `RenderSent` baseline will never remove → **phantom far planets (83, 142 > 59.5) forever.** Shrinking first is also correct on today's edge path (children edge-add on first tick), so it lands safely *before* the level swap.

**The uniform rule is not bypassed (state this in the plan):** the ancestor chain in the registry vs children via delta is a **factoring, not a second visibility mechanism.** A *containing* realm is the degenerate always-in-AoI case — you are physically inside it, so its angular size is effectively infinite. Children obey the one `cot(θ/2)` band via the delta stream. Server-authoritative throughout.

**Files/functions — `crates/connection-plane/src/gateway.rs`:**
- `realm_registry_for_home:1734` → project only the **ancestor chain root→home** (keep `parent.is_none()` + ancestors; **drop `parent == Some(home)` children**). `RealmRegistry{regions,root}` **shape unchanged** — fewer regions, no trailing field, no new variant.
- Gate the shrink on `negotiated_minor >= 6` in `realm_registry_for_home` / `maybe_announce_realm_registry:1763`: `minor≥6` → shrunk registry + children via delta; `minor==5` → full ancestors∪children (today's path). One monomorphic place.

**FOLD-IN (ADV-2 wire axis):** `PROTO_MINOR` is **already 6** in the tree (`version.rs:47`; the memory-note "5" is stale). `RealmSceneDelta` is already a minor-6 variant. So **no `PROTO_MINOR` bump, no new field/variant** — the only wire action is this `minor≥6` gate. A `minor==5` peer (constructible, backward-compat-correct) still receives the full set.

**Migration:**
- `minor==5` path **byte-identical** → `realm_registry_for_home_is_the_faithful_seed_neighbourhood…:7867` **stays green.**
- **EXPLICIT re-tune (ADV-FIX-COV):** the existing test homes to a *leaf* Area, so the new drop-children branch never fires. Add a **System-home** test asserting (a) ancestors root→System kept, (b) every planet-child dropped, (c) a `minor≥6` peer sees the shrunk set / a `minor==5` peer sees the full set. Rework `realm_registry_streams_only_for_an_armed_cluster…:7818` and `realm_scene_delta_routes…minor_6:7941` for the split.

**Gate (100% Tier-A on `vd-connection-plane`):** the System-home test (both drop-branch and both peer arms) + the re-tuned two above.

**What the user sees flying it:** identical to Slice 0 — children now arrive one tick after login via the delta rather than in the registry dump ("no login draw-the-whole-neighbourhood dump," satisfied structurally). Render smoke tolerates the one-tick-later children.

---

## Slice 2 — edge → level render (GATED; byte-identical at inert scale; flyable-equivalent)
**(render + spin-up now read ONE membership)**

**Goal:** replace per-observer edge add/remove events with a per-observer *level set* over the one band, converging and self-healing.

**Files/functions — `crates/sim/src/stub.rs`:**
- New resource `RenderSent(BTreeMap<AccountId, BTreeSet<RealmId>>)` — exact twin of `ForwardedProxyScene:324`; insert default beside the others (~:1342).
- Extract `fn diff_scene_into_delta(new, baseline) -> (Vec<RealmShape>, Vec<RealmId>)` from `on_proxy_scene_set:5126-5140`; call from **both** `on_proxy_scene_set` and the dot emit (one monomorphization, one covered region — HR5/DRY).
- Thread `RenderSent` through `evaluate_realm_aoi:4599` → `aoi_decide:4673`. Delete `render_adds/render_removes:4763-4764`; replace the edge block `4806-4812` with a LEVEL push reusing `next_in:4805`. **The demand union `union_verb:4832` is UNTOUCHED** — render and demand derive from the SAME transition on the SAME band.

**FOLD-IN (ADV-2 Finding 1 — HIGH, byte-identity landmine):** the design's "mirror `4879-4888`, bitwise `|`" is **ambiguous** — `4879` is the proxy reflect loop's *stored≠current* gate (`proxy_sent.get(acct) != Some(&cur)`), which is safe for proxies *only because* their iteration set is empty at walk. The dot emit iterates `render_routes`, which is **non-empty at walk** (there is a player). Following the `!=` anchor would emit a spurious `RealmSceneDelta{added:[], removed:[]}` on tick 1 in the walk/canonical production path → **output byte-identity break + steady per-player wire noise.**
- **Pin the emit gate to diff non-emptiness:** `if !added.is_empty() | !removed.is_empty() { push … }` (the `on_proxy_scene_set:5142` pattern, fed by the extracted `diff_scene_into_delta`), **bitwise `|`** (no short-circuit region).
- **Make `render_sent.insert` unconditional** (like `on_proxy_scene_set:5153`), NOT gated on the diff.
- **Drop the `4879-4888` anchor for the emit gate**; keep that `!=` pattern *only* for the `render_sent.retain` prune of departed dots.

**FOLD-IN (ADV-2 Finding 5 — coverage hygiene):** the level push uses **`render_routes.contains_key(obs) & next_in`** (bitwise `&`, matching `in_range:860` / `on_proxy_scene_set:5142`), not `&&` — no uncoverable false-LHS region.

**FOLD-IN (ADV-2 Finding 3 — correct the claim):** byte-identity holds because **no *delta* is emitted** at inert scale, not because "no `RenderSent` key is written." Under the correct gate, the walk emit loop iterates `render_routes` (non-empty), computes `diff(∅,∅)=(∅,∅)` → no delta, but runs one **output-inert empty-set `insert`** per dot per tick (bounded by player count, pruned by `retain`). State it as: *"no delta emitted; a bounded, output-inert empty baseline may be written."* Do not try to gate the insert to make the claim literal — that adds an uncoverable arm for zero benefit.

**Byte-identity (proven, not asserted):** inert `in_range(false, dist>0) = false` → `next_in` false → level set empty → `diff(∅,∅)=(∅,∅)` → no delta; demand `union_verb(false,false)=None` inert too. Render and demand read the **same `next_in` on the same band**, so render can only leak a delta where demand already leaks a spin-up — and production demand is proven inert. Production (`VD_UNIVERSE_SCALE` absent / `walk`) **byte-identical**. (One pre-existing shared corner: `min_dist == 0` exactly makes even an inert band return `true`; shared with the demand path, so production evidently never hits `dist==0` — not introduced here.)

**Migration:**
- **EXPLICIT re-tune (2 live-fixture tests):** `evaluate_realm_aoi_streams_a_render_delta_on_acquire_and_release:12821` → level-diff (acquire adds / release removes / an unchanged tick emits nothing); `a_proxy_observer_gets_no_render_delta_but_a_dot_does:13175` → level set.
- **UNCHANGED (Class B, ~30 synthetic-fixture AoI tests):** the whole `aoi_transition_covers_every_hysteresis_arm` / `union_verb_*` / `evaluate_realm_aoi_{inert, unsynced, empty, spinup_then_keepalive, grace_then_drop, predictive, two_observers, per_observer_hysteresis, …}` cluster stamps its own 1000 m band → immune to the factor.

**Gate (100% Tier-A on `vd-sim`):** the 2 re-tuned tests + new `diff_scene_into_delta` 4-arm coverage (added-only / removed-only / both / neither, covered once), `render_sent_prunes_a_departed_dot` (twin of `aoi_decide_prunes_expired_retained_occupants:13010`), `render_level_holds_a_passed_realm_across_grace` (no-flicker). Re-run `render_smoke` / `render_boxes_smoke` on the level path.

**What the user sees flying it:** identical static compressed-real view — now driven by the convergent level set (a dropped delta or a fresh shard self-heals to a full re-stream next tick).

---

## Slice 3 — demand cluster → `visual-demand` + migrate the live proofs (FLYABLE: the seamless warp)
**(the un-vacuuming slice; risk concentrates here)**

**Goal:** make compressed-real THE demand scale and re-site the spin-up-ahead / reap-behind proofs onto the now-crossable geometry, in real processes.

**Files/functions — `crates/bins/src/lib.rs`:**
- `UniverseScale::VisualDemand:2174`; `universe_scale_of:2195` arm `"visual-demand"`; `boot_regions_and_movers:2409` arm = `realm_regions_for_config(seed, visual_demand(v_max, dt)) + moving_children_for_config`.
- Repoint `demand_orchestrator_env:976` `VD_UNIVERSE_SCALE "walk-demand" → "visual-demand"`.
- `DEV.move_speed 2.0 → 15.0 (:406)` — single-sourced: rides `VD_SPEED (:936)` to the shard (`shard.rs:116-123` already threads `move_speed·time_multiplier` + `tick_dt` into `boot_regions_and_movers` → `visual_demand`), so the band's `occupant_v_max` **is** the sim's integrated speed.
- **Retire `walk_demand`'s live role:** delete `UniverseScale::WalkDemand:2179` / `"walk-demand"` string / `walk_demand()` live proofs. `walk` (inert) stays the byte-identical production default.

**FOLD-IN (ADV-1 #2 — REQUIRED ARM, not a tuning nicety):** the "streams in AHEAD" mechanism is **not** the geometric factor — it is the predictive horizon in `occupant_child_dist` (`stub.rs:4952`): `min(live, |child − (occ + occ_vel·horizon_s)|)`, `horizon_s = boot_ticks_p99 · tick_dt` (`stub.rs:4689`). The render-add rides the *same* `next_in` transition as the demand, so box and boot both fire at the predicted crossing. **With `boot_ticks_p99 == 0` (the current DEV value, `bins/lib.rs:412`) the predicted term collapses (`occ_vel·0`), `pred == live`, and the box appears exactly at the 59.5 m live crossing with zero boot masking** — the headline proof is **silently vacuous in a real process** while still passing a virtual-clock test where boot is instantaneous.
- **`boot_ticks_p99 > 0` is a required arm of this slice.** Set it to **at least the measured real-process boot p99.** The total lead is `(59.5 − 4.16)/v_max + boot_ticks_p99·dt = 3.69 s + boot_ticks_p99·dt`. This is a closed loop and the correct lever. **Flag: this is the load-bearing item, above "timing tune."**

**FOLD-IN (ADV-1 #3 — reap-side grace tail):** equal factors mean anti-thrash rests *entirely* on the 1 s grace (= 15 m of "ghost hold" behind the ship). Reap-behind lags the geometric exit by `grace + (tear_down − exit)`. **The migrated moving-walk proof's `teardowns_reaped` budget must allow the full grace tail after the exit**, or it asserts too early and flakes. Name the reap-side grace tail (~1 s / ~15 m) in the budget, not just the acquire-side lead.

**FOLD-IN (ADV-1 #4 / #5 — aim at the approach-window position):** the predictive horizon projects the **occupant only**, never the target; planets orbit at ~3 m/s (outer) up to faster inner. Over the boot horizon the target drifts, and the moving 59.5 m sphere shifts. **The proof's ship must aim at the planet's position over the approach window, not its t=0 epoch position** — a naive epoch-aimed shot can graze or miss the moving sphere in the deterministic proof. Add a comment that acquire is predictive-on-occupant-only (asymmetric with release, which already models `v_child` in `need`) — acceptable at 15 m/s, a real cliff for faster occupants (see DEFERRED).

**FOLD-IN (ADV-2 Finding 4 — stop citing a guard that doesn't exist):** `worldgen.rs:864` is a **doc comment, not an assert**; the composer speed cross-check (`stub.rs:1292`) is **not implemented**. Do **not** claim "M-2 boot debug_assert holds," and **do not implement** the deferred `is_live` speed-assert in this work. Reconcile the static-vs-live speed split: `visual_scale`'s static band is authored at `VISUAL_OCCUPANT_V_MAX_MPS` (a static-render authoring constant); the DEV `move_speed 15` threads **only** into the `visual_demand` (live) path via `VD_SPEED`. Keep the static `Visual` dev-launch coherent with that (its band stays authored at the static speed — under equal factors this only shifts `tear_down`/release, so it is cosmetic and correct). If a future change *does* wire the speed-assert, scope it to the threaded `visual_demand` path only.

**Migration (the substantive re-tunes):**
- `rlm_demand_login` (stationary): home at the star; login draws inner 3, outer 2 culled by angular size. A stationary login's byte-identity now rides `visual_demand` (an occupant at origin is nowhere near a spin-up band → no spurious demand, same guarantee). Assert Kepler planet ids + 17/29/49/83/142 distances instead of `"Planet 7"`.
- The moving-walk proof (`node_per_realm_walk` / `rlm_demand_walk`): fly out at 15 m/s → the **outer planet's `RealmSceneDelta{added}` arrives BEFORE the ship reaches its 59.5 m spin-up band and before its shard boots** (requires `boot_ticks_p99 > 0`); the passed inner planet **stays drawn** (grace/velocity-lead hold); `spins_failed == 0`; **`teardowns_reaped ≥ 1`** on fly-out (with the grace tail in the budget); ship aimed at the approach-window position.
- Retire/repoint the `walk_demand` unit proofs (`walk_demand_band_is_crossable…:1817`, `:1861`, `:1876`, `walk_demand_differs_from_walk_only_in_aoi:1907`).
- **UNCHANGED (Class B):** all crossing/transfer/containment e2e (they stamp `aoi: inert` + a hand-built `ContainmentBand` — containment ≠ AoI is a real field boundary), the reconciler verb-injection suite (`realm_lifecycle_e2e` feeds `RealmDemand{verb}` directly), `resolve_universe_scale_*`. `node_per_realm_walk`'s containment crossing survives the wider AoI — only its scale selector swaps.

**Gate (100% Tier-A on `vd-bins`):** re-tuned `rlm_demand_login` + the moving-walk process smoke green; `universe_scale_of` `"visual-demand"` arm; `boot_regions_and_movers` `VisualDemand` arm.

**What the user sees flying it:** the real thing — pilot a compressed-real 5-planet Kepler system at 15 m/s, aim at the outer planet: **its box streams in ahead of you and its server boots because it entered your AoI**; cross in (location flips to that planet); fly back out and **the vacated realm reaps behind you** (after the grace tail). Real processes, real client, boxes, **no loading / teleport / pop**. Server-authoritative throughout (the client never decides visibility). The spin-up-ahead / reap-behind loop — now **non-vacuous** because the geometry is genuinely crossable (outer planet @142 vs spin-up 59.5).

---

## Owed-NOW vs DEFERRED

**Owed NOW (this plan):**
- The uniform `cot(θ/2)` band on the one live geometry (`visibility_factor`, θ_min=8°).
- Compressed-real 150 m / 5-planet Kepler scale as **THE** scale (`visual_scale` static + `visual_demand` live, sharing one geometry).
- Render = per-observer level set over that one band (no edge events, no login dump); render + spin-up read ONE membership.
- Login registry shrunk to the ancestor chain (`minor≥6`, no wire bump).
- Client draws its AoI as boxes (level-set in→drawn / out→not, streams-in-ahead, no flicker, orbits move, nested shells stay).
- The two new tripwires: equal-factor `v_rel=0 → Err` (Slice 0); `boot_ticks_p99 > 0` real-process arm (Slice 3).
- **No `geometry.rs` field, no wire shape change, no `client-render` / `net.rs` / `realm_scene.rs` change.**

**DEFERRED (explicitly NOT this plan):**
1. **Literally-real astronomical distances** — big-world camera-relative f64→f32 floating origin + wide-range depth (the client jitters past ~1e7 m). These slices stay ≤200 m by design.
2. **The nested / re-home "moon-doesn't-vanish-when-you-enter-the-building" recursive world-position compose** + warp-time registry re-announce on a deeper cross (VU-6 / D-RLM-13). **These slices are re-home-free.**
3. **P4/P5 cross-cell `LatticePos` rebase** (`cell != ZERO`) — never reached at ≤200 m.
4. **Meshes** — boxes are the deliberate proxy stage.
5. **Faster-occupant cliff (P8 ships):** the geometric 59.5 m lead does **not** scale with speed — only `boot_ticks_p99` does, which both widens over-spin and projects along a staler planet position (the acquire path models occupant motion but not orbital `v_child`). The 15 m/s plan is sound; faster occupants are a real deferred cliff, not a free extrapolation.

---

## Flags needing a user decision

1. **`boot_ticks_p99` value (Slice 3).** Must be an empirically measured real-process boot p99 (not 0, or the headline proof is vacuous). Recommend: measure the real boot p99 on the demand cluster and set `boot_ticks_p99` to at least that; keep the proof's lead budget `= 3.69 s + boot_ticks_p99·dt`. **Confirm you want this measured-and-set as part of Slice 3 rather than a hand-picked constant.**

2. **The genuinely-idle live occupant (`v_rel = 0`) under a single factor — DEFERRED design choice.** Slice 0 closes the immediate panic (documented precondition + `Err`-not-panic test) without an epsilon, preserving "hysteresis = velocity-lead only." But a *future* preset that wants a live realm to stay up around a **stationary** occupant has no valid band under one factor. Options when that arrives: (a) a minimum-relative-velocity floor, (b) a graceful inert fallback in `to_regions` for zero-`v_rel` shells, or (c) accept "always-on shells need a nonzero authoring speed." **No decision needed now — flagging that this is the one place the single-factor model is genuinely constrained.**

3. **System-SOI / cull headroom (standing flag, not a decision now).** System 150 ⊂ Galaxy 180 < cull 200 leaves 20 m. **Anyone raising `system_soi` past ~180 must move `MAX_RENDERABLE_EXTENT_M` in the same change** — call it out in the geometry doc so it can't drift.

Files that will be touched, in order: `crates/core/src/worldgen.rs` (Slice 0), `crates/connection-plane/src/gateway.rs` (Slice 1), `crates/sim/src/stub.rs` (Slice 2), `crates/bins/src/lib.rs` (Slice 3).