# Visual-Universe Arc — the testable realm-only fully-wired game state

**Goal (user, 2026-07-23):** the WHOLE universe wired end-to-end, REALMS ONLY (no blocks), TEST THE
WHOLE GAME IN A WINDOW with box/sphere proxies (no meshes). The client must: see a galaxy map
(pointable stars) → see the current star system (planet realms as blue spheres, right relative sizes,
real orbits) → FLY the dot between planet realms (crossing machinery) → WARP (point at a star, engage,
speed up, travel to it — reusing the existing realm-boundary crossing). This is the ACCEPTANCE TEST for
the entire realm layer (transfer-first: prove the realm/motion/crossing foundation before blocks).

Plan vetted by workflow `wf_972b724f` (survey → 3 sequencing proposals → adversary + synthesis),
tree-verified at HEAD `e5f300d`.

## Locked architecture decisions (user-confirmed)

1. **Agnostic client = `RealmRegistry`.** A NEW RELIABLE `ServerControlMsg::RealmRegistry{regions:
   Vec<RealmRegion>, root: RealmId}` (append variant, `PROTO_MINOR` 3), emitted by the gateway at
   session-Active + RE-SENT on warp arrival, AoI-scoped via `realm_neighbourhood_for_held`. The client
   decodes it into its boot `RealmScene` via the existing `RealmScene::from_regions`. Deletes
   `--realm-boxes` as the networked source (kept as an offline dev override). **Full `RealmRegion`
   payload** (it already carries the per-realm seed in `RealmId` + the analytic boundary in `shape`).
2. **Scene-graph shipped; content seed-derived (throughput).** `RealmRegistry` carries only
   `{identity+seed, analytic boundary, position, hierarchy}` — NEVER blocks. The expensive block
   content (P4+ voxels) is seed-derived on the client from the per-realm seed `RealmRegistry` delivers,
   with the server shipping only deltas (player edits) + fully-dynamic realms (ships). This arc ships
   the small scene-graph; it ENABLES (never corners) the seed-derived content layer.
3. **Warp = a generic speed modifier + the EXISTING containment crossing** (System→Galaxy→System is the
   same C-6c re-home). NO bespoke warp path / FSM / teleport (HR3). At VISUAL scale the galaxy fits the
   window so f64 offsets suffice (D-41 coarse ly-cells are the canonical-scale concern, P4/P5).
4. **Moving-container straddler: BUILD THE FULL P8 GUARD NOW** (user chose the robust option). Flying
   into an ORBITING planet SOI is genuinely correct, not accidentally-correct.
5. **Scale = VISUAL** (`visual_scale()` synthetic-mass window preset; canonical real-AU/ly → P4/P5, A2
   already locked). Server authors orbits (author-and-ship); client renders shipped-only (no prediction).
6. **SEAMLESS — a HARD rule for the WHOLE experience (user, emphatic 2026-07-23): NO toggles, NO loading
   screens, NO teleportations, ever.** There is NO "galaxy map" mode. The neighbouring systems' STARS are
   ALWAYS rendered around you, correctly positioned relative to your current system (like the real night
   sky). Warp is a SEAMLESS physical fly-by: point at a star in the actual world, engage, accelerate, and
   PHYSICALLY PASS the other systems as the star field parallaxes, until the destination grows from a
   point into a full system whose planets STREAM IN as you approach (prefetched before the SOI so there
   is NO pop). One continuous camera throughout. At VISUAL scale the galaxy is a ~180m space and each
   system is a ~40m SOI sphere at its galaxy position, so the "stars" ARE the other system spheres — warp
   = flying across the Galaxy realm past them. The acceptance test drives the WHOLE game (minus blocks)
   this way. See [[feedback_seamless_experience]].

## What's already built (the leverage — reused VERBATIM)

- The FA realm-observer MOVING-pose pipeline (`authored_realm_snaps`/`emit_realm_frames` →
  `RealmSnapshotDatagram` → gateway `on_shard_realm_frame` → `RealmView` → `RealmScene::overlaid`).
- The crossing saga + containment detector (`evaluate_realm_boundaries` → fence-CAS, proven System 7 ↔
  Galaxy ↔ System 8 both ways, durable + transient — C-6c).
- The ephemeris (`orbital_state`/`solve_kepler_fixed`) + the visual generator (`generate_system_forest`,
  `visual_scale()`) + the boot selector (`resolve_universe_scale`/`boot_regions_and_movers`, FA-5 S2).
- `LocalFrames`/`frame_context` frame-authority; sphere render (`shape_of` Shell→Sphere already works);
  the `PROTO_MINOR` append machinery (proven twice: `UniverseRate`, `OwnEntity`); vdctl + wgpu readback.

## Delivery rhythm (user, 2026-07-23): RUNNABLE-INTERACTIVE FIRST, gates after

Every slice delivers a **runnable command the USER launches + play-tests BY HAND first**. I verify it
COMPILES + BOOTS (so it's not handed over broken), but the visual/interactive check is the user's. ONLY
AFTER the user confirms it works in-game do I build the automated screenshot/e2e regression gate + commit
(matches [[feedback_dont_commit_broken]]: commit only after the user confirms in-game). So per slice:
(1) build the increment + a `just`/script run command; (2) verify build+boot; (3) USER runs + tests
interactively; (4) after user-confirms → add the screenshot/e2e gate + adversarial review + commit.

## The slice sequence (RUNNABLE first, then the gate + adversarial review + commit)

- **VU-0 — Baseline visible (blue orbiting spheres over the fed file, live QUIC).** ZERO new structure.
  A `just` recipe boots the `VD_UNIVERSE_SCALE=visual` cluster, dumps the matching regions.json, launches
  the client fed that file; a generic Planet→blue color law replaces the seed-hash rainbow. Gate: vdctl
  e2e, `WaitUntil RealmFramesApplied>=2`, two captures, ASSERT a planet sphere's pixels moved. Closes the
  FA-5-owed HR6 live-QUIC loop. **Visible: orbiting blue spheres in the client.**
- **VU-1 — Agnostic client: `RealmRegistry` on join (single system).** The #1 shift, landed early where
  it's trivially the whole scene. Wire: the appended arm + `PROTO_MINOR` 3 + golden roundtrip. Gateway:
  emit the neighbourhood subtree at session-Active. Client: `on_control` decode → `load_scene`. Gate:
  client launched with NO scene file renders the same spheres; `RealmScene` from registry == from the
  fed file (byte-compare); minor-negotiation test; HR5 100% on the new arms. **Visible: same scene, no
  scene file — the universe came from the server.**
- **VU-2 — Multi-system / whole-galaxy generator.** `generate_galaxy_forest(seed, config)` = one
  branchless `for s in 0..n_systems` loop consuming the existing-unused `GalaxyConfig.system_count`, each
  System a Galaxy child at star-map position `f(seed, index)`, each with its orbiting planets. SYSTEM_A =
  the s==0 degenerate; walk/canonical byte-identity holds. Gate: worldgen unit tests (N systems,
  deterministic, valid nesting, no siblings in a neighbourhood). **Visible: green generator suite (galaxy
  is `f(seed)`).**
- **VU-3 — Agnostic client discovers the whole galaxy (star field + AoI-scoped detail).** The client
  receives ALL system STUBS galaxy-wide (position + kind + radius — the STAR FIELD, cheap: a few floats
  per system) PLUS the current system's full planet forest (AoI-scoped). NOT "sibling stubs only" — ALL
  stubs, because every star must be visible. Gate: ASSERT the client holds every system's stub but only
  the OCCUPIED system's planet forest (the fan-out bound: planet forests are AoI-scoped, star stubs are
  galaxy-wide). **Visible: proof the client sees all stars but only its own system's planets.**
- **VU-4 — The SURROUNDING STAR FIELD (always-visible neighbouring systems + world-space warp targeting).**
  NO map, NO mode, NO toggle. The neighbouring systems' stars are ALWAYS rendered around you as points/
  spheres at their correct galaxy positions relative to your current system (they parallax as you move).
  Warp targeting = point at a star in the ACTUAL WORLD (ray-cast against the star spheres) → a selected
  `RealmId::System`; a vdctl `PointAtStar`/`SelectStar` arm for HR6 (no mode-toggle arm — nothing to
  toggle). At visual scale the stars ARE the sibling system SOI spheres (VU-3 delivered their stubs), so
  this is mostly a render tune (a bright star point at each system centre + the SOI sphere) + the pick.
  Gate: vdctl e2e — the star field renders at seed positions relative to the current system; `PointAtStar`
  selects the expected system. **Visible: the real star field around you, pointable — no map, no toggle.**
- **VU-5 — Fly between planet realms + the FULL moving-container straddler guard.** The dot flies across
  a planet SOI → the ONE re-home saga fires, over the agnostic-discovered scene with movers streaming.
  BUILD the moving-container straddler guard (D-45(e)) so crossing into an ORBITING SOI is correct
  (dwell/commit consistency on a moving boundary), not accidentally-correct. Gate: e2e crossing (before/
  after capture, dot re-parents into planet-B's sphere; ASSERT one `CrossingRequest`, no bespoke path) +
  the straddler guard's own tests. **Visible: the dot flying into a planet's sphere and re-homing.**
- **VU-6 — Warp = a SEAMLESS physical fly-by (speed modifier + the SAME crossing + the FA-6 realm-fence
  gate).** Point at a star (VU-4 selection), engage → a GENERIC speed modifier on `integrate` accelerates
  the dot toward the target; it PHYSICALLY LEAVES its System SOI → travels the Galaxy realm PAST the other
  system spheres (the star field parallaxes) → enters the target SOI. ONE continuous camera — NO
  transition screen, NO teleport (the SEAMLESS rule). The FA-6 `realm_fence` gate lands HERE (its first
  live producer, so it's HR5-coverable not inert): gateway forwards `realm_fence`, `RealmView` gates on
  `(realm_fence, frame_id)`, a higher fence supersedes + resets the realm's high-water. Gate: vdctl e2e —
  speed rises, leave A, cross the Galaxy passing sibling spheres, enter B; a capture SEQUENCE (not just
  2 frames) showing the continuous fly-by: A shrinks, siblings pass, B grows from a point; ASSERT the
  SAME saga (no bespoke path); FA-6 regression (a re-homing realm's fresh-but-lower frame_id is ACCEPTED).
  **Visible: a seamless warp sequence — A shrinks, you pass other stars, B grows from a point.**
- **VU-7 — SEAMLESS proximity streaming (the full acceptance loop).** The AoI is PROXIMITY-CONTINUOUS,
  not on-arrival-discrete: as the dot APPROACHES a system its planet forest STREAMS IN (`RealmRegistry`
  re-send, PREFETCHED before the SOI so there is NO pop at the boundary — the SEAMLESS rule); as it
  leaves, the forest evicts (`EventMsg::RealmRemoved` + the `RealmView` removal path). The `RealmSnapshot`
  moving feed is AoI-FILTERED by the SAME proximity relevance predicate (one rule filters BOTH the
  registry + the moving feed). The star STUBS stay galaxy-wide (always visible); only the per-system
  planet DETAIL streams. Gate: the FULL acceptance e2e — boot (agnostic, star field visible) → point at a
  star → warp (seamless fly-by) → arrive (B's planets already streamed in, A evicted, no stale ghost, NO
  pop) → fly between B's planets; ASSERT the stream-in completed BEFORE the SOI crossing (seamless), no
  leaked tracks after N warps. **Visible: the complete seamless loop — the entire realm layer proven in a
  window, no toggles/loading/teleport.**

## No-corner (verified)

Voxels (a planet sphere gains a voxel surface inscribing the same boundary, seed-derived + delta'd),
ships-as-realms (the same author-and-ship law + crossing reuse; the FA-6 realm_fence gate VU-6 lands is
what a warping ship-realm reparent needs), signals (the VU-6 speed modifier is exactly the seam a P9
thruster signal fills), AoI (subtree-scoped from VU-1, filtered in VU-7) all stay strictly additive.

## Corners to AVOID (flagged)

Do NOT ship a whole-galaxy `RealmRegistry` (re-creates the D-9 O(N) fan-out on the reliable plane —
always AoI-scope it). Do NOT plant the `realm_fence` gate inert before warp exists (HR5-uncoverable —
land it WITH its live producer in VU-6).
