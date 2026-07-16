# transfer_frame wiring — authority-pose rebinding (fixes FRAME-REBINDING label lag)

Design workflow `wf_082a86bb`. Verdict **GO_WITH_FIXES** (is_sound:false → under-specification, 5 must-fixes folded).
USER-APPROVED to implement now (2026-07-16).

## The fix
The SOURCE re-expresses the crossed/re-homed pose into the DEST realm's frame via the already-built
`transfer_frame` (crates/core/src/frame.rs:91) at the **3 envelope-BUILD sites** (source-computes-and-ships, per
transfer_protocol.md:324-328 — NEVER dest-re-expresses-on-adopt):
- (a) DURABLE crossing: `build_crossing` (crates/node/src/saga_runtime.rs:755, pose written verbatim at :774 from
  `flush_pose` at :762). Insert `let pose = transfer_frame(&pose, ctx.to_frame, &IdentityFrames)?;` (tick from
  `pose.universe_tick`, no `now` arg). `ctx.to_realm` already on SagaCtx (saga.rs:55).
- (b) D-37 RE-HOME: `build_rehome` (saga_runtime.rs:813, pose at :827 in `ReHomeState::PoseOnly`). SAME insert.
  Re-home reads the SAME stashed `LiveSaga.flushed_pose` (SOURCE frame), so source→third is a SINGLE re-expression
  (no intermediate double-count).
- (c) TRANSIENT batch: `emit_transient_batch` (crates/sim/src/stub.rs:2923-2927, `pose: t.pose` per item). Re-express
  each into the group's `to_frame`.
Keep the 3 DEST-store sites VERBATIM (`.sanitized()` only): apply_crossing (stub.rs:2333), re_home_apply
(stub.rs:1974), adopt_transient_batch (stub.rs:2983) — no re-transform ⇒ exactly ONE transform per hop.

Through P3 every SystemSpace placement is IDENTITY (frame.rs:12), so transfer_frame(SystemSpace{7}→{8}) leaves
pos/vel/orient UNCHANGED, frame→{8}. HUD (pose.rs:91) reads "System 8"; dot stays in box B. P4/P8/P10 supply the real
ephemeris FrameContext — no reshape.

## The 5 folded must-fixes
1. **The RealmId→FrameRef map is REAL** (no forward map exists; only lossy inverse `FrameRef::realm()` pose.rs:73).
   Build a TOTAL `frame_for_realm(realm, provenance)` in vd-core: System(s)→SystemSpace{s}, Planet(s)→PlanetCentered{s},
   Ship(id)→ShipLocal{id}, Station(s)→StationLocal{s} are 1-field; **Area(s)→AreaLocal{planet_seed:PROVENANCE,
   area_seed:s}** needs the parent planet seed (NOT in RealmId::Area) — take it from the boundary's `parent` field
   (RealmBoundary.parent, geometry.rs). Area transfers don't exist through P3, so the Area arm is provenance-parametric
   + deferred, but the map is TOTAL + HR3-clean (never a per-kind feature match).
2. **Post-CAS Err handling** — `EmitCrossing` fires POST-CAS (saga.rs:584, authority already committed to dest). A
   None-drop would strand the dest at origin. On `transfer_frame` Err: ship the SOURCE-frame pose (safe degrade, label
   lags, NEVER drop) + loud log. INERT through P3 (IdentityFrames always Some → Err dead); the proper pre-commit-abort
   is P4 (when the ephemeris ctx can Err).
3. **HR5 monomorphization** — IdentityFrames creates a NEW `transfer_frame` monomorphization whose 2 Err regions are
   structurally unreachable → uncovered under `--fail-under-regions 100`. Refactor transfer_frame's `.ok_or(..)?` Err
   mappings into a MONOMORPHIC helper (branchless-shim discipline) so the generic body has no per-mono branch. Verify
   `just coverage-fast` 100% on the new mono before landing.
4. **Re-anchor the dropped-vs-landed discriminator** on the ONE frame-only gate (p2_transfer_gates.rs:393-398, comment
   :383-384 warns a bare pos check is insufficient) onto the auth-sub flip / non-adopt-origin position BEFORE flipping
   its assertion to seed-8 — else a bare flip silently disables dropped-crossing detection.
5. **Scope the DEFERRED finding-5 flip HONESTLY** — authority-pose rebinding lands; the ghost feed (stub.rs:3625 ships
   the seed-8 pose to the seed-7 ghost host verbatim) + the client `world_pos` composite (view.rs:228-238 collapses all
   frames to one identity arm) STILL owe re-expression at P4/P5. Do NOT blanket-close; scope to "authority rebound".

## Gate flips (RED-to-flip → seed-8)
- tests/tests/p2_transfer_gates.rs:393-398 (+ re-anchor drop discriminator on auth-sub).
- tests/tests/p2_transfer_gates.rs:750-754 (backed by the independent auth-sub flip :735-743).
- tests/tests/crossing_e2e.rs:303-307.
- crates/node/src/saga_runtime.rs:3021-3035 + :6005-6019 (emit_crossing byte-equality → add `dest_flushed_pose()` =
  flushed_pose with frame seed-8, assert THAT).
- crates/node/src/saga_runtime.rs re-home byte-equality tests (~:4942/:5005/:5056) → dest_flushed_pose().
- crates/bins/tests/render_crossing_smoke.rs — PROMOTE `location` from the lagging/annotated signal to a HARD
  `location == "System 8"` wait-gate (the empirical DoD); remove the :480 "cosmetic gap" annotation.

## Slice plan (always-green)
- S0 vd-core: add pub `IdentityFrames` FrameContext + `frame_for_realm` (total, Area provenance-parametric) + the Err
  monomorphic-helper refactor + unit tests (identity flip 7→8; empty-ctx expect_err; frame_for_realm arms). No caller.
- S1: thread `to_frame: FrameRef` onto SagaCtx (saga.rs:34, additive Copy field) set at saga creation from
  frame_for_realm(to_realm, provenance); carry to_frame on the transient Group + TransientStatus::Crossing. Unused. Green.
- S2: wire `build_crossing` (durable) + flip the 2 emit_crossing byte-gates + the 3 e2e frame gates (+ re-anchor drop).
- S3: wire `build_rehome` + flip the re-home byte-gates. HR2 symmetry.
- S4: wire `emit_transient_batch` + flip the transient unit tests.
- S5: acceptance — promote render_crossing_smoke `location == "System 8"` hard gate + the build-site Err-path HR5 test.
- S6: flip DEFERRED.md FRAME-REBINDING to 🟩 scoped "authority rebound (ghost + client-composite owed P4/P5)";
  `just coverage-fast` confirm.
