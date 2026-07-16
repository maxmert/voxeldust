# Batch B / V4 — render_crossing_smoke.rs (design + adversary, FOLDED plan)

Workflow `wf_55dae6f6-f27` (understand → design → hostile adversary). Verdict **GO_WITH_FIXES**;
`is_sound:false` on first design (a FATAL keying flaw), 6 must-fixes folded below.

## Engine-change verdict: NONE (code-verified, high confidence)
A single client on the dual-aware gateway ALREADY receives the DEST snapshot after an
authority re-home, with NO client re-subscribe and NO engine/bin change:
- DEST promote emits `ShardToGateway::SubscriptionReady` on `MsgClass::Control`
  (stub.rs:1829-1840 / 1358-1366).
- Gateway `SubscriptionReady` arm calls `open_sub(from=DEST)` + `AuthorityChanged`, idempotent
  (gateway.rs:1822-1852); thereafter `on_shard_frame(DEST)` fans DEST snapshots to that session
  (gateway.rs:1864-1945, `subscribers_of` 442-447).
- Client is purely reactive (net.rs:197-213); `DeliveredView` is multi-mesh, keyed `(SubId,EntityId)`
  (view.rs:39); `world_pos`/`subs_holding`/`location` all resolve post-crossing (view.rs:171-243).
- Dual mode already supplies `known_shards={SHARD,DEST}` + books the DEST peer (bin/gateway.rs:80-84,
  lib.rs:764-766,780-784).
- GAP being closed: the committed `dual_cluster_crossing_smoke` gates only on the ADMIN DIRECTORY
  flip — NOT on client DEST-snapshot delivery. render_crossing_smoke is the client-observed gate on
  that already-wired leg.

## THE FATAL FLAW (must-fix #1) — RealmScene keys by EXTERIOR realm, drops to_realm
`RealmScene::from_boundaries` keys each box by `b.realm` and DROPS `to_realm`
(realm_scene.rs:115,123-131). An INWARD crossing re-homes to `b.to_realm` (stub.rs:2632-2634). For
the dot to re-home into System(8) the SOURCE trigger needs `realm=7,to_realm=8`; the SOURCE guard
requires `b.realm==hosted==7` (lib.rs:1271). So a single-sourced boxes.json keys box B under
System(7) → collides with box A (also 7) → `DuplicateRealm`, scene won't build; and `expected_box`
(exterior-keyed) can NEVER return System(8). **Single-sourcing is impossible for a crossing.**
→ The VISUAL scene (two realm boxes A@7,B@8) and the crossing TRIGGER (realm=7→to_realm=8) are
DIFFERENT artifacts and MUST be two files. Not a hack — the correct separation.

## The 6 folded must-fixes
1. **Two fixtures.** (i) shard `VD_REALM_BOUNDARIES` trigger `realm=7,to_realm=8`, Aabb/shell placed
   so the +X-walking dot crosses INWARD near +SEP; (ii) client `--realm-boxes` scene: box A@7 + box
   B@8 (guard-free client path, client.rs:114-119). In-test projection rebuilds box B@8 to match.
   Document that box-B geometry is kept identical across the two files by hand (single const).
2. **H1 = real second decode-client.** Stand up a 2nd `LoginClient`-style `DeliveredView` consumer
   (vd_io_prod::mesh) on the same gateway; call `crossing_was_real(&view, dot, source_sub, dest_sub)`
   at the overlap frame. Do NOT ship the proxy (snapshots_applied↑ + authoritative_sub flip) as the
   SOLE H1 — `set_authority` runs from the Control arm independent of any delivered DEST track
   (net.rs:211-213), so the proxy can go green on a dest-never-delivered regression.
3. **AFTER-gate independent of the broken key.** Gate on `DevState.location ==
   SystemSpace{system_seed:8}.label()` (DEST config.frame=SystemSpace{8}, stub.rs:1837 → location,
   net.rs:432-435) PLUS a distance-into-B margin from the CLIENT scene's System(8) box — NOT
   `expected_box` against a scene that cannot contain a to_realm box.
4. **In-test cluster from pub env builders.** Build from `shard_env`/`gateway_env`/`shard_b_env`
   (lib.rs:751-848) pointing VD_REALM_BOUNDARIES at the V4 trigger file — do NOT add a vd-devcluster
   launcher flag (that IS a bin change → user approval). Reproduce the C1 both-realms readiness gate
   (`realms_present` over [System(7),System(8)]).
5. **World-space motion assert.** `own_pos_after.x > own_pos_before.x` by a real margin
   (> SEP - BOX_B_HALF), sampled from the OWN DevState row, IN ADDITION to the pixel mirror-negative
   `dot_pixels_within_box_region(after, dot_after_region, box_a_region)==false`.
6. **GPU/QUIC local-only, fail-loud.** Keep await_listener GPU-precondition fail-loud; bound every
   poll with a named deadline (mirror 45s); name which sub-condition missed (location/pos/pixels/H1);
   no software-raster fallback. `just render-crossing-smoke` recipe, NOT in default gate.

## Assertions (the pixel-space crossing proof)
- H1 `crossing_was_real(view, dot, source_sub, dest_sub)==true` at overlap (2nd client).
- H2-BEFORE `dot_pixels_within_box_region(before, dot_before_region, box_a_region)==true`.
- H2-AFTER `dot_pixels_within_box_region(after, dot_after_region, box_b_region)==true`.
- H2-motion (mirror-negative) `...(after, dot_after_region, box_a_region)==false`.
- location flip `DevState.location: SystemSpace{7} → SystemSpace{8}`.
- world motion `own_pos_after.x - own_pos_before.x > SEP - BOX_B_HALF`.
- no-magenta on BOTH frames; RunManifest records both screenshots (HR6).

## Coverage (HR5)
NO new Tier-A code. All verdict/camera/scene primitives already 100%-covered in vd-client-harness/
vd-client unit tests. render_crossing_smoke is Tier-B process glue (coverage-exempt), cfg-gated
`dev-control,render` (zero tests under plain `cargo test`), LOCAL-only (GPU+real-QUIC). Same
exemption rationale as dual_cluster_crossing_smoke.rs:22-24.

## Open items to confirm during implementation
- box-B SEP + half-extents: disjoint screen rects for box_a_region/box_b_region yet both framed by
  fit_camera_to_scene (render_boxes_smoke uses 500m far box).
- DOT_WORLD_RADIUS brackets the billboard dot's actual pixel footprint (check vd-client-render).
- overlap-window: sample H1 tightly after the walk before SubscriptionClosing evicts the source sub.
