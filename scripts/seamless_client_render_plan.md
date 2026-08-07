# The Proper Seamless Client Render — vetted implementer plan

Vetted via the `seamless-client-render-design` workflow (Opus, 2026-07-29): 4 readers → lead design →
3 adversaries (seamlessness / 100K-scale / re-home-correctness+HR1) → synthesis. The adversarial pass
caught a load-bearing precondition the draft missed — the client's `world_pos` is flat/one-level, so
nested realms scatter; the recursive composition engine (Slice 0) must precede everything. All
`file:line` verified against HEAD. Runnable-first: each slice is flown in the window before it is gated
+ committed.

All major findings verified against real code:
- `world_pos` is a flat one-level identity map (view.rs:223-233; the comment itself admits deeper composition is "OWED, NOT free") — **Adversary 1 F1 CONFIRMED**.
- Up-relay iterates `simulates()` dots only (stub.rs:4917), `coarsen_level: 0` hardcoded "ladder up deeper ancestors is S3" (stub.rs:5054-5070); retained proxies never re-emitted upward — **A2 F2 / A3 F1 CONFIRMED (single-hop)**.
- Send key `cur = (entry.home, ids)` is ids-only; `ProxySceneSet` has `observer`+`realms` only, no poses (stub.rs:4878-4886) — **A2 F4 CONFIRMED**.
- `from_shapes`/`with_delta` skip `finite_extent > MAX_RENDERABLE_EXTENT_M` shells (realm_scene.rs:215-216, 255-256); `with_delta` needs an explicit `removed` list (realm_scene.rs:245-252) — **A1 F4 / A1 F2 / A3 F2 CONFIRMED**.

Here is the final synthesized plan.

---

# THE PROPER SEAMLESS CLIENT RENDER — final implementer-ready contract

Path note (adopted from Adversary 2): worldgen lives at `crates/core/src/worldgen.rs`, not `crates/sim/src/worldgen.rs`. All worldgen citations below use the correct path.

---

## PART 0 — ADVERSARY VERDICTS (each finding: CONFIRMED → fold, or REFUTED → why)

**Adversary "seamlessness-no-pop-no-vanish":**

- **F1 (BROKEN — flat `world_pos`) → CONFIRMED.** Verified: `DeliveredView::world_pos` (view.rs:222-238) returns `pose.pos` unchanged for `PlanetCentered | SystemSpace | GalaxySpace | AreaLocal | StationLocal`; only `ShipLocal` composes (one hull hop). Region `center` is authored parent-relative (worldgen.rs `epoch_offset_in_parent`), so a deep realm renders at its parent-relative offset as if absolute — the planet/moon render astronomically far from a city dot the instant nesting exceeds "parent-at-origin." B2's "already the right structure, additive, no wire change" is false. **FOLD: a new Slice 0 (recursive composition) precedes everything; it is the precondition for every later slice's correctness.**
- **F2 (re-stream merge removal uncomputable) → CONFIRMED** (same defect Adversary 3 F2 raises). `RealmRegistry` is own∪ancestors∪direct-children, never siblings (worldgen.rs `realm_neighbourhood_for`); siblings share the sticky map via a different lane (`RealmSceneDelta`); `with_delta` needs an explicit `removed` set. Diff-remove wipes siblings; add-only ghosts the old vantage. **FOLD into Slice 3 as the add/replace-only-spine + lane-owned-removals decision (below).**
- **F3 (reflected poses freeze on re-home via frame-counter re-stamp) → CONFIRMED.** The per-`RealmId` high-water gate (realm_view.rs:42-49,73) rests on "each RealmId authored by exactly ONE shard, each shard's counter from 0." Re-emitting a reflected pose under the home shard's counter breaks that on a re-home (counter discontinuity → `is_stale` drops every post-cross pose → freeze). **FOLD: the pose lane must carry the FA-6 `realm_fence` / preserve authoring identity; client gate adopts on fence advance.**
- **F4 (Galaxy/Universe not framed) → CONFIRMED.** Verified realm_scene.rs:211-217 / 255-257 skip `finite_extent > MAX_RENDERABLE_EXTENT_M` ambient shells. **FOLD: restate the ancestor-sky claim — finite ancestors (System, Planet) are framed; the ambient root is felt via the always-on starfield backdrop, not a Galaxy box.**
- **F5 (Slice-1 win is an illusion until F1) → CONFIRMED.** `camera_inside` is a point-in-box test over composed world boxes; those are mis-placed until Slice 0. **FOLD: reorder — Slice 0 first, then the cull-kill.**

**Adversary "scale-100k-bandwidth":**

- **F1 (S2c reflect is O(observers × siblings)/tick, not O(depth+children); Slice 4 turns idle send-on-change into per-tick fan-out) → CONFIRMED.** Verified the reflect-down iterates per retained account (stub.rs:4875) and the send gate keys on `(home, ids)` (stub.rs:4878). Moving siblings change every tick → send-every-tick → ~10 MB/s from one hub, uncompressed, precisely where `project_vu_streaming_contract` deferred quantize/delta to P5. **FOLD: correct the C5 cost model; make the dense-hub bandwidth bound a PRECONDITION of the pose lane; split the static-outline lane (send-on-change) from the per-tick pose lane; hard per-observer sibling cap + coarsen ladder (`coarsen_level`, intershard.rs:636, exists unused).**
- **F2 (ancestor recursion single-hop; moon-in-building undelivered) → CONFIRMED** (same as Adversary 3 F1). **FOLD: split the pose work — 4a single-hop (depth-2, the live-window repro), 4b multi-hop up-relay (depth≥3, the moon-in-building). D-RLM-13 stays open until 4b.**
- **F3 (galaxy-wide star-stubs = unbounded client hold) → CONFIRMED.** Client holds `BTreeMap<RealmId, RealmBox>` and composes per frame; "reachable" was never radius-bounded; `generate_starfield` (client-render/src/lib.rs:85-95,442) already satisfies "stars always visible" for far systems. **FOLD: bound the streamed-stub tier to a coarse finite warp-AoI radius; beyond it stays the starfield backdrop, never a held box.**
- **F4 (send-on-change key not addressed) → CONFIRMED.** Appending `poses` without splitting the lane either freezes (key unchanged) or re-ships the full outline `Vec<RealmShape>` every tick (key made per-tick). **FOLD into Slice 4a: two lanes, distinct send policies.**

**Adversary "rehome-correctness-hr1-frame":**

- **F1 (Slice 4 delivers occupied+siblings but NOT ancestors; planet-you-stand-on freezes/detaches at depth≥3; D-RLM-13 mis-closed) → CONFIRMED.** `authored_realm_snaps` is a shard's moving DIRECT children only (stub.rs:768-776); a shard cannot author its own realm's pose; the up-relay is single-hop. **FOLD: 4b builds the multi-hop chain; D-RLM-13 flips green only after 4b's depth≥3 e2e.**
- **F2 (re-stream merge uncomputable; `RealmBox` has no lane provenance) → CONFIRMED.** Verified `RealmBox` = `{shape, frame, center_offset, parent, depth, color_rgba}` (realm_scene.rs:51-69) — no provenance. **FOLD: Slice-3 decision below (add/replace-only spine + lane-owned removals + explicit spine-removal; optional provenance tag).**
- **F3 (re-fire anchored to login-only SessionAttached seam that lacks the new home RealmId on re-home) → CONFIRMED.** A re-home re-points authority via the CommitAuthority/SubscriptionReady path, which names entity+frame+`realm_fence`, not a home `RealmId`. **FOLD: thread the new home RealmId into the re-home commit seam as an explicit step (in Slice 2, prereq for Slice 3).**

**No finding REFUTED.** What all three adversaries independently ruled SOUND and is therefore kept unchanged: the HR1 "each shard ships only self-authored children, client composes" principle (not prediction); the kill-the-cull isolation (inner-shell material touches only the material knob, not the `to_render_prims`→vertex seam, so P4 voxel surfaces are not foreclosed); the append-only `ProxySceneSet` field + PROTO_MINOR discipline; the per-`RealmId` freeze guard for the single-author case.

---

## PART 1 — PLAIN-LANGUAGE SUMMARY (for the human)

Today the window shows you one realm at a time. Fly into a planet and the planet's shell, its moon, and its star all blink out; fly back out and they never return. Three real, located defects cause it — and the adversarial review found a fourth, deeper one hiding under the plan's own optimism.

The deep one: **the client cannot actually place a realm that sits inside another realm.** Its position math is "one level flat" — it treats every realm's position as if measured from the centre of the whole universe, when in fact a city's position is measured from its planet, and the planet's from its star. So the moment anything nests more than one level deep, the pieces of the world scatter astronomically far apart. The plan assumed this stitching was already built and free. It is not built. So the very first thing we do — before touching the vanishing bug — is give the client a proper nested position engine: to place any object, walk up its chain of containers (room inside building inside planet inside star) and add each step, exactly the way any real 3-D engine does. This is not guesswork or prediction: the client only ever adds up positions the servers already sent; it invents no motion of its own.

With that engine in place, the rest becomes the consistent idea the goal demands: **the client draws the whole nested stack at once** — the room you are in, the building around it, the planet under it, the moon beside it, the star above it, the distant stars beyond — and keeps every one because *someone always narrates each one*. We honour the sealed-shard rule: no server ever ships a realm it does not own. Each server ships only its own piece (its children's live positions); the client stitches the pieces. The moon reaches you because the planet's parent narrates it and relays it down the same pipe that already relays your immediate neighbours. The realm you stand inside stops being thrown away and instead is drawn as a shell you look *out of* — the same shell that at P4 becomes the ground under your feet. And crossing a boundary *adds* your new surroundings to what you already hold, never wipes — so nothing you left flickers away and nothing pops in late.

Two honest scope corrections the review forced. First, keeping the moon alive when you go *two or more levels deep* (into a city, into a ship inside a station) needs a relay that hops up more than one container — that machinery does not exist yet, so it is its own explicit slice, and we do not declare the "stale sky" bug fixed until that slice is proven at depth. Second, at a busy hub — a thousand players around one star — narrating every moving neighbour to everyone, every tick, is a real bandwidth wall; so the moving-position feed ships with a hard per-player neighbour cap and a "coarser detail the further up you look" ladder, and we prove the hub stays within budget *before* that feed goes live, not after.

The result, flown in the window: stand on the planet and the moon hangs in the sky; walk into the city and the planet, moon, and stars are all still there and still moving; undock and fly out and the planet shrinks behind you while the next star grows ahead — one continuous camera, no loading, nothing vanishing. Shipped in six flyable slices, each visibly better, each flown by you before it is gated and committed.

---

## PART 2 — THE THREE DECISIONS, AS REVISED (one line + rationale each)

**RENDER MODEL — the client holds a `RealmId`-keyed transform tree and composes world position by walking `parent` root-to-leaf, chaining each hop's static `RealmShape.center` or live `RealmSnap.pose`; it draws the full union (occupied ∪ finite ancestors ∪ AoI children ∪ AoI siblings ∪ bounded star-stubs), culling NOTHING by container-inside-ness or distance.**
Rationale: the flat `world_pos` (view.rs:222-238) is the actual defect — the nested walk is the load-bearing new engine, not "additive," so it leads the arc; the outermost ambient shells stay unframed (realm_scene.rs:215-216) and the "sky" above the finite ancestors is the always-on starfield, not a Galaxy box.

**POSE FEED — every visible level is authored by exactly one shard (its parent) in its own frame and shipped; the client composes; live poses of the occupied realm, its ancestors, and its siblings ride a split S2c lane (static outlines send-on-change; moving poses per-tick, per-observer-capped, coarsened up the ladder) that carries the FA-6 `realm_fence` so authority hands off without a stale-freeze.**
Rationale: HR1 holds (no shard ships a non-owned pose); but the single-hop up-relay (stub.rs:4917, `coarsen_level:0`) means depth≥3 ancestors need a real multi-hop relay (4b), and re-stamping reflected poses under the home shard's counter would re-break the per-`RealmId` freeze guard (realm_view.rs:42-49) unless the `realm_fence` rides along.

**RE-STREAM (VU-6) — on the ambient-realm-change edge, re-fire the neighbourhood registry threaded with the NEW home `RealmId`; the client applies it as ADD/REPLACE-ONLY over the own+ancestor SPINE, never removing anything; sibling/child removals stay owned by their own lanes (S2c empty-set reconcile; `RealmSceneDelta` release), and an explicit spine-removal signal drops ancestors that fall off the chain.**
Rationale: the registry carries no siblings and `RealmBox` carries no lane provenance (realm_scene.rs:51-69), so a registry-driven "diff and remove" is a vanish-or-ghost dilemma — eviction ownership must be explicit and per-lane, and the re-fire seam must first be handed the new home id (it currently is not, gateway login-only at :2805).

---

## PART 3 — THE RUNNABLE-FIRST GATED SLICE PLAN

Every slice: **build → boots → user flies it in the window → gate (client Tier-A 100% region+branch + a capture/e2e) → user confirms → commit.** New helpers (`world_pos` walk, `camera_inside`, the merge classifier) are monomorphic branchless helpers with every arm covered in client Tier-A unit tests (HR5 generic-code gotcha). Ordered so each is flyable and strictly better than the last.

### Slice 0 — The recursive nested-position engine (client-only, no wire) — THE PRECONDITION
- **Change:** rewrite `DeliveredView::world_pos` (view.rs:222-238) from the one-level identity into a recursive walk over a client-held `RealmId → (frame, parent, live-pose|static-center)` map, chaining each hop's relative placement with the proven `transfer_frame` math (frame.rs:94-155, chain round-trip frame.rs:426-460); `RenderSnapshot::world_pos` (render_snapshot.rs:105-108) delegates unchanged; `sync_realm_boxes` (client-render/src/lib.rs:678-720,698) places each box through the walk. The map is populated from the boxes already in `RealmScene` (their `parent` links, realm_scene.rs:51-69) + overlaid live poses from `RealmView`.
- **Flyable win:** in the *current* forest and any hand-authored depth-≥2 fixture, nested realms (a station box under a planet box under a system) finally render *nested* — the child sits inside its parent instead of scattering to the parent's absolute offset. Nothing vanishes yet (that is Slice 1+); this is the geometry becoming correct.
- **Gate:** client Tier-A 100% — every match arm of the recursive walk covered monomorphically, plus a cycle/over-deep guard test; a composition round-trip unit test asserting the composed world pose equals the `transfer_frame` chain for a 3-level fixture; a wgpu-readback capture asserting a depth-2 child box renders *inside* its parent's world box.
- **Contract cites:** view.rs:222-238; render_snapshot.rs:105-108; frame.rs:94-155,426-460; client-render/src/lib.rs:678-720,698; realm_scene.rs:51-69.
- **Wire:** none. Byte-identity inert at walk/static.

### Slice 1 — Stop culling the container; finite ancestors render as sky (client-only, no wire)
- **Change:** add an inside-shell material variant in `spawn_realm_box` (client-render/src/lib.rs:768-777, the `cull_mode: Some(Face::Back)` at :775) selected by a `camera_inside(realm)` point-in-box predicate composed in `sync_realm_boxes` (:678-720) from the Slice-0 world boxes; keep the outward `Face::Back` translucent material for realms the camera is outside. The inside variant is front-face-only at a much lower alpha (a boundary you see out of), avoiding the tint-wash the cull was protecting against.
- **Scope correction (Adversary 1 F4):** the "sky" above is the *finite* ancestor chain (System, Planet) plus the always-on starfield backdrop (client-render/src/lib.rs:85-95) — NOT a Galaxy/Universe box, which stays unframed (realm_scene.rs:215-216).
- **Flyable win:** enter the planet/station — the shell you are inside is now visible as a boundary you see out of, and its finite ancestors + the starfield render as the sky. (Siblings still vanish → Slice 2; the sky is still frozen if the ancestor moves → Slice 4.)
- **Gate:** client Tier-A 100% (`camera_inside` is a branchless monomorphic helper, both arms + boundary-equality covered); a wgpu-readback capture asserting non-black inner-shell pixels when the camera is inside a realm box and unchanged outward rendering when outside.
- **Contract cites:** client-render/src/lib.rs:768-777,678-720,85-95; realm_scene.rs:215-216,650-653.
- **Wire:** none. Byte-identity inert.

### Slice 2 — Deliver S2c siblings (outlines) to the live window + thread the new-home id into the re-home seam
- **Change (a):** wire the already-sim-proven `ProxySceneSet` → `RealmSceneDelta` path (stub.rs:5108-5155, gateway forward at gateway.rs:2877-2895, minor≥6) to actually deliver in the real process; the sole existing e2e stops at the home shard (stub.rs:3407-3423 "cross-PROCESS e2e with a real client is deferred") — this slice closes that. Outlines only (static sibling boxes); moving poses arrive in Slice 4.
- **Change (b), the Adversary-3-F3 prerequisite:** thread the new home `RealmId` into the re-home commit seam (the CommitAuthority/SubscriptionReady path, gateway.rs:2816+) so Slice 3's re-fire has a home id to compute `realm_registry_for_home` (gateway.rs:1743) — the login `SessionAttached` seam (gateway.rs:2805) does not carry it on a re-home. No re-fire yet; this only plumbs the id and asserts it lands.
- **Flyable win:** stand on the planet, enter the station — the moon and orbiting stations (siblings) stay drawn as static boxes instead of vanishing.
- **Gate:** a cross-process e2e (`vdctl` harness) asserting a sibling `RealmId` remains in the published scene across a cross-in; a unit/e2e assertion that the re-home commit carries the new home `RealmId`; client Tier-A 100%.
- **Contract cites:** stub.rs:4868-4890,5108-5155,3407-3423; intershard.rs:646-651; gateway.rs:2877-2895,2816+,1743,2805.
- **Wire:** none new (uses the existing `ProxySceneSet`/`RealmSceneDelta`); internal re-home-seam plumbing.

### Slice 3 — The re-home re-stream (VU-6): what you left returns; add/replace-only spine, lane-owned removals
- **Change (a):** on the ambient-realm-change edge, re-fire `maybe_announce_realm_registry` (gateway.rs:1763) using the Slice-2 new home id (the second, keyed re-fire; VU-6); gate `cfg.armed && negotiated_minor >= 5 && home_rid.is_some()` unchanged.
- **Change (b), the Adversary-1-F2 / Adversary-3-F2 fix:** the client reconciles a re-streamed registry as **ADD/REPLACE-ONLY over the own+ancestor spine** — it never synthesizes a `removed` set from the registry. Route it through a new `with_registry_restream` classifier beside `with_delta` (realm_scene.rs:245-278): first registry = `from_shapes` (net.rs:239-244); subsequent = merge that (i) adds/replaces spine realms present in the new registry, (ii) removes ONLY spine realms named by an explicit spine-removal signal for ancestors that fell off the chain, (iii) touches no sibling/child realm. Sibling removals self-heal via the existing S2c empty-set reconcile (stub.rs:5136-5153); child removals via `RealmSceneDelta` release. Optionally tag `RealmBox` with a provenance class (spine|sibling|child) to make the "touch no non-spine" bound explicit and testable.
- **Flyable win:** cross out of the station back onto the planet, then off the planet into open system — the realms you left return, the new vantage's neighbourhood is set up, siblings/children are neither wiped nor ghosted, and no `RealmId` common to both neighbourhoods churns (no flicker/double-draw).
- **Gate:** an e2e asserting, after a re-home, the scene holds {new own ∪ finite ancestors ∪ children} AND retains still-in-AoI left-behind realms AND drops out-of-AoI ancestors via the spine-removal path, with zero `RealmId` churn for common realms and zero sibling wipe; client Tier-A 100% (both merge branches + the empty spine-removal case covered).
- **Contract cites:** gateway.rs:1763,2804-2811; net.rs:239-244; realm_scene.rs:51-69,202,245-278; stub.rs:5136-5153.
- **Wire:** the explicit spine-removal signal — an appended field/variant on the registry announce (PROTO_MINOR bump, append-only, old-client-skip-by-length test).

### Slice 4a — Live poses for the occupied realm + moving siblings (depth-2 pose lane, split, capped, fenced)
- **Change:** append `poses: Vec<RealmSnap>` to `ProxySceneSet` (intershard.rs:646; `RealmSnap` exists channels.rs:242), PROTO_MINOR bump. **Split the lane (Adversary-2 F4):** the static-outline set stays send-on-change (key `(home, ids)`, stub.rs:4878); the moving-pose set ships per-tick on its own gate. **Cap it (Adversary-2 F1):** a hard per-observer sibling-pose count cap + a coarsen ladder driven by the existing unused `coarsen_level` (intershard.rs:636), so a dense hub cannot fan out unbounded; pull the P5 pose quantize/delta forward for THIS lane. The parent fills `poses` from its own `authored_realm_snaps` (stub.rs:759-776) — self-authored, parent-frame (HR1-clean). **Fence it (Adversary-1 F3):** the reflected poses carry the FA-6 `realm_fence`; the home forwards onto the client `RealmSnapshot` lane; `RealmView` (realm_view.rs:68) overlays agnostically but the per-`RealmId` high-water (realm_view.rs:73) ADOPTS on a fence advance rather than dropping the counter discontinuity at a re-home.
- **Flyable win:** the demand-cluster live-window repro (System 7 → {Planet, Station}, depth-2) is fully alive — the planet you stand on visibly orbits, a sibling moon moves, the sky is not frozen; crossing in/out no longer freezes the neighbour.
- **Gate — bandwidth bound is a PRECONDITION, not the finale (Adversary-2 F1):** a dense-hub load test (N≥1000 observers around one system) asserting the split-lane + cap keeps egress within a named budget; an e2e asserting a moving sibling `RealmId` receives per-tick updates AND survives a re-home without freeze (the fence-adopt path); a frame-correctness round-trip (composed world pose = `transfer_frame` chain); client Tier-A 100% + an append-only wire test (old client skips `poses` by length).
- **Contract cites:** intershard.rs:646,636; channels.rs:242; stub.rs:759-776,4868-4890,4878; realm_view.rs:68,73; frame.rs:426-460.
- **Wire:** `ProxySceneSet.poses` appended (PROTO_MINOR bump) + the `realm_fence` on the reflected pose.

### Slice 4b — Multi-hop ancestor poses (depth≥3: the moon-does-not-vanish-in-the-building)
- **Change (Adversary-2 F2 / Adversary-3 F1):** build the recursion the single-hop relay lacks — (i) re-emit retained proxy occupants UPWARD so the up-relay walks the full ancestor chain (extend stub.rs:4914-4929, which today filters `simulates()`-only, to also relay `RetainedOccupants` up to the grandparent, honouring the `coarsen_level` ladder that stub.rs:5054-5055 explicitly defers); (ii) each ancestor reflects its own direct child (the next-lower ancestor) DOWN toward home along the Slice-4a pose lane. Same cap + fence discipline as 4a.
- **Flyable win:** the headline scenario — stand in a *city* on a planet in a system, and the planet visibly orbits its star, the moon (a sibling of the city, two levels up) hangs and moves in the sky, the star is alive; walk into a ship interior inside a station and none of the outer shells freeze or drop.
- **Gate:** a cross-process depth-≥3 e2e (System→Planet→City→player) asserting the planet AND the moon receive live per-tick poses from inside the city, survive a re-home without freeze, and the multi-hop fan-out stays within the coarsen-ladder budget at a hub; client Tier-A 100%.
- **Contract cites:** stub.rs:4914-4929,5054-5073,5081-5106,309; intershard.rs:636,646.
- **Wire:** none new beyond 4a (the ladder rides the same `poses`+`coarsen_level`).

### Slice 5 — Bounded star-stub tier + prefetch + the no-pop warp capstone (VU-7)
- **Change:** ship reachable-system stubs as static `RealmShape.center` within a **coarse finite warp-AoI radius (Adversary-2 F3)** — a large but bounded reachable-system count; everything beyond stays the existing `generate_starfield` backdrop (client-render/src/lib.rs:85-95,442), never a held `RealmBox`. Split the STUB tier (always-on, capped) from the AoI-scoped DETAIL tier (existing `InterestConfig` factors, worldgen.rs:930-944); verify one predicate filters both registry membership and pose feed; prefetch warms before the SOI (`spin_up_factor > 1`, worldgen.rs:930). Galaxy-scale distance is the planted non-zero-`cell` `LatticePos` (pose.rs:132-147), additive.
- **Flyable win:** the full warp fly-by — fly a system, warp anywhere, the departed system shrinks to a dot (still narrated by the parent), sibling stubs parallax, the destination forest streams from its stub into full detail before you arrive, no pop, stars always visible.
- **Gate:** the VU-7 no-pop capture (no blank/flicker frame across an SOI crossing, no leaked tracks); a stub-count-cap bandwidth test asserting the held box count stays bounded regardless of galaxy size; client Tier-A 100%.
- **Contract cites:** worldgen.rs:383,930-944; pose.rs:132-147; realm_scene.rs:75; client-render/src/lib.rs:85-95,442; stub.rs:4799-4867.
- **Wire:** the non-zero `LatticePos.cell` realization (additive, already planted).

---

## PART 4 — DEFERRALS + DEFERRED.md LEDGER

**D-RLM-13 (stale-render-on-authority-change): STAYS OPEN through 4a; flips 🟩 only after Slice 4b's depth≥3 e2e.**
Rationale (Adversary 2 F2 / Adversary 3 F1, CONFIRMED): the staleness has three roots — no live-pose narrator for the occupied realm/ancestors, no re-stream on the ambient change, AND no multi-hop delivery for ancestors ≥2 levels up. Slice 3 delivers the re-stream (VU-6); Slice 4a delivers depth-2 live poses (closes the live-window 2-level repro); Slice 4b delivers the multi-hop ancestor chain. Closing D-RLM-13 at 4a (as the pre-review draft did) would falsely claim the depth≥3 case the goal explicitly names ("the moon does not disappear when you enter the building"). **Ledger:** `D-RLM-13 — WHAT: live pose + re-stream + multi-hop ancestor delivery on authority change; WHERE: gateway re-fire (VU-6) + ProxySceneSet.poses + multi-hop up-relay (stub.rs:4914-4929); WHEN: this arc Slices 3→4b; flips 🟩 when the depth-≥3 cross-process e2e (Slice 4b) asserts live ancestor pose post-re-home with no freeze.`

**New ledger entries owed:**
- `D-VU-COMPOSE — recursive nested world-position engine: WHERE view.rs:222 (one-level→recursive walk); WHEN Slice 0; 🟩 on the 3-level composition round-trip + nested-render capture. (Precondition for the whole arc.)`
- `D-VU-6 — re-home re-stream, add/replace-only spine + lane-owned removals + new-home-id threaded into the re-home seam: WHERE gateway.rs:1763 re-fire + gateway.rs:2816+ id-threading + net.rs:239 merge classifier + explicit spine-removal signal; WHEN Slices 2(b)+3; 🟩 on the Slice-3 no-wipe/no-ghost e2e.`
- `D-VU-SIB-POSE — split-lane, capped, fenced live sibling/occupied pose lane: WHERE intershard.rs:646 (poses append) + intershard.rs:636 (coarsen ladder) + realm_view.rs:73 (fence-adopt); WHEN Slice 4a; PROTO_MINOR bump; 🟩 on the append-only wire test + dense-hub bandwidth bound + re-home-no-freeze e2e.`
- `D-VU-ANCESTOR-MULTIHOP — multi-hop up-relay + per-level ancestor reflect (the coarsen ladder stub.rs:5054 defers): WHERE stub.rs:4914-4929,5054-5073; WHEN Slice 4b; 🟧 until the depth-≥3 e2e.`
- `D-VU-STUB — bounded galaxy-wide star-stub tier (coarse warp-AoI radius cap) + non-zero LatticePos cell: WHERE worldgen.rs:383, pose.rs:132, client-render/src/lib.rs:85; WHEN Slice 5 (box) / P10 (true ly-scale); 🟧.`

**Deferred beyond this arc (not foreclosed):**
- P4 voxel surface inside the same shell (Slice-1 inner shell → seed-derived ground; the `to_render_prims`→vertex seam realm_scene.rs:557 is untouched by the material knob).
- P5 pose quantize/delta as the general compression (this arc pulls it forward for the sibling pose lane only; the galaxy-wide general case stays P5, `project_vu_streaming_contract`).
- The dense-ambient spatial index for thousands of sibling systems (D-45/P6, worldgen.rs:464-469).
- Ship-as-realm + the P9 Signal bus reuse the same author-and-ship lanes (P8/P9) — the `poses`/`coarsen_level` lane and the compose walk are the substrate.
- The inter-tier mm↔AU rebase in the composition (D-41, pose.rs:139-147) lands at P4/P10 additively — the Slice-0 walk composes render-local `DVec3` today; the `cell`-rebase drops in at the same chokepoint.
- D-35 version-matched (not just time-coherent) hull/interior composition at P8 (view.rs:214-220) — orthogonal, the one-level ShipLocal note already tracks it.

**Where a naive implementation still breaks the rules (kept as guardrails):** broadcasting the whole tree from one shard (HR1/frame-authority violation — never; each ships only its own children); `from_shapes` on a re-stream (wipes what you left — must be add/replace-only spine merge); flipping `cull_mode` alone (tint-wash — needs the distinct inner-shell material + client-local point-in-box selector); the client re-deriving any position (prediction, banned — it composes shipped frames only); a trailing struct field on a frozen wire (append + PROTO_MINOR + skip-by-length test only); AoI-culling star-stubs or distance-dimming any held realm (violates stars-always-visible — AoI membership is the only omission); per-shard fan-out that grows with total galaxy size (100K break — split lanes, per-observer cap, coarsen ladder, bandwidth bound as a PRECONDITION of the pose lane); re-stamping reflected poses under the home counter without the `realm_fence` (resurrects the cross-shard freeze — fence must ride along).