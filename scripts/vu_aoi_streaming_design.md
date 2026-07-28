# VU — Per-Level-Cull AoI Streaming (design + gated slice plan)

Vetted via the `vu-aoi-streaming-design` workflow (4 parallel maps → lead design → adversarial
verify → synthesis, all Opus, 2026-07-28). Realizes the user-agreed premise: **render-AoI is the
union of per-level culls up the containment chain; the occupant's position relays UP; each realm
culls its OWN children; siblings arrive from the shared parent with no special-case.**

Builds ON the just-landed login-only shape stream (VU S1–S3: `RealmRegistry` + `RealmShape` +
`from_shapes`, PROTO_MINOR 5).

---

## 1. Architecture (adversarially hardened — must-fixes folded, marked `[MAJOR-n]`)

- **Each realm's SHARD culls its OWN children** — reuses the existing size-proportional interest band
  (`AoiConfig::in_range`) against the observer's position in *its own* frame; never replicates a
  sibling's/parent's state. Author-and-ship, never replicate-derive.
- **The chain-UNION is emergent on the CLIENT, keyed by `RealmId`** (`RealmScene` is already a
  `BTreeMap`). Every ancestor shard independently streams "*this observer's* near children among *my*
  children" as add/remove deltas; the client merges by key. No central assembler → O(players ×
  neighbourhood), no O(all-realms) bottleneck.
- **THE new mechanism — occupant position relays UP the chain.** Positions currently flow down (parents
  author children's placements); this adds the upward counterpart child→parent→…→root so a sealed-off
  ancestor can cull its own children (incl. the observer's siblings) against where the observer is.
- **`[MAJOR-2]` The relay is a DEDICATED infra arm, NOT the reserved generic `Signal`** — leave
  `InterShardFlow::Signal` reserved for the P9 `{name,value,signature,frequency}` gameplay bus.
  *(Open product decision — see §4 Q1.)*
- **`[MAJOR-3]` The relay carries NO fence** — `FireAndForget` + `Unreliable` (a 20 Hz latest-wins
  datagram on a new `MsgClass::SignalDelta`). A parent authorizes any spin-up it emits with its OWN
  realm fence; a fence on the relay would make it authority-gating → force reliable → break the scale
  premise.
- **`[MAJOR-1]` The parent RETAINS each proxy occupant with a last-known TTL** — folds the *retained*
  set (not the raw per-tick datagram) into its cull. A dropped datagram holds the last-known occupant
  for N ticks → stops render flicker + sibling spin-thrash on the shed-first lane; also supplies the
  vanished-observer eviction for the re-keyed membership.
- **`aoi_decide` becomes per-observer with zero forked math** — re-key membership
  `RealmPath → AoiState` ⇒ `(ObserverId, RealmPath) → AoiState`; per-observer acquire/grace latch
  (velocity-widening + warm-ahead unchanged). One band serves BOTH lifecycle demand (OR into the
  existing `RealmDemand` union → closes the sealed-parent sibling-warm gap) and render (per-(observer,
  child) breakdown drives the stream).
- **Shapes stream incrementally** — append `ServerControlMsg::RealmSceneDelta{added, removed}`
  (reliable Control, PROTO_MINOR 5→6). `RealmRegistry` stays the cold-start / warp re-anchor snapshot
  (it alone carries `root`). Unreliable pose datagram unchanged: "a static shape once, a pose per tick."
- **`[MINOR-1]` `apply_delta` recomputes depth over the inserted boxes' descendant closure** —
  independent per-shard reliable streams have no cross-stream ordering, so a deep child can arrive
  before its parent; frames resolve from the parent *link* (safe), only translucent-layering depth
  needs the recompute.
- **Login / walk / warp all collapse to "AoI membership changed"** — login seeds membership + emits
  adds; walk = in_range transitions per level; warp = velocity-widened bands fire earlier (warm-ahead),
  root-crossing re-anchors via a fresh `RealmRegistry` then deltas resume.
- **Byte-identical at walk/static** — band inert (`InterestConfig::inert`) → cull + up-flow emit
  nothing; `RealmSceneDelta` only on a real change and only `armed && negotiated_minor >= 6`; all
  changes are appended enum variants + one appended `MsgClass`, never a trailing struct field.
- **100K scale (cull → compress → budget)** — per-shard distributed `(ObserverId, RealmPath)` state
  only for observers whose band the shard is in; RealmShape/RealmSnap already the render subset; reuse
  MTU chunking + shed-loud on the unreliable lanes. Galaxy free-fly: observer is a local dot on the
  galaxy shard (most contribution, no relay). District sibling: observer a dot deep in a district
  (least contribution), relays up → shared city folds the proxy → its band catches the neighbour →
  streams it. Same rule at every depth.

---

## 2. Gated slice plan

Gate each: `just gate` (fmt + clippy -D warnings + tests + 100% Tier-A coverage), commit on request.
Every slice byte-identical at walk/static, behind `armed && minor` gates.

- **S0 — Per-observer cull, lifecycle output unchanged** *(vd-sim)*. Re-key `aoi_decide` to
  per-`(ObserverId, RealmPath)`; drop the global `aoi_min_dist` reduction to a per-observer loop; add
  the lazy per-observer eviction (TTL skeleton, empty until S2b). Fold per-observer results into the
  existing `RealmDemand` union. No wire, no render output. Test: single-observer demand byte-identical;
  two observers latch independently (union = old global); A-leaves-as-B-arrives no thrash. `[MINOR-2]`
  verify no *armed multi-observer* regression fixture.
- **S1 — Single-level render-delta stream (galaxy free-fly, end-to-end)** *(wire + client + sim +
  connection-plane)*. Append `RealmSceneDelta`, PROTO_MINOR 5→6; `RealmScene::apply_delta` (+ descendant
  depth recompute) + `net.rs` arm; `ShardToGateway::RealmSceneDelta{observer, added, removed}`; shard
  emits per-observer adds/removes for its OWN in-band children; gateway routes gated `armed && minor≥6`,
  non-empty only. Test: in-proc demand cluster — child enters band → box appears, leaves → removed after
  grace; empty-vec byte-identity + minor<6 withhold. Ships the galaxy free-fly case whole.
- **S2a — The up-flow arm (plumbing, no reception effect)** *(wire + sim + node)*. Append the dedicated
  relay arm (fields: `observer`, `to_realm`, `occupant: StampedPose`, `coarsen_level`; NO fence) +
  `MsgClass::SignalDelta`; classify `FireAndForget`/`Unreliable` (compile-forced). Child resolves parent
  `RealmCoord → NodeId` via directory `HeadRead` (resolve-once, cache, re-resolve on re-home), emits per
  observer; parent decodes and drops. Test: round-trip + closed-taxonomy marker; two-shard emit→decode;
  byte-identity inert; directory resolve unit.
- **S2b — Parent retains + folds the proxy → the DISTRICT sibling case ships** *(sim + connection-plane)*.
  Parent stores each occupant in the last-known-TTL map `[MAJOR-1]`, re-expresses into its frame (compose
  with child placement, additive at IdentityFrames), folds the retained set into `aoi_decide` → sibling
  cull → S1's delta routes it; same proxy makes the parent emit `SpinUp` for the sibling. Test: two-shard
  in-proc — O deep in district, client gains the sibling district, orch gets a SpinUp; loss test (drop
  K<TTL → holds; drop >TTL → clean single removal).
- **S3 — Coarsening ladder + recursion + depth bound + dilation stamp** *(sim + wire)*. Multi-hop re-emit
  (`coarsen_level += 1`); full (pos,vel) for the deepest ~1–2 ancestors, lineage-only above the cull
  horizon; depth-bound ~5–6; cadence throttle by level; thread the `StampedPose` so an ancestor's
  predictive horizon uses its own clock. Realizes "least when deep / most when shallow."
- **S4 — Warp re-anchor (VU-6)** *(connection-plane + client)* — **PARTIAL, real cross-root frame rebase
  is P4/P5**. On a root-shifting warp re-emit a fresh `RealmRegistry`, deltas resume. Box-level now;
  exact LatticePos rebase deferred (DEFERRED.md VU-6).
- **S6 — Distinct `render_aoi` band preset** *(vd-core)* — **DEFER** unless a visual horizon must diverge
  from the shard-warm horizon; if needed reuse `InterestConfig`/`AoiConfig` (HR3, never fork), a second
  baked band per region. v1 uses the one `aoi` band.

**Deferred (P4/P5 / later):** real `FrameContext`/`LatticePos` cross-frame rebase (S3/S4 land the additive
IdentityFrames version now); TLV signal-bag extension of the relay (fixed struct fine for v1); client
session-resume re-baseline of distributed per-observer membership (D-11); aggregate-relay-volume load test
at 100K (reuse the SPIKE-3a latency-gate pattern, with S3 or just before P4).

---

## 3. Open product decisions (the human owns these)

1. **`[MAJOR-2]` Relay channel** — dedicated infra arm (recommended; leaves the generic Signal bus free
   for P9) vs adopt the generic `{name,value,signature,frequency}` Signal envelope now as its first
   consumer. Touches the emphatic Signal-vision law.
2. **Observer identity on the wire** — durable player id (recommended; survives re-home/reconnect, needed
   for session-resume) vs `SessionId` (simpler, changes on reconnect).
3. **RealmRegistry (login snapshot) scope** — narrow to ancestor-context only (recommended; one owner per
   renderable box, no stuck-box) vs keep ancestors ∪ home-children (idempotent double-send).
4. **Render band = lifecycle band, or separate draw distance?** One band (default: render when the shard
   warms) vs a longer visual horizon (S6 + more streamed content + budget).
5. **Precision-ladder cutover depth** — how many ancestor hops get full (pos,vel) before lineage-only
   (1? 2?). Trades sibling-cull fidelity vs up-flow bandwidth.
6. **`[MINOR-3]` Off-chain interior visibility** — a sibling arrives as a box (seed geometry); its
   interior doesn't stream until you cross in. Correct for a room/station; a boundary for an open-world
   surface subdivision. Confirm this is the intended product boundary.
