# Step 5 — SL7 liveness/interest, and the deletion of both occupant relay lanes

**OWNER-DECIDED 2026-08-12 — §8 answers:** (1) all three wire changes APPROVED (ChildLive,
ChildSceneSet, GhostFlow::SpawnV2; the four pose-carrying arms tombstone). (2) The avatar loss is
ACCEPTED; the lawful remote-avatar lane is ledgered in DEFERRED.md as owed, owner-gated, dependent
on the server-authoritative frame-composition rework. (3) The crossing window is MEASURE-FIRST:
the retained-ghost fill is the only continuity; the leaver's blink window is measured in the
crossing e2e against the 100–150 ms interpolation buffer; one-feed-wins lands only if that bound
fails. With these answers the slice plan of §5 is unblocked (slice E's product gate is
pre-answered by decision 2; its landing still re-confirms with the owner per §5).

DRAFT ACCEPTED FOR IMPLEMENTATION, 2026-08-12.

**IMPLEMENTATION BEGAN 2026-08-13 — slice A landed, plus one owner-approved ADDITION.** The flown
tree exposed the missing half of observation: re-home works both ways, but "when I exit the system
its planets freeze" — no lane carried a realm's interior motion UP to an observer standing in its
parent. Owner approved (SL6, 2026-08-13) the UP-OBSERVATION lane: `InterShardFlow::RealmObservation`
— a live child ships the rows IT authors (its own frame, the identical bytes its occupants get) one
hop up; the parent adds the ONE placement it authors and re-fans to its observers, and relays the
restated rows one hop further up (recursion; visibility culls with distance). NO occupant data
crosses; the author's `frame_id` stays sealed at every relay (the down-cascade's discipline).
Landed with it: `InterShardFlow::ChildLive` (slice A's bit: producer at the AoI tail, presence-is-
the-bit, `(fence, at)` last-wins, TTL = the retained-proxy knob) and the `ChildLiveness` store.
PROTO_MINOR → 10. (Slices B+C landed together 2026-08-13 — see the block further down.)
**Gated:** `exiting_the_system_keeps_its_planets_orbiting` (rlm_demand_login, full orbit speed) —
exit past the shell, park in the galaxy, the feed must climb 20 frames and a drawn planet must
move; first run: feed 134→168, best motion 5.04 m.

**OWNER FLIGHT 2026-08-13 (second): everything works — re-home both ways, exited-system planets
keep orbiting. ONE gap found: approaching a NEIGHBOUR star system, its planets do not appear even
inside AoI/visibility.** Root cause, code-confirmed: the SCENE lane (`aoi_decide`'s `dot_visible`)
ships only a shard's DIRECT children's outlines — a galaxy-standing observer gets the systems'
boxes, never a neighbour system's interior. The up-observation ROWS for that interior flow (that is
why the exited system's planets, whose boxes pre-existed in the client scene, keep moving), but a
never-visited system's planets have no scene boxes, and the client draws motion only onto scene
boxes. **CURE = SLICE C's FIRST DELIVERABLE: the shape-lane mirror of `RealmObservation`** — a live
child ships its interior OUTLINES (RealmShape set, centers in its own frame — SL3: its interior view
is its look) one hop up; the parent restates centers by adding its authored placement and folds them
into its observers' scene deltas, recursively, visibility-gated, same sealed-authorship discipline.
This slots into the ChildSceneSet rework this slice already owns.

**SLICES B + C LANDED TOGETHER, 2026-08-13 (PROTO_MINOR → 11).** What landed, in the design's terms:
- **Slice C, up half — `InterShardFlow::RealmShapeObservation`** (the shape mirror above): a live
  child ships its interior look — one outline per roster child at ITS authored placement, plus the
  sets its OWN live children shipped it, already lifted — one hop up, full-set, on the AoI cadence,
  from `emit_realm_frames` BEFORE the authored-empty return (the galaxy gotcha again). The parent
  lifts at receive (`on_realm_shape_observation` → `ObservedInteriorShapes`, TTL = the one knob) and
  `aoi_decide` folds a child's interior into ANY observer's drawn set whose band holds that child.
- **Slice B — SL7 liveness/interest reads the BIT:** `aoi_decide` folds one synthetic
  `ObserverId::Child` per FRESH bit at the parent-authored placement + velocity, reaching the
  child's own `circumscribed_extent` (the generic `reach` term — dist shortened, floored at 0);
  `observers.is_empty()` IS SL7's rule now. Cascade targeting (`active_children` /
  `home_of_active_child` / `any_active_child`) and the entity lane's down-leg re-read `ChildLiveness`
  (the bit's sender = the return address; a re-homed child re-targets with its first heartbeat). The
  parity gauge (`child_live_parity_divergence`) counts old-vs-new disagreement per child per tick,
  dies with slice D. THE HOLD IS ARMED EVERYWHERE: the shard BOOT FENCE refuses an AoI-armed world
  with a zero `VD_HANDOFF_HOLD_TICKS`; static launchers state the same derivation the demand
  spawner hands down (`static_handoff_hold_env` in `common_env` — deliberately NOT a spawn-anchor
  key, so the demand path's own derived value still wins for spawned children).
- **Slice C, down half — `InterShardFlow::ChildSceneSet`** (the §2 rekey, landed): the down-reflect
  is keyed by the LIVE CHILD REALM, never an occupant — `AccountId` left the wire. One set per live
  child = that child's in-range siblings (+ their observed interiors) merged with THIS realm's own
  from-above holding, one id per realm, restated once into the child's frame, send-on-change on
  `(home, ids)`. The receiver (`on_child_scene_set` → `FromAboveScene`) replaces ONE whole-realm
  holding and folds it into every local occupant's scene beside its own outline (own id wins) — the
  per-occupant `ProxySceneSet` EMIT is retired (arm tombstones in slice D with its lane), and the
  depth≥3 orphan case is structurally gone (`frame_conversion_e2e` now measures the LEAF holding a
  set from above: `leaf child_scene_received > 0`).
- **THE MEASURED FIX the gate flushed out:** the parent-resolution HeadRead sat BELOW `aoi_decide`'s
  zero-occupant return — an unoccupied realm could never resolve its parent, so its up-observation
  lanes were mute and a neighbour system's interior could not exist client-side. Moved ABOVE the
  Empty return (an armed realm owes its parent observation even while empty — SL3: visibility is the
  spin-up trigger).
- **GATED:** `a_flying_occupant_streams_a_neighbour_system_in_ahead_then_the_vacated_realm_is_reaped`
  — the honest D-RLM-16 rewrite AND the owner-gap gate: exit home, park inside the neighbour's wake
  band ~10 km out, its shard spins up ahead + its planet BOXES stream in + they MOVE, `location`
  never flips; return ⇒ the vacated neighbour reaps behind (baselines settled so neither climb can
  be satisfied by the home system's own activity). First green 2026-08-13.
Remaining: slice D (tombstone `OccupantInterest` + delete the up-relay/`RetainedOccupants`/parity
gauge), slice E (owner sign-off — entity-lane tombstone), slice F (SpawnV2 cutover, `Delta` death).

**THE COVERAGE-DEBT PUSH (same day, HR5):** the first coverage-fast run since Stage A found 423
uncovered Tier-A regions accumulated across the whole arc. Two systemic causes, both cured
structurally: (1) tracing macros' lazy field closures never evaluate without a subscriber — every
Tier-A test rig now installs a TRACE sink (`init_test_tracing`); (2) expressions inside a passing
`assert!`'s message are regions only failure evaluates — literalized. Plus ~30 targeted tests for
genuinely undriven arms (the up-lanes' dispatch wiring, the entity lane's from-above duties, the
flush/adopt/re-home refusal trio, roster/gateway/worldgen fns…), two dead-code deletions
(`direct_child_frame`/`direct_child_of_frame`, the unreachable child-restate arm in
`emit_realm_frames`) and three defensive-arm → stated-invariant conversions. RESULT: 423 → 18
missed regions (99.98%), lines 52,443/5 missed, branches 1,750/5. The residual 18 own NO source
line in the merged lcov (per-crate-hash duplicate instantiations — a tool artifact class); the
gate stays honestly red on them pending the owner's call on how the gate should treat them
(ledgered as a task). One measured behavior lesson en route: a crossing moves ONE hop into the
DIRECT child even when a grandchild claims the point — the first draft of that test expected
deepest-wins and the machinery correctly refused.

(Original design provenance: produced by a 10-agent arc — 4 code maps → 2 independent designs →
4 adversarial judges, all SOUND_WITH_FIXES; every judge fix folded in below. Companion:
`rehome_one_mechanism.md` §4u/§4v/§4w — the measured defects this discharges.)

Scope note on SL2, applied throughout: SL2 forbids occupant poses crossing REALM boundaries
(realm-to-realm). A shard streaming its OWN occupants to its OWN subscribed clients through the
gateway is not a realm-boundary crossing; that lane (EntitySnap, channels.rs:230) is untouched.

## 1. What is unlawful today (code-mapped, file:line in the workflow record)

- **The ghost feed** ships an occupant's full pose sibling-shard-to-sibling-shard at 20 Hz
  (`GhostFlow::Delta`), in the DEST's frame, and the source writes it VERBATIM into its retained
  dot (`refresh_source_ghost`) — the measured §4u/§4v label corruption, and an SL2 breach.
- **The occupant up-relay** (`OccupantInterest` → `RetainedOccupants`) ships every simulated dot's
  pose+velocity up to the parent every tick, re-relayed level by level (unbounded chain, coarsen
  levels) — SL2 breach; today it drives interest (proxy observers), the Empty gate, AND cascade
  targeting (`active_children`).
- **The entity lane** (`EntityInterest` up / `EntityCascade` down) ships entity sets across realm
  boundaries for cross-realm avatar rendering — SL2's enumerated forbidden case.

## 2. The replacement — one bit and the parent's own numbers

- **ChildLive** (the SL7 occupancy bit): child shard → parent shard, level-triggered heartbeat,
  presence-within-TTL IS the bit. Emitted iff the shard's observer set is non-empty (its own
  occupants ∪ its own fresh child bits), on the existing AoI cadence + once at adopt. Recursive
  by construction: a realm with only a live grandchild has a fresh child bit, so it reports live —
  no depth, no hop count, no pose. Parent stores `{child → last_seen, home NodeId, fence}`;
  stale-fence heartbeats rejected (zombie guard). The bit carries `fence + universe_tick` beyond
  SL7's literal "one bit" — ordering/zombie guards only, never authorizing; named explicitly in
  the owner ask (§8.1).
- **Interest = occupied-child proxy observers**: in `aoi_decide`, the proxy fold over up-relayed
  poses is replaced by one synthetic observer per FRESH direct child, at the placement AND
  velocity this shard already authors (`child_placements`) — zero new data, warm-ahead free
  (SL7 verbatim). Distance is inflated by the child's extent (`max(0, dist − extent)`, a generic
  parameter, no observer-type branch) — the SL7 error bound made operational: warming errs early,
  never late.
- **Liveness / Empty gate**: with Child observers in the observer list, `observers.is_empty()`
  IS SL7's rule (no occupants AND no live child). The Empty self-report to the orchestrator and
  the whole reconciler (arm-B, `ancestor_close`, two-phase drain, ForceReap) are untouched.
- **Cascade targeting**: `active_children`/`home_of_active_child`/`any_active_child` re-read the
  bit table (fresh bit = active; home = sender). `RealmCascade` itself — SL1-legal parent-authored
  placements — is byte-identical; only its targeting datum changes.
- **Shape lane rekeyed per child**: `ProxySceneSet` (routed per-occupant by AccountId, riding the
  pose up-relay) → `ChildSceneSet { child, realms }` per LIVE child at its authored placement.
  AccountId leaves the wire (a data reduction). The child fans to all its local dots' gateways,
  stores ONE `FromAboveScene` (replaces Forwarded/RelayedProxyScene), and restates into its own
  live children — the depth≥3 orphan case dissolves structurally. ⚠ Named SL3 debt, carried
  knowingly: the parent still ships sibling OUTLINES down; it retires when a realm authors its
  own look (SL3 end-state), ledgered in DEFERRED.md.
- **Ghost feed shrunk to its lawful residue**: KEEP the retained dot at demote (return-crossing
  target + the render fill at its OWN-frame demote pose), `HandoffHolds`/`speaks_for`, and
  `Despawn` (band-exit, carries no pose). ADD `GhostFlow::SpawnV2 { entity, source_fence,
  since_tick }` — the take-over proof that closes the hold WITHOUT the dest-frame pose (postcard
  forbids field removal in place). DELETE `Delta`, old `Spawn` (after cutover),
  `refresh_source_ghost`, `SourceGhostMirror`/`is_fed_ghost`. The label corruption dies at the
  root: nothing writes a foreign frame into a promotable dot any more — permanent tripwire
  `entity_rows_foreign_labelled == 0`. (Post-deletion pose writers, verified: the input
  integrator (own frame), `apply_crossing` — guarded by the RECEIVER-side `place_arriving_pose`
  acceptance {own ∪ direct children}, refusal otherwise — and the transient re-advance (own
  frame). The S6 grep-gate anchors on that receiver guard, NOT on any "source always rebases"
  claim, which is false for the verbatim arm.)
- **Fence-7 mid-crossing demand**: layers 1–3 kept (source KeepAlive of the dest lineage;
  orchestrator arrival shield; dest-side buffering — §4w). SL7 additions: `speaks_for` keeps the
  SOURCE's bit set through the hand-off (the parent keeps the neighbourhood alive with no pose
  up-relay), and the DEST emits its first bit at adopt. GAP CLOSED: the hand-off hold is
  MANDATORY on any AoI-armed shard (today it defaults to 0 = disarmed outside demand spawns),
  budget derived from the orchestrator arrival-shield duration already wired.

## 3. The accepted loss (owner product decision — §8.2)

Deleting the entity lane removes cross-realm avatar rendering outside the crossing window. What
replaces it visually is what the laws prescribe: the OCCUPIED REALM as its occupants' proxy (its
box/outline + live motion via the lawful realm lanes). At parent scale a child is small, so
individual avatars in a sibling realm are sub-child-resolution. The lawful FUTURE lane for true
remote avatars — the client holds read subs on visible realms' shards; each shard streams its OWN
occupants to its OWN subscribed clients — is designed-but-unbuilt, owner-gated, and its
server-side frame composition ties into the floating-origin server-authoritative rework
(the client must never compose two shards' frames itself; see §4).

## 4. The crossing-window rendering question (owner decision — §8.3)

During a hand-off the client legitimately holds TWO subs. Judge-verified fact: the client today
folds frames from ANY held sub into one per-entity track (the per-entity authoritative-sub picker
was deliberately deleted; staleness is per-sub only) — so "the client renders exactly one feed"
is NOT current machinery, and cross-sub stale rows can reach the render. The options:
(a) a server-side seam — but the gateway composing was deliberately deleted (its own tombstone
comments) and SL1 says the gateway is the parent of nothing; re-instating any gateway compose is
an owner-only decision; (b) a client-side authoritative-sub rule at promote (one feed wins per
entity) — no composition, but touches the deleted-picker ground; (c) keep the ghost fill as the
only crossing-window continuity (this design's default) and MEASURE the leaver's blink window
(< the 100–150 ms interpolation buffer) with swap-acknowledged keying if it fails.

## 5. Migration — seven landable slices (judge-corrected order)

- **A (additive, zero behaviour)**: ChildLive arm + producer + store + prune, parallel to the old
  lanes, nothing reads it. Parity gauge with STATED divergence classes: transient-only children
  and recursive-liveness children legitimately diverge (the NEW lane is correct there); zero
  divergence is asserted only on the player-scenario class. TTL enters `RlmTuning::validate()`
  (TTL ≥ 2–3 cadences, < empty-grace interplay) in THIS slice. Adopt-edge emit: buffered until
  `ParentRealmNode` resolves (the same discipline as fence-7 layer 3), designed here, not owed.
- **B+C (land TOGETHER — the judge-found regression window)**: B switches liveness/interest/
  targeting reads to the bit and arms the hold on all AoI-armed shards (hold-arming's Empty-timing
  shift measured against a walk/static byte-identity gate); C rekeys the shape lane
  (ChildSceneSet + FromAboveScene). Landing B alone would orphan the depth≥2 scene reflection for
  one slice. Gate: crossing e2e set + sky-doesn't-freeze + spin-up-ahead lead-time vs pre-slice
  baseline + depth≥3 visual run.
- **D**: tombstone `OccupantInterest`; delete the up-relay, `RetainedOccupants`, the mid-handoff
  up-relay, coarsen stats, the parity gauge.
- **E (OWNER SIGN-OFF FIRST — §8.2)**: tombstone the entity lane; rewrite (never delete) the
  visibility scenarios to the new contract.
- **F (last — touches freeze-fix-tagged machinery)**: SpawnV2 cutover, Delta deleted, retained
  ghost stops emitting at hold closure, `entity_rows_foreign_labelled == 0` asserted forever;
  the leaver-blink measurement of §4(c). DEFERRED entries: P5 cross-boundary collision seam
  (the collider registration skeleton is retained for it — it feeds nothing today, measured);
  the owed remote-avatar lane.

Wire discipline: deletions are TOMBSTONES (postcard discriminants never renumbered); one
PROTO_MINOR bump per wire-touching slice; every arm change in the ONE reviewed InterShardFlow
file; `intershard_closed.rs` pins updated per slice. Every slice passes `just gate` (which now
includes the flight-speed crossing) and the e2e set. All new logic is monomorphic straight-line
vd-sim code (HR5), no shard-kind branch anywhere (HR3/HR4), G-IDENTICAL fixture on ≥2 profiles
of THE world (SL5).

## 6. Judge-verified kill-proof of the §4u corruption

`refresh_source_ghost` is the SOLE production writer of a foreign-frame pose into a promotable
dot; it dies in slice F. The receiver-side guard (`place_arriving_pose` accepts only own ∪
direct-child frames, loud refusal otherwise) is the invariant the permanent gate anchors on.

## 7. Main risks (full list in the workflow record)

Warm-ahead resolution coarsens to child extent (measured lead-time gate before deletion);
bit loss/latency degrades cascade targeting not liveness (central backstops keep reaping safe);
a just-crossed player has a ≤ one-cadence sky-motion gap (asserted < interpolation buffer);
slice F alters what bystanders see at promote (leaver vanishes at hold closure rather than
tracking — a fade is new render policy, out of scope unless the owner wants it).

## 8. The owner's questions (SL6: default NO until answered)

1. **Three wire changes**: ADD `ChildLive` (the SL7 bit + fence/tick guards), ADD `ChildSceneSet`
   (the SAME parent-authored geometry ProxySceneSet ships today, minus AccountId), ADD
   `GhostFlow::SpawnV2` (strictly less data than Spawn). Four arms tombstoned
   (`OccupantInterest`, `EntityInterest`, `EntityCascade`, `ProxySceneSet`; `GhostFlow::Delta`/
   `Spawn` retired). Approve each?
2. **The avatar loss** (§3): accept occupied-realm-proxy visibility now, with the lawful
   remote-avatar lane deferred and ledgered?
3. **The crossing-window rendering rule** (§4): (a) is owner-retracted ground; recommend (c)
   measure-first, falling back to (b) one-feed-wins-at-promote only if the blink bound fails.
