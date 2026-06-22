# Slice 1d.4/1d.5 — D-2 Ordered Demote-Before-Promote (the SEAMLESS tear-out)

> Status: **LOCKED** (planned via `wf_f780b04c`, adversarially verified — 3/3 skeptics rejected the
> draft, 7 blocking + 7 major folded in). Closes DEFERRED **D-2** (the 1c.8 promote-before-demote
> interim + the 3-tick vanish). This is the implementation decomposition behind the D-2 registry entry.

## Locked decisions

- **Q2 collision scope (USER, 2026-06):** **feed + registration NOW, physical response at P5.** There is
  no physics/collision system in the tree (`integrate` is pure point translation; rapier named in the
  spec but absent). 1d.5b lands the `GhostFlow` wire + dest `GhostColliderRegistration` (pose-mirror,
  fully testable), backend-agnostic; the rapier/parry kinematic-collider + collision *response* is a
  joint library decision at P5 (per the investigate-libs-together rule). The transition is seamless now;
  players-collide-across-boundary *response* completes at P5.
- **Q3 frame-rebind (resolved from code):** NOT a prerequisite. `view.rs::world_pos` maps every
  `SystemSpace { .. }` (any seed) to `pose.pos` (identity) — "System frames are world-origin in P1.5";
  the galaxy ly-cell offset is P10. So seed-7 (crossed) and seed-8 (dest default) share one world
  origin; `verify_pose_continuity` (world_pos basis) holds structurally. Keep the seed-7 pin verbatim in
  1d.5b; add a belt-and-suspenders fixture asserting both seeds map identity in the test topology. Frame
  *rebind* (seed-7 → seed-8 label) stays deferred to D-27/P10.
- **Q1 delivery channel:** new `TransferControlAck::DeliveredToObservers` variant (landed in 1d.4a) —
  `Accepted` = applied-at-dest (shard, once); delivery = fanned-to-all-observers (gateway,
  per-observer-per-frame) — different watermarks. Keeps 1d.5a a pure predicate body-swap.
- **Q4 watermark storage:** cold sibling map on `Session` keyed `SubId → frame_id`, off the wait-free
  `SubTable`. A 20Hz fan-out load test is owed in 1d.5a (non-blocking).
- **Q5 D-29 strand:** the new `(Demoting, Timeout)`/`(Promoting, Timeout)` re-emit arms give every
  post-CAS leg at-least-once re-drive; no Slice-2 adaptive-deadline machinery needed for this gate.

## The three load-bearing corrections the adversarial pass forced

1. **The freeze is NOT frame-fence-enforced for a single-entity transfer.** `emit_frames` stamps the
   **shard-wide** `realm_fence`; fence-rule-5 fires only on REALM-fence movement (D-25), not single-entity
   demote. → The source freeze is enforced by the Ghost **ceasing to emit** (`Authority::Ghost.simulates()
   == false` filters it out of `emit_frames`), and the **ordering by the saga gate**, not a frame-fence drop.
2. **The dest promote is autonomous, not saga-gated.** `pending_grant_op`/`flip_grant` set `granted=true`
   off the dest's *own* directory poll; the saga FSM has no Promote/Owned transition. → 1d.5b adds a real
   saga-driven `Promote` command + a new `Promoting` state and **deletes** the autonomous adopt-flip.
3. **There is no collision system anywhere.** → "players collide" downgraded to the feed + registration
   (per Q2); the 1d.5b gate asserts ghost registration + pose-mirror, not overlap response.

## Sub-slice decomposition (strict order; each leaves the tree green; only 1d.5b flips the headline)

### 1d.4a — net-new wire + FSM vocabulary (pure-additive, review-in-isolation)
- `transfer_control.rs`: `TransferControl::Demote{transfer,session,src,new_owner_fence,at_tick}` +
  `::Promote{transfer,session,dest,new_fence,at_tick}`; acks `DemoteAck{transfer,drained_seq}`,
  `PromoteAck{transfer}`, `DeliveredToObservers{transfer}`. Extend `transfer()/session()/step_id()`
  (Demote=7, Promote=8). `compensator_of` → all post-commit forward-only ⇒ `None`.
- `saga.rs`: `SagaAction::{Demote,Promote}`, `SagaEvent::{DemoteAcked,PromoteAcked,DestDelivered}`. No
  new state yet.
- `saga_runtime.rs`: extend the exhaustive `ack_to_event` (7→10).
- `io/mod.rs`: **split `MsgClass`** → `GhostReliable` (Reliable: Spawn/Despawn) + `GhostDelta`
  (Unreliable: Delta) — a single Ghost class can't yield per-variant reliability. Add both `reliability()`
  arms + the inbound `match class` arms in stub.rs **as counted no-ops** (real logic in 1d.5b).
- **HR1:** no new InterShardFlow arm — Demote/Promote/DeliveredToObservers ride existing `Saga`/`SagaAck`
  (already `SideEffecting` by `(transfer,step_id)`); the MsgClass split is a transport-reliability
  refinement of the existing `Ghost` arm (stays `FireAndForget`).
- **Gates:** extend the shared `all_commands()/all_acks()` fixtures once (roundtrip+step_id+compensator);
  `ack_to_event` 10 cases; `reliability()` both arms; a producer-absence pin (`Demote`/`Promote` emitted
  nowhere yet — flips in 1d.5b). No DEFERRED flips.

### 1d.4b — attach the Authority FSM to Dot (pure refactor, still poll-driven)
- `Dot` gains `authority: Authority` (login ⇒ `Owned{fence}`; dest adopt ⇒ `Frozen` on PrepareSubscribe).
- `emit_frames` filters on `d.authority.simulates()` (render_ready derived); input gate reads
  `!d.authority.simulates()`.
- `self_fence_foreign_entity` applies `Freeze` then `Demote{new_owner_fence}` and **RETAINS** the dot as
  a Ghost (the first ghost), still poll-discovered.
- **Oracle fix (mandatory):** `inspect_world` (topology.rs) re-derives `held_entities` from
  `d.authority.simulates()`, not `d.granted` — a Ghost is excluded, so a retained Ghost can't false-trip
  `verify_authority_unique`.
- **Honest FG-2 scope:** `granted` stays the predicate truth that `foreign_takeover_target`/
  `crossing_target`/`flush_target` key on until 1d.5b deletes the poll. DEFERRED note: "Authority is the
  emit/input/oracle truth; `granted` is the predicate truth until 1d.5b."
- **Gates:** Ghost emits nothing; Ghost retained in `Dots` after the poll; Ghost excluded from
  `held_entities`; `Authority::apply` re-exercised at new call sites. The capstone pins do NOT flip yet.

### 1d.5a — the (a) predicate (DELIVERY-ONLY) — ✅ DONE, and it landed SEAMLESS early
> RE-LOCK (USER-confirmed, plan `wf_1b64b75e`): 1d.5a is the DELIVERY watermark ONLY. The "+ band exit"
> conjunct moved to 1d.5b — a probe proved it UNSATISFIABLE in the stub (source + dest cap at 0.4 m vs a
> velocity-safety-forced ≥2.1 m band edge → would PARK the saga). Band-exit lands in 1d.5b on a moving Ghost.
- **Observer watermark (server-side, cold):** `Session.delivered: BTreeMap<SubId, frame_id>`, written in
  `on_shard_frame` at the push instant (peeked via `peek_snapshot_frame_id`; only ACCEPTED, past-fence frames
  advance it; removed at the drain-sweep). Off the wait-free SubTable (HR1).
- **Standing predicate (NOT a latched gateway signal):** `every_observer_delivered` recomputes each tick over
  the CURRENT `subscribers_of(dest)` — NON-EMPTY required (anti-vacuous) AND every observer watermark ≥ 1.
  Emitted as `DeliveredToObservers`, latched orchestrator-side as `LiveSaga.dest_delivered`.
- `interim_demote_complete` → `demote_when_delivered_and_exited`: fires `DemoteComplete` when `delivered(live)`
  (single-bool shim; 1d.5b adds `& band_exited`). Plus a `(Demoting, Timeout)` loud-fail arm (pulled forward).
- **✅ SEAMLESS LANDED HERE (earlier than predicted):** delivery-gating delays the source-SUB `ReleaseSubscribe`
  until the dest is delivered, so the source sub HOLDS through the dest's first frame → the FORK-0a two-sub
  overlap manifests → the capstone flipped to `overlap_ticks >= 1` + `vanish_gap == 0` + `verify_no_vanish`/
  `verify_pose_continuity` (GREEN). This is the READ-plane seam; the WRITE-plane ordering is still 1d.5b.
- **Band machinery (OverlapBand/BandMembership, the CI band inequality) → 1d.5b** (where the Ghost moves).

### 1d.5b — the (b) tear-out + the seamless close (flips the headline)
> **⚠️ SUPERSEDED — see `DEFERRED.md` D-2 "✅ LANDED 1d.5b.1" for the as-built design (the main-thread R1
> refinement).** The §1d.5b sketch below was REVISED during implementation: (1) `Promoting` is entered on
> `DemoteAcked` ALONE (NOT `DemoteAcked AND DemoteComplete`) — gating it on delivery would DEADLOCK (delivery
> needs the promote, the promote needs the demote); `DemoteComplete` was DELETED entirely. (2) the release gate
> `Promoting→Releasing` is `PromoteAcked` **AND** `DeliveredToObservers` (the seamless no-vanish gate), not
> `PromoteAcked` alone; an early `DestDelivered` is latched in `Demoting`. (3) the autonomous dest adopt-flip
> (`apply_crossing`'s `Ghost→Owned`) is **RETAINED in 1d.5b.1** — relocating it would DELAY the dest's
> first-Owned moment by the round-trip and re-open the vanish; `on_saga_promote` is a 1d.5b.1 CONFIRMER, and the
> relocation + the source-Ghost collider feed land TOGETHER in 1d.5b.3. The 1d.5b split is .1 (this FSM/egress/
> consumers, poll live) → .2 (poll tear-out + R2 oracle) → .3 (GhostFlow feed + flip relocation + band-exit).
- **Saga ordering gate (as-built, 1d.5b.1):** `Swapping→Demoting` emits `Demote`; **`Promoting`** entered on
  `DemoteAcked` alone, emitting `Promote`; `Promoting→Releasing` on `PromoteAcked` AND `DestDelivered`. The dest's
  real `Ghost→Owned` flip stays in `apply_crossing` (RETAINED until 1d.5b.3); `on_saga_promote` confirms + acks.
- **Crash recovery:** `(Demoting,Timeout)→re-emit Demote` + `(Promoting,Timeout)→re-emit Promote`
  (idempotent, forward-only). A duplicate Demote on an already-Ghost dot is a typed no-op that re-acks
  (in the stub wrapper, not `authority.rs::apply` — keep the FSM pure).
- **Ghost-as-collider registration:** on `Demote` the source emits `GhostFlow::Spawn` (GhostReliable) +
  `Delta`/tick (GhostDelta, 20Hz datagram, dropped by `(source_tick,seq)` latest-wins) + `Despawn` on
  band-exit *after* `in_transfer` clears. Dest inserts a `GhostColliderRegistration` (presence + mirrored
  pose, fence-monotone + seq-monotone via `GhostRefresh`). `Ghost::simulates()==false` ⇒ dest never
  integrates it. Lost Spawn self-heals from the first Delta (Ghost stays `FireAndForget`). A Delta before
  its Spawn is a counted no-op (mirrors `apply_crossing`'s buffer-until-flip).
- **Surgical poll removal:** delete `granted_key_poll_tick` + `foreign_takeover_target` (NOT
  `crossing_target` — the twin-predicate DO-NOT-MERGE) + `self_fence_foreign_entity` + the poll arm of
  `pending_grant_op` + the FOREIGN branch of `on_directory_reply`. **KEEP `realm_recheck_interval`** — its
  realm-head loss-reaction arm is a distinct consumer that would go dead/uncovered (HR5) if the seed were
  deleted. Confirm which tests exercise the realm-recheck arm before touching it.
- **Gates flipped (headline):** capstone `overlap_ticks == 0 → >= 1`; `vanish_gap > 0 → == 0` via
  `verify_no_vanish` + `verify_pose_continuity` (lockstep K=0); `MAX_INTERIM_VANISH_TICKS`/`max_absent_run`
  deleted/subsumed; `flips==1` retained (two-holder window de-dups to one avatar via `AuthorityChanged`);
  the `SystemSpace{seed:7}` pin STAYS (frame-rebind deferred).
- **Gates landed (new):** `ghost_registered_across_band` (Ghost retained + dest registration at new fence
  + `simulates()==false`; NOT physical overlap response); saga proptests for `Demoting`/`Promoting`/Timeout
  arms; Despawn-after-in_transfer-clear; out-of-order-Delta; lost-Spawn-self-heal. The 1d.4a producer-absence
  pin flips.
- **DEFERRED:** D-2 🟧→🟩 (all 5 consequences retired); D-30 🟥→🟩 (in-process seamless half); D-25 stays
  🟥 (per-entity frame fence NOT taken); D-27 advances (ghost registration; frame-rebind still owed);
  D-28/D-29 become runnable.

## Ordering proof (corrected)
- **L1 freeze:** `Demote` requires `fence.is_stale_against(new_owner_fence)` (strictly newer), flips
  `Frozen→Ghost`; the Ghost stops emitting (`simulates()==false`) **before** `DemoteAck` — the ack is the
  proof-of-freeze the saga awaits. Not a frame-fence drop (realm-fence path untouched; other players keep
  rendering).
- **L2 ordering:** dest `Promote` is saga-emitted, reachable only from `Promoting`, entered only on BOTH
  `DemoteAcked` AND `DemoteComplete`. The autonomous adopt-flip is gone ⇒ the dest can't become `Owned`
  until the source is provably Ghost-at-new_fence. Demote happens-before Promote by a cross-shard dependency.
- **L3 no split-brain:** both `Promote`/`Demote` gate on `is_stale_against` a strictly-newer fence ⇒
  two-Owned-at-same-fence is unrepresentable. `verify_authority_unique` sampled mid-flight (now possible —
  `held_entities` excludes Ghosts) finds exactly one Owner; the source is a Ghost at `source_fence ==
  new_owner_fence`. Directory CAS stays the single commit point.
- **Crash matrix (scope HONESTLY — 1d.5b closes the LIVE-process strand, NOT crash-restart):** every post-CAS leg
  has a timeout re-drive that re-fires a saga **still alive in RAM**, so a LIVE-orchestrator slow/lost
  Demote/DemoteAck/Promote self-heals — this closes the 1c.8 live-process poll-strand (a real improvement). It does
  **NOT** deliver crash-RESTART resilience: saga state is RAM-only until the durable WAL (DEFERRED **D-6** / ROB-2 at
  P3), so an orchestrator CRASH still loses in-flight sagas. "Restart-durability" is owed by D-6 at P3 (before the real
  kill-9 crash matrix), NOT by 1d.5b — do not read 1d.5b's green as closing the crash class.

## Residual risks (tracked, non-blocking)
- 20Hz cold-watermark load (Q4) — load test owed in 1d.5a.
- Spawn/Delta/Despawn lifecycle across the two MsgClasses — covered by buffer-or-drop + `(source_tick,seq)`
  gates; worth a focused FaultFabric chaos run.
- `realm_recheck_interval` realm-recheck arm must be confirmed still covered after the granted-key-poll
  removal (1d.5b machinery step 10).
