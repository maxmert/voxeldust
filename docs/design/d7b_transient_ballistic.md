# D-7b — BallisticReadvance + per-tick TRANSIENT-CONSERVATION + loss-budget

LOCKED design (workflow `wf_55d7e981`: ground → 2 designs → 4 adversarial skeptics default-refute →
synthesize). Builds on D-7a (commit `3cd3c33`). Three ATOMIC sub-slices, each gate-green, each flips a
DEFERRED line. The skeptics REJECTED the oracle-excuse cure (Design 2) — confirmed in code that
`batch_goes` never GCs (`saga_runtime.rs` `or_insert` + `Tombstone` removes only `sagas`), so an excuse
keyed on it would mask a stuck both-held on EVERY tick — and adopted the STRUCTURAL drop-before-promote
(Design 1) with a retained-uncounted `Departing` source tier.

## The core correction (why structural, not oracle)

D-7a broadcast `TransientDrop` to BOTH source+dest. Under StaggerPlan tick-skew the dest can promote
(`Arriving→Held`, counted+rendered) at tick T while the source has not yet processed its drop at T+k →
a both-COUNTED-held tick for k ticks, AND `readvance_transients` would advance BOTH = a moving debris
double-rendered on two shards (the cardinal sin, `generic_transfer.md:197`). The fix mirrors the durable
demote-before-promote: the source goes UNCOUNTED+UNRENDERED **before** the dest promotes.

## D-7b.1 — structural drop-before-promote ordering (clears the duplication + double-render BLOCKERS; lands FIRST)

Choreography for batch `b`, source `S`, dest `D`, orchestrator `O` (two ordered round-trips):
- e0 `S` emit: `Crossing → Held{outbound:Some(b)}` (COUNTED), holders `{S}`.
- e1 `D` adopt: insert `Arriving{b}` (UNCOUNTED, UNRENDERED), ack `TransferAck::BatchAdopted`; holders `{S}`.
- e2 `O` on `BatchAdopted`: `handle_batch_adopted` sends `InterShardFlow::TransientRelease{transfer,step_id:TRANSIENT_RELEASE_STEP,fence}` to **SOURCE ONLY** (remove the old dest broadcast push).
- e3 `S` on `TransientRelease` (FirstApply, journal `(transfer,TRANSIENT_RELEASE_STEP)`): flip `Held{outbound:Some(b)} → Departing{batch:b}` (now UNCOUNTED+UNRENDERED), `transients_handed_off += 1`, ALWAYS ack `TransferAck::DropApplied{transfer_id,step_id:TRANSIENT_RELEASE_STEP}`; holders `{}` (the legal zero-held gap).
- e4 `O` on `DropApplied`(RELEASE_STEP): `handle_drop_applied` sends `InterShardFlow::TransientDrop{transfer,step_id:TRANSIENT_DROP_STEP,fence}` to **DEST ONLY** (TransientDrop NARROWED to mean PROMOTE).
- e5 `D` on `TransientDrop` (FirstApply, journal `(transfer,TRANSIENT_DROP_STEP)`): flip `Arriving{b} → Held{outbound:None}`, re-anchor `anchor_fence=drop.fence`, `transients_promoted += 1`, ack `TransferAck::DropApplied{..,step_id:TRANSIENT_DROP_STEP}` (promote-confirm, same variant, distinguished by step_id); holders `{D}`.
- e6 `O` on `DropApplied`(DROP_STEP): sends `InterShardFlow::ReleaseComplete{transfer,step_id,fence}` to **SOURCE ONLY**; `S` on FirstApply removes the `Departing{b}` item.

Holder sequence `{S}→{S}→{S}→{}→{}→{D}→{D}`, NEVER `{S,D}`. Handlers are pure egress on `&runtime`
(FF-1 — neither touches `batch_goes`). Under loss: any lost message leaves the item stuck UNCOUNTED
(→ eventual self-fence handover-loss within budget), never counted at both — promote (e5) is causally
unreachable without DropApplied (e3). The RETAINED `Departing` source copy lets a dest-crash-mid-promote
re-drive the promote against a still-extant copy (vs removal-on-release, which would widen the
permanent-loss window — the LOSS-BUDGET skeptic's confirmed hazard).

Wire (`intershard.rs`): `+TransientRelease`, `+ReleaseComplete` (both `{transfer,step_id,fence}`,
source-only, mirror DemoteCmd); `+TransferAck::DropApplied{transfer_id,step_id}` (release-ack AND
promote-confirm, distinguished by step_id — add to BOTH `transfer_id()`/`step_id()` or-patterns + a
serde-roundtrip test); narrow `TransientDrop` to DEST-only=PROMOTE (no struct change);
`+TRANSIENT_RELEASE_STEP:u32=13` (beside BATCH=11, DROP=12).
Sim (`stub.rs`): `+TransientStatus::Departing{batch:TransferId}`; `is_held()` 4-arm
(`Held|Crossing=>true`, `Arriving|Departing=>false`); split `on_transient_drop` →
`on_transient_release` / `on_transient_promote` + `+on_release_complete`; `+transient_release_noop`;
dispatch arms in `on_directory_reply` (`TransientRelease→release`, `ReleaseComplete→release_complete`,
`TransientDrop→promote`).
Node (`saga_runtime.rs`): `handle_batch_adopted` → SOURCE-only release (remove the dest push_flow);
`+handle_drop_applied` (RELEASE_STEP→TransientDrop/promote to DEST; DROP_STEP→ReleaseComplete to SOURCE);
`+DropApplied` inbound arm beside `BatchAdopted`. Both pure egress on `&runtime`.
Gate: `tests/tests/p3_transient.rs` StaggerPlan 1-item case, per-tick `verify_transient_conservation_tick`
(`len>1`) EVERY tick incl. skipped-node ticks, RED-then-GREEN; assert chosen offset `k <= max_absent_ticks`.

## D-7b.2 — BallisticReadvance advance + transient render path

Closed form (reuse, write no new math): `StampedPose::advanced_ballistic(accel, dt_s, new_tick)`
(`pose.rs:129-146`). Stub Debris: `accel=DVec3::ZERO` → constant-velocity. `(pose0,v0,tick0)` are the
TYPED `StampedPose` fields ALREADY on the wire: `pose0=TransientItem.pose`, `v0=pose0.vel`,
`tick0=pose0.universe_tick` (per-item authoritative origin; the batch-level `source_tick:TickId` is a
LOCAL tick, NOT the origin — Category-A motion is `f(seed, universe_tick)`). **ZERO wire change** for
the linear case. The `state:Vec<u8>` blob (≤`max_state_bytes=128`) stays EMPTY for Debris (reserved for
Guided target / RealmAnchored anchor-ref); do NOT pack pose/vel into it (single-source). postcard f64 =
fixed 8-byte LE → bit-exact cross-cut (a `transient_item_serde_roundtrip` test asserts it).
Core: `+kinematics::advance_continuity(continuity, pose0, accel, dt_s, target_tick) -> StampedPose` —
SHIP ONLY `BallisticReadvance => pose0.advanced_ballistic(..)` and `Frozen => {universe_tick:target,..pose0}`
arms (Guided/RealmAnchored land at P10/P11/P6 with real semantics — NO placeholder arm now).
Sim: `+readvance_transients` system inserted between `process_inbound` and `emit_transient_batch`;
advances ONLY the `is_held()` subset (Departing+Arriving SKIPPED); `dt_s = now.0.saturating_sub(tick0.0)
as f64 * config.tick_dt_s` (ONE straight-line expr, the sole tick_dt_s chokepoint; `saturating_sub` =
monotonic-forward-only, no branch); `adopt_transient_batch` stores `item.pose.sanitized()`. Per-shard
local, zero cross-shard read → `par_iter_mut`-ready for D-7c. The f64 pose NEVER gates a control
decision (only render + data payload); P4/P5's crossing trigger compares integer `cell_index`, not float.
Harness: `+InspectReport.held_transient_poses: Vec<(EntityId,StampedPose)>` (is_held() subset; the
transient twin of `held_poses`; Departing/Arriving excluded) → the moving-Debris RenderTrace.
Gate: moving-Debris POSE-CONTINUITY across the cut + a FAST (Projectile-speed) stagger continuity case +
the bit-exact serde roundtrip. If the worst-case Δ lemma shows stagger-alone `epsilon_pos` insufficient,
widen to `(1 + stagger_offset_ticks + max_fabric_delay_ticks)`.

## D-7b.3 — per-tick conservation + per-scenario loss budget

Harness (`oracle.rs`): `+verify_transient_conservation_tick(reports) -> Result<(),TransientViolation>`
— per-tick, duplication ONLY, `len > 1` (admits the zero-held gap, forbids two), NO excuse helper, NO
`batch_goes` dependency (the permanent-mask hazard is structurally absent), NO `dead` param (the
`_excluding` dead-aware variant lands at D-7d WITH its crash consumer — no empty-set theater now). The
per-tick LOOP lives in the scenario harness (mirroring how `verify_authority_unique` is driven per-tick),
NOT inside the oracle. `+verify_transient_loss_budget(reports, kind) -> Result<(),TransientViolation>` —
SEPARATE, called once at quiescence. `+TransientViolation::{Duplicated{entity,holders,tick:TickId},
LostOverBudget{kind:EntityKind,lost:u64,budget:LossBudget}}` (kept distinct from quiescent `DoubleHeld`).
Loss-budget semantics: `LossBudget` is an ABSOLUTE per-SCENARIO assertion threshold checked ONCE at
quiescence — NEVER a runtime quota. `loss <= budget ⇒ PASS`, `loss > budget ⇒ LostOverBudget`, ANY
duplication ⇒ FAIL regardless of budget. The EXACT counter is a NEW handover-attributable per-kind
`StubStats.transients_lost_in_handover: DetHashMap<EntityKind,u64>` — NOT the gross `transients_dropped`
(which counts `owned.0.len()` incl. settled-resident `Held{outbound:None}` debris; under a 1000-item
burst eviction that gross count would spuriously trip `LostOverBudget(4)` — the LOSS-BUDGET skeptic's
confirmed "wrong population" finding). `self_fence_drop_transients` WALKS `owned.0` before `clear()` and
buckets ONLY handover-status items (SOURCE `Held{outbound:Some(_)}`, `Departing`, `Crossing`; DEST
`Arriving`) keyed by `EntityKind::from_tag(entity.kind_tag())`; settled `Held{outbound:None}` is a
resident-eviction event OUT of budget scope. Gross `transients_dropped` KEPT for ops visibility.
Harness `+InspectReport.transient_loss: Vec<(EntityKind,u64)>`.
Gate: deterministic over-budget self-fence (5>4 → `expect_err(LostOverBudget{Debris,5,LossBudget(4)})`),
within-budget (3<=4 → Ok), resident-eviction-NOT-counted (settled items self-fenced → handover counter 0
while gross increments).

## HR5 (every new branch deterministically covered — no proptest draws, no matches!-false-arm)

- `is_held()` 4-arm: a lifecycle unit test constructs each `TransientStatus` (Departing constructed on
  every release → not a dead arm), `assert_eq!` on the bool.
- `advance_continuity`: each SHIPPED arm (Ballistic/Frozen) covered by a core unit test (assert_eq! on
  the exact output pose) AND re-exercised in the integration binary (per-monomorphization gotcha).
- inline `dt_s`: `now<tick0` test (dt=0, pose unchanged, tick re-stamped) + `now>tick0` test (forward
  motion); `saturating_sub` has no branch.
- `verify_transient_conservation_tick` `>1`: stagger-duplication RED-then-GREEN (red on the pre-fix
  broadcast model). `Ok` by lockstep + fixed-stagger. `expect_err`/`assert_eq!` on the exact `Duplicated`.
- `verify_transient_loss_budget` `>budget`: deterministic 5>4; `<=budget`: 3<=4; resident-NOT-counted.
- new wire arms: serde-roundtrip + the extended `transfer_id()`/`step_id()` or-patterns exercised by
  constructing a `DropApplied`.
- new handlers: FirstApply (happy e2e) + AlreadyApplied re-ack (redelivery unit test, noop counter) +
  the `None`-no-go-token LOUD-no-op arm (stray-ack test).
- `held_transient_poses`/`transient_loss` scrapes: non-empty + empty cases.

## Does not corner D-7c / D-7d (end-goal fit)

Advance dispatches on `KindDef::continuity` (NOT shard kind — HR3); Debris+Projectile share the one
`BallisticReadvance` arm (Projectile = ~4 registry lines, zero advance-code change); DroppedBlock gets
an additive `RealmAnchored` arm at P6. `readvance_transients` is per-shard-local → `par_iter_mut` for
D-7c with zero logic change. The new Release/DropApplied/ReleaseComplete arms are POST-COMMIT
choreography on read-only `&runtime`, NOT FSM states — the saga `BatchCommitting` stays pose-agnostic;
ordering is per-BATCH (1000 items = ONE extra round-trip, not 1000), transients still write ZERO
directory rows. Every new step is journaled (RELEASE=13, idempotent at-least-once) → kill-9
mid-handoff replays cleanly (D-7d); the `Duplicated{tick}` variant pins which tick a cell duplicated;
the dead-aware `_excluding` oracle + dest-crash-into-same-budget land at D-7d WITH their consumers.
