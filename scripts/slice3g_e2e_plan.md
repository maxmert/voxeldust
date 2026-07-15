I now have complete, code-verified grounding. The DEFERRED entry confirms the adversary's C1 root: item #2 says "on `Entry::Occupied` with `now - last_commit_tick >= request_ttl_ticks`, RE-EMIT" — this is exactly the placement that `should_commit`'s strict rising edge makes unreachable for a dwelling dot. The DEFERRED design text itself carries the defect. I have everything needed to finalize the definitive plan.

---

# DEFINITIVE PLAN — task #133 crossing E2E capstone (3f-D4 → 3g → crash-leg → 3h)

## Verdict on the draft + adversary

The draft is code-honest and ~80% right; the adversary's C1/H1/H2/H3/M1/M4 are **all confirmed by firsthand reads**. I have re-anchored every line number and folded the valid findings. The one correction to the adversary: its L3 claim that "`saga_runtime.rs` doesn't exist under `crates/sim`" is right — the orchestrator is `crates/node/src/saga_runtime.rs`; `on_crossing_aborted` is `crates/sim/src/stub.rs:1649`. All anchors below are verified against HEAD.

## Verified ground truth (firsthand, this session)

- **C1 CONFIRMED — the draft's own Q3 is a design defect, not an open question.** `fan_out_crossing` is called ONLY inside `if let Some(dir) = should_commit(...)` at `crates/sim/src/stub.rs:2494`. `should_commit` (`crates/core/src/geometry.rs:642`) returns `Some(Inward)` ONLY when `inward_ticks == tuning.n_entry` — a **strict one-tick rising edge** (pinned by `should_commit_inward_is_a_strict_rising_edge` `geometry.rs:1340` and `should_commit_over_a_full_trajectory_commits_inward_exactly_once` `:1667`). A statically-dwelling stranded dot (DEFERRED #2's literal case) never re-enters `fan_out_crossing`, so the `Entry::Occupied` re-drive can NEVER fire for it. The `handle_crossing_request` comment (`saga_runtime.rs:1421` "the source re-drives the request every tick the entity stays over the boundary") **describes an intent the code does not deliver** — a genuine defect, ledgered but unbuilt. The re-drive MUST move to a latch-scan.
- **H1 CONFIRMED — counters already exist.** `crossing_durable_no_session` (`stub.rs:699`), `crossing_latches_cleared` (`stub.rs:709`), `crossings_requested` (`:690`), `crossings_suppressed_in_flight` (`:694`), `transient_crossings_requested` (`:702`) all exist. Orchestrator-side `crossings_started` (`saga_runtime.rs:479`, bumped `:1441`) and `transient_crossings_granted` (`:492`, bumped `:1482`) exist with getters (`:683`, `:703`). **ONLY `crossings_retried` is new.**
- **H2 CONFIRMED — `crossings_started` is the real composition proof.** It bumps ONLY inside `handle_crossing_request`'s `(Some,Some,Some)` non-`contains_key` arm (`saga_runtime.rs:1441`), i.e. only when a `CrossingRequest` actually became a saga. `crossings_requested >= 1` proves Leg 1 only.
- **H3 CONFIRMED — `rebuild()` ≠ SIGKILL.** FSM `rebuild()` (`saga_runtime.rs:2585`) is a World-drop. The real `kill -9` is `Cluster::kill_and_reap` → `child.kill()` (`crates/bins/src/lib.rs:1088`, "SIGKILL on Unix"), used by `crates/bins/tests/orchestrator_crash.rs` — but that test is a fsync/batch crash, NOT a geometric crossing.
- **M1 CONFIRMED — the abort must take the crossing-origin path.** `PendingAbortReply` stages ONLY when `aborted && crossing_origin && durable` (`saga_runtime.rs:1212`); `is_crossing_origin` (`:1155`) tests the `0x39` id tag. A non-crossing abort clears nothing. `on_crossing_aborted` (`stub.rs:1649`) already: unconditionally acks (`:1659`), clears on exact id-match (`:1670`), re-arms dwell + bumps attempt (`:1677-1681`).
- **M4 CONFIRMED — transient auto-grants.** `handle_transient_crossing_request` (`saga_runtime.rs:1460`) auto-grants when `dir.head(Realm(to_realm))` resolves (`:1467-1482`) — no hand-fed batch trigger. The draft's Q8 pessimism is wrong; build it fully autonomous.
- **No stub-tier autonomous prepare-reject exists** — reject is a saga-tier Rig hand-feed (`saga_runtime.rs:3117`). So a cluster-autonomous abort DOES need an inert lever (M1/Q7 stands).
- `StubConfig` is a `Res` (`evaluate_realm_boundaries` takes `config: Res<StubConfig>` `:2339`), cloned into `CrossingCtx` per tick (`:2371`) — mutable mid-run via `world.resource_mut::<StubConfig>()`, no snapshot staleness (Q2 resolved: safe).
- `run_cut_transfer` scaffold: `p2_cluster` (`tests/src/lib.rs:260`), `p2_cluster_durable_orch` → retained `MemStore` (`:292`), `trigger_transfer` = direct `start_transfer` hand-feed (`:375`), `walk_forward` = `movement:[1,0,0]` (`:696`), `seed_transient_crossing` (`:409`), `read_subject` (`:354`). Render-flip gate: `p2_transfer_gates.rs:719` asserts `dest_sample.frame == FrameRef::SystemSpace{system_seed:7}` (`:751`) + `world_pos.is_finite()` (`:756`) + `!= ZERO` (`:760`); byte-identical sibling at `:652`. `FrameRef::label()` asserted NOWHERE in e2e (draft correct — assert the VALUE).
- DEFERRED D-43 block: `docs/design/DEFERRED.md:3017-3030`; items #1-#8 at `:3020-3027`; Pin at `:3029`.

---

## Sub-slice ordering (revised for C1)

**3f-D4-relocate → 3g-0 (scrape) → 3g-durable → 3g-render → 3g-abort → 3g-transient → 3g-C (crash) → 3h (gate + DEFERRED).**

---

## Sub-slice 3f-D4 — the TTL re-drive as a LATCH-SCAN system (NOT the Occupied arm)

**The C1 fix (mandatory).** The re-drive moves OUT of `fan_out_crossing`'s `Entry::Occupied` arm into a new per-tick pass over `RequestInFlight`, scheduled right after `evaluate_realm_boundaries`. The `Entry::Occupied` arm **stays** a pure suppress (`stub.rs:2652` unchanged) — its job is dedup within one rising edge, which is correct.

**Site:** new fn `redrive_stranded_crossings` in `crates/sim/src/stub.rs`, registered in the same system set as `evaluate_realm_boundaries` (`stub.rs:875`), ordered `.after(evaluate_realm_boundaries)`.

**Why a scan and not the arm:** the stranded-dwelling subject generates no second rising edge, so only an unconditional per-tick scan of the held latches can re-emit. The scan needs, per latched entity: `subject_fence`, `subject_session`, `from_realm`, `to_realm`, `last_commit_tick`, `attempt`. `RequestInFlight` holds only `EntityId → TransferId`; the rest must be recovered from the live `Dots`/`OwnedTransients` + `CrossingProgress` + `RealmBoundaries`. **OPEN Q-A below** — this recovery is the load-bearing unknown.

**Shape (monomorphic; the whole body is the covered surface):**
```rust
/// 3f-D4 (DEFERRED D-43 #2): per-tick RE-DRIVE of a STRANDED durable crossing latch — a delivered-but-
/// unresolved dest (`head(Realm(to))` transiently absent) leaves the `RequestInFlight` latch standing
/// with NO rising edge to re-emit it (`should_commit` is a strict one-tick edge; the `Entry::Occupied`
/// suppress never re-fires while dwelling). This scan re-emits the SAME latched `CrossingRequest`
/// (same `(subject, fence, attempt)` → byte-identical id; the orchestrator's `sagas.contains_key`
/// absorbs a dup that started, an unresolved one re-tries the head reads) once `local_tick -
/// last_commit_tick >= request_ttl_ticks`. `ttl == 0` (default / every current rig) → INERT no-op
/// (early return before any iteration). Authority-gated exactly like `evaluate_realm_boundaries`.
/// Monomorphic — every branch covered once here (HR5).
fn redrive_stranded_crossings(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    dots: Res<Dots>,
    owned_transients: Res<OwnedTransients>,
    mut progress: ResMut<CrossingProgress>,
    in_flight: Res<RequestInFlight>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let ttl = config.request_ttl_ticks;
    if ttl == 0 { return; }                       // INERT default — one covered early-return
    let Some(realm_fence) = authority.0 else { return; };
    for (entity, _latched) in in_flight.0.iter() {
        redrive_one_stranded(
            &config, &clock, realm_fence, ttl, *entity,
            &dots, &owned_transients, &mut progress.0, &mut stats, &mut outbox,
        );
    }
}
```
`redrive_one_stranded` is a second monomorphic helper that: (1) looks up the subject's live `to_realm`/`session`/`fence` (Q-A), (2) reads `state.last_commit_tick`, (3) `if elapsed >= ttl` re-emits the SAME `CrossingRequest{..., attempt: state.crossing_attempt}` and re-arms `last_commit_tick = clock.local_tick`, bumping `stats.crossings_retried`. Each false arm (`None` last_commit, `elapsed < ttl`, subject-not-live, `None` session) is a **distinct** counted no-op per HR5(a) — do NOT fold them into one `_` arm (adversary L1a).

**Counter (the only new field):** `pub crossings_retried: u64` next to `crossings_suppressed_in_flight` at `stub.rs:694`. Do NOT add `crossing_durable_no_session`/`crossing_latches_cleared` (already exist — H1).

**Determinism:** same-`(subject,fence,attempt)` re-mint → byte-identical id; `last_commit_tick` re-write is a pure fn of `clock.local_tick` (universe-clock). Scan order is a `BTreeMap` iteration — deterministic.

**HR5 coverage:** all in `stub.rs`'s own `mod tests`. Test matrix (re-derived for the scan, per adversary L1b):
1. `ttl_zero_is_inert` — scan with `ttl==0` and a standing latch → `crossings_retried == 0` (the early-return false arm).
2. `a_stranded_latch_re_drives_after_ttl` — the headline. `Rig::new()`, `grant_realm()`, set `request_ttl_ticks = k` (mutate `StubConfig` res), insert an OWNED dot INSIDE a planted `Authority` shell so a first rising edge latches it (`crossings_requested == 1`), then advance `local_tick` past `last_commit_tick + k` WITHOUT any terminal and WITHOUT re-crossing (dot dwells statically). Assert the scan re-emits: `crossings_retried == 1`, a SECOND `CrossingRequest` in the outbox with `attempt == 0` + `subject_fence == Fence(1)`, and its id `== crossing_transfer_id(Entity(e), Fence(1), 0)` (byte-identical). **This test is now WRITABLE against a static dwell — the whole point of the relocation.**
3. `re_drive_below_ttl_is_suppressed` — advance only `k-1` → `crossings_retried == 0`.
4. `re_drive_skips_a_subject_that_left` — latch present but subject evicted from `Dots` → the not-live no-op counted, no emit.
5. `re_drive_degrades_on_missing_session` — a Durable-tagged held transient (reuse `a_durable_tagged_transient_degrades` fixture) latched with `None` session → `crossing_durable_no_session` bumps, no emit.

**⚠️ This relocation retires DEFERRED #2 AND #3 together** (#3 is the abort re-arm, already landed at `stub.rs:1677`, but the scan makes the "still-dwelling re-request" it promised actually reachable). Note both in 3h.

---

## Sub-slice 3g-0 — surface the crossing counters to the oracle

**Why:** anti-vacuity (H2) needs `crossings_started` (orchestrator) + `crossings_requested`/`crossings_retried` (source) + `crossing_latches_cleared` visible via `InspectReport`. Currently `inspect_world` (`topology.rs:219-231`) scrapes only `last_input_slot_resume` + `transients_lost_in_handover`.

**Site:** `crates/harness/src/topology.rs` — add fields to `InspectReport` (near `:77`), populate in `inspect_world` (`:219` StubStats block for source-side; the orchestrator `SagaRuntimeRes` block for `crossings_started`/`transient_crossings_granted` — **OPEN Q-B: confirm `inspect_world` already reaches `SagaRuntimeRes` for the ORCH node, or add that scrape**).

**Fields:**
```rust
// InspectReport (source-side, from StubStats — None on ORCH/client)
pub crossings_requested: u64,
pub crossings_retried: u64,
pub crossings_suppressed_in_flight: u64,
pub crossing_latches_cleared: u64,
pub in_flight_latches: Vec<EntityId>,       // for the abort-leg "latch empty" assert
// orchestrator-side (from SagaRuntimeRes — None elsewhere)
pub crossings_started: u64,
pub transient_crossings_granted: u64,
pub pending_abort_replies: usize,           // M1: crossing-origin abort staged (non-empty pre-ack)
```
`in_flight_latches` = `stats`-adjacent `RequestInFlight.0.keys()` scrape; `pending_abort_replies` = `SagaRuntimeRes.pending_abort_replies.len()` (needs a getter — **OPEN Q-C: `pending_abort_replies` is `pub(crate)` at `saga_runtime.rs:502`; add a `pub fn pending_abort_replies_len(&self) -> usize` getter, mirroring `crossings_started()` `:683`**).

**HR5:** straight field copies inside the existing `if let Some(stats)`/`if let Some(runtime)` (already covered); `Default` covers the None arms. **OPEN Q-D:** check whether any `topology.rs` unit test asserts `InspectReport` by full struct-literal `==` (~`:1051`) — the new fields default to 0/empty so most survive, but a literal comparison needs updating.

---

## Sub-slice 3g — the full crossing E2E

**File:** NEW `tests/tests/crossing_e2e.rs` (reuses the `vd_tests` scaffold verbatim — decision locked; moving to `crates/node/tests` would fork the client/oracle plumbing).

**New harness helper** in `tests/src/lib.rs` (near `seed_transient_crossing` `:409`), **param-taking the boundary set** (adversary L2 — cheap now, painful to retrofit for the N≥128 soak):
```rust
/// 3g: plant an AUTHORITY realm boundary on the SOURCE whose crossing hands authority to the DEST
/// realm. `boundaries` is the caller's set (single shell for 3g; the dense soak reuses this with N).
/// INERT until this plant — the composer leaves `RealmBoundaries` empty.
pub fn plant_crossing_boundaries(topo: &mut Topology, boundaries: Vec<RealmBoundary>) { … }

/// The 3g single-shell convenience: a small AUTHORITY shell CENTERED AT THE SPAWN so the dot is a
/// band MEMBER from spawn and commits after `n_entry` dwell ticks (the proven `should_commit` path —
/// no fragile 1000 m traversal). `to_realm = dest_stub_config().realm` (→ the DEST-won realm).
pub fn plant_one_crossing_shell(topo: &mut Topology) { plant_crossing_boundaries(topo, vec![ … ]); }
```

**GEOMETRY — decision (a), dwell-triggered, with M3 honesty.** A small `Authority` shell centered at the dot's spawn offset → the dot is a member from spawn, `should_commit` fires on the `n_entry`-th tick. This reuses the proven path and avoids a fragile 100 mm/tick traversal race. **BUT (adversary M3): this means the dot is born-inside-then-commits, NOT a physical traversal.** 3g proves *trigger→transfer composition*; the *pixel-visible physical traversal* (`project_visual_testable_endstate`) remains owed to D-15/D-30 — state this in the 3h DEFERRED note so a later reader doesn't think the crossing-VISUAL is discharged.

**⚠️ M2 — the plant realm-id MUST byte-match what DEST wins.** The boundary's `to_realm` must equal the realm `DEST` actually holds (`dest_stub_config().realm`), asserted in warmup (`report(DEST).held_realms` contains it). If it names a realm no one holds → `handle_crossing_request` counts `crossing_unresolved` and SILENTLY stalls (no saga, green-looking but vacuous). Hard-assert the warmup gate on the exact `to_realm`.

**Test 1 — `crossing_e2e_durable_dot_crosses_a_planted_boundary`** (the headline):
- `p2_cluster(&fabric, 8)` + a `p1_client` with `walk_forward()`.
- Warmup to tick ~80: `report(SHARD).held_entities` non-empty AND `report(DEST).held_realms` contains `dest_stub_config().realm` (M2 gate).
- `plant_one_crossing_shell(&mut topo)` — NO `trigger_transfer`.
- Drive the cut: `step_until` on `saga_states().starts_with("Freezing")`, `resume_input`, `step_until live_sagas()==0`, `pause_input`, settle.
- Steady-state guard (verbatim from `run_cut_transfer`): SOURCE clear of the entity, DEST holds it.
- **Assertions (H2-strengthened):**
  1. **ANTI-VACUITY (causal, not just emit):** `report(ORCH).crossings_started >= 1` (the request became a saga — the load-bearing proof). PLUS `report(SHARD).crossings_requested >= 1` (Leg 1 fired).
  2. **NO HAND-FEED BY CONSTRUCTION:** the test file contains zero `trigger_transfer`/`start_transfer` calls — enforce as a **code-level invariant** (a grep-gate in 3h, or a module-doc contract), not a comment (adversary H2b).
  3. Directory head flipped: `report(ORCH).directory` for `Entity(entity)` → `AuthorityRef::Shard(DEST)`.
  4. Source Ghost / dest Owned: `report(SHARD).ghost_dots.contains(&entity)` (or absent from `held_entities`); `report(DEST).held_entities` contains it.
  5. Positive latch clear: `report(SHARD).crossing_latches_cleared >= 1`, `report(SHARD).crossings_retried == 0` (no ttl re-drive needed on the happy path — `ttl==0` in the shared rig anyway).
  6. `verify_authority_unique` + `verify_authority_settled`.

**Test 2 — `crossing_e2e_renders_at_dest_after_autonomous_crossing`** (render, SEPARATE per HR5(d) — don't `&&` authority + render): same plant + warmup, drive the `capture_subject` observer, assert `dest_sample.frame == FrameRef::SystemSpace{system_seed:7}` (the DEST renders the CROSSED source-realm pose, not its seed-8 origin), `world_pos.is_finite()` + `!= DVec3::ZERO`. Assert the `FrameRef` VALUE (load-bearing), NOT `label()` (untested in e2e).

**Test 3 — `crossing_e2e_pre_cas_abort_clears_the_source_latch`** (the abort leg, M1-guarded):
- Same plant + warmup. Induce an autonomous PRE-CAS abort via a **one-bool inert knob** (no autonomous cluster reject exists — verified): add `pub reject_next_prepare: bool` to `StubConfig` (default `false`, INERT), consumed ONCE by the dest's Prepare handler to reply `PrepareReject::Spatial(SpatialReject::Obstructed)`. **OPEN Q-E: locate the dest-side cluster Prepare handler that the saga's `PrepareSubscribe` drives — it is NOT in `stub.rs` (no reject path there); trace where the dest replies `Prepared`/`PrepareResult` in the cluster and thread the one-shot there.**
- **M1 assertions (prove the crossing-origin path, not a generic abort):** `report(ORCH).pending_abort_replies >= 1` BEFORE the ack round-trip (the `0x39` crossing-origin tombstone staged Mechanism-Y), then `== 0` after. PLUS: `report(SHARD).crossing_latches_cleared >= 1`, `report(SHARD).in_flight_latches` does NOT contain the entity, `live_sagas() == 0`, and the directory head STAYED `Shard(SHARD)` (never moved).

**Test 4 — `crossing_e2e_transient_crosses_via_batch_grant`** (M4 — fully autonomous, no hand-feed):
- Plant the same shell. Insert an OWNED `Debris` transient on SHARD (`TransientStatus::Held`, pose INSIDE the band) — reuse the `seed_transient_crossing` insert shape but Held not Crossing.
- The dwell fires `fan_out_crossing`'s Transient arm → `TransientCrossingRequest` → orchestrator `handle_transient_crossing_request` AUTO-GRANTS (`saga_runtime.rs:1467`) → batch.
- **Assertions:** `report(SHARD).transient_crossings_requested >= 1`, `report(ORCH).transient_crossings_granted >= 1` (M4 — the real proof), `report(DEST).owned_transients` non-empty, `report(DEST).batch_goes` non-empty. This is a genuine HR2 "second class through one machinery" proof — build it fully, no defeatism.

**Test 5 — `crossing_e2e_is_byte_identical_under_same_seed`** (determinism, M2): run the FULL Test 1 + the Test 2 render observer twice under the same seed; assert identical `InspectReport` traces INCLUDING the render sample (float nondeterminism surfaces in the `capture_subject` frame/world_pos, not the authority flip — mirror `p2_transfer_gates.rs:652`).

---

## Sub-slice 3g-C — the crash leg (honest about tier, per H3)

**Tier 1 (harness, in `crossing_e2e.rs`) — `crossing_e2e_abort_survives_orchestrator_restart`:** REUSE `p2_cluster_durable_orch` (retained `MemStore`, `tests/src/lib.rs:292`). Drive Test 3's abort to where `PendingAbortReply` is staged (`report(ORCH).pending_abort_replies >= 1`) but BEFORE the source's `CrossingAbortedAck` lands (park the fabric delivery of the ack). Then `with_orchestrator → rebuild()`; assert post-restart: `pending_abort_replies` restored, the first `scan_deadlines` re-emits `CrossingAborted`, the source clears the latch (`crossing_latches_cleared >= 1`, `in_flight_latches` empty), directory head never moved, `crossings_started` for the transfer stayed 1.
- **⚠️ H3 HONESTY:** this is a **World-rebuild, NOT a `kill -9`**, and the abort is induced by the synthetic `reject_next_prepare` knob. Name the test `..._survives_orchestrator_restart` (NOT `..._kill9`). It proves "a geometrically-triggered crossing's persisted AbortReply survives an orchestrator restart and clears the source latch" — a real cluster-tier composition of the unit-proven mechanism (`rehydrate_restores_a_persisted_abort_reply_and_reemits_on_first_scan`, `saga_runtime.rs:6541`), but not a process SIGKILL.

**Tier 2 (process, OWED — do NOT build now):** a true `kill -9` through the real bins needs the geometric trigger reachable in the `shard` bin (which boots an EMPTY `RealmBoundaries`), i.e. a boundary-plant env knob in the bin — a P4-adjacent lift. The abort-reply CRASH DURABILITY itself is ALREADY process-proven generically by `orchestrator_crash.rs` (any `StoreKey` survives a real `child.kill()`); only the crossing-specific composition is owed. Ledger it.

---

## Sub-slice 3h — gate + DEFERRED flip (HONEST, per H3)

**Gate:** `cargo llvm-cov clean --workspace` (the stale-profdata gotcha — mandatory after interleaved agent builds) → `just coverage-fast` (confirm the new `redrive_*` arms + `topology.rs` scrapes hit 100% region+branch; expect one iteration to catch an uncovered no-op arm — cover it or `#[cfg_attr(coverage_nightly, coverage(off))]` with justification) → `just gate`.

**DEFERRED.md D-43 flips (`:3017-3030`) — do NOT over-claim:**
- Item **#2 (unresolved-dest re-drive)** → 🟩 **but note the site changed**: the fix is NOT the `Entry::Occupied` arm the entry prescribes (that placement is unreachable per C1) — it is the new `redrive_stranded_crossings` latch-scan. **Amend the entry text at `:3021`** to record the corrected site (the entry currently prescribes a defective placement).
- Item **#3 (CrossingProgress re-arm)** → confirm 🟩 (landed `stub.rs:1677`; the scan makes its promise reachable).
- Item **#1 (abort-reply durability)** → confirm 🟩 (Mechanism-Y landed; 3g Test 3 + 3g-C Tier-1 exercise it end-to-end).
- **Pin (`:3029`):** its condition names "3f-E (the **SIGKILL** abort-crash-leg e2e)". Tier-1 is a **restart, not a SIGKILL** (H3). **Either** relax the Pin wording to "orchestrator-restart abort-crash-leg" and flip it, **OR** leave the Pin OPEN and ledger the process-tier SIGKILL crossing-crash as the true discharge. **Recommend: relax the wording + flip** (the restart IS the meaningful durability proof; the process-tier crossing-crash is a distinct owed line, added below).
- Items **#4, #5, #6, #7, #8** → LEAVE 🟧 with a note "task #133 crossing-e2e landed #1/#2/#3; #4-#8 owed at P4/P5 dense-band / Station-Bay." (#7 DRY-id was to fold into 3f-D — but 3f-D4 does NOT change `crossing_transfer_id`'s signature, so #7 stays owed.)
- **ADD a new owed line** under D-43: "process-tier geometric-crossing SIGKILL crash — needs a boundary-plant env knob in the `shard` bin (empty-registry boot); P4-adjacent."
- **ADD a new owed line:** "the pixel-visible PHYSICAL traversal (dot walks across, not born-inside) — 3g proves trigger→transfer composition via a dwell-at-spawn shell; the visual traversal is D-15/D-30."

**Task closure:** update memory `project_spatial_boundary_trigger_plan` + task #133 to DONE for the crossing-e2e capstone, listing the P4/P5-owed items (#4-#8, process-SIGKILL, visual traversal).

---

## OPEN QUESTIONS — need a code read BEFORE coding

- **Q-A (the load-bearing one, `stub.rs` ~`:2379-2419` + the `Dots`/`OwnedTransients` structs):** the latch-scan re-drive needs, per latched entity, the live `to_realm`, `session`, and `subject_fence`. `RequestInFlight` holds only `EntityId → TransferId`. Can `redrive_one_stranded` recover `to_realm` (re-run `crossing_candidates`/`resolve_winner_ix` for the entity's current pose? or store `to_realm` in `CrossingState` at latch time?), `session` (reverse-lookup `Dots` by entity — the map is keyed by `SessionId`), and `subject_fence` (`dot.authority.fence()`)? **If recovery is awkward, the cleaner design is to widen `CrossingState` (or the latch value) to carry `{to_realm, session, fence}` at emit time** — decide this before writing the scan. This is THE design decision of 3f-D4.
- **Q-B (`topology.rs` `inspect_world`):** does it already reach `SagaRuntimeRes` for the ORCH node (for `crossings_started`/`transient_crossings_granted`/`pending_abort_replies`), or must that scrape branch be added?
- **Q-C (`saga_runtime.rs:502`):** `pending_abort_replies` is `pub(crate)` — add a `pub fn pending_abort_replies_len()` getter (mirror `crossings_started()` `:683`).
- **Q-D (`topology.rs` ~`:1051`):** does any unit test assert `InspectReport` by full struct-literal `==`?
- **Q-E (the dest cluster Prepare handler):** where does the dest reply `Prepared`/`PrepareResult` in the CLUSTER (driven by the saga's `PrepareSubscribe`)? It is NOT in `stub.rs`. Thread the one-shot `reject_next_prepare` there. If the dest-prepare reply is entirely gateway/saga-internal with no stub hook, the knob may need to live elsewhere — locate first.
- **Q-F (Q3 in the draft, now RESOLVED as C1):** confirmed — `should_commit` is a strict rising edge; the re-drive MUST be a latch-scan. No longer open.

## What's reachable NOW vs owed

- **Reachable now:** 3f-D4 as a latch-scan (Q-A decides the state-carry shape), 3g durable + render + transient (all machinery autonomous/verified), the abort leg (needs the one-bool `reject_next_prepare` knob — Q-E sites it), the crash leg at HARNESS/restart tier, 3h gate + the D-43 #1/#2/#3 flips + Pin-relax.
- **Owed (ledger, honest):** D-43 #4-#8 (P4/P5 dense-band / Station-Bay); the process-tier SIGKILL geometric-crossing crash (needs a shard-bin boundary-plant knob); the pixel-visible PHYSICAL traversal (D-15/D-30).

**Relevant files:** `crates/sim/src/stub.rs` (new `redrive_stranded_crossings` + `redrive_one_stranded` after `:2420`; `crossings_retried` field `:694`; `reject_next_prepare` on `StubConfig`; unit tests in `mod tests`); `crates/core/src/geometry.rs:642` (the strict-edge `should_commit` — the C1 root, read-only); `crates/node/src/saga_runtime.rs` (`crossings_started` `:1441`, `handle_transient_crossing_request` `:1460`, `is_crossing_origin` `:1155`, `pending_abort_replies` getter `:502`, the dest Prepare-reject site Q-E); `crates/harness/src/topology.rs` (`InspectReport` `:77`+, `inspect_world` `:219`); `tests/src/lib.rs` (`plant_crossing_boundaries`/`plant_one_crossing_shell` near `:409`); `tests/tests/crossing_e2e.rs` (NEW); `crates/bins/src/lib.rs:1081` + `crates/bins/tests/orchestrator_crash.rs` (real-SIGKILL reference, Tier-2 owed); `docs/design/DEFERRED.md:3017-3030` (D-43 flip + amend #2 site).