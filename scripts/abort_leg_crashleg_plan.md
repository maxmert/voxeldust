All seams verified firsthand. Every open question is resolved; the adversary's CRITICAL-1 and CRITICAL-2 are confirmed real against the code. Here is the definitive plan.

---

# DEFINITIVE PLAN — 3g Test 3 (abort-leg) + Tier-1 crash-leg

Every file:line verified against HEAD this session. Q-1..Q-4 all resolved by code read (below). The adversary's **CRITICAL-1** (rehydrate wipes `last_abort_reply_emit`→0) and **CRITICAL-2** (first emit is throttle-gated) are **CONFIRMED against the code** and are folded in as first-class plan structure, not caveats. Two findings the adversary raised turned out already-satisfied by landed 3g infra (`crossing_latches_cleared`/`in_flight_latches` on `InspectReport`) — that shrinks the change set.

## Resolved open questions (was Q-1..Q-4)

- **Q-1 (was load-bearing):** `GatewayConfig` is `#[derive(Resource)]` (`gateway.rs:86`) and inserted live via `world.insert_resource(config)` in `register_gateway` (`gateway.rs:604`). So `arm_gateway_reject` via `resource_mut::<GatewayConfig>()` **works** — **no dedicated-`Res` fallback needed.** The by-value `config: GatewayConfig` param to `register_gateway` (`:602`) is only the constructor; the live resource is mutable.
- **Q-2:** `inspect_world` **already** scrapes the ORCH `SagaRuntimeRes` in one `if let Some(rt)` block (`topology.rs:163-172`, reading `crossings_started`/`transient_crossings_granted`). `pending_abort_replies` slots straight into that existing branch — no new branch.
- **Q-3:** All 21 `InspectReport { … }` literals in `oracle.rs` and the ones in `client.rs`/`topology.rs` use `..InspectReport::default()` (verified: 21 literals, 21 `..default()` spreads). A new `pending_abort_replies: usize` field (defaults `0`) breaks **no** literal. No struct-literal test edit.
- **Q-4:** `AuthorityRef::Shard(NodeId)` is the arm (`directory.rs:47,58`); `.node()` extracts the `NodeId` across both arms. Assert on `.node() == SHARD` (arm-agnostic), not on the enum variant.
- **NEW (shrinks scope):** `InspectReport.crossing_latches_cleared` (`topology.rs:116`), `.in_flight_latches` (`:120`), `.crossings_started` (`:106`) **already exist and are already populated** by `inspect_world` (StubStats scrape `:246-262`, RequestInFlight `:264-267`). These are landed 3g observables. **Only `pending_abort_replies` is a new field.**

---

## (1) The INERT reject knob

**Field** — new on `GatewayConfig` (`gateway.rs:87-125`, after `self_fence_grace_ticks`):

```rust
/// 3g abort-leg (INERT test lever): when `Some`, the NEXT `PrepareSubscribe` that would
/// otherwise reply `Ready` instead replies `Prepared{ result: Rejected(this) }`, then the
/// gateway self-clears it (one-shot). `None` on EVERY cluster = the 1c `Ready` stub,
/// behaviour-identical. The SOLE way to trigger the pre-CAS abort of a crossing-origin
/// durable saga in a cluster, since the gateway (not the dest stub) is the durable Prepare
/// decider (apply_prepare hardcodes Ready today).
pub reject_next_prepare: Option<PrepareReject>,
```

Set `reject_next_prepare: None` in `build_cluster` (`tests/src/lib.rs:200-221`, the one `GatewayConfig { … }` literal) — every existing cluster stays byte-identical.

**Threading (verified path):** `apply_prepare` (`gateway.rs:1318`) is a free `fn`, called from `on_transfer_control` (`gateway.rs:1196`, param `config: &GatewayConfig`), called from `process_gateway_inbound` (`gateway.rs:776`, param `config: Res<GatewayConfig>`) at the dispatch `on_transfer_control(cmd, &config, …)` (`gateway.rs:821`). To `.take()` the one-shot, thread `&mut Option<PrepareReject>` down this chain:

1. `process_gateway_inbound`: `config: Res<GatewayConfig>` → **`mut config: ResMut<GatewayConfig>`** (`gateway.rs:777`). This system already `&config`-reads elsewhere in the body (`config.orchestrator`, `config.is_known_shard`) — `ResMut` derefs read-only fine, no other edit.
2. `on_transfer_control`: add `reject_next_prepare: &mut Option<PrepareReject>` param; at the dispatch (`:821`) pass `&mut config.reject_next_prepare`.
3. `apply_prepare`: add the same param.

Signature:
```rust
fn apply_prepare(
    session: &mut Session, transfer: TransferId, dest: NodeId,
    stats: &mut GatewayStats,
    reject_next_prepare: &mut Option<PrepareReject>,   // NEW — one-shot lever
) -> Option<TransferControlAck>
```

**Reject site + message** — replace the tail (`gateway.rs:1356-1359`). The not-Active guard (`:1324-1327`, returns `None`, bumps `stats.transfer_unroutable`) **MUST stay above** and MUST NOT consume the knob (the lever fires only for a would-be-`Ready` prepare):

```rust
// ... after the Active guard + `session.transfer = Some(TransferProgress { .. })` (unchanged) ...
let result = match reject_next_prepare.take() {          // consume-once; self-clears
    Some(reject) => {
        tracing::debug!(transfer = transfer.0, ?reject,
            "PrepareSubscribe REJECTED by the one-shot reject_next_prepare lever (3g abort-leg)");
        PrepareResult::Rejected(reject)
    }
    None => {
        tracing::debug!(transfer = transfer.0, "PrepareSubscribe readiness is a 1c stub (Ready)");
        PrepareResult::Ready
    }
};
Some(TransferControlAck::Prepared { transfer, result })
```

Armed with `PrepareReject::Spatial(SpatialReject::Obstructed)`, the gateway emits `Prepared{ transfer, result: Rejected(Spatial(Obstructed)) }` → `reply_ack` (`gateway.rs:957`) → orchestrator `SagaEvent::Prepared(Rejected(..))` → FSM `(S::Preparing, E::Prepared(Rejected(reject))) => abort_from_pre_freeze(ctx, AbortReason::PrepareRejected(reject))` (`saga.rs:804-806`) → `abort_from_pre_freeze` (`:1341-1356`) yields `Aborting{awaiting_thaw:false, awaiting_abort_ack:true}` + `[Send(AbortTransfer), NotifyRejected]` — **no `ThawSource`, no `IssueCommitCas`. The directory CAS never runs; source authority never hands off.** Pre-CAS by construction. No wire change (`PrepareResult`/`PrepareReject`/`SpatialReject` all exist).

---

## (2) Test 3 (abort-leg) — `crossing_e2e_pre_cas_abort_clears_the_source_latch`

New test in `tests/tests/crossing_e2e.rs`. Reuses the `run_autonomous_crossing` warmup shape (`:127-197`) but arms the gateway between the M2 gate and the plant. Since `run_autonomous_crossing` builds its own cluster internally, **inline the warmup+plant** into this test (or add a `reject: Option<PrepareReject>` param to a shared internal driver — inline is simpler for the abort divergence).

Ordered steps + assertions:

```rust
#[test]
fn crossing_e2e_pre_cas_abort_clears_the_source_latch() {
    let fabric = FaultFabric::new(0x3F_AB07, 2);
    let mut topo = p2_cluster(&fabric, 8);
    topo.add_node(Box::new(p1_client(&fabric, CLIENT, AccountId(1000), walk_forward())));

    // 1. WARMUP + M2 gate (verbatim run_autonomous_crossing:140-163): SHARD holds the avatar,
    //    DEST holds dest_stub_config().realm. Capture `subject` (report(SHARD).held_entities.first()).
    step_until(&mut topo, 80, &mut |_| {}, |t| /* held_entities && held_realms */);
    let subject: EntityId = /* report(&topo.inspect_all(), SHARD).held_entities.first() */;

    // 2. PRE-CONDITION (H-2, cheap namespace guard): assert the id the source will latch is 0x39.
    //    crossing_transfer_id(subject, subject_fence, attempt=0) high byte must be 0x39 —
    //    so a driver namespace regression fails HERE with a precise message, not later as
    //    "no pending reply". (subject_fence read from the directory record.)
    assert_eq!(expected_crossing_id(subject, fence) .0 >> 120, 0x39, "the crossing id is 0x39-namespaced");

    // 3. ARM the one-shot reject AS LATE AS POSSIBLE (L-1): immediately before the plant, so no
    //    warmup/re-driven prepare can spend it on the wrong transfer.
    arm_gateway_reject(&mut topo, PrepareReject::Spatial(SpatialReject::Obstructed));

    // 4. PLANT the crossing shell (mints the 0x39 crossing on the geometric dwell).
    plant_one_crossing_shell(&mut topo);

    // 5. Drive to terminal, CAPTURING the transient pre-ack staging (the load-bearing proof).
    let mut saw_pending_pre_ack = false;
    let mut crossings_after_plant = 0u64;
    step_until(&mut topo, 60, &mut |t| {
        let r = t.inspect_all();
        if report(&r, ORCH).pending_abort_replies >= 1 { saw_pending_pre_ack = true; }
        crossings_after_plant = report(&r, ORCH).crossings_started;
    }, |t| live_sagas(t) == 0);

    // ---- ASSERTIONS ----
    let r = topo.inspect_all();

    // (a) NON-VACUITY / Mechanism-Y ran: the 0x39 crossing-origin durable tombstone STAGED a reply
    //     BEFORE the source ack reaped it. This is the discriminator — a non-crossing-origin abort
    //     stages NOTHING (saga_runtime.rs:1212 gate), so >=1 proves the 0x39 path specifically.
    assert!(saw_pending_pre_ack,
        "pending_abort_replies >= 1 BEFORE the ack — the 0x39 crossing-origin reply staged");

    // (b) exactly ONE crossing started after the plant (L-1: proves the knob wasn't spent on a
    //     stray prepare and a second crossing didn't sneak through Ready).
    assert_eq!(crossings_after_plant, 1, "exactly one durable crossing saga started post-plant");

    // (c) the reply was reaped by the source's CrossingAbortedAck.
    assert_eq!(report(&r, ORCH).pending_abort_replies, 0, "== 0 after the ack reaps it");

    // (d) the source cleared its crossing latch (POSITIVE clear, not a timeout).
    assert!(report(&r, SHARD).crossing_latches_cleared >= 1);

    // (e) in_flight_latches no longer contains the subject.
    assert!(!report(&r, SHARD).in_flight_latches.contains(&subject));

    // (f) no live sagas.
    assert_eq!(live_sagas(&mut topo), 0);

    // (g) PRE-CAS PROOF (H-1: head-position is necessary-not-sufficient, so pair it):
    //     (g1) the directory head STAYED at the source — authority never moved.
    let head = /* orch DirectoryRes head(subject).authority.node() */;
    assert_eq!(head, SHARD, "pre-CAS abort: head never moved off SOURCE");
    //     (g2) NO dest adoption ran — the sufficient half. The dest never held/adopted the subject
    //          (a post-CAS-then-compensate path would have advanced a dest ownership counter).
    assert!(!report(&r, DEST).held_entities.iter().any(|(e, _)| *e == subject),
        "pre-CAS: the dest never adopted the subject (no CAS ran)");
}
```

Notes:
- **(a) is the sole load-bearing non-vacuity assertion** (adversary H-2, M-1): `pending_abort_replies` is inserted at abort-commit (`saga_runtime.rs:1218`) and removed the tick the ack lands (`:2344`). Asserting only the final `==0` would be vacuous; the transient `>=1` proves the Mechanism-Y path ran. Because the first `CrossingAborted` emit is throttle-gated (`now≥8`, see crash-leg), the entry persists ≥8 ticks before the ack round-trips, so the observation window is **wide** — `saw_pending_pre_ack` fires reliably (M-1: determinism is fine; the transient window is not narrow here).
- **(g2)** is the H-1 fix: head-stayed-at-SHARD is necessary but not sufficient (a post-CAS-then-compensate also leaves head at SHARD). The dest-never-adopted assertion is the sufficient half. Combined with (a) (Mechanism-Y stages ONLY on the pre-freeze tombstone), pre-CAS is fully pinned.
- `arm_gateway_reject(topo, reject)` — new helper in `tests/src/lib.rs`: `with_node(topo, GATEWAY, |gw| gw.world_mut().resource_mut::<GatewayConfig>().reject_next_prepare = Some(reject))`. Q-1 confirms `resource_mut::<GatewayConfig>()` is valid.

---

## (3) Tier-1 crash-leg — `crossing_e2e_abort_survives_orchestrator_restart` — RESHAPED per CRITICAL-1/2

**The adversary's two CRITICALs are confirmed against the code and REDEFINE this leg:**

- **C-1 CONFIRMED:** `rehydrate` builds a fresh `SagaRuntimeRes::with_tunings(...)` (`saga_runtime.rs:1373`) and restores `pending_abort_replies` (`:1377`) but **does not restore `last_abort_reply_emit`** — it defaults to `UniverseTick(0)`. The re-emit gate `now.0 - last_abort_reply_emit.0 >= redrive_deadline_ticks` (`:2012`, default 8 per `saga.rs:80`) is then trivially true post-restart (resumed `now` ≫ 8). Confirmed by the existing unit test comment at `:6756`: *"The first post-restart scan (last_abort_reply_emit == 0) re-emits immediately."* → **A "heal → re-emit → reap" assertion is structurally guaranteed by the throttle reset and proves nothing about pre-restart state.**
- **C-2 CONFIRMED:** the reply is *staged* at commit tick T (`saga_runtime.rs:1218`) but the first `CrossingAborted` emit waits for a `scan_deadlines` with `now≥8` (`:2012,2020`). A `step_until` predicate on *staging* (`pending_abort_replies>=1`) can trip at T≪8, i.e. **before any emit ever fires** → the "parked ack mid-flight" window may not exist.

**Decisive context I found:** the WAL-restore mechanism is **already proven at the unit tier** by `rehydrate_restores_a_persisted_abort_reply_and_reemits_on_first_scan` (`saga_runtime.rs:6722-6766`) — it seeds a persisted `AbortReply`, calls `rehydrate`, asserts the entry is restored AND the first post-restart `scan_abort` re-emits. **So the crash-leg e2e must NOT re-prove WAL survival (that's covered); its job is (i) capture-before-heal to prove the entry rode the WAL in a *cluster* rehydrate, and (ii) a mandatory negative control.** The heal-tail is a labeled liveness check, not the proof.

Ordered steps + assertions:

```rust
#[test]
fn crossing_e2e_abort_survives_orchestrator_restart() {
    let fabric = FaultFabric::new(0x3F_AB08, 2);
    let (mut topo, store) = p2_cluster_durable_orch(&fabric, 8);   // RETAINED MemStore
    topo.add_node(Box::new(p1_client(&fabric, CLIENT, AccountId(1000), walk_forward())));

    // WARMUP + M2 + capture `subject` + 0x39 pre-condition (as Test 3).
    // ARM the gateway reject (late); PLANT the shell.
    arm_gateway_reject(&mut topo, PrepareReject::Spatial(SpatialReject::Obstructed));
    plant_one_crossing_shell(&mut topo);

    // C-2 FIX — gate the restart on "an emit has occurred AND its ack is parked", NOT on staging.
    // (i) PARK the ack direction FIRST (SHARD -> ORCH), so once the emit fires and SHARD acks, the
    //     ack sits in `unacked` (partitioned = block+requeue, no drop; fabric.rs:34/233). Leave
    //     ORCH -> SHARD healthy so the emit flows.
    fabric.set_policy(SHARD, ORCH, LinkPolicy { partitioned: true, ..Default::default() });
    // (ii) Drive until the SOURCE has RECEIVED a CrossingAborted (proving ORCH emitted at now>=8)
    //      AND cleared/attempted its latch — i.e. a SHARD-side received-abort is observable. Use the
    //      SHARD's crossing latch state: the abort emit -> source receives -> source pushes ack
    //      (now parked) -> source clears the latch. So gate on the SOURCE side, not the staging side.
    step_until(&mut topo, 80, &mut |_| {}, |t| {
        let r = t.inspect_all();
        // the ORCH still has the entry (ack parked, not reaped) AND the SOURCE already reacted
        // (latch cleared) => an emit DID fire and its ack is in flight/parked.
        report(&r, ORCH).pending_abort_replies >= 1 && report(&r, SHARD).crossing_latches_cleared >= 1
    });
    // Assert BOTH halves of the real window before restart:
    let pre = topo.inspect_all();
    assert!(report(&pre, ORCH).pending_abort_replies >= 1, "entry still staged (ack parked, not reaped)");
    assert!(report(&pre, SHARD).crossing_latches_cleared >= 1,
        "an emit FIRED pre-restart and the source acked (the ack is the thing being parked)");

    // REBUILD the orchestrator from the RETAINED store.
    rebuild_orchestrator(&mut topo, &fabric, store.clone());

    // C-1 FIX — THE LOAD-BEARING ASSERTION: capture the instant AFTER rebuild, BEFORE any healed step.
    // This (and only this) proves the entry rode the WAL — not the throttle reset, not in-process survival.
    assert!(report(&topo.inspect_all(), ORCH).pending_abort_replies >= 1,
        "the persisted abort-reply was restored from the WAL by the CLUSTER rehydrate (pre-heal capture)");

    // HEAL + reap — a LABELED liveness check (NOT the crash proof; C-1: the post-restart re-emit is
    // throttle-reset-guaranteed, so this only shows the tail completes, not that the WAL carried state).
    fabric.set_policy(SHARD, ORCH, LinkPolicy::default());
    step_until(&mut topo, 60, &mut |_| {}, |t| report(&t.inspect_all(), ORCH).pending_abort_replies == 0);

    let r = topo.inspect_all();
    assert_eq!(report(&r, ORCH).pending_abort_replies, 0, "liveness: reaped post-restart by the redelivered ack");
    assert!(report(&r, SHARD).crossing_latches_cleared >= 1);
    assert!(!report(&r, SHARD).in_flight_latches.contains(&subject));
    let head = /* orch head(subject).authority.node() */;
    assert_eq!(head, SHARD, "head never moved — pre-CAS abort survived the restart");
}
```

**MANDATORY negative control (C-1) — separate test `crossing_e2e_abort_reply_absent_from_empty_store_rebuild`:** rebuild an identical fixture against `MemStore::new()` (empty) and assert `report(ORCH).pending_abort_replies == 0` immediately post-rebuild. Without this, the pre-heal capture is confounded by in-process-survival illusion; with it, the WAL-restore assertion is proven to be reading the store, not a leftover. This is the anti-theater control the adversary correctly upgraded from "optional" to mandatory.

**Honesty (tier):** name it `..._survives_orchestrator_restart`, NOT `..._kill9`. It is a **World-rebuild via `rebuild_orchestrator`** (`tests/src/lib.rs:1353` — `reregister(ORCH)` + `register_orchestrator_with_store` + `replace_node`), NOT a process `kill -9`. The abort-reply crash *durability* is separately process-proven generically by `orchestrator_crash.rs`; the *crossing-specific* SIGKILL stays owed (see ledger).

**Park caveat:** `set_policy` is per-directed-link. During the abort tail the only SHARD→ORCH Saga-class traffic is the `CrossingAbortedAck`; ORCH→SHARD stays healthy so the emit flows. Acceptable.

---

## (4) The two new accessors

**Getter** — `pending_abort_replies` is private (`saga_runtime.rs:502`). Add in the getter block (near `:683`, mirroring `crossings_started()`):
```rust
#[must_use]
pub fn pending_abort_replies_len(&self) -> usize { self.pending_abort_replies.len() }
```

**Harness field** — add `pub pending_abort_replies: usize` to `InspectReport` (`topology.rs:24`, near the existing `crossings_started` at `:106`), and populate it inside the **existing** ORCH `SagaRuntimeRes` branch (`topology.rs:163-172`, right after `report.crossings_started = rt.crossings_started();`):
```rust
report.pending_abort_replies = rt.pending_abort_replies_len();
```
No new branch (Q-2). No struct-literal edits (Q-3). `crossing_latches_cleared`/`in_flight_latches`/`crossings_started` already exist and are already populated — nothing owed there.

---

## (5) HR5 100% + determinism

**Reject-knob two arms (gateway `mod tests`, `gateway.rs`)** — drive `apply_prepare` directly with an Active session, `assert_eq!` on the full ack (HR5(d), never `matches!`):
1. `apply_prepare_inert_lever_replies_ready` — `&mut None` → `Some(Prepared{ result: Ready })` AND lever stays `None`. (the `None` arm)
2. `apply_prepare_armed_lever_rejects_once_then_self_clears` — `&mut Some(Spatial(Obstructed))` → first call `Some(Prepared{ result: Rejected(Spatial(Obstructed)) })` + lever now `None`; a SECOND call on a fresh Active session → `Ready`. (the `Some` arm + one-shot self-clear)
3. `apply_prepare_not_active_does_not_consume_the_lever` — armed lever + not-Active session → `None`, `stats.transfer_unroutable += 1`, lever STILL `Some`. (guard precedence — the lever survives an inactivity-rejected prepare)

Per HR5(c) the gateway unit tests **own** the two-arm coverage; the e2e tests are composition proof (HR5: integration exercises the full surface or none — here the armed path end-to-end).

**Cite the existing coverage owners (H-3), don't silently create new expectations:**
- The `scan_deadlines` re-emit throttle/reap arms (`saga_runtime.rs:2012-2042`) are **already owned** by `scan_deadlines_reemits_a_pending_abort_reply_once_per_cadence` (`:6561`), `..throttles..` (`:6586`), `..reaps_a_pending_reply..` (`:6601`), `..reemits_while_owned_then_reaps..` (`:6623`). The new e2e naturally hits only `reemit_due==true`+`owner==source`; the other arms stay owned by these unit tests.
- The **non-empty** rehydrate abort-scan arm (`saga_runtime.rs:1364`) is **already owned** by `rehydrate_restores_a_persisted_abort_reply_and_reemits_on_first_scan` (`:6722`). The **empty** arm is hit by the mandatory `MemStore::new()` negative control (§3) plus any non-abort recover. Add an explicit assertion in the control, don't assume.

**New accessors:** `pending_abort_replies_len()` is a straight-line `.len()` (one region, covered by any test reading it — both e2e do). The `InspectReport.pending_abort_replies` copy lives inside the existing `if let Some(rt)` arm (already covered); the `None` arm defaults (covered by any non-ORCH scrape, e.g. `topology.rs:1103`).

**Determinism:** the reject is a pure fn of `(reject_next_prepare, session.phase)`; `.take()` and the `Copy` `PrepareReject` are deterministic; the pre-freeze abort is CAS-free/fence-neutral (no ordering nondeterminism). Crash-leg: seeded fabric, `pending_abort_replies` is a `BTreeMap` (deterministic re-emit iteration, `:1363`), the rehydrate scan (`store.scan(&[ABORT_REPLY])`) is ordered, `set_policy` park/heal are explicit (no seeded delay). Optional determinism sibling: run Test 3 twice under one seed, assert identical `InspectReport` traces (mirrors `p2_transfer_gates.rs:652`) — cheap, not required.

---

## Files touched

- `crates/connection-plane/src/gateway.rs` — `GatewayConfig.reject_next_prepare` (`:125`); `process_gateway_inbound` `Res`→`ResMut<GatewayConfig>` (`:777`) + `&mut config.reject_next_prepare` at dispatch (`:821`); `on_transfer_control` param (`:1196`); `apply_prepare` param + reject tail (`:1318,:1356`); 3 unit tests in `mod tests`.
- `crates/node/src/saga_runtime.rs` — `pending_abort_replies_len()` getter (near `:683`). **No** change to staging/reap/rehydrate (all exist).
- `crates/harness/src/topology.rs` — `InspectReport.pending_abort_replies` field (`:24`); one populate line in the existing ORCH branch (`:172`). `crossing_latches_cleared`/`in_flight_latches`/`crossings_started` **already present** — untouched.
- `tests/src/lib.rs` — `arm_gateway_reject` helper; `reject_next_prepare: None` in `build_cluster` (`:200`).
- `tests/tests/crossing_e2e.rs` — `crossing_e2e_pre_cas_abort_clears_the_source_latch`, `crossing_e2e_abort_survives_orchestrator_restart`, `crossing_e2e_abort_reply_absent_from_empty_store_rebuild` (mandatory negative control) (+ optional determinism sibling).
- `docs/design/DEFERRED.md` D-43 — flip the abort/restart lines; add the owed process-tier crossing-SIGKILL line.

No wire change. No `stub.rs` change (the dest never sees the Prepare — the gateway is the decider).

## Owed (ledger, honest)

- **Process-tier crossing SIGKILL** — a real `kill -9` of the orchestrator through the `orchestrator` bin mid geometric-crossing-abort. Blocked on a boundary-plant knob in the `shard` bin (boots empty `RealmBoundaries`, no in-band way to plant a crossing shell across the process seam). The abort-reply crash durability is already process-proven *generically* by `orchestrator_crash.rs`, and *WAL-restore-specifically* by the `rehydrate_*` unit test — so the crossing-SIGKILL is a composition gap, not a durability gap. New owed line under DEFERRED D-43; keep the Tier-1 test named `..._survives_orchestrator_restart` to hold the distinction.

## Remaining OPEN QUESTION needing a code read before coding

- **OQ-1 (crash-leg gate wording, not a blocker for the abort-leg):** §3 gates the pre-restart window on `report(SHARD).crossing_latches_cleared >= 1` as the "an emit fired + source acked" signal. Confirm no *other* path can bump `crossing_latches_cleared` during the parked-ack window (it should be reachable only via `on_crossing_aborted`'s exact-id-match clear, `stub.rs:1671-1681`, i.e. only after receiving a `CrossingAborted` — which is exactly what we want). Verify `on_crossing_aborted` is the sole writer of `crossing_latches_cleared` (a one-line grep before coding the gate); if another writer exists, gate instead on a dedicated SHARD-side "CrossingAborted received" counter. Everything else is fully resolved.