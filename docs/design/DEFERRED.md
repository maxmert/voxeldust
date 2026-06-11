# DEFERRED — the binding registry of stubs, interims, and not-yet-built proper solutions

This is the **single source of truth** for every place the codebase ships a working interim/stub today with a
*proper solution owed later*. The transfer-first mandate means we accept "working now" — but **never silently**:
every entry below is also pinned in-code (a `tracing::warn`, a `KNOWN LIMIT` module doc, and/or an
"exists-to-be-flipped" test assertion) so it cannot be forgotten, and every entry names **what** is missing,
**where** the interim lives, **when** (which slice/phase) the proper version lands, and its **dependency**.

Discovered and hardened by the P1.5/P2 adversarial audits (workflow run-ids in each entry). Update this file
whenever an interim ships or a proper solution lands (flip the status, keep the history).

Status legend: 🟥 not started · 🟧 interim shipped (proper owed) · 🟩 proper solution landed.

---

## P2 — THE TRANSFER (in progress)

### D-1 🟧 Transfer abort never clears the directory lock (`in_transfer` leak)
- **Missing:** on every abort path the saga leaves `OwnerRecord.in_transfer = Some(transfer)` forever, so the
  subject key can never transfer again (and ghost-despawn stays refused). R1 re-manifesting at the directory.
- **Where:** `crates/node/src/saga_runtime.rs` — no `abort_cas` call site anywhere; the abort tests pin
  `assert_eq!(head.in_transfer, Some(XFER))` ("exists-to-be-flipped") + a `KNOWN 1b LIMIT` module-doc block.
- **When / proper:** **Slice 2.** Terminal abort clears the lock via `DirectoryCore::abort_cas` — INCLUDING a
  stale-fence re-read (a CAS-loser's `expected_fence` is stale by definition, so a naive `abort_cas(expected)`
  also loses; the Slice-2 CAS-re-read loop re-derives the head first). The Slice-2 fix flips the pinned asserts.
- **Dependency:** the Slice-2 at-least-once / adaptive-timeout / CAS-re-read machinery.
- **Source:** Slice-1b audit `wf_34ef74d1` (CPO-1/CPO-2).

### D-2 🟥 Demote→Release has no producer (the saga parks in `Demoting` — the `interim_demote_complete` seam is NOT yet built)
- **Missing:** ANY producer of `DemoteComplete`. Today a committed durable saga reaches `Demoting` and PARKS —
  there is no caller of `deliver(.., DemoteComplete)` of any kind (neither the 1c interim stand-in NOR the
  proper predicate). The proper, event-driven producer tears the source ghost down (a kinematic collider, so
  players collide across the boundary) ONLY when the dest has **delivered the entity to all observers** (a
  server-side watermark, never a client ack — HR1) **AND** the entity has **left the source overlap band**
  (`OverlapBand::update_membership` false on the swept segment `segment_shell_crossing`, gated by
  `width_safe_for`/`K_SAFETY` — velocity-aware hysteresis, NO magic tick count).
- **Where (today):** `crates/node/src/saga_runtime.rs` — the `Committed→Demoting` transition is reached
  (saga_runtime.rs ~line 549 documents the park) but NO `interim_demote_complete` fn exists; the saga stays
  live in `Demoting`. The FSM tail (`Swapping→Demoting→Releasing→Done`) IS complete + proptested in
  `crates/sim/src/saga.rs` (so the park is now ADMIN-VISIBLE via AAA-1 — `admin_snapshot.sagas` shows a saga
  stuck in `Demoting`). The park is also pinned by the `KNOWN 1b LIMIT` module doc.
- **ROB-DEMOTE-PARK consequence (audit `wdznr0x6i`):** because a parked saga never reaches a terminal, it is
  never tombstoned, so `SagaRuntimeRes.sagas` GROWS with every successful transfer under churn. This is NOT
  analogous to the `rejected` LOG (which got a counted-drop cap, D-22/ROB-1): a parked saga is a LIVE in-flight
  transfer, so a force-cap-and-drop would ABANDON the player's transfer — INCORRECT. The ONLY correct bound is
  THIS entry's producer (terminal → tombstone). The growth is admin-visible (AAA-1), never silent; it is inert
  until real transfer churn exists (1d). Do NOT add a dropping cap to `sagas`.
- **When / interim (1c step):** add ONE loudly-named `interim_demote_complete` seam — an unconditional bandless
  stand-in that calls `deliver(.., SagaEvent::DemoteComplete)` (a new CALLER of the existing sink — NOT a magic
  tick, NOT an in-flight-queue injection) + a `tracing::warn` + an "exists-to-be-flipped" test. This drives
  `Swapping→Demoting→Releasing→Done` end-to-end NOW, replacing the 1b silent park.
- **When / proper:** the **post-1d band/ghost P2 slice** swaps that ONE function's body (unconditional → the
  conjoined predicate). **Verified reshape-free (`wf_0ed2dc0c`):** ZERO change to the FSM / `deliver` /
  `run_to_quiescence` / the gateway `ReleaseSubscribe` handler / the frozen `TransferControl` vocabulary.
- **Dependency:** Slice 1d (per-entity `Authority` attach so the dest owns + emits the post-commit entity) +
  the ghost-as-collider + band-instance + observer-delivery-watermark machinery (the band/ghost slice).
- **Source:** 1c design `wf_f3eae69e` + the demote refinement `wf_0ed2dc0c`; status corrected by whole-codebase
  audit `wwg7ydm9y` (the 🟧→🟥 reconcile: no interim seam had actually shipped).

### D-3 🟥 Lease lifecycle: no renewal producer, no expiry reaper (TTL unenforced)
- **Missing:** `OwnerRecord.lease_expires` is written on every grant/renew/commit, but NO node sends
  `LeaseRenew` (no heartbeat producer) and nothing reads `lease_expires` to reap a lapsed record — so a
  crashed node's keys are immortal. Inert for P1's fixed roster; a broken promise the moment nodes can die.
- **Where:** `crates/sim/src/directory.rs` (`renew`/`lease_expires` written, never consumed);
  `crates/node/src/orchestrator.rs` (the only `LeaseRenew` consumer is the receive arm). Pinned by the
  `OwnerRecord.lease_expires` doc in `crates/wire/src/seams/directory.rs` (TTL-ENFORCEMENT-UNIMPLEMENTED).
- **When / proper:** **before the P2 crash/stagger matrix and P3 chaos with real nodes.** Both halves: a
  renewal heartbeat from each authority holder (gateway for `Session`, shard for `Realm`/`Entity`) + an
  orchestrator expiry sweep gated on **unreachable-confirmation** (`transfer_protocol.md`: lapsed lease ⇒
  ownership loss ONLY when the owner is confirmed unreachable — an orchestrator outage freezes recovery,
  never mass-orphans).
- **Dependency:** none structural (the plumbing — `renew`/`revoke`/`now`/`entries` — exists); needs the two
  driver systems + the unreachable-confirmation gate.
- **Source:** whole-codebase audit `wf_43fea0dd` (XSI-1 / SCALE-A).

### D-4 🟥 The reliable `EventMsg` client `MsgClass` arm — carrier for BOTH AoI eviction AND cross-shard signals
- **Missing:** `MsgClass` has Control/Saga/Snapshot/Input/Membership — **no `Event`/`Bulk` arm** — so the
  reliable G→C `EventMsg` family has no transport class to arrive on. TWO independent consumers need this ONE
  arm (build it once; it is shared infra, never per-feature — HR3/DRY):
  - **(a) Per-entity AoI eviction (P2):** the client `DeliveredView` is bounded per-SUB (`drop_sub` on
    `SubscriptionClosing`), but a still-open sub's departed entities are not evicted — that needs
    `EventMsg::EntityRemoved` routed to a per-entity `drop`. Retires the foundation-audit `scalability-1` root.
  - **(b) Cross-shard functional-block SIGNALS → client (P9):** the signal system's gameplay events
    (damage/destroyed/notice and functional-block signal deliveries that surface to the player) ride the SAME
    reliable `EventMsg` arm to the client — opened causally after `SubscriptionOpened` (connection-plane X1).
    The cross-SHARD half of signals rides `InterShardFlow::Signal` (a RESERVED arm, P9) over the N-peer mesh +
    Galaxy Relay; this `EventMsg` arm is only the final shard→gateway→client leg. **No rewrite to land it** —
    the whole-codebase audit `wwg7ydm9y` confirmed signals are additive into the current seams (effect_class
    is exhaustive, the capability lattice already carves `SignalGraphCap`/`GalaxyRelay`).
- **Where:** `crates/client/src/view.rs` doc ("the remaining P2 piece"); `crates/sim/src/io/mod.rs` (`MsgClass`,
  no Event arm); `crates/wire/src/channels.rs` (`EventMsg{Notice,EntityRemoved}` + `BulkMsg{Blob}` declared but
  unroutable); `crates/client/src/net.rs` (routes only Control/Snapshot; others → `ignored`).
- **When / proper:** **P2** adds the arm + `EntityRemoved` eviction (consumer a); **P9** adds the signal
  deliveries (consumer b) on the same arm. Building the arm is registry-only here — no code lands today.
- **Source:** P1.5 foundation audit (deferral) + whole-codebase audits `wf_43fea0dd` / `wwg7ydm9y` (SIG-1).

### D-5 🟥 Client cut cycle (the `net.rs` marker emit) — Slice 1e
- **Missing:** on `ServerControlMsg::RequestCut` the client must record it and emit the triple-sent in-band
  `CUT_MARKER` (`is_cut_marker=true`) on its INPUT flow + a reliable `ClientControlMsg::CutEmitted` on CONTROL.
- **Where:** `crates/client/src/net.rs` — `on_control` currently IGNORES `RequestCut` at the catch-all;
  `assemble_input`/`send_outbound` build `InputDatagram` with `is_cut_marker` hardcoded `false`.
- **When / proper:** **Slice 1e.** (HR1: the client only emits a marker — it never participates in the
  transfer. Slice 1c builds + tests the gateway cut partition with an injected/scripted marker.)
- **1c.2/1c.3 landed (gateway side):** the cut-marker OBSERVER (`on_cut_marker`) consumes an injected
  `is_cut_marker=true` `InputDatagram` on the INPUT flow (1c.3: via `peek_is_cut_marker`, not a full decode) and
  replies `CutConfirmed{marker_seq}` (journaled at step 1; a triple-sent marker re-sends the same ack). 1c.3 also
  INSTALLS the route cut: `FreezeSource` → `apply_freeze` → `store_cut(Some(SeqCut{marker_seq, dest}))`; **1c.4**
  SWAPS the route: `CommitAuthority` → `apply_commit` → `store_commit` (authority := `cut.dest`, fence CARRIED,
  cut := None); `Committed` rides the existing redelivery gate (idempotent, never re-swaps). **1c.5** lands the
  partition: `route_input` returns `Buffer` for `seq > marker`, the gateway holds it in `dest_buffer`, and
  `apply_commit` opens the dest `OpenInputSlot{resume_from_seq=marker}` then drains the buffer to the dest as
  `SessionInput` (in seq order). This lands the TWO HALVES (gateway buffer/drain + dest provisional slot); the
  end-to-end cross-crate INPUT-CONSERVATION gate is OWED (D-28, the 1c.7 gate) — the two halves are never joined
  in one scenario yet, and the recovery producer for a lost buffer does not exist (D-29). The **client** emit
  remains 🟥 for 1e. The `CutEmitted` CONTROL
  message stays an inert no-op until 1e (1c uses the input-flow marker, not `CutEmitted`).
- **Source:** the P2 vertical-slice plan + Slice 1c.2 design `wf_726a51bc` + Slice 1c.3 design `wf_a46c0d9b`.

### D-6 🟧 `PersistCheckpoint` is an in-memory no-op (no durable saga WAL)
- **Missing:** the saga's two durable checkpoints are in-memory only; an orchestrator restart loses every
  in-flight saga.
- **Where:** `crates/node/src/saga_runtime.rs` — `SagaAction::PersistCheckpoint` arm is a LOUD `tracing::warn`
  stub.
- **When / proper:** **P3** — group-committed redb WAL (persist at PREPARE-ack + COMMIT-CAS; re-drive
  idempotently to terminal on restart).
- **Source:** the P2 plan + Slice-1b audit (ROB-2 loud-stub hardening).

### D-7 🟧 Transient transfer: `IssueTransientGo` parks (no batched go-token flow)
- **Missing:** the batched `TransientGo` commit path — a Transient subject reaches the commit point and issues
  the go-token, but there is no flow to complete it, so the saga parks in `CommittingCas` with the key locked.
- **Where:** `crates/node/src/saga_runtime.rs` — `SagaAction::IssueTransientGo` is a LOUD `tracing::warn` stub
  (the FSM `commit_action` fan-out is real + class-aware from line one — HR2).
- **When / proper:** **P3** — the batched `TransientTransferBatch` + `TransientGo` go-token + held-sets, in the
  crash/chaos matrix. The go-token completion must drive the FSM out of `CommittingCas` AND clear `in_transfer`
  — OR `start_transfer` must assert transients never lock CAS-backed `Entity` keys (held-set anchored instead).
- **Source:** the P2 plan + Slice-1b audit (CPO-3).

### D-8 🟧 Transfer dest-input buffer: SOFT cap landed (1c.5); the durable interval-map backstop owed (1d/P3)
- **Landed (1c.5):** the gateway `seq > marker` cut buffer `TransferProgress.dest_buffer: VecDeque<Vec<u8>>`
  (cold `Session`), the `route_input` partition READ (`seq > marker → Buffer`), the drain to the dest at
  `apply_commit`, AND a never-silent SOFT cap: `TransportTuning.max_buffered_inputs` + a counted drop-oldest
  (`GatewayStats.dest_inputs_dropped`, latest-wins input). The cap shipped WITH the fill (NOT deferred) because
  a parked saga (`ReleaseSubscribe` not yet landed; `Bye`-mid-transfer, D-23) can hold the cut open across
  unbounded ticks — the "bounded by one round-trip" premise is false the moment the partition goes live.
- **Still owed (1d/P3):** the gateway buffer is RAM-only soft-state and **1c.5 has NO re-drive producer** —
  `apply_commit` `std::mem::take`-drains the buffer and emits `OpenInputSlot` exactly ONCE, and the saga is
  forward-only past `Committed` (`saga_runtime.rs`), so any drop AFTER that point is PERMANENT: (i) a gateway
  crash mid-cut loses the RAM buffer; (ii) a dest that drops `OpenInputSlot` because its realm lease is late at
  commit (`StubStats.input_slots_deferred`) discards the same-batch `SessionInput` as `UnknownSession`; (iii) the
  soft-cap drop-oldest. The DURABLE cross-shard backstop is the **seq-range interval map** (per-txn,
  reload-from-directory on reconnect; integration.json #1) and the re-drive producer that re-issues the slot +
  re-supplies the buffer — both 1d/P3 (registered as D-29). A HARD/tunable abort-on-overflow POLICY (vs today's
  latest-wins drop) also lands then; the soft cap shrinks that to a policy swap.
- **NAMED LOSS POINT (owed honesty, not yet fixed):** `apply_commit` drains the whole buffer in ONE tick as
  UNRELIABLE `MsgClass::Input` frames — the exact class send-shed (`OutboundStagingCap`) and `BoundedInbox` drop
  FIRST under congestion, while the reliable `OpenInputSlot` survives. So the conserved resume batch is the
  designated shed casualty exactly when it matters (many concurrent crossings = congestion), counted only as
  generic `staging_shed`/`dropped_unreliable`, NOT transfer-attributable. 1c.5 mitigations: the burst is bounded
  by `TransportTuning::DEFAULT_MAX_BUFFERED_INPUTS` (256, an order of magnitude below the per-tick transport caps
  — the drain-burst invariant). The real fix — a RELIABLE `GatewayToShard::ResumeInput` carrier OR making the
  durable seq-range interval map the authoritative conservation mechanism — is a 1d/P3 design decision (the loss
  is inert in single-process 1c.5; the in-mem transport does not shed).
- **Where:** `crates/connection-plane/src/gateway.rs` (`TransferProgress.dest_buffer`, `route_input`,
  `on_client_input` Buffer arm, `apply_commit` drain, `TransportTuning::DEFAULT_MAX_BUFFERED_INPUTS`).
- **Source:** 1c design `wf_f3eae69e` + Slice 1c.5 (the cap-with-fill resolution) + the 1c.5 audit `wf_e3397eb2`.

### D-9 🟥 Snapshot emit is full-world + double-encoded (no AoI / delta)
- **Missing:** `emit_frames` sends the full world every tick with no interest filter, and
  `partition_entities` re-encodes every entity twice per tick — the dominant per-tick CPU/bandwidth cost.
- **Where:** `crates/sim/src/stub.rs` (`emit_frames`); `crates/wire/src/channels.rs` (`partition_entities`).
- **When / proper:** the **P2 interest-management slice** — per-sub relevance sets + delta/baseline encoding;
  the throwaway sizing allocation dies in the same refactor. Negligible at current volumes (a few dots).
- **Source:** whole-codebase audit `wf_43fea0dd` (SCALE-B).

### D-10 🟥 Per-node honesty counters exist on every node but are unobservable at runtime (only the orchestrator publishes)
- **Missing:** the COUNTERS now exist on every node — gateway (`GatewayStats.undecodable`/`inputs_*`),
  stub-shard (`StubStats.undecodable`/`attaches_deferred`), orchestrator (`OrchestratorStats.undecodable`),
  follower (`FollowerState.undecodable`/`rejected_backward`/`epoch_mismatches`), inbox-drop
  (`BoundedInbox.dropped_reliable`). What is missing is **runtime EXPOSURE**: only the orchestrator publishes an
  `admin_snapshot`, so the gateway/shard/follower counters are unreachable by a `curl` operationally.
- **Where:** `crates/node/src/orchestrator.rs` (`admin_snapshot` is orchestrator-only); the gateway/shard bins.
- **When / proper:** **Slice 1c/P3** — extend the orchestrator's `ArcSwap` admin-cell pattern to the gateway
  and shard bins (a `NodeStatsView`); needed before the P3 crash matrix with real nodes.
- **Correction (`wwg7ydm9y`, ROB-E2E-1):** the prior "Counters already exist" line was FALSE — the shard,
  orchestrator, and follower DECODE-failure counters did not exist (the path was `tracing::error!`-only). They
  were ADDED (stub/orchestrator/follower `undecodable`, each test-asserted), so the gap is now purely exposure.
- **Source:** whole-codebase audits `wf_43fea0dd` (ROB-3) + `wwg7ydm9y` (ROB-E2E-1).

### D-11 🟥 Reconnect / session-lifecycle (resume without replay)
- **Missing:** the client never sends `Resume`; no node consumes it; the directory `resume_nonce` is unused;
  closing a client leaves a stale session until a cluster restart.
- **Where:** `crates/client/src/net.rs` (no `Resume` send); the `ResumeTicket` types exist but are undriven.
- **When / proper:** **P2/P3/P7** — P2 must ensure the saga does not assume a stable connection;
  gateway-adoption fence-CAS at P3; full reconnect-without-replay at P7.
- **Source:** the roadmap + P1.5 foundation audit.

### D-20 🟩 Gateway `TransferControl` consumer (replaced the `transfer_control_unhandled` counter)
- **Landed (Slice 1c.2):** `on_transfer_control` consumes `InterShardFlow::Saga(TransferControl)` with per-
  session `TransferProgress` cold state + a `(transfer, step_id)` applied-steps idempotency journal (re-sends
  the recorded `SagaAck` verbatim on at-least-once redelivery). The no-authority-move phases ack now:
  PrepareSubscribe→Prepared{Ready stub}, RequestCut→client `RequestCut` (its `CutConfirmed` deferred to the
  cut-marker observer on the INPUT flow), ThawSource→SourceThawed, AbortTransfer→Aborted. The route-touching
  phases (FreezeSource/CommitAuthority/ReleaseSubscribe) PARK loudly on `transfer_control_parked` (no ack ⇒ the
  saga pins) until 1c.3/1c.4. `transfer_control_unhandled` REMOVED; new counters: `transfer_unroutable`,
  `transfer_control_parked`.
- **Where:** `crates/connection-plane/src/gateway.rs` (`on_transfer_control`, `apply_prepare/_request_cut/
  _thaw/_abort`, `on_cut_marker`, `reply_ack`); the ONE dedup accessor is `TransferProgress::recorded`/`journal`
  (both the redelivery gate and the cut-marker observer go through it), keyed by `IdempotencyKey::TransferStep`.
- **Still owed (carried by other entries):** the route swap + cut partition (1c.3/1c.4); durable journal (D-22);
  the client marker emit (D-5).
- **Source:** Slice 1c.1 (the dispatch-collision fix) + whole-codebase audit `wwg7ydm9y` (registered the
  unpinned interim counter).

### D-22 🟥 Gateway applied-steps journal is RAM-only by design (the durable table is the dest shard's at 1d)
- **Missing:** the gateway's `TransferProgress.applied` (`BTreeMap<(TransferId, u32), TransferControlAck>`) is
  IN-MEMORY; a gateway crash mid-transfer loses the recorded acks. This is BY DESIGN — the gateway is soft-state
  (`connection_plane.md` resume model): on resume it re-registers with the saga, it does NOT replay from RAM.
  The DURABLE `(TransferId, step_id)` `applied_steps` redb table is the **dest shard's** (Slice 1d, D-21) /
  the orchestrator saga WAL's (P3, D-6). This entry exists so no future reader mistakes the per-session RAM
  field for the durable journal.
- **Where:** `crates/connection-plane/src/gateway.rs` — `TransferProgress.applied`.
- **When / proper:** durability is NOT a gateway concern; the durable journal lands at **1d** (shard
  `TransferAck` receiver) and **P3** (saga WAL) — keyed by the SAME `IdempotencyKey::TransferStep` (the ONE
  shared anchor; HR3 one machinery, many STORES). No rewrite: durability is a different backing for the same
  key + the consult-before-effect / record-after-effect discipline.
- **Source:** Slice 1c.2 design `wf_726a51bc`. The shared element is the KEY (`IdempotencyKey::TransferStep`),
  not a forwarding fn: the ceremonial `recorded_step` shim was REMOVED and the gateway's two journal sites
  unified behind `TransferProgress::recorded`/`journal` — corrected by audits `wwk1uh5k9` (DRY-1/F1, the dead
  shim) + `wo2gkj7t7` (DRY-1-RESIDUAL, the second inline path).

### D-23 🟥 Gateway single-slot `transfer: Option<TransferProgress>` assumes one-transfer-per-session (masked by D-1)
- **Missing:** the gateway holds at most ONE in-flight transfer per session (`Session.transfer:
  Option<TransferProgress>`). Two latent bugs are masked TODAY only because the orchestrator serializes one saga
  per session-subject (`DirectoryCore::lock_transfer`) AND the abort-lock leak (D-1) never frees that lock:
  - **RACE-1:** `apply_prepare` defensively REPLACES any existing progress; once D-1 is fixed and a second saga
    can start, a `PrepareSubscribe` for transfer B would discard transfer A's in-flight journal.
  - **WEDGE-1:** a client `Bye` mid-transfer drops the `Session` + its journal; subsequent saga commands for it
    then count `transfer_unroutable` with NO producer to unstick the pinned saga. The real backstop is the
    Slice-2 saga timeout/abort producer.
- **Where:** `crates/connection-plane/src/gateway.rs` — `apply_prepare` (RACE-1, loud `tracing::warn` on a
  replaced different transfer) + the `Bye` handler (WEDGE-1, loud `tracing::warn` when a transfer was in flight).
  Both pinned in-code; the in-flight-survives-foreign-abort property has a test (CP-1).
- **When / proper:** **Slice 2** — alongside the D-1 `abort_cas` lock-clear + the saga timeout producer. Revisit
  whether `Option<TransferProgress>` must become a keyed map (it likely must once SIGNALS multiplex multiple
  correlation streams over one session — the signal-readiness prerequisite the audit flagged).
- **Source:** whole-codebase audit `wwk1uh5k9` (RACE-1 / WEDGE-1, pinned; masked by D-1).

### D-21 🟧 `TransferAck` / `TransferStepRejectReason` are RESERVED wire vocabulary (no consumer yet)
- **Missing:** the shard-bound `Transfer(TransferEnvelope)`-arm RECEIVER. The dest shard's
  ack of a transferred entity-state step (`TransferAck{Accepted|Rejected{reason}}`) is frozen
  in the wire contract but has no producer/consumer — the 1b/1c saga uses the gateway↔saga
  `TransferControlAck` vocabulary, not this shard→orchestrator envelope ack.
- **Where:** `crates/wire/src/intershard.rs` — `TransferAck` + `TransferStepRejectReason`
  (RESERVED doc on each; roundtrip-tested but unconsumed). Renamed from `TransferRejectReason`
  to end the name collision with the live CLIENT-facing `channels::TransferRejectReason` (DRY-1).
- **When / proper:** **Slice 1d** — the shard-bound `StubCrossing` transfer receiver journals
  each step idempotently by `(transfer_id, step_id)` and replies `TransferAck`; that is its
  first consumer (the "freeze arms WITH their first consumer" rule). Reconcile then whether the
  shard step-reject reasons should fold into / map cleanly onto the client-facing reasons.
- **Source:** whole-codebase convergence audit `wf_8d81c753` (DRY-1).

---

## CLOUD / DEPLOY (gated on a deploy-readiness signal from the user — no CI yet)

### D-12 🟥 Client reachability: static address book / gateway-dials-client (CA-1)
- **Missing:** the cloud-correct **reply-on-connection** — the gateway must learn a client's return path from
  its inbound accepted QUIC connection, keyed by the mTLS-asserted NodeId, so a client that dials in is
  replyable without being pre-booked.
- **Where:** `crates/io-prod/src/mesh.rs` + the dev-cluster client seeding (the `CA-1 CRUTCH` comment in
  `crates/bins/src/bin/vd-devcluster.rs`). Pinned by a red-guard `#[ignore]` acceptance test in `mesh.rs` that
  fails by construction today.
- **When / proper:** **M3 — before ANY non-loopback deployment.** BINDING: the deployment story is
  loopback-only until CA-1 lands.
- **Source:** whole-codebase audit `wf_43fea0dd` (CAF-1).

### D-13 🟥 Admin endpoint authentication
- **Missing:** the read-only admin endpoint serves internal topology with no auth — safe only on loopback.
- **Where:** `crates/wire/src/admin.rs` (security-contract doc); the dev cluster binds loopback.
- **When / proper:** **deploy-readiness** — bearer/mTLS on the route before any routable bind. The greenfield
  cloud/deploy manifests are authored fresh at that time (the pre-rebuild `Dockerfile`/root `dev-cluster.sh`/
  `k8s/*.yaml` stack — flat-layout, per-shard-kind images violating HR3, unauthenticated `0.0.0.0` admin — was
  DELETED in CAF-NEW-1; `scripts/dev-cluster.sh` + `vd-devcluster` are the local-process launcher).
- **Source:** whole-codebase audits `wf_43fea0dd` (CAF-2) + `wwg7ydm9y` (CAF-NEW-1, stale stack removed).

---

## PERF / SCALE (negligible now; land with the slice that makes them matter)

### D-24 🟥 Per-tick inbound/session rescans (gateway/orchestrator) — (SCALE-CUTDECODE-1 🟩 resolved 1c.3)
- **SCALE-CUTDECODE-1 🟩 RESOLVED in Slice 1c.3:** `on_cut_marker` no longer full-decodes every input — it calls
  `vd_wire::session_flow::peek_is_cut_marker`, reading only the `(seq, is_cut_marker)` postcard prefix
  (`[varint seq][1 canonical bool byte]`) off the head, never the body.
- **Missing (remaining):** three correct-but-O(n)/per-frame-alloc hot spots, all negligible at current volumes:
  (a) **SCALE-INBOX** — `BoundedInbox::push` linear-scans for the oldest unreliable on every full push;
  (b) **SCALE-5** — per-tick systems re-scan the whole inbound `Vec`, and `drive_pending_sessions` scans ALL
  sessions including `Active` ones; (c) **SCALE-INPUT-COPY** — every forwarded client input is heap-copied
  (`input_bytes: bytes.to_vec()`) into a fresh `GatewayToShard::SessionInput` on the 20 Hz path. The fix mirrors
  the snapshot fan-out: carry the body as the already-available Arc-backed `vd_sim::io::Bytes` (`Arc<[u8]>`)
  end-to-end instead of re-allocating per input/client/tick.
- **Where:** `crates/connection-plane/src/gateway.rs` (`drive_pending_sessions`; `on_client_input`'s
  `bytes.to_vec()`); `crates/sim/src/io/mod.rs` (`BoundedInbox::push`); `crates/node/src/orchestrator.rs` +
  `saga_runtime.rs` (`serve_directory` / `drive_sagas` inbound loops).
- **When / proper:** SCALE-INBOX / SCALE-5 / SCALE-INPUT-COPY
  / SCALE-CLOUD-1 → the **P1/P3 load-test + observability slice** (the load-tests-when-applicable standard): an
  index for active vs pending sessions + a ring/heap for the inbox + an Arc-carried input body + publish the
  admin snapshot on-change or at a sub-tick rate (SCALE-CLOUD-1: the orchestrator bin currently rebuilds+clones
  the FULL directory + saga views every tick at `tick_hz` — `crates/bins/src/bin/orchestrator.rs`), each sized
  by a real bench, not before.
- **Source:** whole-codebase audits `wwk1uh5k9` / `wo2gkj7t7` / `wg9gc765s` / `wdznr0x6i` / `wxwinv5no`
  (SCALE-1C2-1 / SCALE-3-inbox / SCALE-5 / SCALE-CLOUD-1 / SCALE-3-input-copy).

### D-25 🟥 Route swap CARRIES the realm fence; the per-Entity CAS `new_fence` is NOT installed (single-realm 1c)
- **Missing:** `apply_commit` (the 1c.4 `CommitAuthority` route swap) carries `route.fence` UNCHANGED;
  `new_fence` (the `DirectoryKey::Entity` CAS fence — `crates/node/src/saga_runtime.rs`) is consumed (`let _`),
  NOT installed as `route.fence`. **Fence rule 5** (the gateway fences out a DEMOTED REMOTE owner's frames by
  dropping a stale frame fence vs `route.fence` — `frame_passes_fence`) therefore cannot fire intra-shard.
- **Why carry-not-install (NOT a bug):** frames stamp the **REALM** fence (`route` attaches `realm_fence`;
  checked at `frame_passes_fence`); `new_fence` is a **per-ENTITY** CAS fence — a DIFFERENT fence domain.
  Installing the Entity fence as `route.fence` would make the dest's OWN realm-stamped frames stale
  (`realm_fence.is_stale_against(new_fence) == true`, `core/src/fence.rs`) → a black screen. Guarded red/green
  by the 1c.4 test `commit_does_not_drop_the_dest_own_realm_frames`.
- **Where:** `crates/connection-plane/src/gateway.rs` — `apply_commit` (`let _ = new_fence`).
- **When / proper:** **1d / mesh** — once source and dest are DISTINCT realm leases, the dest re-stamps at its
  own realm fence and the swap installs THAT realm fence, so the demoting source's lower-realm-fence frames go
  stale at `frame_passes_fence` (rule 5 fires, with the SAME `is_stale_against` already in use). Inert + correct
  in 1c (one realm lease). `new_fence` stays threaded on the wire + saga (the CAS linearization point) so the
  upgrade is install-on-dest-re-stamp, no reshape.
- **Source:** P2 Slice 1c.4 design `wf_2e0f6c1d` (the R-FENCE / fence-domain resolution).

### D-27 🟧 Dest `OpenInputSlot` is a STATE-FREE provisional input slot; 1d promotes it to a real owned entity
- **Landed (1c.5):** to let the transfer-DESTINATION shard ACCEPT (not drop as `UnknownSession`) the post-
  marker input the gateway drains at commit, `GatewayToShard::OpenInputSlot{session, fence, account,
  resume_from_seq}` mints a provisional `Dot` on the dest: `input_active: true` (applies input via the existing
  `apply_input` + `last_applied_seq` dedup, seeded to `marker_seq`) but `granted: false` — so it is **NOT
  rendered** (`emit_frames` filters `granted`), holds **NO directory record**, and emits **NO `SessionAttached`**
  (the source still owns the client connection — R2). Sent only at `CommitAuthority` (after the directory
  committed authority to the dest), guarded `!granted && dot.gateway == from` (a shard never applies input for a
  session not legitimately routed by its owning gateway — HR1/security; a stale/replayed slot is dropped +
  counted `StubStats.input_slots_stale`). This is the INPUT half of the dest receiver — it LANDS the dest's
  ability to apply the drained batch; it does NOT by itself CLOSE cross-cut input-conservation (the two crate
  halves are never joined in one scenario, and a dropped/lost buffer has no recovery — D-28, D-29).
- **Still owed (1d):** the `granted` promotion — the real per-entity `Authority` attach so the dest OWNS +
  EMITS + RENDERS the entity, the entity-STATE crossing (`StubCrossing`/`Transfer` envelope + durable
  `applied_steps`, D-21/D-22), the real ghost, the dest spatial `PrepareResult::Rejected`, the
  `AbortTransfer`-tears-down-the-dest-slot teardown, and the provisional minted entity replaced by the
  transferred one. 1d is a PROMOTION of this slot (a grant flip + state attach), never a rewrite.
- **Also owed:** the EARLY (prepare-time) `OpenInputSlot` for the gateway-adoption-mid-cut race is a **P3**
  resilience item (the gateway re-drives the slot before the buffer drain); inert in 1c (single process, commit
  emits the slot in the same handler before the drain).
- **Where:** `crates/wire/src/session_flow.rs` (`OpenInputSlot`); `crates/connection-plane/src/gateway.rs`
  (`apply_commit` emit); `crates/sim/src/stub.rs` (`on_gateway_msg` arm, `Dot.input_active`, `apply_input`).
- **Source:** Slice 1c.5 design `wf_9679f7c7` (the dest-side reconciliation of integration.json #1).

### D-28 🟥 Cross-cut INPUT-CONSERVATION is proven only in unjoined per-crate halves (the 1c.7 end-to-end gate)
- **Missing:** the gateway's buffer/drain (connection-plane) and the dest's `OpenInputSlot`/`apply_input`
  (sim) are each unit-tested, but NOTHING wires the gateway's drained output into a real stub `apply_input` in
  ONE scenario. `marker_seq` consistency is asserted on each side against HARDCODED constants (the gateway sends
  `resume_from_seq=marker_seq`; the stub test seeds `last_applied_seq` with a literal), never threaded through.
  `verify_input_conservation` (`harness/src/oracle.rs` — the exactly-once + per-session strict-monotonic oracle
  that would catch a post-marker input lost in the buffer, double-applied across the seam, or applied out of
  order) is NEVER invoked on a transfer scenario; `process_parity.rs` hardcodes `is_cut_marker: false`.
- **Where:** `crates/connection-plane/src/gateway.rs` (1c.5 tests), `crates/sim/src/stub.rs` (1c.5 tests),
  `crates/harness/src/oracle.rs` (`verify_input_conservation`).
- **When / proper:** **Slice 1c.7** — one scenario (harness topology or a connection-plane+sim integration
  test) that drives `Prepare → RequestCut → marker → Freeze → client inputs straddling the marker → Commit`,
  feeds the gateway's drained `OpenInputSlot` + `SessionInput` into a real stub, quiesces, and asserts
  `verify_input_conservation(reports) == Ok` AND that the `marker_seq` the gateway EMITTED equals the one the
  stub SEEDED (threaded, not hardcoded on both sides). The central transfer-first invariant is not "closed"
  until this is green.
- **Source:** the 1c.5 audit `wf_e3397eb2` (cut-conservation-untested-crossseam-and-untracked).

### D-29 🟥 No recovery producer for a lost cut buffer (deferred slot / over-cap / gateway crash)
- **Missing:** when the dest drops `OpenInputSlot` (realm lease late at commit → `input_slots_deferred`), or the
  soft cap drops the oldest, or the gateway crashes mid-cut, the take-drained `dest_buffer` is gone and there is
  NO mechanism to re-issue the slot or re-supply the buffer. `apply_commit` emits `OpenInputSlot` once and the
  saga is forward-only past `Committed` (`saga_runtime.rs`). In single-process 1c.5 this is inert (the directory
  CAS-to-dest implies the dest already holds its realm lease; the buffer is never congestion-shed in the in-mem
  transport), but the moment the dest is a genuinely distinct shard whose lease lands a tick late (k8s pod
  cold-start) the loss fires with zero recovery. Distinct from D-27's prepare-time gateway-ADOPTION-mid-cut P3
  note (a different race).
- **Where:** `crates/connection-plane/src/gateway.rs` (`apply_commit` one-shot drain); `crates/sim/src/stub.rs`
  (`input_slots_deferred` drop); `crates/node/src/saga_runtime.rs` (forward-only past `Committed`).
- **When / proper:** **1d/P3** — either a CONFIRMED `OpenInputSlot` step (dest acks slot-open; the gateway holds
  `dest_buffer` and re-emits on the per-tick retry like `AttachSession`, gating the drain on the confirm), OR the
  durable seq-range interval map (D-8) as the authoritative conservation mechanism (reload-from-directory on
  reconnect; integration.json #1's rare-failure-only `InputGap` becomes the honest signal). Closes WITH D-28's
  end-to-end gate.
- **Source:** the 1c.5 audit `wf_e3397eb2` (openinputslot-deferred-loses-buffer-no-redrive).

### D-14 🟥 `rendered()` recomputed ~3×/display-frame on the client
- **Missing:** the windowed/headless render path recomputes the composited view 2–3× per frame.
- **Where:** `crates/client-render/src/lib.rs` (sync_world + draw_hud + location); `crates/client/src/view.rs`.
- **When / proper:** the **P2 client interest-cap work** — sample `rendered()` once/frame in `Update`, publish
  as a Bevy resource the egui pass reads. Pure perf; negligible at a handful of entities.
- **Source:** P1.5 foundation audit (`scalability-1`).

### D-15 🟥 `walk-to`/`look-at` not wired through `vdctl`
- **Missing:** the closed-loop `walk-to`/`look-at` dev-control commands (the `DevRequest::WalkTo`/`LookAt`
  variants exist but return `Unsupported`).
- **Where:** `crates/bins/src/bin/vdctl.rs` + the client `dev_control` handler.
- **When / proper:** **P2** — needed for the paired 2-client visual scenario (client2 screenshots client1
  crossing). `NavController` math is pure-ready in `crates/client-harness/src/nav.rs`.
- **Source:** Slice-3 T6 scope decision.

### D-16 🟥 G-RENDER-SMOKE is GPU-required / local-only
- **Missing:** a GPU-less path for the visual gate (no software-adapter policy).
- **Where:** `crates/client-render/src/lib.rs` `run_capture` (a loud GPU-precondition log exists).
- **When / proper:** **deploy/CI-readiness** — decided GPU-required-local for now (a software fallback would
  poison the byte-baseline and needs a new dep + joint investigation). Re-evaluate when headless-GPU CI exists.
- **Source:** Slice-3 T6 (cloud-1) + sign-off.

### D-17 🟥 G-RENDER-SMOKE scene contract re-baseline at terrain
- **Missing:** the gate's self-calibrated-corner + content-floor assumptions hold for the reference scene
  (ground plate + pillars); they need re-baselining when P4 terrain fills the frame.
- **Where:** `crates/bins/tests/render_smoke.rs`.
- **When / proper:** **P4** (terrain).
- **Source:** Slice-3 T7 sign-off.

---

## BLOCKING SPIKES (must run before the phase they gate)

### D-18 🟥 SPIKE-3a — datagram delivery under a 4 MB BULK burst on the k3d overlay
- **Blocks:** **P3.** The hand-rolled latency-gate harness (`percentile_unstable`, established by SPIKE-2a)
  should be extracted to a shared `vd-harness` helper when SPIKE-3a adds the second hard latency gate.
- **Source:** `PLAN.md` SPIKE list.

### D-19 🟥 SPIKE-6a (rapier snapshot/restore + cross-binary determinism) blocks P5; SPIKE-10a (dual-frame ship-interior physics) blocks P8.
- **Source:** `PLAN.md` SPIKE list.
