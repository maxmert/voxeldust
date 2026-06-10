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

### D-2 🟧 Demote→Release has no real producer (the `interim_demote_complete` seam)
- **Missing:** the proper event-driven `DemoteComplete` — the source ghost (a kinematic collider, so players
  collide across the boundary) is torn down ONLY when the dest has **delivered the entity to all observers**
  (a server-side watermark, never a client ack — HR1) **AND** the entity has **left the source overlap band**
  (`OverlapBand::update_membership` false on the swept segment `segment_shell_crossing`, gated by
  `width_safe_for`/`K_SAFETY` — velocity-aware hysteresis, NO magic tick count).
- **Where:** `crates/node/src/saga_runtime.rs` — a single loudly-named `interim_demote_complete` seam: an
  unconditional bandless stand-in that calls `deliver(.., SagaEvent::DemoteComplete)` (a new CALLER of the
  existing sink — NOT a magic tick, NOT an in-flight-queue injection). The FSM tail
  (`Swapping→Demoting→Releasing→Done`) is COMPLETE + proptested in `crates/sim/src/saga.rs` and is UNTOUCHED.
  Pinned by a `KNOWN LIMIT` doc + an always-on `tracing::warn` + an "exists-to-be-flipped" test.
- **When / proper:** the **post-1d band/ghost P2 slice** writes the real per-tick dest-shard predicate.
- **Single-point upgrade (verified reshape-free, `wf_0ed2dc0c`):** swap that ONE function's body
  (unconditional → the conjoined predicate); ZERO change to the FSM / `deliver` / `run_to_quiescence` / the
  gateway `ReleaseSubscribe` handler / the frozen `TransferControl` vocabulary.
- **Dependency:** Slice 1d (per-entity `Authority` attach so the dest owns + emits the post-commit entity) +
  the ghost-as-collider + band-instance + observer-delivery-watermark machinery (the band/ghost slice).
- **Source:** 1c design `wf_f3eae69e` + the demote refinement `wf_0ed2dc0c`.

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

### D-4 🟥 Per-entity AoI eviction needs the `EventMsg` client `MsgClass` arm
- **Missing:** the client `DeliveredView` is bounded per-SUB (`drop_sub` on `SubscriptionClosing`), but a
  still-open sub's departed entities are not evicted — that needs `EventMsg::EntityRemoved`, which has no
  transport class to arrive on (`MsgClass` has Control/Saga/Snapshot/Input/Membership — no Event/Bulk arm).
- **Where:** `crates/client/src/view.rs` doc (names this "the remaining P2 piece"); `crates/sim/src/io/mod.rs`
  (`MsgClass`, no Event arm); `crates/client/src/net.rs` (routes only Control/Snapshot; others → `ignored`).
- **When / proper:** **P2** — add the `EventMsg` client `MsgClass` arm + route `EntityRemoved` → a per-entity
  `drop`. Retires the foundation-audit `scalability-1` root (bounds the view at entity granularity, not just
  per-sub).
- **Source:** P1.5 foundation audit (deferral) + whole-codebase audit.

### D-5 🟥 Client cut cycle (the `net.rs` marker emit) — Slice 1e
- **Missing:** on `ServerControlMsg::RequestCut` the client must record it and emit the triple-sent in-band
  `CUT_MARKER` (`is_cut_marker=true`) on its INPUT flow + a reliable `ClientControlMsg::CutEmitted` on CONTROL.
- **Where:** `crates/client/src/net.rs` — `on_control` currently IGNORES `RequestCut` at the catch-all;
  `assemble_input`/`send_outbound` build `InputDatagram` with `is_cut_marker` hardcoded `false`.
- **When / proper:** **Slice 1e.** (HR1: the client only emits a marker — it never participates in the
  transfer. Slice 1c builds + tests the gateway cut partition with an injected/scripted marker.)
- **Source:** the P2 vertical-slice plan.

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

### D-8 🟥 Transfer dest-input buffer has no hard cap / counted drop
- **Missing:** the gateway's `seq > marker` dest buffer (filled while the cut is open) is unbounded.
- **Where:** `crates/connection-plane/src/gateway.rs` — the `TransferProgress.dest_buffer` (lands in Slice 1c).
- **When / proper:** **Slice 2.** Inert in 1c (the in-process saga commits `SourceFrozen→CommitAuthority`
  within one tick — a ~1-datagram window), but a HARD prerequisite once a stalled saga can hold the cut open
  across ticks: a `BoundedInbox`-style cap + counted drop (`crates/sim/src/io/mod.rs` discipline).
- **Source:** 1c design `wf_f3eae69e`.

### D-9 🟥 Snapshot emit is full-world + double-encoded (no AoI / delta)
- **Missing:** `emit_frames` sends the full world every tick with no interest filter, and
  `partition_entities` re-encodes every entity twice per tick — the dominant per-tick CPU/bandwidth cost.
- **Where:** `crates/sim/src/stub.rs` (`emit_frames`); `crates/wire/src/channels.rs` (`partition_entities`).
- **When / proper:** the **P2 interest-management slice** — per-sub relevance sets + delta/baseline encoding;
  the throwaway sizing allocation dies in the same refactor. Negligible at current volumes (a few dots).
- **Source:** whole-codebase audit `wf_43fea0dd` (SCALE-B).

### D-10 🟥 Per-node honesty counters (gateway/shard/follower) unobservable at runtime
- **Missing:** gateway-undecodable / shard-malformed / follower-clock-anomaly / inbox-drop counters are
  maintained but only the orchestrator publishes an admin snapshot — the others are unreachable operationally.
- **Where:** `crates/node/src/orchestrator.rs` (`admin_snapshot` is orchestrator-only); the gateway/shard bins.
- **When / proper:** **Slice 1c/P3** — extend the orchestrator's `ArcSwap` admin-cell pattern to the gateway
  and shard bins (a `NodeStatsView`); needed before the P3 crash matrix with real nodes. Counters already exist.
- **Source:** whole-codebase audit `wf_43fea0dd` (ROB-3).

### D-11 🟥 Reconnect / session-lifecycle (resume without replay)
- **Missing:** the client never sends `Resume`; no node consumes it; the directory `resume_nonce` is unused;
  closing a client leaves a stale session until a cluster restart.
- **Where:** `crates/client/src/net.rs` (no `Resume` send); the `ResumeTicket` types exist but are undriven.
- **When / proper:** **P2/P3/P7** — P2 must ensure the saga does not assume a stable connection;
  gateway-adoption fence-CAS at P3; full reconnect-without-replay at P7.
- **Source:** the roadmap + P1.5 foundation audit.

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
- **When / proper:** **deploy-readiness** — bearer/mTLS on the route before any routable bind. Also reconcile
  `k8s/orchestrator.yaml`'s `0.0.0.0:8080` listener (+ its CLI flags the real bin doesn't parse) at that time.
- **Source:** whole-codebase audit `wf_43fea0dd` (CAF-2).

---

## PERF / SCALE (negligible now; land with the slice that makes them matter)

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
