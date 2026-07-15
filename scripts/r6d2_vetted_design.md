# R-6d2 Vetted Design of Record — additive Durability seam + FSM write-through

Design-refinement workflow wf_ad6e7936 (design + 3 adversarial opus reviews + synthesis). Verdict REVISE (architecture SOUND; 1 CRITICAL [conformance-test vacuity] + 2 HIGH folded into the corrected plan below).
(A) ruling: DEFER the real shared sink to R-6d3 (thread the sink param through the SYNC FSM only, None in prod + MockOutboxSink tests).
(B) ruling: a wildcard-free durability_class() classifier in vd-wire (sibling of effect_class) + rig-driven marker tests.

=====================================================================
## PART 1 — ADJUDICATED SYNTHESIS (design of record)
=====================================================================

The producer-path test rigs exist: `emit_transient_batch_ships_one_envelope_and_marks_outbound` (3173) drives the TransientBatch push, and `the_dest_feed_despawns_on_band_exit_and_deregisters` (5744) drives the Ghost::Despawn push. Both capture the `OutboundBox`. So Review 3's fix (drive-the-path + assert the 4th tuple element == Retained) is buildable against existing rigs. 

I now have complete grounding on all findings. Let me compose the implement-ready design of record.

---

# R-6d2 — IMPLEMENT-READY DESIGN OF RECORD (adjudicated)

## Adjudication summary

I verified every finding against the code. **The plan is fundamentally sound and correctly bounded, but ships an incomplete lockstep enumeration and one VACUOUS conformance test.** Verdict: **REVISE** — three CONFIRMED must-fix findings (one CRITICAL) before code. All are mechanical-or-local; the architecture, the two rulings' direction, and the frozen-seam decision stand.

Adjudication of each reviewer finding, cited:

| Finding | Reviewer | Verdict | Evidence |
|---|---|---|---|
| Conformance-test inner wildcard is vacuous on the generalization axis | R3 CRITICAL | **CONFIRMED** | `TransitionPayload` (intershard.rs:455) has 3 variants; `effect_class` (intershard.rs:203) classifies `Transfer(env)` WITHOUT matching payload (line 208); a `_ => ReDriven` inner arm swallows a future producer-less payload silently. |
| Direct `.0.push((..))` sites break the 4-tuple + need a durability decision | R1 HIGH | **CONFIRMED** | gateway.rs:894 (push_control), :899 (push_to_shard), :907 (push_directory), :1849 (Snapshot); stub.rs:2811 (Snapshot) — all direct pushes bypassing `push_flow`. Plan's §2d/§2f enumerate only `push_flow`. |
| `transport.send` CALLER enumeration incomplete; `send_bytes` is production | R1 HIGH / R3 MEDIUM | **CONFIRMED** | net.rs:285 `send_bytes` is a non-test method; harness client.rs:180/281/344 are in `ScriptedClient` (production; test mod starts :389); orchestrator.rs:417/550, tracer.rs, chaos.rs, topology.rs, +~60 test callers. |
| `on_ack` unconditional release bloats the Ephemeral-heavy Saga lane | R2 MEDIUM / R1 LOW | **CONFIRMED (as R-6d3 concern)** | Saga lane carries TransientBatch (Retained) AND ~9 saga_runtime flows (Ephemeral); `on_ack` (mesh.rs:540) has no per-frame durable bit. `RetainedFrame{frame,framed_len}` (mesh.rs:434) can carry a `retained:bool`. |
| write_frame/peer_writer sink-threading is None-only inert plumbing in R-6d2 | R3 MEDIUM | **CONFIRMED** | `write_frame` (mesh.rs:1513) is async, opens real QUIC streams (1610/1614) — untestable without an endpoint; the `Some` path is reachable in R-6d2 only via direct `assign_and_retain`/`on_ack` mock tests. |
| Stored `encoded` (epoch u32::MAX) reverses vetted L1/A5 | R2 LOW / R3 LOW | **CONFIRMED SOUND** | `encoded` (mesh.rs:588) is `encode_frame` output at `epoch:u32::MAX` (line 583), computed before the real epoch is set (line 596); still owned after the `frame` move into `retry` (597). Storing it is HR3-clean (one framing home) but forces R-6d3 replay to decode-and-re-stamp. |
| R-6d1 did NOT plumb a PeerWriter/spawn_mesh sink handle | R2 LOW | **CONFIRMED** | `rg outbox\|sink\|NodeOutbox` over mesh.rs = zero hits. The plan's "confirm it landed / 1-line add" hedge must be removed. |
| A second exhaustive classifier unreconciled with `effect_class` | R3 MEDIUM | **CONFIRMED** | `effect_class` (intershard.rs:199-289) is the existing exhaustive sibling; orthogonal axis, cannot be reused (coarse `Ghost(_)`/payload-blind `Transfer`), must be co-located + acknowledged. |
| `ReliableLaneSender::new` peer param + ~40 test call-site sweep | plan §3a | **CONFIRMED broader** | `new(incarnation, retry_cap)` (mesh.rs:496); ~40 `assign_and_retain` + ~15 `on_ack` test sites (3-arg) also break. |

Refuted-as-non-issues (the plan is right): the 6-impl list is **complete and exact** (mem/io/mem.rs:342, mesh.rs:1711, lib.rs:162, fabric.rs:515, app.rs:474 [test], net.rs:536 [test]); `Durability` in `sim::io` forms **no cycle**; the datagram hot path is **provably byte-identical** (Durability never reaches the `Reliability::Unreliable` arm, mesh.rs:1657); the **frozen-seam change is genuinely required** (the only channel from the `push_flow`-minted bit to the writer-task-minted `assign_and_retain` is `send()→OutFrame→mpsc`; a marker on the `OutboundBox` tuple alone is dropped at `flush_phase→send()`); the **saga split / boot replay / durable-before-send gate are correctly held to R-6d3**.

---

## (a) The final `Durability` type + the COMPLETE lockstep impl list

Type is **unchanged from the plan** — the enum-over-bool reasoning holds:
```rust
// crates/sim/src/io/mod.rs — beside MsgClass / Reliability / SendError
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Durability {
    #[default] Ephemeral,   // RAM-only; the default for all ~existing sends
    Retained,               // write-through to the durable outbox; producer-less one-shots only
}
```
New frozen sig (sim/io/mod.rs:338):
```rust
fn send(&mut self, to: NodeId, class: MsgClass, bytes: Bytes, durability: Durability) -> Result<MsgId, SendError>;
```

**The COMPLETE lockstep set — impls AND callers (rg-confirmed; the plan's §2b covered only impls):**

*Trait impls (6, all confirmed):*
| Impl | file:line | Change |
|---|---|---|
| MemTransport | sim/io/mem.rs:342 | `_durability` param; ignore |
| FabricTransport | harness/fabric.rs:515 | `_durability`; ignore |
| PerPeerLanes `#[cfg(test)]` | node/app.rs:474 | `_durability`; ignore |
| MockTransport `#[cfg(test)]` | client/net.rs:536 | `_durability`; ignore |
| ProdTransport | io-prod/lib.rs:162 | `durability`; set `OutFrame.durability` (lib.rs:165) — struct-completeness only (see §f) |
| MeshTransport | io-prod/mesh.rs:1711 | `durability`; set `OutFrame.durability` |

*Production `.send()` callers (MUST pass a marker — the plan omitted these):*
- **net.rs:285 `send_bytes`** (production client) → `Durability::Ephemeral` (client emits only Input/Control/Bye — never a producer-less reliable one-shot).
- **harness/client.rs:180, 281, 344** (`ScriptedClient`, production harness) → `Ephemeral`.
- **node/orchestrator.rs:417, 550** → `Ephemeral`.
- **node/tracer.rs:85, 152, 173** → `Ephemeral`.
- **harness/chaos.rs:162, topology.rs:642, 938** → `Ephemeral`.

*Test `.send()` callers (mechanical sweep, `Ephemeral`):* ~60 sites across mem.rs, fabric.rs, mesh.rs (test mod), client.rs (test), the io-prod integration tests (mesh_load.rs, mesh_redelivery.rs, mesh_under_loss.rs:59, spike0a_quinn.rs), bins/tests/process_parity.rs:74/111, tests/p0_gates.rs:51. Grep pin: `rg '\.send\(.*MsgClass'`.

**Invariant to state in the plan:** *no `transport.send` caller anywhere passes `Retained` — the ONLY two `Retained` producers are the `push_flow` sites at stub.rs:2134 (TransientBatch) and stub.rs:2667 (Ghost::Despawn), which reach `send` via the `OutboundBox`→`flush_phase` path, not a direct `.send()`.*

---

## (b) RULING (A): DEFER the real shared sink to R-6d3 — but TIGHTEN the R-6d2 threading boundary

**Decision: option (ii) — thread `Option<&mut dyn OutboxSink>` through the FSM, pass `None` in production, exercise via MockOutboxSink; defer real-sink injection + concurrency + the fsync-gate + boot replay to R-6d3.** This direction is CORRECT and matches the vetted slicing (§6:470-483). Injecting now would either hold a Mutex across the QUIC send path (the exact HIGH-1 deadlock) or pull the whole off-tick-fsync gate forward, collapsing R-6d3.

**But the CONFIRMED R3-MEDIUM forces a boundary correction:** `write_frame` (mesh.rs:1513) is async and opens real QUIC streams — it has NO unit test and is unreachable with a non-`None` sink in R-6d2. Threading the sink THROUGH `write_frame` + `peer_writer` in R-6d2 is genuine **inert None-only plumbing**. So:

**Do NOT thread the sink through `write_frame`/`peer_writer` in R-6d2.** Stop the R-6d2 sink threading at the FSM method boundary:
- `assign_and_retain` and `on_ack` **grow their sink param** (`durable: bool` + `Option<&mut dyn OutboxSink>` on assign; `Option<&mut dyn OutboxSink>` on on_ack) — these are synchronous, pure, and directly MockOutboxSink-tested. The `Some` path is genuinely exercised. **NOT dead.**
- `write_frame`'s call to `assign_and_retain` (mesh.rs:1542) passes `(durable_lowered_from_frame.durability, None)` in R-6d2, and the `peer_writer` `on_ack` call (mesh.rs:1128) passes `None`. The `durable` lowering (`let durable = matches!(frame.durability, Durability::Retained);`) IS live in prod (it computes a real value every reliable send); only the terminal sink is `None`.
- **R-6d3 then** flips `write_frame`/`peer_writer` to thread the real `Arc<Mutex<NodeOutbox>>` handle (added to `PeerWriter`/`spawn_mesh` — CONFIRMED not yet present), adds the durable-before-send gate, boot replay, and the saga closure — and its threading is exercised end-to-end by the SIGKILL process proof.

This keeps EVERY R-6d2 line either live-in-prod (the `Durability` seam, `OutFrame.durability`, the `durable` lowering) or live-in-test (the FSM `Some`-path via mock) — **zero inert plumbing**, tighter than the plan's §3d. R-6d2's own gate does not touch the async writer's new params with a real sink, so nothing untestable is claimed-covered.

**⚠ R-6d3 binding constraint to record now (R2-LOW carry):** `NodeOutbox::commit()` (outbox.rs) blocks on `wait_durable_through`. R-6d3 must NOT call it inline on a tokio worker — stage retain/release under the Mutex (no await held), release the lock, then observe durability off-thread. Recording this now prevents R-6d2's commit-cadence placement from locking in an inline-blocking call.

---

## (c) RULING (B): the §7 conformance test — shape (a), FIXED to be exhaustive-by-construction (no wildcard)

**Decision: shape (a) lands with R-6d2 — BUT the plan's inner match is REJECTED as vacuous and replaced.** Review 3's CRITICAL is confirmed against intershard.rs:455 (three `TransitionPayload` variants) and intershard.rs:208 (`effect_class` classifies `Transfer` payload-blind). The plan's `Transfer(env) => match env.payload { TransientBatch => ProducerLess, _ => ReDriven }` wildcard swallows a future producer-less payload (durable Signal, block-edit forwarding — the stated end-goal) with no compile error → the test passes vacuously → the one-shot is silently lost on a source crash. That is false assurance on the exact axis the task demands generalization.

**The fixed shape — exhaustive at EVERY nesting level, NO wildcard anywhere, co-located with `effect_class` in intershard.rs:**
```rust
/// R-6d §7 conformance: the DURABILITY CLASS of every InterShardFlow arm AND its nested
/// TransitionPayload/GhostFlow variants — a compile-forced, wildcard-free declaration. This is the
/// SIBLING of `effect_class` (which answers idempotency, payload-blind, so cannot be reused here);
/// a variant-add touches BOTH, visibly. A `ProducerLessReliable` arm has NO orchestrator
/// scan_deadlines re-driver — its push site MUST carry Durability::Retained or the one-shot is
/// silently lost on a source crash. Adding ANY variant fails to compile until classified.
#[derive(Debug, PartialEq, Eq)]
pub enum FlowDurabilityClass { ReDriven, ProducerLessReliable, Unreliable }

#[must_use]
pub fn durability_class(&self) -> FlowDurabilityClass {
    match self {
        InterShardFlow::Ghost(g) => match g {
            GhostFlow::Spawn { .. }   => FlowDurabilityClass::ReDriven,   // ongoing feed re-spawns
            GhostFlow::Delta { .. }   => FlowDurabilityClass::Unreliable, // datagram latest-wins
            GhostFlow::Despawn { .. } => FlowDurabilityClass::ProducerLessReliable, // band-exit, no re-driver
        },
        InterShardFlow::Transfer(env) => match env.payload {
            TransitionPayload::InitialSpawn { .. }  => FlowDurabilityClass::ReDriven, // saga re-drives
            TransitionPayload::StubCrossing { .. }  => FlowDurabilityClass::ReDriven, // saga re-drives
            TransitionPayload::TransientBatch { .. } => FlowDurabilityClass::ProducerLessReliable,
            // NO `_` — a new TransitionPayload MUST be classified (the CRITICAL fix)
        },
        InterShardFlow::Directory(_) | InterShardFlow::Saga(_) | InterShardFlow::SagaAck(_)
        | InterShardFlow::DirectoryReply(_) | InterShardFlow::FlushSource(_)
        | InterShardFlow::TransferAck(_) | InterShardFlow::Demote(_) | InterShardFlow::Promote(_)
        | InterShardFlow::TransientRelease(_) | InterShardFlow::TransientDrop(_)
        | InterShardFlow::ReleaseComplete(_) | InterShardFlow::TransientAbandon(_)
        | InterShardFlow::ReHome(_) => FlowDurabilityClass::ReDriven,
        // NO `_` at the outer level either (effect_class's proven discipline)
    }
}
```
Lives in `vd-wire` beside `effect_class` (both are wire-type classifiers; this is NOT a durability field ON the wire type — it is a `&self` classifier fn, no new serialized field, HR1-clean). **DELETE the plan's hand-maintained `PRODUCER_LESS_SITES` const table** (R3-CRITICAL: a stale hand-list is a second completeness trap that passes vacuously for an unregistered flow).

**The two asserting tests (Tier-A; drive the real rigs, which exist):**
1. `every_producer_less_reliable_flow_is_marked_retained` — drive `emit_transient_batch` (rig: stub.rs:3173) and the band-exit despawn (rig: stub.rs:5744), capture the `OutboundBox`, assert the pushed tuple's 4th element `== Durability::Retained`. This pins intent (the classifier) to the code (the push site).
2. `producer_less_set_is_exactly_the_documented_two` — golden pin: `{TransientBatch, Ghost::Despawn}` are the only `ProducerLessReliable` arms today (growing it is a deliberate edit). Exhaustiveness itself needs no runtime assert — the wildcard-free match IS the compile-time mechanism.

**Why this now delivers the claimed guarantee AND generalizes:** a future durable Signal / block-edit-forward is most naturally a new `TransitionPayload` variant → the wildcard-free inner match **fails to compile** until classified → if classified `ProducerLessReliable`, test #1's OutboundBox capture fails unless the push site carries `Retained`. Two gates (compile + test), no vacuity. Rejected: (b) a runtime marker on `InterShardFlow` (couples transport to frozen wire, HR1); (d) a clippy lint (inexpressible).

---

## (d) The deduplicated, severity-ranked must-fix findings (before code)

**CRITICAL-1 — conformance-test inner wildcard vacuity.** Replace the plan's `_ => ReDriven` inner arm with the exhaustive wildcard-free `TransitionPayload` + `GhostFlow` match above; delete the `PRODUCER_LESS_SITES` const; co-locate with `effect_class`; drive the two existing rigs to assert the marker. (Fix in §c.)

**HIGH-1 — complete the `send` CALLER enumeration.** Add net.rs:285 (production `send_bytes`, Ephemeral), harness/client.rs:180/281/344 (Ephemeral), orchestrator.rs:417/550, tracer.rs, chaos.rs, topology.rs (all Ephemeral), + the ~60-site test sweep. State the "no caller ever passes Retained" invariant. (Fix in §a.)

**HIGH-2 — complete the direct-`.0.push` OutboundBox site enumeration.** The 4-tuple breaks gateway.rs push_control (:894), push_to_shard (:899), push_directory (:907), Snapshot (:1849), and stub.rs Snapshot (:2811). **Give the three gateway DRY helpers (`push_control`/`push_to_shard`/`push_directory`) an explicit `Durability` param** (uniform "marker is a per-send parameter" HR3 discipline, matching `push_flow`), all callers passing `Ephemeral`; the two raw Snapshot pushes take a literal `Durability::Ephemeral` 4th element. (Fix in §e.)

**MEDIUM-1 — remove the R-6d1-handle hedge.** State decisively: R-6d1 did NOT plumb a PeerWriter/spawn_mesh sink handle (rg-confirmed zero hits); under Ruling A that is CORRECT — R-6d2 adds NO PeerWriter field; the real `Arc<Mutex<NodeOutbox>>` handle + spawn_mesh field is an R-6d3 add. Delete "confirm it landed / 1-line add here." (Fix in §b.)

**MEDIUM-2 — tighten the R-6d2 sink-threading boundary (no inert async plumbing).** Do NOT thread the sink through async `write_frame`/`peer_writer` in R-6d2 (it is None-only, untestable). Stop threading at `assign_and_retain`/`on_ack` (sync, mock-tested); `write_frame` passes `(durable, None)` and the peer_writer `on_ack` passes `None`. R-6d3 flips both to the real handle. (Fix in §b.)

**MEDIUM-3 — reconcile the two exhaustive classifiers.** Add a plan paragraph: `durability_class` is the sibling of `effect_class` (intershard.rs:199); orthogonal (idempotency vs producer-less-ness); cannot be reused (effect_class is coarse `Ghost(_)` and payload-blind `Transfer`); co-locate them so a variant-add touches both visibly. (Fixed by §c placement.)

**LOW-1 — R-6d3 delete-through efficiency (record, do NOT implement in R-6d2).** `on_ack`'s unconditional `release` will stage a pointless tombstone per acked seq on the Ephemeral-heavy Saga lane once the real sink lands. Record the R-6d3 cure: add `retained: bool` to `RetainedFrame` (set true iff `assign_and_retain` wrote through), guard the release `if rf.retained`. Tightens outbox⊆RAM to an equality on the durable subset. (Not an R-6d2 defect — sink is None; a binding R-6d3 note.)

**LOW-2 — stored-value coherence seam.** Keep ⟦R6d2-A2⟧ (store `encoded`, epoch u32::MAX — HR3-clean, one framing home; CONFIRMED in scope at mesh.rs:588). Record the R-6d2/R-6d3 seam contract: `scan_all`'s value is a FULL framed `ReliableFrame` at epoch u32::MAX; R-6d3 replay MUST `decode_frame::<ReliableFrame>(strip_envelope(val)).bytes` then re-send at epoch 0 (the stored epoch is don't-care). Add an R-6d2 outbox round-trip test asserting `decode_frame` recovers the exact payload, so the override is proven replay-consumable now.

**LOW-3 — correct the ProdTransport reasoning.** §2b's "the mesh writer reads it" is wrong for ProdTransport: its `OutFrame` goes to the loopback bridge writer (lib.rs test infra), which never calls `assign_and_retain`. Reword: "sets `OutFrame.durability` for struct-construction completeness on the shared io-prod `OutFrame`; the loopback bridge ignores it. Only the MeshTransport `OutFrame` reaches `assign_and_retain`." No code change.

**LOW-4 — `on_ack` class-source comment.** `rf.frame.class == the lane's class` by the per-class-lane invariant (lanes: `BTreeMap<MsgClass, ReliableLaneSender>`, mesh.rs:1518); add a one-line comment so a reader doesn't infer a cross-class delete.

---

## (e) The exact FSM + seam + test changes (diff-plan)

**sim/io/mod.rs:** add `Durability` enum (~§a); `send` sig grows trailing `durability: Durability` (line 338); update the trait doc (line 336) with the "consumed only by the mesh reliable lane; every other impl ignores it; datagram ignores it unconditionally" contract note.

**sim/runtime.rs:** `OutboundBox(pub Vec<(NodeId, MsgClass, Bytes, Durability)>)` (line 28); `push_flow` grows `durability: Durability`, pushes the 4-tuple (line 63-72); `push_renewals` passes `Ephemeral` (line 85).

**sim/stub.rs:** `push_flow(g.dest, Saga, &Transfer(env), Durability::Retained)` (line 2134); `push_flow(neighbor.source, GhostReliable, &Despawn.., Durability::Retained)` (line 2667); every OTHER stub `push_flow` (721/735/766/969/1384/1433/1523/1569/1637/1806/1989/2175/2214/2252/2292/2678) → `Ephemeral`; the direct Snapshot push (line 2811) → 4-tuple `.., Ephemeral`.

**connection-plane/gateway.rs:** `push_control` (890)/`push_to_shard` (897)/`push_directory` (902) each grow a `durability: Durability` param, callers pass `Ephemeral`; direct Snapshot push (1849) → `.., Ephemeral`; `reply_ack` (913) uses `push_flow(.., Ephemeral)`.

**node/app.rs:** `flush_phase` (237-259) destructures the 4-tuple, passes `durability` into `send` (249), preserves it on requeue; `requeued`/`shed_over_cap` (270) become 4-tuple, the shed closure reads `.1` via `|(_, class, _, _)|`; the ~10 test `.0.push((..))` sites (338/381/521-525/590/623-627/658) → 4-tuple `.., Ephemeral`.

**io-prod/lib.rs:** `OutFrame` (133) gains `pub(crate) durability: Durability`; both constructors set it — mesh.rs:1720 (`durability` from the send param) and `ProdTransport::send` (165, struct-completeness only, §f); `ProdTransport::send` sig grows the param (163).

**io-prod/mesh.rs:**
- `ReliableLaneSender` (453) gains `peer: NodeId`; `new(peer, incarnation, retry_cap)` (496); **update ~40 `new` + `assign_and_retain` + ~15 `on_ack` test call sites** (literal `NodeId`/args).
- `assign_and_retain` (566) grows `(durable: bool, sink: Option<&mut dyn OutboxSink>)`; AFTER the RAM insert (line 599), the monomorphic straight-line shim: `if durable { if let Some(sink) = sink { sink.retain(&OutboxKey{peer:self.peer, class, incarnation:self.incarnation, seq}, &encoded); } }` (reuse `encoded` from line 588; the `Err` returns at 589/594 precede the insert ⇒ a shed frame is never mirrored).
- `on_ack` (540) grows `mut sink: Option<&mut dyn OutboxSink>`; in the retire loop after `retry_bytes` subtract, `if let Some(s) = sink.as_deref_mut() { s.release(&OutboxKey{peer:self.peer, class:rf.frame.class, incarnation:self.incarnation, seq:self.base}); }` (+ the LOW-4 class-invariant comment).
- `write_frame` (1513): `let durable = matches!(frame.durability, Durability::Retained);` (the ONE below-seam lowering); call `assign_and_retain(local, frame.class, &frame.bytes, durable, None)` (1542) — `None` in R-6d2 (MEDIUM-2). Datagram arm (1657) untouched.
- peer_writer `on_ack` call (1128): pass `None` in R-6d2.
- **NO** `write_frame`/`peer_writer` sink PARAM in R-6d2 (MEDIUM-2 defers it); NO `replay_lanes` change (replay never retains/acks — confirmed `replay_batch` reads `retry.values()` only).

**io-prod/outbox.rs:** unchanged (reuse `OutboxSink`/`OutboxKey`/`NodeOutbox`).

**Tests:**
- *Tier-A (sim, 100%):* `push_flow` both variants → assert the 4th element; `Durability::default()==Ephemeral`; `flush_phase` requeue-preserves-durability under back-pressure; `shed_over_cap` reliable-first order unchanged with a Retained/Ephemeral mix; the two §c conformance tests. (HR5 note per §f — Durability is CARRIED, never matched, in sim.)
- *Tier-B (io-prod, MockOutboxSink):* `durable_true_with_sink_writes_through`; `durable_false_records_nothing`; `durable_true_no_sink_noop`; `on_ack_deletes_through_retired_prefix`; `on_ack_stale_epoch_releases_nothing`; `commit_batches_all_lanes`; `peer_field_builds_the_key`. Plus ONE real-`NodeOutbox` glue test: `assign_and_retain(..,true,Some(real)); commit(); scan_all()` returns the framed bytes AND `decode_frame::<ReliableFrame>(that).bytes` == payload (LOW-2 proof).

**Gate:** `cargo test --workspace` green; Tier-A sim 100% region+branch; io-prod Tier-B floor held/ratcheted. **No process/boot/delivery test** (R-6d3/R-6d4).

---

## (f) HR3 / HR5 / frozen-seam / no-dep / hot-path rulings

- **HR3 — CLEAN.** ONE `ReliableLaneSender` FSM (gains `peer` + params, no fork); ONE `OutboxSink`. `Durability` is a per-send PARAMETER at `push_flow` and at the three gateway DRY helpers (HIGH-2 makes it uniform) — exactly like `MsgClass`, decided per-send, NEVER a `match ShardProfile`/`NodeKind`, never a lane fork. Shard and gateway both call the identical marked helper for their producer-less one-shots.
- **HR5 — MET.** *Sim (Tier-A):* `Durability` is CARRIED, never matched — there is NO discriminant branch in sim (the `flush_phase`/`shed_over_cap` destructures bind it, they don't branch on it), so both variants are covered by construction once each is pushed once; no `matches!` false-arm (assert on the tuple element per HR5(d)). The ONLY enum-discriminant read is the io-prod (Tier-B) `matches!` lowering in `write_frame`, covered by the mock `durable=true/false` cases. *The FSM write-through* is a monomorphic (`&mut dyn` object-safe, one monomorphization) branchless shim per HR5(a); the four `{durable T/F}×{sink Some/None}` combos are the mock cases; the `None`-in-prod false-arm is covered by production runs. No per-monomorphization gotcha.
- **Frozen `sim::io` seam — TOUCHED additively, lockstep, owned (HIGH-2).** `Transport::send` grows a trailing `Durability` across all 6 impls in the same slice; mem/fabric/prod/mock/perpeerlanes accept-and-ignore, only mesh acts (`OutFrame.durability`→`write_frame`→`assign_and_retain`). The `Store` trait and `Inbound` enum are NOT touched. Justified: the bit is minted above the seam (`push_flow` caller) and consumed below it (writer-task `assign_and_retain`); the ONLY channel is `send()→OutFrame→mpsc`, so a marker on the `OutboundBox` tuple alone dies at `flush_phase→send()`. This is why the vetted §8 "send unchanged" text is mechanically impossible — the plan correctly overrides it.
- **ProdTransport `OutFrame.durability`** is set for struct-construction completeness on the SHARED io-prod `OutFrame`; the loopback bridge writer ignores it (test infra, no reliable-lane FSM). Only the MeshTransport `OutFrame` reaches `assign_and_retain` (LOW-3 corrected).
- **No new dependency — MET.** `Durability` is a new `sim::io` enum; the write-through reuses the R-6d1 `OutboxSink`/`NodeOutbox` over the existing redb.
- **20 Hz unreliable datagram hot path — BYTE-IDENTICAL.** `Durability` never reaches the `Reliability::Unreliable` arm (mesh.rs:1657); `assign_and_retain` is reliable-only; the `durable` lowering lives only inside the `Reliable` arm; `send_datagram` is untouched. Confirmed zero cost.

---

## (g) FINAL VERDICT

**REVISE.** The architecture, both rulings' direction (defer-the-sink; conformance-test-shape-a), and the frozen-seam decision are SOUND and code-accurate. But three CONFIRMED must-fix findings block implement-readiness as written:

- **CRITICAL-1** — the conformance test as specified is VACUOUS on the exact generalization axis the task demands (the `Transfer(env) => match { _ => ReDriven }` wildcard silently mis-classifies a future producer-less `TransitionPayload`, and the hand-maintained `PRODUCER_LESS_SITES` const compounds it). Fixed by the wildcard-free exhaustive `TransitionPayload`+`GhostFlow` match co-located with `effect_class`, deleting the const, driving the two existing rigs.
- **HIGH-1 / HIGH-2** — the lockstep enumeration is INCOMPLETE: it omits the production `send_bytes` caller (net.rs:285) + the harness/orchestrator/tracer callers + the direct-`.0.push` OutboundBox sites (gateway ×4, stub ×1). All Ephemeral; the gateway helpers should take the marker as a param for HR3 uniformity.

Fold in CRITICAL-1, HIGH-1, HIGH-2, and the MEDIUMs (remove the R-6d1-handle hedge; stop the sink-threading at the FSM boundary, not through async `write_frame`; reconcile the two classifiers) and the plan becomes **SOUND_TO_IMPLEMENT**. The LOWs (record the R-6d3 delete-through `retained:bool` cure + the commit-blocking constraint; pin the epoch-u32::MAX stored-value seam contract with a round-trip test; correct the ProdTransport reasoning; add the on_ack class-invariant comment) are non-blocking but should be carried into the doc.

Key files/lines: `crates/sim/src/io/mod.rs:336-338` (seam + `Durability`), `crates/sim/src/runtime.rs:28,63,85` (OutboundBox+push_flow), `crates/sim/src/stub.rs:2134,2667,2811` (Retained sites + direct Snapshot), `crates/connection-plane/src/gateway.rs:890,897,902,1849` (direct-push helpers), `crates/node/src/app.rs:237-270` (flush_phase+shed), `crates/client/src/net.rs:285` (production send_bytes caller), `crates/harness/src/client.rs:180,281,344` (ScriptedClient callers), `crates/io-prod/src/lib.rs:133,162-170` (OutFrame+ProdTransport), `crates/io-prod/src/mesh.rs:453,496,540-554,566-601,1128,1513-1542,1657,1711` (FSM+write_frame+peer_writer+MeshTransport), `crates/wire/src/intershard.rs:93-166,199-289,400-484` (InterShardFlow+effect_class sibling+GhostFlow/TransitionPayload for the conformance classifier).

=====================================================================
## PART 2 — ORIGINAL DESIGN (fuller detail the synthesis corrects)
=====================================================================

I have complete grounding across all six impls (`MemTransport`, `MeshTransport`, `ProdTransport`, `FabricTransport`, `PerPeerLanes` [node/app.rs test], `MockTransport` [client/net.rs test]) plus the real client send path (`send_bytes` calling `transport.send`). Now I have everything to produce the implement-ready plan.

Here is the implement-ready R-6d2 plan.

---

# R-6d2 — IMPLEMENT-READY PLAN: additive `Durability` seam + FSM write-through

## 0. Grounding deltas vs the task framing (read first — two corrections)

Two things the task prompt states are **wrong against the actual code** and reshape the plan:

- **`push_flow` + `OutboundBox` live in `crates/sim/src/runtime.rs` (Tier-A `vd-sim`), NOT `io-prod/runtime.rs`.** `io-prod/src/runtime.rs` is the bin-shell `EnvConfig`/tick-pacer — it has no `push_flow`. `OutboundBox(pub Vec<(NodeId, MsgClass, Bytes)>)` is the sim-side staging vec (runtime.rs:28). The `Durability` marker therefore rides a **4th tuple element on `OutboundBox`** and `push_flow` grows the param **in Tier-A sim** — this is where HR5 100% bites hardest (every arm must be covered), and it is the correct home for a marker that "crosses the frozen `sim::io` seam" since `Durability` itself must be a `sim::io` type.
- **The saga `AwaitAdopt` split (CRITICAL-2/3) is R-6d3, NOT R-6d2.** The vetted plan (§6, lines 470-478) scopes R-6d2 to *only* the seam + FSM write-through/delete-through; the `saga.rs:922` phase-split, boot replay, `TransientDiscard`, and the durability-before-send gate are all R-6d3. The task's item-2 ("FSM WRITE-THROUGH") is R-6d2; the saga closure is not. I hold that boundary — R-6d2 is a **pure plumbing + write-through slice with no behavior change to delivery** (nothing fsyncs-before-send yet; that gate is R-6d3).

Everything else in the task framing is confirmed. The 6 impls are confirmed exactly: `MemTransport` (sim/io/mem.rs:343), `MeshTransport` (io-prod/mesh.rs:1712), `ProdTransport` (io-prod/lib.rs:163), `FabricTransport` (harness/fabric.rs:516), `PerPeerLanes` (node/app.rs:475, **#[cfg(test)]**), `MockTransport` (client/net.rs:537, **#[cfg(test)]**). Real client send path is `send_bytes` (net.rs:283) calling `transport.send`.

---

## 1. The `Durability` type

**Name / location / shape:**
```rust
// crates/sim/src/io/mod.rs — beside MsgClass / Reliability / SendError (the frozen seam vocab)
/// Per-send producer-intent marker (R-6d): does this reliable one-shot need to survive a SOURCE
/// process crash? `Retained` frames are write-through-mirrored to the durable outbox below the seam
/// and replayed on boot; `Ephemeral` frames ride the RAM retry buffer only (re-driven by their
/// producer's own scan_deadlines if lost). A property of the INDIVIDUAL send, stated at the push_flow
/// site — NEVER inferred from MsgClass (Saga/GhostReliable each carry BOTH kinds) and NEVER a match on
/// ShardProfile/NodeKind (HR3). The datagram (Unreliable) path ignores it entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Durability {
    /// RAM-only; lost on a source crash, re-driven (if at all) by the producer. The default.
    #[default]
    Ephemeral,
    /// Write-through to the durable outbox before send; replayed on boot. Producer-less one-shots only.
    Retained,
}
```

**Enum, not bool — three reasons:**
1. **Self-documenting at every call site.** `push_flow(dest, Saga, &flow, Durability::Retained)` reads as intent; `push_flow(dest, Saga, &flow, true)` is a mystery bool (the codebase's `parse_or`/reliability all use named enums — `Reliability::{Reliable,Unreliable}` is the exact sibling precedent).
2. **`#[derive(Default)] = Ephemeral`** makes the "features once, add-a-marker-only-when-durable" scope model (⟦A2⟧) explicit and gives a clean `..Default::default()`-style default for the 30+ existing `push_flow` sites without a magic `false`.
3. **Extensibility without a wire break** — if a future tier needs `RetainedAcrossReceiver` or a fsync-cadence hint, it is an additive enum variant, not a second bool. (The bit stays a *single* enum today — no premature variants.)

**Why in `sim::io` (not io-prod):** it crosses the frozen `Transport::send` seam, so every impl (incl. `MemTransport` in Tier-A sim) must name it. It cannot live in io-prod (sim would then depend on io-prod, inverting the dependency law). It sits next to `MsgClass`/`SendError`, the other seam-crossing send vocab. The io-prod outbox never sees `Durability` — it sees the already-resolved `durable: bool` the FSM passes (the enum→bool lowering happens once, in `write_frame`).

---

## 2. The frozen-seam change + every impl (HIGH-2, rg-confirmed) + `OutFrame` + `push_flow`

### 2a. `Transport::send` new signature (sim/io/mod.rs:338)
```rust
fn send(&mut self, to: NodeId, class: MsgClass, bytes: Bytes, durability: Durability)
    -> Result<MsgId, SendError>;
```
Additive trailing param. **Justification for touching the frozen seam** (owned, per HIGH-2): the durability bit is minted above the seam (the `push_flow` caller decides) but consumed below it (the writer task's `assign_and_retain`); the only channel from caller to writer is `send()→OutFrame→mpsc`; `send`'s frozen sig carries no field for it, so a marker placed *only* on the `OutboundBox` tuple is dropped at the `flush_phase→send()` boundary. A sibling `send_retained` method was rejected in the vetted design (§8): it forces every impl (incl. `MemTransport`) to implement a second method it no-ops, and forks the FIFO/`MsgId` accounting. One param, one path.

Update the `Transport` doc block (mod.rs:336) with a one-line contract note: *"`durability` is a producer-intent hint consumed ONLY by the io-prod mesh reliable lane (write-through to the durable outbox); every other impl (mem/fabric/prod/mock) accepts and ignores it — behavior-identical to Ephemeral. The Unreliable datagram path ignores it unconditionally."*

### 2b. The complete impl list + exact change to each

| Impl | File:line | Change |
|---|---|---|
| `MemTransport` | sim/io/mem.rs:343 | Add `_durability: Durability` param; **ignore** (no fsync). Body byte-identical. |
| `FabricTransport` | harness/fabric.rs:516 | Add `_durability: Durability`; ignore. `Tracked` unchanged. |
| `PerPeerLanes` (#[cfg(test)]) | node/app.rs:475 | Add `_durability: Durability`; ignore. |
| `MockTransport` (#[cfg(test)]) | client/net.rs:537 | Add `_durability: Durability`; ignore. |
| `ProdTransport` | io-prod/lib.rs:163 | Add `durability: Durability`; thread into `OutFrame.durability` (loopback bridge — no outbox itself, but the field must flow so a future loopback outbox is not a re-plumb; today the mesh writer reads it). |
| `MeshTransport` | io-prod/mesh.rs:1712 | Add `durability: Durability`; set `OutFrame{ .., durability }`. |

Every impl that constructs an `OutFrame` (mesh + prod) sets `.durability`; the four test/mem/fabric impls simply accept-and-drop the param (`_durability`).

### 2c. `OutFrame.durability` field (io-prod/lib.rs:133)
```rust
pub(crate) struct OutFrame {
    pub(crate) to: NodeId,
    pub(crate) class: MsgClass,
    pub(crate) bytes: Bytes,
    pub(crate) msg_id: MsgId,
    pub(crate) durability: Durability,   // NEW — mesh writer lowers to `durable: bool` in write_frame
}
```
Both `OutFrame` constructors set it: mesh.rs:1720 and prod lib.rs:165. `#[derive(Debug)]` stays (Durability derives Debug).

### 2d. `OutboundBox` tuple + `push_flow` (sim/runtime.rs:28, :63)
`OutboundBox` grows to a 4-tuple:
```rust
pub struct OutboundBox(pub Vec<(NodeId, MsgClass, Bytes, Durability)>);
```
`push_flow` grows the param:
```rust
pub fn push_flow(&mut self, to: NodeId, class: MsgClass, flow: &InterShardFlow, durability: Durability) {
    let bytes = crate::io::bytes(postcard::to_allocvec(flow).expect("closed wire enums serialize infallibly"));
    self.0.push((to, class, bytes, durability));
}
```
`push_renewals` (runtime.rs:80) passes `Durability::Ephemeral` (renewals are idempotent + heartbeat-covered — never durable).

### 2e. `flush_phase` (node/app.rs:237-259) threads the marker into `send`
`std::mem::take` now yields `(NodeId, MsgClass, Bytes, Durability)`; the loop destructures the 4th and passes it:
```rust
for (to, class, bytes, durability) in pending {
    if blocked.contains(&to) { requeued.push((to, class, bytes, durability)); continue; }
    match transport.send(to, class, bytes, durability) {
        Ok(_) => sent += 1,
        Err(SendError::QueueFull(returned)) => { blocked.insert(to); requeued.push((to, class, returned, durability)); }
    }
}
```
`requeued: Vec<(NodeId, MsgClass, Bytes, Durability)>`. **`shed_over_cap` (app.rs:270)** takes `&mut Vec<(NodeId, MsgClass, Bytes, Durability)>` — its `retain`/`filter` closures only read `class` (element `.1`), so the pattern becomes `|(_, class, _, _)|`; no logic change (durability never affects the unreliable-first shed order — a `Retained` frame is *always* reliable, so it already rides the reliable-shed tail).

### 2f. The `Durability::Retained` push sites (enumerate — the ONLY two today)
Per vetted §1.3, the closed producer-less set:
1. **`TransientBatch`** — stub.rs:2134 `push_flow(g.dest, MsgClass::Saga, &InterShardFlow::Transfer(env))` → **add `Durability::Retained`**.
2. **Band-exit `GhostFlow::Despawn`** — stub.rs:2667 `push_flow(neighbor.source, MsgClass::GhostReliable, &..Despawn..)` → **add `Durability::Retained`**.

**Every other `push_flow` site** (all ~30, enumerated by the earlier rg: saga_runtime.rs ×9, orchestrator.rs ×2, gateway.rs ×1, stub.rs's other ~16, push_renewals) → **`Durability::Ephemeral`**. Rationale per site class: saga-runtime flows re-drive via `scan_deadlines`; the `GhostFlow::Spawn` (stub.rs:1523) and `Delta` are covered by the ongoing feed / are unreliable; every ack/reply is orchestrator-re-driven. All confirmed producer-*ful* or loss-tolerant.

> **Note on the ghost Despawn `Retained` choice:** the vetted design lists it (§1.3 #2) but R-6d2 only *marks* it — the write-through happens, but the boot-replay that makes it load-bearing is R-6d3. Marking it now is correct (not dead) because the FSM write-through in R-6d2 *does* mirror it to the outbox; it just isn't replayed until R-6d3. Alternatively, defer marking Despawn to R-6d3 to keep R-6d2's only `Retained` site as TransientBatch — **I rule: mark BOTH now**, because the §7 conformance test (item B) needs the full producer-less set marked to be meaningful, and marking without replay is inert-but-correct (an outbox row written and never read is GC'd, not a loss).

---

## 3. The FSM write-through / delete-through (io-prod/mesh.rs — Tier-B)

### 3a. `ReliableLaneSender.peer` (⟦A4⟧, L5) — mesh.rs:453 struct, :496 `new`
Add `peer: NodeId` field, set in `new(peer, incarnation, retry_cap)`. **Every `ReliableLaneSender::new` call site updates** (the two production sites: write_frame:1541 `or_insert_with(|| ReliableLaneSender::new(frame.to, incarnation, retry_cap))` — `frame.to` is the peer; plus ~20 unit-test sites in mesh.rs get a literal peer, e.g. `NodeId(2)`). No behavior change to the RAM path — the field only builds the `OutboxKey`.

> **Wait — `frame.to` vs writer `dest`.** In `write_frame`, `frame.to == dest` always (the frame was routed to this peer's lane). Use `dest` (the writer's canonical peer) for clarity; assert `frame.to == dest` is already implied by the routing. Set `peer: dest` in `new`.

### 3b. `assign_and_retain` new signature (mesh.rs:566) + the monomorphic write-through
```rust
fn assign_and_retain(
    &mut self, from: NodeId, class: MsgClass, bytes: &[u8],
    durable: bool, sink: Option<&mut dyn OutboxSink>,   // NEW
) -> Result<u64, AssignReject> {
    // ... UNCHANGED through the RAM insert (mesh.rs:597-599):
    self.retry.insert(seq, RetainedFrame { frame, framed_len });
    self.retry_bytes += framed_len as usize;
    self.next_seq += 1;
    // NEW — HR5(a) monomorphic straight-line write-through, AFTER the RAM insert (⟦A3⟧ RAM-first):
    // this whole method is monomorphic (no generics), so the two ifs are countable per-test-case, not
    // per-monomorphization. `encoded` (the epoch-u32::MAX framed bytes from the cap check, mesh.rs:588)
    // is reused — H3 one-encode. The stored value is the ALREADY-FRAMED bytes (the outbox wraps them in
    // its OUTBOX_FORMAT_VERSION envelope internally; R-6d1 outbox.rs:117 encode_value).
    if durable {
        if let Some(sink) = sink {
            sink.retain(&OutboxKey { peer: self.peer, class, incarnation: self.incarnation, seq }, &encoded);
        }
    }
    Ok(seq)
}
```
**Placement:** strictly *after* the RAM insert (mirror ⊆ RAM within a tick, ⟦A3⟧). The `Err` returns (Unframable at :589, BufferFull at :594) happen *before* the insert, so a rejected frame is never mirrored — correct (a shed frame must not leave an orphan outbox row).

**Value stored:** the `encoded` buffer (epoch-u32::MAX framed `ReliableFrame`). This contradicts vetted ⟦A5⟧ which prefers `postcard(ReliableFrame{epoch:0})`. **I resolve to storing `encoded` (the u32::MAX framed bytes) for R-6d2**, because (a) it is a pure write-through of exactly what the RAM retry holds (RetainedFrame carries the same frame), zero transform, and (b) the epoch re-stamp to 0 is a **boot-replay (R-6d3) concern** — R-6d3's `inject_replay` decodes the frame and re-frames at epoch 0 anyway, so the stored epoch is don't-care. Storing `encoded` keeps R-6d2 a mechanical mirror with no re-encode divergence. The `OUTBOX_FORMAT_VERSION` envelope (already in outbox.rs) versions the value so R-6d3 can change the stored shape if needed. **Flag:** if R-6d3's replay decode needs epoch-0 canonical bytes, it re-frames on read — the stored u32::MAX epoch is stripped there, not here. This matches outbox.rs's doc ("opaque already-framed bytes").

### 3c. `on_ack` delete-through (mesh.rs:540)
```rust
fn on_ack(&mut self, ack_incarnation: u64, ack_epoch: u32, ack_through: u64,
          mut sink: Option<&mut dyn OutboxSink>) -> usize {   // NEW param
    if ack_incarnation != self.incarnation || ack_epoch != self.epoch { return 0; }
    let base_before = self.base;
    while self.base < self.next_seq && self.base <= ack_through {
        if let Some(rf) = self.retry.remove(&self.base) {
            self.retry_bytes -= rf.framed_len as usize;
            // NEW — delete-through the retired seq, monomorphic straight-line:
            if let Some(s) = sink.as_deref_mut() {
                s.release(&OutboxKey { peer: self.peer, class: rf.frame.class, incarnation: self.incarnation, seq: self.base });
            }
        }
        self.base += 1;
    }
    debug_assert!(self.base <= self.next_seq);
    (self.base - base_before) as usize
}
```
Note `on_ack` has no `durable` param — it releases unconditionally when a sink is present (releasing a key that was never retained is a redb no-op-delete, harmless; `release` stages a `delete`, idempotent per Store LWW). The `class` comes from the retired frame (`rf.frame.class`), so a per-class lane deletes only its own keys.

### 3d. Threading the sink through the writer (peer_writer / write_frame / replay_lanes / ack path)

The sink is **per-node, held by the `PeerWriter`** (R-6d1 already added the handle field to `spawn_mesh`→`PeerWriter` per the vetted §6 R-6d1 bullet — confirm it landed; if not, it is a 1-line field add here). But **for R-6d2 the ruling in §4 below is DEFER the real shared sink** — so in R-6d2 the sink threaded through the writer is `None` in production, and `MockOutboxSink` in unit tests. The threading *shape* is:

- **`write_frame`** (mesh.rs:1513): gains a `sink: Option<&mut dyn OutboxSink>` param and a `durable: bool` derived from `frame.durability` (`let durable = matches!(frame.durability, Durability::Retained);` — one monomorphic lowering). Passes `(durable, sink.as_deref_mut())` into `assign_and_retain` at :1542. The unreliable arm (:1657) ignores both — **datagram path byte-identical** (Durability never reaches it).
- **peer_writer `w.rx.recv()` arm** (mesh.rs:1140-1157): passes the writer's sink into `write_frame`. In R-6d2 production this is `None` (see §4).
- **peer_writer ack arm** (mesh.rs:1120-1136): `lane.on_ack(e.incarnation, e.epoch, e.ack_through, sink.as_deref_mut())`. **Concurrency caveat**: the ack arm and the send arm both want `&mut` the sink within the same `select!` iteration — but `select!` runs exactly ONE branch per poll, so there is no simultaneous borrow; `sink.as_deref_mut()` is fresh each branch. Fine.
- **`replay_lanes`** (mesh.rs:1439) / **`replay_one_lane`**: replay does NOT re-retain (the rows are already on disk); it re-sends existing retained frames. So `replay_lanes` needs **no sink param** — it never calls `assign_and_retain` or `on_ack`. (Confirmed: replay_batch reads `retry.values()`, no mutation of retry/outbox.) This keeps replay untouched.

### 3e. `commit()` cadence (batched — one fsync per flush-tick)
The `sink.commit()` (fsync barrier) is called **once per writer drain cycle**, NOT per frame. But in R-6d2 with `sink=None` in production, there is nothing to commit. The commit-cadence *placement* is designed here and *activated* in R-6d3: after the `select!` loop drains a burst of sends+acks in one wake, call `sink.commit()` once (batching all lanes' staged retains/releases into one fsync). In R-6d2 the MockOutboxSink tests drive `commit()` explicitly to assert batching. **The real per-tick commit + the durable-before-send gate is R-6d3** (that gate is exactly CRITICAL-1, deferred).

---

## 4. RULING (A): DEFER the real shared sink to R-6d3 — thread the param, pass `None` in production

**Decision: option (ii) — thread the `Option<&mut dyn OutboxSink>` parameter through the FSM + write_frame + the ack path, pass `None` in production, exercise the write-through with `MockOutboxSink` in unit tests. Defer the real-sink injection, the concurrency design, the fsync-before-send gate, and boot replay to R-6d3.**

**Why (ii) over (i):**

1. **The vetted plan already sliced it this way** (§6, lines 470-483): R-6d2 = "seam + FSM write-through/delete-through"; R-6d3 = "boot replay + the durability gate (CRITICAL-1) + the saga closure." Injecting the real shared sink *now* pulls CRITICAL-1's writer-task fsync-before-send and HIGH-1's no-deadlock argument into R-6d2, which the vetted design explicitly places in R-6d3. Re-slicing against the vetted plan would be a design change, not a refinement.

2. **The concurrency design is genuinely premature at R-6d2.** The real question — "one `Arc<Mutex<dyn OutboxSink>>` shared across N peer_writer tasks vs a per-writer handle" — is coupled to the **durable-before-send gate** (CRITICAL-1): the gate blocks a QUIC send on `wait_durable_through(seq)`, and HIGH-1 requires the fsync to run on the store's *off-tick writer thread* (⟂ the tokio peer_writer) so no tokio worker parks on the fsync. That whole "no-await-under-lock + fsync-thread-independence" argument is R-6d3's load. Designing the Mutex sharing in R-6d2 — before the gate exists — would design a lock whose contention profile (does a peer_writer hold it across a `commit().await`?) is undetermined until the gate lands. **Building the concurrency now = building it blind.**

3. **The threaded-`None` param is NOT dead-plumbing**, because:
   - The `MockOutboxSink` FSM unit tests (§6) exercise `assign_and_retain(.., durable=true, Some(&mut mock))` and `on_ack(.., Some(&mut mock))` — every write-through/delete-through branch is *executed and asserted* in R-6d2's own test suite. The param has live callers with observable effects.
   - The `durable: bool` lowering in `write_frame` and the `Durability` seam threading are *fully live in production* (they flow caller→OutFrame→write_frame every tick), even though the sink they'd feed is `None`. The plumbing carries a real value; only the terminal sink is deferred.
   - HR5: with `sink=None` in production, the `if let Some(sink)` false-arm is covered by production runs *and* the unit tests cover the `Some` arm — both arms exercised (see §7 HR5).

4. **Each slice stays independently gate-able + non-dead**: R-6d2 gates on "FSM mirrors to a MockOutboxSink correctly + the seam threads a marker end-to-end, zero delivery-behavior change." No process test, no boot, no fsync-timing. R-6d3 then flips `None→Some(real NodeOutbox)`, adds the gate, adds boot replay, adds the saga closure — and *its* gate is the composed SIGKILL process proof (R-6d4).

**Exactly what R-6d3 then adds (the deferred set):**
- Inject the real `NodeOutbox` handle into the writer: **one `Arc<Mutex<NodeOutbox>>` per node, cloned into each `PeerWriter`** (the vetted L5 contract: "the NodeOutbox is shared by all writer tasks behind one handle; retain/release keys are disjoint per (peer,class,inc,seq), so redb LWW-per-key is safe"). The lock is held only across the *synchronous* `retain`/`release`/`commit` staging calls — **never across an `await`** (the mesh discipline). The actual fsync runs on the RedbStore's off-tick writer thread (store.rs), so the Mutex is released before any blocking. **No lock across a QUIC send.**
- The durable-before-send gate (CRITICAL-1): in `write_frame`'s reliable arm, `if durable { stage + commit + wait_durable_through(seq) }` *before* `write_reliable_frame`, with the fsync on the off-tick thread (HIGH-1 no-deadlock).
- Boot replay (MEDIUM-1, HIGH-3): `scan_all()→send(.., Retained)` in ascending order before `build_app` consumes the transport; `gc_below(incarnation)` after fresh-row durability.
- The saga `AwaitAdopt` phase-split + `TransientDiscard` (CRITICAL-2/3).

**Rejected alternative (i) inject-now:** would either (a) hold a Mutex across the send path (violates no-await-under-lock, the exact deadlock HIGH-1 guards) or (b) require the full off-tick-fsync-thread gate to land in R-6d2, collapsing R-6d3 into R-6d2 and making the slice un-gate-able without the process proof. Rejected on both counts.

---

## 5. RULING (B): the §7 forgotten-marker conformance test — shape (a), an exhaustive `InterShardFlow`-arm classification test, is FEASIBLE and lands with R-6d2

**Decision: shape (a) — an exhaustive `match` over `InterShardFlow` (and its `GhostFlow` sub-arms) classifying each into `{ReDriven | ProducerLessReliable | Unreliable | NotAShardOneShot}`, asserting that the classification is total (a new arm fails to compile), and pinned to the actual push sites by a companion assertion. It generalizes beyond TransientBatch.** Feasible; lands with R-6d2.

**The problem it solves:** an author adds a new producer-less reliable flow, forgets `Durability::Retained` at the push site, and loses the one-shot silently on source crash with no compile error. A test cannot *directly* observe "this flow has no re-driver" (a semantic property of the orchestrator's `scan_deadlines`). But it *can* force the author to make a **deliberate, compile-checked declaration** per flow, and cross-check that declaration against the push sites.

**The concrete shape (lives in `crates/sim/src/io/mod.rs` or a new `crates/sim/tests/durability_conformance.rs`, Tier-A):**

```rust
/// R-6d §7 conformance: the DURABILITY CLASS of every InterShardFlow arm — a compile-forced,
/// exhaustive declaration. Adding an InterShardFlow variant (or a GhostFlow lifecycle arm) FAILS TO
/// COMPILE here until the author classifies it. A `ProducerLessReliable` arm is one with NO orchestrator
/// scan_deadlines re-driver — its push site MUST carry Durability::Retained or it is silently lost on a
/// source crash. This makes the ⟦A2⟧ "forgot the marker" failure a BUILD failure, not a silent loss.
#[derive(Debug, PartialEq, Eq)]
enum FlowDurabilityClass { ReDriven, ProducerLessReliable, Unreliable, Internal }

fn classify(flow: &InterShardFlow) -> FlowDurabilityClass {
    match flow {
        InterShardFlow::Transfer(env) => match env.payload {
            // TransientBatch is producer-less (source-shard emit, no orchestrator re-driver).
            TransitionPayload::TransientBatch { .. } => FlowDurabilityClass::ProducerLessReliable,
            _ => FlowDurabilityClass::ReDriven,        // crossing envelopes re-drive via the saga
        },
        InterShardFlow::Ghost(g) => match g {
            GhostFlow::Despawn { .. } => FlowDurabilityClass::ProducerLessReliable,  // band-exit, no re-driver
            GhostFlow::Spawn { .. }   => FlowDurabilityClass::ReDriven,   // covered by the ongoing feed / respawn
            GhostFlow::Delta { .. }   => FlowDurabilityClass::Unreliable, // GhostDelta, latest-wins
            // ... every current GhostFlow arm, exhaustively
        },
        InterShardFlow::Saga(_) | InterShardFlow::SagaAck(_)
            | InterShardFlow::Directory(_) | InterShardFlow::DirectoryReply(_)
            | InterShardFlow::Demote(_) | InterShardFlow::Promote(_)
            | InterShardFlow::ReHome(_) | InterShardFlow::TransferAck(_)
            | InterShardFlow::FlushSource(_) | InterShardFlow::TransientRelease(_)
            | InterShardFlow::TransientDrop(_) | InterShardFlow::ReleaseComplete(_)
            | InterShardFlow::TransientAbandon(_) => FlowDurabilityClass::ReDriven,
        // NO wildcard `_ =>` — a new arm MUST be classified here (compile-forced).
    }
}
```

**The two asserting tests:**

1. **`every_producer_less_reliable_flow_is_marked_retained`** — the load-bearing pin. For each `ProducerLessReliable` flow, assert its **actual push site passes `Durability::Retained`**. This is done by a **table cross-check**: a `const PRODUCER_LESS_SITES: &[(&str, Durability)]` listing each site by name and its marker, asserted `== Retained`, PLUS a test-only re-invocation that captures the `OutboundBox` after driving each producer path (the stub already has `rig.tick(...)` harnesses — e.g. the TransientBatch emit test at stub.rs and the Despawn test at stub.rs:5674) and asserts the pushed tuple's 4th element is `Retained`. The `classify` fn is the *registry of intent*; the OutboundBox capture is the *proof the code matches intent*.

2. **`classification_is_exhaustive_and_producer_less_set_is_the_documented_two`** — asserts the `ProducerLessReliable` set is exactly `{TransientBatch, Ghost::Despawn}` today (a golden pin: growing it is a deliberate edit), and that `classify` compiles (the exhaustive match is the compile-time guard — no runtime assert needed for exhaustiveness; the *absence of a wildcard* is the mechanism).

**Why this is the right shape and generalizes:**
- **It turns "forgot the marker" into a build failure** in the realistic path: a new producer-less flow adds a variant → `classify`'s wildcard-free match fails to compile → author must classify it → if they mark it `ProducerLessReliable`, test #1's OutboundBox capture fails unless the push site carries `Retained`. Two gates: compile (must classify) + test (must mark).
- **It generalizes beyond TransientBatch** — durable Signals and block-edit forwarding (the end goal) will each add an `InterShardFlow` arm; each forces a `classify` decision and, if producer-less, a marked push site. The mechanism is flow-shaped, not TransientBatch-specific.
- **Honest about the semantic gap**: it does NOT *automatically* infer "no re-driver" (impossible statically). It forces a *human declaration* and *cross-checks the declaration against the code*. That is the strongest feasible guard — (b) a runtime registry/marker on the flow type is heavier (a new field on every flow, a wire concern) and (c) a doc-comment convention is unenforced. Shape (a) is the maximal build-time enforcement.

**Rejected (b) runtime marker-on-flow-type:** would put durability *on the wire type* (`InterShardFlow`), coupling a transport concern to the frozen wire contract (HR1/frozen-seam violation) and duplicating the per-send marker. Rejected. **Rejected (d) "narrower lint":** clippy cannot express "producer-less ⇒ Retained." Rejected.

---

## 6. Test plan

**Tier-A (sim, 100% region+branch) — the seam-param coverage:**
- `push_flow` with each `Durability` variant → asserts the 4-tuple's 4th element (extend runtime.rs tests:117). Both `Ephemeral` and `Retained` arms.
- `Durability::default() == Ephemeral` (one-liner golden).
- `flush_phase` threads durability through to `send` and preserves it on requeue — extend the app.rs flush tests (:506, :587): a `Retained` frame that back-pressures requeues *with* its `Retained` marker (assert the 4th element survives the requeue). This covers the `requeued.push((.., durability))` arm.
- `shed_over_cap` with a mix of `Retained` (reliable) + `Ephemeral` frames: assert the reliable-first shed order is unchanged (durability doesn't reorder). Covers the `|(_, class, _, _)|` closure.
- **The §5 conformance tests** (both, above) — Tier-A.
- Every `MemTransport::send` call in existing tests compiles with the new param (mechanical). The `_durability` ignore arm needs no *branch* coverage (it's an unused binding, not a branch) — but confirm no clippy `unused` (prefix `_`).

**Tier-B (io-prod) — the FSM write-through, MockOutboxSink:**
```rust
struct MockOutboxSink { retained: BTreeMap<[u8; OUTBOX_KEY_LEN], Vec<u8>>, commits: u32, released: Vec<[u8; OUTBOX_KEY_LEN]> }
impl OutboxSink for MockOutboxSink { /* records exact key bytes + value; commit() bumps a counter */ }
```
Tests (synchronous, no tokio, no redb):
1. **`durable_true_with_sink_writes_through`** — `assign_and_retain(from, Saga, b"x", durable=true, Some(&mut mock))` → mock has exactly one row at key `(peer, Saga, inc, 0)` with value == the framed bytes.
2. **`durable_false_records_nothing`** — `durable=false, Some(&mut mock)` → mock empty. (The `if durable` false arm.)
3. **`durable_true_no_sink_is_a_noop`** — `durable=true, None` → no panic, RAM insert still happens, returns `Ok(0)`. (The `if let Some` None arm.)
4. **`on_ack_deletes_through_the_retired_prefix`** — retain 0,1,2 durable; `on_ack(inc, epoch, 1, Some(&mut mock))` → mock released keys `{seq0, seq1}`, seq2 retained.
5. **`on_ack_stale_epoch_releases_nothing`** — the early-return arm with a sink present → no release (covers the `!= incarnation` guard doesn't touch the sink).
6. **`commit_batches_all_lanes`** — two lanes (Saga, GhostReliable) each retain into the SAME mock, one `commit()` → `commits == 1`, both rows present. (Batching contract.)
7. **`peer_field_builds_the_key`** — `ReliableLaneSender::new(NodeId(7), ..)` → the retained key's peer bytes == 7.

**Glue test (io-prod, real `NodeOutbox`):** already covered by R-6d1's `retain_scan_release_round_trip`. Add ONE R-6d2 test: `assign_and_retain(.., true, Some(&mut real_node_outbox)); commit(); scan_all()` returns the framed bytes — proves the FSM→real-sink path (not just the mock). Uses a temp redb file (the R-6d1 `TempOutbox` guard pattern).

**No process test, no boot test, no delivery-behavior test in R-6d2** — those are R-6d3/R-6d4. R-6d2's gate: `cargo test --workspace` green + Tier-A sim 100% + io-prod Tier-B floor held/ratcheted (the outbox.rs module is already 99.56%/100%; the new FSM branches ratchet it up).

---

## 7. HR + seam compliance rulings

- **HR3 (one tooling) — CLEAN.** ONE `ReliableLaneSender` FSM (gains `peer` + two params, no fork); ONE `OutboxSink` seam. `Durability` is a per-send **parameter** at `push_flow` (exactly like `MsgClass`), decided by the caller per send — **never** a `match ShardProfile`/`NodeKind`, never a lane fork. The shard and orchestrator both call the identical `push_flow(.., Durability::Retained)` for their producer-less one-shots. No shard-kind match anywhere in the path.

- **HR5 (Tier-A 100%) — MET, with the OR-pattern/monomorphic discipline:**
  - The `Durability` param flows through Tier-A sim (`push_flow`, `OutboundBox`, `flush_phase`, `shed_over_cap`). Every arm is coverable: both `Durability` variants are pushed in tests; the `flush_phase` requeue-with-durability arm is hit by the back-pressure test; `shed_over_cap`'s closure reads only `class` (the durability binding is `_`, not a branch). No `matches!` false-arms (use `assert_eq!` on the tuple element per HR5(d)).
  - The `MemTransport`/mem no-op arm: `_durability` is an **ignored binding, not a branch** — no region to cover beyond the method being called (it is in the existing send tests).
  - The io-prod FSM write-through is **Tier-B floor** but written as an **HR5(a) monomorphic branchless shim**: `assign_and_retain`/`on_ack` are monomorphic (no generics), so the `if durable { if let Some }` regions count per-test-case, and the MockOutboxSink cases (§6 #1-3) exercise all four combinations `{durable T/F} × {sink Some/None}`. No per-monomorphization gotcha (the sink is `&mut dyn`, object-safe, one monomorphization).

- **Frozen `sim::io` seam — TOUCHED additively, lockstep, owned.** `Transport::send` grows a trailing `Durability` param across **all 6 impls in the same slice** (mem/fabric/prod/mock/perpeerlanes/mesh); mem/fabric/prod-loopback/mock/perpeerlanes accept-and-ignore, only mesh acts (via `OutFrame.durability`→`write_frame`→`assign_and_retain`). The `Store` trait is **NOT touched**. The `Inbound` enum is not touched. This is the HIGH-2 owned additive change — justified (the bit cannot be inferred from `MsgClass` nor carried on the frozen `OutFrame` without the seam param).

- **No new dependency — MET.** `Durability` is a new enum in `sim::io`; the write-through uses the existing R-6d1 `OutboxSink`/`NodeOutbox` over the existing `redb` `RedbStore`. No crate added.

- **20 Hz unreliable datagram hot path — BYTE-IDENTICAL.** `Durability` never reaches the datagram arm: `write_frame`'s `Reliability::Unreliable` arm (mesh.rs:1657) ignores `frame.durability`; `assign_and_retain` is reliable-only (mesh.rs:1528 gate unchanged). The `durable: bool` lowering happens only inside the `Reliability::Reliable` arm. The datagram send path (send_datagram, mesh.rs:1698) is untouched. Confirmed zero cost.

---

## 8. Assumptions + rejected alternatives (flagged)

- **⟦R6d2-A1⟧** The saga `AwaitAdopt` split, boot replay, durable-before-send gate, and `TransientDiscard` are R-6d3 — R-6d2 has **zero delivery-behavior change** (write-through mirrors to an outbox that is `None` in production; nothing fsyncs-before-send). If a reviewer wants the write-through *live* in production in R-6d2, that pulls CRITICAL-1/HIGH-1 forward — rejected (re-slices the vetted plan).
- **⟦R6d2-A2⟧** Store the `encoded` (epoch-u32::MAX) framed bytes in R-6d2, not the epoch-0 canonical form — the re-stamp is R-6d3's replay concern; the `OUTBOX_FORMAT_VERSION` envelope lets R-6d3 change the stored shape. If R-6d3's replay needs epoch-0 bytes it re-frames on read. (Differs from vetted ⟦A5⟧, which is a boot-replay-tier concern not yet live.)
- **⟦R6d2-A3⟧** Mark band-exit `GhostFlow::Despawn` `Retained` NOW (not deferred to R-6d3) — inert-but-correct (mirrored, not yet replayed) and required for the §5 conformance test to be meaningful.
- **⟦R6d2-A4⟧** `ReliableLaneSender::new` gains `peer` as the FIRST param; ~20 unit-test call sites updated mechanically (literal `NodeId`).
- **Rejected:** a `bool` durability field (opaque at call sites; §1); a `send_retained` sibling seam method (forces every impl to implement/no-op a 2nd method, forks MsgId accounting; §2a); injecting the real shared sink in R-6d2 (premature Mutex/concurrency design blind to the CRITICAL-1 gate; §4); a runtime durability-marker on the `InterShardFlow` wire type (couples transport to frozen wire, HR1; §5 rejected-b); a clippy lint for "producer-less ⇒ Retained" (inexpressible; §5 rejected-d).

**Slice boundary (what R-6d2 delivers, gate-able alone):** the `Durability` seam type + the 6-impl lockstep + `OutFrame.durability` + `OutboundBox` 4-tuple + `push_flow`/`flush_phase`/`shed_over_cap` threading + the 2 `Retained` push sites + `ReliableLaneSender.peer` + `assign_and_retain`/`on_ack` write-through/delete-through against a **threaded-but-`None`-in-prod** sink + the MockOutboxSink FSM tests + the §5 conformance test + the Tier-A seam-param coverage. **Gate:** `cargo test --workspace` green, Tier-A sim 100%, io-prod Tier-B floor held. No process/boot/delivery test (R-6d3/R-6d4).

Key files/lines: `crates/sim/src/io/mod.rs:336-345` (seam + new `Durability`), `crates/sim/src/runtime.rs:28,63,80` (`OutboundBox`+`push_flow`), `crates/node/src/app.rs:237-261,270` (`flush_phase`+`shed_over_cap`), `crates/io-prod/src/lib.rs:133,163` (`OutFrame`+`ProdTransport::send`), `crates/io-prod/src/mesh.rs:453,496,540,566,1513,1541-1542,1712,1120-1136` (FSM+write_frame+peer_writer+`MeshTransport::send`), `crates/io-prod/src/outbox.rs` (existing `OutboxSink`/`OutboxKey`, reused unchanged), `crates/sim/src/mem.rs:343` / `crates/harness/src/fabric.rs:516` / `crates/client/src/net.rs:537` / `crates/node/src/app.rs:475` (the 4 no-op impls), `crates/sim/src/stub.rs:2134,2667` (the 2 `Retained` sites), `crates/sim/src/saga.rs:922` (the split — R-6d3, NOT this slice).