I have all the verified findings, refutations, and verdicts in the prompt. This is a synthesis task — no further file reading is needed since the verdicts already contain the confirmed evidence. Let me produce the definitive report.

---

# GOAL-AUDIT — Voxeldust Engine — 2026-07-13 (HEAD after 42d7b73)

## VERDICT

**NO CRITICAL FINDINGS.** The base is AAA-sound and not cornered on any end-goal pillar. Everything flagged is a **latent, phase-scoped** concern in the OWED crossing-abort slice (3f-D) or its pre-P4 trigger — inert today behind the empty `RealmBoundaries` registry. There are **0 CRITICAL, 3 HIGH-class defects** (2 of which are the same 3f-D plan flaw re-found; effectively **2 distinct HIGH issues** + a set of MEDIUM/LOW), none reachable in the shipped tree. Fix them **before** task #133 plants the first real realm band.

---

## CONFIRMED CRITICAL + HIGH (ranked)

### H1 — OWED 3f-D crossing-origin discriminator (Option 3-A) is UNSOUND against the committed WIRE-vs-HEAD fence split
*(re-found 5×: rated HIGH twice, MEDIUM three times; the sharper HIGH framing is correct — treat as the top action item)*

- **Area:** `crates/node/src/saga_runtime.rs` + `scripts/slice3f_optionB_plan.md` (Sub-slice 3, Q1).
- **Evidence:** `handle_crossing_request` mints `ctx.transfer = crossing_transfer_id(req.subject, req.subject_fence)` over the **WIRE** fence (`saga_runtime.rs:1314`) but sets `ctx.expected_fence = subj.fence` over the **CURRENT HEAD** fence (`:1332`); the doc at `:1297-1300` states these are "distinct by design" and diverge when the head advances between latch and resolve. `crossing_transfer_id` is a pure FNV over `postcard(subject, fence)` (`intershard.rs:663-673`; unit test at `:1317` asserts a differing fence ⇒ a differing id). The plan's *preferred* discriminator (`slice3f_optionB_plan.md:70,73`) recomputes `crossing_transfer_id(ctx.subject, ctx.expected_fence) == ctx.transfer` = `id(subject, HEAD) == id(subject, WIRE)` → **FALSE on exactly the fence-advanced crossing** the design supports. The plan's own supporting citation (`intershard.rs:626-628`) is misattributed — that range is `TransientCrossingRequest` fields, not the id derivation.
- **Why it matters:** 3f-D exists to positively clear the source `RequestInFlight` latch on a pre-CAS abort. With 3-A, the `pending_abort_replies` insert false-negatives on the fence-advanced path → no `CrossingAborted` emitted → the durable latch is stranded forever (the exact HR2 liveness hole 3f-D is built to close), and it ships **green** because the fence-advanced case is the untested one (happy-path tests use a stable fence).
- **Fix:** Reject Option 3-A. The plan's own Q1 gate (`:206`) already routes to **3-B** when "the orchestrator mints the id by another means" — which the committed code does. Answer Q1 **3-B**: either retain the WIRE `subject_fence` on `SagaCtx` and recompute against the minting fence, or add an explicit `is_crossing_origin` flag behind a **versioned `SagaSnapshot`** store-format bump (pre-prod genesis-wipe acceptable if stated). Also viable: gate on `from_realm != to_realm && class == Durable` (immutable fields already on `SagaCtx`). Resolve BEFORE writing a line of 3f-D; correct the misleading "preferred/verified" gloss and the mis-cited line-70 evidence.

### H2 — Durable `CrossingRequest` is emitted EXACTLY ONCE, not re-driven; an unresolved dest-realm strands the entity, and 3f-D does NOT cover it
- **Area:** sim trigger / wire durability classification / orchestrator consumer.
- **Evidence:** `fan_out_crossing`'s durable arm emits `CrossingRequest` only on `Entry::Vacant`; `Entry::Occupied` is a pure `crossings_suppressed_in_flight += 1` no-op (`stub.rs:2572-2597`). The latch clears only via `on_saga_demote` (commit, `:1568`) or `on_crossing_aborted` (`:1635`, no production producer). `should_commit` is a strict rising edge (`inward_ticks == n_entry`, `geometry.rs:642`) — it does not re-fire while resident. `request_ttl_ticks` is `0` everywhere and read by **no** consumer. The transport reliable-lane retry covers a **lost-in-flight** request but NOT a **delivered-but-unresolved** one (cumulative ack retires the frame; `io-prod/src/lib.rs:117`). An unresolved crossing never starts a saga (no `SagaCtx`, no tombstone), so 3f-D's `pending_abort_replies` cannot cover it. The `ReDriven` attribution in `intershard.rs:485-491` and `saga_runtime.rs:1305` ("the source re-drives it / boundary re-emission") is **false** for the entering-crossing path.
- **Why it matters:** A dest realm being unresolved (`Realm(to_realm)` head absent) is a real steady-state at P4/P5 — a realm shard mid-lease, not-yet-claimed, or partitioned. An entity crossing in emits ONE request, gets counted `crossing_unresolved`, and is permanently suppressed — a silent liveness strand on the transfer-first spine (prove authority/abort robust BEFORE features).
- **Fix:** Wire `request_ttl_ticks` to a real re-request: on `Entry::Occupied` with `now - latch_tick >= request_ttl_ticks`, RE-EMIT `CrossingRequest` (the orchestrator's `sagas.contains_key` guard absorbs a duplicate that DID start a saga; an unresolved one re-tries the head reads). Alternatively have the `crossing_unresolved` arm reply a retriable NAK the source re-arms on. Correct the two `ReDriven` comments to attribute re-drive to the transport reliable lane, and add a DEFERRED 3f/P4 entry for the unresolved-dest strand distinct from the abort-reply.

---

## MEDIUM / LOW LEDGER

*(All inert today — `RealmBoundaries` default-empty, `evaluate_realm_boundaries` authority-gated + empty-candidate early-return. Sequence with task #133 / 3f-D, never after the first real band.)*

| # | Sev | Area | One-line | Fix owner |
|---|-----|------|----------|-----------|
| M1 | MED | crossing trigger / abort egress | Durable pre-CAS abort strands the source latch — **no production `CrossingAborted` producer**; the no-TTL "positive clear" is half-built (only the source *consumer* landed as a 3f-prep plant). | Land 3f-D **before** planting the first real band; add the binding DEFERRED entry. |
| L1 | LOW | crossing re-arm after abort | Clearing the latch (3f-D) is necessary but not sufficient: `CrossingProgress` dwell (`inward_ticks`/`was_member`/`last_commit_tick`) is never re-armed, so a still-dwelling entity won't re-request until it physically re-crosses. | Reset/drop the `CrossingProgress` entry on abort-clear; add an abort-while-dwelling scenario. Fold into 3f-D. |
| L2 | LOW | wire/node id derivation | `crossing_transfer_id` (vd-wire, tag 0x39) and `rehome_transfer_id` (vd-node, tag 0x37) are byte-identical FNV-1a bodies differing only by tag — DRY hazard on the determinism primitive (HR3). | Hoist `namespaced_transfer_id(tag, subject, fence)` into vd-wire; both call it. |
| L3 | LOW | evaluate_realm_boundaries scale | Trigger is **O(N·M²)** per tick (not the claimed O(N·M)) via `boundary_depth`'s inner O(M) parent-chain scan + a per-subject `Vec` alloc; **no DEFERRED ledger entry** (the landed slice's plan promised one and dropped it). | Precompute per-boundary depth once into `Vec<u32>` (→ O(N·M)); add a `D-TRIGGER-SCALE 🟥` entry (per-cell boundary index = the D-41 grid). |
| L4 | LOW | fan_out_crossing panic | `subject_session.expect()` in the `(Authority, Durable)` arm is network-input-reachable: the arm is guarded by `durability_of(entity)` (the id tag), NOT by loop-of-origin; `adopt_transient_batch` inserts wire `item.entity` with **zero kind validation**, so a Durable-tagged batch item → held-transient → `.expect(None)` panics the shard. Contradicts HR2 (kind-generic batch). Producer is a mutual-TLS authenticated peer, so not an untrusted-client DoS. | Route on the loop not the kind (typed `SubjectKind`), OR count-and-`return` instead of panic; validate `durability_of == Transient` at the `adopt_transient_batch` chokepoint. |
| L5 | LOW | CrossingState cooldown | Anti-thrash cooldown is **per-ENTITY** while the dwell is **per-BOUNDARY** — a legitimate back-to-back station→bay Authority crossing within `k_dwell` is dropped (regression test uses `Interest`, which arms no cooldown, so the collision is untested). | Make the cooldown per-(entity,boundary) to match the dwell; add an Authority-effect A-then-B regression. Fold into task #133 (Station/Bay). |
| L6 | LOW | wire H2 same-fence re-cross | `CrossingAborted{subject,transfer}` has no generation; a same-fence re-cross re-derives the identical id, so a redelivered stale abort can wrongly clear a fresh latch (double-free). Plan flags as Q5. | Decide the wire-payload cure UP FRONT with 3f-D: append `generation` to `CrossingAborted`, store `(TransferId, generation)` in `RequestInFlight`; do NOT ship the dedup-set fallback. |
| L7 | LOW | DEFERRED registry | 3f-D abort-reply durability is tracked only in inline comments + the untracked scratch plan — **not in DEFERRED.md** (the binding registry; the plan's own Sub-slice 0 mandates adding it). | Add the `🟥` entry (WHAT/WHERE/WHEN), flip 🟩 when 3f-D + the SIGKILL crash-leg pass. |
| L8 | LOW | dead config field | `StubConfig.request_ttl_ticks` is declared + set in 4 sites, **read nowhere** — a planted-but-unwired knob; violates one-config-home discipline. | Remove until 3f consumes it, OR assert INERT-by-design. (Overlaps H2's fix — H2 would give it a reader.) |
| L9 | LOW | winner_ix identity | `CrossingState.winner_ix` is a **raw slice index** into `RealmBoundaries`; the dwell reset compares indices not `RealmId`. Stable only while the registry is immutable — a P4/P5 dynamic reorder would alias dwell to a different boundary. (Note: keying on `RealmId` alone is insufficient — coincident mouths share a realm; needs a stable per-boundary id, part of task #133.) | Key the dwell reset on stable boundary identity, not the raw index. Cheap now, expensive post-P4. |
| L10 | LOW | counter honesty | `crossing_unresolved`/`crossing_subject_gone` increment **per redelivered request** (no dedup guard), unlike `crossings_started` (guarded) — one stuck entity reads as a 20 Hz storm of distinct failures. | Guard symmetrically, OR rename to a `_requests` rate gauge. |
| L11 | LOW | orchestrator scale | `batch_goes` `BTreeMap` is UNBOUNDED (D-7d GC owed) — the concrete RSS wall under a hundreds-in-one-location transient burst; `scan_deadlines`/`reap_lapsed_leases` are per-tick linear scans (D-32). | Land D-7d `batch_goes` GC + deadline-ordered structures before a dense-transient soak. Already ledgered. |

---

## END-GOAL PILLAR READ

| Pillar | Read | Basis |
|--------|------|-------|
| **Blocks** (stations/cities/ships from blocks) | **PREPARED** | No block-machinery defect surfaced this pass. The only block-adjacent risk (L5 station→bay cooldown) is a latent trigger refinement folded into task #133 before Station/Bay Authority boundaries wire. |
| **Signals** | **PREPARED** | No signal-system finding; untouched by the crossing-abort concerns. |
| **Cross-shard** (transfer/authority) | **PREPARED, with mandatory pre-P4 work** | The saga spine, directory CAS, and fence discipline are sound. But the crossing-*trigger* layer above it is half-built: H1 (unsound 3f-D discriminator), H2 (single-shot unresolved strand), M1/L1/L6 (abort-clear + re-arm + double-free) must land as one 3f-D slice BEFORE task #133 plants a real band. Not cornered — the fixes are additive and don't touch the frozen wire beyond an append-only `CrossingAborted{generation}` grow decided up front. |
| **Multisharding meshes** | **PREPARED** | Mesh transport (reliable lane, cumulative ack, incarnation guard) is intact; H2 correctly attributes re-drive to the transport lane once its comments are fixed. Producer of the crossing path is a mutual-TLS peer (L4 is not an untrusted-client DoS). |
| **Mesh-render** | **PREPARED** | No render/client finding this pass; renderer-free client lib keeps the choice deferred at zero cost per prior audits. |
| **PvP** | **PREPARED** | No PvP-specific defect; action_bits tripwire and density fixture remain green per prior state. |
| **Hundreds-in-one-location** | **PREPARED, one untracked wall** | Correctness proven (`p1_volume_dense_hundreds`). Two scale items to track before a soak: L3 (trigger O(N·M²) + missing ledger entry — the one **untracked** wall) and L11 (`batch_goes` RSS — already ledgered D-7d). Both inert today; neither corners the design (both are additive, same D-41 cell grid serves the pruning). |

---

## WHAT THE REFUTED FINDINGS WERE (honesty trail)

Six findings were investigated and **REFUTED** — recording them so the confirmations are auditable:

1. **"CrossingRequest single-shot ⇒ source-crash silently loses the crossing"** — REFUTED. Assumed dwell/latch state *survives* a crash; it does not (`CrossingProgress`/`RequestInFlight` are in-RAM `Default` resources, reset EMPTY on restart). The restored still-in-band durable dot re-evaluates from a fresh `CrossingState` and **re-emits** — the `ReDriven` classification is truthful for the *crash* case. (Note: this is distinct from H2, which is the *delivered-but-unresolved* case, where no crash resets the latch — H2 stands.)
2. **"Commit-path latch clear is not id-gated (H2 symmetry)"** — NOT_A_BUG. The asymmetry is deliberate: abort is pre-CAS (tight re-cross window ⇒ id-gate load-bearing); Demote is a post-CAS commit terminal whose duplicate paths are all bounded to *before* the saga completes, mutually exclusive with a re-latched T2. A stale Demote can never arrive after T2 is latched.
3. **"3f-D Option 3-A unsound" (as CRITICAL/reproducible bug)** — REFUTED *as a bug in the tree*; the math is correct but there's zero implementing code and the subject is an untracked scratch plan whose own Q1 gate routes an honest reader to 3-B. **This exact observation is CONFIRMED as H1** at HIGH/MEDIUM as a **plan-quality** defect to fix before coding — not as a shipped bug.
4. **"3f-D H2 id-aliasing double-free"** (as a live defect) — NOT_A_BUG in the audited tree: doubly unreachable (no production `CrossingAborted` producer, no populated `RealmBoundaries`), already ledgered to the right phase. **Retained as L6** (the wire-payload *decision* to make up front).
5. **"No live-cluster durable outbox (`VD_OUTBOX_PATH` unset)"** — REFUTED as a live data-loss bug. The "safe because all emitters are `#[cfg(test)]`" premise is false (both are production Bevy systems), but it is safe by **data-inertness** (no crossing ⇒ no `TransientBatch`/Ghost produced), already found + downgraded in the 2026-07-11 audit and ledgered D-6 as a pre-P4 deploy TODO.
6. **"`winner_ix` raw index foot-gun"** (as MEDIUM) — the code facts are accurate but it's a future-feature-contingent design note (runtime-mutable `RealmBoundaries` doesn't exist), and the proposed `RealmId` key is itself unsound (coincident mouths share a realm). **Retained as L9** at LOW, correctly scoped to task #133.

**Pattern:** the honest discriminator throughout was *reachable-NOW vs. plan/latent/phase-scoped*. Every "HIGH shipped bug" framing collapsed to a plan defect or an inert pre-P4 concern; nothing in the running P3 tree breaks determinism, loses data, or forces a rewrite.