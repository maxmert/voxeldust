Decisive on Q1/gateway: `SagaView` (`admin.rs:65`) exposes only `transfer`, `state`, `since` — it does **not** render `session` or `gateway`. And `active_transfers()` (`:607`) exposes `subject/source/dest` — not `session`. So `SessionId(0)`/`gateway` are **never surfaced in any admin view**. The adversary's HIGH claim that `SessionId(0)` is "surfaced in the admin `SagaView`" is **factually wrong** — I read the struct. The only remaining exposure is `SagaSnapshot` persistence (`:1263`), which is internal, not operator-facing.

I have full seam confirmation on every load-bearing claim. Here is the definitive finalized plan.

---

# D-43 #9 — DEFINITIVE fix: crossing-triggered transient starts THE `BatchHandoff` saga

## Root cause (confirmed at the seam)

`handle_transient_crossing_request` (`crates/node/src/saga_runtime.rs:1467-1488`) emits `TransientCrossingGrant` and **never calls `start_transfer`**, so `runtime.sagas` has no entry under `crossing_transfer_id(req.subject, req.src_realm_fence, 0)`. When the dest's `BatchAdopted{transfer_id}` arrives, `deliver` (`:1109-1114`) does `runtime.sagas.get_mut(&transfer)` → `None` → silent early-return. The adopt-before-drop handoff never runs: `orch.batch_goes == []`, dest stuck `Arriving`, never `Held`. **The fix is: start the saga at grant time, keyed on the same `batch`.**

## Adversary adjudication (verified against code, not prose)

| Adversary finding | Verdict after code read | Evidence |
|---|---|---|
| **CRITICAL: per-entity keying defeats HR2 batching** | **REAL mechanism, but NOT a blocker of THIS fix — must be ledgered, not folded in.** See §5. | `fan_out_crossing:2662` emits one `TransientCrossingRequest{subject: Entity(e)}` **per crossing entity**; orch keys `batch = crossing_transfer_id(Entity(e), …)`; `emit_transient_batch:2743 .entry(batch)` groups by it → 1-item groups. Confirmed. **BUT** it is the *pre-existing, whole-chain, both-class* keying — the durable path (`:2635`) is identically per-entity. Re-keying is a wire-shape redesign of the request/grant/source-flip chain (grant carries a single `subject`, `:643`; source flips one item, `:1622`). The adversary itself concedes "not mechanical... ripples." |
| **HIGH: `SessionId(0)` surfaced in admin `SagaView` + persisted** | **HALF-WRONG.** `SagaView` (`admin.rs:65`) exposes ONLY `transfer/state/since` — NOT `session`/`gateway`. `active_transfers()` exposes `subject/source/dest` — NOT `session`. **No admin surface reads it.** The only carry is internal `SagaSnapshot` persistence. Downgrade to the real residual risk: a *future* class-blind reader/collision. **Adopt `SessionId::NONE` — cheap, correct — but it is a HR6-clean hardening, not the crash-safety hole the adversary claimed.** | Read `views():591`, `active_transfers():607`, `SagaView` struct. |
| **MED: no transient rehydrate/SIGKILL proof** | **REAL. Land WITH the fix** — a first-time-reachable persisted state. | §7. |
| **MED: composition regression is topology-level** | **REAL. Required, not optional.** The `saga_runtime` rig simulates the dest, so it proves "saga starts" but not "silent early-return cured end-to-end." | §6 Test B. |
| Crux, timing, HR2-reuse, idempotency, id-match (1–5) | **All HOLD.** Verified. | §2, §4. |

**Bottom-line correction to the adversary:** the CRITICAL is a *real limitation* but re-keying is **NOT "THE fix" and is NOT safely mechanical** — folding it in would balloon D-43 #9 into a multi-seam wire redesign (subject-set-aware grant, source `Crossing` grouping). The correct engineering is: land the per-entity saga-start (correct, idempotent, unblocks D-43 #9), and open a **first-class SCALE slice** to re-key the batch on the realm-pair. This is honest, not a green-light of a "non-fix": per-entity is **correct** (one saga per crossing entity, adopt-before-drop lands, dest ends Held) — it is just **not yet coalesced**. That distinction is load-bearing and I am not eliding it.

---

## §1 — Ordered implementation steps (seam-accurate)

**Step 0 (precondition — Q1 resolved to YES):** `crates/core/src/ids.rs:77` — add
```rust
impl SessionId { pub const NONE: SessionId = SessionId(u128::MAX); }
```
plus a unit test asserting `SessionId::NONE != SessionId(0)`. (High byte `u128::MAX` cannot collide with the sequential connection-plane session ids, which start low.) **Rationale:** `SessionId(0)` is a valid allocatable id; a transient saga has no session, so it must carry a typed sentinel, not a real-collidable literal — even though no admin view reads it today (verified), it is persisted into `SagaSnapshot` and a future class-blind reader must not alias onto real session 0.

**Step 1 (THE fix):** `crates/node/src/saga_runtime.rs`, `handle_transient_crossing_request`, `Some(rec)` arm — after the existing grant `push_flow` + counter (`:1482`), **keep the grant emit UNCHANGED** and append the guarded saga start:

```rust
runtime.transient_crossings_granted += 1;

// D-43 #9: START the SAME BatchHandoff saga the machinery drives (HR2 one-machinery), keyed on
// `batch` == the id the dest acks BatchAdopted under, so `deliver` routes the adopt onto a LIVE
// AwaitAdopt saga instead of the silent None early-return. The grant/emit leg is UNCHANGED: the
// source emits on the Held->Crossing flip, NOT on this go-token (see below), so no double-emit.
// The `contains_key` fast-skip avoids a redundant pending push when the saga is already live from a
// prior tick's re-request (`process_starts:1504` is the AUTHORITATIVE one-saga guard). A transient
// takes NO directory lock (`locks_directory_key(Transient)=false`, :1516), so a debris burst stays
// burst-isolated.
if !runtime.sagas.contains_key(&batch) {
    let ctx = SagaCtx {
        transfer: batch,
        session: SessionId::NONE,                    // INERT — transient short-path never reads ctx.session
        subject: DirectoryKey::Realm(req.to_realm),  // INERT provenance — a transient never enters the directory
        expected_fence: rec.fence,                   // the dest realm-lease fence the go-token commits at
        source: from,                                // transport-origin connection = the source shard
        dest: rec.authority.node(),
        class: DurabilityClass::Transient,
        needs_provision: false,
        from_realm: req.from_realm,
        to_realm: req.to_realm,
    };
    runtime.start_transfer(ctx, from);               // gateway arg INERT for Transient — pass `from`, not a fabricated NodeId
}
```

This is the exact HR3/HR2 twin of the durable `handle_crossing_request:1425-1441` (build ctx → `runtime.start_transfer(ctx, gateway)`), differing only in `class: Transient` (the intended policy fan-out) and the inert `session`/`subject`/`gateway` a session-less debris batch has no value for.

---

## §2 — The grant / go-token reconciliation (RESOLVED — no double-emit, grant path NOT replaced)

Verified in code, not prose:

- **The source emits its batch on the GRANT, driven purely by `Held→Crossing`.** `on_transient_crossing_grant` (`stub.rs:1610`) field-assigns the grant's four fields into `TransientStatus::Crossing` with **zero read of any orchestrator saga state** (`:1622-1630`). `emit_transient_batch` (`stub.rs:2714`) groups every `Crossing` item and `push_flow_durable(g.dest, …)` the envelope **directly to the dest** (`:2778`), independent of any saga.
- **The go-token is orchestrator-LOCAL, wire-silent.** `IssueTransientGo` (`saga_runtime.rs:974-984`) pushes a `BatchGo` and feeds `CasWon` back into the FSM on the same tick (`events.push_back`). It emits **nothing on the wire** — not to source, not to dest — so it cannot re-trigger the source emit.

**Therefore: one batch on the wire (the `Held→Crossing` flip), one saga (the orchestrator's private commit authority). Starting the saga at grant time is invisible to the source → no double-emit.** The grant-flips-and-emits path is the **correct required source-side leg** and is left untouched; what was MISSING is the orchestrator-side saga that commits the go-token and owns `AwaitAdopt→…→Done`. Both legs key on the SAME `batch` → once the saga exists, `deliver` (`:2254-2263 → :1109`) routes `BatchAdopted` onto the live `AwaitAdopt`. HR2-clean: same `start_transfer`, same `BatchHandoff` FSM (proven at `transient_subject_commits_via_the_go_token_not_a_cas:3209`), `saga.rs` needs **zero change**.

---

## §3 — The exact Transient `SagaCtx` (every field, production source in-scope at `:1470`)

| field | value | seam justification (verified) |
|---|---|---|
| `transfer` | `batch` = `crossing_transfer_id(req.subject, req.src_realm_fence, 0)` (`:1470`) | THE key `deliver` looks up (`:1110`); identical to grant `batch` (`:1479`), source `Crossing{batch}` (`stub.rs:1628`), envelope `transfer_id` (`stub.rs:2763`), dest `BatchAdopted{transfer_id}` (`stub.rs:2831`). All from the same pure fn, `attempt=0` fixed → redelivery re-derives identically. |
| `session` | `SessionId::NONE` — **INERT** | The transient short-path (`start`→`BatchCommitting`→`IssueTransientGo`→`CasWon`→`BatchHandoff`→every `EmitTransient*`) reads `ctx.transfer/source/dest/expected_fence/class` ONLY — never `ctx.session` (verified `:974-1049`, `:1648-1701`). Drives NO gateway route-swap. |
| `subject` | `DirectoryKey::Realm(req.to_realm)` — **INERT provenance** | `locks_directory_key(Transient)=false` (`:1516`) short-circuits, so `subject` is never a CAS/lock key. Never enters the directory. |
| `expected_fence` | `rec.fence` (== grant's `dst_realm_fence`, `:1478`) | Becomes `new_fence` on `CasWon` (`:982`) = the sole promote authority carried into every `EmitTransient*` egress. |
| `source` | `from` (fn param `:1465`, transport-origin = source shard) | Target of `EmitTransientRelease/ReleaseComplete/Abandon` (`:992/1015/1029`). |
| `dest` | `rec.authority.node()` (`:1476`) | Target of `EmitTransientPromote/Discard` (`:1003/1043`); the node the source shipped the envelope to. |
| `class` | `DurabilityClass::Transient` | The HR2 fan-out driver at `start()`/`commit_action()`. |
| `needs_provision` | `false` | Durable-warp-only. |
| `from_realm` / `to_realm` | `req.from_realm` / `req.to_realm` | Faithful provenance (never a key for a transient). |

**The `gateway` 2nd arg:** pass `from` (INERT). Stored on `LiveSaga.gateway` (`:1525`) but the transient short-path never reads it, and **`views()` does not render it** (verified `:591-599` — only `transfer/state/since`). Passing a real in-scope NodeId avoids fabricating a sentinel.

---

## §4 — Idempotency (redelivered request → no second saga)

Two guards, both already in code, both required:
1. **`process_starts:1504` `contains_key`** — the **authoritative** one-saga gate; its doc-comment (`:1500-1503`) already calls out the transient-no-lock clobber risk explicitly. A `ReDriven` re-request whose saga is live is absorbed.
2. **The handler `contains_key` fast-skip (Step 1)** — steady-state optimization: prevents a redundant `pending` push, mirroring the durable arm's structure (`:1423-1424`).

`crossing_transfer_id` is a pure fn (`intershard.rs:691`, `attempt=0` fixed). No directory lock (`:1516`) → a mass debris burst never serializes against a durable saga (HR2 burst isolation, structural). Dest-side `adopt_transient_batch` journals by `(transfer, step)` (`stub.rs:2804`) → re-adopt re-acks without re-inserting.

---

## §5 — SCALE: the CRITICAL, adjudicated honestly (the ONE design decision — DECIDED)

**The fact (confirmed):** `batch` keys on `req.subject = DirectoryKey::Entity(e)` → per-entity. N distinct debris crossing the same boundary same tick = N batches = N sagas = N go-tokens = N fsyncs. `emit_transient_batch`'s group-by collapses to singletons. HR2's literal "batched TransientGo" is NOT achieved at entity granularity. The density fixture (`p1_volume_dense_hundreds`, N≥128) crossing a boundary → 128 sagas/wave.

**The decision (DECIDED — do NOT silently ship per-entity as "batched"):**

1. **Land per-entity NOW as the D-43 #9 fix.** It is **correct** — one saga per crossing entity, `contains_key`-idempotent under redelivery, adopt-before-drop lands, dest ends Held, single-holder oracle passes. It is **not coalesced**. The silent-early-return bug (dest stuck Arriving) is CURED. This is the minimal seam-accurate change.

2. **Open a first-class SCALE slice (`D-43 #9-SCALE`), ledger it in `docs/design/DEFERRED.md` as 🟧, task-tracked, with the explicit end-state:** re-key the transient batch on the **realm-pair + `src_realm_fence`** (`crossing_transfer_id(DirectoryKey::Realm(to_realm), src_realm_fence, 0)` or a dedicated `(from,to,src_fence)` seed) so `emit_transient_batch`'s existing group-by collapses N items → 1 envelope / 1 saga / 1 go-token / 1 fsync (the G-TIER promise). This is **NOT mechanical** — it ripples through:
   - `TransientCrossingRequest` (`intershard.rs:630`) — currently one per entity carrying a single `subject`; must become realm-scoped OR the orch must coalesce many same-realm requests into one batch id.
   - `TransientCrossingGrant` (`:642`) — a realm-keyed grant covers many subjects; the reply becomes subject-set-aware or subject-free (the source-side `on_transient_crossing_grant:1622` currently flips exactly one item).
   - `fan_out_crossing:2661` request emission grouping.

**Why NOT fold it into THIS fix (correcting the adversary):** the adversary calls re-keying "THE fix, not deferrable" — but the same critique admits it "ripples... not mechanical," and it touches **3 frozen `vd-wire` shapes + the source-side flip**. Bundling a wire redesign into the one-arm early-return cure inflates blast radius and delays the correctness fix that unblocks the whole transient path. Per-entity is correct-and-shippable; realm-keyed is an optimization with a wire-contract cost. **The engineering-correct call is: correctness now, coalescing as a scoped, reviewed, wire-versioned slice.** This is not a ledger-hidden non-fix — it is a scoped decomposition of a correctness fix from a throughput optimization, with the optimization explicitly owned (not "recommend a DEFERRED line").

**OPEN QUESTION Q3 (needs a code read before the SCALE slice, NOT before THIS fix):** does the density soak actually route hundreds of *distinct transient entities* across a boundary in one tick in the current fixtures, or is the realistic dense case the *same* entity re-requesting (which per-entity + `contains_key` already coalesces to one saga)? Read `p1_volume_dense_hundreds` and the crossing driver it uses. If the realistic corner is same-entity redelivery, per-entity is already adequate and the SCALE slice is genuinely deferrable to when true multi-entity debris storms exist (P4/P5 physics debris). **This gates the SCALE slice's priority, not the D-43 #9 landing.**

---

## §6 — Tests

### Test A (unit, `saga_runtime.rs` 3f-C suite ~`:6100`) — the direct D-43 #9 pin

`transient_request_grant_starts_a_batchhandoff_awaitadopt_saga`: after a resolved grant, assert exactly ONE live saga under `batch`, parked in `SagaState::BatchHandoff{ phase: AwaitAdopt, new_fence: realm_fence }` (use `assert_eq!`, not `matches!` — HR5(d)), `ctx.class == Transient`, `ctx.source == SOURCE`, `ctx.dest == DEST`, `ctx.session == SessionId::NONE`, `batch_goes() == vec![(BatchId(batch), realm_fence)]`, `batch_go_writes() == 1`, and the grant emit UNCHANGED (`transient_crossings_granted == 1`).

**Idempotency extension** to `transient_request_redelivery_regrants_same_batch:6144`: after the second request, `assert_eq!(rig.live(), 1)` — one batch, one saga.

### Test B (composition — the 3g Test 4 extension — `crates/harness/src/topology.rs`) — REQUIRED, not optional

`transient_crossing_composes_end_to_end_dest_owns_and_go_token_recorded`: a 3-node topology (orch A + source shard B realm FROM + dest shard C realm TO). Seed a transient positioned to dwell over the B→C boundary; **populate the `RealmBoundaries`/dwell state** to fire the rising edge (`evaluate_realm_boundaries` is inert in prod through P3, `stub.rs:831` — tests populate it, exactly as `a_transient_crossing_emits_a_transient_crossing_request:8030` does). Step N ticks; then from `topo.inspect_all()`:
- `!orch.batch_goes.is_empty()` ("was `[]` before the fix" — the exact D-43 #9 symptom).
- `!dest.owned_transients.is_empty()` ("was stuck Arriving before the fix").
- `held_fence == go_fence` (adopt-before-drop landed the authority at the go-token fence).
- `topo.oracle().verify_single_transient_holder()` passes (no {source,dest} both-held window, `oracle.rs:403-411`).

**This is the regression the fix most needs** — the `saga_runtime` rig *simulates* the dest ack (`batch_adopted` helper), so only topology-level proves the silent-early-return is cured with a REAL dest reacting to a REAL envelope.

**OPEN QUESTION Q2 (needs a code read before writing Test B):** confirm whether a 2-shard-plus-orch transient-crossing topology fixture exists to extend. The closest scaffold is `inspect_reports_directory_and_shard_ground_truth:~1000` (single-shard) — grow it to 3 nodes, or check `dev_cluster_smoke.rs`/`process_parity.rs` for an existing multi-shard harness. If none exists, building it is part of Test B's cost and must be scoped.

### Test C (crash-safety — `saga_runtime.rs`) — REQUIRED (first-time-reachable state), see §7.

---

## §7 — HR5 + determinism + crash-safety

**HR5 100% region+branch:**
- New branching = ONE `if !runtime.sagas.contains_key(&batch)`. True arm (first request) covered by Test A; false arm (already-live) covered by the redelivery extension. Keep it a plain `if`, not `&&` (no short-circuit uncoverable edge — HR5(d)).
- `SagaCtx` build is straight-line — a branchless shim (HR5(a)).
- The auto-drive reuses code already at 100% (`transient_subject_commits_via_the_go_token_not_a_cas:3209`). `saga.rs` unchanged → no new FSM regions.
- `SessionId::NONE` const + its `!=` test: trivially covered.

**Determinism:** `crossing_transfer_id` pure (postcard of `(subject, fence, 0)`); `contains_key`/`start_transfer` push to `Vec pending` drained in order; `IssueTransientGo`'s `batch_goes` insert idempotent + order-independent (`BTreeMap`); no wall-clock, no default-hasher `HashMap`. Fully deterministic under the virtual clock.

**Crash-safety (inherited, ONE new test required):**
- Persistence is class-blind: `commit_result` stages `SagaSnapshot{ctx, state, gateway, since, flushed_pose}` under `StoreKey::Saga(transfer)` on every non-terminal transition (`:1257-1269`); the go-token separately persists as `GoTokenSnapshot` under `StoreKey::BatchGo(batch)` (`:1186-1194`); rehydrate (`:1331-1357`) re-inserts with `since=UniverseTick(0)` so `scan_deadlines` re-drives on the first post-restart tick.
- **The gap the adversary correctly flags:** today the ONLY production `start_transfer` caller is the **durable** crossing handler, so a **transient** saga is **never persisted in prod yet**. The moment this fix lands, a `Transient BatchHandoff{AwaitRelease/AwaitPromote/AwaitComplete}` `SagaSnapshot` becomes **first-time-reachable** in the store. "Inherited automatically" is a hypothesis about generic code, not a tested fact.
- **Test C (land WITH the fix, deterministic):** persist a transient saga mid-`BatchHandoff{AwaitRelease}`, drop + rehydrate the runtime, assert the saga re-inserted under `batch` (with `SessionId::NONE` intact) + re-drives its pending `EmitTransientPromote` on the next tick + `batch_goes` restored from `StoreKey::BATCH_GO`. Locks the newly-reachable transient persistence path (project discipline: SIGKILL/rehydrate proof for every persisted class).

---

## §8 — Files to change

| what | file:line | change |
|---|---|---|
| Step 0 sentinel | `crates/core/src/ids.rs:77` | add `impl SessionId { pub const NONE = SessionId(u128::MAX); }` + `!= SessionId(0)` test |
| **THE fix** | `crates/node/src/saga_runtime.rs:1482` (`handle_transient_crossing_request`, `Some(rec)` arm) | after the grant emit, `contains_key(&batch)`-guarded Transient `SagaCtx` + `runtime.start_transfer(ctx, from)` |
| FSM | `crates/sim/src/saga.rs` | **NO CHANGE** — `BatchHandoff` short-path complete |
| source | `crates/sim/src/stub.rs` | **NO CHANGE** for THIS fix (per-entity keying inherited faithfully); the SCALE slice touches `fan_out_crossing:2661` + `emit_transient_batch:2743` + `on_transient_crossing_grant:1622` |
| Test A | `crates/node/src/saga_runtime.rs` ~`:6100`/`:6144` | grant-starts-saga pin + redelivery `live()==1` |
| Test B | `crates/harness/src/topology.rs` (grow `~:1000` to 3-node) | composition: `dest.owned_transients` + `orch.batch_goes` non-empty + single-holder oracle |
| Test C | `crates/node/src/saga_runtime.rs` | transient rehydrate-mid-`AwaitRelease` proof |
| SCALE ledger | `docs/design/DEFERRED.md` | new 🟧 entry `D-43 #9-SCALE` (realm-key coalescing) + task-track |

---

## Summary of the decision + OPEN QUESTIONs

**The fix (minimal, correct, seam-accurate):** append a `contains_key`-guarded Transient-`SagaCtx` `start_transfer(ctx, from)` immediately after the grant emit in `handle_transient_crossing_request` (`:1482`), grant leg UNCHANGED. `SessionId::NONE` + `gateway=from` are inert-by-construction (verified: no admin view reads them). Idempotent via the existing `process_starts:1504` guard. Zero `saga.rs`/`stub.rs` change. HR2/HR3-clean (same machinery), deterministic, crash-safe (inherited + one new test).

**Precondition promoted:** Q1 → **Step 0** (`SessionId::NONE`, cheap, done first) — but downgraded from the adversary's HIGH: no admin surface reads it (verified `views()`/`active_transfers()`), the residual risk is only a future class-blind persistence reader.

**Required-with-the-fix (adversary MEDs sustained):** Test B (topology composition — the true regression) and Test C (first-time-reachable transient rehydrate).

**The ONE explicit design decision (adversary CRITICAL, adjudicated):** ship **per-entity** as the D-43 #9 fix (correct, uncoalesced), and open a **first-class ledgered SCALE slice** to re-key on the realm-pair for true N-into-1 batching — **NOT** fold the wire-shape redesign into this fix (it ripples 3 frozen `vd-wire` shapes + the source flip; the adversary's own critique concedes it "is not mechanical"). This is a scoped decomposition, not a hidden non-fix — per-entity genuinely cures the silent-early-return; realm-keying is a throughput optimization with a wire-contract cost.

**OPEN QUESTIONs still needing a code read before their respective work:**
- **Q2 (before Test B):** does a 2-shard+orch transient-crossing topology fixture exist in `topology.rs`/`dev_cluster_smoke.rs`, or must the 3-node harness be built? (`inspect_reports…:~1000` is the closest single-shard scaffold.)
- **Q3 (before the SCALE slice, NOT before THIS fix):** does the density soak route hundreds of *distinct* transient entities across a boundary in one tick, or is the realistic dense case *same-entity* redelivery (already coalesced to one saga by `contains_key`)? Read `p1_volume_dense_hundreds` + its crossing driver — this gates the SCALE slice's priority.

Key files: `crates/node/src/saga_runtime.rs:1467-1488` (edit site), `:1499-1531` (`process_starts` guards), `:974-1049`/`:1648-1701` (transient egress — confirms session/gateway inertness), `:591-616` (`views`/`active_transfers` — confirms neither renders session/gateway), `crates/sim/src/stub.rs:2655-2666` (`fan_out_crossing` per-entity request — SCALE root), `:2714-2786` (`emit_transient_batch` group-by), `:1610-1635` (`on_transient_crossing_grant` single-item flip), `crates/wire/src/intershard.rs:630-648` (request/grant shapes — SCALE wire cost), `:691` (`crossing_transfer_id` — re-key point), `crates/core/src/ids.rs:77` (`SessionId` — needs `NONE`).