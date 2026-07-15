I have all the evidence I need. `Dots(pub BTreeMap<SessionId, Dot>)` — the source knows the session at trigger time (map key), and `Dot.session_fence` exists. The durable loop at :2332 uses `.values_mut()` which discards the key, so threading session requires `.iter_mut()`. Also `Dot` at :105 does NOT store a `session` field directly (only `entity` + `session_fence`), so the session comes from the map key.

I now have exhaustive verified evidence. Producing the definitive plan.

---

# SLICE 3f — DEFINITIVE IMPLEMENTATION PLAN
## Orchestrator crossing-consumer + producer-durable abort-reply egress

**Verification status:** Every seam below verified against HEAD (worktree `worktree-new-system`). The adversary's three CRITICAL findings (C1 lost-abort strands the entity, C2 real session/gateway blocker, C3 id-contract-resolved-but-unstated) are **all correct and confirmed**; they are folded in as either resolved decisions or blocking OPEN QUESTIONS. The draft's mechanical seam facts are correct except the specific line offsets and one wrong TTL set-site, corrected below.

---

## A. VERIFIED GROUND TRUTH (what actually exists at HEAD)

**Source side (Slices 3c–3e) is ALREADY LANDED.** The wire types, the id function, the source producer, and the source consumers all exist and are live:
- Wire structs `CrossingRequest`/`TransientCrossingRequest`/`TransientCrossingGrant`/`CrossingAborted` — `crates/wire/src/intershard.rs:583-624`. **`CrossingRequest` carries `{subject, from_realm, to_realm, subject_fence}` — NO `session`** (`:583-589`).
- `crossing_transfer_id(subject, subject_fence)` — `intershard.rs:635` (FNV-1a, tag `0x39`, branchless).
- All four arms classified `FlowDurabilityClass::ReDriven` — `intershard.rs:471-474`. **The `ReDriven` doc (`:281-283`) states it is sound ONLY because "a PRODUCER re-drives it on a crash (the orchestrator saga's `scan_deadlines` re-emits the step)."**
- Source producer `fan_out_crossing` — `stub.rs:2533`; durable `Vacant` arm latches + emits + arms cooldown (`:2561-2577`); `Occupied` arm is a **pure suppress no-op** `crossings_suppressed_in_flight += 1` (`:2578`). The source **never re-emits `CrossingRequest` while latched.**
- Source consumers (live): `on_crossing_aborted` (`stub.rs:1624`, exact-id-guarded clear at `:1635`), `on_transient_crossing_grant` (`stub.rs:1592`), dispatched at `stub.rs:3154`/`3146`. `on_saga_demote` clears the latch **presence-keyed** (`in_flight.0.remove(&entity).is_some()`, `:1568`) — NOT id-keyed.
- `Dots(pub BTreeMap<SessionId, Dot>)` (`stub.rs:178`); `Dot` has `entity` + `session_fence` but no `session` field — **the session is the map key.** The durable crossing loop at `stub.rs:2332` iterates `.values_mut()`, **discarding the session key.**

**Orchestrator side (3f) is NOT built.** The `drive_sagas` inbox `match postcard::from_bytes::<InterShardFlow>` is at `saga_runtime.rs:1924`; it has NO `CrossingRequest`/`TransientCrossingRequest` arm — both fall through the catch-all `_ => {}` at **`:1995`** and are **silently dropped today**.

**Key structures (verified offsets):**
- `SagaCtx` — `sim/src/saga.rs:35`, 10 fields incl. `session: SessionId` (`:37`) and `expected_fence` (`:40`). **No `crossing_origin`.**
- `LiveSaga` — `saga_runtime.rs:175`, 7 fields, **no origin marker**.
- `SagaSnapshot` — `saga_runtime.rs:138`, 5 fields (`ctx, state, gateway, since, flushed_pose`), **no marker**.
- `PendingStart` — `saga_runtime.rs:216`, 2 fields `{ctx, gateway}`.
- `start_transfer(ctx, gateway)` — `saga_runtime.rs:509`; body is `self.pending.push(PendingStart { ctx, gateway })`.
- `process_starts` — `saga_runtime.rs:1239`; destructures `for PendingStart { ctx, gateway }` at **`:1246`**; the sagas-map key is `ctx.transfer` **verbatim** (`:1251` contains_key guard, `:1263` lock_transfer, `:1267` insert). **C3 confirmed: the id rides `ctx.transfer`; nothing mints over it.**
- `run_to_quiescence(ctx, gateway, state, actions, dir, outbox, epoch, now, flush_pose)` — `saga_runtime.rs:755`; 3 call sites: `process_starts:1281`, `deliver:1017`, `process_rehome_starts:1630`.
- `ClearTransferLock` executor — `saga_runtime.rs:981` (`dir.abort_clear(ctx.subject, ctx.transfer)`). `NotifyRejected` executor pushes to `rejected` at **`:976`**.
- Terminal-abort FSM edge emits `[ClearTransferLock, Tombstone]` — `saga.rs:1436-1437`. **Terminal catch-all `(terminal @ (Done | Aborted), _) => (terminal, vec![])` — `saga.rs:1268`: a Timeout on an already-`Aborted` saga emits NOTHING.**
- `commit_result` tombstone: `runtime.sagas.remove(&transfer)` + snapshot DELETE — `saga_runtime.rs:1085-1092`. **A tombstoned saga is gone from `runtime.sagas` and cannot receive a future Timeout.**
- `rehydrate` LiveSaga reconstruction — `saga_runtime.rs:1191-1206` (field-by-field; `dead_observed_since`/`dest_adopted` re-derive to RAM defaults; nothing threads any origin marker).
- Durable `start` immediately emits `PrepareSubscribe { session: ctx.session, dest }` routed to `gateway` via the `Send` arm at `run_to_quiescence:778-779`. **C2 confirmed: even the START (not just abort) drives a gateway-routed, session-bearing command.**
- `Session` directory record exists: `DirectoryKey::Session(SessionId)` → `OwnerRecord{authority: AuthorityRef::Gateway(node), fence}` ("which gateway speaks for a connected client", `directory.rs seam:18`). `head(key) -> Option<OwnerRecord>` — `directory.rs:644`. `AuthorityRef::node()` — `directory.rs seam:56`. **But there is NO `Entity→Session` reverse index anywhere in the codebase.**
- `push_flow(to, class, flow)` defaults `Durability::Ephemeral`; `push_flow_durable(to, class, flow, Durability::Retained)` is the producer-less path — `runtime.rs:63`/`:78`. A `debug_assert!` (`:92`) fires if a `ProducerLessReliable` flow rides `Ephemeral`.
- `request_ttl_ticks` (`StubConfig`, `stub.rs:100`): set=`0` at **3 sites** — `stub.rs:3545`, `bins/src/bin/shard.rs:100`, `harness/src/topology.rs:1023`. **ZERO read sites.** (The draft's 4th site `tests/src/lib.rs:81` is WRONG — it does not exist; the field is declared+set at exactly these 4 lines total, one being the decl.)

---

## B. THE THREE OPEN QUESTIONS THAT BLOCK IMPLEMENTATION

**These MUST be resolved (by the reads noted, plus a design decision the user should ratify) before ANY code is written. Two of them force wire/frozen-seam changes that the draft assumed away.**

### OQ-1 (BLOCKER, was C2) — the durable crossing saga needs a REAL `session` + `gateway`; neither is available at the orchestrator today. **This forces a `CrossingRequest` wire append + source-side session plumbing.**

**Status: RESOLVED as a real blocker; the draft's "placeholder is sound" fallback is provably WRONG.** Verified: a durable saga starts in `Preparing` and *immediately* emits `PrepareSubscribe { session: ctx.session, dest }` to `gateway` (`saga.rs:740-744` → routed `run_to_quiescence:778-779`); the abort compensators `ThawSource`/`AbortTransfer` also carry `session` to `gateway` (`saga.rs` Aborting arms). A placeholder session/gateway would **misroute the very first message of every crossing**, not just the abort.

The orchestrator has no way to derive `session` from a `CrossingRequest`: the request carries only `{subject, from_realm, to_realm, subject_fence}`, the `Entity` `OwnerRecord` carries no session, and **no `Entity→Session` reverse index exists**. But the SOURCE has the session in hand at trigger time (`Dots` is keyed by `SessionId`; `fan_out_crossing` runs inside the per-dot loop).

**DECIDED resolution (pending user ratification of the wire-grow, per the "investigate libs/seam changes together" standing rule):**
- **Append `session: SessionId` to `CrossingRequest`** (`intershard.rs:584`). This is a frozen-seam wire append — treat it with the same care as R-2a/CA-1 (which the memory shows grew the seam deliberately). It is postcard-safe (field append) and the flow is already `MsgClass::Saga` reliable.
- **Source plumbing:** change the durable crossing loop at `stub.rs:2332` from `.values_mut()` to `.iter_mut()` so the `SessionId` key is in scope, thread `session` through `evaluate_one_subject` → `fan_out_crossing`, and set it on the emitted `CrossingRequest`. (The transient loop at `:2350` does NOT need this — see OQ-1b.)
- **Orchestrator resolution:** in `handle_crossing_request`, set `ctx.session = req.session` and resolve `gateway = dir.head(DirectoryKey::Session(req.session)).map(|r| r.authority.node())`. If the `Session` record is absent (client already gone), this is a **counted, un-startable drop with an abort-reply to the subject owner** (the entity can't cross to a dead client's route) — see 3f-2.

**Remaining read to confirm before coding:** confirm `Dot`/the source loop can cheaply expose the `SessionId` key without breaking the eviction/`retain_live` borrow structure at `stub.rs:2315-2323` (the `.values_mut()`→`.iter_mut()` change is inside the same `dots.0` borrow). Low-risk but must be eyeballed. **This is the single largest scope item the draft omitted entirely.**

### OQ-1b — does the TRANSIENT consumer need a session? **NO — verified.**
A transient `start` takes the SHORT PATH (`saga.rs:727-732`): `BatchCommitting` → go-token, **never** `PrepareSubscribe`, never a session-bearing gateway command. `TransientCrossingRequest` correctly carries no session and needs none. 3f-3 is unaffected by OQ-1. (This asymmetry is real and correct — do not add a session to the transient request.)

### OQ-2 (BLOCKER, was C1) — the abort-reply is a single-shot with NO re-driver; on any lost `CrossingAborted` the entity is stranded forever. **This is the make-or-break correctness hole.**

**Status: RESOLVED as a real hole; the draft's "at-least-once transport + Timeout re-drive suffices" is provably FALSE.** Verified chain: (1) `CrossingAborted` fires only in the `ClearTransferLock` executor (`saga_runtime.rs:981`), reachable only on the single `Aborting→Aborted` terminal edge (`saga.rs:1436`); (2) that same run tombstones the saga — `commit_result` does `sagas.remove` + snapshot DELETE (`:1085-1092`); (3) a Timeout can never re-drive it because the saga is GONE from `runtime.sagas`, and even if it weren't, the terminal catch-all `(Aborted, _) => (Aborted, vec![])` (`saga.rs:1268`) emits nothing. (4) The source never re-requests (`fan_out_crossing` `Occupied` = suppress no-op, `stub.rs:2578`). (5) `CrossingAborted` is `FlowDurabilityClass::ReDriven` (`intershard.rs:474`), whose invariant (`:281`) **requires a producer** — which a tombstoned saga is not. **So a lost `CrossingAborted` = permanent latch leak + permanent orchestrator amnesia = the entity never re-crosses.** This is exactly the D-6 #1 class of hole the project already closed for other flows via the durable outbox.

**DECIDED resolution — make `CrossingAborted` genuinely producer-less-durable (the project's existing D-6 #1 pattern), NOT a new ack protocol:**
- Emit the abort-reply via **`push_flow_durable(ctx.source, MsgClass::Saga, &CrossingAborted{..}, Durability::Retained)`** so the **R-6d durable outbox** mirrors + replays it across an orchestrator crash — the same mechanism `TransientBatch`/`GhostFlow::Despawn` already use.
- **Reclassify `CrossingAborted` from `ReDriven` to `ProducerLessReliable`** in `intershard.rs:471-474` (split it out of the shared arm). This is REQUIRED for correctness AND is self-enforcing: the `push_flow` `debug_assert!` (`runtime.rs:92`) will then **fail loud** if any site emits `CrossingAborted` on `Ephemeral` — converting the strand-hazard into a build/test failure, exactly the tripwire that classifier exists for.
- **Leave `CrossingRequest`/`TransientCrossingRequest`/`TransientCrossingGrant` as `ReDriven`** — they retain live producers (the source re-latches and the request is re-driven by the source's continued residence; the grant is re-driven by the source re-requesting; the source-side idempotency absorbs redelivery). Only `CrossingAborted` loses its producer at emit (the saga tombstones the same tick).

**This retires the draft's 3f-5(ii-B) "drop `request_ttl_ticks`" as the leak-closer** — the leak is closed by the durable outbox, not a TTL. `request_ttl_ticks` is still dropped, but as pure dead-code cleanup, NOT as the correctness mechanism (see 3f-6).

**Remaining read to confirm before coding:** verify the orchestrator's `OutboundBox` at the `drive_sagas`/`run_to_quiescence` level is wired to the **same** R-6d durable-outbox sink the shard/gateway bins use (memory task #108 "wire the durable outbox LIVE into shard/gateway boot" — confirm the ORCHESTRATOR bin got the same wiring, or the `Retained` marker is inert on the orchestrator). If the orchestrator outbox is NOT yet R-6d-backed, OQ-2's resolution is blocked on that wiring and the reclassification alone is insufficient. **This is the highest-priority code read.**

### OQ-3 (CONFIRM-ONLY, was C3) — `start_transfer`'s id-acceptance IS resolved; state it, test it, and fix the now-stale wire doc.
**Status: RESOLVED in the seam.** `start_transfer(ctx, gateway)` uses `ctx.transfer` as the sagas-map key verbatim (`saga_runtime.rs:1251/1263/1267`); nothing mints over it. So a caller setting `ctx.transfer = crossing_transfer_id(subject, subject_fence)` gets the exact-id guarantee. No re-derivation, no minting. **The plan does NOT need a new `start_crossing_transfer` entry to "accept" the id** — the id already rides `ctx.transfer` on the existing entry. The only thing the crossing needs beyond `start_transfer` is the `crossing_origin` tag; whether that justifies a new entry vs. a param is a design choice (3f-1). **Required: (a) state this explicitly, (b) add a unit test that a hand-set `ctx.transfer` becomes the map key, (c) FIX the now-stale wire doc at `intershard.rs:627` which says the orchestrator derives the id "(`start_transfer`)" — update it to name the actual crossing entry.**

---

## C. THREE LEDGER RESOLUTIONS (decided)

### (i) MED transient-latch symmetry → **ACCEPT-WITH-DOC (do not latch the transient path).**
The durable latch (`RequestInFlight`) satisfies a hard correctness invariant (exactly one durable saga per `(subject, fence)`, because a durable transfer mutates the authoritative `OwnerRecord` under a directory lock). The transient path has **no directory record, no per-entity lock, no CAS** (`locks_directory_key(Transient)=false`, `saga_runtime.rs:1301`). A re-spammed `TransientCrossingRequest` is idempotent at three verified layers: (1) wire `effect_class` keys `FencedKey{src_realm_fence}` (`intershard.rs:397`) → transport dedup; (2) the orchestrator re-grants the **same** deterministic `batch` id; (3) the source `on_transient_crossing_grant` no-ops a grant not matching `Held{outbound:None}` (`stub.rs:1604-1615`). Latching it would re-introduce the exact machinery HR2 says transients don't need, for zero correctness gain. **Action: a doc-comment on `fan_out_crossing`'s transient arm (`stub.rs:2581`) stating the asymmetry is deliberate and cross-referencing the three dedup layers. No code.**

### (ii) dead `request_ttl_ticks` → **DROP (pure dead-code removal), demoted from correctness-mechanism.**
Confirmed: declared `stub.rs:100`, set=`0` at `stub.rs:3545`/`shard.rs:100`/`topology.rs:1023`, **zero read sites**. The user LOCKED "PROPER fix, no bounded TTL" and the `CrossingAborted` wire doc (`intershard.rs:617-618`) says "never a bounded TTL window." Per OQ-2, the leak is closed by the **durable outbox**, not a TTL — so the field is genuinely dead and is removed as cleanup. **Action: delete the field + its 3 set-sites + update the `StubConfig` doc (`stub.rs:95-100`). Gate: land this AFTER OQ-2's durable-outbox resolution is confirmed (if OQ-2's read reveals the orchestrator outbox is NOT R-6d-backed and a fallback re-driver is needed, this field could be the vehicle — so do not delete until OQ-2 is closed).**

### (iii) LOW `on_saga_demote` exact-id symmetry → **DO NOT gate; keep the presence-keyed clear; DOC it. But VERIFY the stale-duplicate-Demote hazard first (the draft hand-waved it).**
The clear at `stub.rs:1568` is `if in_flight.0.remove(&entity).is_some()` — **presence-keyed, not id-keyed** (the draft called it "unconditional"; it is at-most-once-on-presence). The one-saga-per-entity directory invariant (`lock_transfer` serializes) makes an id-guard *usually* vacuous. **However, the adversary's M1 stale-duplicate concern is legitimate and must be checked before we commit to "leave it":** under at-least-once delivery, can a duplicate OLD `Demote` (for a superseded transfer) arrive at the source AFTER the entity has committed, been re-owned, re-crossed, and re-latched a NEW id — whereupon the presence-keyed clear wrongly frees the NEW latch? **Required read before finalizing (iii):** trace whether a re-crossed entity is even still SIMULATED at the old source (if authority moved to the dest, the old source no longer holds the dot, `self_fence_foreign_entity` no-ops, and there's no new latch at the old source to wrongly free — making it safe). If that holds, **Action: doc-only on `stub.rs:1565-1570` explaining the presence-key is safe because a committed Demote's entity has left this source (no re-cross window here, unlike the re-drivable abort path).** If it does NOT hold, gate the Demote clear on exact id too (accepting the false-arm coverage cost). **This is a bounded read, not a redesign — but it must happen; do not doc "safe" on faith.**

---

## D. ORDERED SUB-SLICES (seam-accurate; each compiles + gates independently unless bundled)

> **Bundling note (folds H2):** 3f-1a/1b/1c touch `PendingStart`/`LiveSaga`/`SagaSnapshot`/`process_starts`/`rehydrate` and **must land atomically** — adding a field to `PendingStart` (`:216`) breaks the destructure at `process_starts:1246`, and the `LiveSaga` insert (`:1267`) + `commit_result` snapshot (`:1116`) + `rehydrate` reconstruction (`:1191`) all reference the new field. Any partial edit fails to compile. Land 3f-1 as one commit.

### 3f-0 — PRECONDITION READS (no code; resolve OQ-1/OQ-2/OQ-3 + (iii)).
Confirm: (a) orchestrator `OutboundBox`→R-6d durable-outbox wiring (OQ-2, **highest priority**); (b) source loop `.values_mut()`→`.iter_mut()` borrow safety (OQ-1); (c) stale-duplicate-Demote residence trace (iii). **Gate the whole slice on OQ-2(a): if the orchestrator outbox is not durable, 3f cannot be made leak-free as designed and the plan needs a re-scope (a dest-side ack or a retained-record re-driver).**

### 3f-1 — origin marker + session plumbing + crossing start entry (ATOMIC; folds H1, H2, C3, OQ-1 orch-half, OQ-3).
**File:** `crates/node/src/saga_runtime.rs` (+ `sim/src/saga.rs` if `SagaCtx.session` needs no change — it doesn't; `session` already exists).
- **(a)** `PendingStart` (`:216`): add `crossing_origin: bool`.
- **(b)** `LiveSaga` (`:175`): add `crossing_origin: bool` with doc: "3f: set by the crossing start path; gates the terminal-abort `CrossingAborted` egress to `ctx.source`. **PERSISTED** — a parked crossing saga that aborts post-rehydrate must still reply (durably, via the R-6d outbox), else the source latch leaks."
- **(c)** `SagaSnapshot` (`:138`): append `crossing_origin: bool`. Write it in `commit_result`'s snapshot build (`:1116-1122`) from `live.crossing_origin`. **Thread it in `rehydrate`'s `LiveSaga` reconstruction at `:1191-1206`** as `crossing_origin: snapshot.crossing_origin` (H1: the draft omitted this site — it is a decoded field, no default possible; without this edit the code does not compile).
- **(d)** Crossing start entry. **Two viable shapes — the reader must pick one (OQ-3 clarifies the id already rides `ctx.transfer`, so this is only about the `crossing_origin` tag + `session`):**
  - **Option A (new entry, matches draft):** `pub fn start_crossing_transfer(&mut self, ctx: SagaCtx, gateway: NodeId)` pushing `PendingStart { ctx, gateway, crossing_origin: true }`. `start_transfer` (`:509`) sets `crossing_origin: false`.
  - **Option B (bool param, less surface):** `start_transfer(ctx, gateway, crossing_origin)` with one flag; three call sites updated.
  - **Recommendation: Option A** — it is DRY-honest (matches how `rehome` is a distinct start path) and keeps `start_transfer`'s existing 2-arg signature stable. Doc must state: "`ctx.transfer` MUST equal `crossing_transfer_id(ctx.subject, ctx.session_derived_fence)` — the id the source latched; the caller derives it (asserted by 3f-6)."
- **(e)** `process_starts` (`:1246`): update the destructure to `PendingStart { ctx, gateway, crossing_origin }` and thread `crossing_origin` into the `LiveSaga` insert (`:1267-1278`). The `rehome` insert (`saga_runtime.rs:~1614`) sets `crossing_origin: false`.
- **(f)** **FIX the stale wire doc (OQ-3c):** `intershard.rs:627` — change "the orchestrator (`start_transfer`)" to name the crossing entry.

**Tests (unit, `saga_runtime.rs` `#[cfg(test)]`):**
- `crossing_start_tags_the_live_saga`: hand-set `ctx.transfer = crossing_transfer_id(subject, F)`, `start_crossing_transfer`, run a tick, assert `runtime.sagas` is keyed on that exact id (OQ-3b — the id-acceptance proof) AND the live saga's `crossing_origin == true`. `start_transfer` tags `false` (both bool arms).
- `crossing_origin_survives_rehydrate`: persist a parked crossing saga (`crossing_origin=true`), `rehydrate`, assert it decodes `true` (extends an existing rehydrate test).

**Coverage:** both bool arms covered by the two starts; the `SagaSnapshot` field rides the existing persist/decode sites; `assert_eq!` on the id key (not `matches!`). No generic body.

### 3f-2 — durable `CrossingRequest` consumer (folds C2 orch-resolution, H3, decision-1).
**File:** `saga_runtime.rs`, new arm in the `match` at `:1924`, before `_ => {}` (`:1995`). Because the transient arm needs `from` (see 3f-3), **capture `from` in the outer arm** — change the `Inbound::Wire { from, class, bytes }` binding at `:1894` to carry `from` forward (currently only `(class, bytes)` survive the match; bind `from` into a var visible in the `for` body).

New monomorphic helper (branchless-shim discipline: ALL branching here, none in a generic):
```
fn handle_crossing_request(runtime, dir, outbox, req: CrossingRequest) {
    let transfer = crossing_transfer_id(req.subject, req.subject_fence); // id uses the WIRE fence (matches source latch)
    // 3-arm match (NOT let-else) so the reply-target branch is a covered region (HR5):
    match (dir.head(req.subject), dir.head(DirectoryKey::Realm(req.to_realm)), dir.head(DirectoryKey::Session(req.session))) {
        (Some(subj), Some(dest_rec), Some(sess_rec)) => {
            // start: ctx.source = subj.authority.node(); ctx.dest = dest_rec.authority.node();
            // ctx.expected_fence = subj.fence (HEAD fence = CAS expectation); ctx.transfer = transfer (WIRE fence id);
            // ctx.session = req.session; gateway = sess_rec.authority.node(); class = Durable; needs_provision = false;
            runtime.start_crossing_transfer(ctx, gateway); runtime.crossings_started += 1;
        }
        (Some(subj), _, _) => { // dest-realm OR session unresolved, but subject owner known → abort-reply so the latch clears
            outbox.push_flow_durable(subj.authority.node(), MsgClass::Saga,
                &InterShardFlow::CrossingAborted(CrossingAborted { subject: req.subject, transfer }),
                Durability::Retained); // OQ-2: producer-less, MUST be Retained
            runtime.crossing_unresolved += 1;
        }
        (None, _, _) => runtime.crossing_subject_gone += 1, // no owner to reply to — counted drop
    }
}
```
**Critical (folds ii-A):** the id derives from `req.subject_fence` (the WIRE fence = the source's latched fence), while `ctx.expected_fence` = `subj.fence` (the current HEAD fence for the CAS). These are equal in the happy path but the **id derivation MUST use the wire fence** so it matches the source latch even if the head advanced between latch and resolve.

**H3 folded — the None-branch thrash hazard:** replying `CrossingAborted` on a merely-*not-yet-propagated* dest-realm/session record can thrash (clear latch → re-cross → still-unresolved → re-abort). **Resolution:** the abort-reply on an unresolved dest is CORRECT for a permanently-absent realm but WRONG for a not-yet-resolved one. Since the orchestrator cannot distinguish them from a single `head` miss, **the abort-reply here must be the LAST resort, not the first.** Options for the reader to decide in 3f-2: **(a)** park the request for a bounded resolve-attempt budget (a small counted retry in `runtime`, determinism-safe) before replying abort — trades immediate thrash for a bounded park; **(b)** since a realm's owner is a durable directory record that is present-or-absent deterministically in the single-orchestrator in-process model (no propagation delay in-proc — the realm record either exists or doesn't), the immediate abort-reply is safe NOW and the thrash only becomes real under N-orchestrator partition (D-32). **Recommendation: (b) with a ledgered tripwire** — in-process, `head(Realm)` has no propagation race, so immediate abort is correct today; add a DEFERRED note that under D-32 partition this needs the bounded-park of (a). **This must be confirmed by reading whether realm records can be transiently absent in-proc (they cannot in the single-orchestrator model — the realm is registered at boot).**

**Counters** (mirror `liveness_notices`/`sends_shed` style, `saga_runtime.rs:1904`/`:1913`): `crossings_started`, `crossing_unresolved`, `crossing_subject_gone`.

**Tests (unit):**
- `crossing_request_starts_a_tagged_saga`: seed `Entity` subj@F/SOURCE + `Realm(to)`@DEST + `Session`@GATEWAY; feed the request; run two ticks; `assert_eq!` live-saga `transfer == crossing_transfer_id(subject, F)`, `source==SOURCE`, `dest==DEST`, `session==req.session`, `gateway==GATEWAY`, `crossing_origin==true`, `crossings_started==1`.
- `crossing_request_unresolved_dest_emits_durable_abort`: seed subj + session but NO `Realm(to)`; assert one `CrossingAborted{subject,transfer}` to SOURCE **pushed Retained** (assert via the outbox durability field), `crossing_unresolved==1`, no saga.
- `crossing_request_unresolved_session_emits_durable_abort`: seed subj + realm but NO `Session`; same abort-reply assertion (OQ-1: a missing session route can't start a durable saga).
- `crossing_request_vanished_subject_counted_drop`: no subj record; `crossing_subject_gone==1`, no outbox.
- `crossing_request_redelivery_is_one_saga`: feed twice; `live()==1`, `crossings_started==1` on the second (the `contains_key` guard at `:1251` + `lock_transfer` refusal at `:1263`).

**Coverage:** monomorphic helper; the 3-arm match covers all head-resolution outcomes; `assert_eq!` on id equality + the whole outbox tuple. No generic/`matches!`/`&&`.

### 3f-3 — transient `TransientCrossingRequest` consumer (folds M2).
**File:** `saga_runtime.rs`, sibling arm at `:1924`, using the captured `from`. Monomorphic helper:
```
fn handle_transient_crossing_request(dir, outbox, runtime, req, from: NodeId) {
    match dir.head(DirectoryKey::Realm(req.to_realm)) {
        Some(rec) => {
            let batch = crossing_transfer_id(req.subject, req.src_realm_fence);
            outbox.push_flow(from, MsgClass::Saga, &InterShardFlow::TransientCrossingGrant(
                TransientCrossingGrant { subject: req.subject, dest: rec.authority.node(),
                    to_realm: req.to_realm, dst_realm_fence: rec.fence, batch }));
            runtime.transient_crossings_granted += 1;
        }
        None => runtime.transient_dest_unresolved += 1, // decision (i): no reply — the source re-requests; idempotent
    }
}
```
- **Grant target = `from`** (the transport origin) — a transient has no directory `OwnerRecord` to look up (`from` is the only authoritative source address; the seam-map §4 identified this). The grant is `ReDriven` (the source re-requests) — plain `push_flow`, NOT `Retained`.
- **`batch` id** = `crossing_transfer_id(req.subject, req.src_realm_fence)` — per-subject, deterministic; a redelivery re-derives the same batch, absorbed by the source's `on_transient_crossing_grant` no-op.
- **M2 folded — `dst_realm_fence = rec.fence` staleness:** the grant carries the *current* realm-lease head fence; the batch commits LATER via the go-token. **Required note in the plan (and a code comment):** state what consumes `dst_realm_fence` downstream — the source's `TransientStatus::Crossing.dst_realm_fence` (`stub.rs:1609`), threaded into the batch payload's anchor. Verified earlier: the transient path takes NO directory CAS (`intershard.rs:567-569` — "NOT yet a stale-reject guard; the in-process saga-ordered path makes a stale handoff unrepresentable; a fence-rule-1 reject lands with the cross-host mesh transport, P3+"). **So a stale `dst_realm_fence` is TOLERATED in-proc today; ledger the cross-host stale-reject to P3+ (already ledgered in the wire doc — cross-reference it, no new work).**

**Tests (unit):**
- `transient_request_grants_resolved_dest`: seed `Realm(to)`@DEST/RF; feed `TransientCrossingRequest` from SOURCE; assert exactly one `TransientCrossingGrant{subject, dest:DEST, to_realm, dst_realm_fence:RF, batch}` to SOURCE (Ephemeral); `transient_crossings_granted==1`.
- `transient_request_unknown_realm_drops`: no `Realm(to)`; `transient_dest_unresolved==1`, no outbox.
- `transient_request_redelivery_regrants_same_batch`: feed twice; `assert_eq!` the two batch ids.

**Coverage:** monomorphic; 2-arm `Option` match; `assert_eq!` on the whole grant struct (`PartialEq`). No generic/`matches!`/`&&`.

### 3f-4 — the producer-DURABLE abort-reply egress (folds C1/OQ-2, H4).
**File:** `saga_runtime.rs` `ClearTransferLock` executor (`:981`) + thread `crossing_origin` into `run_to_quiescence`.
- Add a `crossing_origin: bool` param to `run_to_quiescence` (`:755`). **Three call sites (folds H2's compile-fanout):** `process_starts:1281` (pass `live.crossing_origin` — read it before the `run_to_quiescence` call), `deliver:1017` (pass the live saga's flag), `process_rehome_starts:1630` (pass literal `false`).
- In the `ClearTransferLock` arm:
```
SagaAction::ClearTransferLock => {
    let _ = dir.abort_clear(ctx.subject, ctx.transfer);
    if crossing_origin {
        // OQ-2: producer-less (the saga tombstones this same tick) → MUST be Retained so the R-6d
        // durable outbox replays it across an orchestrator crash (else the source latch leaks forever).
        outbox.push_flow_durable(ctx.source, MsgClass::Saga,
            &InterShardFlow::CrossingAborted(CrossingAborted { subject: ctx.subject, transfer: ctx.transfer }),
            Durability::Retained);
    }
}
```
- **Why here not `Tombstone`:** `Tombstone` also fires on the committed `Done` edge (`saga.rs:1245`); `ClearTransferLock` is abort-exclusive (`saga.rs:1436`, the only emit site). On COMMIT the latch clears via `on_saga_demote` (`stub.rs:1568`).
- **Reclassify `CrossingAborted` → `ProducerLessReliable`** (`intershard.rs:471-474`): split it out of the shared `ReDriven` arm into its own arm returning `ProducerLessReliable`. This makes the `push_flow` `debug_assert!` (`runtime.rs:92`) enforce the `Retained` marker at every emit site — the strand-hazard becomes a loud test failure. Update the `durability_class` conformance test + the arm doc.
- **H4c folded — does a crossing abort ALSO push to `rejected` (double-signal)?** Verified: the terminal-abort edge emits `[ClearTransferLock, Tombstone]` (`saga.rs:1437`) — it does **NOT** emit `NotifyRejected`. `NotifyRejected` is emitted only on the EARLY provision/prepare-reject edges (`saga.rs:1353/1375`), which for a crossing saga precede the `Aborting` state. **So a crossing that reaches `ClearTransferLock` does NOT also push to `rejected`** — no double-signal. BUT a crossing saga that aborts at the *prepare-reject* stage (`saga.rs:1375` `NotifyRejected` + later `ClearTransferLock`) WILL both push to `rejected` AND emit `CrossingAborted`. **Required: confirm this is acceptable** — `rejected` is the client-facing 1c channel (drained by CPO-4, `saga_runtime.rs:352-359`), `CrossingAborted` is the source-latch-clear; they serve different consumers and both firing is correct (the client learns the transfer failed; the source clears its latch). **Add a test asserting a prepare-rejected crossing emits BOTH, and that this is intended (they are orthogonal signals).**

**Tests (unit):**
- `crossing_abort_replies_durable_crossing_aborted`: start a crossing saga; drive to a pre-CAS abort (existing abort-test helpers, e.g. `TransferControlAck::Aborted` at `saga_runtime.rs:2754`); assert exactly one `CrossingAborted{subject,transfer}` to `ctx.source`, `transfer==crossing_transfer_id(subject,fence)`, **pushed Retained**.
- `non_crossing_abort_emits_no_crossing_aborted`: same abort via `start_transfer` (`crossing_origin=false`); assert NO `CrossingAborted` (the `if crossing_origin` FALSE-arm coverage).
- `crossing_commit_emits_no_crossing_aborted`: crossing saga to `Done`; assert no `CrossingAborted` (commit clears via Demote).
- `crossing_prepare_reject_emits_both_rejected_and_aborted` (H4c): assert both channels fire and document orthogonality.

**Coverage:** `if crossing_origin` both arms covered (true→emit / false→skip). The threaded param at 3 call sites: rehome→literal `false` (existing rehome tests), the two live paths→the flag (true by crossing tests, false by existing non-crossing transfer tests). No generic; `assert_eq!` on the decoded flow + durability.

### 3f-5 — ledger docs + dead-code drop (folds i/ii/iii; ordered AFTER OQ-2 confirmed).
- **(i)** doc-comment on `fan_out_crossing` transient arm (`stub.rs:2581`) — asymmetry deliberate, 3 dedup layers.
- **(ii)** drop `request_ttl_ticks` — field (`stub.rs:100`) + 3 set-sites (`stub.rs:3545`, `shard.rs:100`, `topology.rs:1023`) + `StubConfig` doc (`stub.rs:95-100`). **Only after OQ-2 confirms the durable outbox is the leak-closer.**
- **(iii)** doc on `on_saga_demote` presence-keyed clear (`stub.rs:1565-1570`) — **after the 3f-0 residence trace confirms safety**; if unsafe, gate on exact id instead.
- **H4a/H4b coverage tests (folds H4):** add (a) `on_saga_demote` with an ABSENT latch (the `.is_some()` FALSE arm — a non-crossing durable Demote) → covered no-op; (b) `on_transient_crossing_grant` redelivery hitting the `Some(_)|None` no-op arm. **These are SOURCE-side (`stub.rs`) tests for already-landed 3e code whose false arms 3f newly exercises end-to-end — add them here to keep HR5 green.**

### 3f-6 — capstone end-to-end test (the proof; folds ii-A id-contract).
**File:** new `crates/node/tests/crossing_trigger_e2e.rs` (dedicated node-level integration test wiring a source stub + orchestrator, mirroring existing transfer parity tests; in-mem io + virtual clock, no sockets). **Prefer a new file over extending `tests/src/lib.rs`** so the crossing surface is isolated.

**Durable happy path (dot crosses SOI → re-homes → label flips):**
1. Seed `Entity(dot)`@F/SOURCE, `Realm(planet_soi)`@RF/DEST, `Session(sid)`@GATEWAY.
2. Drive the SOURCE stub so `dot` crosses the `Authority` Shell into `planet_soi` → emits ONE `CrossingRequest{subject, from_realm, to_realm, subject_fence:F, session:sid}` and latches `L = crossing_transfer_id(subject, F)`. **Assert `L` is in `RequestInFlight`.**
3. Route to the orchestrator; step. **Assert** a `crossing_origin` saga started, `transfer==L`, `source==SOURCE`, `dest==DEST`, `session==sid`, `gateway==GATEWAY`. **ii-A assertion: `L == saga.transfer == crossing_transfer_id(subject, F)`.**
4. Drive to commit (feed FlushSource/Freeze/CAS acks via existing helpers). **Assert** the directory head for `Entity(dot)` flips to `Shard(DEST)`@F+1, and `Demote{transfer:L}` reaches SOURCE.
5. **Assert** SOURCE's `on_saga_demote` cleared the latch (`RequestInFlight` no longer holds `dot`; `crossing_latches_cleared` bumped). **ii-A: the id-contract closes — `L==saga.transfer==Demote.transfer` and the latch is gone.**

**Transient leg:** seed a `Transient` crossing into `planet_soi`; assert one `TransientCrossingGrant{dest:DEST, dst_realm_fence:RF, batch:crossing_transfer_id(subject, src_realm_fence)}` to SOURCE; drive `Held→Crossing`; assert `transient_grants_applied` bumped.

**Abort path (the 3f-4 durable-reply proof):** new crossing for `dot2` (latches `L2`); drive to a pre-CAS abort. **Assert** `CrossingAborted{subject, transfer:L2}` reaches SOURCE **and was pushed Retained** (durable-outbox mirror asserted if the harness exposes it), `on_crossing_aborted` clears `dot2`'s latch on exact-id match, `crossing_latches_cleared` bumped, and a fresh crossing re-latches (proving re-cross freedom).

**Crash-leg (the C1/OQ-2 proof — the highest-value new test the draft lacked):** start a crossing saga for `dot3`, park it (e.g. `Freezing`), **simulate an orchestrator kill-9 + rehydrate BEFORE the abort** (the harness kill-9 analog per memory task #86/#108), then drive the rehydrated saga to abort. **Assert** the `CrossingAborted` still reaches SOURCE (via the durable outbox replay) and `dot3`'s latch clears — proving the strand-hazard (OQ-2) is closed, not just for a live orchestrator but across a restart. **If 3f-0 reveals the orchestrator outbox is not R-6d-backed, this test will FAIL — which is the correct gate.**

**Coverage:** e2e exercises the full new surface (both request arms + durable abort egress + both clear paths + the crash replay). Unit tests in 3f-2/3/4 give the 100% region/branch on helpers in isolation; the e2e is the integration proof (HR5(c): integration tests exercise the full surface). All asserts `assert_eq!` on decoded flows / directory heads / stat counters, never `matches!`.

### 3f-7 — `just gate` (fmt + clippy -D + tests + coverage-fast).
Run `cargo llvm-cov clean --workspace` first (stale-profdata gotcha, per memory). Verify `git status` is clean of any concurrent review-agent mutations before gating.

---

## E. WHAT CHANGED FROM THE DRAFT (honest delta)

1. **OQ-1/C2 (session+gateway) is a real BLOCKER, not placeholder-able** — the draft's "option (a) placeholder is sound" is provably wrong (verified `PrepareSubscribe` routes to `gateway` at saga start). Resolution forces a **`CrossingRequest.session` wire append + source `.values_mut()`→`.iter_mut()` plumbing** — scope the draft entirely omitted.
2. **OQ-2/C1 (lost abort strands the entity) is a real BLOCKER** — verified the abort-reply is single-shot + the saga tombstones same-tick + `ReDriven` requires a producer that no longer exists. Resolution: **`push_flow_durable(...Retained)` + reclassify `CrossingAborted` to `ProducerLessReliable`**, leaning on the existing R-6d durable outbox (gated on 3f-0 confirming the orchestrator outbox is R-6d-backed). The draft's "at-least-once transport suffices" and "TTL-drop closes the leak" are both retired.
3. **H1** — the draft omitted the `rehydrate` reconstruction site (`:1191`); it is added (a compile requirement).
4. **H2** — 3f-1 is now explicitly ATOMIC (the `PendingStart` destructure at `:1246` + insert + snapshot + rehydrate must land together).
5. **H3** — the None-branch thrash is addressed (immediate abort safe in-proc where realm records don't propagate; ledgered park for D-32).
6. **H4** — added the missing false-arm coverage tests (`on_saga_demote` absent-latch, `on_transient_crossing_grant` redelivery) and verified the `rejected`-vs-`CrossingAborted` double-signal (orthogonal, both correct; tested).
7. **M2** — `dst_realm_fence` staleness is documented as in-proc-tolerated (no CAS; cross-host reject ledgered to P3+, already in the wire doc).
8. **Ledger (ii)** — `request_ttl_ticks` is 3 set-sites (draft's `tests/src/lib.rs:81` is wrong); dropped as cleanup, NOT as the leak-closer.
9. **OQ-3/C3** — id-acceptance IS resolved via `ctx.transfer`; a new `start_crossing_transfer` entry is for the `crossing_origin` tag, NOT to "accept" the id (the id already rides `ctx.transfer`). Added the explicit statement, a unit test, and the stale wire-doc fix (`intershard.rs:627`).
10. **UNKNOWN-3** (draft's `req.orchestrator_self`) — confirmed spurious; the None-branch reply target is `subject_rec.authority.node()`. Dropped.
11. **UNKNOWN-2** (transient batch coalescing) — correctly deferred; ledger a DEFERRED entry "transient crossing grants are per-subject; coalesce to one batch id per (to_realm, tick) at G-TIER."

**Bottom line:** the transient consumer (3f-3), the source side (3c–3e, already landed), the id math, and the mechanical hooks are all correct and buildable. The **durable consumer (3f-2) and the abort-reply (3f-4) are NOT safe to build until OQ-1 (wire append + source plumbing) and OQ-2 (orchestrator durable-outbox confirmation) are resolved by the 3f-0 reads.** OQ-2's read (is the orchestrator `OutboundBox` R-6d-backed?) is the single highest-priority gate — if it is not, the whole slice needs a re-scope.

Relevant files: `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/node/src/saga_runtime.rs`, `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/sim/src/saga.rs`, `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/wire/src/intershard.rs`, `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/sim/src/stub.rs`, `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/sim/src/runtime.rs`, `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/wire/src/seams/directory.rs`, `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/sim/src/directory.rs`.