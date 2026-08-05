# Arrival-race design — slices 8-9 (the destination-realm reap window)

Produced by a 9-agent design workflow, 2026-08-05. Ground phase read the source; three opposed
designs (prove-it-closed / inbound-ledger / extend-existing); independent judges; synthesis.

**CRITICAL CORRECTION THIS DESIGN ESTABLISHED:** the "slice 7 mid-saga reap shield" is NOT
implemented. The teardown check contains the clause, but nothing in the running system ever sets
that flag on a REALM (only on an entity), so the clause is permanently true and shields nothing.
The unit test asserting the clause is load-bearing hand-builds a state the real system cannot
produce. There is no slice 7 to implement — the dead clause is DELETED in step 9c.

---

sign slices 8-9: close the arriving-occupant race the existing in_transfer shield does not cover",
  "agentCount": 9,
  "logs": [],
  "result": {
    "recommendation": "# 1. IS THE RACE REAL — **YES**

Yes, and the thing the brief believed was already protecting it does not exist. The teardown check has a clause meaning "don't shut this realm down, something is being handed over" — but nothing in the running system ever sets that flag on a *realm*. The only code that sets it sets it on an individual player or object, never on a place. So that clause is always true and shields nothing; the unit test that "proves" it is load-bearing hand-builds a value the real system cannot produce. And even if it were set, it is cleared at the exact instant ownership flips — one step *before* the arrival happens — so it would cover the wrong half of the handoff anyway. What actually protects the destination today is something else entirely: a heartbeat the departing shard sends every tick saying "keep the place I'm sending this player to alive". Here is the story of how it fails. A player flies toward a station; the moment they cross the boundary the departing shard starts that heartbeat, and it keeps beating all the way through the handshake. Ownership flips. One tick later the departing shard is told to let go — and at that same instant it stops the heartbeat, because the heartbeat is tied to "a handover is in progress" and as far as it can tell the handover is finished. But the player is not there yet. The receiving realm is holding their data frozen, not yet running them, so it honestly reports itself as empty; nobody is asking for it; and the one remaining step — "you own them now, wake them up" — is a message that can be lost or refused and, unlike every earlier step, has no give-up path: it simply retries forever. Meanwhile the orchestrator's countdown runs: about four seconds of nobody asking, half a second of draining, then it kills the realm that the ownership record says is holding a live player. In the healthy case that final message lands in a fraction of a second, so the budget is roughly twenty-eight times larger than it needs to be — but **nothing anywhere states or checks that relationship**, and there are three cheaper doors in that do not need a stall at all: the departing shard losing its lease mid-crossing stops the heartbeat immediately; a batch of non-persistent things crossing over (dropped items, projectiles) never gets a heartbeat at all; and a cluster configured with the reaper switched on and the visibility system switched off runs the killer with the heartbeat entirely disabled. This is exactly the shape of the two prior failures in this subsystem: a silent, timing-held guarantee that everyone believed was structural.

# 2. THE RECOMMENDATION

Never shut a realm down while the orchestrator's own list of in-progress handovers says someone is on their way there. The orchestrator already keeps that list; it already writes it to disk so it survives a restart; and the shutdown decision already reads that list every tick for a different reason. So the fact and the thing that needs the fact are the same object, in the same process, on the same tick — no message to lose, no countdown to beat, no number to tune. Treat a pending arrival exactly like a request to keep the place alive rather than as a special veto, so that the destination's whole chain of parent realms is held up with it — otherwise the player lands on a planet whose star system was just shut down. Count the shield by name every time it is the *only* thing holding a realm up, so a wedged handover shows as a climbing number instead of a silence. Exclude the one kind of handover that has no destination at all (the recovery hand-off used when a machine dies, which parks indefinitely by design and carries a made-up destination) — including it would pin garbage forever on the first machine failure. And put a generous, derived upper limit on how long a single arrival may hold a realm up: not because the limit is the safety mechanism — membership in the handover list is — but so that a permanently stuck handover can never leak a realm forever. When the limit trips it lifts loudly, naming the realm and the stuck handover, and the system falls back to the recovery path it already has for orphaned players.

# 3. WHY NOT THE OTHERS

**prove-it-closed (null result + proof).** Its central claim — that the shield is continuous and the race is closed — is falsified by its own evidence: the heartbeat stops at the demote, one tick *before* the arrival, and the remaining coverage is a timing tail with no assertion behind it. Its secondary claim (that both halves have never run in one process) is also wrong — `crates/bins/tests/rlm_demand_login.rs::a_flying_occupant_re_homes_into_an_inner_planet_no_boundary_flap` already runs an armed reconciler with live `visual_demand` AoI and a real cross-in. **Grafted in:** its V3 *executed mutation control* (re-run the same fixture with the shield disarmed and require the reap to reappear — if that control does not turn red the slice is invalid), its land-the-failing-case-first discipline, and its arming cross-check (armed reaper + inert AoI is a real reachable configuration with the shield entirely off), which I promote from "candidate sixth slice" into 9c.

**inbound-ledger (arrival census projected into `desired_alive`).** Right instinct, over-built surface: a per-realm cell with counts and two anchors reintroduces state to keep consistent, and its derived constant is self-contradictory (`POST_COMMIT_ACK_STEPS = 4` against a list that yields seven shielding post-commit arms). **Grafted in — and this is the decisive borrow:** feeding the arrival into `desired_alive` as a third arm rather than into `teardown_ready` as an eighth clause, which buys ancestor-closure and descendant protection for free; the bounded age cap with loud expiry; the INERT-first slicing; and retiring the dead clause outright rather than leaving a comment correction.

**extend-existing (saga-derived veto set).** Taken as the base mechanism — it is the one with no threshold in the correctness path, it reuses a borrow and a schedule edge that already exist (`reconcile_realm_lifecycle` already holds `Res<SagaRuntimeRes>`, chained straight after `drive_sagas_core`), and its "count the veto by name" answer to the seven-silent-clauses problem is exactly right for this subsystem's history. **Rejected two things:** its no-cap position (see §4), and its unaudited projection — `rehome_ctx` (`crates/node/src/saga_runtime.rs:1954-1959`) fabricates `to_realm: RealmId::System(entity as u64)` under an in-source `// PLACEHOLDERS` comment, and those sagas park in `ReHoming` indefinitely by design, so a naive map over all live sagas permanently pins a garbage realm id per orphaned entity. **Also moved** the shield from teardown-veto to `desired_alive`, because a teardown-only veto leaves the destination's parent reapable.

# 4. THE LEAK QUESTION

**What is structurally impossible to leak:**

1. **The set is derived, never stored.** It is recomputed each sweep from `runtime.sagas`. There is no clearing path to forget, no eviction rule, no second source of truth. `commit_result` already does `runtime.sagas.remove(&transfer)` at every tombstone (`saga_runtime.rs:1303`) and deletes the durable snapshot in the same barrier — the shield disappears the same sweep, for free.
2. **Aborts un-shield by classification, not by code.** `Aborting`/`Aborted` classify as non-shielding, so a pre-commit abort (which sends the occupant back to the source) drops the shield immediately rather than after a timeout.
3. **Restart-safe without extra work.** `SagaSnapshot` persists the full `ctx` including `to_realm` (`saga_runtime.rs:191-199`), so the shield re-forms on the first post-restart sweep — while `rlm_quiesced_until = recovery_grace_ticks` already blocks all teardown across the boot window.
4. **The recovery hand-off is excluded by an explicit classifier**, not by a filter-on-provenance smell: the shield asks *"does this saga have a real arrival destination?"*, and a D-37 standing re-home moves ownership of a key between machines without changing realm membership at all — so it answers *no*. Exhaustive match, no wildcard, so a new saga shape does not compile until someone decides. (Best fix if it is cheap: stop fabricating and make the placeholder fields `Option`, which makes the exclusion structural rather than classified.)
5. **Ledger growth is bounded.** A shielded cell with no head cannot retire (retirement requires it to be outside the closure), so this is the one place the shield *can* hold ledger memory — bounded by the cap below, and pinned by a test.

**What is NOT structurally bounded, stated plainly:** a crossing saga wedged post-commit in `Promoting`/`Releasing` re-drives forever and never aborts (`deadline_for` gives those states only `redrive_deadline_ticks` and no abort path — correctly, since aborting post-commit loses the entity). **Nothing stops that pinning a realm alive forever.** So:

6. **A per-saga age cap.** Anchor it on a new `opened: UniverseTick` set at insert — **not** on `LiveSaga.since`, which `scan_deadlines` re-arms to `now` on every fire (`saga_runtime.rs:248-249`); anchoring there would refresh the shield forever, which is precisely the leak. `opened` is RAM-only, re-derived to `now` on rehydrate — the established precedent of `dead_observed_since` and `dest_adopted`. Derivation:

   `arrival_shield_ticks = max(abort_deadline_ticks + POST_COMMIT_STEPS × redrive_deadline_ticks, teardown_drain + teardown_cooldown + 1)`

   where `POST_COMMIT_STEPS` is the count of acked steps on **one** path from the commit CAS to `Done` — defined precisely and pinned by a test that recounts it from the classifier, so adding a state forces the constant to move. Expiry evaluated **per saga**, not per realm, so one wedged crossing on a busy realm cannot alarm every sweep forever.

   The cap is a **leak bound on a pathological input, not the correctness mechanism.** Correctness comes from set membership, which is zero ticks wide by construction. That distinction is the whole argument for why a timeout is acceptable here and was not acceptable as the primary shield.

7. **On expiry the fallback is a designed path, not data loss.** The realm becomes reapable; its node is then latched dead; the head is force-reaped; the D-37 standing re-home is exactly the machinery for an entity whose owner record names a dead node; and the source's crossing request re-fires into the unresolved arm until the demand loop re-spins the realm. Loud at every step.

# 5. SLICE PLAN

Two slices, three steps each. Every step is independently verifiable; only 9a wants a live run.

### Slice 8 — build the oracle and land the failing case (INERT: zero behaviour change)

**8a — the reproduction, RED-then-asserted.** In `tests/tests/` (new file), drive a real durable crossing to the CAS, deliver the `Demote` (clearing the source latch and the last keep-alive), **withhold the `Promote`**, advance `demand_ttl + drain + 1` ticks, and assert a `Kill` **is** emitted for the destination realm today.
*Acceptance:* `a_stalled_promote_lets_the_dest_realm_be_reaped_today` passes, with a comment naming 9a as the flip. Also assert `dir.head(DirectoryKey::Realm(dest)).in_transfer == None` on **every** tick of the saga — the executable retirement of the "slice 7 is already implemented" belief. **Test-provable.** This is the highest-value artifact in the arc: the race stops being a reading argument.

**8b — the classifier + projection (`vd-node`, EXECUTE side).** `LiveSaga.opened`; an exhaustive no-wildcard `arrival_dest(&LiveSaga) -> Option<RealmId>` returning `None` for `Aborting`/`Aborted`/`Done` **and for the re-home shape**; `SagaRuntimeRes::arriving_dest_realms() -> BTreeSet<RealmId>`. Nothing consumes it.
*Acceptance:* full-variant classification table (covers `Done`/`Aborted`, unreachable via integration since they tombstone first); `arriving_dest_realms_excludes_a_parked_rehome_saga` — the leak that would have shipped; `..._includes_a_transient_batch_dest` — the gap the keep-alive never covered; `..._is_empty_after_tombstone`; `rehydrate_restores_the_arrival_shield_set`. **Test-provable.**

**8c — kernel plumbing, records only (`vd-sim`, DECIDE side).** `reconcile` takes `arriving: &BTreeSet<RealmId>` as *data*; computes `shielded` and, where it would be load-bearing, records `delta.arrival_shielded` — **and still emits the identical `Kill`**. Runtime counters (`arrival_shield_vetoes` monotone, `arrival_shield_gauge`) into the JSON admin `RlmView`.
*Acceptance:* `arrival_shield_is_recorded_but_inert_in_slice_8` — the realm from 8a is named in the delta and still killed; `reconcile_is_run_twice_identical_with_an_arriving_set`; the existing byte-identity gates unchanged. **Test-provable.**

### Slice 9 — flip it, bound it, retire the dead clause

**9a — consume: the third arm of `desired_alive`.** `desired_alive(cell, now, tuning, running_live, arrival_pending)`, so the destination's whole parent chain is held by `ancestor_close` and its ancestors by `has_desired_descendant`. Continue recording `arrival_shielded` **only when the arrival is the sole reason** the realm is desired, so the counter's healthy baseline is zero.
*Acceptance:* invert 8a → `a_stalled_promote_shields_the_dest_realm_from_the_reap`; the arrival keeps the destination's **parent** in the closure; `an_aborted_crossing_releases_the_shield_within_one_sweep`; `a_transient_batch_dest_realm_is_shielded_too`; and **the mutation control** — same fixture, same seed, shield disarmed, the reap **must** reappear (if it does not, the slice is invalid and must not land). **Needs ONE in-game run** on the demand cluster: the regression risk of 9a is realms living *longer*, so fly the Planet→System→Planet round trip and confirm (i) teardown-behind still fires and (ii) `arrival_shield_vetoes` reads **zero** after a healthy crossing — a non-zero value in-game would mean destinations routinely reach full teardown-readiness mid-crossing, i.e. the race is not latent at all, which is a bigger finding than the fix.

**9b — the leak bound.** `arrival_shield_ticks` derived as in §4.6, into `RlmTuning` with a `validate` arm and the existing `resolve_rlm_tuning` threading; per-saga expiry; `arrivals_shield_expired` counter plus a `tracing::error!` naming realm, transfer, state and age.
*Acceptance:* the derivation test recounting `POST_COMMIT_STEPS` from the classifier; `arrival_shield_ticks > drain + cooldown`; a pending arrival older than the cap does **not** shield and increments the counter; **one wedged saga on a busy realm alarms once, not every sweep**; a shielded headless cell retires once the cap lifts (no ledger growth); default stays `0` / INERT. **Test-provable.**

**9c — retire the dead clause + close the config gap.** Delete `TeardownFacts::in_transfer` and `& !facts.in_transfer`; update `teardown_ready_requires_every_clause` to the surviving clauses; add a `DEFERRED.md` entry stating that a realm-key transfer lock has no producer until realm mobility lands (P8/P10) and must return **with** its producer and an end-to-end test — noting that the new shield already subsumes its intent, since a realm-subject saga is a live saga. Plus the boot cross-check: refuse to boot with the reconciler armed and the shards' AoI inert.
*Acceptance:* a pinning test that `locks_directory_key(Transient) == false` and that the only realm-keyed saga context is the transient batch; `the_arrival_shield_is_armed_exactly_when_the_reconciler_is`. **Test-provable.**

**Rule posture:** no wire change (`PROTO_MINOR` stays 7), no `InterShardFlow` arm, no store-format change, no shard gains a capability, no match on shard kind (the one match is on the transfer FSM's own shape), `BTreeSet`/`BTreeMap` throughout, no generics added, DECIDE/EXECUTE preserved, INERT-then-flip. **One honest correction to the compliance story:** routing the arrival through `desired_alive` means it can now *drive a `SpinUp`*, not merely withhold a `Kill` — that is a real action, it goes through the existing fenced minting path unchanged, and it should be argued explicitly rather than hidden behind "the shield authorises nothing".

**Rough size:** ~120 production lines across `vd-sim`, `vd-node`, `vd-bins`; ~420 test lines.

# 6. OPEN QUESTIONS FOR THE USER

1. **When an arrival's shield times out, which failure do you want?** (a) *Recommended:* lift the shield loudly and let the realm be reaped — the player's ownership record then names a dead node and the existing orphan-recovery machinery takes over. (b) Pin the realm forever and alarm — never risks a player, but leaks one live realm per wedged handover, permanently, in a system meant to scale to 100k. I have designed for (a) with a generous derived limit; (b) is a one-line change if you'd rather never risk the player.

2. **Should a pending arrival be allowed to *spin a realm back up*, or only to stop it being shut down?** Feeding the arrival into the desired set gives the parent-chain protection you need (otherwise the player lands on a planet whose system just died), but it also means an arrival at a realm whose head has vanished will start a new one. That is the correct self-heal, and it is also a pod launch triggered by a handover. I recommend allowing it; say if you want desire-without-spawn.
---

# USER DECISIONS — 2026-08-05 (both open questions answered; treat as settled)

**Q1: when an arrival's shield times out — which failure?**
**DECIDED: option (a).** Lift the shield LOUDLY and let the realm be reaped. The player's ownership
record then names a dead node and the existing orphan-recovery machinery takes over. Rationale: a
permanent per-wedged-handover realm leak is unacceptable in a system targeting 100k. Implement the
generous DERIVED limit (§4.6 / step 9b) with the `arrivals_shield_expired` counter and the
`tracing::error!` naming realm, transfer, state and age. The limit is NOT the safety mechanism —
membership in the live-handover set is — it exists solely so a permanently stuck handover cannot
leak a realm forever.

**Q2: may a pending arrival SPIN A REALM BACK UP, or only prevent shutdown?**
**DECIDED: allowed to spin up.** Feed the arrival into `desired_alive` as the third arm (step 9a),
which is what buys parent-chain protection via `ancestor_close` — without it a player lands on a
planet whose star system was just reaped. Accepted with eyes open: this means an arrival at a realm
whose head has vanished will START a new one. That is the correct self-heal AND it is a real action
(a fenced pod launch triggered by a handover), not a passive veto. The design's own compliance note
stands: argue this explicitly in the commit, do not hide it behind "the shield authorises nothing".

Both answers match the design's recommendations; nothing in the plan changes.
