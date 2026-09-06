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

### D-31 🟧 HR2's `TransferableKind` trait (the per-kind serialize/spawn/precondition/rebind_refs seam) is unbuilt — only the static `KindDef` half landed
- **LANDED (the static half):** `core/src/entity_kind.rs` has `EntityKind` + `KindDef` (DurabilityClass/GhostPolicy/
  ContinuityModel/LossBudget/blob_schema) + the registry data, fully tested. This is HR2's data half.
- **MISSING (the behavior half):** the `TransferableKind` TRAIT itself — the per-kind `serialize` / `spawn` /
  `precondition` / `rebind_refs` methods (+ the `register!` macro + the `kind_blob_evolution` writer-N+1/reader-N CI
  gate) — does NOT exist. `grep -rn TransferableKind crates/` returns only doc-comments. Consequently the transfer
  envelope carries an OPAQUE `state: Vec<u8>` (`TransitionPayload::StubCrossing.state` / `InitialSpawn.state` /
  `TransientItem.state` in `wire/src/intershard.rs`) that no registry trait produces or consumes — and in the live
  path it is produced as `vec![]` (the pose crosses as a TYPED field on the crossing, not through the trait). There is
  no kind-generic serialize, no spawn-from-tags reconstruction, no `precondition` spatial gate, and crucially no
  `rebind_refs` (the method that re-homes a ship's `ChildOf` passengers and a block's frame anchor at the dest).
  The envelope's sibling `schema_version: u16` field (`wire/src/intershard.rs` `TransferEnvelope`) is likewise
  stamped-but-unvalidated at the receiver today — the per-kind version-FLOOR check (writer-N+1/reader-N, the
  `kind_blob_evolution` gate) is part of THIS seam and lands with the TLV blob (1d.6). By contrast the
  `universe_epoch` sibling is an EXACT-match fail-safe that is NOT deferred: it is validated NOW at BOTH
  pose-placing ingresses — the crossing (`sim/src/stub.rs` `on_transfer_envelope` → `crossings_epoch_mismatch`)
  AND the D-37 forward re-home adopt (`sim/src/stub.rs` `on_re_home` → `re_home_epoch_mismatch`; `ReHomeCmd`
  gained a `universe_epoch` field, stamped by `build_rehome`) — so transfer_protocol §3.3 ("no entity placed at a
  stale celestial position") is UNIFORM across every pose-placing leg (audits `wf_032b80eb` + `wf_2c963246` H1).
  **Cold-start ordering caveat (audit `wf_2c963246`, MEDIUM):** the stub schedule does NOT explicitly order
  `observe_clock_syncs` (the follower) `.before` the inbound dispatch (unlike the orchestrator's explicit `.chain` in
  `orchestrator.rs`), so a crossing/re-home processed on a tick before the FIRST `ClockSync` (epoch still default
  `EpochId(0)`) would be FALSE-rejected against a valid in-epoch leg. Self-healing (the saga re-drives at-least-once
  and is accepted once the epoch syncs) + practically unreachable (a Transfer/ReHome arrives many ticks after
  login→grants→sync), so MEDIUM — but make the ordering explicit (`.before(process_inbound)`, matching the
  orchestrator) when the stub schedule is next touched, so the fail-closed gate's epoch-known precondition is
  structural rather than insertion-order luck.
- **WHY this is an INTERIM, not a defect (the deferral is sound):** the trait's methods have NOTHING to do yet. Only
  single-dot Players cross today, and a Player's entire transferable state is its pose — already carried as a TYPED
  `pose` field on the `StubCrossing` (never the opaque blob). `rebind_refs` has no refs to rebind until a COMPOUND kind
  exists (a ship with `ChildOf` passengers — P8; a dropped block with a frame anchor — P6). `serialize`/`spawn` have no
  per-kind state beyond the pose until the TLV blob format exists (1d.6). Building the full trait NOW would be
  speculative scaffolding (the smallest-correct discipline forbids it). The retrofit is purely ADDITIVE: the
  `state: Vec<u8>` field IS the exact insertion point — the blob will flow THROUGH the trait with zero envelope change.
- **WHAT the audit actually caught (the honesty hole this entry closes):** `PLAN.md` schedules `TransferableKind` at
  **P0**, yet it was absent from BOTH the code AND this ledger — invisible to the "a phase isn't done until its
  DEFERRED entries flip green" gate. The DEFECT was the UNLEDGERED absence, not the deferral. This entry is the fix.
- **When / proper (the trait lands incrementally with its first real consumer):**
  - `serialize` + `spawn` (typed `Blob` assoc-type, total over `EntityKind::ALL`) + the `register!` macro + the
    `kind_blob_evolution` CI gate — with the **TLV blob (Slice 1d.6)**, so `StubCrossing.state` flows through the trait.
  - `rebind_refs` — with the first COMPOUND kind: `DroppedBlock`'s frame anchor at **P6**, the ship+`ChildOf` bundle at **P8**.
  - `precondition` — with the typed spatial-precondition gate (`PrepareResult::Rejected{spatial_reason}`, the hull-trap
    class) when real placement constraints exist (**P5/P8**).
- **Pin (exists-to-be-flipped):** `core/src/entity_kind.rs` module doc names the missing trait + this entry; the
  `state: Vec<u8>` field docs in `wire/src/intershard.rs` name 1d.6 as where it stops being opaque. Flips 🟩 when a
  NON-Player kind crosses end-to-end via the registry trait (not a special-cased dot) under the `kind_blob_evolution` gate.
- **Dependency:** the TLV blob (1d.6) for serialize/spawn; a compound kind (P6 block / P8 ship) for rebind_refs.
- **Source:** the full-architecture audit `wf_65b95cbf` (Finding B, the one HR-load-bearing P0 deliverable absent from
  both code and ledger).

### D-32 🟥 Directory partition-by-region: the in-process DIRECT CAS + flat `config.orchestrator` addressing is the cross-region / N-orchestrator scaling blocker (unledgered until now)
- **CURRENT (correct for P2):** ONE orchestrator owns the WHOLE keyspace; the commit CAS is a DIRECT in-process call
  (`saga_runtime.rs` `IssueCommitCas` → `DirectoryCore::commit_cas(ctx.subject, …)` on the orchestrator's own
  single-threaded schedule — documented at `saga_runtime.rs:10-11`). Every directory-op / saga-ack send site addresses
  the single flat `config.orchestrator: NodeId` UNCONDITIONALLY, regardless of which `DirectoryKey`/`RealmId` it
  concerns (`stub.rs` ~375/392/408/608/980/1124; `gateway.rs` ~87 + its push sites). There is NO `RegionId` /
  `coordinator_of` / `resolve_coordinator` symbol anywhere — a cross-region transfer (subject key on coordinator A,
  dest realm on coordinator B) has **no representation**: the DIRECT CAS is correct ONLY while one orchestrator owns
  everything.
- **WHY this is correct now, not a defect:** at P2 there is exactly one orchestrator; partitioning the directory by
  region is the binding THOUSANDS-scale seam (`transfer_protocol.md:283`) that should NOT be built yet (premature —
  the partition map, cross-coordinator routing, and 2-coordinator commit have no consumer until N-orchestrator scaling).
  The DEFECT the audit caught is that the in-process DIRECT CAS was **invisible to the green-gate** — `transfer_protocol.md`
  flags the seam but no DEFERRED entry named `commit_cas`'s in-process locality as the cross-region blocker (SCALE-CLOUD-1
  under [[D-24]] is only about the admin-snapshot rebuild cost, NOT CAS locality). This entry closes that honesty hole.
- **When / proper (additive — code at N-orchestrator scaling, NOT before):** (a) introduce ONE
  `fn coordinator_of(key: &DirectoryKey) -> NodeId` resolver (returns `config.orchestrator` today) called at every
  directory-op / saga-ack send site, so the future constant→key-lookup swap is ONE function, never a 13-site
  scatter-rewrite (the scatter class that killed the old project); (b) the saga must ROUTE a `CommitCas` op to the
  subject key's coordinator (not call `commit_cas` locally) once keys span coordinators — a cross-coordinator commit
  (still ONE logical CAS, the directory partition owner is the single writer per region). Until then `config.orchestrator`
  is the degenerate single-region case.
- **Pin (exists-to-be-flipped):** `saga_runtime.rs` `IssueCommitCas` executor doc-comment names the in-process DIRECT
  CAS + this entry. Flips 🟩 when the `coordinator_of` resolver + cross-coordinator commit routing land and a transfer
  whose subject + dest span two coordinators commits correctly.
- **Dependency:** N-orchestrator horizontal scaling (post-P3; the gateway adoption/ResumeTicket scaling is the sibling).
- **Source:** the full-architecture audit `wf_7f83e6b3` (F1).

### D-33 🟥 The Directory CAS is single-key only — no atomic N+1-key (ship + `ChildOf` passengers) commit primitive (the directory-side gap, distinct from D-31's blob-side `rebind_refs`)
- **CURRENT (correct for P2):** `directory.rs` `commit_cas` is strictly SINGLE-key — one `key: DirectoryKey`, one
  `records.get_mut(&key)`, one `Fence::cas_next`, flips ONE authority. `SagaCtx` carries exactly one `subject:
  DirectoryKey` + one `expected_fence` (it is `Copy`, fixed at creation); `commit_action` issues exactly one
  `IssueCommitCas`. No `ChildOf` / `bundle` / `reparent` / `Vec<DirectoryKey>` concept exists in the directory/saga/runtime.
- **WHY this is correct now, not a defect:** only single, childless entities (Player dots) cross at P2 — a single-key CAS
  is exactly right. The compound shape is owed only when ships exist (P8).
- **The design requires the opposite shape at P8 (`transfer_protocol.md` §8 + :283):** the "walk inside a flying ship"
  compound handoff must bump the `Ship(ShipId)` key AND re-parent every `ChildOf(ShipId)` `OwnerRecord` in ONE atomic
  transaction — "one saga, one commit, N+1 keys" — so a passenger's authority can NEVER flip on a different tick than
  its hull's during a host/dock crossing (the §8.1 passenger-split-brain-relative-to-hull class).
- **The ledger gap this closes:** [[D-31]] tracks `rebind_refs` as the per-kind BLOB method (re-homes a passenger's
  child/frame refs at the dest) — a DISTINCT change from the directory-side ATOMIC MULTI-KEY COMMIT primitive on the
  commit seam. The blob re-home and the multi-key CAS are two different owed pieces; D-31 named only the first.
- **When / proper (additive at P8, records never reshape):** grow the primitive to
  `commit_cas_bundle(primary: DirectoryKey, slaved: &[DirectoryKey], expected, new_owner)` that CASes the ship key and
  re-parents the `ChildOf` records under ONE lock; the single-key `commit_cas` stays as the degenerate N=0 case (HR3 —
  ONE machinery, policy fan-out). `SagaCtx` grows an optional slaved-key set.
- **Pin (exists-to-be-flipped):** `directory.rs` `commit_cas` doc names the single-key limit + this entry. Flips 🟩
  when a ship + ≥1 `ChildOf` passenger transfer commits atomically (both authorities flip on the same fence/tick).
- **Dependency:** P8 ships (the first compound kind); composes with [[D-31]] `rebind_refs` (blob side) for the full bundle.
- **Source:** the full-architecture audit `wf_7f83e6b3` (F2).

### D-34 🟥 Gateway has NO per-session home/authority field — login landing + `Bye` detach both hardwire `config.shard` (a multi-shard session leak + an unledgered inversion of the orchestrator-owned Spawn Resolver)
- **CURRENT (correct for single-login-shard P2, WRONG for N-shard):** the `Session` struct carries `subs:
  BTreeMap<NodeId, SubRecord>` (the shards it is *actually* subscribed to, EMPTY at creation — `gateway.rs` ~795 —
  filled only by `open_sub` on SubscriptionReady) but NO per-session *home/authority* field. Every CONTROL-plane
  decision reads the global flat `config.shard` scalar: route birth (~798), login sub (~1502), `AttachSession` on
  grant (~1675) + retry (~1724), and the `Bye` `DetachSession` (~850). In single-login-shard P2 these coincide; in any
  N-shard cluster they diverge.
- **A1 — the `Bye` detach leak (the present-but-masked half):** on `Bye` the gateway sends `DetachSession` to
  `config.shard` UNCONDITIONALLY. After even ONE transfer (player now homed on dest B, source A's sub closed — the
  1d.2/1d.3 path), this detaches A and never tells B to free its saga-attested `SessionTable` entry → B leaks until
  lease-TTL reap. TTL enforcement now LANDED ([[D-3]] — the reaper reaps the lapsed directory `Session` lease), but the
  reaper clears the DIRECTORY record, not B's LOCAL saga-attested `SessionTable` entry, so that local entry still
  leaks until B's own lease-recheck observes the loss — this item's concern remains. This is
  the INVERSE of the `"any sub != config.shard"` anti-pattern the abort path explicitly forbids (`gateway.rs` ~987-991).
  MASKED today only because no test does Bye-AFTER-transfer and there is one login shard. **NOTE the cheap "iterate
  `session.subs.keys()`" fix is WRONG:** `subs` is empty at login, so it would regress the basic login→Bye detach —
  the fix genuinely needs the authority field below, not a sub-scan.
- **A2 — the login landing inversion (the structural half):** the binding spec assigns the landing decision to the
  ORCHESTRATOR — `integration.json` res #15 ("One owner: orchestrator … Spawn Resolver") + `connection_plane.md:190`
  invariant B1 ("a stateless Spawn Resolver computes the initial InterestSet from the player's durable checkpoint
  position"). The read plane (SubTable/`known_shards`) was split for N-shard fan-out; the *where-does-a-session-land*
  axis was not. With one scalar, two players checkpointed on different planets via one gateway both attach to one shard.
- **When / proper (ONE additive field closes BOTH, at first-multi-shard / P3):** add a per-session
  `home_shard`/`authority: NodeId` field on `Session` that initializes from `config.shard` today but is SET by an
  orchestrator reply (the stateless Spawn Resolver / Directory head at login) and UPDATED on every transfer commit;
  route attach / detach / initial-route off THAT field. Then the P3 swap is one populate-the-field change, never a
  call-site scatter (the scatter class that killed the old project). The `Bye` detach then targets the session's true
  current authority (+ any composited subs), never `config.shard`.
- **Pin (exists-to-be-flipped):** `gateway.rs` `Bye` `DetachSession` site + the session-creation site name the
  `config.shard` assumption + this entry. Flips 🟩 when a session that transferred A→B is `Bye`'d and B frees its
  `SessionTable` (a multi-shard Bye test: login A → transfer to B → Bye → assert B detached), AND login lands a session
  on an orchestrator-computed home shard.
- **Dependency:** P3 first-multi-shard cluster (the orchestrator Spawn Resolver round-trip); composes with the
  [[D-32]] `coordinator_of` partition seam.
- **Source:** the full-architecture audit `wf_b82d1a67` (A1 + A2, one root cause).

### D-35 🟥 Client ship-interior compositing is TIME-coherent but NOT version-matched — the §8.1 "buffer the newer stream until both at a matching version" obligation is unbuilt (and the `world_pos` doc over-claims it is free)
- **CURRENT (correct for P2 single-sub, WRONG for P8 ship-on-its-own-shard):** `view.rs::world_pos` composes a
  `ShipLocal` interior through its hull as `hull.pos + hull.orient * interior.pos`, sampling the hull at the SAME
  `cursor` (time-coherence). Benign today — interior + hull are one entity-set on one sub at one `frame_id`, trivially
  same-version. But when the ship rides its OWN shard (P8), hull and interior arrive on DIFFERENT lossy datagram subs.
- **MISSING (the binding §8.1 obligation — `transfer_protocol.md:382` + `sealed_shards.md:130-132` Finding #9):**
  VERSION-MATCHED buffering — hold the newer stream until both are at a matching `(source_tick, version)`, or the
  client renders "new hull pose + stale anchor" for a tick = the passenger-jumps-relative-to-hull bug §8.1 exists to
  kill (the precise AAA-seamless defect). The wire carries NO correlation field: `EntitySnap` is `{entity, pose}`
  (`channels.rs` ~173, no `parent_version`); `SnapshotDatagram` has only a per-sub `frame_id`.
- **The doc over-claim corrected by this entry:** `view.rs` ~210-214 claimed the composition refinement "lands here
  WITHOUT reshaping call sites" — FALSE: honoring §8.1 needs a NEW wire field (`parent_version` on `EntitySnap`/
  `SnapshotDatagram`) AND a hold-buffer in the view, both reshaping the snapshot decode path. The doc is corrected to
  say the wire reshape is REQUIRED, not free, so a P8 implementer does not under-budget it.
- **When / proper (P8 — ships on their own shard + the multi-sub composited client):** (a) a `parent_version` /
  matching `(source_tick, version)` field on `EntitySnap`/`SnapshotDatagram`; (b) a hold-buffer in `world_pos`
  deferring compositing until hull + interior match version, asserting
  `freeze_gap + InputGap + stagger_skew + parent_match_hold < interp_buffer_headroom` (`sealed_shards.md:132`). The
  seam SHAPE is right (one `world_pos` chokepoint, one hull level — §8.2) — only the version gate + wire field are owed.
- **Pin (exists-to-be-flipped):** `view.rs` `world_pos` doc names §8.1 + this entry. Flips 🟩 when a hull and its
  interior delivered on two lossy subs composite without a passenger jump under hull-datagram loss (a harness drop test).
- **Dependency:** P8 ships; composes with [[D-31]] `rebind_refs` + [[D-33]] atomic multi-key CAS (the server side of the same compound handoff).
- **Source:** the full-architecture audit `wf_b82d1a67` (Theme B).

### D-36 🟥 DestDelivered-starvation stall: the saga timeout producer cures lost saga-ACKS, not a starved delivery watermark
- **Missing:** a `Promoting{promote_acked: true}` saga whose DEST frame-delivery PATH is blocked (the gateway's
  standing `every_observer_delivered` watermark stays false) stays half-open FOREVER — the Slice-2a producer re-drives
  the `Promote` (re-acking `PromoteAck`, `promotes_redelivered` climbs) but CANNOT re-arm `DeliveredToObservers`,
  which `recompute_delivery_watermarks` derives per-tick from the observer frame set, INDEPENDENT of the saga
  round-trip. So the producer's "R1 cured" claim is honestly NARROWED to lost SAGA-ACKS.
- **Where:** `crates/connection-plane/src/gateway.rs` `recompute_delivery_watermarks`; the saga `Promoting` gate
  (`saga.rs` `promoting_advance`). The stall is admin-visible (climbing `now - since`), NEVER silently wedged.
- **When / proper:** **P3** — a re-poke of the delivery path / rising-edge watermark (the producer drives saga-ack
  recovery; a starved delivery watermark needs a delivery-path nudge, a distinct mechanism).
- **✅ Forward note RESOLVED (Slice 2d, was holistic audit `wf_dd38151d`):** the Slice-2a producer's `Promote`
  re-drive presumes `ctx.dest` is LIVE; after a D-37 CELL-2 re-home it would aim at the now-dead original dest. Slice
  2d carries `rehome_target` in `Promoting`, so a re-homed `Promoting` Timeout re-drives `A::ReHomeAdopt → target` (the
  live re-home target), never `Promote → ctx.dest`. The starved-DELIVERY-watermark wedge above is the DISTINCT
  remaining D-36 item (the producer re-drives the saga-ACK; a starved watermark still needs a delivery-path nudge).
- **Source:** Slice-2a design `wf_9f22c70d` (finding #3); boundary-documented, not silently wedged.

### D-37 🟧 Permanent participant-kill recovery: ENTITY forward re-home LANDED (CELL 1 + CELL 2); CELL 3 standing re-home + Realm/Ship re-home owed (design wf_6efc70f1)
- **✅ LANDED (Slices 0/1/2a/2b/2c/2d, all gate-green 100% Tier-A region+branch; gated on D-3 + D-6, both landed):**
  the FENCED FORWARD RE-HOME machinery — when a transfer PARTICIPANT is permanently KILLED mid-flight, the
  orchestrator confirms it dead (D-3 `is_confirmed_dead`) and recovers the ENTITY to a LIVE owner instead of
  re-driving toward a corpse forever. The fence-monotone invariant is load-bearing: every re-home commits at
  `fence+1` through the ONE commit point (`commit_cas`), so a resurrected dead owner is strictly stale.
  - **Slice 0 (f49cdda) CELL 1 — kill SOURCE in Demoting (post-commit):** the dest is the already-committed owner;
    `(Demoting, SourceUnreachable) → Promoting` SELF-PROMOTES it (`scan_deadlines` injects SourceUnreachable once
    the source is confirmed dead). End state `EntityRecoveredRealmOrphaned{entity_at: DEST}` (was `ParkedHalfOpen`).
  - **Slice 1 (8a95899):** `ShardProfile::satisfies` (capability match, HR3) + `select_rehome_target` (lowest LIVE
    capable shard, deterministic BTreeMap tie-break).
  - **Slice 2a (f71cafb):** the DEDICATED `InterShardFlow::ReHome(ReHomeCmd)` arm + `ReHomeState::PoseOnly` payload
    + `RE_HOME_STEP` + effect_class (user decision: a proper new arm, NOT a `Promote` reuse).
  - **Slice 2b (be0170a):** the re-home FSM (`S::ReHoming` + `ReHomeTo` event + `ReHomeCommit`/`ReHomeAdopt` actions
    + arms; `CasLost` → clean Aborted no-op) + the executor (commit_cas re-pointed at the target; emit the ReHome
    adopt from the flushed pose).
  - **Slice 2c (<2c commit>) CELL 2 — kill DEST pre-freeze:** the saga commits to the dead dest then FORWARD
    re-homes — `rehome_event_for`'s `Promoting` branch (gated on the CURRENT directory owner's liveness, NOT the
    stale `ctx.dest`, so it fires AT MOST once per owner-death — the fence never runs away) + `select_rehome_target`
    + the roster (`SagaRuntimeRes.roster` from `OrchestratorConfig.roster`; the cluster maps stubs → empty profile)
    + the stub `on_re_home` adopt (CREATES an Owned dot from the pose at a FRESH target — no ghost to flip). End
    state `EntityRecoveredRealmOrphaned{entity_at: SHARD}` (DEST dead ⇒ SHARD the lowest live shard).
  - **Slice 2d (`<2d>`) the re-home adopt's SELF-SUFFICIENT re-drive egress (holistic audit `wf_dd38151d` HIGH):**
    `SagaState::Promoting` gained `rehome_target: Option<NodeId>`; a re-homed `Promoting` (Some) Timeout re-drives
    `A::ReHomeAdopt → target` (re-read from `flushed_pose`, idempotent at the target) instead of `A::Promote →
    ctx.dest` (the confirmed-dead original dest). So the re-home adopt no longer depends on transport at-least-once —
    `scan_deadlines` drives the re-solicit. Closes the SECOND producer-less phase [[D-6]] precondition 1 named.
- **✅ Slice 3a LANDED — the STANDING re-home CONTROL PATH (kill SOURCE pre-freeze, CELL 3):** the freeze timeout
  aborts (`abort_with_thaw`, gateway-acked, fence-neutral `abort_clear`) + tombstones, leaving the directory at the
  now-DEAD source, UNLOCKED, with NO live saga. The expiry reaper (`reap_lapsed_leases`) now, on a confirmed-dead +
  lapsed + UNLOCKED **`Entity`** key, enqueues a `PendingReHome` (RAM within-barrier hand-off; the durable artifacts
  are the locked record + the armed saga, so a kill-9 in the enqueue→drain window is covered by the reaper
  re-detecting the still-dead UNLOCKED record on reboot — no new WAL family). `process_rehome_starts` then picks a
  LIVE capability-matched target (`select_rehome_target`), LOCKS the key (`lock_transfer` — so the next sweep skips
  it), and ARMS a fresh saga that PARKS in `ReHoming{target}` via `saga::start_rehome`. CONSERVATIVE split: authority
  STAYS at the dead owner (no CAS), NOTHING is emitted (HR1: no fabricated pose — `flushed_pose: None`). NO live target
  (whole-pool death) → the entry DROPS, the reaper retries next sweep (interval-paced, never a forced incapable
  re-home). **The reaper arms ONLY a dead-owner Entity that is BOTH `in_transfer`-unlocked AND has NO live saga
  (`subject_has_live_saga` cross-checks `runtime.sagas`, audit `wf_3b9eb7f0`):** `commit_cas` CLEARS the lock at the
  commit point while a POST-commit `Promoting`/`ReHoming` saga lives on (it re-homes a dead committed owner ITSELF
  via `scan_deadlines`), so an unlocked key can still be saga-owned — the cross-check makes "one re-home arm per key"
  an ENFORCED invariant (the lock alone does not, post-commit), preventing a double-arm + a leaked parked saga. A
  LOCKED Entity, an unlocked Entity a live saga still owns, and Realm/Ship are all LEFT (Slice 4 / the owning saga). The re-home `TransferId` is
  deterministically derived (FNV over `(subject, prev_fence)`, namespaced `0x37` — never collides with a
  client/gateway id). Proven by 5 deterministic unit tests (`saga::start_rehome_arms_parked_in_rehoming`;
  `reap_in_freezing_orphan_enqueues_a_pending_rehome_and_arms_a_parked_saga`; `reaper_leaves_a_locked_dead_entity_*`;
  `process_rehome_parks_when_no_live_target`; `process_rehome_skips_an_already_locked_key`), 100% Tier-A region+branch.
- **✅ Slice 3b LANDED — the crash-matrix flip:** Slice-3a is now wired into the `p3_crash_matrix` CELL-3 cell. A
  CELL-3-SPECIFIC reaping cluster (`p2_cluster_reaping`: non-zero `reaper_interval_ticks` + short `lease_ttl_ticks` —
  NOT a global change; the other cells keep the inert default so a short lease never spuriously lapses + reaps their
  live owners) makes the reaper fire in the full kill scenario, and the `Scenario.standing_rehome` flag runs the FULL
  quiesce window (the early `live_sagas == 0` break is skipped — the aborted saga tombstones BEFORE the reaper arms
  the re-home). `p3_kill_source_in_freezing_arms_a_standing_rehome` now asserts `ParkedHalfOpen{SOURCE}` (the existing
  asserter: a LIVE parked re-home saga + the directory at the dead node + the dead-aware oracle's exact `HeldNowhere`
  — never a false pass). The byte-identical seed-replay canary still holds (the re-home is deterministic). Was
  `DeadOwnerOrphan{SOURCE}` before Slice 3.
- **Still owed (Slice 4):** generalize the re-home from Entity to **Realm/Ship** keys (ships/stations/cities are
  Realms — PLAN.md:82,141) — the reaper's Realm/Ship arm + the Realm adopt effect + D-33 N+1 CAS for ships — AND turn
  `start_rehome`'s empty action list into `ReHomeCommit{expected: prev_fence, target}` (the CAS off the corpse,
  fence-monotone) + the entity ADOPT via `ReHomeState::Snapshot` (the new owner opens the RealmId-keyed redb made
  single-writer by the CAS — HR1-clean, never the sealed dead store; D-6/P7); G-IDENTICAL on ≥2 ShardProfiles; flip
  the cured cells to `SettledAt` once the dead owner's REALM also re-homes (the `RealmHeldNowhere` residual the
  `EntityRecoveredRealmOrphaned` intermediate honestly surfaces).
- **Deferred sub-items (accepted user-decided scope, 2c-UNREACHABLE today, recorded so the green gate is honest):**
  - **The live-resurrected-stale-held RESURRECT test:** the harness CANNOT resurrect a killed node IN-PLACE
    (`FaultFabric::kill` sets killed+crashed; `reregister` drops the old World → empty Dots, no stale claim survives).
    A true resurrect-after-rehome test needs NEW harness Dot-seeding support. 2c ships the dead-aware-oracle
    fence-monotone proof instead (the re-home moves authority uniquely past the corpse's fence).
  - **The stub ENTITY-head self-fence twin:** `self_fence_lapsed_realm` (`stub.rs`) is REALM-keyed only; there is NO
    periodic entity-head self-fence on a held Owned dot whose `Entity` directory head advanced past its held fence.
    2c-UNREACHABLE (it requires the in-place resurrection the harness lacks; the dead-aware oracle excludes the
    corpse), so a sound deferral — the `re_home_without_realm` path is where the entity-head twin would attach.
  - **~~The PROD roster is EMPTY~~ STALE, corrected by the Stage-C audit (2026-08-14):** the prod
    orchestrator has filled its roster from `VD_ROSTER` (empty profiles) since Track R/1d.2, so the
    old "a prod re-home PARKS" safety argument was FALSE — the selection was live and REALM-BLIND
    (the audit's HR3 critical). ★FIXED: `select_rehome_target` now takes ONLY the live directory
    owner of the stashed pose's own realm (the one node whose `place_arriving_pose` accepts the
    frame; usually the source, or whoever the realm re-homed to) — no roster fallback exists; no
    placeable owner ⇒ PARK (honest red, re-resolved each scan fire). The pose-less STANDING re-home
    parks at the DEAD owner (no pose ⇒ nothing placeable to name) with the key locked; the Slice-4
    adopt must select pose-aware when the checkpoint pose lands.
  - **The D-36 starved-watermark park after re-home:** the re-homed entity at a fresh target has no gateway sub
    delivering frames, so `DeliveredToObservers` may starve and the saga parks in `Promoting` (never Done). The entity
    IS recovered (directory + held-set), so the cell end-state holds; the client-input re-route to the new target is
    the connection-plane refinement owed with the ResumeTicket adoption ([[D-36]]).
  - **✅ The re-home ADOPT now has an orchestrator re-drive egress (Slice 2d, was holistic audit `wf_dd38151d` HIGH):**
    a re-homed `Promoting` carries `rehome_target: Some(target)`, so a `Promoting` Timeout re-drives `A::ReHomeAdopt →
    target` (the live target) NOT `A::Promote → ctx.dest` (the dead original dest); `scan_deadlines` drives it once the
    directory names the live target (`rehome_event_for` → `Timeout`). The re-home adopt no longer leans on the
    FaultFabric's at-least-once redelivery — saga recovery is SELF-SUFFICIENT. Closes the SECOND producer-less phase
    that [[D-6]] precondition 1 named; that additive FSM-field fix is now the proven template for AwaitAdopt's owed
    egress. Covered by `a_rehomed_promoting_timeout_redrives_the_adopt_to_the_live_target` (FSM Some-arm) +
    `a_rehomed_promoting_redrives_the_adopt_to_the_live_target_via_scan_deadlines` (the END-TO-END producer chain:
    scan_deadlines → rehome_event_for → deliver → emit_rehome re-sends the ReHome to the live target, NOT a Promote to
    the dead dest). The CELL-2 crash matrix does NOT exercise this re-drive — its first adopt lands over the perfect
    FaultFabric link, so reverting the Some-arm leaves the matrix green; those two targeted tests RED instead (the
    regression guard the matrix lacks — empirically confirmed, review `wf_a401ca14`).
- **Source:** design `wf_6efc70f1` (judge-panel) + the empirical crash matrix; review `wf_688a65d9` (DONE_NO_CRITICAL
  on the code) + holistic audit `wf_dd38151d` (DONE_NO_CRITICAL; surfaced the re-home-adopt producer-less-phase HIGH,
  cured by Slice 2d) + Slice-2d focused review `wf_a401ca14` (DONE_NO_CRITICAL; its MEDIUM — the re-drive lacked an
  end-to-end guard, the crash matrix passing it via fabric redelivery — closed by the two targeted re-drive tests +
  this honest test-credit). This ledger sync closes their ledger-honesty findings.

### D-38 🟩 HR4's literal G-IDENTICAL gate — the first `assert_feature_anywhere` LANDED (Spherical Shell + Cartesian Aabb crossing); the `reanchor()`-forcing variant re-owed at P5
- **LANDED (the foundation):** `crates/sim/src/capability.rs` has the validated `ShardProfile` capability DAG (private
  fields, `ShardProfile::build()` the only ctor), the 5 canonical kinds as DATA, the coherence test
  `canonical_profiles_are_coherent`, and the negative gate (incoherent profile fails loud, `capability.rs:227`). This is
  HR4's STRUCTURAL half: features are written against capability traits and a shard type is zero new code.
- **LANDED (the behavior half — the crossing feature; Slice 4a / task #133):** a NAMED `assert_feature_anywhere` test
  (`crates/sim/src/stub.rs` mod tests) runs ONE identical crossing-feature fixture (`drive_inward_crossing_feature`, a
  single closure body) on BOTH a **Spherical** profile (`profiles::planet()`) with a **Shell** boundary AND a
  **Cartesian** profile (`profiles::station()`) with an **Aabb** boundary, tying each run to its `VoxelGeometry`
  (Spherical⇔Shell, Cartesian⇔Aabb). INWARD only. The assertions are STRONG, not a bare count: each run emits EXACTLY
  ONE inward `CrossingRequest` whose `to_realm` == the boundary's `to_realm` (proving the box/shell actually GATED), the
  membership provably crossed the create edge (start OUTSIDE the destroy edge → end INSIDE the create edge, anti-vacuity),
  and the Aabb run drives a DIAGONAL segment (≥2 non-zero axes) so `segment_aabb_crossing`'s slab-corner path is genuinely
  exercised. The new `RealmBoundary::aabb(...)` ctor + `OverlapBand::for_box_velocity_safe(...)` (both `crates/core/src/geometry.rs`,
  100% Tier-A) mirror `shell(...)`/`for_soi_velocity_safe(...)` in NORMALIZED Chebyshev units so the ONE fixture drives both.
  **LIMIT (batch-2 closure sweep, finding 6):** both runs share ONE `Rig::new()` shard kind — the fixture varies the child
  REGION SHAPE, and the Spherical⇔Shell / Cartesian⇔Aabb ties are equality asserts on the PROFILE OBJECTS, not the hosting
  rig. The two-SHARD-KIND run of one identical fixture body is `drive_swept_crossing_feature` (D-PLACE-1: the same closure
  on `NodeKind::StubShard` AND `NodeKind::Shard(planet)`). Fold the two when the profile's NodeKind is wired through
  `drive_inward_crossing_feature`.
- **STILL OWED (the `reanchor()` half — P5):** the D-38 doc (the `capability.rs` KNOWN-LIMIT header) mandates ONE fixture forcing a
  `reanchor()`. `FrameSpace`/`SphericalSpace`/`CartesianSpace` + `reanchor()`/`AnchorGen` do not exist yet (P4/P5), so the
  crossing feature is the FIRST but not the reanchor variant. The `assert_feature_anywhere` HARNESS is now real; the
  reanchor-forcing fixture is re-owed at P5 (block-edit on planet AND ship). The gate is no longer absent — it exists and
  is green for the crossing feature.
- **WHY this is an INTERIM, not a defect (the deferral is sound):** there is ZERO feature code to diverge yet — the only
  sim is the stub shard (points in empty space), and real capability-bearing features begin at **P4** (terrain) / **P6**
  (block edits, the first true G-IDENTICAL fixture — "block edits on planet AND ship"). A G-IDENTICAL harness now would
  assert sameness over an empty feature set: scaffolding with nothing to protect (the smallest-correct discipline forbids
  it). The `FrameSpace` seam it tests (`SphericalSpace`/`CartesianSpace` + `reanchor()`/`AnchorGen`) itself lands at P4/P5.
- **WHAT the audit caught (the honesty hole this entry closes):** CLAUDE.md HR4 + PLAN.md schedule G-IDENTICAL as a
  PERMANENT gate, yet it was absent from BOTH the code AND this ledger — invisible to the "a phase isn't done until its
  DEFERRED entries flip green" gate. The DEFECT would be the UNLEDGERED absence (the same class [[D-31]] closed for HR2's
  `TransferableKind` and [[D-32]] closed for the directory partition seam), not the deferral. This entry is the fix.
- **When / proper:** the `assert_feature_anywhere` harness + the first G-IDENTICAL fixture LANDED at Slice 4a (the
  crossing feature). The `reanchor()`-forcing fixture lands with the first `FrameSpace`-bearing feature — **P5/P6** (block
  edits on a planet `SphericalSpace` AND a ship `CartesianSpace`, one fixture forcing a reanchor), per the PLAN.md P6
  delta. Every transition phase from P4 onward then adds its own fixture to the now-real harness.
- **Pin (flipped 🟩 for the crossing feature):** `crates/sim/src/stub.rs` `assert_feature_anywhere` runs one identical
  crossing fixture green on a Spherical Shell + a Cartesian Aabb profile; `crates/core/src/geometry.rs` `RealmBoundary::aabb`
  / `OverlapBand::for_box_velocity_safe` are the ctors it drives. The `reanchor()` variant is the only owed half (P5).
- **Dependency (for the reanchor half):** `FrameSpace` (`SphericalSpace`/`CartesianSpace`, P4/P5); the first
  `FrameSpace`-bearing feature (P5/P6).
- **Source:** the b0304ca holistic audit `wf_2e97cbe7` opened this; Slice 4a / task #133 discharged the crossing half
  (design + adversary review `scripts/slice4_design_adversary.md`).

### D-39 🟧 Three orphaned-but-additive responsibilities — design-mandated, unbuilt, and (until now) UNLEDGERED (holistic audit `wf_ed40e95e`)
All three are ADDITIVE at their feature phase (no landed byte/decision changes shape) — the DEFECT was their absence
from this registry, invisible to the "a phase isn't done until its DEFERRED entries flip green" gate (the same
honesty-hole class [[D-31]]/[[D-32]]/[[D-38]] closed). Ledgered here so each lands with a green-gate pin.
1. **Cross-shard BLOCK-EDIT / world-mutation forward-to-realm-owner routing.** PLAN.md:100 binding mandate
   ("cross-shard block edits are FORWARDED to the realm owner — single-writer redb, never written locally") +
   `integration.json` orphaned[3]. No code, no `ClientControlMsg` world-mutation arm, unrepresentable in the gateway's
   single-authority `route_input`. ⚠️ RISK: a P6 implementer writes the edit locally on the authority shard = the exact
   HR1-forbidden anti-pattern. **Owed (P6):** a reliable discrete edit action + gateway realm-ownership resolution via
   the directory `Realm` key (reuse the [[D-32]] `coordinator_of` resolver) + the `InterShardFlow::BlockEdit` carrier
   (a reserved arm; additive). WHEN: P6 (block edits + persistence). **Second consumer (audit `wf_2c963246`, P11):**
   the SAME reliable client→shard discrete-action carrier is what the PvP FIRE trigger needs — `action_bits`
   (`crates/wire/src/channels.rs`) rides the UNRELIABLE latest-wins `InputDatagram` today, and its own comment
   promises "discrete world-mutating actions ride reliable channels (v1.1)", a channel that does NOT yet exist.
   Hitscan/projectile hit-reg must never LOSE a fire event, so P11 combat reuses this ONE reliable C→S arm (the
   `ClientControlMsg` discrete-action family + the exhaustive-match `MsgClass` extension — additive, never a
   per-feature fork; the SAME build-once shared infra as block-edit). Pin the `channels.rs` `action_bits` comment to
   name combat as the second consumer so the "v1.1" promise is ledgered, not floating.
2. **Gateway↔shard CONNECTION POOL (`pool_size_K`).** `connection_plane.md` §3 (Attack-2.minor) specifies K conns/pair;
   the mesh currently uses a single reliable stream per pair (which CORRECTLY satisfies `transfer_protocol.md` §1.4 —
   the down-grade here corrects an over-claim). A landed design knob, unimplemented + unledgered. NOT a pre-feature
   blocker, but should land before any multi-shard scale/load run. WHEN: P6-scale / the first multi-shard soak.
3. **Warp `AwaitProvision` child-saga machinery.** The FSM has a bare `vec![]` stop where `AWAIT_PROVISION` belongs —
   no `ProvisionIntent` persist-before-spawn, no Spawn-Resolver, no de-provision compensation, and no dedicated ledger
   entry. **Owed (P10):** the full provisioning child saga per PLAN.md (persist-intent-before-spawn; self-terminating
   unwanted pods; idle-GC). WHEN: P10 (warp + provisioning). Additive — the saga FSM already fans out at CasWon/CasLost.
4. **Space STATIONS / planet-CITIES have no `RealmId`/`FrameRef`/`ShardProfile` representation yet** (`core/src/pose.rs`
   `RealmId` = Planet/System/Ship; `sim/src/capability.rs` the 5 canonical kinds) — the design names them Realms built
   from blocks but there is no station/city kind. Additive when P8 (ships/stations) lands (a new `RealmId` arm + a
   `ShardProfile` capability config — zero new transfer code, HR2/HR3); ledgered now so it is not discovered late.
   WHEN: P8.
5. **Gateway subscriptions are keyed by shard `NodeId` — one sub per (session, shard).** `gateway.rs`
   `subs: BTreeMap<NodeId, SubRecord>` + `SubTable::lookup(shard)` + the shard-keyed snapshot tag bake the
   ONE-realm-per-shard assumption into the routing KEY (CONSISTENT with today's design — planets are single-cluster,
   ships get their own shard — so additive, NOT a defect now). But a STATION shard hosting multiple docked-ship realms
   a client renders at once (the natural .4 station model) would collapse them into one sub (`open_sub` on the same
   shard `NodeId` overwrites the prior `SubRecord`). If any shard ever hosts >1 client-subscribed realm, the `by_shard`
   sub key + the shard-keyed snapshot must become `(shard, realm/sub)`-keyed — a route-identity + snapshot-tag +
   `open_sub`/`close_sub` change. WHEN: P8 (with .4). Source: audit `wf_032b80eb` (integration-1).
6. **Ghost-carried replicated combat-STATE blob (health/shield/anim/pilot-flags).** `transfer_protocol.md:107` binds
   that ghosts carry a small replicated blob alongside pose; the shipped `GhostFlow::Spawn`/`Delta`
   (`crates/wire/src/intershard.rs`) carry pose + fences ONLY. Cross-boundary PvP (A on shard 1 shoots B, a ghost
   owned by shard 2) needs B's combat state on A's shard to render the health/downed state + gate a hit before
   forwarding the fire-event to B's owner. ADDITIVE (GhostFlow is INTERNAL mesh wire, kind-generic kernel): lands as a
   NEW `GhostFlow` VARIANT (a FIELD-append to Spawn/Delta is NOT postcard-safe — `channels.rs` append rule), read-only
   display state; the authoritative hit is applied at the ghost's OWNER via the D-39.1 forward path. WHEN: P11 combat
   (or P5 if cross-boundary collision-response needs it earlier). Pinned in the `GhostFlow` doc-comment. **Lockstep
   site (audit `wf_9f26b8cb`):** the GhostNeighbor-insert + `GhostFlow::Spawn` emit is byte-identical in BOTH
   `promote_apply` and `re_home_apply` (`sim/src/stub.rs`, DRY-pinned at both); when this blob lands its new variant,
   EXTRACT a shared `register_and_spawn_source_ghost` helper so the field-append touches ONE place (re_home_apply is
   exercised only by the D-37 kill-cell tests, not the happy-path gate, so an un-extracted edit silently drifts).
   Also feeds [[D-42]] (the rewind reads this combat state). Source: audit `wf_fd6a4b9d` (pvp-readiness).

### D-40 🟧 Tier-B PROCESS-tier coverage %c-merge for the spawned node BINARIES (the deeper half of the HR5 Tier-B ratchet)
- **✅ LANDED (the io-prod half):** the `coverage-io-prod` recipe enforces a RATCHETED regions floor (`tier_b_floor`,
  currently 90, baseline ~92.5%) on io-prod's OWN in-process unit tests — deterministic (no SIGKILL counter loss), wired
  into `gate` via `coverage`. This closes the holistic-audit `wf_da9e2be2` HIGH: io-prod (the live crash-durability +
  mesh path) now has a real, non-theater coverage signal that cannot regress silently.
- **OWED (the binaries half):** the full process-tier merge — instrument the spawned node BINARIES (orchestrator /
  gateway / shard / client) under `LLVM_PROFILE_FILE=…%p-%m%c` (continuous mode MANDATORY: the harness SIGKILLs them, so
  an atexit flush is lost), merge across the children, and report a ratcheted floor. `orch-crash-cov` already accumulates
  the profraws via `--no-report` but has NO `report --fail-under` step; the `process_parity`/`dev_cluster_smoke`/
  `client_load` binaries are likewise uninstrumented. **Owed: a `coverage-process` recipe (show-env + the %c merge +
  `report --fail-under` at a ratcheted floor) folded into `coverage`/`gate`.** Honestly never 100% (a SIGKILL can lose
  the final flush — exactly why it is a ratcheted FLOOR, not a 100% gate). WHEN: the process-tier coverage slice (a
  coverage-infra pass; not blocking — the io-prod logic is already floored in-process).

### D-1 🟩 Transfer abort clears the directory lock (Slice 2a)
- **✅ CLOSED (Slice 2a):** the terminal `Aborted` edge now emits `SagaAction::ClearTransferLock` →
  `DirectoryCore::abort_clear(subject, transfer)` — the STALE-FENCE re-read (a CAS-loser/aborter's `expected_fence`
  is stale by definition, so a naive `abort_cas(expected)` would also Lose; `abort_clear` re-derives the head and
  clears the lock IFF still held by THIS transfer, NEVER steals another saga's lock). Authority is UNCHANGED (abort
  keeps the source owner) and the abort is **FENCE-NEUTRAL** (`abort_clear` does NOT bump — abort-path fix, see the
  D-6 abort-path note: an abort is not an ownership change, no crossing is ever in flight to fence out where the
  bump fired, and a bump would strand the surviving source / wedge a logout `LeaseRevoke`). Idempotent (a re-driven
  terminal whose lock already cleared = `Lost` no-op, driven by the lock not the fence). The two pinned
  "exists-to-be-flipped" asserts (`prepare_rejection_aborts_and_tombstones_with_typed_feedback`,
  `cas_loss_unwinds_with_a_thaw`) are FLIPPED from `in_transfer == Some(XFER)` to `== None` (authority still
  `Shard(SOURCE)`, fence UNCHANGED). An aborted subject can immediately re-transfer (R1's directory-side
  manifestation is cured).
- **RACE-1/D-23 closed-without-unmasking:** `ClearTransferLock` fires at the TERMINAL edge (after BOTH compensator
  acks), so the gateway's `apply_abort` has pruned the session journal before any re-transfer's `PrepareSubscribe`
  could discard it — see D-23.
- **Source:** Slice-1b audit `wf_34ef74d1` (CPO-1/CPO-2); closed by Slice-2a design `wf_9f22c70d` + audit.

### D-2 🟩 Demote→Release tail: the ORDERED, fence-enforced demote-before-promote tear-out is COMPLETE (1d.5b.1→.3d)

> **⚠️ READER NOTE — current state (post-1d.5b.2):** the two **✅ LANDED** blocks below (1d.5b.1, 1d.5b.2) are the
> CURRENT truth. 1d.5b.1 landed the saga-pushed ordered demote-before-promote FSM + consumers; 1d.5b.2 TORE OUT the
> 1c.8 source granted-key poll, so the saga-pushed `Demote` (`on_saga_demote`) is now the **SOLE** source-demote
> driver — `granted_key_poll_tick` and the per-entity poll no longer exist. The historical-interim descriptions in
> the numbered consequences (1–5) and the 1d.1-layering note below describe the NOW-REMOVED 1c.8
> promote-before-demote poll model; they are RETAINED as the why-it-was-broken record, **not** the current state.
> **✅ D-2 NOW GREEN (1d.5b.3 COMPLETE):** .3b landed the strict demote-before-promote ordering (the dest promote
> relocated into `on_saga_promote`) + the GhostFlow source-Ghost collider FEED; .3c landed the band-exit Despawn +
> dual-registry teardown (the source-ghost lifecycle END); .3d landed the PER-TICK mid-flight `verify_authority_unique`
> with the **single W1 excuse** (post-CAS source-still-Owned `DirectoryDisagrees`) — the zero-Owned gap needed NO new
> excuse (the existing `in_flight_to_owner` path already covers it; empirically `HeldNowhere` never fires mid-flight)
> and the two-Owned overlap is UNREACHABLE under strict demote-before-promote (so NO uncoverable excuse was added).
> Residuals are SEPARATE owed items, NOT D-2: the orchestrator-side teardown gate (Slice-2/D-6), the cooperative
> in-memory freeze vs the fence-enforced freeze (`transfer_protocol` §2.4), and the symmetric Realm/Ship-half
> mid-flight excuse (P8/P10). **✅ the lost-`Demote`/`Promote` re-drive producer LANDED in Slice 2a** (the
> `scan_deadlines` `Timeout` producer re-drives a never-acked post-commit saga; closes D-1 too — see D-1 🟩).
- **LANDED (1c.8 — interim):** `interim_demote_complete` now exists in `crates/node/src/saga_runtime.rs` — an
  UNCONDITIONAL bandless stand-in, invoked from `drive_sagas` AFTER the ack loop, that for every live saga in
  `Demoting` calls `deliver(.., SagaEvent::DemoteComplete)` (a new CALLER of the existing sink — NOT a magic
  tick, NOT an in-flight-queue injection) with a LOUD `tracing::warn` naming it the interim, + the
  "exists-to-be-flipped" unit tests (`interim_demote_complete_drives_a_demoting_saga_to_done` and its
  not-Demoting skip twin). This drives `Swapping→Demoting→Releasing→Done` end-to-end NOW. The gateway's
  `ReleaseSubscribe` handler flipped from PARK to acking `Released` (`apply_release`, the only existing-vocabulary
  ack the tail needs — `release_subscribe_acks_released`). `ROB-DEMOTE-PARK` is RETIRED: a successful saga now
  reaches a terminal and tombstones, so `SagaRuntimeRes.sagas` no longer grows per transfer (the only correct
  bound, as this entry always said). FF-1 holds (the `DemoteComplete→Send(ReleaseSubscribe)` emits no
  synchronous CAS feedback — loop depth stays 1; stated in a code comment).
- **⚠️ ORDERING INVERSION — the 1c.8 interim does NOT honor the binding ordered Demote-before-Promote invariant**
  (`transfer_protocol.md` §2.1 "the #1 fatal" split-brain mitigation; `integration.json` blocking-resolution #1
  — the COMMIT / route-swap-ordering issue, same ref as `generic_transfer.md`: "Keep
  transfer_protocol's Demote-before-Promote ordering AND the gateway lease-epoch/fence frame-drop fence"). 1c.8
  does the OPPOSITE — **promote-before-demote, both as independent unordered directory polls**: the DEST promotes
  via its own post-CAS adopt-`HeadRead` grant-flip (`stub.rs` `flip_grant`), and the SOURCE demotes LATER via its
  own granted-key poll (`stub.rs` `granted_key_poll_tick` → `self_fence_foreign_entity`, a LOCAL `dots.remove`).
  There is NO saga-driven `Demote`/`DemoteAck` and NO ordering gate. CONSEQUENCES, stated plainly so a 1d
  implementer does not under-budget or mistake this for the permanent design:
  1. **Transient two-holder / different-fence window** — between the dest grant (`@new_fence`) and the source's
     next poll-drop (`@old_fence`), BOTH shards report `granted`. `verify_authority_unique` would REJECT this
     mid-window (WrongHolderCount / FenceMismatch); the 1c.8 gate only samples authority POST-quiesce (after the
     two-window precondition poll resolves), so the invariant is asserted at steady state, never proven mid-flight.
     It is MASKED today (not prevented): `render_ready=false` (D-27) means neither emits client frames, and the
     single-shard gateway routes only to the dest post-swap — so there is no client double-vision, but the
     AUTHORITY-level double-hold is live.
     **⚠️ 1d.3 EMPIRICAL FINDING (the flip UNMASKED the window — and it is a VANISH, not an overlap):** Slice 1d.3
     flipped `render_ready=true` on the dest crossing-apply, so the dest now emits frames. The expected payoff was a
     SEAMLESS two-holder overlap (source still rendering while the dest comes up, composited to one). Measured
     against the code (zero-fault lockstep, `p2_transfer_gates`), the OPPOSITE happens: the SOURCE sub closes
     ~3 ticks BEFORE the DEST sub opens at the client, so the avatar renders NOWHERE for those ticks — a real
     client-visible VANISH, NOT an overlap. Root cause: the source-sub close is the saga's `ReleaseSubscribe`, fired
     by the **unconditional** bandless `interim_demote_complete` (it is NOT gated on the dest being adopted /
     delivered-to-observers), and so it wins the race against the dest's slower multi-round-trip adopt
     (`SubscriptionReady`→`open_sub`). The proper (a)-predicate (dest **delivered to all observers** before
     `DemoteComplete`) is EXACTLY the gate that closes this — i.e. the seamless no-vanish visible-crossing gate is
     BLOCKED on D-2(a)+(b), not on anything in 1d.3's render-only scope. 1d.3 DID land the visible crossing (the dest
     renders the crossed seed-7 pose from the dest sub); it did NOT — and structurally cannot, pre-D-2 — land the
     seamless overlap. **✅ FLIPPED IN 1d.5a (EARLIER than this finding predicted — the prediction is corrected
     here):** the (a) delivery predicate ALONE closed the render vanish. The finding above said the seamless overlap
     was BLOCKED on (a)+(b); the reality is that gating `DemoteComplete` on dest delivery delays the saga's
     `ReleaseSubscribe` (the source-SUB close) until AFTER the dest is delivered+rendered — so the source sub HOLDS
     its track through the dest's first frame, the FORK-0a two-sub overlap (designed in 1d.2) finally manifests, and
     the client de-dups to ONE render via `AuthorityChanged`. The capstone
     `p2_dod_the_cross_shard_crossing_renders_at_the_dest_at_the_crossed_pose` now asserts `overlap_ticks >= 1` +
     `vanish_gap == 0` + the `verify_no_vanish`/`verify_pose_continuity` oracles (GREEN). The RENDER seamless is a
     READ-plane property (the source-sub release TIMING); it is INDEPENDENT of the (b) WRITE-plane ordering tear-out,
     which is still owed (the source still poll-demotes its authority, the dest still autonomously promotes — no
     fence-enforced demote-before-promote yet). So 1d.5a delivered seamless RENDER; (b) delivers authority-ordering
     ROBUSTNESS + the GhostFlow collider feed + the band-exit Despawn.
  **✅ LANDED 1d.5b.1 (the WRITE-plane ordering MACHINERY — `0218dc8` wire arms + this slice):** the saga FSM now
  carries the ORDERED tail `Swapping → Demoting{dest_delivered} → Promoting{promote_acked,dest_delivered} →
  Releasing` (`crates/sim/src/saga.rs`): RouteSwapped pushes the saga `Demote` to the source; the source's
  `DemoteAcked` ALONE (the R1-deadlock fix — NOT gated on delivery) advances to `Promoting` + pushes the `Promote`
  to the dest; the source sub releases only once BOTH `PromoteAcked` AND `DestDelivered` land (`promoting_advance`,
  the seamless gate; an early `DestDelivered` is latched in `Demoting` and carried forward so it is never lost — no
  park). The runtime (`saga_runtime.rs`) routes `Demote`→`ctx.source` / `Promote`→`ctx.dest` as pure egress and the
  1d.5a interim `DemoteComplete` delivery-pass + its `LiveSaga.dest_delivered` latch are DELETED (the single release
  path is now the `Promoting` gate — the capstone can no longer shortcut, so it HONESTLY exercises the round-trips).
  The source consumer `on_saga_demote` (`stub.rs`) does the REAL `Owned→Frozen→Ghost` (reusing
  `self_fence_foreign_entity` at the real `cmd.transfer`/`new_owner_fence`), fence-idempotent with the still-live
  poll, acking `DemoteAck` UNCONDITIONALLY (never wedge). **STILL OWED — split across 1d.5b.2/.3:** (i) the source
  granted-key POLL is NOT torn out yet (it coexists idempotently); → 1d.5b.2 (poll tear-out + the R2 mid-flight
  `verify_authority_unique` oracle excuse). (ii) the dest's REAL `Ghost→Owned` flip is STILL `apply_crossing`'s
  autonomous promote (RETAINED — relocating it into `on_saga_promote` would DELAY the dest's first-Owned moment by
  the multi-tick round-trip and re-open the vanish); `on_saga_promote` is a fence-idempotent CONFIRMER (journals
  `PROMOTE_STEP` + acks `PromoteAck`) in 1d.5b.1; → 1d.5b.3 relocates the flip INTO `on_saga_promote` co-landed with
  the GhostFlow source-Ghost collider FEED that keeps the consequent later-promote window seamless, + the band-exit
  Despawn. So the strict demote-before-promote ORDERING invariant (dest-Owned only AFTER source-DemoteAck) is NOT
  yet enforced (apply_crossing promotes the dest early); it lands in 1d.5b.3 with its own gate.
  **✅ LANDED 1d.5b.2 (poll TEAR-OUT — the saga `Demote` is the SOLE source-demote driver):** the 1c.8 source
  granted-key poll is GONE (`stub.rs`): `granted_key_poll_tick`, the `poll_granted_keys` arg + the `pending_grant_op`
  poll-HeadRead branch, the `on_directory_reply` FOREIGN-owner self-fence reaction, and the inert `SELF_FENCE_TRANSFER`
  const are all DELETED; `self_fence_foreign_entity` now takes a real `TransferId` (no `Option`/fallback) and is called
  ONLY by `on_saga_demote`. The realm-lease recheck (`realm_recheck_interval`) is KEPT (realm-key self-fence only).
  Coverage-preserving: the deleted poll tests' coverage moved to the saga-demote tests + a NEW `the_saga_demote_on_an_
  unheld_entity_is_a_clean_noop_but_acks` (the no-match arm, whose only producer was the poll reply).
  **⚠️ EMPIRICAL FINDING (probe vs the capstone, `realm_recheck_interval=4`) — the design's two-Owned assumption was
  REFUTED, the mid-flight authority oracle is genuinely 1d.5b.3 work:** the avatar's Owned-holder timeline is
  `[SHARD] t7..t18 → [] t19..t20 → [DEST] t21..`. There is NO two-Owned overlap at the AUTHORITY layer (the dest's
  adopt is SLOWER than the source's saga `Demote`, so it promotes at t21 AFTER the source demotes at t19); the handoff
  is a 2-tick ZERO-Owned window (t19–t20) — the demote-before-promote gap, INVISIBLE to the client (the render gates
  stay seamless via last-frame hold + the dest's t21 frame; this gap is pre-existing from 1d.5b.1, just unmeasured).
  Consequence: a `verify_authority_covered` (`>= 1` Owned) mid-flight gate was DESIGNED then REJECTED — it forbids the
  intrinsic zero-Owned handoff gap. A strict `verify_authority_unique` (`== 1`) mid-flight gate must EXCUSE THREE legal
  transfer windows (post-CAS source-still-Owned `DirectoryDisagrees`; the zero-Owned gap via dest-pending; any overlap)
  — that excuse logic + the per-tick wiring land in 1d.5b.3 (co-landing with the apply_crossing relocation that makes
  the ordering strict). 1d.5b.2 adds NO new authority oracle; the capstone's RENDER gates carry the player-visible
  seamless guarantee (which is the property that matters), and `verify_authority_unique` stays the post-quiesce gate.
  **⚠️ LOSS OF AUTONOMOUS RECOVERY (disclosed, P3/Slice-2 scope):** the torn-out poll was ALSO an independent recovery
  path — it would self-fence a stranded source on the next recheck tick even if the saga `Demote` never arrived. The
  saga `Demote` is now the SOLE driver with NO self-heal, and there is NO production `Timeout` PRODUCER yet (`saga.rs`
  HONEST SCOPE), so a lost/never-acked `Demote` strands the source `Owned` (a two-holder split-brain with the
  autonomously-promoted dest), surfaced only as a parked saga in the admin staleness view. Correctly P3/Slice-2 scope
  (the at-least-once / adaptive-timeout / re-drive machinery) — only the disclosure is owed here; the parking itself is
  already documented honestly at `saga.rs` (the `Demoting`/`Promoting` timeout re-emit arms).
  **✅ LANDED 1d.5b.3a (carrier prereq, render-neutral):** `MsgClass` gains `GhostReliable` (Spawn/Despawn,
  Reliable) + `GhostDelta` (Delta, Unreliable latest-wins) (`sim/io/mod.rs`). UNROUTED — no emitter/consumer yet
  (the source-ghost feed lands at 1d.5b.3b), so zero render-timing change (the capstone trace is byte-identical).
  **✅ LANDED 1d.5b.3b (the COUPLED CORE — strict ordering + the source-ghost collider FEED):** the dest
  `Ghost→Owned` promote RELOCATED out of `apply_crossing` (which now only stores the crossed pose; the dot stays
  Ghost) INTO `on_saga_promote` (the real promoter, pose-before-promote guarded), so demote-before-promote is
  STRICT — the dest becomes Owned ONLY on the saga `Promote`, after the source demoted. The dest read-sub
  (`SubscriptionReady`) moved from the adopt flip to `on_saga_promote`, so the client's render authority moves to the
  dest only at promote (it stays on the SOURCE sub until then). The dest-INITIATED ghost feed (`GhostFlow` owner →
  ghost-host, `PromoteCmd.source` tells the dest the host): on_saga_promote registers the source (`GhostColliderRegistration`)
  + sends `GhostFlow::Spawn`; `feed_source_ghosts` streams `GhostFlow::Delta` (MsgClass::GhostDelta, monotone seq);
  the source `on_ghost_flow` writes into the retained ghost via `AuthorityCmd::GhostRefresh` (`SourceGhostMirror` =
  pure freshness/dedup, FG-2 single-truth). `emit_frames` widened to `simulates() | is_fed_ghost | is_retained_ghost`
  — the RETAINED source ghost SELF-EMITS its frozen last-Owned pose to fill the demote→Promote window (the feed
  CANNOT fill it — nothing is Owned then), the feed fills the post-Promote window. **The capstone stays SEAMLESS
  (`verify_no_vanish`, `max_absent_run==0`, `overlap_ticks>=1`) across the lengthened strict handoff.** Routing is
  DIRECT shard↔shard mesh (`push_flow` by NodeId, no gateway hop); no new libraries.
  **✅ LANDED 1d.5b.3c (band-exit teardown — the source-ghost lifecycle END; the .3b per-tick leak CLOSED):** the dest
  (owner) detects band-exit in `feed_source_ghosts` — the owned entity's distance from the crossing ANCHOR (captured at
  promote on `GhostNeighbor.anchor`; immutable bookkeeping, FG-2-clean — the live pose is still read from the `Dot`)
  leaves the seed-derived overlap band (`OverlapBand::for_motion(move_speed·dt)`, velocity-safe, NEW in
  `core/geometry.rs`) — and emits `GhostFlow::Despawn` (RELIABLE carrier; ⚠️ over the AT-MOST-ONCE prod mesh a lost
  Despawn LEAKS a permanent phantom — the dest deregisters same-pass + the source has no re-detection, so there is NO
  producer to re-send: a self-emitting frozen ghost (render) + a phantom collider forever. Tracked as the FOURTH
  instance of D-6 #1's producer-less-flow CLASS; the redelivering transport / a source staleness reaper is the cure) +
  DEREGISTERS the feed (`GhostColliderRegistration`). The source `on_ghost_flow` Despawn arm now TEARS DOWN: removes
  the `SourceGhostMirror` entry AND the retained ghost DOT (the source stops self-emitting + being a collider) —
  IDEMPOTENT + counted (`ghost_despawns` / `ghost_despawn_no_host` / dest `ghost_band_exits`). Strictly POST-release
  **[SUPERSEDED IN PART by Step 5 slice F, 2026-08-14 (minor 15):** the POSE FEED half of this machinery
  (`Spawn`+`Delta`+`refresh_source_ghost`+`SourceGhostMirror`+the fed-ghost emit) is DELETED — it was the §4u/§4v
  foreign-frame corruption's carrier. The take-over proof is the pose-free `GhostFlow::SpawnV2` (RETAINED —
  which also RETIRES the ⚠️ lost-Spawn instance of the producer-less class noted above); the retained ghost
  self-emits only while the Source hold is open, and bystanders' clients EVICT the leaver at hold closure via
  the minor-14 remove message (TTL expiry = the no-proof backstop). Band-exit Despawn + the anchor sweep are
  UNCHANGED and still owed the redelivering-transport cure for a lost Despawn — though its blast radius shrank:
  the leaked dot no longer self-emits (hold closed) and no longer strands a drawn phantom (the eviction already
  landed); the leak is now a silent inert dot + a collider skeleton until realm teardown.]**
  (the destroy edge is many per-tick steps out, so the entity walks well past the demote→promote→release handoff before
  exiting), so the RETAINED DOT is removed only at band exit — while the leaver's VANISH already happened at hold
  closure (slice F). The capstone (rewritten 2026-08-14, Stage-C audit :229 — its old `max_absent_run==0` "NO vanish
  across the band-exit teardown" claim had become unfalsifiable once the ghost stopped emitting at hold closure) is
  `p2_dod_band_exit_tears_down_the_retained_dot_after_the_leaver_vanished_at_hold_closure`: it asserts the Source hold
  is CLOSED at settle while the dot still stands (the vanish-at-hold-closure half), then the band-exit Despawn tears
  the dot down (`ghost_band_exits`/`ghost_despawns` exactly once each), with `max_absent_run==0` kept as a continuity
  floor on the REAL `DeliveredView`. The .3b feed-forever (dest feed-pass + source self-emit every tick per committed
  transfer) is CLOSED. Routing stays DIRECT shard↔shard mesh.
  **⚠️ INTERIM (.3c) — the motion-scaled stub band:** `OverlapBand::for_motion` sizes the band off per-tick travel
  (the stub has no realm-center/SOI geometry) and the anchor is the CROSSING pose (a local-boundary approximation).
  Production realm bands use `for_planet_soi`/`for_system_soi` anchored at the realm center (P4/P5 spatial geometry) —
  band-exit then has no anchor-at-crossing artifact. The teardown machinery is geometry-agnostic (only the band +
  anchor source change), so the swap is additive.
  **⚠️ OWED (.3c) — the orchestrator-side teardown gate:** the "refuse a Despawn that races a live saga" guard the spec
  named on the directory `in_transfer` field (DEAD — cleared at commit-CAS) CANNOT live on a `vd-sim` shard (it cannot
  see the orchestrator live-saga set; the dependency rule forbids sim→node — full-audit `wf_3fee0260` skeptics 4/4).
  The shard-LOCAL stand-in is STRUCTURAL: the source tears down ONLY a retained `Ghost` dot (a RE-OWNED `Owned` dot is
  refused — `remove_retained_ghost`), and the destroy-edge sizing keeps band-exit post-release. **⚠️ Correction
  (audit `wf_3b9eb7f0`):** the `in_transfer` lock does NOT by itself prevent a concurrent same-key saga POST-commit
  (`commit_cas` CLEARS it at the commit point — as this entry's own note above states) — so "one saga per key" was
  OVERSTATED as a lock guarantee. For the reaper-driven STANDING re-home it is now an ENFORCED invariant via
  `subject_has_live_saga` (the reaper cross-checks `runtime.sagas`, not just `in_transfer`); the GENERAL
  orchestrator-side one-saga-per-key gate (+ a harness one-saga-per-subject-key oracle) + ghost-lifecycle crash
  recovery remain owed at **Slice-2** / **D-6**. **⚠️ saga `(Demoting, DestDelivered)` latch arm is
  now PRODUCTION-DEAD** (with the relocated SubscriptionReady the dest sub/frames cannot exist while the saga is in
  Demoting — `DestDelivered` can only arrive in Promoting+); kept as a defensive/harness-only arm (annotated in
  `saga.rs`; its `an_early_delivery_in_demoting` test injects the event directly, so HR5 coverage holds).
  **⚠️ OWED at P8/P10 (realm-mobility re-drive — full-audit hardening `wf_fde6c77f`):** the dest `Promote` arm
  assumes the dest holds its realm. A `Promote` racing a realm SELF-FENCE (the realm reassigned/revoked while an
  entity transfer into it is in flight) is now a counted no-op (`promote_without_realm` — DEGRADE, never panic, the
  realm-owner guard mirrors every sibling handler), the saga re-drives. UNREACHABLE in P2 (no realm-revoke
  producer; the entity `in_transfer` lock keys on `DirectoryKey::Entity`, never `::Realm`, so it does not serialize a
  realm move against an in-flight entity transfer). The proper recovery (the saga `Promoting`-timeout re-drive
  producer + serializing realm moves vs in-flight entity transfers) co-lands with **D-3** (lease lifecycle) +
  **Slice-2** (the timeout/re-drive machinery), before P8/P10 multi-realm mobility.
  **✅ LANDED 1d.5b.3d (per-tick mid-flight AUTHORITY-UNIQUE — D-2 GREEN):** `verify_authority_unique` is now asserted
  EVERY tick across the transfer (capstone `p2_dod_authority_is_unique_every_mid_flight_tick`), not just post-quiesce.
  It gained ONE tight excuse — **W1**: the post-CAS, pre-demote `DirectoryDisagrees` window (the source still holds the
  subject `Owned` at the old fence while the directory records the dest) is excused via a monomorphic `excuse_w1`
  (`oracle.rs`) gated on FOUR conjuncts — a LIVE saga for the entity, `saga.source == holder`, the record names
  `Shard(saga.dest)`, and the held fence is STALE vs the record fence — reached ONLY after the `len == 1` guard, so it
  can NEVER mask a split-brain (unit-proven: `a_split_brain_during_a_live_saga_is_still_caught_never_excused`). The
  LIVE-saga ground truth is `SagaRuntimeRes::active_transfers() -> Vec<ActiveTransfer{subject,source,dest}>` surfaced on
  `InspectReport.active_transfers` (orchestrator-only). The W2 zero-Owned gap needed NO new excuse — the existing
  `in_flight_to_owner` path already returns Ok (the dest is `pending` during the gap; empirically `HeldNowhere` never
  fires) — and the two-Owned overlap is UNREACHABLE under strict demote-before-promote, so no uncoverable excuse was
  added. `verify_authority_settled` now also rejects a non-empty `active_transfers` (the W1 excuse is mid-flight only).
  2. **Cooperative in-memory freeze, contra the spec** — `transfer_protocol.md` §2.4 states "the freeze is enforced
     by the **fence**, not by cooperative in-memory state." The source freeze IS still cooperative in-memory
     (`self_fence_foreign_entity` does a local `Authority` flip to a RETAINED Ghost — since 1d.4b it KEEPS the dot,
     no `dots.remove` — with NO fence pushed to the gateway to drop stale source frames). 1d.5b.1's saga-pushed
     `Demote` drives the SAME in-memory flip via `on_saga_demote`; the fence-enforced (gateway frame-drop) freeze
     remains the owed item.
  3. **Crash double-hold** — an orchestrator/source crash AFTER `CasWon`+`OpenInputSlot` but BEFORE the source's
     next poll leaves a PERSISTENT stale source grant. The Tier-3 "abort to last-known directory owner" recovery
     CANNOT un-stick it (last-known owner is already the DEST). Within the already-deferred saga-durability gap
     (D-6 `PersistCheckpoint` stub / ROB-2) but WIDENED by promote-before-demote — flagged here, not silent.
  4. **Poll-best-effort, no re-drive, unreachable at `realm_recheck_interval==0`** — the source demote has NO
     durable `Demote` record and NO re-drive (a lost reply strands it; same class as D-29). `realm_recheck_interval`
     defaults to `0` (`stub.rs` `config()`, `harness::topology`), which DISABLES the source poll entirely — the
     1c.8 gate only works because `tests/src/lib.rs` sets the interval to `4` (both shards inherit it via
     `..stub_config()`, but it is the SOURCE poll that CLOSES the demote). A CAS-backed Entity transfer against
     an `interval==0` source would never demote (permanent two-holder). `interval>0` is therefore an undocumented
     transfer PREREQUISITE today; the proper saga-pushed `Demote` removes the dependency.
  5. **Logout-vs-transfer race (SF-1)** — `pending_grant_op` orders the `departing` (LeaseRevoke) arm BEFORE the
     granted-key poll arm, and `foreign_takeover_target` requires `!departing`, so a `granted && departing` dot is
     structurally blind to a foreign takeover and can strand. Inert today (no 1c.8 scenario issues `DetachSession`
     mid-transfer); owed a unit test once the second (logout) producer lands.
- **STILL OWED (proper, post-1d band/ghost slice) — TWO distinct pieces, NOT one predicate body-swap:**
  - **(a) the demote PREDICATE — ✅ LANDED IN 1d.5a, then SUPERSEDED by 1d.5b.1's single Promoting gate:** the
    1d.5a interim was a delivery-pass (`interim_demote_complete` → `demote_when_delivered_and_exited`) synthesizing
    a `DemoteComplete` when a STANDING server-side delivery watermark latched `LiveSaga.dest_delivered`. **1d.5b.1
    DELETED that pass + the `DemoteComplete` event + the `LiveSaga.dest_delivered` latch:** the SAME standing
    watermark (`gateway.rs` `Session.delivered: BTreeMap<SubId, frame_id>` advanced in `on_shard_frame`, the
    recomputed `every_observer_delivered` conjunction over the current dest observers — the set
    `subscribers_of(dest)` indexes off `by_session.subs`, NON-EMPTY required, anti-vacuous) now emits
    `DeliveredToObservers` → `SagaEvent::DestDelivered`, fed DIRECTLY into the FSM's `Promoting` release gate
    (`promoting_advance`, with `PromoteAck`); an early `DestDelivered` is latched in the FSM's `Demoting.dest_delivered`
    field and carried forward (never lost). HR1 gateway-internal, never a client ack. This delivery-gating is what
    closed the render vanish (above). A `(Promoting, Timeout)` / `(Demoting, Timeout)` self-re-emit arm is the
    re-drive LANDING PAD — but ⚠️ there is NO production `SagaEvent::Timeout` PRODUCER yet (no `now - since` deadline
    scan in the orchestrator; owed at Slice-2 — see the D-3/lease + saga-timeout items). So a production never-delivered
    Demoting saga currently PARKS (emits nothing), visible ONLY as growing `now - since` in the admin staleness view;
    the loud-fail-via-`step_until`-cap is a TEST-tier property, NOT a production guarantee. Do not read this arm as a
    live safety net until Slice-2 injects `Timeout` on a deadline.
    **✅ band-exit RESOLVED at 1d.5b.3c — as a SEPARATE post-release ghost-lifecycle event, NOT a saga release-gate
    term:** the earlier plan folded "entity left the source overlap band" into the demote/release predicate
    (`dest_delivered & band_exited`). 1d.5b.1 instead made release = `PromoteAck & DeliveredToObservers` (NO band term —
    gating release on band-exit would PARK the saga: an SOI-sized band is unsatisfiable in the stub, probe
    `wf_1b64b75e`). Band-exit now drives the retained Ghost's `Despawn` (the COLLIDER lifecycle), DISTINCT from sub
    release (the RENDER lifecycle): the sub releases fast on delivery; the ghost stays a collider until the entity
    leaves the band (`.3c` above). The exit is measured by the DEST on its OWNED pose vs the crossing anchor
    (`OverlapBand::update_membership`, hysteresis-only — NOT the swept `segment_shell_crossing`: the motion band is
    velocity-safe, 0.1 m/tick < the 0.2 m gap, so a body cannot tunnel the band in one tick and hysteresis alone is
    sufficient; the swept primitive stays covered in `core` for the future fast-body SOI case).
  - **(b) the source-side ORDERED, fence-enforced demote (a TEAR-OUT, NOT a body-swap):** replace the `stub.rs`
    `granted_key_poll_tick`/`self_fence_foreign_entity` POLL with a SAGA-pushed `Demote`/`DemoteAck` driving the
    per-entity `authority.rs` `Owned→Frozen→Ghost` FSM — the source flips to a **retained ghost-as-collider** (so
    players still collide across the boundary — a HARD requirement) and STOPS emitting because of the **fence**
    (lease-epoch pushed to the gateway), demote-before-promote, BEFORE the dest promotes.
    **⚠️ 1d.4b PROGRESS (part of (b) landed):** the `authority.rs` FSM is now ATTACHED and the source self-fence
    RETAINS the dot as a Ghost via `Owned→Frozen→Ghost` (no more `dots.remove`) — so the source ghost now EXISTS and
    STOPS emitting (`simulates()==false`). What (b) owed (UPDATED post-1d.5b.2): the SAGA-PUSH ✅ DONE (the demote is
    now driven by the saga's `Demote`/`DemoteAck` via `on_saga_demote`, 1d.5b.1; the poll `granted_key_poll_tick` is
    DELETED, 1d.5b.2); the ORDERING ⏳ owed-1d.5b.3 (the dest STILL autonomously promotes off `apply_crossing` — the
    strict relocation lands in .3); the **fence**-enforced freeze ⏳ still cooperative (no lease-epoch pushed to the
    gateway — P3); the GhostFlow collider FEED ⏳ owed-1d.5b.3 (the ghost exists but emits no delta). The **reshape-free
    promise (`wf_0ed2dc0c`) covers ONLY (a)** — (b)'s remainder is a re-architecture.
- **⚠️ 1d.1 LAYERING (stated honestly so the tear-out is not under-budgeted):** the 1d.1 pose crossing adds
  machinery BESIDE this interim, NOT a migration of it. (i) The source pose flush (`FlushSource`→`SourceFlushed`)
  is NET-NEW and read-only — it does NOT replace the source self-fence drop (in the 1c.8 interim that drop was
  POLL-discovered; post-1d.5b.2 it is the saga `Demote` driving `self_fence_foreign_entity` via `on_saga_demote`, the
  poll torn out). (ii) Because the dest adopts LATE under
  promote-before-demote (the crossing, emitted at CAS, races ahead of the dest's `OpenInputSlot`→`HeadRead`→flip),
  the dest BUFFERS the crossing (`PendingCrossings`) and drains it at the adopt flip — a 1c.8-model workaround for
  the adopt-ordering race, not the permanent design. (iii) The crossing is NOT gated on its dest ack: the SAGA
  reaches `Done` whether or not the crossing landed (the source self-fence is independent of the crossing). ⚠️
  CORRECTED (holistic audit `wf_ed40e95e`): this is STALE re: gateway visibility — since the 1d.5b.3b relocation
  moved the dest read-sub announce INTO `promote_apply` behind the crossing-landed gate (stub.rs:1460), the gateway's
  `DeliveredToObservers` watermark NOW depends on the crossing landing, so a crossing LOST over the at-most-once mesh
  parks `Promoting` forever (see D-6 #1's third producer-less phase). The saga FSM still reaches `Done` source-side,
  but the dest does not become Owned — the wedge is real over the prod transport, masked by the FaultFabric. The
  proper tear-out (b) SUBSUMES all three: the saga-pushed ordered `Demote` retains the source as a ghost until the
  dest is fully ready (pose INCLUDED, gated on the crossing ack), at which point the dest buffer AND the
  cooperative poll AND the net-new flush-beside-poll layering all retire into one fence-enforced handoff.
- **⚠️ 1d.3 AUDIT FOLLOW-UP (the e2e render-origin variant — `wf_c3e4f1b7` Finding 1):** the 1d.3 capstone
  `p2_dod_the_cross_shard_crossing_renders_at_the_dest_at_the_crossed_pose` ASSERTS the dest renders the crossed
  seed-7 pose (never the seed-8 origin-adopt default), but it CANNOT catch the render-flip being MISplaced to the
  adopt grant arm (`stub.rs` flip_grant Adopted) instead of `apply_crossing`: this fixture buffers the crossing and
  drains+flips it in the SAME tick BEFORE `emit_frames`, so a misplaced flip never emits a seed-8 frame. That
  property is held at the UNIT tier (`stub.rs::the_adopt_grant_flip_holds_authority_announces_the_sub_without_render_or_attach`)
  and the capstone now documents this honestly (negative-control (f) + docstring). The genuine e2e coverage — a
  capstone variant that DELAYS the crossing envelope past the adopt handshake (the crossing-AFTER-adopt ordering,
  where the misplaced flip WOULD emit the origin frame) — needs the D-2 reorder machinery and lands WITH the
  `verify_no_vanish`/`verify_pose_continuity` wiring here. Until then the unit test holds the line (a real
  render-origin regression still turns `cargo test --workspace` RED — vd-sim is a workspace member).
- **When / proper (UPDATED — substantially LANDED 1d.5a/1d.5b.1/1d.5b.2; see the LANDED blocks + the READER NOTE):**
  the source demote is now FENCE/SAGA-driven (the poll-dependent test was flipped — DELETED — in 1d.5b.2 when the saga
  `Demote` became the sole driver). The capstone flipped from the interim BOTH-SIDED pin (`overlap_ticks == 0` +
  `0 < vanish_gap <= MAX_INTERIM_VANISH_TICKS`) to the SEAMLESS gate (`verify_no_vanish`/`verify_pose_continuity` +
  `overlap_ticks >= 1` + `max_absent_run == 0`) at 1d.5a. What remains owed is the 1d.5b.3 back-half (strict ordering
  relocation + GhostFlow feed + band-exit + the mid-flight authority oracle).
- **Dependency:** Slice 1d (per-entity `Authority` attach so the dest owns + emits the post-commit entity) +
  the ghost-as-collider + band-instance + observer-delivery-watermark machinery (the band/ghost slice).
- **Source:** 1c design `wf_f3eae69e` + the demote refinement `wf_0ed2dc0c`; interim landed in Slice 1c.8;
  ordering-inversion honesty + tear-out scoping from the 1c.8 audit `wf_13656136`. **The LOCKED 1d.4/1d.5
  implementation decomposition (4 slices, adversarially verified `wf_f780b04c`) is
  `docs/design/slice_1d4_1d5_ordered_demote.md`** — note the 3 corrections it folds in (freeze is
  saga-gated not frame-fence-enforced; the dest promote is currently autonomous and needs a real saga
  `Promoting` state; no collision system exists yet so 1d.5b lands the ghost FEED+registration, response @P5).

### D-3 🟩 Lease lifecycle COMPLETE: all 6 slices LANDED (config + heartbeat + flap fault + CSCALE-1 tracker + expiry reaper + self-fence-before-grant on BOTH the shard Realm and the gateway Session) (design wf_24c1ecc5)
- **✅ LANDED (all 6 slices, all gate-green 100% Tier-A region+branch):**
  - **Slice 0 (5e73fdf) — config foundation:** `DirectoryTuning` + the lease-liveness knobs
    (`lease_renew_interval_ticks`, `min_renews_before_lapse`, `self_fence_grace_ticks`,
    `max_self_fence_grace_ticks`, `reaper_interval_ticks`, `recovery_grace_ticks`) + `LivenessTuning`
    (`n_consecutive_unreachable`, `unreachable_window_ticks`, `retry_delay_ticks_hint`), each with a
    `validate()` (the ordering chain, gated on an ACTIVE heartbeat so inert configs pass). ALL defaults INERT.
  - **Slice 1 (3907c93) — the heartbeat PRODUCER:** `OutboundBox::push_renewals` (ONE branchless generic
    shim, HR3); the shard renews its `Realm` + every granted, non-departing `Entity` and the gateway renews
    every Active `Session`, each on its own LOCAL cadence (`lease_renew_interval_ticks`, INERT at 0). So
    `lease_expires` IS now refreshed by a live producer.
  - **Slice 2 (ca756c7) — the harness FLAP fault:** `LinkPolicy.flap_until_tick` + `FaultFabric::flap` (a
    recoverable `NodeUnreachable` with the fresh-seq fix so the subject auto-redelivers) — the in-process
    stand-in for an io-prod blip. Consumed by the Slice-3 CSCALE-1 flap cell.
  - **Slice 3 (68ec2d2) — the CSCALE-1 tracker (CSCALE-1 CLOSED):** the insert-only / never-cleared
    `dead_participants` set is GONE — replaced by an evidence-gated, clearable `LivenessTracker` (a peer is
    confirmed dead only after `n_consecutive_unreachable` notices within `unreachable_window_ticks` with no
    intervening successful inbound; cleared on ANY inbound Wire at the top of `drive_sagas`). The DESTRUCTIVE
    dest-abandon is moved off the cheap `redrive_deadline_ticks` onto the LARGE `abort_deadline_ticks`,
    measured from a NEW per-saga RAM-only `dead_observed_since` (NOT `since`, which `scan_deadlines` re-arms
    every fire). `LivenessTuning` threaded through `OrchestratorConfig`→`with_tunings` (incl. the recover
    path `rehydrate`, re-audit fix) — prod default `n = 3` (the CSCALE-1 margin), dev/test `n = 1`
    (kill-equivalent). Proven by unit tests (all tracker branches + the recover-margin) + the e2e flap cell
    (a blip toward a HEALTHY dest lands the batch, zero abandon, non-vacuous). Re-audited DONE_NO_CRITICAL
    (`wf_4d2ca7ae`).
  - **Slice 4 (650911d) — the orchestrator expiry REAPER + CAP freeze:** finally CONSUMES
    `lease_expires`. `should_reap` = a 3-if CAP AND-gate (quiesce-elapsed AND lapsed AND confirmed-dead);
    `reap_lapsed_leases` runs INSIDE the D-6 group-commit barrier (before the directory reconcile) so a
    revoke is durable that tick (COMP-2, no kill-9 resurrection), once per `reaper_interval_ticks` (re-armed
    via `last_reap_tick`; INERT at 0). The D-37 boundary: FULLY revokes a dead `Session` key (client
    reconnects via its ResumeTicket) but LEAVES dead Realm/Entity/Ship for D-37 forward re-home; `revoke`
    refuses an `in_transfer`-locked key (a saga owns it). `liveness_quiesced_until` set on rehydrate =
    `ceiling + recovery_grace_ticks` (belt-and-suspenders atop the RAM-empty tracker — the primary CAP
    freeze). `admin_snapshot.leases` surfaces lapsed-pending leases (negative `ticks_remaining`) — never a
    silent wedge. Focused review DONE_NO_CRITICAL (`wf_31119adb`).
  - **Slice 5 (e528d26 shard + <gateway commit>) — self-fence-before-grant (the split-brain cure):** a
    holder that has gone un-CONFIRMED past `self_fence_grace_ticks` (a partition from the orchestrator —
    the reactive recheck reply never arrives) HARD-STOPS its OWN authority on the partition-surviving
    `local_tick` BEFORE the orchestrator's reassign window opens. The split-brain-safe timer is
    `local_tick - last_confirmed > grace`, keyed on the ROUND-TRIP confirmation (NOT the fire-and-forget
    renewal emit, which keeps flowing under partition); it matches the audited Slice-0 validate chain
    (`ttl < grace <= ttl + max`). **Shard (5a):** `RealmConfirmedAt` + `self_fence_lapsed_realm`
    (RealmAuthority=None + drop transients). **Gateway (5b):** `SessionPhase::SelfFenced` +
    `Session.confirmed_at` + the `HeadRead{Session}` recheck channel + `self_fence_lapsed_sessions` +
    a reactive arm (foreign/absent head). A SelfFenced session is excluded from input/frames/renew/
    recheck/re-drive; promotion to Active is AwaitingAttach-only so a `SessionAttached` straggler can
    never resurrect it (the review CRITICAL). DRY: ONE shared `lease_self_fence_due` predicate + ONE
    `due_this_tick` cadence guard (closes the holistic-audit DRY item) across shard + gateway. A node-side
    `validate_self_fence_cadence` (grace armed ⇒ recheck channel exists AND `grace >= 2*recheck`) rejects
    the mass-self-fence misconfig at boot (the review HIGH — the orchestrator can't see the node-side
    recheck knob). Reviewed `wf_15a14bb0` (CRITICAL straggler-resurrection + HIGH cadence-guard found,
    both fixed + re-verified SAFE).
- **Residual / forward (NON-blocking):** (a) LOW defense-in-depth (review `wf_31119adb`): an
  `unreachable_window_ticks < lease_ttl_ticks` orchestrator-side validate cross-check — distinct from the
  node-side cadence guard now landed. (b) A SelfFenced gateway Session lingers inert until `Bye` /
  the D-37/P3 ResumeTicket adoption re-homes it (the adoption path is unbuilt — D-37).
- **✅ CSCALE-1 CLOSED (Slice 3, `68ec2d2`) — was: whole-codebase audit `wf_2de9063f`, HIGH.** The
  BEFORE-state (now historical): the D-7d dead-resolution used a kill-only `Inbound::NodeUnreachable` as its
  dead-vs-slow stand-in, so in io-prod a SINGLE recoverable write blip (a 20s idle-reap / VXLAN drop of a LIVE
  peer; the writer auto-redials next frame) inserted the peer into an insert-only / NEVER-cleared
  `dead_participants` set, and a due `BatchHandoff` resolved to `DestUnreachable` → `EmitTransientAbandon` +
  Tombstone = irreversible loss of a HEALTHY in-flight batch (the cheap `redrive_deadline_ticks`, so a blip +
  a few ticks sufficed). FIXED: the evidence-gated `LivenessTracker` (N-consecutive + clear-on-ack) means a
  blip never confirms a healthy peer, and the destructive abandon now waits the LARGE `abort_deadline_ticks`
  from `dead_observed_since`. The e2e flap cell exercises the exact blip path the old code abandoned on. (The
  io-prod transport that PRODUCES the real blip is still owed — see DEFERRED.md D-6 transport-redelivery; the
  in-process flap fault is its deterministic stand-in.)
- **Where (residual — slices 4–5):** `crates/sim/src/directory.rs` (`renew`/`lease_expires` written; the
  expiry reaper that CONSUMES `lease_expires` is owed Slice 4); `crates/node/src/saga_runtime.rs` (the
  `LivenessTracker` is built + consumed by the BatchHandoff resolution; Slice 4 adds the directory reaper
  reading `is_confirmed_dead`, Slice 5 the proactive self-fence timer).
- **When / proper:** **before the P2 crash/stagger matrix and P3 chaos with real nodes / any multi-pod
  deploy.** FOUR parts: (1) ✅ a renewal heartbeat from each authority holder (gateway `Session`, shard
  `Realm`/`Entity`) — LANDED, Slice 1; (2) an orchestrator expiry sweep gated on **unreachable-confirmation**
  (lapsed lease ⇒ ownership loss ONLY when the owner is confirmed unreachable — an orchestrator outage
  freezes recovery, never mass-orphans) — owed, Slice 4; (3) CSCALE-1 hardening — N-consecutive
  `NodeUnreachable` within a window before the dead bit + **clear it on the next successful ack**, and gate
  the DESTRUCTIVE abandon behind the LARGE `abort_deadline_ticks` (not the cheap `redrive_deadline_ticks`) —
  owed, Slice 3; (4) ✅ a harness **"flap" fault** (a recoverable `NodeUnreachable`) — LANDED, Slice 2. The
  CSCALE-1 cure (3) is now testable RED via the landed flap fault (4); Slice 3 lands the cells + the tracker.
- **⚠️ SCALE (filed per the holistic audit `wf_9a986473`):** `scan_deadlines` is an O(live-sagas) full scan
  per tick (`saga_runtime.rs` — bounded by the deadline re-arm, like `views()`/`active_transfers()`), and the
  Slice-4 reaper adds an O(directory) full scan per `reaper_interval`. At MMO scale (thousands of in-flight
  sagas / leased keys) both want a deadline-ordered min-heap / per-node lease index instead of a full walk —
  a D-32-adjacent scale optimization, additive (no rework). Inert at P2/P3 in-process size. **WHEN: the redb
  backend / N-orchestrator (D-32) era**, alongside the directory-reconcile write-amp (DEFERRED.md D-6).
- **Dependency:** none structural (the plumbing — `renew`/`revoke`/`now`/`entries` + the heartbeat + the flap
  fault — now exists); needs the tracker rework (Slice 3) + the reaper (Slice 4) + self-fence (Slice 5).
  Gates D-37 (durable re-home).
- **Source:** whole-codebase audits `wf_43fea0dd` (XSI-1 / SCALE-A) + `wf_2de9063f` (CSCALE-1, HIGH) +
  `wf_9a986473` (the scan_deadlines scale entry + the D-3 ledger sync); design `wf_24c1ecc5` (the 6 slices).

### D-4 🟧 The reliable `EventMsg` client lane — consumer (a) AoI eviction LANDED (minor 14, 2026-08-14); consumer (b) P9 signals still owed
- **The lane as BUILT (2026-08-14; the paragraphs below are updated to the landed truth):** `EventMsg`
  rides INSIDE `ServerControlMsg::Event` on the existing reliable Control lane — no new `MsgClass` arm
  was needed (the original sketch below named one; the appended-variant pattern on the Control lane is
  the established client-facing shape and serves both consumers identically). TWO independent consumers
  share this ONE arm (built once — HR3/DRY):
  - **(a) Per-entity AoI eviction — LANDED (minor 14):** `EventMsg::EntityRemoved{entity, at}` routed
    to the client's per-entity eviction + resurrect guard. Retires the foundation-audit `scalability-1`
    root for the entity-leaves case. **Residuals, both slice F's:** (i) OBSERVER-LEAVES is not covered —
    when a player crosses away, nothing evicts the OLD realm's bystanders' figures from the LEAVER's
    own view (their rows stop at the one-space filter; the tracks freeze off-screen-frame) — the VU-6
    scene-reset re-stream or a sub-scoped eviction decision owns it; (ii) LOGOUT MID-CROSSING — ★CURED by slice F
    (2026-08-14): the retained ghost emits only while its Source hold is open, and the hold-TTL expiry
    fans the eviction the take-over proof never delivered — the strand ends in a vanish. Residue: the
    DEST-side `GhostColliderRegistration` entry leaks until realm teardown (its no-dot skip is counted
    per tick, `ghost_feed_skipped`) and the silent retained dot lingers as an inert return target — both
    bounded by the realm's own lifetime; the composited-subs detach clause remains owed beside them.
  - **(b) Cross-shard functional-block SIGNALS → client (P9) — still owed:** the signal system's gameplay
    events (damage/destroyed/notice and functional-block signal deliveries that surface to the player) ride
    the SAME `ServerControlMsg::Event` arm as appended `EventMsg` variants — opened causally after
    `SubscriptionOpened` (connection-plane X1).
    The cross-SHARD half of signals rides `InterShardFlow::Signal` (a RESERVED arm, P9) over the N-peer mesh +
    Galaxy Relay; this `EventMsg` arm is only the final shard→gateway→client leg. **No rewrite to land it** —
    the whole-codebase audit `wwg7ydm9y` confirmed signals are additive into the current seams (effect_class
    is exhaustive, the capability lattice already carves `SignalGraphCap`/`GalaxyRelay`).
- **Where (as landed):** `crates/wire/src/channels.rs` (`EventMsg` + `ServerControlMsg::Event`);
  `crates/wire/src/session_flow.rs` (`ShardToGateway::EntityRemoved`, the mesh leg, RETAINED durability —
  a producer-less reliable one-shot); `crates/sim/src/stub.rs` (`push_entity_removed` + the two
  permanent-stop emit sites + the loud no-lease suppression counter); `crates/connection-plane/src/gateway.rs`
  (`fan_entity_removed` — fence-checked, minor-gated, owner-session-skipped);
  `crates/client/src/{net,view}.rs` (routing, eviction, resurrect guard at the wire-taught tick rate).
  `BulkMsg{Blob}` remains declared-unroutable (P4/P6 terrain chunks).
- **When / proper:** consumer (a) DONE; **P9** adds the signal deliveries (consumer b) on the same arm.
- **★ESCALATED BY STEP 5 SLICE E (2026-08-13), RESOLVED (2026-08-14, owner picked the message):**
  slice E invalidated the deferral's precondition (the entity relay was the thing keeping bystander
  copies fresh; with it gone, every observed crossing stranded a FROZEN figure once the retained ghost
  despawned). **CONSUMER (a) IS LANDED — THE REMOVE MESSAGE (PROTO_MINOR 14):** the shard emits
  `ShardToGateway::EntityRemoved{realm_fence, entity, at}` at every permanent stop (the band-exit
  ghost despawn; the detach completing at the directory; NOT the pre-grant provisional drop, which
  never emitted); the gateway fans it to the shard's Active subscribers as
  `ServerControlMsg::Event(EventMsg::EntityRemoved{entity, at})` — per-subscriber fence-checked
  (`stale_removals_dropped`, its own counter: a wrongly-dropped removal strands a phantom) and
  minor-gated ≥ 14; the client evicts the track and arms a RESURRECT GUARD (`removed_at`: a straggler
  row stamped ≤ the removal is refused + counted; a strictly newer row is a genuine return and clears
  it; pruned on a derived window). The own-identity marker deliberately SURVIVES its own removal (the
  leaver's client legitimately receives one from the realm it just left — clearing would orphan the
  one-space filter forever). Proven end to end in the two-player chain scenario: the bystander's real
  client HOLDS the leaver's track at the crossing instant and it VANISHES through the production lane,
  and stays gone. The reshape of `EventMsg::EntityRemoved` (gaining `at`) was lawful uniquely because
  the enum had no producer since P1.5 — no negotiated wire ever carried its old shape.
  **Consumer (b) — P9 gameplay signal deliveries — rides this SAME `Event` arm as appended `EventMsg`
  variants; still owed at P9.** ★Slice F LANDED (2026-08-14): the leaver's eviction is fanned at hold
  closure (the SpawnV2 proof + the TTL expiry); the despawn-site emit stays for band-exit (idempotent
  on clients that already evicted).
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

### D-6 🟧 Durable saga WAL: S0–S5 + the redb backend (Slice P3-PERSIST-1: C1/C2 + Slice D α/β/γ/δ) LANDED — orchestrator crash-durable & SIGKILL-mid-fsync-proven; the redelivering transport is 🟩 (R-3'→R-4e, R-5 capstone proves producer-less at-least-once for a source-that-stays-up); only the deploy preconditions (#1 AwaitAdopt SOURCE-CRASH egress = M3+L5, #3 WAL version+Tombstone, durable-outbox, durable-root allow-list) remain OWED
- **▶ Slice P3-PERSIST-1 — the redb backend (design `wf_83d5a428`, judge-panel of 4; user-decided: Store A redb
  now + single-file/split-ready-seam):** ONE generic `RedbStore` behind the frozen `sim::io::Store` seam in
  `crates/io-prod/src/store.rs`, TARGETING the ORCHESTRATOR (Store A: directory + saga WAL + clock ceiling — the
  the bin wiring landed in Slice D-gamma — the prod orchestrator now opens `RedbStore` from `VD_STORE_PATH`; the
  shard per-RealmId Store B (P6/P7) reuses the SAME type verbatim (variance is DATA — path + keyspace — never CODE,
  HR3). Sub-slices: **✅ C1 LANDED (`63b637d`, + fail-loud hardening from audit `wf_66cb8f06`)** — the SYNCHRONOUS
  durable backend (commit() = one redb WriteTransaction + inline fsync; 6 durability tests across a real reopen; redb
  2.6.3 into the lock, the keep-list Store lib's first consumer; NOT yet wired → inline fsync blocks no tick).
  **FAIL-LOUD ADAPTER CONTRACT** (the infallible `Store` seam must never degrade a redb fault to a silent wrong
  answer): a READ fault in `scan`/`is_empty` PANICS (a silent empty would make `rehydrate` misread a faulting
  non-empty store as GENESIS = clock-reset + orphaned-WAL data loss — the re-audit `wf_66cb8f06` HIGH, now cured); a
  WRITE fault in `commit` is ALL-OR-NOTHING + RETAINS the staged batch (a per-key apply error aborts the txn, never a
  partial fsync — the re-audit MEDIUM, matching MemStore's all-or-nothing fold). A refusal is never a loss.
  **✅ C2 LANDED (design `wf_caad488e`, judge-panel 9/9/9; user-decided approach A for PvP: see below).** The OFF-TICK
  fsync so the sim thread never blocks on disk. APPROACH A = COMMIT-BLOCKS-ON-PRIOR (depth-1
  pipeline): a single named `vd-store-writer` thread + a BOUNDED crossbeam channel; `commit()` BLOCKS until the
  PREVIOUS batch is durable (≈0 wait at 50Hz; correct counted back-pressure under a disk stall — durability over
  liveness), then `mem::take`s the staged map + assigns a monotone seq + sends `(seq, batch)` + RETURNS (never touches
  disk on the sim thread); the writer `recv()` + `try_recv()` drain-COALESCES into ONE redb
  WriteTransaction + fsync, then bumps `last_durable: Arc<AtomicU64>` (Release) + notifies a condvar. **IMPL
  REFINEMENT (caught in C2):** under depth-1 the channel can NEVER fill (block-on-prior throttles before a 2nd batch
  submits), so the design's channel-Full back-pressure path was DEAD — block-on-prior ITSELF is the back-pressure
  (counted: `DurabilityHandle::backpressure_stalls`, incremented when a `commit` actually waits); the drain-coalesce
  loop is kept only as the seam for a future depth>1 mode (vestigial at depth-1). The block uses a `Mutex`+`Condvar`
  the writer notifies (park, not hot-spin); the `AtomicU64` stays the lock-free source of truth. Depth-1 is a C2
  INVARIANT: it bounds crash-loss to ≤1 batch. **⚠️ CORRECTION (review `wf_186fc41d`):** the earlier claim that
  block-on-prior makes the per-tick reconcile scan read "exactly T-1 by construction" is FALSE — the OFF-TICK writer
  RACES the reconcile's redb `begin_read` (which sees only PRE-call commits); block-on-prior waits for S_{i-1}, so the
  tick-(i+1) reconcile scan may observe S_{i-1} OR S_{i-2} depending on the writer's async progress, and a STALE scan
  that misses a row resurrects the COMP-2 zombie row on recover. Therefore the INCREMENTAL per-mutation reconcile
  (precondition #2) is CORRECTNESS-load-bearing (it stages each revoke's delete at revoke-time with NO read-back, so
  the race never arms) and was a **HARD Slice-D-BLOCKING precondition of the RedbStore swap — NOT the write-amp-only
  optimization it was framed as.** ✅ RESOLVED: D-alpha (commit `02db31d`) landed the incremental per-mutation
  reconcile (no read-back), so the race never arms — and the orchestrator bin now wires `RedbStore` (D-gamma,
  `orchestrator.rs` `RedbStore::open`), no longer MemStore. The 4/5 boot-only scan consumers (rehydrate) are
  unaffected — a fresh open reads a quiescent redb. (Alternatives to #2 if ever
  needed: flush/watermark-gate the scan, or reconcile from a pure-RAM directory snapshot with no read-back.) The
  durability watermark reaches the bin via a SIDECAR `DurabilityHandle` returned
  ALONGSIDE the `Box<dyn Store>` from `open()` — the FROZEN infallible `sim::io::Store` seam + `commit()`'s `()`
  signature stay BYTE-FOR-BYTE unchanged (MemStore + the sole caller + the deterministic harness churn ZERO). `Drop`
  joins the writer (flushes pending — graceful shutdown loses nothing). `flush_blocking()` + a `StoreTuning {
  writer_channel_depth }` (no magic numbers). Crash proof (all 4 SIGKILL cases SAFE): an effect leaves ONLY after the
  bin observes `last_durable >= seq_T` (Variant-A deferred-by-one-tick flush, Slice D), and `last_durable` bumps ONLY
  after the fsync (Release→Acquire happens-before) — so every shipped effect is durable-justified; the converse is the
  safe direction (recovery re-derives idempotently). C2 ships the PRIMITIVE + accessors + 8 DETERMINISTIC tests (the
  adapted durability/contract tests with `flush_blocking`, the watermark-advances-per-batch test, and the
  Drop-joins-the-writer-and-flushes-pending graceful-shutdown test). **Owed tests (need a `#[cfg(test)]` writer-pause
  hook, mirroring io-prod's `BridgeControl`):** the explicit block-on-prior-depth + backpressure-count + the
  in-process fault-injection crash analog (drop the in-flight batch = crash case 2/3) — block-on-prior makes those
  hard to trigger deterministically without a pause gate; the AUTHORITATIVE crash proof is the Slice-D process-tier
  real-SIGKILL cell anyway. **✅ Writer-fault HARDENING LANDED (review `wf_186fc41d` findings 1/2/4):** the writer
  RETRIES a failed fsync `WRITER_FSYNC_MAX_RETRIES`x (it still HOLDS `merged` — a TRANSIENT blip recovers losing
  nothing), then EXITS on a permanent fault; `wait_durable_through` now `wait_timeout`s + detects a DEAD writer
  (`JoinHandle::is_finished`) → PANICS LOUD rather than silently hang the synchronous orchestrator thread; the commit
  retain arms now log LOUD (no silent retain). **Still-owed refinement: a durable OUTBOX** — on a PERMANENT fsync
  give-up the consumed batch CONTENT is dropped (a regression vs C1's staged-RETAIN); the fix is a WAL-of-the-batch
  (a mere in-memory retry has nothing to retry once `commit()` `mem::took` + sent). Not C2-blocking: recovery
  rehydrates the last durable state (one all-or-nothing txn, no split-brain) + the Slice-2a producer re-drives
  idempotently — the outbox only AVOIDS that re-drive. **PvP rationale:** the orchestrator is
  NOT the combat hot path (hit-reg is shard+gateway, P5/P11) — it is only on the TRANSFER/authority path; the
  PvP-critical property is never duplicating/wedging a player mid-transfer, which A guarantees by construction; deeper
  pipelines (B/C) trade durability for throughput the batch-bounded single orchestrator does not need (the real
  scale lever is the per-Realm PARALLEL fsync pool for Store B, already designed). **Slice D (design `wf_2d4ab8d4`,
  decision A — renew participates) is wiring the durable bin in 4 gate-green sub-slices:**
  - **✅ D-alpha LANDED (`02db31d`)** — precondition #2, the INCREMENTAL directory reconcile. `DirectoryCore` carries
    a per-tick `dirty: BTreeMap<DirectoryKey, Option<OwnerRecord>>` delta set (every actual-row-change arm of
    grant/renew/revoke/lock/commit/abort_cas/abort_clear stages `Some`=PUT / `None`=DELETE; no-change arms stage
    nothing; within-tick collapse free); the group-commit barrier drains `take_dirty()` straight into the same
    `commit()`, REPLACING the O(directory) delete-all-then-put-current scan. This CURES the COMP-2 race the C2 review
    surfaced: the reconcile no longer READS the store back, so the off-tick writer can never race a reconcile read
    (the prior "scan reads T-1" coupling is gone). 14 directory unit pins + a saga_runtime DIFFERENTIAL ORACLE
    (incremental == full, byte-for-byte, across PUT-new/refresh/replace/DELETE + a kill-9 rebuild). 100% Tier-A.
  - **✅ D-beta LANDED (`de911e1`)** — split `step_tick` into `run_schedule()` (stages+submits the durable batch) +
    `flush_outbox(TickPrologue)` (sends the egress); `step_tick` is a thin back-to-back wrapper (harness/MemStore
    path BYTE-FOR-BYTE unchanged). The seam the Variant-A deferred-flush needs. 100% Tier-A.
  - **✅ D-gamma LANDED** — wire `RedbStore` into the orchestrator bin. `VD_STORE_PATH` REQUIRED (no in-memory
    fallback) + HR1 persistent-volume guard (a `/tmp`/`$TMPDIR` path is a HARD boot reject unless the explicit
    dev/test `VD_STORE_EPHEMERAL_OK` opt-in is set — `orchestrator_env` sets it for dev clusters + the process
    tests, whose stores live in the slot work dir reaped by `down`). The PARKED-FLUSH loop: defer tick T's outbox
    until batch T is durable (`DurabilityHandle::wait_durable_through` parks on the writer notify, fails LOUD if the
    writer died) — persist-before-effect, the wait IS the disk-stall back-pressure (~0 at 50Hz). `DurabilityHandle`
    gained `last_submitted()` / `writer_alive()` / `wait_durable_through()` (the shared `park_until_durable` core +
    a `writer_alive` flag the writer clears on its single exit). RedbStore≡MemStore differential (the RedbStore arm
    of the oracle: faithful Store ⇒ incremental==full ON REDB) + a handle unit test; process tier (process_parity /
    dev_cluster_smoke / client_load) GREEN on the real RedbStore-backed orchestrator over QUIC. The boot-warn is
    NARROWED to the one remaining precondition (at-most-once transport). io-prod Tier-B; Tier-A still 100%.
    **3-lens focused review (persist-before-effect / concurrency / env-robustness) — NO CRITICAL/HIGH in the
    ordering or concurrency (lost-wakeup + graceful-shutdown false-panic both REFUTED); fixes landed:** the writer
    now publishes death on ANY exit incl. a panic-unwind (`WriterExitSignal` Drop-guard) and `commit` FAILS LOUD
    (panic) on a disconnected channel instead of silently retaining (closing a latent persist-before-effect hole
    when the writer dies caught-up); `VD_STORE_EPHEMERAL_OK` is STRICT-parsed (a present-but-unrecognized value is
    a loud error, never fail-open); the ephemeral guard CANONICALIZES (collapses `/tmp`→`/private/tmp`,
    `/var`→`/private/var`) + rejects `/private/tmp`; the bin-side flush-gate stall now counts in
    `backpressure_stalls()`; the Clock-key/seq invariant + shutdown-relies-on-re-drive are documented in the loop.
  - **✅ D-delta LANDED** — the process-tier SIGKILL-mid-fsync crash proof. A content-keyed writer-pause hook
    (`store-test-hooks` feature, ABSENT from release; `StoreTuning.pause_on_key_prefix` + a marker file written
    by the writer thread itself — the honest, decoupled "submitted-but-pre-fsync window is open" signal, since
    the bin loop BLOCKS on the persist-before-effect gate the instant the sentinel batch fails to become durable
    and so cannot signal). The orchestrator (under the feature, `VD_STORE_TEST_SENTINEL_SEED`) plants a distinct
    realm grant B once the shard's realm A is already in a PRIOR batch ⇒ block-on-prior makes A durable BEFORE
    B is submitted; the writer parks on B pre-fsync. `orchestrator_crash` SIGKILLs in that window
    (`Child::kill`+`wait`, no zombie race) and proves on restart: A present, B LOST (the ≤1-batch claim), clock
    resumes FORWARD, no zombie saga / orphan lock — plus the ANTI-THEATER twin (the same `cluster_bootstrapped`
    predicate is FALSE on a fresh store). `boot_guard` (no feature) locks the HR1 ephemeral-store guard's reject
    AND accept arms (closes the review LOW). `directory_store_key` shim (vd-node) gives the test the exact
    barrier key-bytes (no drift; covered by a Tier-A unit test). `just orch-crash` recipe + `orch-crash-cov`
    (`%c` continuous-mode merge for the SIGKILLed child) + `store-test-hooks` in `lint-combos`, `orch-crash`
    wired into `gate`. io-prod/bins Tier-B; Tier-A still 100%. **Slice D COMPLETE.**
  - **✅ D-6 #1 PARTIAL FLIP (R-4e4 / R-5 `mesh_under_loss.rs`, `2123c19..`): the TRANSPORT-LAYER redelivery is 🟩 GREEN.**
    The redelivering `MeshTransport` (R-3' RX ledger + per-lane replay + R-4a idle-after-blip timer + R-4b bounded
    retry + R-4d SendShed) is now proven AT-LEAST-ONCE across a blip for a PRODUCER-LESS reliable flow whose SOURCE
    STAYS UP — the R-5 capstone drives a real `GhostFlow::Despawn` (band-exit, no saga, no `scan_deadlines` re-driver)
    across a `drop_connections` blip + idle and asserts B receives it EXACTLY ONCE via the timer alone, the envelope
    round-trips byte-exact, `gap_drop==0`, ZERO `NodeUnreachable` (blip≠death), the endpoint survives. This subsumes
    the GhostReliable Despawn, the durable `EmitCrossing` (D-6 3rd phase), and the D-37 re-home (cured in 2d) FOR THE
    SOURCE-STAYS-UP case. **The `MeshTransport` is NO LONGER "at-most-once" — the boot-warn premise is retired for the
    source-up path.**
  - **⚠️ SUPERSEDED-IN-PART by R-6d3c (see the LANDED block below, ~line 1697):** the saga half of #1 below is DONE —
    the `(BatchHandoff, SourceUnreachable)` arm was SPLIT by phase (post-adopt self-promote; `AwaitAdopt` +
    `SourceUnreachablePreAdopt` → discard-to-dest + count), the dest `on_transient_discard` poisons a late replay so no
    orphan strands, and the saga.rs comment was corrected. So "the self-promote matches ANY phase via `..`" and "the
    comment is INACCURATE for AwaitAdopt" (in the paragraph below) are HISTORICAL — do NOT read them as live bugs. What
    remains CA-1/L5-gated is only the confirm-dead TRIGGER (the AwaitAdopt re-solicit egress that MAKES
    `is_confirmed_dead(source)` reachable) + the lost-discard reliability.
  - **✅ CA-1 S3/S4 LANDED (spec `scripts/ca1_s3s4_perfect_guarantee_spec.md`) — the over-discard residual is CLOSED
    (the achievable HARD guarantee):** the `AwaitAdopt` re-solicit egress now exists (the new
    `InterShardFlow::ReSolicitBatch` arm, discriminant 16 → 17-arm set, `RE_SOLICIT_STEP=18`; emitted by the FSM's
    `(BatchHandoff{AwaitAdopt}, Timeout) => EmitReSolicit` arm every re-drive — the phase's first orch→source egress,
    so a dead source's send fails and `is_confirmed_dead(source)` becomes reachable). The over-discard hole the
    R-6d4 residual named is closed by the DECOMPOSED cure: (a) SAFETY — a monotone `dest_adopted` latch on `LiveSaga`,
    set in a PRE-SCAN inbox pass (`latch_adopted_from_inbox`, BEFORE `scan_deadlines` — the intra-tick ordering that
    makes it hard: latching in `deliver`, inside the post-scan drain, would be one tick too late on the exact
    budget-maturity tick). In `rehome_event_for` the latch drives a `defer_to_dest` gate: once the dest has adopted an
    `AwaitAdopt` batch it is the HOLDER-OF-RECORD, so its fate decides — the source-death resolutions
    (discard-if-never-adopted / self-promote-if-post-adopt) apply ONLY when `!defer_to_dest`; an adopted batch defers
    to the dest-death ladder (dest alive → re-drive + let the drain advance the phase; dest dead → the budget-gated
    `DestUnreachable` abandon+tombstone). **The post-impl review (`wf_1eb86848`) caught a REGRESSION in the first cut —
    a bare `!dest_adopted` guard let the source-death arm preempt the dest-death terminal, WEDGING the both-dead
    double-crash (dest adopted-then-crashed + source dead) into a perpetual `Timeout`→`EmitReSolicit` loop (pre-CA-1-S3
    that case resolved cleanly via the dest-dead `else if`, since the source trigger was inert). The `defer_to_dest`
    restructure fixes it: the double-crash now terminates via `DestUnreachable` (admin-visible via
    `dest_unreachable_resolutions`), the pre-change terminal restored.** (b) LIVENESS — the DEST re-emits `BatchAdopted`
    every tick it holds `Arriving` (`redrive_pending_adoptions`, gated purely on Arriving-presence, one ack per DISTINCT
    batch), so a transiently lost ack never strands the latch; (c) DETECTION — the `ReSolicitBatch` probe (above). A
    live source no-ops the probe (counted). This chose the "check dest-adopted" cure over the `abort_deadline_ticks >
    ack-bound` numeric gate (the perfect guarantee, not a bounded margin). **STILL OWED (deploy-bundle, ledgered — NOT
    reachable-now blockers):**
    (1) the ABSOLUTE-ZERO refinement — a sustained dest→orch shed of EVERY `BatchAdopted` across the whole abort
    window still over-discards (the orch cannot prove adoption without receiving one ack; in that corner the dest is
    anyway likely `is_confirmed_dead(dest)` → the dest-dead arm fires instead). Closing it needs a NON-SHEDDABLE ack
    (a transport priority/reservation lane so the adoption ack can never be shed under R-4b overload) — a bigger
    transport change, owed with the deploy bundle; (2) the permanent-orch-death TERMINATION of the dest re-drive
    depends on `self_fence_grace_ticks > 0` (the self-fence drops an orphaned `Arriving` item → the re-drive stops).
    **The SHIPPED prod default is `0` (shard.rs:49, `VD_SELF_FENCE_GRACE`; `validate_self_fence_cadence` checks only the
    operator-supplied value — nothing forces `>0`), so in the default config an orphaned `Arriving` batch re-drives
    `BatchAdopted` to a dead/absent orch every tick INDEFINITELY** (finite test rigs bound it by run length; the
    double-crash wedge fix above removes the compounding orch-side loop). This is a COUPLED cloud prerequisite, not a
    free-standing nice-to-have: the `grace > 0` boot assertion must land WITH the split-brain lease-timing coherence
    (`lease_ttl < grace <= lease_ttl + max`) on the deploy config validation — NOT a half-measure now.
    **[Cloud-ready k3d Slice 2 — CLOSED for the cloud profile]** `VD_BUILD=cloud` now makes `resolve_d3`
    (`io-prod/src/boot.rs`) DERIVE an ACTIVE D-3 set (`DirectoryTuning::cloud` + `LivenessTuning::cloud`) AND REJECT
    re-zeroing it, so the "shipped default is 0 / boots green with no split-brain protection" hole is gone whenever
    the profile is cloud. **The split-brain reassign TIMING was ALSO fixed here** (an adversarial post-impl review +
    an 8-agent design workflow, `wf_e392275a`/`wf_692089c5`; user chose "Strong-AND"): the pre-existing D-3 mechanism
    the cloud profile ACTIVATES had a verified ~ttl→grace two-holder window because `should_reap` reassigned a lapsed
    key at `lease_expires` (=ttl) while a holder self-fences at `grace > ttl` — the `+max` term was dead code. THE FIX:
    (a) `should_reap` now gates on `now > lease_expires + max_self_fence_grace_ticks` (the `+max` is load-bearing) AND
    on a PERSISTENT (monotone-latched) confirmed-dead signal — reassign happens only after the holder has PROVABLY
    self-fenced (zero zombie window), and the latch (`LivenessTracker::is_latched_dead`, cleared on ack, empty on
    rehydrate) survives the redial-backoff pulse so the ttl+max deadline never orphans; (b) `DirectoryTuning::cloud`
    sizes `max = THETA_MAX*grace + M - ttl = 5·hz` so the reassign horizon `ttl+max` STRICTLY outlasts even a
    `THETA_MAX`(=2)-CPU-throttled holder's self-fence (`THETA_MAX*grace < ttl+max`, `const`-asserted at hz∈{10,20,50}
    + `validate()` re-checked); (c) the earlier `SKEW_FACTOR`-in-`unreachable_window` mechanism was INERT (the window
    never enters `should_reap`) and is DROPPED — the window is now just `run_spread.max(n·hint)`. The node bins read
    the derived grace/recheck via `vd_bins::resolve_node_d3` (the old direct `VD_*` reads + inline
    `validate_self_fence_cadence` retired into the resolver). Findings 3/4/5 from the review also fixed:
    `VD_BOOT_DURABLE_ROOT` (the M3 incarnation ledger) is now force-validated in cloud (was fail-open on an emptyDir);
    the durable-root temp check rejects an ANCESTOR-of-temp root too; the DevTest orchestrator keeps its prod-safe
    `n=3` confirm-dead (was silently regressing to `n=1`). **DEPLOYMENT REQUIREMENTS (USER-RATIFY):** the orchestrator
    pod must run at Guaranteed QoS (a CPU reservation — the non-throttled reference frame); `THETA_MAX=2` is a policy
    const to RATIFY against a throttled-pod load test (Slice 5/6); the failover latency for a dead shard is now the
    full `ttl+max` = 7 s @50Hz (the Strong-AND price of the zero-zombie guarantee vs the OR fast-path's ~1.2 s) —
    shrink `ttl` to reduce it while preserving the margin. **MARGIN-BUDGET assumption (post-impl review F1, MEDIUM,
    NOT a shipped defect — the LAN config holds with wide headroom):** the enforced `THETA_MAX*grace < ttl+max` folds
    THREE terms into its `hz` (1 s @50Hz) margin: self-fence GRANULARITY (`+THETA_MAX`, the holder fences at
    `grace+1`), the renew/recheck ANCHOR DECOUPLING (`+renew_interval` = `hz/2`, since `lease_expires` tracks the
    reply-less renew while `confirmed` tracks the recheck round-trip), and WALL-CLOCK LATENCY (RTT + renew jitter,
    un-ticked). The tight DETERMINISTIC bound is `THETA_MAX*(grace+1) + renew_interval <= ttl+max` (@50Hz `327 <= 350`,
    leaving 23 ticks ≈ 0.46 s one-way latency headroom — ample on a k3d LAN). It is DOCUMENTED (the `THETA_MAX` doc-
    comment) but NOT yet code-enforced in `validate` (the shipped derivation satisfies it by construction, so no
    footgun today); PROMOTE it into `validate` + RATIFY the wall-clock latency budget with the same throttled-pod load
    test if a deployment ever tightens `max` toward `THETA_MAX*grace`. Also (F2, LOW): `VD_LEASE_TTL` moved from
    required→defaulted in the shared `resolve_d3` (a shard/gateway never set it + cloud derives it) — inert in DevTest,
    every rig sets it explicitly. **The RESIDUAL is FAIL-OPEN, not fail-safe:** a k8s manifest
    that OMITS `VD_BUILD=cloud` boots DevTest (inert D-3, NO split-brain protection) silently — so the **Slice-4
    manifest DoD** is that EVERY server pod sets `VD_BUILD=cloud` + BOTH durable roots (each footgun-arm is proven
    fail-loud in a real binary by `crates/bins/tests/cloud_preflight_process.rs`, incl. the green-boot positive
    control). DevTest stays intentionally inert (in-process rigs + the loopback process tier have no split-brain to
    protect against); (3) the real-cloud
    k3d CrashLoop/reschedule e2e that exercises the `ReSolicitBatch` send-failure → confirm-dead path across a NAT'd
    reschedule (driven by CA-1 `update_peer_addr`); (4) SCALE refinements (idempotent + sheddable today, non-blocking):
    the dest re-drive is UNCONDITIONAL every `Arriving` tick, so it emits a duplicate `BatchAdopted` for the whole
    (brief) NORMAL happy-path handoff window too — interval/first-seen gating would trim this steady-state orch-bound
    traffic while keeping the re-drive present across the maturity tick; and `latch_adopted_from_inbox` is a SECOND full
    postcard decode of every Saga-class inbound each tick (the drain re-decodes) — a shared adopted-this-tick
    `BTreeSet<TransferId>` populated once would remove the duplicate decode. Both are pure common-path overhead worth
    trimming at MMO scale, neither a correctness issue. **Also ledgered (prose):** the spec's "a single lost ack cannot
    strand the latch" is precise only POST-first-ack (the monotone latch needs one delivered ack EVER); the
    first-adopt-exactly-at-maturity-with-that-ack-lost corner is a strict sub-case of residual (1).
  - **[Cloud-ready k3d Slice 3 — k8s liveness/readiness probes | LANDED + 3× adversarial review folded]** ONE
    `probe_router` (`/healthz` + `/readyz`, bare status codes, HR3) served by ALL three bins, SEPARATE from the
    auth-gated `admin_router` (HR1: an unauth kubelet `httpGet` presents no token; the bare code leaks no cluster
    state). The PURE decision surface is Tier-A `vd_node::health` (100% region+branch — `is_live`,
    `is_confirmed_fresh`, `shard_authority_ready`, `shard_ready`, `gateway_ready`, `orch_ready`, `health_report`,
    `ProbeTuning`); the axum GLUE + the ONE `Instant::now` staleness ref is Tier-B `io-prod::probe` (`HealthSource`
    trait injection = the per-role polymorphism, never a match-on-kind). LIVENESS is drain-safe (`draining | is_live`:
    a SIGTERM-edge park stops the heartbeat but reads LIVE so kubelet never SIGKILLs mid-fsync) + wedge-detecting
    (a frozen tick loop's stale heartbeat → 503 → restart). READINESS de-routes booting/clock-unsynced/
    partitioned-self-fenced/session-full/draining. Orchestrator readiness is its OWN serving bit (NOT
    `cluster_bootstrapped()` — that would cold-start-DEADLOCK). SIGTERM edge de-routes `/readyz` BEFORE the drain +
    `VD_SHUTDOWN_LINGER_MS` preStop-park. Ports via `devproto` `PROBE_*_OFFSET` (4/5/6); addrs emitted into the
    `cluster.env` + `vd-slot` contracts. Proven by `crates/bins/tests/probe_endpoints.rs` (converge-ready,
    orch-alone-ready bootstrap, sigterm-de-route-stays-live, partitioned-shard-notready-stays-live). **3× adversarial
    review (`wf_40ac2919`) folded 4 real defects:** (CRIT) `publish_tick` was a non-atomic load-then-`store` that
    could CLOBBER a concurrently-set `draining` (signal task on another thread) → un-drain a terminating pod as its
    FINAL state → routed live traffic to a dying pod + 503 mid-fsync — FIXED to `ArcSwap::rcu` (compare-and-retry ⇒
    `draining` monotone), + a 4000-round race regression test; (HIGH) armed `shard_authority_ready` IGNORED `held`, so
    a never-granted shard read Ready off the default `RealmConfirmedAt(0)` for its first `grace` ticks if clock-sync
    beat the grant — FIXED to `held & (is_confirmed_fresh | grace==0)` (now a true dual of `lease_self_fence_due`);
    (MED) `enforce_cloud_preflight` had NO `VD_PROBE_ADDR` requirement (the one cloud footgun that failed OPEN — a
    probe-less pod boots always-healthy) — FIXED with `CloudProfileError::MissingProbeAddr` + negative test; (LOW)
    probe addrs now in the `cluster.env`/`vd-slot` contracts so S4 tooling never hand-derives probe ports.
  - **[Cloud-ready k3d Slice 4a — GATEWAY partition-aware readiness | LANDED + review folded]** Closed the S3
    NAMED blocker (gateway `gateway_ready` was partition-BLIND). `gateway_ready` gained a 4th arg `sessions_live`
    = `vd_node::health::gateway_sessions_live(GatewaySessions::freshest_session_confirmed(), local_tick, grace)`,
    the session-servicing analogue of the shard's `shard_authority_ready`/`RealmConfirmedAt`: a total
    gateway↔orchestrator(directory) partition freezes every session's `confirmed_at`, so the freshest going stale
    ⇒ the gateway can renew NO session lease ⇒ /readyz de-route. Inert `grace==0` (DevTest) ⇒ always live
    (byte-identical). **The adversarial review (`wf_8261a0ea`) caught a real HIGH I FIXED:** an `Active`-only max
    was DEAD for the exact total partition it targeted — the proactive self-fence (`self_fence_lapsed_sessions`)
    fires at the SAME `local_tick - confirmed > grace` threshold EARLIER in the same `step_tick`, flipping every
    session `Active → SelfFenced` before readiness samples, so the detector read `None` (falsely live). CURE:
    `freshest_session_confirmed` maxes `confirmed_at` over `{Active ∪ SelfFenced}` (SelfFenced retains its frozen
    confirmed = the partition evidence; a fresh Active still dominates so a healthy/partial-partition gateway stays
    Ready), proven by the REAL-schedule rig test asserting `Some(1)` post-self-fence (not a hand-fed value).
    GENERAL lesson: a readiness gate sampled after `step_tick` that shares a threshold with a self-fence must read
    the POST-transition state. **RESIDUAL (acknowledged, ledgered):** the zero-session / pre-`Active`-only blind
    spot — a gateway that never had an Active session (only in-flight logins, `confirmed_at`=0 sentinel excluded)
    rests on `clock_synced` (orch→gw only; does NOT prove gw→orch mint capability). Proper cure = a
    session-independent gateway↔orch directory heartbeat (the shard has `RealmConfirmedAt` even with 0 players; the
    gateway lacks a standing key to recheck). Deferred past S4.
  - **[Cloud-ready k3d Slice 4b — the k3d manifests | AUTHORED + static-validated + review folded]** `deploy/k3d/`
    (00-namespace, 10-configmap, 20-networkpolicy, 30-orch, 40-gateway, 50-shard): 3 StatefulSets (per-pod
    volumeClaimTemplates for the M3 boot-counter + the cloud-required store root — a Deployment/emptyDir would wipe
    the monotone counter = the R-6a dedup-loss landmine), 3 headless Services ALL with `publishNotReadyAddresses:
    true` (the cold-start cure — each pod's entrypoint resolves the OTHER two at its own boot, so a NotReady peer's
    A-record must still publish). Cloud env in ONE ConfigMap (VD_BUILD=cloud, **VD_TICK_HZ=50** + VD_TICK_DT=0.02
    together, the durable roots; NO forbidden knob); the mTLS + real auth Secrets minted IMPERATIVELY by
    `just k3d-secrets` (never in git). Probe stanzas satisfy the 3 inequalities at 50Hz (liveness 5×6=30s >>
    stall_deadline 400ms; grace orch 30 / gw+shard 15 >= drain; readiness de-routes on the SIGTERM edge).
    3 CODE CHANGES: (1) `docker/entrypoint.sh` + Dockerfile COPY — DNS-resolves the SocketAddr-only VD_PEERS from
    the headless pod DNS (getent ahostsv4) then `exec`s the bin as PID 1 (SIGTERM drain path), with an empty-book
    guard; the direct-bin docker smokes are unaffected (no ENTRYPOINT/CMD kept); (2) `vd_bins::validate_tick_pair`
    (+ shard.rs wiring) — a RUNTIME `VD_TICK_DT == 1/VD_TICK_HZ` guard (the compile-time assert covers only the DEV
    const; a ConfigMap retune of one-not-the-other would silently integrate at the wrong dt); (3) `vd-devcluster
    gen-authkey` (+ `auth_keypair_hex_from_seed`) — mints a real Ed25519 pair from /dev/urandom (cloud vetoes the
    dev 0x42 key); pubkey → gateway Secret, signing seed → clients out-of-band. `just k3d-{validate,up,load,secrets,
    apply,down,dod,all}` (context-pinned; k3d-validate = the offline author+static-validate path). The adversarial
    review (`wf_e5b6d490`, entrypoint-dns + code-changes CLEAN) folded 1 MEDIUM: VD_MAX_BUFFERED_INPUTS drifted to
    64 (the design misread the constant) — FIXED to 256 (== DEFAULT_MAX_BUFFERED_INPUTS / the DEV parity value).
    **LIVE BRING-UP VALIDATED (`just k3d-all` on cluster `voxeldust-newsystem`, Jul 2026):** image built (178MB) +
    imported, all 13 objects applied, ALL 3 pods `1/1 Running` with **0 restarts**, `/healthz`+`/readyz`=200 on every
    pod, the SHARD holds its realm (`directory: authority=shard:node-3`) ⇒ bootstrapped, `/metrics` served. The
    flagged residuals did NOT hit (no fsGroup/local-path EACCES — boot-counters wrote fine; no cold-start DNS
    deadlock — publishNotReadyAddresses worked). ONE DoD-recipe bug found + FIXED live: `cluster_bootstrapped()` is a
    Rust METHOD, not a serialized snapshot field, so `k3d-dod` now reads the directory for a `"authority":"shard:"`
    entry (the real bootstrap signal). Residuals still ledgered: NetworkPolicy is a no-op on k3d/flannel (declarative;
    reach admin via port-forward). **✅ S6 RESCHEDULE-RECOVERY LIVE-PROVEN (auto-pusher, Jul 2026):** the "rollout
    restart re-seed day-1 / no auto-pusher" crutch is RETIRED — `just k3d-reschedule-e2e` (fresh cluster `voxeldust-s6`)
    kills `vd-shard-0`, the StatefulSet reschedules it to a NEW podIP (10.42.0.17→.18, asserted non-vacuous), and the
    cluster RE-BOOTSTRAPS on its own: the in-process peer-resolver on BOTH survivors re-resolves the shard's DNS name +
    pushes the new IP via `update_peer_addr` (direct evidence: `peer-resolver: re-plumbed a peer to its new address
    peer=3 addr=10.42.0.18:9000` on orch + gateway), so INITIATED reliable traffic re-plumbs with NO manual intervention
    (reply-on-connection only ever covered replies). Asserts ONLY the reliable control plane (D-18: unreliable snapshots
    don't flow on the Docker-Desktop k3d overlay). Same slice hardened the harness: `60-agent.yaml` moved to
    `deploy/k3d/agent/` (the non-recursive base `apply` no longer pulls the ImagePullBackOff agent Job into k3d-dod's
    `wait -l app=vd`); k3d-dod cold-start timeouts 120s→300s. See [[project_peer_addr_autopusher]]. Owed by S5:
    in-cluster agent-HR6 continuous testing (`k3d-agent`). ✅ S5b (vdctl WalkTo/LookAt → nav + DevState orient) DONE
    — see D-15.
  - **STILL OWED after Slice D + the partial flip** (separate items, ledgered): precondition **#1 (NARROWED to the
    SOURCE-CRASH residual):** `BatchHandoff::AwaitAdopt`'s producer-less phase, IF the SOURCE crashes before the dest
    adopts, is NOT covered by the transport (the retry buffer is RAM, dies with the process). **This is NOT
    "producer-less-with-no-recovery":** the saga HAS a source-crash resolution — `(BatchHandoff, SourceUnreachable)`
    self-promotes the dest (`saga.rs:915-922`), triggered by `rehome_event_for` once `is_confirmed_dead(ctx.source)`.
    That trigger is gated on **M3** (durable monotone incarnation — else a sub-second CrashLoop restart at equal/lower
    incarnation silent-Dedups + never confirms dead) **AND L5** (addr-reread — a source rescheduled to a new addr is
    never re-dialed, so its death is never confirmed) — both already-tracked HARD deploy preconditions (R-6 + CA-1).
    ⚠️ **BUT M3+L5 are NECESSARY-NOT-SUFFICIENT for #1** (review `wf_182e1507` LOW): they make the confirm-dead
    trigger FIRE, but the `(BatchHandoff, SourceUnreachable)` self-promote (saga.rs:915-918) matches ANY BatchHandoff
    phase via `..` and does NOT check dest-adopted, so if the source crashes IN `AwaitAdopt` — entered on CasWon,
    saga.rs:815-819, exited only by `E::BatchAdopted` — BEFORE its `TransientBatch` envelope reached the dest, the
    self-promote promotes a dest that NEVER received the batch ⇒ silent loss. Closing THAT window REQUIRES (not
    "optionally") the durable-outbox / the 2d re-solicit-egress (an `AwaitAdopt` Timeout re-prompting the source to
    re-emit) — the true SELF-SUFFICIENT cure; confirm-dead alone is a lean. (Ledger: the saga.rs:508/909-911
    "every BatchHandoff phase is post-adopt ⇒ zero loss" comment is INACCURATE for `AwaitAdopt`; the self-promote arm
    should arguably be gated to post-adopt phases, OR the AwaitAdopt-pre-delivery loss window is the exact residual
    the durable-outbox closes — a pre-existing saga property, not R-4e4's, recorded here so #1 is not mistaken as
    M3+L5-complete.) NONE of this blocks the R-5 TRANSPORT-layer proof (which flips only the source-stays-up path).
    Precondition **#3** (`WAL_FORMAT_VERSION` +
    `universe_epoch_id` byte + fallible quarantining decode replacing the bare `.expect` decodes) + the
    `StoreKey::Tombstone` family; the **durable-outbox** refinement (a permanent-fsync-fault re-drive that loses
    nothing); the **durable-root ALLOW-list** (review HIGH: a prefix deny-list cannot enumerate every ephemeral
    mount — the production enforcement is a `VD_STORE_DURABLE_ROOT` the deploy points at its mounted volume, with
    `VD_STORE_EPHEMERAL_OK` the sole escape; owed WITH the deploy preconditions, no production deploy yet); a
    graceful-shutdown flush (only needed once a RELIABLE non-re-driven egress exists — none today). The redb
    backend does NOT directly unblock [[D-37]] Slice 4 (needs Store B + P7 `player_ckpt`, P6/P7); [[D-36]] unaffected.
- **✅ S0–S3 LANDED (design `wf_0f8321dc`, doc `d6_saga_wal.md`):** the orchestrator now durably persists its
  saga set + go-tokens + directory + clock ceiling through the `sim::io::Store` seam (`MemStore` staged/committed
  two-tier; redb is the io-prod backend, DEFERRED — NO new dep). `commit_result` stages the QUIESCENT
  `SagaSnapshot` at the write-back (the single point that knows `final_state`); `drive_sagas`' tail is the ONE
  group-commit barrier (persist-before-effect, before the flush phase). `register_orchestrator_with_store` →
  `rehydrate` (genesis-or-recover); recovered sagas re-arm `since=0` so the EXISTING Slice-2a Timeout producer
  re-drives them forward (no new recovery path). Atomic-MemStore co-commit ⇒ directory+saga always CONSISTENT
  post-crash (a Swapping snapshot ⟺ dest-owned directory) ⇒ NO bespoke head-re-read (the C4/C5 intra-tick
  split-brain is structurally precluded in the mem tier; it reappears only at io-prod's independent files, where
  the explicit head-re-read is owed). The directory barrier RECONCILES (delete-durable + put-current) so a
  REVOKED record is not resurrected (audit COMP-2 fix). Proven by rig kill-9 tests: a durable saga in `Demoting`
  survives + re-drives the Demote forward; a transient go-token survives (`batch_go_writes` preserved); a revoked
  record stays gone; a re-trigger of a live transient is a guarded no-op. `PersistCheckpoint` is now a documented
  no-op (the write-back + barrier subsume it — the loud-warn stub is GONE). Gate-green at 100% Tier-A.
- **✅ S4–S5 LANDED (this slice):** the harness kill-9 REBUILD analog — `FaultFabric::reregister` (fresh
  transport on the prior identity; clears the crashed node's inbound = RAM loss, but the global at-least-once
  unacked ledger HOLDS in-flight messages for redelivery) + `Topology::replace_node` (drops the old World, swaps
  the rehydrated one in), each at 100% with a `#[should_panic]` for the reclaim-an-existing-id guard. The e2e
  cells (`tests/tests/p3_orch_kill.rs` via `run_orch_kill_transient`): a live TRANSIENT batch caught mid-
  `BatchHandoff` survives an orchestrator `crash`+rebuild — the rehydrated saga re-drives on the redelivered
  acks and the debris settles at DEST with zero loss + per-tick conservation across the whole outage+recovery;
  the ANTI-THEATER twin rebuilds against a FRESH empty store and recovers NOTHING (proves the retained WAL is
  load-bearing, not in-process World survival). NB the cell uses `crash` (restart-able) not `kill` (permanent),
  because recovery of the producer-less `AwaitAdopt` phase depends on the held redelivery — see the ⚠️ below.
- **✅ The prod orchestrator BINARY is now DURABLE** (D-gamma, `crates/bins/src/bin/orchestrator.rs`
  `RedbStore::open` from `VD_STORE_PATH`): a restart RE-HYDRATES (clock resumes forward, in-flight transfers
  restore + re-drive). The boot `tracing::warn` is NARROWED to the one remaining deploy precondition (the
  at-most-once transport / the producer-less-phase egress below). No production deployment until that lands.
- **⚠️ Three io-prod / real-deploy PRECONDITIONS (audit `wf_7cc86404`, all HIGH, none break P2/P3 correctness —
  proven against the in-process MemStore + harness at-least-once model the S0–S5 cells run on):**
  1. **A CLASS of PRODUCER-LESS FLOWS depends on transport REDELIVERY the production `MeshTransport` does not
     provide.** ⚠️ THE ROOT, stated as a class (not a fixed count — three holistic audits each surfaced another
     instance, `wf_dd38151d`/`wf_ed40e95e`/`wf_259d14c0`): the at-most-once `MeshTransport` (`mesh.rs` `peer_writer` —
     `NodeUnreachable`-and-drop; a buffered-but-unacked reliable frame is lost SILENTLY, only the in-flight frame
     bounces) breaks ANY flow whose sole delivery is one emission with no re-driving producer. Known instances:
     **(saga phases, scan_deadlines-re-drivable)** `AwaitAdopt`, the D-37 re-home adopt (CURED, Slice 2d), the durable
     entity-STATE crossing; **(direct shard↔shard GhostFlow, NO saga / NO scan_deadlines at all)** the one-shot
     ghost BAND-EXIT `Despawn` (dest emits once + deregisters same-pass, stub.rs:2606/2632; a lost Despawn leaks a
     self-emitting phantom source ghost — render + collider — forever; `wf_259d14c0`). **THE ROOT CURE for the WHOLE
     class is the redelivering / retry-until-acked `MeshTransport`** (identity_persistence.md:122) — ONE fix, all
     instances; the per-flow re-solicit egresses below are INTERIM band-aids (and the GhostFlow instance has no saga
     to attach one to, so for it the transport fix or a source-side staleness reaper is the ONLY cure). A real-mesh
     transfer-WITH-band-exit + crossing-loss process test is owed alongside (the at-least-once FaultFabric masks the
     whole class — `dev_cluster_smoke` covers bring-up + SIGKILL only). **WHEN: the whole class lands before any real
     multi-shard rolling deploy.** Per-instance detail follows.
     **✅ ROOT-CURE PROGRESS (the redelivering `MeshTransport`, plan in memory `project_redelivering_transport_plan`,
     designs `wf_2a366f96`/`wf_b469b353`): R-1' (6f4614d) + R-2a (4bf9b06) + R-2b (aae1f3c) + R-3' (this commit) LANDED,
     gate-green. The mesh is now AT-LEAST-ONCE across a connection blip FOR A LANE THAT KEEPS CARRYING TRAFFIC — a reliable
     frame survives a `drop_connections` break and arrives EXACTLY ONCE (proven end-to-end in `mesh_redelivery.rs`); the
     idle-after-blip re-drive is the still-owed R-4' timer (KNOWN GAP below, data retained not lost).** R-1' split
     the hot 20Hz path into a bare `DatagramFrame` (byte-identical, zero PvP cost); R-2a grew the reliable frame to
     `ReliableFrame{from,class,incarnation:u64,epoch:u32,seq:u64,bytes}` + unified `OutFrame`. **R-2b = the SENDER lane
     FSM:** `ReliableLaneSender` (per-(peer,class), lazy) with BUFFER-FIRST seq (assign+retain THEN write — a failed
     write never burns a seq) + epoch-bump-replay on a re-dial (replay the unacked window ascending re-stamped at the
     new epoch — the cross-stream-race cure) + the per-frame OVERSIZE reject (un-framable ⇒ `reliable_shed` + bounce,
     never retained, so it can't poison the lane); `peer_writer` owns one peer-level connection + the lane map, full
     connection drop on a write error. `MeshReliabilityTuning` (validate-loud at boot) + `process_incarnation` (a 5th
     `MeshConfig::new` arg; bins read `VD_PROCESS_INCARNATION`) + 5 never-silent `MeshStats` counters. **A pre-impl
     adversarial design review (`wf_2a366f96`) caught a CRITICAL (a replay-vs-new-frame double-write — the SAME silent-dup
     class that got the FIRST R-1 attempt reverted) + 4 HIGH BEFORE any code; the structural `if stream.is_none() {
     write replay_batch() } else { write only new }` (no fall-through) designs it out.** **SMALLEST-CORRECT DEVIATION
     from the vetted design:** the cumulative-ack RETIRE (`on_ack`+`base`) is the sender half of the R-3' ack protocol
     and the buffer byte-SHED (`retry_bytes`) is R-4' — they have NO R-2b runtime consumer, so they land WITH their
     consumers (not shipped dead/`#[allow(dead_code)]`). So R-2b's retry buffer GROWS un-drained (the non-draining
     window below). **R-3' NOW READS the header + drains the retry buffer (see the R-3' paragraph below), so the mesh is
     AT-LEAST-ONCE across a connection blip WHEN TRAFFIC CONTINUES ON THE (peer,class) LANE** (the re-dial + `replay_batch`
     of the unacked window fires inside `write_frame`'s `connection.is_none()` block, which the next reliable send
     triggers). **KNOWN GAP (post-impl review `wf_a909be64`, HIGH → chartered to R-4'/R-5', data RETAINED not lost):** an
     IDLE-after-blip lane — a lone `GhostReliable` Despawn that blips then goes quiet — leaves its unacked tail in
     `lane.retry` (buffer-first; the receiver ledger survives) but does NOT re-drive it until the next reliable send on
     that lane re-dials. There is no sender-side retransmit TIMER yet (`peer_writer`'s select has only ack + rx arms).
     **✅ R-4a LANDED (this commit; design+3-review wf_40f7b3eb caught 3 CRITICALs pre-code, post-impl review wf_b1d0610c
     SHIP): the sender-side retransmit/redial TIMER + per-lane confirm-dead — the idle-after-blip gap is CLOSED.** A
     `peer_writer` 3rd biased-select arm (guarded by the PER-LANE `any_lane_owes`, the C1 cure — NOT `connection.is_none()`)
     re-drives an idle lane's retry buffer off the sender's own clock (no follow-up send needed); the R-3' inline
     `sleep(backoff).await` is REMOVED (the backoff IS the timer deadline, non-blocking). `confirm_unreachable_after_retries`
     is now consumed: a per-lane `consecutive_failures` (bumped ONLY on a lane's OWN failed replay — the C2 cure, never the
     connection-drop fan-out; reset per-lane — the H2 cure) gates the `NodeUnreachable` bounce, so a blip that RECOVERS =
     ZERO bounce (proven by `an_idle_after_blip_lone_frame_is_re_driven_by_the_timer` + the zero-bounce blip test); a
     genuinely-dead peer still bounces after N. Buffer-first-before-dial: a first-dial failure leaves the frame RETAINED +
     re-driven (never a silent drop). io-prod Tier-B 93.44% ≥ 90. **✅ R-4b LANDED (this commit; focused adversarial review
     aa95e10c SHIP — the accounting is provably correct): `retry_bytes` byte-cap + shed-loud = the M1 unbounded-growth
     cure.** `RetainedFrame{frame, framed_len}` stores the FROZEN worst-case (`u32::MAX`-epoch) framed length (the H3 fix —
     it upper-bounds every re-stamp, so accounting never drifts as the epoch varint grows); `retry_bytes` tracks the total
     in LOCKSTEP (`assign_and_retain` adds, `on_ack` subtracts the SAME number). `AssignReject{Unframable, BufferFull}` —
     BufferFull = PRODUCER BACKPRESSURE (refuse the NEW send; NEVER shed a retained frame, since the receiver is strictly
     contiguous so dropping any seq wedges/loses the window); both are shed-loud (`reliable_shed` + a bounce, no timer arm,
     no connection drop — a full buffer = a dead ack path, `reliable_acked` stuck-at-0 corroborates). The mpsc bound is the
     primary producer backpressure for a slow-but-alive path (D2 tier 1); the byte cap is the tier-2 dead-ack shed. Review
     fix: `validate()` now requires `retry_buffer_max_bytes >= MAX_STREAM_FRAME_BYTES + 4` (the length prefix) so a single
     maximal framed frame can always be retained (was off by the envelope). **STILL OWED to flip this fully 🟩:** R-4c
     (`LivenessTuning::validate_against(&SagaTuning)` — the geometric-backoff
     confirm-ticks ≤ abort_deadline invariant, boot-asserted, the C3 cure), R-4d (MeshStats→`/metrics` MetricsSource, M4),
     R-4e (N-peer real-QUIC load test, L7 — also the natural home for a `replay_lanes` conn_died-mid-pass coverage test:
     the dial-succeeds-then-write-fails arm is Tier-B-uncovered today, floor still met, review wf_b1d0610c LOW). R-5' (the
     `mesh_under_loss.rs` real-mesh capstone driving an actual PRODUCER-LESS flow — a raw `GhostReliable` Despawn — across
     the break, PROVING the idle-after-blip re-drive end-to-end, retiring the per-flow band-aids). The loopback bridge keeps its simple
     per-stream seq (single-stream test infra; NO lane FSM). R-6 = durable outbox + durable boot-counter incarnation
     (P6/P7).
     **✅ POST-R-4b HOLISTIC AUDIT (wf_9d72e38c, DONE_NO_CRITICAL — 0 CRITICAL, 0 in-scope-unaddressed HIGH; confirm-dead +
     shed COMPOSE cleanly with the saga LivenessTracker, frozen seam intact, 20Hz hot path zero-cost). SHARPEST QUESTION
     DECISIVELY RESOLVED: at the DEFAULT tuning there is NO confirm-dead-vs-saga-abort live hazard — R-4c is DEFENSE-IN-DEPTH
     (a mis-tuning guard), NOT a fix for a live strand.** The proof (keep in the R-4c PR): (1) a live-but-slow peer emits ZERO
     transport bounces — `owes_redelivery` needs `stream.is_none()`, a connected-but-slow peer keeps `stream.is_some()` so the
     timer never fires + `consecutive_failures` stays 0 (ack-lag is R-4b's shed job, which does NOT feed the LivenessTracker);
     "slow" and "unreachable" are STRUCTURALLY DISJOINT transport states. (2) For a DEAD peer the transport confirms (~0.35–1.55s
     @ default confirm=3 × geometric backoff 50ms→5s) FASTER than the saga's DESTRUCTIVE abort, which is DOUBLE-gated
     (`is_confirmed_dead` AND `abort_deadline_ticks`=1.2s from `dead_observed_since`) and strictly downstream — the transport
     can only make evidence accrue, never abort independently. (3) The only live-but-slow abort (pre-freeze Timeout, saga.rs
     ~621/641/665) is liveness-INDEPENDENT and PRE-DATES R-4a.
     **✅ R-4c LANDED (this commit; design+3-review wf_dd37d7a5 caught a CRITICAL + HIGH pre-code — the naive formula summed the
     WRONG series segment): `LivenessTuning::validate_against` — the GEOMETRIC cross-config invariant, boot-asserted in the
     orchestrator (the ONE process holding both clocks).** Pure Tier-A fns in saga.rs: `backoff_series_sum(lo,hi)` (half-open
     window of the redial series `min(backoff_min·2^(i-1), backoff_max)`), `duration_to_ticks_ceil` (ceiling, div-by-zero-guarded),
     `transport_run_spread_ticks(N,n)` = the ticks spread of the n-notice run at fires N..N+n-1 = series `[N+1, N+n)` — the LATE
     near-cap segment (the CRITICAL fix: the naive `[1,n)` early-segment gave 7 ticks vs the real 24 @ 20Hz, a 3.4x-too-narrow
     window that would orphan a genuinely-dead peer). `validate_against(confirm_retries, backoff_min, backoff_max, tick_hz)`
     asserts `unreachable_window_ticks >= run_spread` (raw args ⇒ NO sim→io-prod edge; supersedes the coarse linear `validate()`
     window floor). Orchestrator reads the ACTUAL `mesh_cfg.reliability.confirm_unreachable_after_retries`/`redial_backoff_min/max`
     + hoisted `VD_TICK_HZ`, cross-validates LOUD after `liveness.validate()`. Dev 50Hz n=3 → run_spread 60 ≤ window 64 (a real
     4-tick margin, gate-surfaced). **SCOPED-DOWN from the vetted design (2 deliberate deviations, ledgered below):** (i) DROPPED
     the check-(b) abort-vs-confirm ordering invariant — the audit PROVED a live-but-slow peer emits ZERO transport bounces (never
     reaches confirmation), so check-b's "false-abandon" premise has no real path, and the abort grace (1.2s) comfortably covers
     the ~400-800ms redial cadence at confirmation time; the abort-grace adequacy is a SOFT tuning concern (revisit if a
     recovering-blip-on-a-5s-backoff case ever bites), NOT a hard invariant. (ii) DEFERRED the M2 transport env-plumb
     (`VD_MESH_*`) — check-a validates against the ACTUAL (default) `mesh_cfg` the mesh runs with, so the invariant WORKS today;
     the env-plumb only adds tuning the transport side (an operator convenience) and lands as a small follow-up. **AUDIT-REFINED
     OWED ITEMS (fold into the named slices):** [→R-4c follow-up, M2] env-plumb the transport clock
     (`VD_MESH_CONFIRM_RETRIES`/`VD_MESH_BACKOFF_MIN_MS`/`_MAX_MS`, or `MeshConfig::from_env`) so `validate_against` has TWO
     tunable sides (today it validates the tunable saga window against the default transport backoff). [→R-4c/R-4d, M3 — a REAL gap R-4b
     INTRODUCED] a `WriteFail::Shed` bounce (BufferFull/Unframable) pushes `Inbound::NodeUnreachable` BYTE-IDENTICAL to a
     dead-peer bounce, and `saga_runtime` `record_unreachable`s it UNCONDITIONALLY — a BufferFull shed toward an inbound-QUIET
     peer (a frozen source shard) can accrue to `n_consecutive_unreachable` and FALSE-confirm a live-but-ack-stalled peer.
     Non-triggering at default (clear-on-ack fires on ANY inbound Wire; a shed is one-shot per new send, no timer/counter), but
     the clean cure is a DISTINCT `Inbound::SendShed` arm the saga routes to a METRIC, never to `record_unreachable` (an
     additive `Inbound` seam change → frozen-wire review). Scope BEFORE a real deploy.
     **✅ R-4d1 (M3) LANDED (design+3-review `wbt45dxj6` caught a HIGH + 6 MEDIUMs pre-code; post-impl review `wf_d11fdb03`):**
     `Inbound::SendShed{to,class,undelivered,reason:ShedReason{Unframable,RetryBufferFull}}` added to the frozen `sim::io`
     seam (reliability()=Reliable); `WriteFail::Shed(ShedReason)` maps `AssignReject` at the mesh reject-site (never leaks the
     private reject across the crate boundary) → `peer_writer` emits `SendShed` (NOT `NodeUnreachable`) + a debug_assert that
     the shed is reliable-lane-only; `saga_runtime` routes `SendShed`→`runtime.sends_shed` counter, NEVER `record_unreachable`
     (regression test proves a burst of sheds leaves `liveness_notices==0` + peer un-confirmed-dead); `app.rs` `TickReport`
     gains a DISTINCT `shed` count (not folded into `unreachable`). Coverage deviation (HR5): the four fabric/mem-driven Tier-A
     consumers (tracer, harness client/chaos/fabric-test) that can NEVER produce a shed use an exhaustive OR-PATTERN merge
     (SendShed folded with the nearest covered arm) — preserves compile-time exhaustiveness without an uncoverable region;
     the load-bearing distinction (seam/saga/app) stays distinct+injected-covered; the design's `TraceEvent::SendShed` was
     DROPPED (never-constructed → uncoverable derives). Caught + fixed a design-missed consumer (`mesh.rs` filter_map arm). NEW
     io-prod integration test proves the `RetryBufferFull`→`SendShed` emit path end-to-end (min buffer + near-MAX frames =
     deterministic shed, asserting no `NodeUnreachable`). Gate: Tier-A 100% region+branch, Tier-B 93.44%, clippy/fmt clean.
     [→R-4d, M4] the 7+ `MeshStats` counters
     (`reliable_shed`/`reliable_acked`-stuck-at-0=dead-ack-path/`gap_drop`=MUST-BE-0/…) are BLIND in prod — every bin discards
     `_control`, `/metrics` serves hardcoded 0, `metric_names::ALL` omits them; wire `MeshControl::stats()` via a `MetricsSource`
     BEFORE any soak (a soak with invisible alarms proves nothing).
     **✅ R-4d2 (M4) LANDED (design+3-review `wbt45dxj6`; post-impl review `wf_dbc1ca7a`): the 10 mesh counters are now
     scraped on the orchestrator `/metrics`.** `metric_names::ALL` +9 (`_total` counters + the legacy un-suffixed
     `vd_datagrams_dropped_too_large` kept as a frozen scraped name); a `MetricsSource` trait (mirrors `SnapshotSource`) with a
     flat `MetricValues` + live `MeshMetrics(Arc<MeshControl>)` (reads `stats()` per scrape — a pure atomic load off the hot
     path) + `FixedMetrics`; `render_metrics` EXHAUSTIVELY destructures `MetricValues` (no `..`) so a new field is a compile
     error until paired (the registry-vs-struct completeness guard); 2-arg `admin_router` at both call sites; the orchestrator
     RETAINS its `MeshControl` (was discarded). Tests: render pairing + both `find` branches + struct↔registry completeness +
     the `/metrics` HTTP scrape reflecting live values + a NEW integration test proving the live `MeshMetrics`→`MeshControl`
     bridge mirrors `stats()` field-for-field. Gate: Tier-A 100%, Tier-B ~93%, clippy/fmt clean.
     **[→ M4 follow-up, before a soak dashboard]** `render_metrics` emits `# TYPE {name} untyped` for EVERY metric,
     including the new `_total` COUNTERS — Prometheus convention pairs a `_total` name with `# TYPE counter`. Pre-existing
     (the removed `render_metrics_shell` did the same); `rate()`/`increase()` still work on untyped, but a scraper/linter that
     keys off TYPE will not classify these as counters. Give `metric_names` a per-name type tag and emit the right TYPE line
     when the first soak dashboard is authored (post-impl review `wf_dbc1ca7a` LOW).
     **R-4d2 (M4) SCOPE (vetted `wbt45dxj6` HIGH): orchestrator-ONLY this slice.** The draft wanted gateway+shard `/metrics`
     listeners too, but that is UNDELIVERABLE in the dev cluster: `ClusterAddrs` has ONE `admin` field, `VD_ADMIN_ADDR` is set
     only in `orchestrator_env`, and `SlotPorts` allocates ONE `admin` port — so gateway/shard read `VD_ADMIN_ADDR` unset
     (spawn no listener = dead capability) or, via `common_env`, collide three binds on one port. So R-4d2 lands the mesh-metrics
     capability wired into the orchestrator's existing `VD_ADMIN_ADDR` axum serve (`MetricsSource`/`MetricValues`/`MeshMetrics`,
     `metric_names::ALL` +9, a `render_metrics` that exhaustively destructures `MetricValues` so a new field is a compile error,
     the 2-arg `admin_router` at BOTH call sites incl. the `admin.rs:88` in-crate test helper). **[→ per-node admin ports, owed
     with the gateway/shard soak]** gateway/shard mesh counters stay UNSCRAPED until real port work: 3 admin fields on
     `ClusterAddrs`, 3 `SlotPorts` ports at new offsets, 3 distinct per-node `VD_ADMIN_ADDR`s, curl-all-three in the smoke test.
     [→R-4e, M6+cloud] make R-4e REAL-QUIC (not the loopback
     bridge) with (a) a CORRELATED multi-peer outage (drop ≥50% simultaneously) asserting `inbound_dropped_reliable`≈0 for
     surviving-peer traffic — else coalesce confirm-dead to ONE notice/peer/window or give NodeUnreachable a priority lane; (b) a
     pod-reschedule/address-change scenario (L5: `peer_writer` captures `addr` once at spawn — a moved peer is dialed stale
     forever; recovery needs address re-plumb); (c) the H2 RecvLedger-lock contention MEASUREMENT.
     **R-4e DESIGN VETTED (design+3-review `wf_e096eefb`, SOUND_TO_IMPLEMENT; full spec `scripts/r4e_vetted_design.md`; caught a
     CRITICAL — the `inbound_dropped_reliable==0` assertion was a drain-vs-fill FLAKE, cured by sizing the receiver inbox to the
     full backlog). SUB-SLICED: R-4e1 (load harness + baseline, H2 UNCHANGED) → R-4e2 (H2 per-peer RecvLedger re-key) → R-4e3
     (correlated-outage N=8 all-pairs + L5 `#[ignore]` red-guard + `conn_died` coverage) → R-4e4 (R-5 `mesh_under_loss.rs`
     producer-less capstone + the D-6 #1 PARTIAL-flip). ✅ R-4e1 LANDED: `crates/io-prod/tests/mesh_load.rs` — fan-in (N senders
     → 1 receiver), a `LoadShape` config (all consts homed; `VD_MESH_LOAD_NODES` env, default 16), the SIZED receiver inbox
     (`senders*frames_per_sender+256` ⇒ `inbound_dropped_reliable==0` STRUCTURAL not a race), completion-gated EXACT assertions
     (per-`from` seq-set == 0..frames, per-sender `reliable_acked`, gap/stale/dedup==0) + a derived generous progress deadline;
     `just mesh-load` recipe (pinned N=64, validated GREEN 5/5 on the dev host, N=128 also green ~0.8s — the fan-in is
     correctness-bound not wall-clock-bound on loopback; re-validate the target host fd/port ceiling before a cloud soak). H2
     UNCHANGED (proves the single-lock baseline is CORRECT under fan-in; the re-key is R-4e2).
     **✅ R-4e2 LANDED (post-impl review `wf_13851c8e`, 2 opus concurrency lenses + synth → SOUND_TO_COMMIT, ZERO
     correctness/liveness/deadlock defects): the H2 per-peer RecvLedger re-key.** `RecvLedger` is now
     `Arc<RwLock<BTreeMap<NodeId, Arc<Mutex<BTreeMap<MsgClass, RecvState>>>>>>` (std-only, no new dep) — a frame from peer P no
     longer serializes against peer Q. Outer `RwLock` shared-READ on the hot path; write-locked ONLY on a first-frame-from-a-new-
     peer (`or_insert_with` idempotent for the concurrent-first-frame race). Inner per-peer `Mutex` HELD across
     classify+push_inbox+`*st=before` rollback (the atomicity invariant — two `serve_data_stream` tasks for one (peer,class) overlap
     across a redial). `ack_egress` copies acked_keys then two-level-reads (never nests acked_keys inside an inner lock); kept
     per-key (no single-peer collapse). Lock order outer-read→inner→inbox, never reversed; all poison-safe. Review VERIFIED
     byte-identical behavior vs the single-lock version + N=64 fan-in green 3/3 + Tier-B mesh.rs 93.98%. **2 LOWs, ledger-only:**
     (i) `ack_egress`'s outer-peer-miss arm is a dead region unreachable-under-trust (acked_keys populated only after the peer is
     inserted) — TOLERATED by the Tier-B 90% floor (a `coverage(off)` would over-engineer a region the tier deliberately tolerates;
     the defensive None-guard stays per the "correct-even-if-multi-peer-ever-possible" invariant); (ii) std `RwLock` has no
     writer-fairness guarantee — a first-frame WRITE could theoretically be delayed under sustained read pressure (bounded, once
     per peer, empirically fine at N=64); if a churn-heavy soak ever shows first-frame admission latency, a writer-preferring
     `parking_lot::RwLock` is the cure but a NEW-DEP JOINT-INVESTIGATION decision, NOT to be adopted unilaterally.
     RESIDUAL (R-5-audit): after the ledger re-key the node-wide `SharedInbox` Mutex is the NEXT RX serialization point — per-peer
     inbox partition / lock-free MPSC drain is a future scaling slice; the ledger re-key alone does NOT deliver full RX isolation.
     **NEW RESIDUAL (/goal `wf_58cc14fb`, MEDIUM — the one UNCAPPED receive structure): the per-peer `RecvLedger` outer map + the
     `acked_keys` set GROW FOREVER.** A first frame from a NodeId inserts a never-removed outer entry (mesh.rs `classify_and_deliver`
     first-frame arm) + an `acked_keys` insert (`serve_data_stream`); neither is ever evicted. Bounded + fine for the static-roster
     dev/cloud tier (a fixed peer set), but under HIGH PEER CHURN or a spoofed-NodeId flood it is an unbounded memory leak / DoS
     vector — contrast the retry buffer (byte-capped, R-4b) and the inbox (BoundedInbox). NOT a live hazard at the current tier
     (peers come only from the trusted static book/roster over mTLS). **Owed (M6/cloud, before an untrusted/churny peer set):** evict
     a peer's ledger+acked entry on last-activity (gated on the durable incarnation so a legit reconnect re-seeds cleanly) OR cap the
     tracked-peer count with a loud shed. Cheap to add when peer identity becomes dynamic; ledger-only for now.
     **✅ R-4e3 LANDED (tests-only, NO product change): the correlated-outage proof + the L5 red-guard.** Item 3
     (`a_correlated_half_cluster_outage_never_evicts_surviving_peer_traffic`, mesh.rs): 8 nodes all-pairs, kill 4 SIMULTANEOUSLY,
     drive the 4 survivors; gated on survivor-1 receiving every frame from the 3 live survivors loss-free AND the dead-lane bounce
     stressor being ACTIVELY present (≥ dead_lanes NodeUnreachable in the same inbox) — asserts `inbound_dropped_reliable==0`.
     PASSES ⇒ **the coalesce cure is NOT needed** (the backoff-paced re-bounce cadence can't out-produce the drain, as the R-4b
     audit predicted); the `last_bounced_unreachable` coalesce stays a documented if-it-ever-reds fallback, not built. Item 4
     (`l5_a_rescheduled_peer_at_a_new_address_is_reachable`, `#[ignore]`d): a RED GUARD mirroring `ca1_reply_on_connection` — A's
     book points B at a stale addr (B binds elsewhere = a rescheduled pod); asserts the TARGET (B reachable at its real addr) and
     FAILS BY CONSTRUCTION today (verified: "timed out; got []" — A dials the once-captured stale addr forever). Flips green when
     the addr re-plumb (CA-1/provisioning) lands; the exact line to flip is named in the test. **Item 5 (`replay_lanes`
     conn_died-mid-pass coverage) DEFERRED (design-sanctioned "time-box + ledger"):** the dial-ok-then-write-fails `Err(())` arm is
     a tight cross-lane race (the connection must die in the window between two lane writes in one replay pass) — not
     deterministically triggerable without an injectable fault seam, and it is Tier-B DEFENSE-IN-DEPTH already above the 90 floor
     (io-prod ~93.98%). Owed with R-4e5/an injectable-write-fault seam (xref R-4a post-impl review `wf_b1d0610c` LOW); NOT worth a
     flaky race test now.
     **✅ R-4e4 LANDED: the R-5 producer-less capstone (`crates/io-prod/tests/mesh_under_loss.rs`) + the D-6 #1 PARTIAL flip
     (transport-layer redelivery → 🟩, above).** A real `GhostFlow::Despawn` (band-exit, producer-less — no saga/scan_deadlines)
     is driven across a `drop_connections` blip + idle; B receives it EXACTLY ONCE via the R-4a timer alone, the envelope
     round-trips byte-exact, `gap_drop==0`, ZERO `NodeUnreachable` (blip≠death), endpoint survives — stable 5/5 standalone
     + green in the full concurrent io-prod suite, ~0.22s. Run by
     `just mesh-load` alongside the N-peer load gate. **R-4e COMPLETE** (R-4e1 load harness / R-4e2 H2 RX re-key / R-4e3
     correlated-outage+L5 / R-4e4 R-5 capstone all landed); the D-6 #1 SOURCE-CRASH residual is narrowed + gated on M3+L5 (above).
     [→R-6, CLOUD-BLOCKING,
     SHARPENED] confirm-dead AMPLIFIES the M3-incarnation wall-clock hazard: a CrashLooping peer is confirmed-dead (correct) but
     its fast-restart reliable traffic is silently Dedup/Stale-dropped ⇒ clear-on-ack never fires ⇒ a split-brain-ish stall; the
     durable monotone boot-counter is a HARD precondition before ANY real deploy, not a dev residual. [→P9] pin an
     `intershard.rs` Signal forward-note: R-4b's BufferFull shed is PRODUCER-BACKPRESSURE (not infinite reliable buffering) — a
     cross-shard Signal fan-out (radio/functional-block-to-leased-subscribers) must claim its OWN shed-degrade policy + its own
     `MsgClass`, never inherit a silent `NodeUnreachable`.
     **R-2b cross-slice contracts (binding for R-3'):** (a) the receiver MUST dedup by (peer,class,incarnation,
     seq) EPOCH-AGNOSTICALLY — epoch gates ONLY the high-water-advance race, never seq retirement; a torn frame is
     discarded wholesale (`read_one_reliable_frame` already None-on-short-read). (b) The per-class lane split REMOVES
     cross-class ordering on a link — every reliable consumer must be order-INDEPENDENT across classes (verified for the
     gateway dispatch; a cross-class ordering dependence must ride the SAME class, never the transport). (c) The full
     connection drop on a write error is REQUIRED for old-stream cleanup — never downgrade to a stream-only reset
     without a receiver-side stream reaper (the stream-only-error optimization is deferred). (d) ✅ SATISFIED in R-3':
     `common_env` (bins/src/lib.rs) exports a per-launch wall-clock-ms `VD_PROCESS_INCARNATION` so a dev restart comes up
     at a strictly-higher incarnation than a peer's surviving ledger holds (no silent-Dedup collision). The durable
     boot-counter that also survives a wall-clock REWIND is R-6 (P6/P7). **R-2b
     coverage scope (post-impl review `wf_93fc5909`, which caught + I FIXED an off-by-the-header oversize poison-window
     CRITICAL before commit):** the SENDER FSM correctness (buffer-first, replay-batch exactly-once, the framed-oversize
     reject incl. the boundary, the reconnect-resets-an-existing-lane path) is proven by the io-prod unit+integration
     tests (mesh.rs 94.70%). The LIVE end-to-end multi-frame replay over a REAL redial (a peer that drops then RECOVERS
     mid-stream) needs `MeshControl::drop_connections()` — R-3'/R-5' (`kill()` cannot blip-recover) — so that
     end-to-end proof is the R-5' `mesh_under_loss.rs` capstone, NOT an R-2b gap.
     **✅ R-3' LANDED (this commit; design+3-adversarial-review `wf_b469b353` — 13 blocking findings incl. 2 CRITICALs
     resolved BEFORE code; R-2b+R-3' are ONE correctness unit).** THE RECEIVER: a node-wide `RecvLedger` (`(peer,class)
     → RecvState{incarnation,epoch,hw,primed}`) that SURVIVES connection teardown + a pure `classify_reliable` verdict
     ladder (incarnation ▸ epoch ▸ contiguity; StaleEpoch decided BEFORE any hw compare = the reverted-R-1 cross-stream
     cure). **CRITICAL#1 (the reverted CRITICAL's twin) designed out:** the higher-incarnation A1 arm resets to
     `{hw:0,primed:false}` and FALLS THROUGH — a reset-window reorder surfaces as Gap (a MUST-BE-0 alert), NEVER a silent
     Dedup burying a lower never-delivered seq. THE ACK LANE: acks ride the reverse direction ON THE SAME connection they
     arrived on — the data receiver's `serve_connection` opens ONE `STREAM_KIND_ACK` uni stream (cumulative, idle-flushed,
     cancel-safe write OUTSIDE the select); the data sender's `peer_writer` gains an `accept_uni` ack-reader on its OWN
     dialed connection → `on_ack` retires the prefix (clamped to `next_seq`, monotone `base`, epoch+incarnation matched).
     **CRITICAL#2 (ack topology): the "MANDATORY AckRouter" that 2 of 3 reviewers demanded was REJECTED** — an AckRouter
     couples two independent connections' liveness (a deadlock during a one-directional redial); the same-connection
     reverse stream is buildable (quinn Connections are symmetric) and needs no NodeId→mpsc registry. **Folded refinement
     (a review gap): on a reliable inbox-drop the whole `RecvState` ROLLS BACK to its pre-classify snapshot** (advancing
     hw without delivering = permanent silent loss; contiguity, not the epoch check, is the anti-burying guard so the
     rollback can't re-open the CRITICAL). A 1-byte `STREAM_KIND` tag demuxes DATA/ACK (a raw `read_exact(1)` BEFORE the
     framing loop — `read_one_reliable_frame` stays byte-identical, so the loopback bridge is untouched). `common_env`
     now exports a per-launch wall-clock-ms `VD_PROCESS_INCARNATION` (the restart-collision precondition; a `dev_cluster_smoke`
     test asserts two launches strictly increase). Tests: 16 pure classify/on_ack unit tests (each arm equality-asserted,
     incl. the reset-window-reorder-⇒-Gap-not-Dedup case) + `mesh_redelivery.rs` (happy-burst-acks-retire, drop_connections
     blip exactly-once, lone-frame idle-flush, sender-restart-higher-incarnation). `reliable_acked` is a 6th counter (a
     stuck-at-0 = a dead ack path). NOTE: acks flow promptly (per-frame `ack_due`), so a blip's replay re-covers only the
     un-acked tail (optimal) — the receiver dedup path is exercised deterministically by the pure `classify_dedups_*` test,
     not asserted in the integration blip (which asserts no-loss + no-dup + a real disruption). **R-3' latent note (review
     `wf_a909be64` LOW, provably correct TODAY, tied to R-6): `AckFrame` carries ONE `incarnation` scalar over its
     per-class `entries`** — sound under the one-connection-one-sender-incarnation invariant (a `peer_writer` stamps every
     DATA frame with its own `local`/incarnation, so one accepted connection is uniform; `ack_egress` mints a fresh
     `last_sent` per connection). IF R-6's durable-incarnation work ever lets one accepted connection be REUSED across a
     sender restart (two incarnations on one connection), this must move `incarnation` INTO `AckEntry` (per-class) or the
     sender's `on_ack` silently rejects the mismatched entries = a dead-ack-path (retry grows to the R-4' shed), not loss.
     **R-3' HOLISTIC AUDIT owed items (wf_d0a91a43, DONE_NO_CRITICAL — 0 CRITICAL, 2 latent HIGH correctness-first-deferred
     with LEDGER-not-rework; the transport COMPOSES soundly + advances the end-goal). LOAD-BEARING (synth: "must not be
     lost"):** **[H2, P3-load scaling]** the `RecvLedger` is ONE node-wide `Mutex<BTreeMap<(NodeId,MsgClass),RecvState>>`
     locked on EVERY reliable inbound frame (held across the nested inbox lock) — the wrong GRANULARITY for
     hundreds-in-one-location: it reintroduces a node-wide receive-plane serialization point the SEND lanes don't have
     (contradicts the mesh.rs:6-8 per-peer-isolation promise for the RX direction). NOT a correctness bug (short hold, fine
     P1/P2/P3); the clean fix when due = key the outer map's Mutex PER-PEER (`serve_connection` already = one peer) + drop
     the ledger lock before `push_inbox` (snapshot verdict, re-lock only on the rare reliable-drop rollback). Measure it
     under the load test (below); do NOT rework before P3-load. **[M4, before any real soak/deploy]** the 7 never-silent
     `MeshStats` counters (incl. `reliable_acked` stuck-at-0 = DEAD-ACK-PATH alert, `gap_drop` = MUST-BE-0) are NOT on the
     ops surface — `metric_names::ALL` omits them, `render_metrics_shell` hardcodes 0, and NO bin feeds `MeshControl::stats()`
     to `/metrics`. So the alarms the design promises are invisible in prod (a `tracing::warn` only). Additive fix (off the
     frozen seam + hot path, sequence with R-4'/R-5' as ONE ops slice): add the 7 names to the review-gated `metric_names::ALL`
     + a `MetricsSource` trait mirroring `SnapshotSource` reading `stats()` + the `BoundedInbox` tallies; the shard/gateway/orch
     bins hold their `MeshControl` and feed `/metrics`. **CORRECTLY-OWED (named future slices):** [H1→R-5'] the mem `FaultFabric`
     preserves cross-class FIFO but the mesh does NOT — pin with a process-parity test (or per-class-split the fabric) so a
     cross-class-order-dependent consumer REDs in-process (contract now in the `Transport` seam doc). [M3→R-6, CLOUD-BLOCKING]
     `VD_PROCESS_INCARNATION` = wall-clock-ms is UNSAFE under k8s (CrashLoopBackOff restarts sub-second ⇒ EQUAL incarnation ⇒
     silent-Dedup LOSS; NTP/reschedule clock-skew ⇒ LOWER ⇒ StaleIncarnation-drops all restarted traffic) — R-6's durable
     monotone boot-counter is a HARD precondition on the SAME gate as the AwaitAdopt egress before ANY real deploy, NOT a "dev
     residual".
     **R-6 DESIGN VETTED (design+3-review `wf_6c81f81c`, SOUND_TO_IMPLEMENT after de-scoping k3d; spec `scripts/r6_vetted_design.md`).
     SUB-SLICED: R-6a (boot-counter core) → R-6c (L4 AckEntry per-class incarnation) → R-6b (LOOPBACK CrashLoop e2e proof) →
     R-6d (durable outbox, follow-slice). ✅ R-6a LANDED: `crates/io-prod/src/boot.rs` — `BootCounter` (a 24-byte checksummed
     sidecar file: ABSENT⇒genesis at `genesis_floor.max(1)`; VALID⇒`checked_add(1)`; CORRUPT⇒FAIL LOUD never-reset-to-lower;
     atomic write-tmp→fsync→rename→fsync-dir, durable BEFORE `spawn_mesh` so a re-used value is never wired; MONOTONE + immune
     to a wall-clock rewind, no `SystemTime` in the increment path) + a shared `check_durable_path` (durable-root ALLOW-list
     when declared, else the temp DENY-list; `canon_lenient` resolves symlinks for an uncreated subdir). `bins::resolve_process_
     incarnation` precedence: VD_PROCESS_INCARNATION explicit WINS (dev/test via common_env + future orchestrator-issued) → else
     VD_BOOT_STATE_DIR durable self-counter (genesis_floor=launch_incarnation, VD_BOOT_DURABLE_ROOT allow-list, VD_BOOT_STATE_
     EPHEMERAL_OK escape) → else FAIL LOUD. std-only NO new dep, NO frozen-seam touch, NO HR3 fork. 8 boot + 4 bins unit tests;
     process_parity green (real bin boot path); Tier-B boot.rs 94.17%. Post-impl review `wf_b241ed1c`.
     **✅ R-6c LANDED (c7a7255): the ack `incarnation` moved from a single `AckFrame` scalar (`ack_egress` overwrote it each
     loop iteration = last-class-wins) INTO `AckEntry` (per-class); `peer_writer` fan-out calls `on_ack(e.incarnation, …)` per
     entry — removes the latent silent send-stall R-6a's first-class sender-restart could reach on a reused connection carrying
     two classes at different incarnations. Greenfield hard cutover, below the frozen seam.**
     **✅ R-6b LANDED — M3 CAPSTONE 🟩: `crates/bins/tests/boot_counter_crashloop.rs` (Tier-B, gated via `just orch-crash` with
     `--test-threads=1`) is the LOOPBACK CrashLoop e2e proof. A real SIGKILL + sub-second shard restart, over the real QUIC mesh,
     PAIRED: RED control (fixed `VD_PROCESS_INCARNATION` reused across the restart) ⇒ the orchestrator silently DEDUPS the
     restarted shard's reliable realm re-grant (observed `vd_dedup_drop_total`+2/+3 over 3 runs); GREEN (durable boot-counter ⇒
     strictly-higher incarnation) ⇒ the receiver's A1 ladder resets the dedup high-water ⇒ Accepted, `dedup_drop`+0/`stale`+0.
     Same shard re-grant behavior; only the incarnation source differs — so the pairing PROVES the boot-counter fixes exactly what
     the wall-clock version breaks. Adversarial review (opus) verdict SOUND_WITH_NITS — both nits (gate-wiring with a thread guard;
     RAII temp cleanup) folded before commit. This flips the M3 milestone: `VD_PROCESS_INCARNATION`=wall-clock is no longer the
     only incarnation source; the durable monotone boot-counter is proven end-to-end across a real process crash.** Only R-6d
     (durable outbox, the AwaitAdopt SOURCE-CRASH egress = D-6 #1 residual) remains in the R-6 arc.
     **R-6d DESIGN VETTED (design+3-adversarial-review+synthesis wf_0af5fa8e; full spec `scripts/r6d_vetted_design.md`).
     Verdict REVISE — the core layering DECISION is SOUND + code-confirmed, but 3 CRITICALs proved the transport outbox
     ALONE does NOT close D-6 #1; the synthesis folded all fixes into a corrected 4-sub-slice plan (implement-ready).
     LAYERING (fact): Shape (A′) TRANSPORT-OWNED write-through mirror of `ReliableLaneSender.retry`, its OWN fsync, NOT
     atomic with any above-seam saga step — Shape (B) saga-runtime-owned is INFEASIBLE (the `(peer,class,incarnation,seq)`
     key is minted async INSIDE the writer task `assign_and_retain`, does not exist above the seam). SCOPE: producer-less
     one-shots ONLY (`TransientBatch` + `GhostFlow::Despawn`) via a per-send `Durability` marker — NEVER all reliable
     traffic (no fsync-per-send), NEVER an HR3 fork. **3 CRITICALs (all REQUIRED, none alone closes D-6 #1):** (C1) the
     durability fsync-gate must live in the WRITER task (`write_frame` Reliable arm), NOT `flush_outbox` (which only
     enqueues → `send()` returns a `MsgId`, no seq yet); (C2) a NEW `InterShardFlow::TransientDiscard` arm + dest-side
     `on_transient_discard` — the assumed dest-side GC of orphaned `Arriving` items does NOT exist, so abandon-then-replay
     strands a silent `Arriving` orphan (poison `TRANSIENT_BATCH_STEP` so a late replayed adopt is AlreadyApplied); (C3)
     `(BatchHandoff, SourceUnreachable)` self-promotes WRONGLY from `AwaitAdopt` (empty dest) AND is UNREACHABLE there
     (no orchestrator→source lane ⇒ parks forever) — split the arm by phase + add an `AwaitAdopt` re-solicit egress
     (DEFERRED.md:280 FSM-field template) so confirm-dead is reachable + resolve as accounted loss (discard-to-dest +
     `batch_lost_source_crash` count), budget-gated on `abort_deadline_ticks` (a source that RESTARTS within budget wins
     via its outbox replay). **ADDITIVE FROZEN-SEAM CHANGE (HIGH-2, owned): `Transport::send` grows a `Durability` param
     across all 4 impls (Mem/Mesh/Prod/Fabric) lockstep — mem/fabric no-op it.** SUB-SLICES: R-6d1 (per-node RedbStore
     for shard+gateway + writer durability handle; `open_node_outbox` reuses `check_durable_path`; tick-loop split gated
     on `outbox.is_some()` so `step_tick` stays byte-identical) → R-6d2 (`OutboxSink` seam [retain/release/commit/
     scan_all/gc_below, 26-byte BE key] + FSM write-through/delete-through + the additive `send(durability)`) → R-6d3
     (the C1 durability gate + boot replay [MEDIUM-1 send-based, HIGH-3 GC-after-durable] + the C2/C3 saga+dest closure)
     → R-6d4 (named gates: the mock-backed both-ends-restart replay proptest [io-prod pure] + the SIGKILL-source-in-
     AwaitAdopt process proof with a RED no-outbox control). HR1/HR3/HR5/no-new-dep all MET; boot replay deterministically
     Accepted (verified vs classify_reliable A1). NOT near-term (follow-slice, gated behind M3✅+L5); LARGER than the
     original sketch (seam-touching + a new wire arm + saga FSM changes).
     **POST-M3 /goal AUDIT (wf_3191067b, 6 opus dimensions + adversarial-verify + synth): DONE_NO_CRITICAL — 0 CRITICAL,
     0 in-scope-unaddressed HIGH; 4 dimensions HEALTHY (compose/DRY/end-goal-readiness/HR-seal+coverage), 2 CONCERNS
     (scale-PvP + cloud, both roadmap-truth not M3 regressions); all 10 findings MEDIUM/LOW + already honestly ledgered.
     RULED: the AwaitAdopt orphan HAS siblings (D-6#1 producer-less class: AwaitAdopt / durable EmitCrossing / GhostFlow
     Despawn) but they are EXHAUSTIVELY enumerated with ONE shared root cure (redelivering transport + the D-37-2d
     re-solicit template) — R-6d is a clean bounded increment, NOT hidden rot; searched the transient tier + saga FSM for
     a 5th un-GC-d orphan, found none new. Two cheap correctness-hygiene fixes DONE this pass (zero behavior change): F1 —
     corrected the FALSE saga.rs:909 "every BatchHandoff phase is post-adopt ⇒ zero loss" comment to flag the AwaitAdopt
     exception + point to R-6d; F7 — narrowed the orchestrator boot-warn from the stale unqualified "mesh is at-most-once"
     to name the actual residual (at-least-once for source-stays-up per R-1..R-5+M3; only a SOURCE crash in AwaitAdopt
     loses its batch until R-6d). F2 (orchestrator VD_STORE_PATH hand-rolled deny-list → reuse the tested `check_durable_
     path` allow-list, closes the emptyDir-defeats-the-guard cloud hole) FOLDS INTO R-6d1 (which opens per-node stores
     through the same helper). No re-audit required.**
     **✅ R-6d1 LANDED (1a=79439d8; 1b this commit). R-6d1a: `crates/io-prod/src/outbox.rs` — the durable outbox STORAGE
     PRIMITIVE (OutboxKey 26-byte BE key with a stable class byte-map [not the implicit discriminant]; the OutboxSink
     seam retain/release/commit/scan_all/gc_below; redb-backed NodeOutbox with a SYNCHRONOUSLY-DURABLE commit = the
     durable-before-send gate; opaque already-framed value behind a 1-byte format-version envelope, HR1-clean). 8 tests;
     Tier-B outbox.rs 99.56% region/100% branch. R-6d1b: `open_node_outbox` bins helper (VD_OUTBOX_PATH; the SAME
     `check_durable_path` guard — allow-list under VD_STORE_DURABLE_ROOT else temp deny-list; VD_OUTBOX_EPHEMERAL_OK
     escape; absent→None so step_tick stays byte-identical) + 4 tests. **F2 CLOSED: the orchestrator VD_STORE_PATH guard
     now uses `check_durable_path` (allow-list catches a k8s emptyDir the old deny-list missed) + `parse_bool_env` (DRY);
     the shared guard's Ephemeral message genericized (`*_EPHEMERAL_OK`, serves 3 callers).** Gate: bins lib 9, io-prod
     lib+boot 9, `just orch-crash` (boot_guard/orchestrator_crash/boot_counter_crashloop all green — F2 regression-safe),
     clippy/fmt clean, Tier-B TOTAL 94.28%. NEXT = R-6d2 (the additive `send(Durability)` seam across all 4 Transport
     impls + the ReliableLaneSender write-through/delete-through wiring NodeOutbox into the FSM).**
     **POST-R-6d1 /goal AUDIT (wf_225ebe05, 6 opus dims + verify + synth): DONE_NO_CRITICAL — ALL 6 dimensions HEALTHY
     (incl. the sharpened DRY/modular/elegant lens), 0 CRITICAL, 0 unaddressed HIGH; 5 MEDIUM (3 R-6d2/d3-gated, 2
     scale ceilings owed at cloud) + 7 LOW, all ledgered. Ruled R-6d1+F2 a SOUND/DRY/modular/elegant foundation for
     R-6d2/d3 (check_durable_path is THE ONE durable-path guard; NodeOutbox a clean RedbStore facade; OutboxSink
     right-sized; commit() durable-before-send gate verified correct). ✅ FOLDED MEDIUM-1 (this pass): `MsgClass` is
     de-facto WIRE-FROZEN (rides every postcard frame via its variant index + the outbox key byte) but had NO reorder
     guard, unlike `class_to_byte`/`intershard.rs` — added an APPEND-ONLY doc warning + `msgclass_wire_discriminant_is_
     frozen_append_only` golden test (postcard [0]..[6], Tier-A) so a reorder fails the build. **TWO items KEPT ON THE
     R-6d2 CRITICAL PATH (do NOT defer past it): (i) DONE now — the MsgClass wire-freeze test; (ii) the §7 producer-less
     `Durability` marker CONFORMANCE test (exhaustive-over-InterShardFlow that every producer-less reliable arm carries
     the marker, turning a forgotten-marker silent-loss into a build failure) — lands WITH R-6d2 when the marker exists.**
     No re-audit required.**
     **R-6d2 DESIGN VETTED (design-refinement+3-review+synth wf_ad6e7936; spec scripts/r6d2_vetted_design.md). Verdict
     REVISE — architecture SOUND but a CRITICAL caught PRE-CODE: the design agent's §7 conformance test was VACUOUS
     (a `Transfer(env) => match { TransientBatch => ProducerLess, _ => ReDriven }` wildcard would silently swallow a
     future producer-less payload, e.g. a durable Signal/block-edit forward — false assurance on the exact generalize
     axis). Corrected + 2 HIGH (incomplete send-caller + OutboundBox-push-site enumeration). RESOLUTIONS: Durability
     enum {Ephemeral(default),Retained} in sim::io; `send` grows a trailing `durability: Durability` LOCKSTEP across
     ALL 6 impls (Mem/Fabric/PerPeerLanes/Mock ignore; Prod/Mesh→OutFrame.durability) + ~70 caller sites (all Ephemeral
     except the 2 push_flow producer-less sites); OutboundBox 4-tuple + push_flow param (Tier-A sim/runtime.rs, NOT
     io-prod). (A) RULING: DEFER the real shared sink to R-6d3 — thread the sink param through only the SYNC FSM
     (assign_and_retain/on_ack, MockOutboxSink-tested), pass None in prod, do NOT thread inert plumbing through the
     async write_frame/peer_writer. (B) RULING: a wildcard-free `durability_class()` classifier in vd-wire (sibling of
     effect_class) — a new arm/GhostFlow/TransitionPayload variant fails to compile until classified — + a rig-driven
     marker test. **✅ R-6d2a LANDED (this commit): `FlowDurabilityClass{ReDriven,ProducerLessReliable,Unreliable}` +
     `InterShardFlow::durability_class()` (wildcard-free, exhaustive at every nesting level) in wire/intershard.rs +
     the `durability_class_pins_the_producer_less_reliable_set` golden test (drives every_arm, pins the producer-less
     set = EXACTLY {Ghost::Despawn, Transfer(TransientBatch)}). This is the ANTI-VACUITY core: a future durable Signal
     is a new TransitionPayload → compile-forced classification → the marker test fails unless its push carries Retained.
     Tier-A 100% (intershard.rs 100% region/branch/fn), clippy clean.** NEXT = R-6d2b (the big churn: the Durability
     seam + all 6 impls + ~70 callers + OutboundBox/push_flow + the FSM write-through + MockOutboxSink tests + the
     rig-driven marker-on-push test). THEN R-6d3 (the durable-before-send GATE in the writer + boot replay + the C2/C3
     saga+dest closure) → R-6d4 (both-ends-restart proptest + SIGKILL-source-in-AwaitAdopt proof).**
     **✅ R-6d2b LANDED (the Durability seam threaded end-to-end): `Durability{Ephemeral(default),Retained}` in sim::io;
     `OutboundBox` is a 4-tuple carrying it; the mesh `OutFrame` + `write_frame` lowering consume it; `push_flow_durable`
     at the 2 producer-less sites (stub TransientBatch + band-exit Ghost::Despawn) push `Retained`, all other pushes
     default Ephemeral. ⚠️ DESIGN DEVIATION FROM THE VETTED R-6d2 SPEC (owned, flagged for post-impl review): the spec
     chose EXPLICIT 4-arg `send`/`push_flow` at EVERY call site (~150 sites, `.send(` unscriptable due to mpsc/channel
     ambiguity). Instead pivoted to 3-arg `send`/`push_flow` DEFAULTS delegating `Ephemeral` + `send_durable`/
     `push_flow_durable` for the rare producer-less exceptions — MORE DRY (the 99% common case is a clean 3-arg call),
     far less churn, and the marker-on-push CONFORMANCE test is now the load-bearing guarantee (it drives the real stub
     rigs + asserts the 2 producer-less flows push Retained via `durability_class`). Landed the `producer_less_reliable_
     flows_push_with_the_retained_marker` test (Tier-A) + a `Rig::tick_raw` durability-preserving harness helper.
     DEFERRED to R-6d2c: the FSM WRITE-THROUGH itself — `assign_and_retain(.., durable, sink)`/`on_ack(.., sink)` +
     the `OutboxSink` handle injection + MockOutboxSink FSM tests (today `write_frame` LOWERS `frame.durability` to a
     `durable` bool but the sink is not yet wired — the `let _durable` bridge). Gate: workspace clippy -D clean (20m
     rebuild), vd-sim 82+conformance, vd-node 190, vd-wire 63, full `--tests` compiles.
     **✅ POST-IMPL REVIEW DONE (wf_ec38e817, 3 opus lenses + synth): verdict FIX_BEFORE_COMMIT + pivot ruling
     KEEP_WITH_HARDENING (do NOT revert). The pivot is structurally sound — the frozen seam is correct additive-
     lockstep (`send` default delegates `send_durable(Ephemeral)`, non-recursive; all 6 impls satisfy send_durable), the
     20Hz datagram hot path is BYTE-IDENTICAL (durability read only in the Reliability::Reliable arm), HR1/HR3/HR5 all
     MET, the DRY win real. But a reviewer EMPIRICALLY PROVED a HIGH: a future NESTED producer-less variant (the exact
     P9 Signal / P6 BlockEdit path — a new TransitionPayload/GhostFlow) classified ProducerLessReliable slips the wire
     golden pin VACUOUSLY (`every_arm` hand-maintained + `arm_tripwire` outer-only ⇒ len()==2 passes), shipping pushed
     Ephemeral = D-6 #1 silent-loss with a GREEN suite. FIXES FOLDED before commit: (1) NESTED-variant tripwires
     `payload_tripwire`/`ghost_tripwire` (wildcard-free, intershard_closed.rs) — a new nested variant now FAILS
     COMPILATION until classified (exhaustive-by-construction, the load-bearing fix); (2) a RUNTIME guard debug_assert
     in `push_flow_durable` (a ProducerLessReliable flow pushed Ephemeral panics — fires on EVERY producer-less push in
     EVERY debug/test build, stronger than explicit-4-arg which forced only that SOME value be stated; release hot path
     untouched) + a `#[should_panic]` test proving it + covering the panic arm; (3) gateway LOW-1 note (a future gateway
     producer-less flow MUST use push_flow_durable). Gate after hardening: vd-wire 3 (tripwires) + vd-sim 191 (guard +
     conformance) + workspace clippy -D clean + Tier-A 100% region/branch/fn. CONFIRMED (LOW-2): R-6d2b is
     plumbing+classification ONLY — the `let _durable` bridge (mesh.rs) is unconsumed; the actual OutboxSink write-
     through + boot replay that CLOSE D-6 #1 are R-6d2c/R-6d3 (a source crash of a TransientBatch/Despawn is STILL lost
     today; this slice only prepares the cure). NEXT = R-6d2c (assign_and_retain/on_ack sink + OutboxSink injection +
     MockOutboxSink FSM tests).**
     **✅ R-6d2c LANDED (the FSM durable write-through / delete-through — io-prod-ONLY, per vetted design wf_5ca0a751,
     SOUND_TO_IMPLEMENT with 1 HIGH + 1 MEDIUM folded): `ReliableLaneSender` grows `peer: NodeId` (the DEST — the
     `OutboxKey.peer`, distinct from `ReliableFrame.from`=local sender; `on_ack` had no `NodeId` otherwise) + a
     `RetainedFrame.retained: bool` (the strict-subset flag) + a ONE-home `outbox_key(class, seq)` helper (MED-1 fix,
     single-sources peer+incarnation). `assign_and_retain(from, class, bytes, durable, sink)` mirrors ONLY a Retained
     frame to the `OutboxSink` (`durable && sink.is_some()`) with the epoch=u32::MAX framed value (the L2 replay
     contract — byte-identical to `write_reliable_frame`, `decode_frame`-round-trippable, no re-encode); both rejects
     (Unframable/BufferFull) PRECEDE the insert so a shed frame is never mirrored (outbox ⊆ retained ⊆ never-shed).
     `on_ack(.., sink)` releases ONLY the `rf.retained` subset in the retire loop (an Ephemeral-heavy Saga lane stages
     no tombstone per acked ephemeral seq); the incarnation/epoch guard still returns 0 BEFORE the loop (a stale ack
     releases nothing). §b ruling CONFIRMED against code: `replay_batch` (`&self`) + `on_write_error` re-SEND but never
     re-RETAIN, and OutboxKey excludes epoch, so a redial epoch-bump neither doubles nor orphans a row. PROD PATH
     UNCHANGED: `write_frame` now consumes the `durable` bool (was `let _durable`) and passes `(durable, None)`; the
     peer_writer passes `on_ack(.., None)` — the sink is `None` until R-6d3 injects the real `Arc<Mutex<NodeOutbox>>`,
     so a Retained send is BYTE-IDENTICAL to Ephemeral today (no outbox row written). Tests: MockOutboxSink + T1–T7
     (retain-exact-key+value / durable-false-no-record / no-sink-noop / on_ack-releases-only-retained /
     no-sink-releases-nothing / stale-ack-releases-nothing / redial-no-double-retain) + T8 real-`NodeOutbox` glue
     (scan_all→decode_frame DIRECTLY, LOW-2 fix — no double-strip) + T9 integration (a `send_durable(.., Retained)` over
     real QUIC loopback drives the `matches!` TRUE arm in `write_frame`, HIGH-1 fix — the FSM unit tests never touch
     `write_frame`). The ~57 test-caller sweep (26 `new`+PEER / 38 `assign_and_retain`+`false,None` / 16 `on_ack`+`None`)
     was a paren-matching transform restricted to the test module (all 3 prod call sites untouched). Gate: vd-io-prod
     100 lib + all integration green, clippy -D clean, Tier-B floor `--fail-under-regions 90` PASS (TOTAL 94.47%,
     mesh.rs 94.70%), workspace build green. R-6d3 PRECONDITIONS carried (LOW-3: R-6d2c STAGES retains but never
     `commit()`s — R-6d3 MUST call `commit()` before the QUIC send or every staged retain is inert-not-on-disk;
     LOW-4: the SAME sink MUST reach both `assign_and_retain` AND `on_ack` for a lane or a durable row leaks). NEXT =
     R-6d3 (the durable-before-send fsync GATE + the real `Arc<Mutex<NodeOutbox>>` sink injection through peer_writer +
     boot replay `scan_all`→re-drive + `gc_below` sweep + the saga/dest AwaitAdopt closure — this actually CLOSES
     D-6 #1 — then R-6d4 the both-ends-restart replay proptest + the SIGKILL-source-in-AwaitAdopt e2e proof).**
     **✅ POST-IMPL REVIEW DONE (wf_38cdb045, 3 opus lenses [FSM-correctness / test-rigor / HR+seam-integrity] +
     synth): verdict COMMIT_CLEAN — NO CRITICAL/HIGH, NO production correctness bug; prod provably byte-identical
     (both call sites pass literal `None` ⇒ `retained = durable && sink.is_some()` is always false in prod). The
     retained-subset invariant is airtight by code trace (retain 1 site, release 1 site, same `retained` predicate,
     `retry.remove` co-located with release, `outbox_key` single-sourced, epoch deliberately NOT a key field so
     redials can't diverge keys, `encoded` owned, both shed rejects precede the write-through). The review's real
     catch (MEDIUM F1, this project's HR5 real-not-theater standard): the MockOutboxSink tests were MUTATION-BLIND —
     `on_ack` releases on the stored `retained` BOOKKEEPING bit, so neutering the actual `s.retain()` STORE write
     left T4/T5/T6 green (T8 never acked ⇒ no full lifecycle through redb). FOLDED BEFORE COMMIT (cheap, reuse T8
     scaffolding): (F1) T8 extended — after retain+commit+scan(len==1) it now `on_ack(.., Some(&mut ob))`+commit and
     asserts `scan_all().is_empty()`, driving the FULL retain→ack→release→empty lifecycle through REAL redb (catches
     either half becoming a no-op + pins the retain↔delete key match); (F2) T8 now asserts the FULL stored value
     `== expected_encoded(..)` through the real `encode_value`/`decode_value` envelope (not just `rf.bytes` — R-6d3
     boot-replay dedups on from/incarnation/seq); (F3) new T10 pins the multi-release `sink.as_deref_mut()` re-borrow
     (2 durable seqs retired by 1 ack ⇒ both released in order — a `sink.take()` regression would release only seq0
     and slip every other test); (F4) code comment noting the write-through's before-insert placement is an
     INTENTIONAL deviation from wf_5ca0a751 §3 (behaviour-identical, sidesteps move-then-borrow). F5 (T9 proves
     region-coverage not durable-behaviour) needs no change — already honestly documented. Gate after fold: vd-io-prod
     101 lib (+T10) + all integration green, clippy -D clean, Tier-B `--fail-under-regions 90` PASS (TOTAL 94.46%,
     mesh.rs 94.67%), workspace + vd-bins build green.**
     **✅ R-6d3a LANDED (the durable-before-send GATE + real shared-outbox injection — io-prod + a `None` arg at every
     bin, per vetted design wf_e8bea726 SOUND_TO_IMPLEMENT): `spawn_mesh` gains `outbox: Option<SharedOutbox>` where
     `SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>` — ONE store shared by every per-peer writer task
     (`OutboxKey.peer` keeps rows disjoint). `write_frame`'s Reliable arm is now BLOCK A (lock the sink for THIS span
     only: `assign_and_retain` retain + a NON-BLOCKING `submit_barrier`; guard dropped) → BLOCK B (OUTSIDE the lock, on
     a cloned `DurabilityHandle`: `wait_durable_through(batch_seq)` so the row is fsynced BEFORE the wire) → BLOCK C
     (the verbatim QUIC send). Ephemeral / no-sink ⇒ no lock, `gate=None`, byte-identical fast path (the 20Hz datagram
     path untouched). `peer_writer` locks the SAME `Arc` once per ack fan-out and threads it into `on_ack` (release-
     through, LOW-4). ⚠️ MF-3 CONCURRENCY DEVIATION FROM THE VETTED SYNTH (implementer finding, adversarially verified
     opus, FLAW CONFIRMED + FIX SOUND): the synth's MF-1 fix (`submit_nonblocking` + depth-2 channel + try_send-panic-
     on-Full) PANICS under ≥3 concurrent distinct peer-writers piling batches into a depth-2 channel before one fsync
     drains (the "≤1 in-flight" proof is per-writer-iteration, NOT per-node — exactly the "source fans TransientBatch to
     many dest peers + Despawn to neighbors" load). CURE: the OUTBOX store gets its OWN generous
     `OUTBOX_WRITER_CHANNEL_DEPTH` (named const 256 ≫ peer count, HR no-magic-numbers) so `Full` is an impossible-
     capacity tripwire not a load path; `submit_nonblocking` stays block-on-prior-FREE with seq-assign+`try_send` UNDER
     the lock (the sole out-of-order-send hazard avoided); block B waits on a SINGLE `bar.seq` (durability monotone ⇒
     subsumes prior). The outbox needs NO depth-1 crash-loss bound (unlike the orchestrator store): an un-fsynced
     in-channel batch is an un-SENT frame (the gate withholds the wire until durable), re-emitted by the restarted
     source — proven leg-by-leg (durable-before-send / seq-durability monotonicity / crash-safety; no false-durable, no
     out-of-order, bounded memory). New: `store.rs` `submit_nonblocking` + `DurabilityHandle::already_durable()` test-
     ctor; `outbox.rs` `OUTBOX_WRITER_CHANNEL_DEPTH` + `submit_barrier` + open forces the depth; `mesh.rs` `SharedOutbox`
     + the gate + `MockOutboxSink::submit_barrier`. Tests: outbox `submit_barrier` unit (monotone durable seqs +
     None-on-empty) + T-DBS-1 (real `NodeOutbox` + real QUIC: a Retained send is durable on disk BEFORE delivery,
     released on ack through the shared sink) + T-DBS-2 (Ephemeral fast path leaves the outbox empty). SCOPE: bins pass
     `None` (inert-safe like 2c) — R-6d3b opens+boot-replays the real store; so D-6 #1 is STILL open (3a builds the gate
     but no bin writes to it yet). Gate: vd-io-prod 102 lib + all integration green, clippy io-prod + vd-bins -D clean,
     Tier-B `--fail-under-regions 90` PASS (TOTAL 94.05%), workspace build green. NEXT = R-6d3b (open the per-node store
     in shard/gateway `main` + the boot replay `scan_all`→re-drive→`gc_below` + the tick-loop split) — THEN R-6d3c (the
     saga AwaitAdopt split + dest TransientDiscard) actually CLOSES D-6 #1, then R-6d4 the SIGKILL e2e + proptest.**
     **✅ POST-IMPL REVIEW DONE (wf_cfdde7cc, 3 opus lenses + synth): verdict COMMIT_CLEAN — the MF-3 fix is re-derived
     SOUND in the landed code (seq-assign+try_send atomic under the lock ⇒ FIFO seq-ordered drain + monotone
     last_durable; guard dropped before block B ⇒ no cross-peer serialization; depth 256 ⇒ Full unreachable for ≤256
     peers; Ephemeral byte-identical). NO CRITICAL, NO MUST-FIX-BEFORE-COMMIT — every finding ships INERT (all bins pass
     `None`). FOLDED BEFORE COMMIT (cheap, improve THIS slice): **F3** a fail-loud boot check in `spawn_mesh` (peer count
     > `OUTBOX_WRITER_CHANNEL_DEPTH` ⇒ Tuning error, ONLY when an outbox is wired) — closes the silent panic-at-scale
     ceiling; **F4** the durable-path None-barrier is now a RELEASE-safe fail-loud `WriteFail::Down` (was a debug-only
     assert compiled out of release ⇒ a store-mid-Drop race could send un-durable); **F5** `#![warn(clippy::
     await_holding_lock)]` on io-prod (mechanical guard for the guard-dropped-before-await invariant; verified clean
     crate-wide); **F1-doc** T-DBS-1 no longer overclaims a deterministic block-B ordering pin (it exercises the regions
     + proves durable-write-through-to-redb, but the off-tick writer races `scan_all` so a block-B deletion is caught
     only ~1/8 — the DETERMINISTIC ordering pin is R-6d4's `pause_on_key_prefix`); **F7** T-DBS-2 commits before scanning.
     ⚠️ TWO HARD GATES ON R-6d3b (the slice that flips `None`→live sink) — NOT commit blockers for 3a (gate inert): **F2**
     the block-B `wait_durable_through` is a SYNCHRONOUS condvar park; on the 2-worker tokio runtime two peer-writers
     parked in block B occupy BOTH workers and STARVE the mesh RECV/ACK/ACCEPT I/O path for the fsync duration (the sim
     tick loop is on the MAIN thread, NOT starved — reviewer self-corrected). R-6d3b MUST land one of: `spawn_blocking`
     the park / an async `Notify`-based wait / raise `worker_threads` (documented) + a ≥2-concurrent-durable-send
     progress test. **F1-pin** land the deterministic `store-test-hooks` `pause_on_key_prefix` ordering proof (row NOT in
     `scan_all` until block B waits) — R-6d4 (or R-6d3b). Gate after folds: vd-io-prod 103 lib/unit + all integration
     green, clippy io-prod (+await_holding_lock) + vd-bins -D clean, Tier-B PASS. (F6: store.rs region % is mostly
     pre-existing helper/monomorph surface + the new panic tripwires — not a regression; floor is on TOTAL.)**
     **✅ R-6d3b DESIGN DONE (wf_ab97227a, verdict REVISE with all fixes folded; F2 async-wait SOUND_TO_IMPLEMENT).
     Sub-sliced: R-6d3b-1 (io-prod, LANDED below) = the F2 async block-B wait; R-6d3b-2 = bin wiring + boot replay via
     an io-prod-internal `NodeOutbox::replay_all` (folds finding A `pub(crate)` `OutboxKey`/`ReliableFrame` ⇒ the
     replay MUST live in io-prod, not the bin; B `QueueFull` no-swallow + a pre-send peer-membership check; C resolve
     the incarnation ONCE — re-calling `resolve_process_incarnation` double-increments the durable BootCounter; D
     fence on the `last_submitted` high-water not a `scan_all` count; STRICT gc-after-all-durable, HIGH-3). F1 (the
     deterministic `pause_release` ordering pin) + `two_peers_do_not_starve_workers` + the writer-death-wake test →
     R-6d4 (store-test-hooks tier). R-6d3c (saga AwaitAdopt split + dest TransientDiscard) = the NEVER-restart closure
     that actually retires D-6 #1.**
     **✅ R-6d3b-1 LANDED (io-prod ONLY — the F2 fix, makes block B production-safe; inert in prod, bins still None):
     the durable-before-send block-B wait is now ASYNC so a stalled producer-less durable send YIELDS its tokio worker
     instead of parking it — closing the R-6d3a F2 gate (on `worker_threads(2)`, two peer-writers parked in a SYNC
     durability park occupy BOTH workers + starve the mesh recv/ack/accept I/O). `DurabilityHandle` gains
     `durable_notify: Arc<tokio::sync::Notify>` + a NEW `wait_durable_through_async(seq)` (fast-path ⇒ return; else
     fetch_add backpressure then loop{ enroll `notified()` BEFORE the re-read (lost-wakeup-free) → recheck →
     `timeout(WRITER_WAIT_POLL, notified).await` → on elapsed fail-loud iff writer dead }). The writer
     `notify_waiters()` AFTER `last_durable.store(Release)` alongside `cv.notify_all()`; `WriterExitSignal::Drop` fires
     it too (finding-E prompt death-wake). The SYNC `wait_durable_through` is BYTE-UNCHANGED — the orchestrator
     persist-before-effect gate (orchestrator.rs:280, MAIN thread) is untouched; `durable_notify` is NOT a `RedbStore`
     field (it never reads it — no dead code), only cloned into the writer + handle at `open`. mesh.rs block B:
     `wait_durable_through_async(batch_seq).await` (the guard from block A still drops BEFORE it — the
     `#![warn(clippy::await_holding_lock)]` lint now MECHANICALLY proves that). Tests: 2 store.rs async-wait unit tests
     (fast-return-when-durable; returns-after-a-real-submit-fsynced) + T-DBS-1 runs the async block-B PATH end-to-end
     over real QUIC. Gate: vd-io-prod 105 lib + all integration green, clippy -D (incl await_holding_lock) clean,
     Tier-B `--fail-under-regions 90` PASS (TOTAL 94.09%, store.rs 90.43%↑), workspace build 0.
     **✅ POST-IMPL REVIEW DONE (wf_75a225d0, 3 opus lenses + synth): verdict COMMIT_CLEAN — concurrency-fidelity +
     no-regression lenses both CLEAN (enroll-`notified()`-BEFORE-recheck ⇒ lost-wakeup-free; writer stores(Release)
     THEN notify_waiters; timeout backstop + writer-death panic bound liveness; cancel-safe at the sole `w.rx.recv()`
     cancel point; SYNC `wait_durable_through` byte-unchanged ⇒ orchestrator untouched; `durable_notify` correctly NOT
     a `RedbStore` field ⇒ no dead code; `await_holding_lock` mechanically proves the guard drops before block B). ONE
     real finding, dispositioned FIX-IN-R-6d4 (NOT a commit blocker): the 2 async tests + T-DBS-1 are MUTATION-BLIND to
     the async WAKE — a reviewer NEUTERED `notify_waiters()` and BOTH async tests still passed (they fell into the
     100ms `WRITER_WAIT_POLL` re-read); T-DBS-1 pins the durable-before-send OUTCOME, which BOTH the sync and async
     waits satisfy, so it does NOT distinguish the swap. A reliable wake-mechanism / non-serialization regression guard
     needs the `store-test-hooks` writer-pause (a default-tier latency test would be racy) ⇒ OWED at R-6d4 alongside F1
     (the deterministic ordering pin) + `two_peers_do_not_starve_workers` + the writer-death-wake test. The landed CODE
     is concurrency-correct (the coverage gap is test-theater honestly deferred, not a defect); the async timeout/panic
     arms ride the Tier-B floor like the sync sibling (F-2, R-6d4). NEXT = R-6d3b-2 (flip the sink live: open_node_outbox
     → SharedOutbox → spawn_mesh in shard/gateway main + the boot replay).**
     **⚠️ R-6d3b-2 DESIGN CORRECTED before impl (implementer + opus verifier — TWO CONFIRMED HOLES in the vetted §2
     `replay_all`; full corrected design in scripts/r6d3b_vetted_design.md tail). HOLE 1 DEADLOCK (certain): §2 called
     `shared.lock().replay_all(&mut self, transport, ..)` holding the shared `std::sync::Mutex` across send+fence, but a
     replayed `send_durable(Retained)` re-enters a peer_writer's block A which re-`.lock()`s the SAME mutex to
     re-mirror ⇒ the peer_writer blocks its tokio worker, the fence waits forever for a durability bump only that
     blocked writer can make (2 blocked writers also exhaust worker_threads(2)). HOLE 2 PREMATURE-GC DATA LOSS: §2's
     high-water fence `wait_durable_through(last_submitted())` reads `last_submitted` ONCE, but it bumps ASYNC as
     peer_writers reach block A ⇒ the fence can pass with only k<N fresh rows submitted, then `gc_below` sweeps the OLD
     rows whose fresh re-mirror never landed = the exact D-6 #1 loss. CORRECTED DESIGN (verified deadlock-free + no
     premature-gc): a FREE fn `replay_outbox(&SharedOutbox, transport, peers, new_incarnation)` — LOCK-SCOPED: brief
     lock to snapshot rows + a `DurabilityHandle` clone + `base=last_submitted` (⇒ ADD `fn durability(&self) ->
     DurabilityHandle` to `OutboxSink`); RELEASE; lock-free decode+peers-membership-check+`send_durable_with_retry`
     (peer_writers re-mirror freely — no deadlock); lock-free COUNT-ANCHORED fence `last_submitted >= base + N` (each
     replay send = exactly one submit; a previously-retained row re-frames IDENTICALLY ⇒ N is exact, no Unframable
     shed), fail-loud-bounded, THEN `wait_durable_through(base+N)`; then ONE brief lock for atomic `gc_below`+`commit`.
     Sub-slice for impl: R-6d3b-2a (io-prod replay MACHINERY + a real-QUIC test proving re-drive+deliver+gc WITHOUT
     deadlock/premature-gc), R-6d3b-2b (bin wiring: shard/gateway main open+wrap+pass + the boot-replay call +
     resolve-incarnation-ONCE, finding C).**
     **✅ R-6d3b-2a LANDED (io-prod ONLY — the boot-replay MACHINERY, from the corrected design; inert until 2b wires
     the bins): `SharedOutbox` moved to `outbox.rs` (pub) + `fn durability(&self) -> DurabilityHandle` added to
     `OutboxSink` (NodeOutbox → its clone; MockOutboxSink → `already_durable()`); `ReplayError` +
     `REPLAY_SEND_MAX_RETRIES`/`REPLAY_POLL_BACKOFF`/`REPLAY_FENCE_DEADLINE` (named, one home); `decode_value_payload`
     (decode_frame→`.bytes`, the PAYLOAD not the frame — finding A, keys read in-crate); `send_durable_with_retry`
     (peers-membership pre-check ⇒ `Unroutable` loud; QueueFull ⇒ bounded retry ⇒ `LaneStuck` loud; NEVER swallows —
     finding B); and the pub `replay_outbox(shared, transport, peers, new_incarnation)` — LOCK-SCOPED (brief scan-lock
     to snapshot rows + a `DurabilityHandle` clone + `base=last_submitted` → RELEASE → lock-free decode+route+send →
     lock-free COUNT-anchored fence [`last_submitted >= base+N` bounded by `REPLAY_FENCE_DEADLINE`, then
     `wait_durable_through(base+N)`] → ONE brief lock for atomic `gc_below`+`commit`), so the peer-writers re-mirror
     the replayed sends FREELY (HOLE-1 deadlock cured) and gc runs STRICTLY after all N fresh rows are durable (HOLE-2
     premature-gc data-loss cured; N exact — a previously-retained row re-frames identically ⇒ none shed). Tests:
     outbox `decode_value_payload` (recover+reject-garbage) + `send_durable_with_retry` (QueueFull×3-then-succeed via
     a FlakyTransport mock) + a real-QUIC end-to-end `replay_outbox_redrives_retained_rows_delivers_and_gcs_the_prior_
     incarnation` (pre-seed 2 rows @incarnation-1 → A@incarnation-2 wired to the shared outbox + B → replay → B gets
     both payloads + scan_all has NO incarnation<2 rows; the test COMPLETING is the no-deadlock proof). Gate:
     vd-io-prod 108 lib + all integration green, clippy -D clean, Tier-B `--fail-under-regions 90` PASS (TOTAL 94.29%,
     outbox.rs 97.71%), workspace build 0. `resolve_process_incarnation` is NOT called here (finding C — the bin passes
     `new_incarnation` in 2b). NEXT = R-6d3b-2b (shard/gateway `main`: `open_node_outbox` → wrap `SharedOutbox` → pass
     to `spawn_mesh` + call `replay_outbox` before `build_app`, resolve-incarnation ONCE) ⇒ CLOSES the D-6 #1 RESTART
     case.**
     **✅ POST-IMPL REVIEW DONE (wf_7183b0d1, 3 opus lenses + synth): verdict COMMIT_CLEAN — BOTH headline holes
     VERIFIED FIXED in the landed code (the reviewer confirmed the +1-per-submit mechanic the fence relies on:
     `submit_nonblocking` bumps `last_submitted` by exactly one per block-A submit; the ack/release path stages a
     delete but NEVER submits; so `base+N` is reached iff all N fresh rows submitted ⇒ no premature gc; the lock is
     held only for the brief scan-snapshot + brief gc, never across send/fence ⇒ no deadlock). Findings A/B/C/D hold.
     FOLDED before commit (cheap, default-tier — the boot-safety proofs this slice EXISTS for): the LOW
     `base.checked_add(n).expect(..)` overflow guard (disarms the `already_durable` u64::MAX foot-gun for 2b's own
     mock tests) + two abort-before-gc tests (`replay_outbox_bails_on_an_unroutable_row_without_gc` +
     `..._undecodable_..._without_gc` — each asserts `Err` AND the prior-incarnation rows SURVIVE, gc skipped) + the
     doc NIT (BufferFull-shed is fail-SAFE: no submit ⇒ FenceTimeout ⇒ gc skipped ⇒ no loss). DEFERRED to R-6d4
     (store-test-hooks tier): the mutation-proof no-premature-gc DETERMINISTIC pin (writer-pause) alongside F1. Gate
     after folds: vd-io-prod 110 lib + all integration green, clippy -D clean, Tier-B PASS (TOTAL 94.40%, outbox.rs
     98.29%), workspace build 0.**
     **✅ HOLISTIC /goal AUDIT (wf_c158c7d0, 6 opus dims + synth, HEAD 1789e49): DONE_NO_CRITICAL — 5/6 SOUND
     (compose / scale / DRY-HR / robust-crash / feature-readiness), 1 CONCERNS (test-coverage) whose only HIGH is a
     MISSING future LOAD gate (D-9), not a broken thing. No CRITICAL, no HIGH-that-is-a-real-defect-in-the-built-
     surface. Rewarded: the compiler-closed InterShardFlow + runtime debug_assert (future Signal/BlockEdit silent-loss
     = build failure), the count-anchored no-premature-gc fence (every error arm ?-bails before gc — verified
     fail-safe), the SIGKILL-mid-fsync orchestrator proof. BINDING FORWARD INPUTS (latent TODAY — bins pass None — but
     go LIVE the moment R-6d3b-2b flips a bin off None; fold into the 2b/2c design BEFORE wiring):
     • R-6d3b-2b RC-2a: `replay_outbox` error DISPOSITION must honor the OUTBOX_FORMAT_VERSION quarantine promise
       (outbox.rs:47) — QUARANTINE + loud-count + boot-PROCEED for `Undecodable` (a permanent poison row) and a
       roster-diff `Unroutable` (a peer legitimately gone from the book); refuse-to-boot ONLY for transient
       `LaneStuck`/`FenceTimeout`. Else a fail-loud 2b boot WEDGES on one poison/absent-peer row. + a poison-row
       boot-progress test.
     • R-6d3b-2b RC-2b: `send_durable` folds `TrySendError::Full | Closed` into ONE `QueueFull` (mesh.rs:1913-1916) —
       split `Closed` → `Inbound::NodeUnreachable`/a LaneDead signal (+ observe the peer_writer `JoinHandle`s) so a
       DEAD lane surfaces as unreachable, not transient backpressure; else `send_durable_with_retry` spins ~10s on a
       corpse before `LaneStuck` instead of failing fast.
     • R-6d3c RC-3 (cheap pre-stage available now): split the `rehome_event_for` BatchHandoff phase-wildcard
       (saga.rs:922) — AwaitAdopt on a confirmed-dead source → `Timeout`/PARK (admin-visible) NOT `SourceUnreachable`/
       self-promote; removes the only in-code dead-source→silent-promote path ahead of the full R-6d3c closure.
     • RC-1 (correctly P6-deferred, NOT blocking): the hundreds-in-one-location snapshot LOAD gate — `emit_frames`
       (stub.rs:2752) is whole-realm-broadcast O(entities×clients), largest fixture CLIENTS=32; land the N-in-one-realm
       ratcheted-bytes fixture before any dense-PvP soak, the AoI reshape at P6 (D-9, DEFERRED.md).**
     **✅ R-6d3b-2b LANDED (the durable outbox is now LIVE in the bins — CLOSES the D-6 #1 RESTART case; per design
     wf_e91a84e5 + post-impl review wf_d926fe08 verdict COMMIT_CLEAN): Sub-slice A (RC-2a, outbox.rs) — `replay_outbox`
     → `Result<ReplayCounts{replayed,quarantined}, ReplayError{LaneStuck,LaneDead,FenceTimeout}>`; a roster-gone peer
     or an undecodable frame is QUARANTINED (warn+count+continue, RETAINED) so the boot PROCEEDS (one bad row never
     wedges the node); the count-fence targets `base + replayed` (NOT rows.len() — a quarantined row never submits);
     gc uses the NEW `gc_replayed(&replayed_keys)` [sweeps ONLY the re-driven keys] not `gc_below` — the F1 fix (my
     original `gc_below(new_incarnation)` was a D-6 #1 regression that would sweep a recoverable roster-gone row);
     `replayed==0` early-returns with no fence + no gc; the `new_incarnation` param dropped (gc is key-based).
     Sub-slice B (RC-2b, mesh.rs) — a NEW pub supertrait `ReplayTransport: Transport` + `lane_alive` (via
     `tx.is_closed()`); `send_durable_with_retry` fast-fails `LaneDead` on a dead lane (no ~10s corpse-spin);
     `send_durable` UNCHANGED (Full|Closed fold to QueueFull with NO msg_id consumed — F2; the FROZEN sim::io seam is
     untouched, verified by an empty `git diff crates/sim/src/io/`). Sub-slice C (bins) — a NEW DRY
     `vd_bins::boot_mesh_and_replay(env, runtime, trust)` used by BOTH shard + gateway (HR3): resolves the incarnation
     ONCE (finding C — no double BootCounter increment), opens+wraps the outbox, `spawn_mesh(Some)`, `replay_outbox`
     BEFORE `build_app`; the GW-1 assert moved before; `ReplayError` gained Display+Error. Replay fences on DURABILITY
     not DELIVERY ⇒ no boot-hang on down-at-boot peers (their rows RETAIN + the R-4a timer re-drives). Post-impl review:
     2/3 lenses COMMIT_CLEAN; the 3rd (bins-wiring) FIX_BEFORE_COMMIT was a SCOPE/coverage matter, not a defect — the
     core is correct on every no-loss/no-premature-gc/no-hang/F2/frozen-seam/resolve-once axis (independently re-traced).
     Tests: io-prod quarantine×2 (unroutable/undecodable RETAINED) + fast-fail-LaneDead + refuse-on-dead-lane-no-gc +
     the real-QUIC MIXED-F1 end-to-end (1 re-driven SWEPT + 1 quarantined RETAINED) + a bins in-process Some-path glue
     test (`boot_mesh_and_replay_wires_a_live_outbox_and_returns` — closes the F-A "glue never runs" gap; catches a
     spawn-arg/ordering regression). Gate: vd-io-prod 112 lib + all integration, vd-bins green (incl. process_parity),
     clippy -D (io-prod+bins) clean, Tier-B `--fail-under-regions 90` PASS (TOTAL 94.19%, outbox.rs 97.09%), workspace
     build 0. OWED to R-6d4 (post-impl review, honestly deferred): the process-tier VD_OUTBOX_PATH SIGKILL-restart e2e +
     boot-counter-by-exactly-1 (F-A full); the `LaneStuck`/`FenceTimeout`/overflow-expect/`Display` arm tests (F-C).
     BINDING forward-input to R-6d (when a gateway durable-to-client flow lands): departed-client peer entries in the
     gateway `VD_PEERS` would re-drive forever + trip the peer-count>256 guard (F-F) — gated behind `None` today.
     ⇒ D-6 #1 RESTART case CLOSED; the NEVER-restart case is R-6d3c (saga AwaitAdopt split + dest TransientDiscard).**
     **📐 R-6d3c DESIGN DONE (wf_9e5f126d, 3 opus lenses ALL SOUND_TO_IMPLEMENT + synth, 4 folded must-fixes) — ready
     to implement (NOT yet coded). SCOPE resolved: the PURE-LOGIC never-restart closure lands NOW; the confirm-dead
     TRIGGER is CA-1-gated. Sub-slices: R-6d3c-1 (vd-wire, Tier-A) — a NEW `InterShardFlow::TransientDiscard`
     (reusing the flat `TransientHandoff`, an OUTER arm so the nested tripwires are untouched) + `TRANSIENT_DISCARD_
     STEP=17` (APPENDED, preserves postcard discriminants) through ALL 3 closed-set gates (effect_class
     SideEffecting/TransferStep, durability_class `ReDriven` [NOT ProducerLessReliable — that would trip the
     push-Ephemeral debug_assert with no re-emitter + break the producer_less.len()==2 pin], the golden byte-freeze,
     the "16-arm" header). R-6d3c-2 (vd-sim, Tier-A) — `on_transient_discard`: REMOVE `Arriving{batch==transfer}` +
     POISON `(transfer, TRANSIENT_BATCH_STEP)` via journal_step [so a LATE outbox-replayed `adopt_transient_batch` is
     `AlreadyApplied` ⇒ never re-inserts an orphan — the ACTUAL dest cure, lands COMPLETE] + bucket the loss into
     `transients_lost_in_handover` by kind + a discard-specific counter. R-6d3c-3 (vd-sim FSM + producer) —
     `SagaEvent::SourceUnreachablePreAdopt` + `SagaAction::{EmitTransientDiscard,CountBatchLostSourceCrash}`; split the
     `saga.rs:922` BatchHandoff wildcard BY PHASE (post-adopt → self-promote unchanged; AwaitAdopt → discard-to-dest +
     count + Tombstone, terminal-on-first-fire, budget-gated on abort_deadline_ticks). MUST-FIXES folded: (M1) the
     budget-gate is a restart-window WIDENING, NOT the anti-double-resolution proof — that is the phase-structure
     (post-adopt makes the AwaitAdopt arm unreachable) + the adopt-poison (a late BatchAdopted at a Done saga is a
     terminal-absorb no-op); (M2) the `scan_deadlines` counter match has a `_ => {}` wildcard — the new counter arm
     MUST be added there (the ONE place a missing arm compiles green) + covered by BOTH a scan_deadlines test AND a
     direct-deliver test; (NIT-A) the budget-gate affected test is saga_runtime.rs:4293 (AwaitPromote) ONLY, NOT :3806
     (Demoting); the EntityKind::from_tag Err arm needs a test (Tier-A 100% branch). NET NOW: the dest can NEVER strand
     an Arriving orphan regardless of replay timing + the saga resolves to a bounded accounted terminal. STILL
     CA-1-gated (⇒ D-6 #1 never-restart "closed ON delivered-discard"): the AwaitAdopt RE-SOLICIT egress (idempotent
     orch→source re-prompt) that makes `is_confirmed_dead(source)` REACHABLE in AwaitAdopt (fires the discard) AND the
     lost-discard reliability (a lost fire-once discard + a late batch replay re-orphans — same CA-1 reliability gate
     as the trigger). Full crash proof (SIGKILL-source-in-AwaitAdopt + zero-orphaned-Arriving) = R-6d4.**
     **✅ R-6d3c LANDED (the NEVER-restart CORRECTNESS closure, per the vetted design of record; all 3 sub-slices in ONE
     Tier-A pass). R-6d3c-1 (vd-wire): `InterShardFlow::TransientDiscard(TransientHandoff)` APPENDED (postcard
     discriminants preserved) + `TRANSIENT_DISCARD_STEP=17`, through all 3 closed-set gates (effect_class
     SideEffecting/TransferStep folded into the transient group; durability_class `ReDriven` folded into the big group —
     the golden `producer_less.len()==2` pin STAYS 2, proving the arm does NOT grow the outbox set); `arm_tripwire` +
     `every_arm` + the in-module g_sealed loop + the step-disjointness pin (now the FULL 7–17 space) + the "16-arm"
     header. R-6d3c-2 (vd-sim): `on_transient_discard` REMOVES `Arriving{batch==transfer}` + POISONS `(transfer,
     TRANSIENT_BATCH_STEP)` (a late replayed `adopt_transient_batch` is `AlreadyApplied` ⇒ never re-inserts an orphan —
     the ACTUAL dest cure, proven by `discard_before_adopt_poisons_so_a_late_replay_never_orphans`) + buckets the loss by
     kind into `transients_lost_in_handover` + the discard-specific `transients_discarded_source_crash` counter;
     idempotent by `TRANSIENT_DISCARD_STEP`; ack-FREE dispatch arm in `on_directory_reply` (mirrors `TransientAbandon`);
     the corrupt-tag `from_tag` Err arm covered. R-6d3c-3 (vd-sim FSM + vd-node producer): `SourceUnreachablePreAdopt` +
     `EmitTransientDiscard`/`CountBatchLostSourceCrash`; the `saga.rs` BatchHandoff arm SPLIT by phase (post-adopt →
     self-promote unchanged; AwaitAdopt → discard-to-dest + count + Tombstone, terminal-on-first-fire); the crossed
     events fall through the total no-op catch-all (2 FSM no-op tests). `rehome_event_for` BatchHandoff now
     phase-discriminates + BUDGET-gates the source-dead path (a restart-within-budget wins its outbox-replay race);
     `batch_lost_source_crash` counter + accessor; the M2 fold — the counter arm added to the `scan_deadlines`
     `_ => {}` wildcard (the one place a missing arm compiles green), covered by BOTH a scan_deadlines test AND the
     rehome_event_for producer test; NIT-A — only `scan_deadlines_resolves_a_batch_handoff_with_a_dead_participant`
     (AwaitPromote, source-dead now budget-gated) MODIFIED, `:3806` (Demoting) untouched; M1 — the late-BatchAdopted-at-a-
     tombstoned-saga no-op proven in the scan_deadlines test. NET: the dest can NEVER strand an `Arriving` orphan
     regardless of replay timing, and the saga resolves to a bounded accounted terminal (loud
     `batch_lost_source_crash`/`transients_discarded_source_crash`) instead of self-promoting an empty dest. Gate: vd-wire
     66 + vd-sim 200 + vd-node 84 all green; full workspace green. ⇒ D-6 #1 RESTART case 🟩 (R-6d3b-2b); NEVER-restart
     CORRECTNESS 🟩 closed ON DELIVERED-DISCARD by R-6d3c; NEVER-restart DETECTION-IN-PROD (the confirm-dead trigger +
     the AwaitAdopt re-solicit egress + the lost-discard reliability) 🟧 CA-1/L5-gated; SIGKILL-source-in-AwaitAdopt e2e
     + both-ends-restart proptest = R-6d4.**
     **✅ R-6d3c POST-IMPL REVIEW DONE (wf_f3bed49f, 3 opus lenses + synth, verdict FIX_BEFORE_COMMIT → all blockers
     FIXED). Headline claims CONFIRMED SOUND: dest no-loss + adopt-poison (both interleave orders orphan-free), the
     phase-split + crossed-event no-ops (no double-resolution), the closed-set landing (discriminant unshifted, verified
     by a reviewer's throwaway postcard-discriminant probe; ReDriven not producer-less; producer_less.len()==2 intact),
     the M2 scan_deadlines counter arm (exercised by a real scan test). BLOCKERS FIXED: (1) [MEDIUM, real regression I
     introduced] the shared `dead_observed_since` budget anchor conflated the source-dead and dest-dead causes — a
     cause-switch (dest confirmed dead → dest RECOVERS via record_ack → source confirmed dead) measured the source's
     restart-race budget from the DEST's stale first-dead tick, firing the destructive source resolution early
     (reachable-NOW in post-adopt phases, which emit orch→source egress). CURED: the anchor is now keyed by the dead
     NodeId (`Option<(NodeId, UniverseTick)>`) via a monomorphic `dead_budget_elapsed` helper that RE-ANCHORS on a
     participant switch + a `rehome_event_for_reanchors_the_budget_on_a_source_dest_cause_switch` guard test. (3) [LOW]
     the canonical arm enumeration in `wire/src/lib.rs` said "15 arms" (the copy `intershard_closed.rs` names as
     authoritative) — updated to 16 + `TransientDiscard`. DEFERRED (non-blocking): finding 2 [→ R-6d4] once the
     AwaitAdopt re-solicit egress lands, gate `abort_deadline_ticks > transport max-reliable-ack bound` (or check
     dest-adopted) so a live dest's in-flight `BatchAdopted` never gets over-discarded — ledgered above (~line 950);
     finding 4 [doc] the stale saga-phase-wildcard residual annotated SUPERSEDED. Scope split judged HONEST (no
     over-claim). Re-gate after fixes: full workspace + clippy -D + coverage-fast (Tier-A 100%) all green.**
     **✅ HOLISTIC /goal AUDIT DONE post-R-6d3c (wf_062637c0, HEAD ff6cbb7; 5 read-only Explore+opus lenses + synth):
     VERDICT DONE_NO_CRITICAL — architecture SOUND (no rewrite), R-6d3c a clean regression-free never-restart closure,
     zero CRITICAL, every dense-crowd/PvP wall honestly additive. FIX-NOW: none. FIX-BEFORE-P4: none. Three cheap
     LEDGER items owed (recorded so the "phase-not-done-until-entries-flip" gate catches them):
     • **F1 (poison durability, lands with the CA-1 confirm-dead trigger):** `on_transient_discard` poisons `(transfer,
       TRANSIENT_BATCH_STEP)` into the in-RAM `AppliedSteps` (stub.rs:615,2380); a DEST restart between the discard and a
       late adopt-replay would lose the poison and reopen the orphan. NOT reachable-now (the discard is only emitted once
       the CA-1/L5 confirm-dead-toward-source trigger fires, and the durable `applied_steps` table is D-22 — both arrive
       TOGETHER). When the trigger lands, the poison + the Arriving removal MUST write to the durable D-22 `applied_steps`
       table in the SAME barrier. No code change at this posture (not P4-blocking; the trigger is not P4).
     • **F2 (loss-budget oracle shape, FIX-BEFORE-P11):** `oracle.rs:440-455` judges the transient loss budget as a
       CUMULATIVE-ABSOLUTE count summed flat over all shards for the whole scenario, vs a per-entity `LossBudget(u16)`
       (entity_kind.rs). A long P11 dense-PvP firefight soak with the crash matrix injecting genuine kills will sum
       handover losses past the absolute bound at a perfectly healthy per-crash FRACTION → a false RED on a spec-correct
       system. TEST-SCAFFOLD gap ONLY (the runtime never sheds to stay in budget — the budget JUDGES, never CAUSES). Before
       the P11 firefight fixture relies on this gate, re-base `LossBudget` to a RATE/FRACTION (lost/attempted ppm) or a
       per-window / per-injected-crash bound. Runtime path unchanged; not P4-blocking (P4 is terrain, no combat volume).
     • **F3 (observability, M4/CA-1-era):** a QUARANTINED-retained outbox row (roster-gone / undecodable — outbox.rs:452-476,
       fail-SAFE: retained, never swept, never mis-decoded) is surfaced only in the boot log, not a live `MeshStats`/admin
       gauge — a roster-gone row re-quarantines every boot and can accumulate on the PV with no alertable cloud signal. Add
       an `outbox_retained_rows` gauge / admin report with the CA-1-era cloud-deploy hardening. Correctness-neutral.
     • F4 (cosmetic DRY, optional): `on_transient_abandon`/`on_transient_discard` share the FirstApply-drain + per-kind
       loss-bucket tail (the kind-attribution line appears 3×) — an optional `bucket_loss_by_kind` extraction, not
       load-bearing. NITs (no action): corrupt-tag counted-not-bucketed is the intentional HR2 no-decode-to-default stance;
       the discard `AlreadyApplied` reuses the generic `transient_release_noop`; `replay_outbox` `checked_add().expect()` is
       test/mock-only reachable. Direct /goal answers all GREEN: complementary, DRY, robust (fail-safe every arm), scalable
       (G-TIER one-row-per-batch holds; walls ledgered+additive), cloud-ready (modulo CA-1/L5/M3), elegant, not error-prone,
       AAA/PvP/hundreds-in-one-location/new-feature-ready on track (RESERVED wire arms + seams shaped additively).**
     **📐 R-6d4 DESIGN DONE (wf_ca465bbc, 1 designer + 3 adversarial opus lenses + adjudication; read-only Explore agents
     per feedback_review_workflows_readonly). Verdict REVISE → all 7 blockers folded → SOUND_TO_IMPLEMENT; full design of
     record in scripts/r6d4_vetted_design.md. The review earned its cost: it caught that B4 as first-designed was
     MUTATION-BLIND (killing notify_waiters would NOT turn it red — the exact wf_75a225d0 gap R-6d4 exists to close) ⇒
     needs R-6d4-M (a one-line prod death short-circuit); that F2's numeric abort_deadline>ack-bound guard is ILL-DEFINED
     now (a live-but-slow dest emits ZERO transport bounces — slow≠unreachable — so it measures the wrong clock) ⇒ DEFER
     to CA-1, land only an inert tripwire; that B4's "drop the paused store" self-hangs (writer join) ⇒ permanent-fsync-
     fault exit; and that D's pause-seed produces a LOST (not durable-unacked) row + a killed shard has no admin surface.
     Sub-slices: M (prod death short-circuit) → C1/Cseam/C2 (replay error arms) → A (both-ends-restart proptest) →
     B1/B2/B3/B4 (store-test-hooks deterministic+mutation-sensitive pins) → D (process-tier SIGKILL-restart +
     boot-counter-by-1) → F2 (inert CA-1 tripwire). CA-1 boundary OUT (the SIGKILL-source-in-AwaitAdopt real-QUIC e2e).**
     **✅ R-6d4-M + R-6d4-C LANDED (io-prod hardening batch 1). M (prod, store.rs): the durable-writer death now
     short-circuits BOTH durability waits — `wait_durable_through_async` collapses its match into ONE shared post-await
     death-check (a death `notify_waiters` fails loud IMMEDIATELY instead of re-enrolling a fresh `notified` the fired
     notify would never wake — the old code cost a full WRITER_WAIT_POLL); `park_until_durable` checks death on EVERY
     wake (not only `res.timed_out()`); both call the new `#[cold] writer_died_panic(seq) -> !` (one fail-loud site).
     A strict correctness sharpening (prompt fail-loud; happy path unchanged) + the enabler for B4's mutation-sensitivity
     (its full mutation-proof lands with B4). C (outbox.rs): the replay error arms the R-6d3b-2b review left uncovered
     (F-C) — a `ReplayLimits` test-speed seam (`replay_outbox_with_limits`; prod delegates with `::release()`, byte-
     identical) drives `LaneStuck` (live lane full past a tiny retry cap ⇒ refuse-boot, row survives) + `FenceTimeout`
     (a re-driven row that never re-submits ⇒ fence times out at base+1, row survives) in ms; a `MockOutboxSink`
     (already_durable ⇒ base=u64::MAX) drives the count-fence `checked_add` overflow `#[should_panic]`; + the
     `Display`/`std::error::Error` impls for all 3 variants. Gate: vd-io-prod 116 lib green, clippy -D clean,
     coverage-io-prod Tier-B floor PASS (region 94.41% ≥ 90). NEXT: R-6d4-A (proptest) → B1-B4 (pins, need the
     OutboxKey-prefix + WRITER_WAIT_POLL-override + fsync-fault seams) → D (process-tier) → F2 (tripwire).**
     **✅ HOLISTIC /goal AUDIT DONE post-R-6d4-M+C (wf_ddc01e94, HEAD 336c914; 3 read-only Explore+opus lenses + synth):
     VERDICT DONE_NO_CRITICAL — zero criticals. M is PRODUCTION-CORRECT (verified end-to-end incl. the pinned tokio
     `Notify` source: `notified()` captures the counter at creation so a bump racing enroll→await is never lost; the
     death panic requires BOTH `last_durable < seq` AND `!writer_alive` so a live-slow writer never false-aborts; the
     writer stores `writer_alive=false` Release THEN `notify_waiters()` ⇒ prompt real death). R-6d investment PROPORTIONATE
     (protects EXACTLY the 2 producer-less arms {Ghost::Despawn, TransientBatch}; Delta=Unreliable, all saga/dir=ReDriven
     are OFF the durable path; ~1 fsync/batch — critical-path traffic, not gold-plating). No drift, no scale-regression,
     HR1/HR2/HR3 hold (shard-kind grep empty), ReplayLimits REFUTED as premature-abstraction (release()==the consts,
     prod byte-identical). Findings (all LOW/NIT): (1) **BINDING GATE** — M's death branches (store.rs writer_died_panic
     + the two death-checks) ride the region-only Tier-B floor (`coverage-io-prod` gates `--fail-under-regions 90` with
     NO `--branch`), NOT a real fail-loud test; **R-6d4-B4 MUST land a REAL covered `#[should_panic]` writer-kill test
     flipping them from floor-tolerated to PROVEN before R-6d4 is called done** (safe to ship M ahead of B4: strict fail-
     loud superset + timeout backstop retained). (2) H2 RecvLedger inner-lock across push_inbox → LEDGER-ONLY, measure at
     P3/P6 D-9 dense-crowd fixture. (3) D-9 whole-realm snapshot broadcast → LEDGER-ONLY, verified-additive, the
     mandatory N-in-one-realm per-client-bytes floor stays a hard gate before any dense-PvP soak. (4) NIT F3 outbox
     retained-row gauge → M4-era. (5) NIT the outbox.rs replay-seed `MockOutboxSink` name-collides with the mesh.rs
     recording mock → optional cosmetic rename to `OverflowSeedSink` when R-6d4-B next touches outbox.rs (no action req).**
     **✅ R-6d4-B4 LANDED — the /goal-audit BINDING GATE on M is SATISFIED (M's death branches are now PROVEN, not
     region-floor-tolerated). `store.rs` gains TWO store-test-hooks seams: `fail_fsync_on_key_prefix` (a content-keyed
     PERMANENT fsync fault — the writer force-fails the marked batch's fsync, exhausts `WRITER_FSYNC_MAX_RETRIES`, then
     `break 'drain`s + DIES deterministically, vs `pause_on_key_prefix` which parks forever + hangs Drop's join) +
     `wait_poll_override` (overrides `WRITER_WAIT_POLL` for the async wait; `DurabilityHandle` gains an always-present
     `wait_poll` field, release=const so prod is byte-identical). Test `async_wait_wakes_and_fails_loud_promptly_on_
     writer_death`: enrolls an async `wait_durable_through_async(seq)`, kills the writer via the fault, asserts the task
     PANICS via the R-6d4-M death short-circuit with the "died before seq" message + PROMPTLY (elapsed < 2s) while
     `wait_poll_override` is 10s — so a neutered death `notify_waiters()` / a lost R-6d4-M short-circuit would hang to
     ~10s ⇒ RED (the exact wf_75a225d0 mutation the old 100ms-poll tests could not catch). `run_writer` gained an
     `#[allow(clippy::too_many_arguments)]` (9 args under the feature); the orchestrator bin's `StoreTuning` literal
     lists the 2 new fields (None). Gate: vd-io-prod 116 lib (no feature) + 118 (store-test-hooks incl. B4) green,
     clippy -D clean BOTH configs, coverage-io-prod Tier-B floor PASS (region 94.43% ≥ 90), workspace + vd-bins(feature)
     build 0. FORMAL coverage-INSTRUMENTATION of the death branches (a coverage pass WITH store-test-hooks) rides with
     R-6d4-B1/B2 (which also need the feature-coverage). NEXT R-6d4: B3 (two-peers-no-starve) → B1/B2 (ordering +
     no-premature-gc, pause-hook pins) → A (both-ends-restart proptest — needs a controllable-DurabilityHandle fence
     seam, its own focused model design) → D (process-tier) → F2 (tripwire).**
     **✅ R-6d4-B1 + B2 + B3 LANDED (the deterministic + mutation-sensitive store-test-hooks pins) + the feature-coverage
     GATE (closes the audit "REAL not theater" note — B4/B1/B3's hook branches are now INSTRUMENTED, not merely
     region-tolerated). B1 `a_row_is_not_visible_to_scan_all_until_block_b_waits` (outbox.rs, store-test-hooks): opens a
     NodeOutbox whose writer PARKS pre-fsync on the retained row's exact key (`pause_on_key_prefix = key.to_bytes()`);
     while the marker proves the writer parked, `scan_all` is EMPTY + the row is not-durable (durable-before-send
     ordering; a synchronous-fsync submit ⇒ RED). B2 `gc_never_runs_before_the_replayed_rows_are_durable_mutation_proof`
     (outbox.rs, plain #[test] ⇒ runs in the default gate): a NEW `DurabilityHandle::controllable()` seam (test-driven
     submitted/durable atomics) + a `RemirrorTransport` (bumps `submitted` per send = block A, never `durable`) + a
     background `replay_outbox` that PARKS at `wait_durable_through` — while parked, the 2 recoverable prior rows MUST
     still be present (gc is fenced behind DURABILITY not submission; moving gc before the wait ⇒ sweeps them un-durable
     = the D-6 #1 loss ⇒ RED), with the anti-vacuity guard (submitted≥target && durable<target); after bumping durable,
     gc sweeps them. `MockOutboxSink` refactored (a `durability` field + a REAL `gc_replayed`). B3
     `two_parked_durability_waits_do_not_starve_a_third_worker_task` (store.rs, store-test-hooks): worker_threads(2), TWO
     `wait_durable_through_async` waits parked on paused writers + a THIRD mesh-I/O-proxy task that STILL completes
     (async YIELDS the worker; the sync park would occupy both ⇒ the 3rd starves ⇒ RED). Justfile: NEW
     `coverage-io-prod-hooks` (io-prod cov WITH store-test-hooks, floored) wired into `coverage` — so the crash-proof
     hook surface is gated alongside the release surface. Gate: vd-io-prod 117 (no-feature, incl. B2) + 121 (feature,
     incl. B1/B3) green, clippy -D clean both configs, coverage-io-prod 94.45% + coverage-io-prod-hooks 94.74% (both ≥
     90). REMAINING R-6d4: A (both-ends-restart proptest — reuses `controllable()` for its fence seam, own focused model
     design) → D (process-tier SIGKILL-restart + boot-counter-by-1) → F2 (inert CA-1 tripwire); the mesh-block-B-uses-
     async coupling guard (B3's "should") is minor-OWED (the `await_holding_lock` lint + T-DBS-1 already cover it).**
     **📐 R-6d4-A DESIGN DONE (wf_ec22b736, 1 designer + 3 adversarial opus lenses + adjudication, read-only Explore;
     3× REVISE → 12 folded must-fixes → SOUND_TO_IMPLEMENT; full DOR scripts/r6d4a_vetted_design.md). The both-ends-
     restart replay PROPTEST — drives the REAL `replay_outbox` through arbitrary crash/restart/redeliver interleavings
     and cross-checks disk state against a HAND-MAINTAINED `Ref` computed from an INDEPENDENT source (the anti-vacuity
     core — an agreement assert is load-bearing). Pieces: a `ModelOutboxSink` (committed/staged two-set = the RAM/disk
     crash boundary, mirrors NodeOutbox+RedbStore), a `DedupLedger` backed by the REAL `classify_reliable` receiver
     ladder (exposed `#[cfg(test)] pub(crate)` in mesh.rs — so an incarnation-reset mutation turns it RED, not a
     hand-rolled dedup), a `CaptureTransport` (bumps the `controllable()` submitted+durable atomics per send = block A+B
     synchronously; `fail_after`/`lane_alive` drive the LaneStuck/LaneDead arms via `replay_outbox_with_limits(fast)`).
     4 invariants each argued NON-vacuous + NON-falsely-red: INV-1 no-loss (`sink.committed==Ref.committed` + independent
     `scan_all` decode; staged-lost-to-crash is CORRECT loss), INV-2 no-orphan (`gc_swept ⊆ expected_deliveries`, ∩
     committed = ∅), INV-3 exactly-once (SPLIT: source-idempotence [2nd BootReplay ⇒ replayed==0] + receiver-exactly-once
     [real classify_reliable]), INV-4 accounting (`replayed+quarantined==scanned`). Anti-vacuity: a 4-flag in-strategy
     coverage floor (crash-with-staged / redrive-after-crash / redeliver-of-delivered / quarantine) asserted after the
     run + 3 hand-written witnesses + a crash-mid-replay regression (fail_after ⇒ LaneStuck ⇒ ?-bail ⇒ no partial sweep,
     then fresh replay ⇒ dedup to one effect) + a differential-vs-real-redb witness. ChaCha-seeded `TestRunner`
     ([0x6d;32], 1024 cases, 0..=48 ops) ⇒ byte-reproducible. PREREQ edits: `OutboxKey` derive +PartialOrd/Ord/Hash;
     mesh receiver ladder `#[cfg(test)] pub(crate)`; `proptest` io-prod dev-dep. Plain `#[test]` (rides `controllable()`)
     ⇒ runs in the default coverage-io-prod gate. SCOPE: A proves cross-crash exactly-once + no-loss/no-orphan
     accounting; B2 still owns the block-B no-premature-gc PARK (both survive, non-overlapping). IMPLEMENT NEXT
     (checkpointed after the vetted design — a large fresh unit, not rushed at the tail of a marathon session).**
     **✅ R-6d4-A LANDED (the both-ends-restart replay proptest — the headline robustness proof). Per the DOR:
     `ModelOutboxSink` (committed/staged two-set RAM/disk crash boundary, validated by the differential-vs-real-redb
     witness) + a `DedupLedger` driving the REAL `classify_reliable` via an opaque `RecvCell` (mesh.rs `#[cfg(test)]
     pub(crate) recv_test_hooks` — no visibility widening) + a `CaptureTransport` (bumps the `controllable()`
     submitted+durable atomics per send = block A+B synchronously; `fail_after`/`lane_alive` drive LaneStuck via
     `replay_outbox_with_limits`). A HAND-MAINTAINED `Ref` cross-checks the sink after every op (INV-1 no-loss +
     independent scan_all; the accounting `replayed+quarantined==scanned`; source-idempotence; receiver-exactly-once).
     Anti-vacuity: a 4-flag in-strategy coverage floor (crash-with-staged / redrive-after-crash / redeliver-deduped /
     quarantine) ASSERTED after the 1024-case ChaCha-seeded run + 5 hand-written witnesses (incl. crash-mid-replay:
     LaneStuck ⇒ ?-bail ⇒ no partial sweep, then fresh replay ⇒ dedup to one effect). THE FUZZER EARNED ITS KEEP
     IMMEDIATELY — it caught TWO real model-fidelity bugs during bring-up: (1) a boot cloned the sink WITH its
     uncommitted staged retain, and replay's final `commit()` drained it back in ⇒ FIX: a boot is a (re)start, staged
     (RAM) is empty — clear it first; (2) a falsely-red cross-time `gc_swept ∩ committed` invariant (a key legitimately
     re-committed after a boot gc'd it) ⇒ FIX: dropped it, the load-bearing no-orphan is the INV-1 equality (Ref keeps
     quarantined rows; a gc-of-a-quarantined-row mutant ⇒ RED). PREREQ prod edits: `OutboxKey` +PartialOrd/Ord/Hash;
     mesh `recv_test_hooks` (test-only); `proptest` io-prod dev-dep. Plain `#[test]` (rides `controllable()`) ⇒ runs in
     the default coverage gate. Gate: vd-io-prod 124 (no-feature, incl. 7 proptest) + 128 (feature) green, clippy -D
     clean both, coverage-io-prod 94.61% + coverage-io-prod-hooks 94.90% (both ≥ 90). REMAINING R-6d4: D (process-tier
     SIGKILL-restart + boot-counter-by-1 — needs a NodeOutbox::seed_reliable_row seam + an in-process receiver) → F2
     (inert CA-1 tripwire). Then CA-1 → P4 voxels.**
     **✅ HOLISTIC /goal AUDIT DONE post-R-6d4-A (wf_0b8c712a, HEAD 39f4c59; 3 read-only Explore+opus lenses + synth):
     VERDICT DONE_NO_CRITICAL — architecture SOUND; the proptest is GENUINELY load-bearing (not theater), the 3 prereq
     prod edits are SAFE, the R-6d4 investment is PROPORTIONATE (no drift). The auditor ran a LIVE MUTATION (mutated
     classify_reliable Reset→Dedup) + verified `cargo build --release` (no test-item leak), the `#[cfg(test)]` gating of
     recv_test_hooks/controllable/fail_fsync/wait_poll_override, and that `OutboxKey`'s derived Ord == the big-endian
     to_bytes order (MsgClass decl order == class_to_byte, doubly golden-pinned). Findings (all LOW/NIT): (1) **LOW,
     FIXED IN PROSE now** — the proptest's dedup-oracle over-claimed "incarnation-reset mutation ⇒ RED": the mutation
     experiment proved the A1 incarnation-reset arm is NOT exercised (the oracle feeds each row's STORED incarnation),
     only the SEQ-dedup arm bites (via DestRedeliver); the A1/epoch/Gap arms are covered by mesh.rs's classify_reliable
     unit tests. Tightened the module + DedupLedger doc-comments to state this scope honestly (no code/coverage change —
     the proptest's OUTBOX/REPLAY proof [no-loss/no-orphan/accounting/source-idempotence] + seq-dedup integration is
     unaffected + real). (2) NIT — class_byte round-trip test pins uniqueness not `class_to_byte(ALL_CLASSES[i])==i`
     (near-zero risk, both orders from one decl); (3) NIT — differential-vs-redb witness uses only MsgClass::Saga (the
     2 class golden pins cover the class axis); (4) NIT — f[1] post-crash-redrive flag doesn't certify A1 fired (same
     root as #1); (5) NIT — differential witness pins the commit surface not the staged crash boundary (proven on
     RedbStore). All LEDGER-ONLY except #1 (prose, done). No CRITICAL/HIGH, no vacuous test, no prod-surface leak, no
     silent-loss, no HR violation, no drift.**
     **📐 R-6d4-D DESIGN DONE (wf_bec6d1a6, 1 designer + 3 adversarial opus lenses + adjudication, read-only Explore; 3×
     REVISE → 8 folded must-fixes → SOUND_TO_IMPLEMENT; full DOR scripts/r6d4d_vetted_design.md). The process-tier
     SIGKILL-restart + boot-counter-by-EXACTLY-1 proof (the RESTART case at the real-process tier; the in-process proof
     is R-6d4-A). Plan: (seed) a store-test-hooks `NodeOutbox::seed_reliable_row(from, peer, class, incarnation, seq,
     payload)` — builds+frames a ReliableFrame via a new `frame_reliable` private helper (framed_row delegates, DRY;
     ReliableFrame stays pub(crate)), retain()+commit() (DURABLE-on-return via commit's wait_durable_through — NO pause
     hook), returns the durable batch seq. (reader) a store-test-hooks `BootCounter::current(path) -> Result<Option<u64>,
     _>` thin wrapper over the private read (keeps MAGIC+checksum in one place). (bin) a NEW purpose-built
     `vd-outbox-testnode` — a plain `[[bin]]` with a `#[cfg(feature=store-test-hooks)]` main() body (NOT required-features
     ⇒ CARGO_BIN_EXE_* always defined for the test): boot-1 (VD_OUTBOX_TEST_SEED SET) `resolve_process_incarnation` FIRST
     [mints v1 — the F-CRITICAL: else v2==v1+1 is unprovable] → open_node_outbox → seed_reliable_row → assert durable →
     write .seeded marker → idle (park, keep outbox open, NO spawn_mesh/replay — else replay would gc the seed); boot-2
     (seed UNSET) boot_mesh_and_replay + tick. The env is built BY HAND without VD_PROCESS_INCARNATION (so the durable
     counter is used). (test) crates/bins/tests/outbox_sigkill_restart.rs: an IN-PROCESS receiver B (real MeshTransport,
     stays up), reap-then-rebind-SAME-addr (Cluster::kill_and_reap, boot_counter_crashloop pattern — NOT
     orchestrator_crash's fresh-addr), poll .seeded → read v1 → SIGKILL+reap → boot-2 → 30s deadline-drain B's inbound
     for the re-drive (first-contact Accept) → assert v2==v1+1 (an INDEPENDENT exactly-once-incarnation check, not a
     delivery precondition) + a fresh-store 0-re-drive anti-theater twin. (gate) add to orch-crash + orch-crash-cov
     (%p-%m%c continuous-mode merge, --test-threads=1); Tier-B honesty (a SIGKILL loses the final counter flush).
     CA-1 boundary honest: the SIGKILL-source-in-AwaitAdopt real-QUIC e2e stays OUT (AwaitAdopt egress empty; needs
     CA-1+M3). IMPLEMENT NEXT (checkpointed after the vetted design — a large fresh process-tier unit: a new bin + test +
     2 io-prod seams + justfile, not rushed at a marathon tail). After D: F2 (inert CA-1 tripwire) closes R-6d4.**
     **✅ R-6d4-D LANDED (the process-tier SIGKILL-restart proof — the RESTART case at the REAL-process tier, closing
     the loop the in-process R-6d4-A proptest opened). Implemented exactly to the vetted design: (seam-1, outbox.rs)
     `#[cfg(store-test-hooks)] NodeOutbox::seed_reliable_row(from, peer, class, incarnation, seq, payload) -> u64`
     frames a `ReliableFrame` via a new private `frame_reliable` (the test `framed_row` delegates — DRY; ReliableFrame
     stays pub(crate)), `retain()`+`commit()` (DURABLE-on-return via commit's `wait_durable_through`, NO pause hook),
     returns the durable batch seq. (seam-2, boot.rs) `#[cfg(any(test, store-test-hooks))] BootCounter::current(path)
     -> Result<Option<u64>, _>` — a read-only peek over the private `read` (v1 before the kill / v2 after; MAGIC+
     checksum stay in one place). (bin) NEW `vd-outbox-testnode` — a PLAIN `[[bin]]` (NOT required-features ⇒
     CARGO_BIN_EXE_* always defined) with a `#[cfg(store-test-hooks)]` main body (release compiles only an exit-2
     stub): boot-1 (VD_OUTBOX_TEST_SEED SET) `resolve_process_incarnation` FIRST [mints v1 — else v2==v1+1 is
     unprovable] → `open_node_outbox` → `seed_reliable_row` → assert `is_durable_through` → write `.seeded` → park
     (NO mesh/replay, else replay gc's the seed); boot-2 (seed UNSET) `boot_mesh_and_replay` → `.ready` → park. Env
     built by hand WITHOUT VD_PROCESS_INCARNATION (so the durable counter is used). (test) crates/bins/tests/
     outbox_sigkill_restart.rs: in-process receiver B (real MeshTransport, stays up — a killed shard has no admin
     surface), `Cluster::kill_and_reap` + rebind-SAME-addr (boot_counter_crashloop pattern), poll `.seeded` → read v1
     → SIGKILL+reap → boot-2 → 30s deadline-drain B for the re-drive (first-contact Accept) → assert v2==v1+1 (an
     INDEPENDENT exactly-once-incarnation check, NOT a delivery precondition) + a fresh-store 0-re-drive anti-theater
     twin (proves the positive isn't a false-positive from an unrelated delivery). (coverage) the two io-prod seams
     get their OWN in-crate unit tests (`current_reads_the_counter_without_incrementing` [boot.rs, plain], and
     `seed_reliable_row_writes_one_durable_scannable_row` [outbox.rs, #[cfg(store-test-hooks)]]) so the seams are
     honestly instrumented by coverage-io-prod(-hooks), NOT merely by the uninstrumented vd-bins process test — REAL
     not theater. (gate) wired into `orch-crash` + `orch-crash-cov` (%p-%m%c continuous-mode, --test-threads=1).
     GATE GREEN: outbox_sigkill_restart 2/2 (both the redrive + the anti-theater twin); io-prod 130 lib + all
     integration green both configs (one flaky quinn-loopback timing test re-confirmed clean 125/125 — test-only
     change cannot touch a default-config prod path); clippy -D clean (io-prod ×2 + vd-bins store-test-hooks);
     coverage-io-prod 94.81% + coverage-io-prod-hooks 95.11% (both ≥ 90 floor; outbox.rs 98.16% under hooks — the
     seed seam now instrumented); vd-bins default (exit-2 stub) + workspace build clean. Tier-B honesty preserved: a
     SIGKILL loses the final counter flush (why io-prod is a ratcheted floor, never 100%). CA-1 boundary honest: the
     SIGKILL-source-in-AwaitAdopt real-QUIC e2e stays OUT (AwaitAdopt egress empty ⇒ needs CA-1+M3, ledgered).
     REMAINING R-6d4: F2 (the inert CA-1 tripwire at the saga AwaitAdopt discard site) closes the reachable-now arc.**
     **✅ HOLISTIC /goal AUDIT DONE post-R-6d4-D (wf_607da1cf, HEAD 2ec392e; 5 read-only Explore+opus survey lenses →
     adversarial refute-each-finding → opus adjudication): VERDICT DONE_NO_CRITICAL — 0 critical, 0 high. drift_check =
     PROPORTIONATE, NO DRIFT (R-6d4-D closes the RESTART half of D-6 #1 — a lost reliable one-shot is the mechanism
     behind a phantom cross-boundary collider from a lost GhostFlow::Despawn, so it is core transport-robustness, not
     peripheral; correctly bounded both ends — does not overreach past the CA-1+M3 line, does not under-prove). 2 of 11
     survived findings were DISMISSED on verification (the seed epoch-divergence "bug" is a FALSE POSITIVE — mesh.rs
     retains the epoch=u32::MAX bytes byte-identical to frame_reliable; the exactly-once happens-before concern is
     structural since incarnation is a required data-dependency of the send path). The other 9 are all LOW/NIT. FOLDED
     this pass (doc/comment/test-only, no prod-logic change): [F6] kill_and_reap now documents the load-bearing wait()
     reap (same-addr restart determinism); [F9] vd-outbox-testnode hoists the tokio runtime + trust into the boot-2 arm
     so "boot-1 builds no mesh" is STRUCTURALLY true; [F5] the orch-crash-cov comment now covers ALL SIGKILL-only
     children (incl. the two outbox_sigkill_restart boots), not just the orchestrator. LEDGERED-FORWARD (bigger, honest
     deferrals): [F3, the audit's TOP follow-up] D-40 — fold the process-tier %c merge into an ENFORCED
     `cargo llvm-cov report --fail-under-regions <floor>` over the merged %p-%m%c profraws (scoped to the spawned bin
     paths incl. vd-outbox-testnode) and wire into `coverage`/`gate`; until then the mirrored io-prod unit tests are the
     enforced floor (the SIGKILL-restart coverage rests on the ledger + those mirrors, not yet a gate). [F1] the
     per-node-PVC-vs-orchestrator-issued incarnation choice stays a JOINT-INVESTIGATION user-decision (VD_PROCESS_INCARNATION
     precedence makes it a zero-rework supersede). [F7] the anti-theater twin's 5s empty-B settle could become a
     deterministic replayed==0 marker (surface ReplayCounts from boot_mesh_and_replay) — optional hardening, non-vacuous
     as-is (the counter-by-1 assert blocks a fully vacuous green; the 5s clock starts only AFTER replay fast-returns on
     the empty outbox). CA-1 SCOPING NOTE for when it lands [F2+F4]: the CA-1 slice must cover BOTH halves — (i) inbound
     reply-on-connection AND (ii) the outbound addr re-plumb (L5: peer_writer must re-read the peer addr from an
     updatable topology source on each redial, not a spawn-time cfg.peers copy), with an acceptance criterion that a peer
     rescheduled to a new IP is redelivered-to, not confirm-dead-bounced; and cross-link the OUTBOX_WRITER_CHANNEL_DEPTH=256
     fail-loud fan-out ceiling (F-F) to the hundreds-of-shards sizing note (additive escape hatch: raise the const /
     per-peer outbox stores — do NOT raise speculatively; 256 is generously above the static-roster tier). WON'T-FIX
     [F8]: the two same-named MockOutboxSink test mocks (outbox.rs vs mesh.rs) are distinct private test modules with zero
     compile ambiguity — closed as harmless, no rename churn (retires the perpetual "opportunistic rename" ledger nit).**
     **✅ R-6d4-F2 LANDED (the INERT CA-1 tripwire — closes the R-6d4 reachable-now arc). Design wf_fc8c02b4
     (1 designer + 3 adversarial opus lenses + adjudicator, read-only Explore): the designer's proposed numeric
     `SagaTuning::validate_against_transport` boot-time cross-config validator was REJECTED by all 3 lenses + the
     adjudicator on TWO critical grounds — (i) WRONG CLOCK / LANE CONFLATION: the over-discard risk is a live dest's
     `BatchAdopted` ack lost on the DEST→ORCHESTRATOR TransferAck lane (redelivered on THAT lane's own backoff clock),
     but the designer's bound was derived from the dead-SOURCE confirm-redial series — the wrong clock (cf. the R-4c
     author's note ~saga.rs:289 declining a similar cross-check because a live-slow peer emits ZERO transport bounces);
     the dest-lane bound is genuinely ILL-DEFINED until the CA-1 re-solicit cadence exists; and (ii) the numeric guard
     would FAIL at the prod default (abort=24 < the corrected bound) — a real inertness violation — and contradicts the
     prior adjudicated disposition (r6d4_vetted_design.md item 4) that deferred it. FOLDED to the correct shape: an
     INERT DOCUMENTED TRIPWIRE — a doc marker at BOTH discard sites (the FSM arm `(BatchHandoff{AwaitAdopt},
     SourceUnreachablePreAdopt)` in saga.rs, and its producer `rehome_event_for`'s `AwaitAdopt→SourceUnreachablePreAdopt`
     mapping in saga_runtime.rs — the exact two places the CA-1 author must edit to wire the re-solicit egress, so the
     marker is unmissable in that diff) carrying the CORRECTLY-stated invariant: once CA-1/L5 makes
     `is_confirmed_dead(source)` reachable, this discard does NOT check dest-adopted, so a live dest whose ack was lost
     would be over-discarded ⇒ BEFORE wiring the egress you MUST either (a) check dest-adopted, OR (b) gate
     `abort_deadline_ticks >= the DEST-LANE max reliable-ack redelivery bound` (guard at orchestrator boot beside
     `LivenessTuning::validate_against`). Coverage: doc-comments ONLY ⇒ ZERO new Tier-A regions/branches ⇒ HR5 100% by
     construction, no new tests, no debug_assert panic-arm trap (the review's key HR5 point: a bare `debug_assert` on a
     runtime value is an uncoverable branch; a doc marker + the const `_: () = assert!` idiom are the coverage-clean
     tools — the const marker was skipped as a near-tautology, DEFAULT_ABORT>=DEFAULT_REDRIVE already enforced by
     `SagaTuning::validate`). Inert-today: both FSM/producer bodies are byte-identical; no runtime path changes. THE
     NUMERIC GUARD (dest-adopted check OR the dest-lane `abort_deadline >= ack-redelivery-bound` boot invariant) STAYS
     OWED/CA-1-gated — it lands WITH the CA-1 re-solicit egress, against the CORRECT dest-lane clock, with its own tests.
     This CLOSES the reachable-now R-6d4 arc (M+C+B+A+D+F2). NEXT: CA-1 (no-DNS peer addressing) — the last cloud
     precondition, which will also carry the L5 outbound addr re-plumb + this F2 numeric guard.**
     **📐 CA-1 DESIGN ANALYSIS (4 workflows: wf_cbc28339 decomposition + trust/security, wf_f1bdd063 S1 mechanism,
     wf_8b9e6320 Slice-1a unified-dispatcher; each 1 designer + adversarial opus lenses + adjudicator, read-only
     Explore). NOT YET IMPLEMENTED — the design did NOT converge to SOUND_TO_IMPLEMENT; CA-1 is a DEEPER transport
     re-architecture than "reply on a connection" sounded, and the analysis below is the running start. USER DECISIONS
     already made (locked): M6 = cap the learned table + a not-churny-deployable gate (NOT full RecvLedger eviction);
     trust = sender-asserted NodeId at the single-PSK tier + authority-split (a learned entry never shadows a booked
     one) + a release-present SPIFFE cross-check marker; the S3 re-solicit egress = a NEW appended InterShardFlow
     variant (never a repurposed no-op); NO new external dep (reuse ArcSwap for S2; io-prod learned table is
     Arc<Mutex<BTreeMap<NodeId,quinn::Connection>>>); the real-cloud k3d proof is a SEPARATE deploy-readiness-gated
     slice (loopback proofs first). Decomposition: S1 inbound reply-on-connection → S2 outbound L5 ArcSwap re-plumb +
     MeshControl::update_peer_addr → S3 AwaitAdopt re-solicit egress (new InterShardFlow arm) → S4 the F2 dest-lane
     numeric guard (only if a well-defined prod-default-passing bound derives from S3's cadence; else F2 stays inert).
     THE HARD BLOCKER surfaced (VERIFIED line-by-line in mesh.rs): the mesh pins application stream ROLES to QUIC
     ACCEPTOR-vs-DIALER, not connection-level bidirectionality — serve_connection (the DATA reader + ack_egress +
     datagram reader) runs ONLY on ACCEPTED conns; a DIALER's only reader is ack_reader_task, which drops every
     non-ACK stream. So (a) "reply on a held accepted connection" is WIRE-BROKEN both ways (reply DATA lands in the
     dialer's ACK-only reader and is dropped; the acceptor's learned-lane ack-reader waits forever for an ACK stream
     the dialer never opens ⇒ window never retires ⇒ false NodeUnreachable at a LIVE peer); (b) adding a second
     accept_uni loop on one connection RACES (quinn hands each incoming uni stream to exactly one waiter
     non-deterministically ⇒ DATA lands in the ACK loop and vice-versa). The FIX shape that IS sound: ONE unified
     per-connection accept_uni dispatcher that reads the 1-byte STREAM_KIND tag and routes DATA→classify_and_deliver
     (node-wide RecvLedger, StaleEpoch cross-stream cure intact) / ACK→the owning peer_writer, running on EVERY
     connection. THE UNRESOLVED CRUX (why it did not converge): AckFrame/AckEntry carry NO NodeId (lib.rs), so an ACK
     on an ACCEPTED connection cannot be routed to the right peer_writer by frame content — it needs a
     learned-identity registry keyed by the first authenticated frame.from (NodeId→ack-watch), which is exactly 1b's
     learned-peer machinery. Net: a "1a pure-refactor" (merge the two accept loops) has NO behavioral effect and NO
     honest RED→GREEN gate in the both-booked topology (both directions already ride each side's OWN dialed conn,
     served correctly today); the actual reply-over-accepted-connection behavior + the ACK-routing registry are
     irreducibly 1b. So CA-1's real first unit is "unified dispatcher + learned-identity ACK registry + learned lane"
     TOGETHER — a single larger deliberate transport slice, not a quick refactor prequel. This is a load-bearing
     change to the R-3'→R-6 redelivery core (the most safety-critical component); it warrants a dedicated,
     deliberate effort, not slice-by-slice quick wins. HELD for a user decision on investment level (defer to
     deploy-readiness + do P4 now, vs commit to the CA-1 transport redesign now). The two RED guards
     (ca1_reply_on_connection, l5_a_rescheduled_peer, both #[ignore]d) remain the flip-targets when it lands.
     ►► USER CHOSE: commit to the CA-1 redesign NOW (full design→implement→review→gate rigor).**
     **📐 CA-1-CORE DESIGN CONVERGED — SOUND-TO-IMPLEMENT blueprint (wf_bb68a021, the definitive consolidated pass;
     1 designer + 3 adversarial opus lenses [R-3'-redelivery / ack-routing-architecture / coverage-lifecycle-scope]
     + adjudicator). All lenses REVISE (not REJECT); architecture SETTLED; the adjudicator left 4 folded must-fixes
     each with a concrete resolution ⇒ the blueprint below IS the implementation plan (the NEEDS_ANOTHER_ROUND label
     is the conservative "confirm the fold" — the fold is fully specified here). ACK-ROUTING DECISION = Solution (C)
     per-connection ack sink resolved at spawn (NOT (A) a NodeId→ack-watch registry keyed by sender-asserted
     frame.from — that would route ACKs by spoofable identity + add a hot-path lock; NOT (B) a NodeId ACK-stream
     header — an unneeded wire change, since the directional single-peer-per-connection topology makes the routing
     key simply "which connection did this ACK arrive on"). BUNDLE (one landable io-prod slice, flips
     ca1_reply_on_connection with an honest DELIVERY+reliable_acked-advances gate):
     (1) UNIFIED DISPATCHER — add pure `enum StreamKind{Data,Ack}` + `fn stream_kind(u8)->Option<StreamKind>` (near
     STREAM_KIND consts); ONE per-conn `dispatch_streams` accept_uni loop reads the 1-byte tag and spawns
     `serve_data_stream_body` (DATA) or `drain_ack_stream` (ACK), unknown⇒drop. Replaces BOTH serve_connection's
     DATA-only accept loop AND ack_reader_task (the VERIFIED two-accept_uni-loops-per-conn race cure: quinn hands
     each uni to exactly one waiter). (2) MUST-FIX #1 SYMMETRIC ack_egress — factor `serve_any(conn,…,ack_out)`
     called by BOTH the accept path AND the dial path (ensure_connection Dial arm), so EVERY connection creates its
     own acked_keys+ack_due, spawns ack_egress, and runs the dispatcher. WITHOUT this the dialer never emits acks ⇒
     A's reply reaches B but reliable_acked never advances (Gate ii unsatisfiable). ack_egress is already symmetric-
     safe (keys (peer,class) from the node-wide ledger, per-entry incarnation, write outside select). Thread
     inbox/stats/ledger into ensure_connection + write_frame + replay_lanes. Dialer datagram RX stays dropped ⇒
     this slice is RELIABLE-uni reply-on-connection only (say "reliable-uni superset", not "behavioral superset").
     (3) LEARNED LANE — `type LearnedPeers=Arc<Mutex<BTreeMap<NodeId,LearnedConn>>>`; `struct LearnedConn{conn,
     ack_rx: watch::Receiver<Option<AckFrame>>}`. Record site in serve_data_stream_body on the FIRST frame whose
     from != local AND NOT in `booked` (AUTHORITY SPLIT — a booked NodeId is never learned, no shadow lane), under
     `learned_peers_max` (new MeshConfig field; loud `learned_peers_rejected` MeshStats counter + snapshot on cap);
     `learned.lock()` is a SEPARATE statement OUTSIDE the inner ledger Mutex (lock order outer-read→inner→inbox —
     the one hard code-review gate all lenses agreed on); REFRESH PREDICATE = insert iff !contains_key OR cached
     conn.close_reason().is_some() (dead-conn eviction only); SPIFFE MARKER = read conn.peer_identity() + warn-if-
     None (log only, NOT an authority check — the shared cluster cert can't bind NodeId; authority = booked-exclusion
     + the mTLS channel). `enum ConnSource{Dial(SocketAddr),Learned{table,dest}}` replaces PeerWriter.addr;
     ensure_connection Learned arm NEVER dials (adopts the freshest held conn, Err on absent/dead ⇒ Down⇒timer⇒
     re-read = stale-send-at-most-once). (4) send_durable LAZY-SPAWN + MUST-FIX #4 corpse-lane — treat a
     present-but-CLOSED lane as a MISS (get(&to).is_some_and(|l|!l.tx.is_closed())), remove the stale entry, consult
     LearnedPeers, lazily spawn a Learned peer_writer (+insert PeerLane) else QueueFull. MeshTransport gains the
     spawn bundle {handle, endpoint, learned, stats=Arc::clone of the SAME MeshControl Arc [else Gate ii false-
     negatives], booked, incarnation, reliability, outbox, connections, backoff_min/max, outbound_capacity}. (5)
     MUST-FIX #2 acc_ack_rx re-assignable — peer_writer gains `ack_rx_override: Option<Receiver>` (Some for a learned
     lane, from LearnedConn.ack_rx); the Learned adopt re-reads BOTH conn+ack_rx on refresh; the select's changed()
     future is re-created per iteration (no borrow held across an adopt). (6) DEAD-LEARNED-LANE single-bounce —
     same-pass `break` when dest is absent-from-LearnedPeers AND the confirm bounce just fired (confirm_and_maybe_
     bounce re-bounces every cycle, so termination must be same-pass); loop-exit closes the PeerLane ⇒ #4 re-consults.
     Booked lanes NEVER terminate. R-3'/R-4 RE-CHECK: ledger keys (from,class) connection-agnostic; StaleEpoch/
     contiguity per-(peer,class); ack_egress acks only its own acked_keys; on_ack incarnation/epoch guard — all
     bodies lifted VERBATIM, hold. TESTS (Tier-B ≥90 regions, measure both coverage-io-prod + -hooks): stream_kind
     3 arms (pure); runtime unknown-tag; ca1_reply_on_connection rewrite (NO book! macro — BTreeMap::from + the
     cluster() reserve-port dance; A book EMPTY, B books A; assert B receives 0xA AND a_ctl reliable_acked≥1 AND NO
     NodeUnreachable{to:b}); reply-on-RE-connection (B blip+re-dial+re-send ⇒ reliable_acked→2 on v2 — the acc_ack_rx
     refresh gate); corpse-lane re-spawn; dead-lane ≥1 AND ≤1 bounce then QueueFull; cap-full reject (explicit);
     authority-split (booked never learned). Existing mesh_redelivery/mesh_load/mesh_under_loss MUST stay green
     (learned path dormant on booked clusters; dialer now a reliable-uni superset). NO new dep; HR1/HR3 preserved
     (spawn_mesh sole MeshTransport ctor; frozen sim::io::Transport seam untouched). IMPLEMENTING IN TREE-GREEN
     STAGES: dispatcher+serve_any symmetric refactor (existing tests green) → learned lane + send_durable lazy-spawn
     + gate → RE-connection/corpse/dead-lane robustness. L5 outbound ArcSwap re-plumb (S2), the AwaitAdopt re-solicit
     egress (S3, new InterShardFlow arm), and the F2 dest-lane numeric guard (S4) are the follow-on slices.**
     **✅ CA-1 CORE (reply-on-connection) LANDED (Stages 1+2+3-HIGH) — the k3d no-DNS peer-addressing blocker is
     CLEARED for the mesh. Implemented per the blueprint: (S1) the unified per-connection accept_uni dispatcher
     (stream_kind DATA/ACK tag-routing) replacing serve_connection's DATA loop + the deleted ack_reader_task — the
     two-accept_uni-loops-per-connection RACE eliminated — + symmetric ack_egress on EVERY connection (dialer now
     emits acks too); (S2) LearnedPeers/LearnedConn + ConnSource{Dial,Learned}, the learn_dial_in_peer record site
     (authority split = booked/self never learned; learned_peers_max cap + loud learned_peers_rejected counter;
     SPIFFE marker; separate learned.lock() outside the ledger inner Mutex), ensure_connection Learned arm (adopt
     held conn, never dial, Err on absent/dead), send_durable lazy-spawn with corpse-lane-as-miss + a dead-conn
     pre-check + ack_rx_override; MeshTransport spawn bundle (stats the SAME Arc as MeshControl). ADVERSARIAL
     POST-IMPL REVIEW (wf_5310c8fb, 3 read-only opus lenses) VERIFIED every R-3'/R-4/R-6 invariant PRESERVED (one
     ledger Arc both directions — StaleEpoch cross-stream cure STRENGTHENED; contiguity, torn-ack cancel-safety,
     epoch-bump, retransmit rearm, drop_connections/kill teardown [serve_tasks Vec aborts BOTH tasks], R-6d3a
     durable-before-send gate all intact; lock order correct; no-dup-lane; HR1/HR3/no-dep/no-magic-number hold) AND
     caught ONE HIGH the design+I both missed: a learned lane whose accepted connection DIES had its ack watch
     close ⇒ changed()==Err perpetually ⇒ the biased unguarded select arm-1 starved the send+retransmit arms ⇒
     hot-spin + wedge (latent — the happy-path gate keeps both ends alive). FIXED (S3-HIGH): on ack-watch-Err for a
     LEARNED lane, loud-bounce any undelivered window then TERMINATE ⇒ send_durable respawns over the peer's
     re-dialed connection — unifying the HIGH + Stage-3 #2 (RE-connection via respawn) + #4 (no bounce-storm). GATE
     GREEN: io-prod 129 lib + all integration pass (0 failed, 0 ignored — the ca1_reply_on_connection gate is LIVE:
     asymmetric A(empty-book)/B(books A), A learns B then replies over the held conn, DELIVERED + reliable_acked>=1
     + no false NodeUnreachable; + ca1_learned_lane_terminates_when_its_accepted_connection_dies [the HIGH-fix
     regression]; + ca1_learned_peers_table_cap_rejects_and_counts; + stream_kind 3-arm); clippy -D clean;
     coverage-io-prod 95.06% + mesh.rs 95.09% (>= 90 floor). Tier-A (coverage-fast 100%) UNAFFECTED — CA-1 is
     io-prod-only. OWED REFINEMENTS (ledgered, NOT correctness blockers — the terminate-and-respawn is fail-safe +
     consistent with the NodeUnreachable-then-producer-redrive model): (a) window-preserving in-place re-adopt (keep
     the unacked window across a learned-connection blip instead of terminate+producer-redrive — matches booked-lane
     redelivery, an optimization); (b) M6 LearnedPeers eviction (a learned-then-vanished entry is resident until the
     cap or a same-NodeId re-dial — bounded, static-roster-safe; M6 must cover LearnedPeers, not just RecvLedger);
     (c) the deeper coverage of a few learn_dial_in_peer arms (already-live-keep / dead-evict) + a full
     RE-connection respawn e2e; (d) LOW: dialer-side ack_egress opens one idle ACK stream per dialed conn in the
     booked topology (lazy-open optimization). FOLLOW-ON SLICES: S2 (L5 outbound ArcSwap re-plumb + update_peer_addr,
     flips l5_a_rescheduled_peer), S3 (AwaitAdopt re-solicit egress, new InterShardFlow arm), S4 (F2 dest-lane
     numeric guard); then the real-cloud k3d CrashLoop/reschedule e2e; then P4 voxels.**
     **✅ MILESTONE /goal AUDIT post-CA-1-core (wf_1ccb876a, HEAD 42d7f94; 5 read-only Explore+opus survey lenses →
     adversarial refute → synth): HAS_CRITICAL_OR_HIGH (0 critical, 2 HIGH — BOTH FIXED this fold). drift_check =
     PROPORTIONATE, NOT drift (CA-1 is tightly-scoped transfer-first hardening, io-prod-only, Tier-A untouched; the
     adversarial-review-caught HIGH being fixed pre-milestone is the discipline working). scale_endgoal = scales on
     the addressing/authority axis; hundreds-in-one-location rides AoI subs/datagrams NOT one durable lane per
     player (the 256-slot submit channel is shard-to-shard-sized — architecturally sound, not yet load-proven, D-9
     AoI owed). HIGH-1 FIXED (mesh.rs): the learned-lane conn-death bounce loop keyed on `owes_redelivery()` (which
     requires stream.is_none()), but at conn-death the stream is stale-but-Some ⇒ it UNDER-fired ⇒ SILENTLY dropped
     an acked-to-producer, unacked-in-flight window (a no-silent-loss violation on the exact reply path). Cured by
     bouncing on `!lane.retry.is_empty()` (a non-empty in-flight window regardless of stream state). HIGH-2 FIXED
     (celestial.rs, doc-only): the module claimed a cross-binary determinism CI gate "protects" solve_kepler, but
     none exists (the p0_gates test only re-runs the SAME binary on one host). Corrected to state the real
     cross-target-cpu build-and-diff gate is OWED (SPIKE-6a) and lands with P4 terrain; AUTHORITY never depends on
     the bit-equality (poses ship from source) so it is determinism-hygiene, not an authority hole. NEWLY-SURFACED
     OWED (ledgered, non-blocking): (LOW-9) the HIGH-1 loud-bounce BODY is un-exercised — the conn-death regression
     asserts only termination/no-hot-spin; a deterministic bounce-assertion e2e needs an ack-PAUSE test hook (die
     the accepted conn AFTER a learned-lane write but BEFORE its ack) — owed with a mesh test-hook; (LOW-6) when the
     durable outbox goes live on a churny tier, spawn_mesh's OUTBOX_WRITER_CHANNEL_DEPTH boot guard must bound
     booked + peak-active-LEARNED durable lanes (cfg.peers.len() + learned_peers_max), not booked alone (inert today
     — bins pass outbox=None); (LOW-11) the io-prod Tier-B floor is regions-only + package-total (not --branch, not
     per-file) — a known Tier-B honesty note. P4 PRECONDITIONS the audit sharpened (all ledgered): the REAL
     cross-binary determinism gate (SPIKE-6a, + exact-pin glam =0.30.x); the D-41 re-centering MATH + its
     exists-to-be-flipped tripwire (four sites atomic); the FrameSpace/AnchorGen seam (build reusing core::Fence as
     AnchorGen; extend build_app per-capability, never a shard-kind match); the D-38 G-IDENTICAL fixture; the D-9
     within-realm AoI filter. Verdict: the CA-1 core is correct + robust after the 2 HIGH folds; the endgoal is
     gated on S2 outbound-initiate + D-9 AoI + a real determinism gate landing before/at P4.**
     **✅ CA-1 S2 LANDED (outbound L5 re-plumb) — the no-DNS addressing is now COMPLETE (reply + INITIATE). A
     rescheduled/newly-provisioned peer is dialable at its CURRENT address, refreshed at runtime with NO static
     book edit + NO DNS. Implemented: `type PeerTopology = Arc<ArcSwap<BTreeMap<NodeId,SocketAddr>>>` (reuse of the
     workspace arc-swap dep — added to io-prod/Cargo.toml, NO new external dep), seeded from cfg.peers at
     spawn_mesh; `ConnSource::Dial(SocketAddr)` → `ConnSource::Dial(PeerTopology)` so ensure_connection's Dial arm
     RE-READS the peer's current addr on EVERY dial (None ⇒ Err ⇒ Down ⇒ retransmit re-reads next fire, for a peer
     with no current address); `MeshControl::update_peer_addr(peer, addr)` rcu-publishes the new addr (lock-free
     copy-on-write readers) + proactively closes+forgets any stale dialed connection so the peer_writer re-dials
     the new addr on its next attempt. Booked addrs are updatable ONLY via this trusted control surface — never
     via untrusted dial-in (which only populates LearnedPeers — the CA-1 authority split). GATE GREEN: the
     l5_a_rescheduled_peer_at_a_new_address_is_reachable guard FLIPPED (was the R-4e3-item-4 #[ignore]d red guard)
     — A INITIATES to a rescheduled B at its real addr after update_peer_addr, B receives it; io-prod lib 129 pass;
     clippy -D clean; coverage-io-prod 95.09% (mesh.rs 95.15%, ≥ 90). REMAINING CA-1 follow-ons: S3 (AwaitAdopt
     re-solicit egress, new InterShardFlow arm — the D-6 #1 detection trigger), S4 (F2 dest-lane numeric guard);
     then the real-cloud k3d CrashLoop/reschedule e2e (update_peer_addr is the provisioning hook it drives); then
     P4. Owed (ledgered): a fully-prompt live-reschedule re-dial (wake the parked peer_writer + null its local
     conn — today the proactive registry close makes the re-dial happen on the writer's NEXT operation, correct
     not instant); the update_peer_addr None-Dial-arm (peer removed) + the proactive-drop path want a direct unit
     test (the l5 test covers the Some/re-plumb path). S2 POST-IMPL REVIEW (wf_838899e4, SOUND_NO_CRITICAL — 0
     crit/high, verified no redelivery-core regression + rcu lost-update-safe + None-arm no-spin + l5
     deterministic) flagged ONE narrow LOW TOCTOU: an in-flight dial that read the OLD addr at ensure_connection's
     topology-load can complete + register a stale connection into the ConnRegistry AFTER update_peer_addr's
     proactive remove already ran, escaping the proactive close (the sub-ms window between the dial's topology
     read and its registry insert). SELF-HEALING in every realistic reschedule (the old addr is dead ⇒ the stale
     conn fails on its next write ⇒ Down ⇒ re-dial reads the NEW addr); the update is only DELAYED (never lost),
     and only if the OLD endpoint is still live-and-accepting during that window (a NAT/routing re-point, not a
     teardown). Doc-only at this trusted-control-surface tier; the ledgered fully-prompt-wake refinement
     (generation-checked writer wake + null-local-conn) subsumes it, OR a re-read-topology-vs-registered-addr
     check before the connection.is_some() early-return.**
     **📐 CA-1 S3+S4 DESIGN — DEFERRED TO THE DEPLOY-READINESS BUNDLE (2 design rounds wf_a9abdcc2 + wf_46104935,
     1 designer + 3 adversarial opus lenses + adjudicator each; both NEEDS_ANOTHER_ROUND). DECISIONS LOCKED (user):
     source handler = Option B (detection-only re-solicit — no source re-emit; recovery is already covered by the
     R-6 durable outbox + R-4e transport; only a DEAD source needs detection); scope = S3+S4 COUPLED (S3's
     reachable trigger ARMS the over-discard, so its guard must land with it). S3-B shape is SOUND + specified: a
     NEW appended InterShardFlow::ReSolicitBatch(TransientHandoff) (the 18th arm, discriminant 17, RE_SOLICIT_STEP
     =18 — the header/lib.rs closed-set counts LAG at 16/need fixing to 18), ReDriven, on the (AwaitAdopt, Timeout)
     FSM arm (today a no-op catch-all) → EmitReSolicitAdopt → Ephemeral push_flow toward ctx.source → a dead
     source bounces NodeUnreachable → is_confirmed_dead(source) reachable → the existing SourceUnreachablePreAdopt
     discard fires. ⚠️ BUT S4 (the over-discard guard) IS UNSOUND AS A TIMING BOUND (the TRUE BLOCKER, verified
     against real code): a lost BatchAdopted (dest→orch) is pushed Ephemeral via plain push_flow (stub.rs) with NO
     producer redrive, and the transport only redelivers on STREAM-DROP (owes_redelivery = stream.is_none() &&
     !retry.is_empty(), mesh.rs) — NOT on a live lane whose acks stopped. So a lost BatchAdopted on a live lane is
     NEVER redelivered, and NO abort_deadline_ticks bound can guarantee it arrives before the discard (the code's
     own saga.rs:961 + R-4c saga.rs:289 tripwires already flag this cross-check ILL-DEFINED). THE SOUND CURE is a
     MECHANISM change, not a validator: make BatchAdopted a PRODUCER-REDRIVEN ack (a dest-side re-emit until the
     saga advances, symmetric with S3's re-solicit) — then a live-adopted dest's ack ALWAYS eventually arrives →
     exits AwaitAdopt → no discard; a never-adopted dest never sends it → discard fires correctly. NO timing bound
     needed; the redrive distinguishes the cases. (Also owed: the ReSolicitBatch-carries-a-transfer-id vs the
     FireAndForget "no transfer trigger" written-contract tension — resolve doc-vs-semantics; and the conservation
     oracle does NOT backstop a wrong bound — an over-discard is a phantom uncounted-Arriving loss, not a dup, so
     the oracle stays green.) WHY DEFER (SAFE): without S3 the AwaitAdopt egress stays EMPTY ⇒ is_confirmed_dead(
     source) is UNREACHABLE in prod ⇒ the over-discard NEVER ARMS (the F2-tripwire invariant) ⇒ the system is safe
     as-is; S3's ONLY payoff (detecting a dead source mid-transient-handoff) is reachable ONLY in a real cloud
     deploy (a pod dying), which is deploy-readiness-gated. So S3 (detection reachable) + the redriven-BatchAdopted
     cure + S4 land TOGETHER as the deploy-readiness bundle (with the k3d CrashLoop proofs that exercise them),
     NOT now. NEXT: P4 voxels (the milestone audit confirmed P4 ready; first slices = SPIKE-6a determinism gate +
     FrameSpace/AnchorGen + D-41 re-centering math).**
     **⚠️ k3d CLOUD test DE-SCOPED (review CRITICAL, D-12 BINDING): the mesh uses a static literal-IP peer book with NO DNS/
     service resolution — two k3d pods CANNOT address each other until CA-1 (reply-on-connection) lands. So R-6 proves M3 on a
     LOOPBACK CrashLoop test (R-6b, no pod network); the k3d StatefulSet+PVC + real-cloud CrashLoop/reschedule proof is a separate
     CA-1 + L5 + D-13(admin-auth) + M3 slice.** [→R-6 follow-up, DRY] the orchestrator VD_STORE_PATH deny-list guard should be
     lifted into the same shared `check_durable_path` (R-6a added the shared helper + the boot path's allow-list but did NOT
     retouch the orch-crash-tested store boot path — a small DRY cleanup owed). [→R-6, JOINT-INVESTIGATION] self-count-PVC vs
     orchestrator-ISSUED incarnation for hundreds of dynamically-provisioned shards is a user-decision (the R-6a precedence
     `VD_PROCESS_INCARNATION`-wins makes the orchestrator-issued path a zero-rework supersede, so R-6a does NOT lock it in). [L7→R-4'/R-5', load-test mandate] `mesh_redelivery.rs` is 2-node/1-class/1-lane — add an N-peer (16-64)
     sustained-reliable-into-one-node test (no-loss/no-dup, `reliable_acked` keeps pace, no RX-plane collapse under a
     wall-clock bound); it's the companion to R-5' `mesh_under_loss.rs` and the test that empirically surfaces H2. [M2→R-5'
     belt-and-suspenders] add a source-side retained-ghost STALENESS REAPER for the one-shot `GhostFlow::Despawn` band-exit
     (#4 below) — it has NO saga so a per-saga re-solicit structurally cannot reach it, and a lost Despawn = a phantom
     cross-boundary COLLIDER (violates players-physically-collide). **✅ L4 LANDED (R-6c): `incarnation` MOVED from
     `AckFrame` (a single last-class-wins scalar `ack_egress` overwrote each loop iteration) INTO `AckEntry` (per-class).**
     `ack_egress` stamps each entry with its `(peer,class)` incarnation; the `peer_writer` fan-out calls `on_ack(e.incarnation,
     e.epoch, e.ack_through)` per entry. Removes the latent silent send-stall that R-6a's first-class sender-restart could reach
     if a connection is ever reused across a restart carrying two classes at different incarnations. Greenfield hard cutover (no
     rolling-version mix pre-first-deploy). New test `r6c_a_per_class_ack_incarnation_retires_each_lane_against_its_own` proves
     each lane retires against ITS own incarnation + that a wrong-incarnation ack retires nothing (the misfire the fix removes).
     Below the frozen seam (AckFrame is `pub(crate)`). io-prod full suite green, Tier-B mesh.rs 94.30%. [L5→provisioning slice] `peer_writer`'s `addr` is captured once at spawn — a
     NodeId that moves IP needs the writer to RE-READ its address (not just re-dial), so the dynamic-address (CA-1/orch
     provisioning) slice must re-plumb the address, not only the connection.
     **(1a) `BatchHandoff::AwaitAdopt`** — the first-identified instance. A producer-less phase has NO `scan_deadlines` re-drive
     egress, so a lost message is resent ONLY by the harness `FaultFabric` (at-least-once, surviving a receiver crash).
     **⚠️ STATUS UPDATE (post-R-4a/R-4b, /goal `wf_58cc14fb`):** the io-prod `MeshTransport` is NO LONGER purely at-most-once —
     after R-1'..R-4b the sender path (`peer_writer`/`confirm_and_maybe_bounce`, mesh.rs ~1101-1361) is AT-LEAST-ONCE across a
     blip for a traffic-carrying reliable lane, and R-4a closed the idle-after-blip re-drive (see this entry's own 996-1049 log).
     So the STILL-OWED piece for `AwaitAdopt` is narrower than "at-most-once": (i) the genuinely-idle producer-less re-drive PROOF
     (R-4e4/R-5 `mesh_under_loss.rs`), and (ii) the SOURCE-CRASH residual — the retry buffer is RAM, so a source restart before
     the dest adopts is not covered by the transport (the saga self-promote at saga.rs:915-922 recovers it, but its
     `SourceUnreachable` trigger is gated on M3 durable-incarnation + L5 addr-reread). `AwaitAdopt`
     (saga.rs:563-571,631-634): the dest adopts off the source's envelope; the orchestrator only awaits `BatchAdopted`.
     **Owed cure (preferred): a real re-solicit egress on Timeout** (orchestrator re-prompts the source to re-emit the
     `TransientBatch`, or the dest to re-ack) so saga recovery is SELF-SUFFICIENT and stops depending on an un-promised
     transport property — a designed slice (NEW saga action + shard handler, full architecture first per the no-hack
     rule). Alternative: a sender-side durable outbox / retry-until-acked transport (identity_persistence.md:122).
     **✅ The D-37 re-home ADOPT was the SECOND such phase (holistic audit `wf_dd38151d`) — now CURED in Slice 2d
     (commit `<2d>`):** a re-homed `Promoting` carries `rehome_target: Some(target)`, so a `Promoting` Timeout
     re-drives `A::ReHomeAdopt → target` (the live target) NOT `A::Promote → ctx.dest` (the dead original dest a
     re-home fired BECAUSE of); `rehome_event_for`'s `Promoting` branch returns `Timeout` once the directory names the
     live target, so `scan_deadlines` drives the re-solicit. That additive FSM-field fix (one `Promoting` field + the
     Timeout-arm branch + `a_rehomed_promoting_timeout_redrives_the_adopt_to_the_live_target`) is the PROVEN TEMPLATE
     for `AwaitAdopt`'s owed egress. **WHEN (`AwaitAdopt`): before the redb backend / any real rolling deploy.** Until
     then, `AwaitAdopt` orchestrator/peer-crash recovery is proven ONLY vs the FaultFabric.
     **➕ THIRD producer-less phase (holistic audit `wf_ed40e95e`): the durable entity-STATE crossing.**
     `EmitCrossing` (the dest's authoritative pose) is emitted at `CasWon→Swapping` (saga.rs ~:681) and re-emitted
     ONLY by the Swapping-Timeout arm (~:833); once the saga advances Swapping→Demoting→Promoting NOTHING re-emits it
     (Demoting-Timeout re-emits Demote; Promoting-Timeout re-emits Promote/ReHomeAdopt). Over the at-most-once mesh a
     single LOST crossing ⇒ the dest never journals `STUB_CROSSING_STEP`, `promote_apply` DEFERS Ghost→Owned forever
     (stub.rs:1441) yet acks `PromoteAck`, so the gateway's `DeliveredToObservers` never latches and the saga parks
     in `Promoting` (the R1 wedge). Masked in every cell by the FaultFabric's at-least-once redelivery (the stub.rs:1443
     DEFER is reached only as the transient promote-before-crossing race there, never a permanent wedge). **Owed cure:
     the SAME 2d template — a Demoting/Promoting Timeout that ALSO re-emits `EmitCrossing` (idempotent via the dest
     `STUB_CROSSING_STEP` journal dedup, exactly as the Swapping-Timeout arm), OR gate Promoting→Releasing on a
     crossing-applied ack — and it is SUBSUMED by the redelivering transport (the ONE root cure for the whole class
     above, the DRY choice). WHEN: with `AwaitAdopt` (the same deploy precondition).** The
     stub.rs:1443 pin is corrected to name this; the D-2 1d.1-layering note (iii) below is corrected (the crossing IS
     now load-bearing for `DeliveredToObservers` since the 1d.5b.3b sub-announce relocation).
     **➕ FOURTH instance — DIRECT GhostFlow, NOT a saga phase (holistic audit `wf_259d14c0`): the one-shot ghost
     BAND-EXIT `Despawn`.** On band exit the DEST owner emits `GhostFlow::Despawn` ONCE on the rising edge
     (stub.rs:2606) and deregisters the feed in the SAME pass (stub.rs:2632); the SOURCE tears down its retained ghost
     ONLY on RECEIVING that Despawn (stub.rs:1695) — it has no band-exit detection of its own. Over the at-most-once
     mesh a single LOST Despawn ⇒ the source's retained ghost LEAKS FOREVER: it keeps self-emitting its frozen
     last-Owned pose (`is_retained_ghost`) AND stays a kinematic collider — a permanent phantom render source + a
     phantom cross-boundary collider (violating "players physically collide"). Distinct from the saga phases: there is
     NO saga + NO `scan_deadlines`, so a per-saga re-solicit (cure a) CANNOT cover it. **Owed cure: the redelivering
     transport (it rides `GhostReliable`, so the root fix subsumes it), OR a dest "departing-tier" re-emit-until-acked,
     OR a source-side retained-ghost STALENESS REAPER (tear down a retained ghost that has received no Delta for N
     ticks).** WHEN: with the class (the same deploy precondition). DEFERRED.md .3c (the `GhostFlow::Despawn` note ~:498)
     previously said only "a lost Despawn would leak the collider" — it now names the at-most-once + one-shot +
     dest-deregisters + no-producer = permanent-phantom consequence.
  2. **Idle-tick fsync-skip (write-amp).** ✅ The O(directory) delete-all+put-all reconcile is GONE — D-alpha (commit
     `02db31d`) landed the INCREMENTAL per-mutation `dirty`-delta drain (`directory.rs` `take_dirty` →
     `saga_runtime.rs` barrier, no read-back), curing both the write-amp cliff AND the COMP-2 race. STILL OWED: skip
     the durable write + fsync entirely on a FULLY IDLE tick (today the barrier unconditionally stages the Clock key
     every tick, so it submits + fsyncs every tick). ⚠️ SAFETY (holistic audit `wf_ed40e95e` HIGH, RESOLVED-as-safe):
     the parked-flush persist-before-effect gate stays correct under this skip — an idle tick has no state change, so
     it stages no durable delta AND produces no state-DEPENDENT egress (only the loss-tolerant `ClockSync`, which
     depends on the durable clock CEILING persisted at reservation, not this tick's batch). `last_submitted()` at
     flush time therefore ALWAYS covers every flushed effect's state (effects depend on past-or-same-tick commits; a
     same-tick state change stages a delta ⇒ is submitted ⇒ reflected in `last_submitted`). The fsync-skip implementer
     MUST only skip when `staged` is empty (≡ no state change). + a SCALE soak guard (large stable directory, zero
     transfers). WHEN: a later write-amp pass (NOT blocking — correctness already holds).**
  3. **Durable WAL records carry NO schema version** (saga_runtime.rs:124-150 snapshot structs; bare
     `postcard::from_bytes(...).expect` decodes at :833/:848/:856/:873). Safe in-process (same binary wrote the
     bytes); a one-way boot-panic door the instant redb persists across a rolling deploy with a changed
     `SagaCtx`/`SagaSnapshot` shape (D-33 slaved-keys / 1d.6 TLV blob both grow it before prod). **Owed: a
     `WAL_FORMAT_VERSION` prefix per persisted value (or one genesis-stamp record), validated in `rehydrate` BEFORE
     the family scans (fail-loud typed `BootError` or migrate), AND per-record FALLIBLE decode that quarantines +
     counts (`wal_records_skipped`) a torn row instead of aborting the whole boot. WHEN: the FIRST persisted-shape
     change OR the redb swap, whichever is first.**
- **Still owed (engine):** the **redb `Store` backend + off-tick fsync** (io-prod). The **`StoreKey::Tombstone`
  family + the C8 redeliver-PREPARE drop** (a redelivered trigger for an ALREADY-COMPLETED transfer is not yet
  dropped on recover — the live-saga clobber is guarded, but the post-completion zombie needs the persisted
  tombstone; owed with the 1d idempotent-trigger slice). The **explicit head-re-read** for io-prod independent-file
  directory/saga disagreement. **`batch_goes` WAL GC is now a HARD redb precondition** (it is durable+unbounded
  now, not just RAM — the WAL `delete` must be staged alongside the in-RAM `batch_goes.remove` at the shard's
  terminal retire; co-gated with D-7d Slice 2 GC).
- **✅ DURABLE orchestrator kill-9 recovery LANDED (this slice, `p3_orch_kill.rs`):** the POST-commit headline —
  a player crossing while the operator kill-9s the orchestrator mid-`Demoting` survives a `crash`+rebuild: the
  rehydrated saga re-drives the demote/promote forward to `Done`, the avatar settles at DEST, AUTHORITY-UNIQUE,
  zero loss + an anti-theater control (fresh store recovers nothing). Reuses `run_orch_kill_durable` on the same
  `rebuild_orchestrator` + `p2_cluster_durable_orch` machinery as the transient cell.
- **✅ RESOLVED (abort-path fence-neutrality, design `wf_5cfc96b0`, adversarially verified, high-confidence):** the
  PRE-commit abort-with-surviving-source fence divergence is FIXED. Root cause: `abort_clear` bumped the directory
  fence "to fence-out a stale crossing", but that bump-guard (`in_transfer == Some(this)`) is reachable ONLY
  pre/at-CAS (a winning `commit_cas`/`abort_cas` clears the lock at the commit point; post-commit phases never
  abort), where authority never moved AND no crossing was ever emitted (`EmitCrossing` is post-`CasWon`-only) — so
  the bump was PROVABLY spurious wherever it fired, and it stranded the surviving source one fence behind
  (directory@N+1 vs owner@N). That was not merely an oracle nit: a post-abort logout `LeaseRevoke{fence: N}` (and a
  re-transfer keyed at N) is permanently REFUSED by `directory.revoke`'s exact-fence requirement against N+1 — a
  real logout/reassignment WEDGE. **Fix (A): `abort_clear` is now FENCE-NEUTRAL** — clears the lock, leaves the
  fence put (an abort is not an ownership change). `directory.fence == source.entity_fence` (FENCE-9) holds, so no
  re-sync machinery / new message is needed (rejected fix B's source head-re-read stays torn out). The PRE-commit
  orchestrator-kill cell now lands GREEN (`AbortedToSource`, with the asserter strengthened to run AUTHORITY-UNIQUE
  so a future re-introduced bump REDs). Tests flipped: `abort_clear_*` (no-bump) + new `abort_clear_is_fence_neutral`
  pin; `cas_loss_unwinds_with_a_thaw` (Fence 6→5). **⚠️ D-33 forward note:** the future `commit_cas_bundle` abort
  twin MUST inherit fence-neutrality across all N+1 keys (hull + every `ChildOf` passenger, possibly on different
  source shards) — else a compound abort replicates the divergence per passenger / leaves passengers at mixed
  fences relative to their hull. The `abort_clear_is_fence_neutral` pin guards against silently reintroducing it.
- **Still owed (durable crash matrix):** the full **C1–C9 phase × victim sweep + the cut-warmup durable cell**
  (the S5 machinery — `run_orch_kill_durable` + `rebuild_orchestrator` + `p2_cluster_durable_orch` — generalizes;
  only the durable fixtures differ; the post-commit, pre-commit-abort, and anti-theater cells already land).
- **Source:** the P2 plan + Slice-1b audit (ROB-2) + the D-6 design `wf_0f8321dc` + the holistic audits
  `wf_132becc3` (COMP-2 directory-resurrect fix, D6-ROB-1 non-durable-bin warn, D6-1 tombstone/clobber) and
  `wf_7cc86404` (S0–S5 DONE_NO_CRITICAL; the 3 io-prod preconditions above).

### D-7 🟧 Transient transfer: D-7a/b/c + D-7d Slice 1 LANDED (park closed, ballistic, conservation, loss-budget, G-TIER, DURABLE-UNAFFECTED, kill-9 crash resolution); D-7d Slice 2 (burst-scale + dual-death + GC + cap-split + mixed-realm) owed
- **✅ CLOSED — the park (D-7a):** the `IssueTransientGo` LOUD-warn stub that parked a transient saga in
  `CommittingCas` is GONE. The synthesis took the OR-branch: a Transient subject takes a distinct SHORT FSM PATH
  (`saga.rs` `SagaState::BatchCommitting` — `start` enters it directly, skipping Prepare/Cut/Freeze, and `CasWon`
  ends it at `Done` with no Swapping/Demote/Promote), `locks_directory_key(Transient)=false` so it NEVER takes a
  directory lock (burst isolation is STRUCTURAL — a debris burst writes zero directory rows), and the executor
  records the batched go-token into `SagaRuntimeRes.batch_goes` + feeds `CasWon` (one orchestrator write per
  batch, G-TIER). The source→dest set hand-off is the shard↔shard ADOPT-BEFORE-DROP choreography (the demote-
  before-promote twin): SOURCE emits `TransferEnvelope{TransientBatch}` → DEST adopts as the uncounted `Arriving`
  tier + acks `TransferAck::BatchAdopted` → orchestrator emits `InterShardFlow::TransientDrop` → SOURCE drops the
  `Held` items + DEST flips `Arriving→Held`. `TRANSIENT-AUTHORITY-HELD` (`oracle.rs`, no directory cross-check —
  go-token + realm-fence anchored) holds; the e2e gate `tests/tests/p3_transient.rs` proves a 1-item Debris batch
  crosses with no DoubleHeld, no vanish, no loss. Gate-green at 100% Tier-A region+branch.
- **✅ D-7b.1 LANDED (design `wf_55d7e981`, doc `d7b_transient_ballistic.md`):** the STRUCTURAL
  drop-before-promote that makes per-tick `TRANSIENT-CONSERVATION` hold under tick-skew. The D-7a
  broadcast `TransientDrop` (which could double-hold for the stagger window — the dest promotes while
  the source still counts) is REPLACED by the ordered `TransientRelease → DropApplied → TransientDrop`
  (=PROMOTE) ` → DropApplied → ReleaseComplete`: the source flips `Held→Departing` (the new UNCOUNTED
  retained tier) BEFORE the dest promotes, so the holder set transits `{S}→{}→{D}`, never `{S,D}`.
  `oracle::verify_transient_conservation_tick` (`len > 1`, no excuse) is asserted EVERY tick of a
  STAGGERED crossing (`tests/tests/p3_transient.rs` `..._under_stagger`). One `TransientHandoff` struct
  serves the three command arms (DRY); `with_batch_token` shares the ledger lookup. Gate-green at 100%
  Tier-A.
- **✅ D-7b.2 LANDED:** the closed-form `BallisticReadvance` — `kinematics::advance_continuity`
  (dispatch on `KindDef::continuity`, NOT shard kind; reuses `StampedPose::advanced_ballistic`; ships
  the `BallisticReadvance` + stamp-only arms, Guided/RealmAnchored owed at P10/P11/P6) + the
  `readvance_transients` system (advances the `is_held()` subset each tick from its OWN origin; the
  uncounted Arriving/Departing tiers skipped; `par_iter_mut`-ready) + `continuity_of(EntityId)` (unknown
  tag → Frozen, never a panic) + adopt-time `.sanitized()` (never trust a wire pose) +
  `InspectReport.held_transient_poses` (the moving-Debris render-continuity trace). The e2e proves a
  moving debris (and a projectile-speed one UNDER stagger) re-advances CONTINUOUSLY — its dest pose lies
  on the closed-form trajectory `pos0 + vel·(t−t0)·dt_s` (no teleport, no double-advance: source
  increment + dest re-advance compose to one trajectory). `accel = ZERO` (a stub is empty space; P5's
  SphericalSpace introduces seed-derived gravity — same primitive). Gate-green at 100% Tier-A.
- **✅ D-7b.3 LANDED (D-7b COMPLETE):** the per-kind handover-attributable loss budget. A new
  `StubStats.transients_lost_in_handover: DetHashMap<EntityKind,u64>` (distinct from the gross
  `transients_dropped`) is filled by `self_fence_drop_transients`, which buckets ONLY
  `TransientStatus::is_in_handover()` items (the `Held{Some}`/`Crossing`/`Departing` source + `Arriving`
  dest tiers) per kind — a settled `Held{None}` is a resident eviction OUT of budget scope, a corrupt
  kind tag is counted gross but not bucketed (the HR2 `from_tag` Err arm). `oracle::
  verify_transient_loss_budget(reports, kind)` sums it per kind across shards and compares to the
  kind's `LossBudget` (`LostOverBudget` iff `lost > budget`; loss WITHIN budget is tolerated,
  duplication is failure regardless). So a 1000-resident burst eviction can never spuriously trip a
  tiny per-kind budget (the LOSS-BUDGET skeptic's confirmed "wrong population" hazard, fixed). Unit-
  gate: over-budget (5>4), within (3≤4), mixed-kind disentangling, resident-excluded, corrupt-excluded.
  Gate-green at 100% Tier-A. **D-7b (BallisticReadvance + per-tick conservation + loss budget) is DONE.**
- **✅ D-7c LANDED (design `wf_364d34ba`, doc `d7c_transient_burst.md`):** the G-TIER one-write-per-batch
  scale proof + `DURABLE-UNAFFECTED-BY-BURST` differential. The crown insight the 4 skeptics caught:
  `batch_goes.len()==K` is FALSE-GREEN (idempotent `or_insert` collapses a 1000-same-key-write regression
  to `len==1`), so the ONLY observable that turns a per-item-write regression RED is a monotonic
  `SagaRuntimeRes.batch_go_writes: u64` (incremented in `commit_result`'s loop BEFORE `or_insert`, surfaced
  on `InspectReport`). 7c.1 (the Tier-A observable + reader, unit-asserted `==1`) + 7c.2 G-TIER
  (`p3_gtier_burst_write_rate_is_batch_count_not_item_count`: 1000-item batch ⇒ `sum(batch_go_writes)==1`,
  one ledger entry, ZERO directory rows, `held_at_dest==1000`, nothing lost; `…scales_with_batch_count`:
  K=3 distinct batches ⇒ writes==3, NOT the cap-split which is broken at HEAD) + 7c.3 DURABLE-UNAFFECTED
  (`durable_subset(reports, subject, session)` projects only the burst-invariant durable footprint —
  EXCLUDES `owned_transients`/`batch_goes`/`trace_bytes` which legitimately move; `assert_eq!(base, burst)`
  byte-identical under a concurrent 1000-burst, two-sided SENSITIVITY + a real-burst SPECIFICITY control).
  Both variants step a FIXED tick window so they sample tick-aligned (a renewed lease is then burst-
  independent by construction). Multithreading verdict: SEQUENTIAL — rayon NOT adopted (flagged for the
  user, not unilaterally taken; µs-scale work, byte-identical-replay guarantee, parallelism already at the
  shard granularity). Gate-green at 100% Tier-A. **D-7c (G-TIER + DURABLE-UNAFFECTED) is DONE.**
- **✅ D-7d SLICE 1 LANDED (design `wf_f82daa5e-56b`, doc `d7d_transient_crash.md`):** the saga-owned
  transient handoff TAIL + dead-aware crash resolution — the P3 kill-9 headline for the debris class. The
  transient adopt-before-drop choreography moved OUT of the ledger-driven read-only handlers INTO a
  `SagaState::BatchHandoff{phase, new_fence}` FSM tail (replacing tombstone-at-commit) the ONE
  `scan_deadlines` producer re-drives (HR2: the transient twin of the durable Demoting/Promoting tail, no
  second machine); `on_release_complete` now acks `DropApplied(TRANSIENT_COMPLETE_STEP)` (the new
  `SourceRetired` signal) so the tail reaches Done. Dead resolution: `dead_participants` (fed by a
  kill-only `Inbound::NodeUnreachable`) + a `scan_deadlines` gate → a dead SOURCE self-promotes the dest
  from the go-token (zero loss); a dead DEST abandons the source's retained copy as accounted
  loss-within-budget (new `InterShardFlow::TransientAbandon` + `on_transient_abandon`), terminal-on-first-
  fire (no D-37 bounce). Dead-aware oracle twins (`verify_transient_*_excluding`). 3 crash cells
  (source-kill → `BatchCommittedAt{DEST}`, dest-kill → `BatchDroppedWithinBudget{Debris}`, crash-resurrect
  control proving the kill-only gate fires NO spurious resolution). Gate-green at 100% Tier-A; holistic
  audit DONE_NO_CRITICAL. Deviations from the design (correctness-improving, recorded in the doc): NO
  `GcGoToken` in Slice 1 (it would break the D-7c `batch_goes` gate + `TRANSIENT-AUTHORITY-HELD`);
  resolution counters live on the orchestrator (the dest can't distinguish a self-promote).
- **⚠️ DUAL-PARTICIPANT death (audit D7D-1, MEDIUM, owed Slice 2):** if BOTH source AND dest are killed
  mid-handoff, the source-first `scan_deadlines` gate selects `SourceUnreachable` → a promote aimed at the
  DEAD dest (never applied), and NO loss path fires (`on_transient_abandon` is never reached) → the debris
  is held at no live shard AND counted in no loss bucket = a SILENT, uncounted loss. NOT an invariant
  break (no double-hold / no corruption — the cardinal invariant holds unconditionally); a loss-ACCOUNTING
  gap under a simultaneous double-kill crash-storm (debris-only, loss-tolerable). Proper fix: a COUNTED
  orchestrator-side drop on dual-death so the loss enters the budget gate, WITH a dual-victim harness
  `Scenario` (the current single-`victim` field cannot express it). The D-7d doc's conservation claim is
  scoped to single-participant kills accordingly.
- **Still owed (D-7d Slice 2+):** the BURST-scale crash cells (1000-item wholesale self-promote;
  cluster-level over-budget DEST-kill → `LostOverBudget` honest RED) + the dual-death counted-drop above;
  `StubConfig.max_items_per_batch` (the source-side wire-frame cap — DEFERRED whole from D-7c because a
  cap-split is lossy-if-triggered under the derived-id dedup journal and `MsgClass::Saga` is reliable, so
  the sub-batch→distinct-go-token wiring must land WITH it here) + **audit COMP-1 (LOW):** the io-prod
  RELIABLE-arm `FrameError::TooLarge` currently mis-classifies a caller-side framing fault as transport
  DEATH (tears the connection / surfaces `NodeUnreachable`) — split `write_wireframe`/`peer_writer` so a
  `TooLarge` is a counted drop + `tracing::error` that NEVER tears down the connection (mirror the datagram
  arm's `TooLarge` guard) + a wire-level test asserting an over-cap reliable `TransferEnvelope` does NOT
  surface as `NodeUnreachable`; no trigger today (K=1 single-DEST, no >1 MiB frame constructible), lands
  WITH the cap; MIXED-KIND / MULTI-DEST-REALM burst (a
  3rd shard + a `seed_transient_crossing_to(dest, realm)` parameterization — structurally unreachable
  through the current `dest=DEST`/`to_realm=System(8)`-hardcoded API); bounded GC of completed go-tokens
  (`batch_goes` unbounded — the oracle needs the live record until a drop-completion signal Slice 2
  co-designs; GC MUST fire at `on_release_complete`, the shard's terminal retire, NOT a timeout — and NOT
  on the crash-path Done, which would break the D-7c quiescence count); the durable go-token WAL
  (in-memory, [[D-6]]); the ORCHESTRATOR-kill cell ([[D-6]] — `batch_goes`/live-saga set are in-memory);
  D-3 lease-lapse liveness as the real crash-vs-partition discriminator + the "flap" fault (replacing the
  kill-only `NodeUnreachable` heuristic — the prod 20s-idle-reap-of-a-live-peer path is gate-invisible today).
- **Shard-side handoff is O(K·M), not O(batch) (audit D-7c SCALE-1, LOW):** the one-write-per-batch claim
  is honest for the ORCHESTRATOR control plane ONLY (doc `d7c_transient_burst.md` lines 36/86). On the
  shard, `on_transient_release`/`on_transient_promote`/`on_release_complete` (`stub.rs`) each scan the
  WHOLE `OwnedTransients` map filtered by `batch == cmd.transfer`, so the 3 handoff phases cost
  O(K-concurrent-batches × M-resident) and `readvance_transients` is O(M)/tick. Structurally unreachable
  today (`seed_transient_crossing` hardcodes a single DEST/realm → K=1), so it bites only at P11 combat
  scale. PROPER fix (lands WITH the multi-batch/multi-realm burst above): index `OwnedTransients` by batch
  (a secondary `BTreeMap<TransferId, BTreeSet<EntityId>>` of in-handover members maintained at status
  transitions) so the handoff phases touch O(batch_size); keep `readvance_transients` a full scan (it is
  genuinely all-held work). Per the load-tests-when-applicable standard, add a shard-tick burst-soak load
  test (~10k resident × ~100 concurrent bursts) BEFORE P11 relies on the burst path.
- **Where:** `saga.rs` (`BatchCommitting`), `saga_runtime.rs` (`locks_directory_key`, `IssueTransientGo` executor,
  `batch_goes`, `handle_batch_adopted`), `stub.rs` (`OwnedTransients`/`TransientStatus`, `emit_transient_batch`,
  `adopt_transient_batch`, `on_transient_drop`, self-fence drop), `wire/intershard.rs` (`TransferAck::BatchAdopted`,
  `InterShardFlow::TransientDrop`, `TRANSIENT_BATCH_STEP`/`TRANSIENT_DROP_STEP`), `core/ids.rs` (`BatchId`).
- **When / proper:** D-7b/c/d (this slice cycle); the crash cells compose with the P3 Slice-1 matrix.
- **Source:** the P2 plan + Slice-1b audit (CPO-3) + the D-7 design `wf_2eaa4df7` (3 blockers → short-FSM-path).

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

### D-9 🟩 THE INTEREST — one body per cube of occupants, to the observers that hold it (BUILT 2026-09-05, foundation slice 2, owner-approved "yes to all")
- **✅ BUILT (2026-09-05; the angle re-ruled 2026-09-06).** The rule: an occupant is shipped to an observer
  inside the occupant's REACH — its look (`Dot::look_extent_m`, born as `vd_core::look::OCCUPANT_FIGURE_EXTENT_M`
  = 1 m until the character states a body) at THE DRAWABLE ANGLE, one pixel at the reference view (45° over
  720 rows, `vd_core::geometry::drawable_theta_min_rad`, ~869 m for a 1 m figure, ~1.7 km for a 2 m
  character). ★ OWNER 2026-09-06: the first build used the realm WAKE angle (1.5°, ~76 m) and the owner
  refused it — *"players should be seen from far enough"* — so a person is shipped while it can still be
  DRAWN (cheap), where a realm is woken when it is a dot worth WAKING (expensive). The reference view is
  ONE home in core; the renderer's capture height and the harness's fitted field of view read it — plus a
  lead that grows with the closing speed over the interpolation buffer plus one tick
  (`vd_wire::channels::INTERP_BUFFER_MS`, now the ONE home the client's tuning reads too). The shard sorts
  its rows into cubes of one reach in its OWN frame (`crates/sim/src/stub/interest.rs`); every dot it holds
  observes from its cube; an observer TAKES the cubes within its step count and KEEPS a cube one step
  further out while it holds a figure the observer was shown (no flapping at a boundary). One body per cube
  ships on the appended `ShardToGateway::FrameFor { recipients }` (mesh minor 30, disc 14) to the observers
  holding it; a cube every observer holds rides the older whole-realm `Frame`, and a row that cannot be
  placed in this frame ships to everybody, counted. An occupant that left an observer's hold is told to
  that observer ONCE on `ShardToGateway::EntityOutOfInterest` (disc 15, Control, Retained), which the
  gateway delivers as the EXISTING `EventMsg::EntityRemoved` — the client evicts a figure only on a sound
  signal, never on silence, and a later row re-creates it; the client wire is unchanged. The gateway fans a
  named body only to its recipients (`on_shard_frame`), counted apart when a name is stale
  (`interest_recipient_unsubscribed`, admin 63) and per removal delivered (`interest_removals_fanned`, 64).
  The double encode died with it (each cube's rows are encoded once per chunk).
- **MEASURED (the owed gate, `crates/sim/src/stub/tests/interest.rs`):** four groups of 32 occupants a
  ten kilometres apart in one realm — every observer receives its own 32 rows, 128 × 32 rows shipped where
  whole-realm emission shipped 128 × 128; the dense case (128 in one place) keeps everyone seeing everyone.
  Two neighbours share one whole-realm body; two strangers each get a body naming them alone; an occupant
  walking out of reach is told once, silence after, and its return is silent.
- **STILL A NUMBER TO WATCH:** with today's one-metre figure a lone person is shipped inside ~870 m plus the
  lead; the character's real body roughly doubles it, and the suit's light adds the reach by light on top
  (the limiting-magnitude rule realms use), which needs a light datum per occupant. A hull with people
  aboard is a realm and follows its own reach.
- **STILL OWED:** the realm-lane twin (`RealmFrame`/window fan-out O(observers×movers), D-RLM-18 item iii) is
  a different lane and stays where it is ledgered.
- *(the original entry, kept for the record)* Snapshot emit was whole-realm broadcast + double-encoded — NO within-realm per-entity AoI (the load-bearing hundreds-in-one-location seam; see [[D-41]])
- **Missing:** `emit_frames` builds ONE `Vec<EntitySnap>` from EVERY emitting dot in the realm (filtered only by
  the authority-state `emits()` + the MTU budget, NEVER by observer) and the gateway `on_shard_frame` re-tags ONE
  shared `Arc` body per `SubId` and refcount-fans IDENTICAL bytes to every subscriber (the SCALE-1 optimization).
  There is NO per-entity relevance/AoI filter anywhere, so every client in a location receives every other entity's
  pose every tick. `partition_entities` additionally re-encodes every entity twice per tick (a sizing pass + the
  per-chunk encode). At **hundreds of users in ONE location** (one realm, one shard, one sub) this is
  O(entities×clients) ≈ N× the necessary per-client bytes at 20 Hz — the canonical dense-crowd / firefight wall.
- **✅ BASELINE FIXTURE LANDED (Jul 2026) — the owed "N-in-one-realm" load proof (was: largest was 32 sessions):**
  `tests/tests/p1_gates.rs` test `p1_volume_dense_hundreds_walk_under_invariants` runs N≥128 sessions in ONE realm
  (env-RAISABLE `VD_DENSITY_CLIENTS`, clamped so CI always proves ≥128) and asserts EVERY invariant holds at scale
  (authority-unique per tick, all-N spawned, ≥80%-of-ideal applied inputs, no InputLog eviction, every dot sees all N,
  no reliable shed, wire-truth, input-conservation + authority-settled at rest). It REPORTS (never gates) the D-9 O(N²)
  fan-out as an outbound-MESSAGE-COUNT baseline (NOT bytes — `Stepped.sent` is messages, a proxy for the byte fan-out) —
  measured n=128: gateway_msgs_total=357248, peak=1920 msgs/tick, elapsed 7.6s. So "hundreds in one location" is
  CORRECTNESS-proven; the D-9 byte-volume reshape below is the remaining owed work (P6, additive, client wire stays
  frozen). A soak at higher N prints the growing quadratic curve without failing (it is deferred, not gated).
- **NOT covered by the cross-shard InterestSet:** `connection_plane.md:189` / `PLAN.md:141` interest governs which
  SHARDS a session subscribes to + ghost-band membership between neighbor shards — it CANNOT cull co-located players
  who all share one shard/realm/sub. Also DISTINCT from [[D-4]](a) (client-side `DeliveredView` eviction of
  band-departed entities via `EventMsg::EntityRemoved` — render cleanup, not the shard-emit/gateway-fanout) and from
  [[D-24]](d) SCALE-FANOUT-COPY (only the per-frame `subscribers_of` `Vec` alloc — the smaller issue).
- **Where:** `crates/sim/src/stub.rs` (`emit_frames` ~2710); `crates/connection-plane/src/gateway.rs`
  (`on_shard_frame` ~1798); `crates/wire/src/channels.rs` (`partition_entities` double-encode).
- **Shape of the fix (a RESHAPE of the emit/sub model, NOT a free delta layer and NOT a from-studs rewrite):** push
  interest INTO the shard as a per-cell / interest-group spatial index so `emit_frames` produces per-interest-group
  snapshots; the gateway still Arc-shares ONE body per GROUP (re-key "one body per sub" → "one body per cell") so
  SCALE-1's technique is PRESERVED. The double-encode dies in the same refactor. This composes with the per-cell
  grid [[D-41]] needs for a possible region split. **Scope precisely (audit `wf_2c963246`):** only the
  CLIENT-FACING `SnapshotDatagram` wire shape stays FROZEN; the INTERNAL shard→gateway seam DOES change — the shard
  must emit per-cell bodies (a new appended field/arm on `ShardToGateway::Frame` — postcard append-safe, as
  `SubscriptionReady` already did) AND the gateway must learn each session's cell membership (a session→cell index;
  today session→sub is keyed 1:1 by shard `NodeId`). That is the SAME re-key as [[D-39]] sub-bullet 5
  ((shard) → (shard,cell/sub)), so D-9 and D-39.5 are ONE coupled slice — still additive (no landed decision undone,
  the Arc-share survives), just not "a shard-local index under a frozen wire". A P6 implementer must budget BOTH.
- **When / proper:** RE-DATED — the original "the P2 interest-management slice" label is STALE (P2's transfer gates
  landed without it; the project is now ~P3 and AoI is unbuilt, so the "a phase isn't done until its DEFERRED entries
  flip" gate could not catch it). Lands with the real interest work (the P6 InterestSet schedule, or earlier if the
  dense-crowd target is prioritized), and MUST be measured before any hundreds-in-one-location / dense-PvP load run.
- **Owed gate (load-tests-when-applicable):** an N-in-one-realm load fixture (today's largest is 32 sessions,
  `tests/tests/p1_gates.rs`) asserting per-client snapshot bytes scale with VISIBLE-NEIGHBOR count, not realm
  population — converts "negligible, trust me" into a measured floor.
- **Source:** whole-codebase audits `wf_43fea0dd` (SCALE-B) + `wf_032b80eb` (PvP/large-scale: H1, the within-realm AoI gap).

### D-10 🟧 PARTIAL — Per-node honesty counters exist on every node; the ORCHESTRATOR + GATEWAY now publish, the SHARD/follower do not yet
- **RG-4 progress (RLM RG-4a, committed `bd01f5a`..`f744a5c`):** the GATEWAY now publishes its own
  `/admin/snapshot` (the `GatewayView` — all 22 `GatewayStats` counters + `sessions_open` + `dynamic_shards`),
  through the SAME lock-free `PublishedSnapshot` cell the orchestrator uses (one shared `vd_io_prod::admin`
  contract + `vd_bins::spawn_admin_server`, both bins). The reactive-greeting `presence_announces` + the
  demand `dynamic_shards`/`home_bootstrap_timeouts` are now curl-able and the RG-4c demand-login e2e reads its
  proof from them. The mesh `learned_peers_rejected` shed counter is scraped on `/metrics`. REMAINING: the
  STUB-SHARD + follower counters (`StubStats`/`FollowerState.undecodable`) are still unexposed — a demand-spawned
  shard has no admin endpoint yet. Flips 🟩 when the shard bin grows the same published-snapshot admin cell.
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
- **1d.1 update:** the dest shard's `AppliedSteps` (1d.0) now has its FIRST consumer — `on_transfer_envelope`
  consults-then-records by `(transfer, STUB_CROSSING_STEP)` before applying a crossing. It remains IN-MEMORY (the
  durable redb backing is still P3, this entry's point). **Two bounds are now OWED** (both need terminal-awareness
  the dest shard lacks, so both land with the journal-retention slice — neither grows per-tick in a healthy run):
  (1) `AppliedSteps` retention (drop a transfer's steps on its terminal, as 1d.0 flagged); (2) `PendingCrossings`
  cleanup (an entry whose entity NEVER adopts — a misrouted crossing — would linger). Recorded on the resources.
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
- **⚠️ PARTIAL (Slice 2a):** D-1's lock-clear is now LIVE, but RACE-1 is NOT unmasked — `ClearTransferLock` fires at
  the TERMINAL `Aborted` edge (after BOTH compensator acks), by which point the gateway's `apply_abort` has pruned
  `Session.transfer`, so an immediate re-transfer's `apply_prepare` finds NO stale journal to discard. WEDGE-1's
  PRE-FREEZE case is now backstopped by the producer (a wedged pre-freeze saga times out → aborts → the lock clears).
  STILL OWED: WEDGE-1's POST-COMMIT case (a `Bye` after commit drops the session+journal while the dest already owns
  — abort is wrong + there is no session to ack; needs gateway re-adoption, P3) + the keyed-map `TransferProgress`
  (the SIGNALS slice, once multiple correlation streams multiplex one session).
- **When / proper:** the keyed map + WEDGE-1-post-commit at the SIGNALS slice / P3 (the producer + terminal-edge
  lock-clear handled the Slice-2a-scoped halves).
- **Source:** whole-codebase audit `wwk1uh5k9` (RACE-1 / WEDGE-1, pinned; masked by D-1); Slice 2a `wf_9f22c70d`.

### D-21 🟩 `TransferAck` got its first consumer (Slice 1d.1): the `StubCrossing` receiver + the source pose flush
- **Landed (1d.1):** the shard-bound `Transfer(TransferEnvelope)`-arm RECEIVER now exists. The DEST consults the
  1d.0 `AppliedSteps` journal by `(transfer_id, STUB_CROSSING_STEP)`, applies the sanitized pose to the adopted
  dot (FirstApply), and replies `TransferAck::Accepted` (re-ack only on redelivery). The SOURCE replies a NEW
  `TransferAck::SourceFlushed` arm (carrying its pose + drain watermark) to the orchestrator's new `FlushSource`
  request. All of `FlushSource`, `TransferAck::Accepted`, and `TransferAck::SourceFlushed` are produced AND
  consumed end-to-end — proven by `p2_dod_cross_cut_input_is_conserved_exactly_once`, whose load-bearing
  discriminator is the dest pose's FRAME (the crossing carries the SOURCE realm frame seed-7, which the dest's
  own input integration can never produce — its adopt-default is the DEST frame seed-8), so the gate goes RED if
  the crossing is dropped (verified by mutation; a `pos != ZERO` check could not — the dest's own drained input
  also moves it off origin).
- **Residual (one reserved sub-variant):** `TransferAck::Rejected{reason}` has NO producer yet. A `StubCrossing`
  is emitted POST-commit / forward-only, so spatial admissibility is gated PRE-commit at `PrepareSubscribe`
  (`PrepareResult::Rejected{Spatial}`) — the dest never rejects a committed crossing on spatial grounds (it would
  STRAND the entity). `sanitized()` is the post-commit network-trust chokepoint; a below-fence crossing is a
  silent counted drop (`crossings_stale`, fence rule 1), not a `Rejected`. `Rejected` awaits a non-spatial
  producer (version-floor / epoch / unknown-kind on the TLV blob — a later slice, with the kind registry +
  version-floor handshake). Reconcile then whether the shard step-reject reasons fold onto the client-facing ones.
- **Where:** `crates/wire/src/intershard.rs` (`FlushSource`, `TransferAck::SourceFlushed`, the two new
  `InterShardFlow` arms, `FLUSH_SOURCE_STEP`/`STUB_CROSSING_STEP`); `crates/node/src/saga_runtime.rs`
  (`emit_crossing`/`build_crossing`, the `SourceFlushed` stash+decode); `crates/sim/src/stub.rs`
  (`on_flush_source`, `on_transfer_envelope`, `apply_crossing`, `PendingCrossings`).
- **Source:** whole-codebase convergence audit `wf_8d81c753` (DRY-1); CONSUMED in Slice 1d.1.

---

### D-48 🟥 Nothing we write to disk carries a FORMAT version, and a shipped saga's PHASE SET cannot evolve — a code change that alters the persisted snapshot has no defined migration path
- **What's missing:** the durable records are postcard-encoded values (`SagaSnapshot`, `OwnerRecord`, per-batch
  `BatchGo`, `Tombstone`, `Clock`) under 1-byte-tagged `StoreKey`s, and **no format version exists anywhere** —
  `grep` for `WAL_VERSION`/`STORE_VERSION`/`SCHEMA_VERSION` across `crates/` returns nothing. `TRANSFER_n`
  (`wire/src/intershard.rs`) is the WIRE version and says nothing about the store. Two distinct consequences,
  only the first of which is presently ledgered (as D-6's owed item #3, "WAL version+Tombstone"):
  1. **Format.** Change a field in `SagaSnapshot`, redeploy over an existing store, and rehydrate decodes
     postcard bytes against the wrong shape. Decode-to-`Default` is BANNED for Durable kinds, so the only
     honest outcome is a LOUD REFUSAL — but nothing today *enforces* which outcome you get.
  2. **Phase evolution (no home until now).** Recovery re-hydrates a QUIESCENT PHASE SNAPSHOT and re-drives it
     through the existing per-phase `Timeout` arms. If a later slice inserts, removes, renumbers or reorders a
     saga phase, an in-flight saga persisted under the OLD phase set has no defined meaning to the new binary.
     This is not a wire problem (one process writes and reads it) and no version floor covers it. It is
     invisible today only because we have never rolled a code change over a non-empty store with transfers in
     flight — i.e. the first rolling redeploy is the first time it can bite.
- **Changes to implement:**
  1. A `StoreFormat` stamp (its own `StoreKey`) written once at init and verified on every `RedbStore::open` /
     `rehydrate`. Mismatch = loud refusal — never a genesis, never a best-effort decode.
  2. Per-family migration at rehydrate: read the old shape, convert **field-by-field**, write the new. NEVER
     re-use bytes across versions and never a "same-bytes" shortcut, even when two versions look identical
     today (the discipline in `scripts/rivet_codebase_analysis_20260811.md` §7.7). Add a writer-N+1/reader-N
     test in the same shape as the `kind_blob_evolution` gate owed under D-31.
  3. **The phase-evolution decision** — either (a) durable steps addressed by a stable, insertion-tolerant
     LOCATION plus a per-step VERSION, so a step can be inserted into or removed from a saga that is already
     mid-flight and replay detects divergence loudly (reference: Rivet's gasoline ordinate scheme —
     `docs-internal/engine/GASOLINE/WORKFLOW_HISTORY.md` + `history/cursor.rs`, coordinates `{1}`, `{1,4}`,
     `{0.1}`, `{2,11,4.1}` with a `HistoryDiverged` error; ranked #1 harvest item in
     `scripts/rivet_codebase_analysis_20260811.md` §7.1), or (b) a WRITTEN policy that all in-flight transfers
     are DRAINED before any deploy that changes the phase set. (b) is legitimate and much cheaper — it makes
     the constraint operational rather than structural — but it must then be enforced by the deploy procedure,
     not by hope, and it forecloses hot-patching a wedged transfer.
  4. The `Tombstone` half already named in D-6 owed #3.
- **Where:** `crates/io-prod/src/store.rs` (`RedbStore::open`); `crates/node/src/saga_runtime.rs`
  (`SagaSnapshot`, the `commit_result` write-back, `rehydrate`); the `Store` seam + `StoreKey` in
  `crates/sim/src/io`; and the deploy procedure itself for option 3(b).
- **When:** item 1 lands **with the D-47 deploy preconditions** — it is the cheap half and it converts an
  undefined decode into a refusal. Items 2–3 **before the first rolling redeploy over a live store**. Not
  before: at in-process scale every store is fresh, so this is genuinely inert today.
- **Source:** the durability audit of 2026-08-11 (grep-verified absence of a store format version), sharpened
  by the Rivet codebase comparison. Extends D-6 owed item #3 with the phase-evolution half, which had no home.

---

### D-49 🟩 The coverage-fast artifact class: per-record report rows owning no source-line miss in the merged report — filtered by an objective rule (owner: option 3, 2026-08-15)

- **The artifact class, measured:** cargo-llvm-cov counts coverage rows PER COMPILED RECORD, and Tier-A
  compiles the same source into many objects (each crate's own test binary plus every dependent crate's).
  Three shapes of row therefore show as "missed" while every source LINE and every BRANCH SIDE of the
  MERGED report is covered: (1) a `?` operator's never-taken early-return micro-region (a one-column span
  on a covered line); (2) a lazy closure argument's body (`map_or_else` defaults, assert messages) never
  evaluated; (3) a per-crate-hash duplicate instantiation of a span another record fully covers. Measured
  2026-08-15 on the pinned toolchain: 27 such rows (17 regions, 4 functions, 5 lines, 1 branch shadow)
  and ZERO missed lines in the merged lcov — the raw `--fail-under-*` gate was red on rows no source line
  backs. A toolchain bump was tried first and rejected by measurement: nightly-2026-07-21 +
  cargo-llvm-cov 0.8.7 reproduced the identical totals (no artifact drop, no new miss), so the pin stays
  at nightly-2026-06-06.
- **The rule (owner-picked option 3, 2026-08-15 — docs/design/owner_decisions_2026-08-15.md item 8),
  objective, never a blessed location list:** `scripts/coverage_gate.py` makes the coverage-fast pass/fail
  decision from the merged report. A REAL miss — a merged-lcov `DA` line with zero hits, or a `BRDA` side
  never taken — still FAILS the build with its location. Every other missed row is DROPPED, and the
  dropped count is PRINTED every run (shed-loud), with the never-executed micro-spans listed, so growth
  in the artifact class stays visible in review even though it does not fail the build.
- **The accepted residual, stated honestly:** a genuinely-new never-evaluated closure body or `?` early
  return on an otherwise-covered line is dropped by the same rule (it owns no missed line). The printed
  per-run counter + span listing is the tripwire for that class; a genuinely dead function or arm still
  fails (its lines go unhit ⇒ real missed `DA` rows).
- **Where:** `scripts/coverage_gate.py` (the decision), `justfile` `coverage-fast` (produces the merged
  json+lcov and delegates the verdict).
- **When flipped:** 2026-08-15 — the four real branch misses the merged report DID back (nav's non-finite
  brake arm, an `assert!(a && b)` split, the echo pin's third condition, the detector stamp invariant's
  panic arm) were covered by tests in the same change, never exempted; `coverage-exemptions.toml` stays
  empty.

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

### D-44 🟧 N-shard roster generalization for the k3d deploy (Track R / 1d.2 M-2 scope split) — 3-shard PROCESS cluster + acceptance capstone LANDED (S5b); the fully-GENERIC `Vec`-of-N still owed for k3d StatefulSet ordinals
- **⚠ AMENDED (2026-08-14, THE-world cluster arc — see D-WORLD-1..9 below):** `ClusterShape` is now
  `Single | Dual | Chain | Demand`, and every pre-booked realm DERIVES from `world_roster` (the one
  derivation point) — never a named const. `Planet(7)`/`Station(7)`/`Area(7)`/`System(8)` named realms
  THE world does not contain, so the six-shard `Forest` could never boot: four of its shards died at
  `guard_regions_nest` (0 ambient roots) and stayed invisible because bring-up never polled
  `Cluster::first_exited` (D-WORLD-7/8). The `Vec`-of-N k3d generalization below is UNCHANGED and still
  owed — and now SMALLER, because `realm_shards(shape, addrs, p)` already returns a `Vec<RealmShard>`
  that is a function of the world. The station/area addr + `DevPortScheme` slots stay RESERVED and
  unused, retired with that work (no Tier-A port renumbering). Read the present-tense "PROVEN" claims
  below through this amendment: they describe clusters that no longer exist, kept as history.
- **Landed (Track R / 1d.2 Batch A):** the LOCAL 2-process dual-shard crossing playground — a single extra
  DEST shard (`SHARD_B = NodeId(4)`, realm `DEV.realm_seed_b`) wired as a clean 2-shard extension: the
  `--dual` launcher flag, `shard_b_env`, `ClusterAddrs.shard_b`, `DevPortScheme` `SHARD_B_OFFSET`/
  `PROBE_SHARD_B_OFFSET`, the gateway `VD_KNOWN_SHARDS` union + orchestrator `VD_ROSTER`, and the C1
  both-realms readiness gate (`AdminSnapshot::realms_present`). ~~PROVEN by
  `crates/bins/tests/dual_cluster_crossing_smoke.rs` (a real dot re-homes SOURCE→DEST over process QUIC).~~
  **⚠ CORRECTED (D-WORLD-8, 2026-08-14):** that proof stood on an INJECTED born-inside boundary file
  whose geometry existed in no world, and the gate later sat RED for the one D-WORLD-8 cause (a cluster
  naming realms THE world does not contain). The smoke is REWRITTEN on THE world's own 150 m home shell
  (Dual, zero injection) and is green there — the sentence stays as history of what the old green meant.
- **Landed (S5b, 2026-07-18) — the 3-shard PROCESS cluster + acceptance capstone:** the `--triple` launcher
  mode adds the GALAXY between-space shard (`GALAXY = NodeId(5)`, realm `System(GALAXY_SEED=1)`) as a THIRD
  stub shard, so a durable dot walks the FULL seed-forest chain `System 7 → Galaxy → System 8` and back over
  a REAL 3-shard process cluster. The `dual: bool` env-builder param became a `ClusterShape { Single, Dual,
  Triple }` DATA enum (HR3: the peer/clock/roster/known-shards lists GROW with the shape — never a shard-kind
  branch); added `galaxy_env` (twin of `shard_b_env`, seed-forest boot, NO `VD_REALM_BOUNDARIES` override),
  `ClusterAddrs.{galaxy,galaxy_probe}`, `DevPortScheme` `GALAXY_OFFSET`/`PROBE_GALAXY_OFFSET`
  (`RESERVED_NODE_PORTS` 9→11), and the STRICT all-three-realms `--triple` readiness gate over
  `[System(7), System(1), System(8)]`. The G1 worldgen change (`GALAXY_R_M` 1000→180, `MAX_RENDERABLE_EXTENT_M`
  100→200) makes the client render the Galaxy as the CONTAINING box so the dot is never orphaned in the
  between-space. PROVEN at boot: `vd-devcluster up --triple` grants all three realms to node-3/4/5, each shard
  plants its seed neighbourhood (counts 5/4/3). The client-observed pixel capstone is
  `crates/bins/tests/triple_cluster_crossing_smoke.rs` (GPU-gated `just triple-crossing-smoke`): a REAL durable
  player, driven by `WalkTo`, whose location label flips `System 7 → System 1 → System 8 → System 1 → System 7`
  with the Galaxy-box containment asserted in pixels at the gap waypoint.
- **Owed (GPU pixel-capture walk re-base, task #149):** the `--triple` co-hosting cluster now THRASHES under
  node-per-realm, so `triple_cluster_crossing_smoke.rs` + `just triple-crossing-smoke` were RETIRED (deleted).
  The headless `crates/bins/tests/node_per_realm_walk.rs` (`just node-per-realm-walk`, in `just gate`) is the
  interim CI walk gate, and the live `--demand` playground window is the interim VISUAL proof (the `--forest`
  shape was RETIRED by the Stage-C batch-1 cluster rework — shapes are now Single|Dual|Chain|Demand on THE
  world, and a cluster naming realms the world does not contain fails loud at boot). The GPU pixel-capture walk
  proof (the client-observed `location`-label + Galaxy-box-containment capstone) is OWED a re-base onto the
  `--demand` cluster (a headless `client --capture --realm-boxes` over demand-spawned shards);
  `render_crossing_smoke` (Stage-C batch 1) carries the CROSSING half in pixels on `--dual` today.
- **Landed (NODE-PER-REALM, 2026-07-18, task #149) — the `ClusterShape::Forest` SIX-shard cluster:** the user
  decided re-home should be NODE-PER-REALM (each realm on its OWN node), so EVERY re-home is a uniform CROSS-NODE
  saga and the source==dest (co-hosted) degenerate case never arises. The `Forest` shape stands up
  orchestrator + gateway + SIX single-realm shards — System 7, Planet 7, Station 7, Area 7, Galaxy (System 1),
  System 8 — with NO `VD_HELD_REALMS` co-hosting. This proved out the DATA-driven seam this ledger anticipated:
  a `RealmShard { node, realm, quic, probe }` descriptor + `ClusterShape::extra_realm_shards(shape) -> Vec<RealmShard>`
  that the orchestrator/gateway/source-shard env builders + the launcher iterate (never a shard-kind branch),
  plus a `VD_REALM_KIND` (`system|planet|station|area`) env so the shard bin boots a Planet/Station/Area realm
  (before this a shard could only host `System(seed)`; the frame comes from the seed forest region). Added the
  `PLANET_A/STATION_A/AREA_A_SHARD` node consts, six `ClusterAddrs` fields + six `DevPortScheme` offsets
  (`RESERVED_NODE_PORTS` 11→17), `realm_shard_env` (twin of `shard_b_env`/`galaxy_env`), the `--forest` launcher
  flag + all-six-realms readiness gate, and re-pointed `scripts/crossing-playground.sh` to `--forest`. ~~PROVEN by
  `crates/bins/tests/node_per_realm_walk.rs` (a REAL durable player, `WalkTo`-driven over process QUIC, walks
  System 7 → Planet 7 → Area 7 → Planet 7 → System 7 → Galaxy → System 8: every leg ARRIVES with no freeze, and
  the subject Entity's directory fence stays ≤ 12 — one clean commit per crossing, no thrash).~~
  **⚠ CORRECTED (D-WORLD-7, 2026-08-14): the walk was structurally unrunnable AND inert.** Four of the
  six Forest shards named realms THE world does not contain and died at boot — invisibly, because the
  gate never polled `Cluster::first_exited` — and EVERY leg target (x = 15/25/35/65) sat INSIDE THE
  world's 150 m home shell, so the green run crossed nothing. The gate is REWRITTEN on the Chain shapes
  (see D-WORLD-7); this paragraph stays as the record of what the old green claimed. The `Dual`/`Triple`
  co-hosting shapes are KEPT unchanged (their `extra_realm_shards` is empty — byte-identical). This REPLACES the
  old `cohosted_input_freeze_repro.rs` (which reproduced the co-hosting FREEZE the source==dest patches tried to
  cure; those two sim-level patches were REVERTED — node-per-realm makes them unnecessary).
- **Missing (for the k3d/#123 deploy):** the wiring is still a FIXED-shape ENUM (Single/Dual/Triple/Forest =
  1/2/3/6 NAMED shards — one const + one twin builder + one `ClusterAddrs` field pair + reserved ports per
  shard). k8s scales shards as StatefulSet ordinals of arbitrary N, so for the cloud deploy the named
  `SHARD_B`/`GALAXY`/`PLANET_A`/… consts + `shard_b_env`/`galaxy_env`/`realm_shard_env` builders must collapse
  into `ClusterAddrs.shards: Vec<RealmShard>`, a `DevPortScheme` `Vec<(shard, probe)>` per index, and N-entry
  `VD_KNOWN_SHARDS`/`VD_ROSTER`/`VD_CLOCK_PEERS` built from a shard-count. `Forest` proved the `RealmShard`
  descriptor + `extra_realm_shards` are the RIGHT seam for that generalization (a `Custom(Vec<RealmShard>)` arm,
  or a `shard_count`-driven builder) — the fully-generic `Vec`-of-N is now a mechanical collapse, not a redesign.
- **Where:** `crates/bins/src/lib.rs` (`SHARD_B`/`GALAXY`/`PLANET_A/STATION_A/AREA_A_SHARD`, `ClusterShape`
  incl. `Forest` + `RealmShard`/`extra_realm_shards`, `ClusterAddrs`, the `*_env` builders incl. `realm_shard_env`,
  `realm_kind_token`/`realm_from_kind_seed`), `crates/devproto/src/lib.rs` (`*_OFFSET`/`SlotPorts`),
  `crates/bins/src/bin/{shard,gateway,orchestrator,vd-devcluster,vd-slot}.rs`, `scripts/crossing-playground.sh`.
- **When / proper:** the **cloud-ready k3d slice (#123)** — the fully-generic `Vec`-of-N. HR3 stays clean:
  adding a shard is a roster/list EXTENSION (a `ClusterShape` value), never a code branch on a shard kind.
- **NOTE (S3, 2026-07-17):** the durable-dot MULTI-HOP `CutTimeout` that was co-ledgered here is RETIRED by S3
  (server-timed cut) — see the C-6c note below, `tests/tests/three_shard_round_trip_durable_e2e.rs`, and now
  the S5b process-tier capstone. No session-route migration was needed.
- **🟩 RESOLVED (2026-07-18, task #149) — every re-home is the ONE uniform orchestrator saga; source==dest is the degenerate case (the local short-circuit is DELETED):**
  **The generic cure landed.** The user rejected the earlier local re-home short-circuit as an edge-case special-path: "Re-home should be the same for any realm based on the coordinates through the same mechanism (the orchestrator) — if you build the generic solution right, all edge cases are covered." So the special `RealmRelabels` relabel + `evaluate_one_subject` `held_here` branch + `local_rehome` + the `CrossingCtx.cohosted` borrow are **DELETED**. A re-home into a co-hosted realm now emits the SAME `CrossingRequest`/`TransientCrossingRequest` as any other; the orchestrator resolves `head(Realm(child))` to the same node (a source==dest transfer) and the ONE saga drives it to completion — post-S3 the cut is server-timed (no `Cutting`→`CutTimeout` stall) so the same-node saga completes cleanly: the gateway self-acks the cut, the directory CAS just bumps the fence, the route swap is an idempotent no-op. The subject's OWNING realm is now derived from its POSE FRAME (`FrameRef::realm()`, `owning_realm` helper) instead of a relabel map. **TWO source==dest hazards are GUARDED** (the second was found by the source==dest e2e, not the original investigation): (1) `register_and_spawn_source_ghost` creating a self-ghost feed loop when `cmd.source == this node` — skip the register + Spawn when `source == self_node`; (2) the ordered `Demote`-before-`Promote` running on the SAME dot: the demote self-fences it to `Ghost{source_fence: CAS-fence}` and a strict-newer `AuthorityCmd::Promote` at the SAME fence would `StaleFence`, so `promote_apply` RE-OWNS the dot at `cmd.new_fence` directly when `cmd.source == self_node` (a legitimate equal-fence re-ownership — the CAS committed THIS node as owner; the route swap completing). **Co-hosting is KEPT** — it is now PURELY a PLACEMENT: the grant/affirm of child-realm heads (`held_realms`/`CoHostedAuthority`/`realm_neighbourhood_for_held`) is what makes `head(Realm(child))` resolve here so the saga can start. This unlocks peeling a crowded Area onto its OWN server with ZERO mechanism change (its head just moves to a different node; the saga is identical). **PARENT-PROVENANCE (`to_parent`) THREADED — the "Area label never flips" fix, retiring the "Area frame owed at P4" degrade note:** `rebind_pose_to_dest(pose, to_realm, to_parent)` now takes the dest realm's parent; the SOURCE detector fills it from the container region's `parent` (a deterministic worldgen fact), and it rides `CrossingRequest`/`TransientCrossingRequest`/`TransientCrossingGrant`/`SagaCtx`/`TransientStatus::Crossing`/`LatchedCrossing` (wire minor bumped to 3) to every `rebind_pose_to_dest` call site — so an `AreaLocal { planet_seed, area_seed }` frame FORMS (previously the parent was hardcoded `None` and Area frames never formed → the HUD label stuck on the parent planet). **[⚠ STALE at HEAD — corrected 2026-08-14 (Stage-C audit :685/:812): `rebind_pose_to_dest` was later DELETED by the placement arc (D-PLACE-1); NO production code reads `to_parent` anywhere on the thread. The Area frame forms at the RECEIVER from its own ROSTER (`arrival_frame` → `RealmRegions::hosted_frame`, which carries the planet parent losslessly), so the datum never needed to cross. `to_parent` is now a ★DEAD carried-but-unread wire field, tombstoned in the field docs and owed a flag-day removal — see D-WIRE-1.]**
  the seed forest nests Planet 7 (SOI r=10 at x=20), Area 7, Station 7 as CHILD regions of System 7 that the
  System-7 shard's containment detector EVALUATES; a shard may CO-HOST them so `head(Realm(child))` resolves.
  A CONTINUOUS `WalkTo` along the straight +X axis THROUGH Planet 7's SOI (x∈[10,30]) re-homes System 7 →
  Planet 7 (the deepest container) via the uniform saga; with the child realm co-hosted the resolve returns
  THIS node (source==dest) and the saga completes, so nothing strands. A shard HOSTS its realm AND deeper CHILD
  realms it CO-HOSTS: `StubConfig.held_realms: BTreeSet<RealmId>` (default `{realm}` → every single-realm shard
  byte-identical); the directory grant/affirm generalizes to a per-held-realm map (`CoHostedAuthority` beside
  `RealmAuthority`). The co-hosted hop is CLIENT-OBSERVABLE (the delivered own-entity FrameRef advances, now
  reaching the Area frame) exactly like a network re-home; the client can't tell them apart. In the `--triple`
  demo the System-7 shard co-hosts `{System 7, Planet 7, Station 7, Area 7}` (`VD_HELD_REALMS`, set by the
  launcher via `held_realms_env`); the Galaxy/System 8 shards stay single-realm.
  **Anti-vacuity tests:** sim-tier `stub::tests::containment::a_dot_re_homing_into_a_cohosted_child_emits_a_crossing_request_with_its_parent, stub::tests::containment::a_dot_re_homing_into_a_non_cohosted_child_emits_a_crossing_request, stub::tests::containment::a_held_transient_re_homing_into_a_cohosted_child_emits_a_transient_request_with_its_parent, stub::tests::containment::a_cohosting_shard_affirms_the_child_realm_head_into_its_own_authority_map, stub::tests::handoff::a_same_node_re_home_registers_no_self_ghost_and_spawns_none, stub::tests::crossing::owning_realm_reads_the_pose_frame_when_nameable, stub::tests::crossing::owning_realm_names_the_galaxy_and_the_universe_too`; the end-to-end source==dest saga (Planet 7 ↔ Area 7 on one co-hosting node, the frame flips to `AreaLocal{7,7}`, the head stays on-node, and a committed re-home does not re-fire) is `tests/tests/crossing_same_node_e2e.rs`.
  **STILL DEFERRED (P4+):** dedicated hosting SHARDS for planets/stations/areas (co-hosting keeps them on the
  parent-system shard; a busy planet split to its OWN shard is the P4+ spatial-index/roster item, D-44/D-9) —
  but now a ZERO-mechanism-change move (co-hosting is pure placement; the saga is already uniform).
- **Source:** dual-shard cluster design + adversary `scripts/dual_shard_cluster_design_adversary.md` (M-2);
  S5b is the acceptance capstone of the generic-coordinate-rehome arc.

---

### D-47 🟥 Durable state survives a process crash but NOT the loss of its HOST — one local redb file, no replication, no backup, and a volume-less rescheduled pod is indistinguishable from a legitimate first boot
- **What IS proven (so this entry is not mistaken for the crash gap, which is closed):** `RedbStore` commits one
  redb `WriteTransaction` at redb's **default** durability — verified in the dependency source, `redb-2.6.3`
  `src/transactions.rs:835` sets `InternalDurability::Immediate` ("guaranteed to be persistent as soon as
  `commit` returns"), and we never lower it. The off-tick writer bumps `last_durable` ONLY after that fsync
  (Release), the bin gates effect egress on `is_durable_through(seq)`, and depth-1 commit-blocks-on-prior bounds
  crash-loss to ≤1 batch — proven by a real `kill -9` parked in the submitted-but-pre-fsync window
  (`crates/bins/tests/orchestrator_crash.rs`, sentinel-key writer park + a writer-written marker file), with an
  anti-theater FRESH-store control. D-6 covers this and is silent on everything below.
- **What's missing:** any durability beyond that ONE disk. `grep -i 'replica|replication|raft|paxos|quorum'`
  over `crates/` returns no implementation. Three consequences:
  1. **Permanent host/disk loss is unrecoverable.** Gone with it: every in-flight saga WAL, the durable
     **directory head — THE commit point** — and the clock ceiling. The saga WAL tolerates re-drive; a lost
     directory head cannot be reconstructed from anything else in the system.
  2. **★ A volume-less pod looks exactly like a first boot.** `RedbStore::open` already refuses to treat a
     NON-empty store as genesis (the `wf_66cb8f06` cure), but an ABSENT store legitimately IS genesis on first
     init — so the dangerous case is indistinguishable from the safe one *by construction*. Under k3d/k8s a pod
     rescheduled onto another node without its volume boots clean, and our own anti-theater control
     (`p3_orchestrator_kill_9_without_a_durable_store_recovers_nothing`) states precisely what that means:
     nothing recovers, silently, and the rebuilt orchestrator treats the reset clock as legitimate.
  3. **No backup / point-in-time restore**, so operator error or a corrupt file has no recovery path at all.
- **Changes to implement, in order:**
  1. **The deploy precondition (minimum, before ANY multi-node deploy).** The store must live on a durable
     volume whose identity is PINNED to the orchestrator identity, and boot must refuse to run as genesis
     unless explicitly authorized: (a) a k8s StatefulSet with a PersistentVolumeClaim — never `emptyDir`,
     never `hostPath`, never `/tmp` — one PVC per orchestrator identity; (b) a **store-identity stamp**
     (cluster id + orchestrator id + universe epoch + the D-48 store format version) written at init and
     verified on every open, mismatch = LOUD REFUSAL to boot; (c) an explicit opt-in
     (`VD_STORE_ALLOW_GENESIS`, or the existing durable-root allow-list) REQUIRED for a legitimately empty
     first boot, so "no volume attached" can never masquerade as one. This converts silent data loss into a
     refusal — the house rule: **a refusal is never a loss.**
  2. **Backup + restore:** a redb snapshot/copy hook, a restore path, and a restore test. Evaluate redb's
     two-phase-commit ("Paranoid") durability and savepoints here — a knob we currently do not use.
  3. **Replication (the proper solution).** Only the **directory family** strictly needs it (see consequence 1).
     The `Store` seam already supports separate families/files (PLAN.md:126), so the directory family can move
     to a replicated backend behind the SAME frozen seam with **no change in sim/node**. Backends to
     investigate JOINTLY (investigate-libraries-together — do not adopt unilaterally). Reference read: Rivet's
     Epoxy per-key Paxos, whose immutable-value design buys local reads —
     `scripts/rivet_codebase_analysis_20260811.md` §4 and §7.2, **including the finding that their own scheme
     cannot RELOCATE a key yet** (their "reservation chains" future work), which is why it is a reference and
     not a drop-in for a directory whose whole job is relocation.
- **Where:** `crates/io-prod/src/store.rs` (`RedbStore::open`, `apply_batch`, `DurabilityHandle`);
  `crates/bins/src/bin/orchestrator.rs` (the `VD_STORE_PATH` wiring + the boot precondition warn); the
  k3d/deploy packaging (task #123).
- **When:** change 1 is a **HARD precondition of the first real multi-node deploy** (task #123 / D-44), sitting
  alongside the existing at-most-once-transport precondition in D-6 — the boot warn should name both until they
  land. Change 2 with the same slice. Change 3 on the same deploy-readiness signal and no earlier: at
  in-process / single-host scale it is genuinely inert.
- **Source:** the durability audit of 2026-08-11 (all facts measured from the tree this session), prompted by
  the Rivet comparison in `scripts/rivet_codebase_analysis_20260811.md` §5. **Not previously ledgered** — D-6
  proves crash durability and says nothing about host loss, so this was a silent gap in the registry itself.

---

## PERF / SCALE (negligible now; land with the slice that makes them matter)

### D-41 🟧 "Hundreds of users in ONE location" — DIRECTION RESOLVED (adaptive-hierarchical-horizontal + tiered-i64 coordinate base); plant-now seams owed before P4/P5/P6 harden the single-anchor/single-writer/single-cluster assumptions
- **RESOLVED (user-confirmed, Jun 2026):** the direction is **adaptive-per-realm HIERARCHICAL HORIZONTAL** partitioning
  (universe→quadrant→galaxy→system→planet/city→sub-region; the split factor is per-location `ShardProfile` capability
  DATA — small planet=1 shard, dense city-hub=N, big planet ~6-8; ships are mobile nested frames traversing it via the
  SAME transfer/saga/ghost kernel). The crux is the **coordinate base, not the topology** (the topology seams are
  verified additive). Decided base: a **TIERED INTEGER lattice** — a position is an `i64` CELL anchor + a bounded f64
  local OFFSET (floating-origin model); the FINE tier (star system + everything inside: planets, ships, cities,
  sub-regions) uses `i64`@**mm** (spans ~1 ly radius, mm-exact, sub-micron locally), the COARSE tier
  (galaxy/quadrant/universe = the nav/warp/star MAP, never physics) uses `i64`@**ly or AU**; the integer TYPE is uniform
  `i64`, the UNIT is keyed by the `FrameRef` tier. Cross-frame/cross-shard re-base is EXACT INTEGER arithmetic
  (intra-tier = subtract cell anchors; inter-tier = exact mm↔ly conversion at the DISCRETE SOI/warp transfer) — zero
  drift, bit-deterministic (serves the byte-identical-replay gate), so "an error of several cm never occurs" holds BY
  CONSTRUCTION where things interact (within a system). The enabling fact: the **star system is the largest
  *interaction* domain** — nothing collides between stars, inter-system travel is warp (a discrete transfer) — so
  cm-exactness is bounded to a system, exactly where the fine lattice lives. Substrate = `glam::I64Vec3` (already a
  dep); HAND-ROLL `LatticePos` in vd-core (`big_space` is a math/recenter REFERENCE only, NOT a dep — f32 + Bevy-coupled
  + no determinism). Designs: brief `wf_c4157f73`, impl-plan `wf_73c0d67f`.
- **PLANT-IN-THE-BASE-NOW (before P4/P5/P6 — behaviour-identical by default: every world at cell 0, offset == today's f64 pos):**
  (1) `StampedPose.pos` → an `i64`-cell + bounded-f64-offset lattice (THE single most important plant — it rides the
  FROZEN wire that P4 physics / P5 rapier / the client render path / every transfer+ghost+snapshot harden against;
  deferring = the ~14-file wire rewrite this entry exists to prevent); (2) exact-integer re-base in `transfer_frame` +
  `origin_cell` on the non-wire `FramePlacement`; (3) ONE config home (per-tier unit + cell_edge as a power-of-two number
  of mm) + a margin assertion + a drift-free proptest (normalize round-trips; re-base A→B→A returns the exact cell;
  inter-tier conversion exact+reversible); (4) the shared-anchor-per-location CONTRACT + a >1-anchor-capable `FrameSpace`
  shape (default 1; soften the single-cluster HARD ERROR to per-region) — the mechanism for bit-exact cross-region
  no-prediction collision; (5) a degenerate `RegionId{WHOLE}` on the realm/directory key (single-key `commit_cas`
  unchanged) + a non-radial `OverlapBand` constructor (degenerate-unused); close D-39.4 (Station/City realm kinds). Plus
  [[D-9]] per-cell AoI (owed REGARDLESS — same grid primitive as the region split). HR5: the cell↔offset
  split/normalize are MONOMORPHIC branchless helpers; DETERMINISM: the integer is authoritative/replayed, the f64 offset
  is derived/tolerance-only (never a float op in the authoritative path).
- **✅ LANDED (plant-now item 1 — the frozen-wire SHAPE; impl-plan `wf_73c0d67f`):** `StampedPose.pos` is now
  `LatticePos { cell: I64Vec3 (private), offset: DVec3 (private) }` (`core/src/pose.rs`), constructed at the cell
  origin via `LatticePos::local`, moved-in-frame via the cell-PRESERVING `LatticePos::map_offset`, and read via
  `.offset()` — behaviour-IDENTICAL through P3 (cell is always ZERO, offset == the old f64 `pos`), verified by the
  full suite + a non-zero-cell postcard round-trip proving the integer cell rides the frozen wire bit-exact. The
  ~20-site migration (core/sim/client/harness/wire/tests) landed as ONE atomic build-green slice (Tier-A 100%).
  **Smallest-correct scoping:** only the WIRE SHAPE landed (the before-P4-critical part, cheapest now before features
  multiply the sites); the lattice MATH — `normalize`/cell-crossing + the per-`FrameRef` tier unit (FINE i64@mm /
  COARSE i64@AU) + `cell_edge` config + the exact inter-tier (mm↔AU) conversion + per-tick re-centering (the
  bounded-offset invariant) + cross-cell interp rebasing — is DEFERRED to its first consumer (**P4/P5 re-centering**
  is the first producer of a non-zero cell; galaxy ly-cells at P10). **"PURE-ADDITIVE then" — precisely (audit
  `wf_fd6a4b9d` HIGH):** the WIRE is additive (no shape change); the cell-PRESERVING move sites (`stub.rs` integrator
  via `map_offset`, `pose.rs` `advanced_ballistic`/`sanitized`) are additive at P4/P5 (re-centering is a `.normalize()`
  ADD, the cell already carried) — the earlier integrator used `local` (which ZEROES the cell, a latent teleport
  foot-gun + a false "preserved" comment); that was fixed in THIS slice by adding `map_offset` + switching the
  integrator to it. The cell-WRITE sites (`transfer_frame`'s `LatticePos::local(new_pos)` — plant-item 2; the
  `world_pos`/`RenderPose` which drops the cell at `interp::sample`) are NOT yet cell-aware: each REPLACES its
  provisional cell-0 write with destination-cell math at P4/P5+ — additive (no landed decision undone) but a
  body-REWRITE at the site, not a free delta. **Distinct cell-READ sites that gain cell-aware re-basing ADDITIVELY
  (no provisional cell-0 write to replace — audit `wf_9f26b8cb` precision):** `GhostNeighbor.anchor` (`stub.rs`, a
  bare `DVec3` captured at promote/re-home) + `ghost_band_exited`'s `(pose.pos.offset() - anchor).length()` distance
  — these only READ `.offset()`, so the P4/P5 re-centering slice converts the anchor to a `LatticePos` + the
  band-exit to a cell-aware difference at the SAME time it converts the interp + `transfer_frame` (named here so the
  re-centering implementer does not miss this second offset-only site, the cross-region-collision band geometry).
  Items (2)-(5) remain owed.
- **✅ THE CELL MATH WENT LIVE (real-scale cell activation, 2026-08-18/19 — supersedes the stale
  "cell is always ZERO" notes above):** the ONE subtraction is `vd_core::pose::Separation`; every
  producer normalizes (`from_metres` public, `local` crate-private, `map_offset`→`translated`,
  `at_rest` normalized, generator centres + placements + Motion arms normalized); `transfer_frame`
  is integer-first with normalized output; the containment verdict is INTEGER (`region_verdict`);
  the interp/`RenderPose`/`GhostNeighbor` items previously listed as owed were verified already
  cell-correct. STILL DORMANT: `Tier::Coarse` + `convert_tier`'s cross-tier arm (P10; refused —
  `guard_root_representable` is the named trigger).
- **DEFER (additive once the shapes exist):** galaxy/quadrant/universe `FrameRef` levels + ly-cells (P10); the
  multi-anchor `FrameSpace` MACHINERY (P4/P5); N>1 live sim + the density rebalancer + region-store redb sharding + the
  cross-region GhostFlow runtime consumer (gated on the single-orchestrator soak below + a benched per-shard
  colliding-player budget); the cross-region collision-RESPONSE solver + its seam-straddling single-authority tie-break
  RULE (P5; NAMED now: exactly ONE region-shard resolves any straddling contact — rapier is never re-simulated
  cross-host); the Signal arm (P9, rides the planted band). The VERTICAL path (parallel rapier islands) stays the
  per-shard compute fallback, SPIKE'd before P5 if a benched system hits the single-rapier ceiling. **A STANDING
  cross-region overlap needs an `AuthorityChanged` producer for the OWNED sub (audit `wf_9f26b8cb`):** one entity
  owned by region-shard A also appears as a ghost in region-shard B's snapshot, so a client subscribed to BOTH subs
  receives it twice; `DeliveredView::chosen_subs` (`client/src/view.rs`) de-dups via `AuthorityChanged`-else-lowest-sub
  — the right mechanism — but `AuthorityChanged{entity,sub}` is produced TODAY only by the transfer route-swap
  (`gateway` `apply_commit`/`store_commit`), so a standing (no-transfer-in-flight) overlap falls back to "lowest sub"
  (not authority-correct — the ghost-sub could win, rendering a stale kinematic ghost over the owned pose). Owed: a
  steady-state `AuthorityChanged` producer for the owner-sub + the "owner beats ghost" rule; ADDITIVE (the view-side
  consumer already generalizes to N subs), lands with the region-split client work — load-bearing for multi-mesh-render.
- **The gap (the original problem this resolves):** every spatial partition in the base is a RADIAL SHELL around a body centre — `RealmId` is one-owner-per-body
  (`core/src/pose.rs` Planet/System/Ship), `OverlapBand` edges derive from an SOI radius (`core/src/geometry.rs`
  `for_planet_soi`/`for_system_soi`/`for_motion`; `segment_shell_crossing` tests `|p|<=r`), `DirectoryKey::Realm` is
  whole-realm with a single-key `commit_cas` (`sim/src/directory.rs`), a shard holds ONE `config.realm`, and a second
  cluster on one planet is a binding HARD ERROR (`sealed_shards.md:236`, `PLAN.md:42`). So splitting ONE crowd across
  shards — with cross-boundary collision/ghosts the way the SOI machinery does it BETWEEN bodies — has no representation:
  no non-radial interior boundary, no second `SurfaceAnchor`, no sub-realm/region directory key. The design's stated load
  target everywhere is "dozens" (`PLAN.md:181`, `transfer_protocol.md:283`, `identity_persistence.md:113`); the user's
  "hundreds in one location" target had NO ledger entry until now.
- **WHY this is a decision, NOT a crisis (nothing landed is wrong):** the load-bearing pieces are UNBUILT — `FrameSpace`
  lands P4/P5, the single-cluster assertion lands P6, per-realm single-writer redb is not wired into a real planet shard
  yet. The transfer/ghost/saga KERNEL already moves individual entities cross-shard with ghosts and is geometry-AGNOSTIC;
  per-`Entity(EntityId)` authority is already SEPARABLE from per-`Realm` ownership; `OverlapBand` already carries 3
  constructors and grows boundary shapes additively. So the missing piece is an ADDITIVE future subsystem the base does
  not foreclose — but it WILL be foreclosed if P4/P5/P6 harden single-anchor/single-writer/single-cluster before the
  decision is made.
- **DISTINCT from [[D-32]]:** D-32 partitions the directory keyspace BETWEEN realms across N orchestrators and explicitly
  keeps "no cross-key transaction ever spans regions" (`transfer_protocol.md:283`) — that is INTER-realm. D-41 is
  INTRA-realm: splitting ONE realm's live sim + cross-boundary collision across shards. The `(RealmId, region-range)`
  intra-realm sharding exists only as design prose (`sealed_shards.md:236,379`) with no pin and no seam shape — this entry
  closes that honesty hole.
- **The options weighed (HORIZONTAL chosen — see RESOLVED above; VERTICAL kept as the per-shard compute fallback):**
  - **(a) Vertical (one beefy shard):** parallelize `step_tick` + the rapier solver (rapier ISLANDS / sub-stepping)
    WHILE preserving the byte-identical-replay determinism gate (single-threaded today guarantees it — a parallel solve
    needs a deterministic merge) and the integer-quantized physics→control boundary; SPIKE this BEFORE P5 commits to one
    single-threaded rapier context (`node/src/app.rs:93,121` — the per-tick compute ceiling for hundreds of colliding
    capsules). Set a benched per-shard colliding-player budget so "hundreds" has a defined shard count.
  - **(b) Horizontal (intra-realm region shards):** a multi-anchor `FrameSpace`, a `(RealmId, region-range)` directory
    key, and a NON-radial inter-region band type distinct from SOI. If chosen, plant the degenerate-now seam shapes (a
    region-range component on `DirectoryKey` that is whole-realm today; a `FrameSpace` that admits >1 anchor defaulting to
    one; an inter-region band type) so the future split is populate-the-field, not a band/RealmId/directory rewrite — the
    [[D-32]]/[[D-34]] `coordinator_of`/`home_shard` discipline.
- **Synchronized-crowd CROSSING (the dynamic side — fleet/raid/evac/PvP boundary churn):** a Durable crowd moving across a
  boundary TOGETHER produces one saga + one single-key CAS PER player (only Transient kinds have the batched
  `TransientGo` token — `sealed_shards.md:255`). **NOTE (audit `wf_2c963246` precision):** the group-commit fsync is
  ALREADY amortized to ~1/tick by the barrier (`saga_runtime.rs` drains a whole tick's directory mutations + saga
  snapshots into ONE `Store::commit()`), so this is NOT fsync amplification — the real surge cost is O(N) per-player
  saga state-machines + single-key CASes plus the O(live-sagas)/tick `scan_deadlines` walk ([[D-3]] already flags the
  deadline-index fix). Decide, when load-testing the transfer path with a SYNCHRONIZED N-player crossing, whether the
  orchestrator can DRAIN O(N) per-player sagas/CASes within the tick budget during a surge (a saga-throughput /
  commit-pipeline-depth question — the same one the single-orchestrator soak below covers), NOT a per-player-fsync one.
- **Single-orchestrator interim soak (owed NOW, before the [[D-32]]/[[D-3]] N-orchestrator era):** add a saga-throughput
  soak (N concurrent durable sagas on distinct keys + a transient burst) asserting the orchestrator drains within tick
  budget and `scan_deadlines`(O(live-sagas)/tick) + the reaper(O(directory)) stay bounded — index them by a deadline-ordered
  structure ([[D-3]] already flags this). Proves the single-orchestrator interim holds through P3–P10 and turns "partition
  later" into a measured-headroom decision.
- **Pin (exists-to-be-flipped):** direction is now RESOLVED (above); flips 🟩 when plant-now items (1)-(5) land as
  reviewed shapes (degenerate-to-today, proptested) BEFORE P4/P5/P6 harden the single-anchor/single-writer/single-cluster code.
- **⚠️ MECHANICAL-GUARD GAP (audit `wf_9f26b8cb`, owed before P4/P5/P6) — PARTIALLY CLOSED (Jul 2026):** the guard-gap
  was that plant-items (2)-(5) are pinned ONLY in prose doc-comments, so the "a phase isn't done until its DEFERRED
  entries flip" gate cannot mechanically catch a P4/P5/P6 slice that hardens single-anchor/single-cluster WITHOUT first
  softening it to per-region.
  - **✅ LANDED — the `action_bits`-inert tripwire ([[D-39]].1):** `crates/sim/src/stub.rs` test
    `action_bits_are_inert_two_datagrams_differing_only_in_action_bits_integrate_identically` asserts two
    `InputDatagram`s differing ONLY in `action_bits` (0 vs `u32::MAX`) integrate to the IDENTICAL authoritative pose
    today (the integrator reads only movement+look). It flips RED the day the reliable client→shard discrete-action arm
    makes the sim consume `action_bits`, so a regression can never silently gate PvP fire-registration (or P6
    block-edit-forward) onto the lossy UNRELIABLE input datagram. A live green gate, no longer prose-only.
  - **🟥 STILL OWED — the region/anchor tripwires (>1 anchor representable / region-range `DirectoryKey` / inter-region
    non-radial band), OWED WITH the P4/P5 re-centering work:** these are NOT landable now because their SUBJECT TYPE is
    absent in code as of HEAD ab8fae8 — there is no `FrameSpace`/`SurfaceAnchor` struct (only doc-comments), `DirectoryKey`
    (`wire/seams/directory.rs`) has no `RegionId`/region-range arm (`RegionId` has zero code defs), and `OverlapBand`
    (`core/geometry.rs`) is a scalar radial-shell with only radial constructors (no band-shape discriminant). Writing a
    tripwire against a non-existent field is empty-set theater that false-greens and would ITSELF corner the P4/P5 seam.
    These land WITH plant-items (2)-(5) (the degenerate-now `RegionId{WHOLE}` / multi-anchor `FrameSpace` / non-radial band
    shapes) during the P4/P5 re-centering — the tripwire and the seam it guards are the SAME slice. (The LatticePos
    multi-cell round-trip — plant-item (1) — is already covered by `pose.rs` tests, so no coordinate tripwire is owed there.)
- **Source:** whole-codebase audit `wf_032b80eb` (PvP + large-scale: H2 partition; the rapier/step_tick ceiling; the
  synchronized-crossing batch gap; the single-orchestrator interim soak) + design brief `wf_c4157f73` + impl-plan `wf_73c0d67f`.

### D-42 🟥 Server-side LAG-COMPENSATION (pose-history rewind for hit-registration) has NO home — the one PvP pillar absent from code AND every design doc (audit `wf_9f26b8cb`)
- **The gap:** the binding standards mandate NO client prediction + players physically COLLIDE + a 100-150 ms interpolation
  buffer (`PLAN.md:32`). With those three together, what a shooter sees of a victim is 100-150 ms (2-3 ticks) in the PAST,
  so fair server-authoritative hitscan/projectile hit-reg REQUIRES the authoritative shard to REWIND the victim's collision
  body to where it was at the shooter's render instant ("favor the shooter" — standard for every no-prediction authoritative
  shooter). The code keeps ONLY the current pose per entity (`Dot.pose: StampedPose`, `Dots: BTreeMap<SessionId, Dot>` —
  `sim/src/stub.rs`); no shard buffers past authoritative poses, and there is NO design-doc or DEFERRED treatment (the only
  "lag compensation" string in the tree is in `docs/audit/rust_netcode.json` describing a REJECTED library's feature). Every
  OTHER PvP pillar (D-39.1 reliable fire carrier, D-39.6 ghost combat-state blob, the applied_damage TLV) IS ledgered with a
  green-gate pin; lag-comp is the one invisible to the "a phase isn't done until its DEFERRED entries flip" gate.
- **WHY ADDITIVE, not rework:** no landed code contradicts it — `Dot` simply GAINS a bounded per-entity pose-history ring +
  a rewind-on-fire path keyed by the fire event's source/client tick. The wire ALREADY carries the timestamps a rewind keys
  on (`SnapshotDatagram.source_tick/universe_tick`; `GhostFlow::Delta.source_tick/seq`). So nothing is undone — it is a new
  consumer + a ring. MEDIUM (not CRITICAL): nothing is broken at 943a51e; the risk is a LATE, cross-shard-entangled retrofit
  at P11 with no reserved shape.
- **The cross-shard wrinkle (why it is non-trivial-additive — pin the shape NOW so P11 is not painted into a corner):** when
  shooter and victim are on DIFFERENT shards the victim is a GHOST on the shooter's shard (lossy `GhostFlow::Delta` @20Hz),
  so the rewind must reconcile the shooter shard's local ghost history vs the victim OWNER shard's authoritative history. The
  authoritative rewind happens at the VICTIM'S OWNER shard (the [[D-39]].1 hit-forward target) using ITS history; the shooter
  shard's ghost history validates only the shooter's claimed aim. So [[D-39]].6 (the ghost combat-state blob) + .1 (the fire
  carrier) must stay shaped to FEED a rewind.
- **When / proper:** **P11 combat.** Dependency: [[D-39]].1 reliable fire carrier + [[D-39]].6 ghost combat blob. A bounded
  per-entity authoritative-pose ring sized to `interp_buffer_ms + max ghost-feed staleness` (a `TransportTuning`/`TransferTuning`
  config field — NO magic number). Pinning it now is ZERO code; it keeps the GhostFlow/fire-carrier shapes honest about feeding
  a rewind. Flips 🟩 when the ring + the rewind-on-fire path + the cross-shard owner-rewind rule land at P11.
- **Source:** holistic audit `wf_9f26b8cb` (PvP lens — the only genuinely-unledgered PvP subsystem).

### D-RLM-2 🟧 The durable RLM demand-ledger snapshot (`StoreKey::Rlm`, next free tag) — DEFERRED with machine-checked evidence (RLM Step 4b; spec `scripts/rlm_step4_persistence_spec.md`)
The orchestrator's RAM `DemandLedger` (`crates/sim/src/rlm.rs`) is NOT persisted: on a kill-9 rebuild it comes up EMPTY and self-heals from the re-asserted (`ReDriven`, every tick) demands within ~1 demand round-trip. The RLM Step-4a crash-recovery FREEZE (`RlmTuning::recovery_grace_ticks`, armed in `saga_runtime::rehydrate` from the RAW recovered ceiling, applied via `RlmReconcilerRes::arm_quiesce` in `register_orchestrator_with_store`) blocks teardown for one full demand-TTL past the recovered ceiling, bridging the window until demands re-accrue. A durable snapshot (a `StoreKey::Rlm(RealmPath)` record on the next free tag — tags 6/7 are now taken by RLM Step 5b's `RlmLaunch`/`RlmWater` launch-ledger families — + `LedgerCell` serde + dirty-cell staging into the D-6 group-commit barrier; full stage/restore/`draining_since=None` design in the spec §3) would save only ≈1 tick of re-spin latency and CANNOT replace the freeze (it can lose ≤1 off-tick batch). It is DEFERRED, not owed.
- **Machine-checked deferral evidence:** `rlm.rs` `mod crash_replay_proptest` — the 1024-case fixed-seed `rlm_reconcile_survives_arbitrary_crash_interleavings_deterministically` (+ `just rlm-soak`, ≈30k ops, release-gated, in `just gate`) runs the shipped RAM-heal arm AND a durable-snapshot dry-run arm over the SAME crash/reorder/dup op histories and asserts INV-SNAPSHOT-EQUIVALENCE (identical settled reap-sets, valid because `recovery_grace == demand_ttl` ⇒ a snapshot's stale demand ages out exactly when the freeze lifts). It PASSES with NO divergence ⇒ the snapshot changes no teardown decision the armed freeze + arm-B don't already achieve. The proptest ALSO caught + drove the fix of a real Step-3 determinism bug (the streak-based `empty_confirmed` was order-sensitive for a same-tick parent-`KeepAlive` + child-`Empty`; now keyed on `last_demand < last_empty`, both order-invariant maxes).
- **Pin (exists-to-be-flipped):** the `DemandLedger` doc (`rlm.rs`) names the RAM-only self-heal + this entry; `StoreKey` reserves NO tag 6 yet (HR5: no inert uncovered reserved code). Flips 🟩 (lands the spec §3 snapshot at 100%) ONLY IF the divergence oracle or a soak surfaces a restart-window strand the freeze + arm-B do not cover (D-RLM-2's binding trigger). Until then it stays reserved-prose-only.

### D-RLM-4 🟥 Warm-spare-pool for realm spin-up latency — DEFERRED to RLM Step 7 (warp), seam kept open (user: "defer, but don't forget"; RLM Step-5 vet OQ-2, spec `scripts/rlm_step5_real_spawner_spec.md` §5)
RLM Step 5 hides cold spin-up latency GENERICALLY via the predictive-AoI (F7) term — the spin-up radius widens by `boot_time × occupant_velocity` so a demand-launched `vd-shard` finishes booting before the occupant reaches the realm's SOI. A warm-spare-pool (a handful of pre-booted idle shards, so a demand ASSIGNS a waiting shard instead of cold-forking) is a PURE OPTIMISATION on top — it removes the fork+runtime-init slice of the latency but is NOT a correctness need, and its right size/policy is only measurable against a real warp fly-by (Step 7).
- **WHAT is deferred:** the pool + the assign-a-coord-to-a-blank boot handshake.
- **WHERE it lands:** entirely behind the FROZEN `RealmSpawner` contract, as a strategy inside the `LaunchBackend` seam (`rlm_step5_real_spawner_spec.md` §1.1) — `spawn_realm(coord)→NodeId` is unchanged; the reconciler/wire/upstream never know. When built it MUST be the GENERIC form: a pool of **profile-agnostic BLANK shards**, NO mode branch, NO per-realm-kind pool (else it is an automatic REWORK per the no-corners mandate).
- **WHEN:** RLM Step 7 (warp E2E), once warp latency is measured against a physical fly-by and the pool's value + tuning can be proven. Deferring costs nothing — it drops in as a backend change with zero upstream rework.
- **Pin (exists-to-be-flipped):** the Step-5 spec §5 + §1.1 `LaunchBackend` seam doc name this entry. Flips 🟩 when the blank-shard pool + assign handshake land at Step 7.

### D-RLM-5 🟥 The fork-crash orphan window (v1-committed, forked, crash before the pid v2-commit) — DEFERRED to RLM Step 5e (spec `scripts/rlm_step5_real_spawner_spec.md` §8 5e)
`SpawnCore::spawn_realm` (RLM 5c-1) is write-ahead: it commits a v1 launch-intent (`pid: None` + the pre-fork cookie) BEFORE the fork, then a v2 intent (`pid: Some`) AFTER the launch confirms. `rehydrate` DROPS a v1-only (`pid: None`) intent — reconstructing an unconfirmed launch risks a double-spawn. The narrow window where the child WAS forked but the orchestrator crashed before the v2 commit therefore leaves a running `vd-shard` whose intent rehydrate deleted → an orphan the rebuilt orchestrator has no ledger row for.
- **WHAT is deferred:** reaping that orphan. It is bounded (≤1 in-flight spawn) and never a double-spawn (F2 mints a fresh id next time); the cookie was persisted pre-fork so the orphan is IDENTIFIABLE.
- **WHERE it lands:** RLM Step 5e's cookie-guarded orphan sweep — the same `/whoami` cookie-probe + `pgid`/live-pod scan that reaps a crashed child reaps this one (it is just an orphan with no live ledger row). NOT reachable to test until 5e builds the orphan-adopt path + a real kill-9-mid-spawn process gate.
- **PROGRESS (5e-4 committed d7c34d5):** the orphan-ADOPT path this entry's eventual reaper reuses now EXISTS — `LaunchBackend::adopt` + `ProcLaunchBackend`'s `Slot::Orphan` + the `/whoami` cookie-probe (`is_alive`) + the signal-only `teardown`, with `SpawnCore::rehydrate` re-adopting each `pid:Some` survivor. Proven in the process gate's ADOPT+cookie-probe leg.
- **SCOPE (5e-5 committed):** the 5e-5 kill-9 gate proves no-orphan / no-double-spawn / no-id-reuse-of-a-live-child for the **PRE-FORK mid-fsync window ONLY** — `flush()` parks strictly before `backend.launch()`, so a SIGKILL there forks NO child and NO orphan is produced (witnessed directly by the child probe never answering `/whoami`). It deliberately does NOT exercise THIS entry's window (child forked, crash before the v2 commit): that requires new hook machinery a content-keyed pause cannot provide (v1 and v2 share the tag-6 `RlmLaunch` key), AND an orphan-head sweep that exists NOWHERE today (verified: `reconcile` iterates only the ledger/closed set; `reconcile_launches` iterates `minted`; `live_nodes` probes only `live` — the dropped `pid:None` orphan enters none).
- **WHERE it lands (re-pointed to 5f):** the orphan-head sweep — a per-tick directory-head scan cross-checked against the path-keyed launch ledger, with a `/whoami` cookie-probe ForceReap, INCREMENTAL/bounded for 100K — needs the ARMED reconciler (the bin hardcodes `RlmTuning::default()` = inert) and must be co-designed with 5f's cadence + probe budget. Interim safety: the orphan is bounded to ≤1 in-flight spawn (`spawn_realm` serializes under the mutex), never a double-spawn (F2 mints a fresh id), and is identifiable (the cookie is persisted pre-fork).
- **Pin:** the `rehydrate` "drop partial" comment (`crates/node/src/rlm_spawn.rs`) names this entry. Flips 🟩 when 5f's orphan-head sweep lands + reaps it.

### D-RLM-6 🟥 A running descendant's peer view does not refresh when an ancestor churns — DEFERRED to RLM Step 5e/5f (RLM Step-5d vet `wf_9e6e54c7` D1; spec §8 5d amendment)
RLM 5d's `VD_PEERS` ancestor closure (`closure_peers`, `crates/node/src/rlm_spawn.rs`) is computed at the child's SPAWN instant and baked into the child's boot `cfg.peers` → dial lanes (the sole lane-create site is the boot loop; `update_peer_addr` mutates only the addr book, creating no lane). So a child's reachability to its ancestor chain is FIXED at boot: an ancestor ABSENT at spawn, or one that RESTARTS under a new incarnation (fresh monotone `NodeId`, F2) while the descendant stays live, is NOT reachable by that already-running descendant — reply-on-connection cannot repair it (it needs the descendant to be the DIALER, and the descendant has no addr/lane for the new incarnation). This is the node-per-realm frozen-player class ([[project_node_per_realm_rehome]]) if left unhandled once the reconciler is armed.
- **CORRECT ONLY (5d's standing precondition):** under parent-first spawn ordering within a sweep (the reconciler's `BTreeMap` path-prefix ordering spins ancestors before descendants) AND no ancestor churn under a live descendant. 5d is inert (reconciler unarmed), so this never fires in 5d.
- **RESOLVED (user decision 2026-07-25, RLM 5e vet `wf_71fd3e40` D-RLM-6 judge panel A=47/B=80/C=80): mechanism = C, LAZY RESOLVE-ON-MISS.** A live descendant keeps its boot-time peer book UNTIL a miss (an upward send fails / a downward parent-authored frame goes stale) proves the lane dead; only THEN does it re-resolve the ancestor's CURRENT `(NodeId, addr)` from an authority and re-plumb its own dial lane. PULL, not push: cost is paid only by descendants actually crossing a broken hop during the rare outage; dormant leaves + deep chains pay nothing; the orchestrator holds ZERO new per-ancestor subscriber state (the anti-cornering win at 100K). Chosen over B (eager push — needs a reverse `ancestor→{descendants}` index to hold/shard/crash-recover + an O(subtree) push storm on a high-ancestor restart) and A (re-spawn subtree — DISQUALIFIED: bouncing live realms is loading/teleport-class disruption, violates the seamless hard rule). Honest correction from the vet: C's "zero new wire" holds ONLY in k8s (per-realm Service DNS re-lookup); at the process/dev tier (no DNS) BOTH B and C need ONE new address-bearing arm (a `PeerLocate{node}→PeerLocated{node,addr}`, NOT a repurpose of `RealmDemand`/`ReSolicitBatch`) — so the decision rests on scale/robustness/anti-cornering (where C wins), not wire count.
- **WHAT 5e does (pick + ledger ONLY):** records C here; builds the substrate C leans on (adopt/cookie/`/whoami`/F2-fence — the fence gates the head flip so a resolve can never plumb a RETIRED incarnation). **5f implements** the miss-trigger (`NodeUnreachable`/stale-watermark), the re-resolve + tick-jittered backoff, and the `PeerLocate` addr arm — with a **HARD gate**: ancestor crash+respawn while a descendant stays live → the descendant re-reaches the new ancestor within bounded ticks. **The addr-delivery arm MUST be validated in the process-tier gate (no DNS there) — the single point that cannot be hand-waved.** Related **D4** (first-contact `Unreliable` Input dropped before the freshly-spawned dest is learned by the gateway) is also a 5f HARD gate (real re-home onto a fresh dest → zero input-drop across the commit; gate/buffer at the gateway or eager dest→gateway dial at boot).
- **★ BUILT 2026-09-03 (mechanism C, the ruler switch's slice 7; owner: *"implement all"*):** the two
  address arms `InterShardFlow::PeerLocate` / `PeerLocated` (discs 41/42, minor 27, Membership class);
  the MISS as the trigger (`SendError::UnknownPeer` — the mesh's answer for a peer in neither the book nor
  the learned set; the node runtime keeps the frame and asks the node that drives its clock, once per tick
  while a frame waits, `PeerBook` counters); the answer is consumed below the schedule and booked through
  the new seam method `Transport::book_peer` (a no-op on every in-memory fabric); the mesh dials a
  booked-but-laneless peer lazily on the next send (`MeshStats::lazy_dials`, `peers_booked`); the
  orchestrator answers from `RealmSpawner::addr_of` (the launcher's live slot, IPv4 as mapped octets) and
  counts a stranger. Nothing in the sim changed. Unit-level gates: mesh on localhost, node, orchestrator,
  spawner, follower. **The ancestor-churn re-reach gate and the process-tier proof are still owed** — the
  first process-tier use is the ruler switch's inward leg (a hull entering a sibling star system), i.e. the
  flight. The re-resolve-on-stale-watermark trigger (a DOWNWARD miss) is not built: only an UPWARD send
  misses today, and the new parent's first downward statement misses from its own side the same way.
- **Pin:** `closure_peers`' doc-comment LIMITATION names this entry. Flips 🟩 when 5f's ancestor-churn re-reach gate is green.

### D-RLM-8 🟥 Demand ingest is SOURCE-BLIND — `record_realm_demands` accepts a `RealmDemand` from ANY cluster-bundle holder (RLM 5f-3e security vet `wf_a7008b28`, 2026-07-26)
`record_realm_demands` (`crates/node/src/rlm_runtime.rs` ~355-380) destructures `Inbound::Wire { class, bytes, .. }` — DISCARDING the sender — and folds any decoded `RealmDemand` with NO authorization. Cluster mesh membership today = ONE shared mTLS bundle (`docs/design/identity_persistence.md §5`) that the DEV CLIENT also loads (`crates/bins/src/bin/client.rs` `ClusterTrust::from_der_dir`); the mesh accept path delivers every authenticated frame regardless of the peer book (`crates/io-prod/src/mesh.rs` `serve_data_stream_body`→`classify_and_deliver`; the shared cert proves cluster-membership, NOT the NodeId — sender-asserted). So ANY bundle holder can dial the orchestrator directly and inject `InterShardFlow::RealmDemand{child, verb: SpinUp}` on `MsgClass::Saga` → unbounded realm spawn once the reconciler is armed.
- **WHAT is missing:** sender authorization at the ingest — accept a `RealmDemand` only from the gateway NodeId or a NodeId in the orchestrator's own launch ledger/roster (the orchestrator ASSIGNS spawned shard NodeIds, so the legitimate set is known locally), drop+count the rest.
- **WHY it is not fixed now:** on the current trust model the sender NodeId is SELF-ASSERTED (not cryptographically bound to the connection), so a NodeId allow-list is spoofable — a real boundary needs per-node client-facing trust (certs binding NodeId↔connection). Adding a spoofable check now would be security theatre.
- **WHERE it lands / WHEN:** P7 (`identity_persistence`) — per-node client-facing trust distinct from the shared cluster mTLS bundle; once NodeId is connection-bound, add the sender allow-list as defense-in-depth. **BLOCKING before any cloud orchestrator arms `VD_DEMAND`.**
- **Interim guard (see [[D-RLM-7]]):** the 5f-3e cloud preflight refuses `VD_DEMAND` on ANY cloud node, so the vulnerable configuration (an armed cloud reconciler) cannot be deployed until this is closed.
- **SCOPE NOTE (2026-08-14, Step-5 lane cure):** the shard-side lane attestation that landed ([[D-LANE-1]] — receive arms compare the sender against the directory head) does NOT meet this row's objection and does not flip it: NodeId stays self-asserted, so those compares are a correctness bound (the zombie window), never authentication. What DID change the ground under this row: every demand a shard emits now names ITSELF or a DIRECT CHILD (`push_demand`'s structural gate, finding 37), so once P7 lands connection-bound NodeIds the orchestrator can enforce the mirror from the fenced record it already holds locally — the same head the fence CAS trusts, resolved locally, NOT the spoofable allow-list this row rejected. Whether to pre-land that orchestrator-side check on the current trust model is an open owner question (OD-6 of the Step-5 lane audit).

### D-RLM-7 🟧 The 5f-3e cloud demand-arming veto — an interim STUB disabling the demand-driven RLM in cloud until client trust exists (RLM 5f-3e, committed with 5f-3e)
`enforce_cloud_preflight` (`crates/io-prod/src/boot.rs`) fails LOUD if `VD_BUILD=cloud` AND `VD_DEMAND` is set, for ANY node role — because the demand ingest is source-blind ([[D-RLM-8]]) and every cluster-bundle holder is a mesh peer. This DISABLES the demand-driven realm lifecycle (the warp mechanism) in a real cloud deploy; it stays fully live at the process/DevTest tier (the 5f-4 gates).
- **WHAT is deferred:** actually running the demand-driven RLM in cloud with real clients.
- **WHERE it lands / WHEN:** P7 (`identity_persistence`) delivers per-node client-facing trust; that slice closes [[D-RLM-8]] and then RELAXES this veto (from "refuse `VD_DEMAND` in cloud" to "refuse unless client-facing trust is configured"). The enabling condition (`VD_BUILD=cloud` in the shipped ConfigMap) is pinned by `shipped_cloud_manifests_pin_the_cloud_profile` (`crates/bins/tests/cloud_preflight_process.rs`) so a manifest edit cannot silently disarm it.
- **Pin:** the veto's comment (`boot.rs`) + the gateway bin comment name this entry; `CloudProfileError::DemandWithoutClientTrust`. Flips 🟩 when P7 lands and the veto is relaxed to gate on client-facing trust.

### D-RLM-9 🟥 The demand-login bootstrap-TTL fail-safe leg is unproven at the process tier; a never-syncing gateway holds a login open indefinitely (RLM RG-4c, empirical spike 2026-07-27)
- **WHAT is deferred:** a process-tier e2e leg proving a demand login whose home NEVER boots is CLOSED at the gateway's bounded `bootstrap_ttl_ticks` (~140 local ticks / ~2.8s at the DEV budget), not left hanging. The positive login + the reactive-greeting self-heal leg + the `process_parity` non-vacuity control are proven (`crates/bins/tests/rlm_demand_login.rs`, committed with RG-4c); this fail-safe corner is the remaining leg.
- **WHY deferred (empirical finding, not a guess):** the obvious inductions do NOT reach the TTL. With NO orchestrator (or the gateway dropped from the orchestrator's `VD_PEERS`) the gateway never CLOCK-SYNCS — the orchestrator is the clock source — so its seed injector stays inert (the `has_synced` gate), no `RealmDemand` is ever emitted, and the TTL clock never starts: the login sits with `sessions_open == 1` and `home_bootstrap_timeouts == 0` for 60s+. The TTL fires ONLY for a SYNCED gateway whose home genuinely fails to boot, which CA-1 learning (the orchestrator learns the gateway's connection from the inbound demand frame and can deliver the grant WITHOUT the gateway being booked) makes delicate to induce; the likely clean induction is a synced orchestrator with an EXHAUSTED RLM spawn-port band (`VD_RLM_PORT_LIMIT ≈ VD_RLM_FIRST_PORT`) so the spawn fails while the clock still flows.
- **SECOND finding (a possible separate gap):** a gateway that never clock-syncs (partitioned from its orchestrator) ACCEPTS a login into an open session (`sessions_open == 1`, `logins_held_pre_sync == 0`) and holds it INDEFINITELY — no bootstrap-TTL close, no pre-sync hold. Whether a never-syncing gateway SHOULD self-fence / hold / reject such logins is an open robustness question distinct from the demand TTL.
- **Where:** `crates/connection-plane/src/gateway.rs` (`SeedInjectorConfig::bootstrap_ttl_ticks`, `home_bootstrap_timeouts`, the `has_synced`-gated seed injector); the deferred leg lands in `crates/bins/tests/rlm_demand_login.rs`.
- **When / proper:** revisit with RG-5 / the P3 crash-chaos matrix — induce the synced-but-spawn-fails scenario (port-band exhaustion) for the TTL leg, and decide the pre-sync-hold policy.

### D-RLM-10 🟥 The VU-AoI parent-realm resolve keys the parent's directory read on the LOSSY `lowered()` `RealmId` — must migrate to `path()`-keying TOGETHER with the directory (VU AoI S2a-2b, 2026-07-28)
- **WHAT is deferred:** the child shard's up-relay resolves its PARENT realm's authoritative node by HeadRead-ing `DirectoryKey::Realm(parent.lowered())` (`parent_headread_due`/`update_parent_node`, `crates/sim/src/stub.rs`). `lowered()` collapses a whole lineage to a single `RealmId` (Universe→System(0), Galaxy→System(1), and every `System(seed)` across every galaxy by seed alone), so at multi-galaxy scale the parent key is NOT collision-free — the same limitation the DIRECTORY itself has (`DirectoryKey::Realm` is lowered-keyed everywhere).
- **WHY it is safe today (not a crisis):** the one universe where this MIS-resolves — a shard whose parent lowers to the SAME `RealmId` as itself (a `System(1)` hosted directly under a `Galaxy`) — is UNREPRESENTABLE in the lowered-keyed directory (the two realms would share a directory key), and a boot `debug_assert` in `register_stub_shard` REJECTS exactly that self-alias LOUD (`register_stub_shard_rejects_a_parent_that_aliases_its_own_realm_id`). So the resolve is correct for every representable universe; it inherits, never widens, the directory's own key lossiness.
- **WHERE it lands / WHEN:** with the DIRECTORY's migration to the collision-free `RealmCoord::path()` key (see [[D-32]]/[[D-41]] keyspace work) — the two share one lossy key and must move together, else a `path()`-keyed directory paired with a `lowered()`-keyed parent resolve would re-open the alias. Flips 🟩 when the parent resolve + directory both key on `path()`.
- **Pin:** the `[MAJOR guard]` comment + the boot `debug_assert` (`register_stub_shard`) and the `parent_headread_due`/`update_parent_node` doc comments name this entry; `realm_coord.rs`'s "NEVER use `lowered()` as a directory/dedup key at multi-galaxy scale" is the standing law it defers against.
- **WIDENED (2026-08-14, Step-5 lane cure):** the CHILD admission map `ChildRealmNodes` + `update_child_node` (the up-lanes' attestation, [[D-LANE-1]]) key on the same lossy `lowered()` `RealmId`, deliberately matching the directory and this row exactly — all three migrate to `path()`-keying together.

### D-RLM-11 🟩 LANDED (VU-AoI S2c) — sibling RENDER to the traveller's client, drawn ON APPROACH (Option C, reply-down; vetted `vu-aoi-s2c-design` 2026-07-29)
- **LANDED (`372e54f` wire + `ed06bdd` parent-emit + `831d1f3` home-reconcile):** the warmed neighbour realm is now DRAWN on the traveller's client while they are still inside their own realm. The parent REFLECTS each proxy occupant's full current in-range sibling set DOWN to the home shard it already relays with (the up-relay's own return address); the home shard — the sole holder of the dot's client route — RECONCILES the full set against a per-account baseline (`ForwardedProxyScene`) into the UNCHANGED client `RealmSceneDelta` add/remove and forwards it. So the neighbour is visible on APPROACH, continuous through the crossing (client unions by `RealmId`, sticky), torn down on back-away. Sim-tier e2e proven (`e2e_parent_reflects_a_sibling_and_the_home_forwards_it_to_the_client`).
- **THE OPTION DECISION — Option C (reply-down) CHOSEN; the earlier "(B, recommended)" prose above was REFUTED.** The vet found the directory has NO `Account→Gateway` resolution (`DirectoryKey = Session|Entity|Realm|Ship`; an `Entity` head → `AuthorityRef::Shard`, never a gateway), so B was NOT "no wire growth": it needed a new shared-directory `Account→Gateway` record that PUBLISHES the client-connection detail cluster-wide — the opposite of the user-decisive "no cross-realm leakage" constraint. Option A leaked `d.gateway` UP a frozen relay + a new parent→gateway edge. Option C — the parent replies down to the relay SENDER (`from`, provably the dot's home shard) — leaks NOTHING (the parent never learns the gateway), adds NO topology edge, and grows NO client-facing wire (one appended intra-cluster `InterShardFlow::ProxySceneSet`, public `RealmShape` geometry only, HR1).
- **The critical revision the vet forced (all 3 lenses):** an EDGE feed `{added,removed}` addressed to the stored home FAILS on EVERY crossing (the home dies mid-re-home → delivered-then-dropped → permanent ghost). Shipped LEVEL-triggered instead (`ProxySceneSet{observer, realms:full set}`, send-on-change; the home reconciles vs baseline; forward matched by ACCOUNT alone so pre-commit re-home removes land) — a lost/shed reflect or a re-home hand-off self-heals.
- **Pin:** `scripts/vu_aoi_s2c_plan.md` + the `ProxySceneSet`/`on_proxy_scene_set`/`ForwardedProxyScene`/`ProxySentScene` doc comments. **STILL OWED: the pixel-visible dot-crosses-a-boundary acceptance test** (client tier) is now UNBLOCKED (S2b + S2c both in) but not yet written — a VU visual-arc item. Residuals → [[D-RLM-13]].
- **SUPERSEDED by Step 5 (2026-08-13, `step5_sl7_lane_deletion.md`):** the per-OCCUPANT reflect this entry landed was an SL2/SL7 breach (occupant identity + interest crossing realm boundaries) and is DELETED — `ProxySceneSet` is a wire TOMBSTONE (slice D, minor 12) and `on_proxy_scene_set`/`ForwardedProxyScene`/`ProxySentScene` no longer exist. **SUPERSEDED AGAIN (window lane Slice C2, 2026-08-16):** its replacement `ChildSceneSet` + `FromAboveScene` is ALSO deleted — no parent reflects any scenery down any more. The PRODUCT behavior (neighbour drawn on approach, continuous through the crossing) now rides the window lane: every chain level states its own rows straight to the observer's gateway, which stacks them at ONE universe tick. The historical record above stays as the option-decision provenance.

### D-RLM-13 🟥 VU-AoI — the crossing scene-reconcile residual (a departed-sibling stale outline across an authority change) → VU-6 (the warp / scene-reset re-stream)
- **WHAT is deferred:** two bounded stale-render corners at the exact hand-off instant, both the SAME class as the dot render path's shard-crash stale-render (an authoritative re-draw on a scene/authority change, which VU-6 owns):
  - (i) a sibling that DROPS OUT of range in the narrow `[dest-Committed, home-flip-to-B]` sub-window — after the source dropped the dot from `dots.0` but before the parent's last-wins moved `home` to B — is known to neither the old home (dot gone) nor the new home (not told yet). A bounded (≤ one relay cadence) transient ghost, NOT the every-crossing permanent ghost the EDGE feed would have produced.
  - (ii) the OLD home's OWN interior children (the traveller's just-departed realm's sub-boxes) linger after re-home: the render emit loop iterates present dots only, so it sends no `removed` on departure, and `retain_live` silently drops the membership keys. **Pre-existing S1b** (not introduced by S2c); NOT covered by the client `RealmId` union (no `removed` is ever sent). On the S2c side, a `ProxySceneSet` for a departed dot drops the home's forwarded baseline (`on_proxy_scene_set` lazy cleanup, `proxy_scene_orphaned` counter) but does not itself emit a client removal.
- **WHERE it lands / WHEN:** VU-6 (warp re-stream / scene-swap) — an authoritative scene RE-STREAM on an authority change re-draws the client from truth, discharging both. **Pin:** `scripts/vu_aoi_s2c_plan.md` §C/§E + the `on_proxy_scene_set`/`ForwardedProxyScene` doc comments.
- **Step 5 slice D note (2026-08-13):** the S2c mechanisms this entry cites (`on_proxy_scene_set` lazy cleanup, `proxy_scene_orphaned`) are DELETED with the per-occupant lane — see the [[D-RLM-11]] supersession. **Window lane Slice C2 note (2026-08-16):** the `ChildSceneSet`/`FromAboveScene` lane this note re-homed the residual onto is deleted too. The residual CLASS is unchanged: a stale outline across an authority change now manifests at the GATEWAY's composer, where a look is held per subject and pruned on the derived TTL / roster loss — the exact-instant window remains until the authoritative re-stream (the origin-epoch swap, §2.7, already closes the CROSSING half of it).

### D-RLM-12 🟩 OBSOLETE BY DELETION (Step 5 slice D, 2026-08-13) — the machinery this entry guards is gone and its S3 successor can never exist
- **WHAT is deferred:** freeze-hold (carry-forward the proxy's existing latch keys, or a cached last-good composed position) when a retained proxy's relayed frame momentarily cannot be placed, so a transient un-measurable tick does not hard-evict-then-respin (a blink). In S2b the fold does a straight `transfer_frame` → `Err ⇒ None ⇒ not folded this tick`, and `retain_live` hard-evicts a latch absent for one tick (no grace path for a skipped proxy).
- **WHY it is safe today:** in S2b's scope the proxy is a DIRECT-child occupant whose child is always in the parent's region roster, so `transfer_frame` always `Ok`s (proven by `proxy_observer_safe_degrades...` — the `Err` path is reachable ONLY for an out-of-scope coarsen-ladder grand-child the parent does not place, or a stale post-re-home relay, both of which SHOULD fold nothing). So no oscillating `Err`, no blink, in scope.
- **WHERE it lands / WHEN:** S3 (the coarsen ladder — `coarsen_level`, `intershard.rs`), when a parent legitimately receives + places a grand-child relay. Add the carry-forward/last-good-composed guard THEN. Owner: S3.
- **Pin:** `scripts/vu_aoi_s2b_plan.md` §E (F-B1 finding) + the `proxy_observer` doc comment name this entry.
- **CLOSED WITHOUT LANDING (Step 5 slice D):** `proxy_observer`, the retained-proxy fold and its cited proof test (`proxy_observer_safe_degrades…`) are DELETED; `coarsen_level` is a field on a wire TOMBSTONE. The S3 coarsen-ladder this entry deferred TO is structurally impossible now — SL7 recursion (one bit per level, each level treating an occupied DIRECT child as the observer at the placement it authored) is the deep-ancestor keep-alive, and it has no compose step that could `Err` and no per-proxy latch to blink. The risk class died with the machinery; nothing is owed. (Records above kept as history; the entry stays for the trail.)

### D-RLM-14 🟥 The RETURN-crossing surrounding-realm feed does not re-advance on the client — no authoritative re-stream on a cross → VU-6 / floating-origin S6 (round-trip E2E, 2026-08-01)
- **★ SCOPE NOTE (window lane Slices C1+C2, 2026-08-16):** every mechanism this row names is GONE.
  `maybe_announce_realm_registry` was deleted at the Slice-C1 flag day, and the per-tick
  `MsgClass::RealmSnapshot` placements it describes were the OLD realm datagram, tombstoned in
  Slice C2. The DESIGNED cure is landed: the composed feed carries an ORIGIN MARKER, the gateway
  bumps `origin_epoch` at the crossing commit and re-emits a FULL level composed at the same tick
  as the last old-epoch datagram, and the client swaps atomically on the epoch (`window_lane.md`
  §2.7 — "the crossing re-stream and the origin marker are one mechanism"). What remains to CLOSE
  this row is the flip of its own pin: the OBSERVE-not-panic tail of
  `a_planet_to_system_return_commits_both_rehomes_and_the_player_rides` becomes a hard assert.
- **WHAT is deferred:** after a Planet→System RETURN crossing COMMITS, the returned client's SURROUNDING realm feed (the per-tick `MsgClass::RealmSnapshot` placements that keep the neighbour realms orbiting) does not reliably re-advance. The dest's read sub re-opens ONLY from the `SubscriptionReady` the dest emits at `on_saga_promote` (`crates/sim/src/stub.rs`), and the realm SCENE graph is streamed ONLY at a home-entry `Active` promote (`maybe_announce_realm_registry`, `crates/connection-plane/src/gateway.rs` — NOT re-emitted on a cross). So a returned player is live on the dest, commits, and RIDES its own realm, but its neighbours can stall until an authoritative re-stream lands. Distinct from [[D-RLM-13]] (bounded stale-outline ghosts at hand-off) — this is the whole per-tick feed, unbounded until re-stream.
- **WHY it is NOT a regression / safe to defer:** the reported TOTAL freeze (player + world both dead because the return parked on a reaped authority) is FIXED — the return-dest System stays live through the crossing so the return COMMITS (proven by the round-trip E2E reaching `location == "System 7"` and by the deterministic 3c `rlm.rs` twin). *(Mechanism updated 2026-08-14, Step-5 lane cure finding 37: the OUTWARD keep-alive this sentence originally credited is deleted — a shard demands only itself or a direct child (`push_demand`'s structural gate) — and the return-dest is held by arm B of `desired_alive` (the source still speaks for the departing occupant, so it never reports Empty) + `ancestor_close`; the round-trip E2E re-measured green over exactly that path.)* This residual is the NARROWER "neighbours stall after the commit" facet, newly EXPOSED by the new round-trip E2E (`a_planet_to_system_return_commits_both_rehomes_and_the_player_rides`), not caused by the reap fix (which cannot suppress emission) nor by the read-only client render changes (floating-origin S2/S3 `world_pos`).
- **WHERE it lands / WHEN:** VU-6 (warp re-stream / scene-swap) / floating-origin S6 (`anchor_epoch` re-emit at FORK-0a) — an authoritative scene RE-STREAM on an authority change re-opens the read sub + re-ships the neighbour placements from truth, discharging this together with [[D-RLM-13]]. The likely mechanism: re-emit the realm registry/scene (and re-arm the dest read sub) at the return-crossing promote, idempotently. **Repro / pin:** the OBSERVE-not-panic tail of `a_planet_to_system_return_commits_both_rehomes_and_the_player_rides` (`crates/bins/tests/rlm_demand_login.rs`) — flip its `neighbour_feed_live` observation to a hard `>= base + 20` assert when this lands. Owner: VU-6 / floating-origin S6.

### D-48-S1 🟩 THE SAVED-DATA LABEL — LANDED (slice S1, owner ruling 2026-08-24 Q1 condition 1)

- **WHAT LANDED.** Every durable file carries a label naming the world that wrote it, verified BEFORE any
  family is scanned (positional encoding: a wrong-shaped file decodes into RECORDS, not into an error).
  The label holds only fields COMPARED FOR EQUALITY — layout, role, world seed, epoch, and two DERIVED
  generations (`crates/core/src/store_stamp.rs`). The coordinate generation folds over `Tier::ALL`'s
  metres-per-cell, so it CANNOT fail to move when a unit moves — which is the whole reason it exists,
  since S8 re-values the coarse step and a position read under the wrong unit is silently wrong by the
  ratio between them. The world generation folds `vd_physics::worldgen::world_shape_constants()`.
- **THE DECISION IS IN THE PURE CRATE** so the 100 % gate holds every branch; `vd-io-prod` only reads
  bytes and calls it. There is NO unstamped door: `RedbStore::open` REQUIRES the label (a label written
  and never read is worse than none — this tree already held one of those, cured below).
- **THE FOUR OUTCOMES**, each driven separately: absent+empty ⇒ write and proceed; absent+non-empty ⇒
  REFUSE (owner: refuse, do not convert); present+equal ⇒ proceed; present+different ⇒ refuse naming the
  field and BOTH values.
- **THE WAY FORWARD** — `VD_STORE_ALLOW_GENESIS` deletes the file and starts a new world, named in the
  refusal's own message. Deliberately NOT a fallback inside the opener (a wipe that could happen on its
  own eventually would), and reachable ONLY from a label refusal — proven: another error class against an
  existing file leaves it untouched.
- **THREE THINGS FOUND BY MEASUREMENT, not reasoning:**
  1. **We touched a file before refusing it.** The table-ensure is a WRITE and ran first, so a refused
     file had already been modified — the operator is sent to look at something we changed. Order is now
     read, refuse, then write; pinned by a byte-for-byte assertion over the whole file.
  2. **Writing the label broke four durability assertions** that a fresh store is empty. Fixed in
     `is_empty` (a label is not a record), NEVER by weakening them — they are the durability contract's
     own control.
  3. **THE DEPLOY CHECK I WROTE FIRST WAS VACUOUS**: it grepped for the word `OnDelete`, which my own
     explanatory comment contained, so it would have reported success on a rolling update forever. Found
     by planting a rolling update and watching it pass. It now matches the SETTING and is proven to fail.
- **THE RESPAWN LOOP IS CLOSED** (`crates/node/src/rlm_runtime.rs`). A successful fork used to count as a
  successful LAUNCH and clear the failure streak — but a fork only proves a PROCESS started, not that a
  REALM came up. A child that starts and dies at once therefore never engaged the backoff and was
  respawned at the base cooldown forever. **A launch is now proven by the realm's HEAD appearing**, and a
  child that dies without ever registering is counted as the failure it is. This became load-bearing the
  moment a store could refuse: a refusing shard IS a child that starts and dies.
- **A REFUSAL NO LONGER ORPHANS A CLUSTER.** Children run in their own process group and are not killed
  when the orchestrator drops them, and the launch ledger is the only record of them. The orchestrator
  now reaps the recorded process groups through a narrow read-only door
  (`store::scan_refused_for_reaping` — read-only, no writer, opaque bytes, adopts nothing) before exiting,
  and reports the count. Zero means the rows could not be decoded either, and says so.
- **ROLLING UPDATES ARE REFUSED.** All three StatefulSets defaulted to replacing pods one at a time, which
  with the label armed is not a degraded release but a cluster that PARTITIONS ITSELF, each half refusing
  the other's files. All three are `OnDelete`; `just k3d-validate` fails if one drifts back.
- **THE SILENT VERSION DROP IS CURED** (`crates/io-prod/src/outbox.rs`). The one format version this tree
  already had failed on exactly its own event: `scan_all` dropped version-mismatched rows with no count
  and no log, and the replay then reported the same thing it reports for an empty store — so a node came
  back believing it owed nobody anything. Unreadable rows are now counted and reported at error level,
  the two kinds (bad key, bad value) separated, and the rows are SKIPPED, never deleted.
- **STILL OWED (named, not hidden):** cluster identity and orchestrator identity are NOT in the label —
  neither exists as a value anywhere in the tree, and inventing one is a design decision, not a line of
  code (D-47 wants both). Per-family record migration stays deferred under the owner's refuse-don't-convert
  ruling. The byte-golden over the seven durable value shapes is NOT written.
- **GATES:** workspace 2201 passed / 0 failed; Tier-A coverage 100 % of lines and branch sides; fmt and
  clippy clean; `just k3d-validate` green AND proven to fail with a rolling update planted.

### D-S3 🟩 THE COORDINATE UNIT IS REFUSED AT THE HANDSHAKE — LANDED (slice S3, owner-approved 2026-08-24 Q1 conditions 2+3)

- **WHY.** Slice S8 re-values the galaxy's coordinate step. From that day two builds can be the SAME
  version, both healthy, both talking — and every position between them wrong by the ratio between the
  units. Nothing crashes, nothing is logged, a player is simply somewhere else. And **a step change moves
  ZERO bytes**: the edges are compile-time constants, so no golden reddens and no decode fails. Every
  safeguard here had to be built deliberately.
- **CLIENT SIDE — `ProtoVersion` gained `coordinate_generation`**, folded at COMPILE TIME over the
  coordinate tier table (`vd_core::store_stamp::coordinate_generation`, now a `const fn`). **The unit is
  NOT negotiable**: major and minor may differ and still talk (that is what a floor and a minimum are
  for), but there is no lower UNIT two builds can agree to speak — so it is an equality and it refuses.
  The refusal names the field and BOTH values, because it is the one refusal whose cause is invisible
  from outside. Driven at the REAL negotiation site (the gateway's one `negotiate` call), not against the
  type.
- **FLAG DAY, minor 22 → 23, floor with it.** The version rides inside `Hello` and postcard is positional,
  so the field re-labels every byte after it. The tree's own completeness check demanded three things
  before accepting it: a ledger entry per minor, an owner citation with an accepted marker, and a written
  reason for the floor. All three are in `crates/wire/src/version.rs`.
- **FLEET SIDE — the unit is folded into the transport tag** (`intershard_alpn`), so a mismatched node is
  refused **by the transport itself**: no new message, no new wire arm, and the fleet half stays out of
  SL6's ask entirely. **A CORRECTION TO BOTH PLANS, which claimed a diagnosis this route cannot deliver:**
  a tag mismatch produces `no_application_protocol` — a connection error with no field, no value and no
  unit. So the tag buys the REFUSAL only. The DIAGNOSIS is carried elsewhere and deliberately: the tag is
  built to be READ (the unit appears as hex text), every node states it on its admin view
  (`AdminSnapshot.coordinate_unit_tag`, present even on the empty snapshot) and in its boot log. A test
  asserts the admin string and the transport bytes are IDENTICAL — two renderings of one fact is how an
  operator gets sent to the wrong cause.
- **Q1 CONDITION 3 — every position diagnostic names its frame AND its unit.** Not twelve hand-written
  field lists: ONE helper (`vd_core::pose::describe`) that states metres, frame and unit together, so a
  new diagnostic cannot print half of it. A bare number of metres is not a position — the same integer
  counts millimetres in one frame and metres in another.
- **A MISLEADING COMMENT CORRECTED**, and it would have pointed the next reader the wrong way:
  `crates/client/src/interp.rs` claimed the unit "was SHIPPED, not inferred". It is not. **The frame NAME
  ships; the UNIT is a compile-time table on the receiver.** The safety comes from this slice's refusal,
  and the doc now says so.
- **ONE FOLD, THREE USERS** — the durable label (S1), the client handshake and the fleet tag all fold the
  same table. Asserted, so a store and a peer can never disagree about which world they are in.
- **STILL OWED:** the dialling side does not yet log the tag it offered when a connection fails (the plan
  asks for it; the boot log and admin view carry the same string, so an operator can compare two nodes,
  but the failure itself is still silent on the dialling end).
- **GATES:** workspace 2213 passed / 0 failed; Tier-A coverage 100 %; fmt and clippy clean. One
  self-caught defect: the first refusal test joined two conditions with `&&`, which hides one side's false
  branch from the coverage gate — the project's own written rule, and it was split.

### D-S4 🟩 SUBTRACT BEFORE YOU FLATTEN — LANDED (slice S4). A defect that was LIVE on the world as it stands

- **THE DEFECT, not a future one.** A drawn position and the eye were each flattened from the frame
  origin and THEN subtracted. At the star placement radius a position is ~1.53e18 fine cells, whose f64
  spacing is 256 cells = **0.25 m**. So two ships flying in convoy each rounded to a quarter of a metre
  INDEPENDENTLY, and their drawn separation was wrong by up to half a metre and FLICKERED as they moved.
- **★ THE GATE ALL THREE DESIGNS PROPOSED IS GREEN TODAY AND CANNOT FAIL.** Every one of them led with
  *"two things 100 m apart must draw 100.000 m apart — RED BEFORE"*. Re-derived and confirmed by
  measurement: 100 m is exactly 102,400 cells, 102,400 mod 256 == 0, so BOTH endpoints round by the same
  amount, the errors cancel, and the answer is exactly 100.000 m. **Every whole-metre separation is exact
  here for the same reason.** A gate built on one would have passed for ever with the defect standing.
  Kept as a test (`the_whole_metre_gate_the_designs_proposed_is_green_today`) so nobody re-proposes it.
- **THE GATE THAT CAN FAIL** sweeps separations that are NOT multiples of the rounding quantum (e.g.
  102,501 cells = 100.0986 m). Measured: the old path is wrong by up to 0.25 m per axis; the lattice path
  is **exactly zero**, not merely smaller. Both in one test body, in `vd-core` (the property) and in
  `vd-client-harness` (the convoy).
- **TWO DEFECTS, TWO PLACES, DIFFERENT FAILURES — said apart so a gate is not written against the wrong one:**
  - **CLIENT:** cancellation error in a DRAWN SEPARATION. `RenderEye` now carries the eye as a lattice
    position with its unit, and the entity and realm-box draw sites reduce against it before flattening
    (`eye_relative_lattice`). The eye is an `Option` and NOT a defaulted `Tier`: there is no default
    unit, and a position reduced against a guessed one is wrong by the ratio between guess and truth —
    the very defect being removed.
  - **SERVER:** a KEY FLIP near a grid boundary in the child index (written the same day). The division
    was already exact (a power-of-two edge), so the error entered ONLY through the flatten. And this
    index's empty answer is TRUSTED as a positive "not here", so a lost key does not blur a boundary —
    **it makes a realm unenterable**. `cell_key` now keys on the integer cell plus a small local
    displacement, never on a distance from the frame origin. Proven at the placement radius.
- **TWO BUGS IN MY OWN REWRITE, both caught by the tests, both real:** the new key ignored a position's
  sub-cell offset (so every child built with a pure local offset keyed to cell zero), and an index with
  no extent divided by a zero grid edge. Fixed by folding the offset into the local displacement and by
  taking the same early exit `candidates` already takes.
- **STILL OWED:** the plan's measured touch list names thirteen production sites; the three load-bearing
  ones (both draw sites and the index) are converted. The remainder —
  `client-harness/src/camera.rs:455,756`, `verdict.rs:55,70`, `FollowCamera::eye`,
  `pilot_capture_camera` — still build or compare from already-flattened values. They are camera and
  verdict paths, so the error is relative to the DISTANCE to the thing rather than between two drawn
  things; the convoy defect is closed. Convert them with S9, when the magnitudes actually move.
- **GATES:** workspace 2221 passed / 0 failed; Tier-A coverage 100 %; fmt and clippy clean.

### D-S5 🟩 MEMBERSHIP TESTS THE WHOLE TICK'S MOTION — LANDED (slice S5). The module header was a claim; it is now a measurement

- **WHAT THE HEADER SAID, AND WHAT WAS TRUE.** `crates/core/src/geometry.rs` has stated since it was
  written that *"membership evaluation tests the tick's whole motion segment against the shell, so a fast
  body cannot tunnel undetected."* It did not: the verdict sampled the pose and nothing else. Acquisition
  needed a SAMPLE landing at least one inset INSIDE the surface, so **no widening of any band could ever
  buy acquisition** — the property A4 at P8 depends on, and the one that decouples enterability from a
  child's size.
- **BOTH EARLIER PLANS MIS-SIZED THIS, FOR THE SAME REASON.** Each assumed the swept primitives were
  built and merely needed wiring. They ARE built and had zero callers — but their signatures are f64
  METRES, while the shipped verdict decides on **integer cell squares**, on purpose, so the answer is
  exact at every magnitude and identical on every host. Wiring the decimal pair in would have abandoned
  integer exactness on the one path where it matters most. This slice is **new integer swept
  mathematics**, and the decimal pair stays what it now is: a zero-caller oracle with a stated validity
  window (bit-comparable only while `|cell delta| ≤ 2⁵³`, about 59 AU — one f64 ulp equals one Fine cell
  at 4.4e12 m, which is INSIDE the home system, so an equality against it would pass on small fixtures
  and start failing exactly when a test finally ran at real scale, with the ORACLE at fault).
- **THE OVERFLOW, WHICH IS THE HARD PART, VERIFIED INDEPENDENTLY OF THE DESIGN THAT PROPOSED IT.** The
  textbook interior test cannot be computed directly at the distances this world allows:

  | Term | Worst value | Fits signed 128-bit? |
  |---|---|---|
  | `a · d` | 1.276e38 | yes, margin 1.333× |
  | `d · d` | 2.552e38 | **NO** — must be unsigned |
  | the deciding product | 1.628e76 (**254 bits**) | **NO** — needs 256 |

  So 256 bits is the MINIMUM sufficient width and it IS sufficient, with the maximum attained rather
  than estimated. There is no cleverer 128-bit identity: granting a free clip, the largest region an
  i128-only form could cover is 1,895 km — an Earth-sized realm misses by 128×, a star system by 1.2e21×,
  the galaxy by 2.0e36×. **One path, 256-bit, always.** The wide multiply is four 64-bit limb products
  with **zero branches**, checked against an oracle built from a DIFFERENT (32-bit) limb split so the
  test is not the implementation restated.
- **WHAT THE VERDICT NOW ADDS — EXACTLY ONE THING.** "The path dipped inside while BOTH samples were
  outside." Inside-now, inside-then and LEAVING all keep the shipped answer bit for bit. The rejected
  alternative also held a subject for one tick after it left; at a galaxy ceiling that subject is
  thousands of shell radii outside, so it inverts membership at precisely the speeds the machinery
  exists for. The shipped point answer is asked FIRST and returns immediately, so **the upgrade can only
  turn `false` into `true`: no subject that is a member today can stop being one.**
- **★ THE PLAN'S OWN GATE WAS FALSE, AND IT IS REPLACED, NOT WEAKENED.** The plan required *"for every
  subject whose per-tick travel is below the band, the swept verdict must equal the point verdict, bit
  for bit."* **Two cells of travel — 1.95 mm, three orders below the shipped 3 m band — genuinely clips a
  region at every size**, verified at seven real radii from a 5 m station to the galaxy bound: for
  `a = (E,−1,0) → b = (E,+1,0)` both endpoints are outside and the midpoint sits exactly on the surface.
  Written as an equality that gate would be **RED for a correct implementation**, and the cheapest way to
  turn it green would be to weaken the code. Replaced by four arms that can still fail: stationary
  identity; **one-cell equality** (proved from the both-outside inequality, then checked exhaustively
  over 187,008 shell and 135,492 box pairs, with the bound shown ATTAINED at `|d|² = 4` so it is tight
  rather than padded); an exactly-tight integer divergence envelope; and a float-free lattice oracle.
- **EVERY GATE WAS PROVEN FALSIFIABLE BY PLANTING THE DEFECT.** Removing the sweep, dropping the
  nearest-point-is-the-end exit, narrowing the 256-bit compare to 128, replacing the checked subtraction
  with a bare one, enumerating only the endpoint cells, un-seeding the arrival prior, and reverting the
  scan to the instant — each turned a named test red, and each was restored. **One planted defect was
  NOT caught, and that is recorded rather than hidden:** replacing the checked subtraction with a
  WRAPPING one is indistinguishable, and provably so — a wrapped `g` is about 2¹²⁸ while `g·s ≤ (a·d)²`
  requires `g ≤ |a|²`, so the wrapped form answers `false` wherever the checked form does. The guard is
  kept for the debug panic it prevents, and the code now says so instead of claiming more.
- **THREE COMPANION CHANGES LANDED IN THE SAME COMMIT, because without them the slice is inert or unsafe.**
  1. **The arrival prior.** A Ghost is minted with its prior at the realm ORIGIN. Nothing read the field,
     so it was harmless; once membership tests a segment, the first scan after a cross-node arrival would
     sweep a realm-wide line from the destination's centre to the arrival point — through regions the
     subject never visited — straight into a re-home decision. **This is the same defect class as the
     re-home freeze that cost a week and produced a turn-back tag.** Re-seeded at `apply_crossing`.
  2. **The lookup asks about the segment.** `ChildIndex::candidates` reads exactly one cell keyed on one
     point, and its answer is trusted as positive information. A child the subject tunnels through holds
     NEITHER endpoint, so it was skipped before the verdict could be asked — the arithmetic would have
     been correct and unreachable. `candidates_segment` enumerates the endpoint key-box WHOLE (a subset
     would look exactly like a positive "not here") and **abstains** when doing so would touch more grid
     cells than the index has occupied — a cost argument against the index's own size, not a threshold
     someone picked. `worth_asking` reads the abstain flag FIRST: abstaining is ignorance, not a miss.
  3. **Two bare `i64::abs()` cured.** `i64::MIN.abs()` panics in debug — which is the entire test and
     coverage suite — and `normalize` saturates to `i64::MIN`, not to the sanitize domain. S5 did not
     create the hazard; it doubles the traffic through it (two endpoints per region instead of one).
- **GATES:** workspace **2246 passed / 0 failed** (baseline 2221 → +25); Tier-A coverage **100 % region and
  branch**; fmt and clippy clean at `-D warnings`. The coverage gate failed TWICE first, and both times the
  misses were in the NEW TEST CODE, not in the arithmetic — one of them a `&&` short-circuit inside the
  reference implementation, which is the very pattern this project's written rule forbids. Neither was
  exempted: a guard that could never fire was DELETED (a branch nothing can drive is not coverage), and the
  two remaining arms got real fixtures — a region thinner than its own inset (never acquirable at any
  distance) and a ROTATED region (so "the swept change does not touch the decimal arm" is a measurement
  rather than an omission). Both fixes made the gate stronger, not merely green.
- **WHAT THIS DELIBERATELY DOES NOT DO — named in the module header so the next reader inherits the
  limits, not the old false sentence.**
  - **A MOVING REGION sweeping over a PARKED subject is not covered.** Both endpoints are reframed
    through THIS tick's book, so the segment tested is the subject's own displacement, not the relative
    one. The tree's own G-IDENTICAL fixture `drive_swept_crossing_feature` parks the occupant and moves
    the child — **that fixture passes while proving nothing about this slice, and must not be read as its
    acceptance line.** The cure is the prior folded through the PREVIOUS tick's book (one extra ledger
    lookup and one extra frame conversion per region per subject per tick); it is out of S5 because the
    one-book form is what keeps the stationary answer structurally identical. **WHEN:** with the relative-
    motion work at P8.
  - **`Obb` is not swept.** A rotated box needs a rotation, and no rotation of an integer lattice vector
    is an integer lattice vector, so a swept `Obb` would decide on a square-cornered box while its point
    form decides on a rounded-corner distance field — a **1.46 m** disagreement at the shipped 2 m outset
    between the two halves of one verdict. No shipped region is an `Obb`. **WHEN:** with the first
    rotated region.
  - **One-tick fly-through is detected but not damped.** Acquiring a child and leaving it within one tick
    issues a re-home the next tick reverses. **Measured, not argued:** the band's own hysteresis damps it
    while one step of travel stays inside the band (the acquire edge sits an inset in, the release edge an
    outset out), and stops damping it above — both halves are pinned by a test that would have come out
    either way. This is the same class as the eighteen alternating crossings in eighteen seconds parked
    red at `crates/bins/tests/node_per_realm_walk.rs:241`. **WHEN:** S6's speed-sized bands, which is that
    test's own un-ignore condition.
  - **`guard_quantum_band` still does not exist.** The verdict's doc justifies discarding the sub-cell
    residual by naming a fence with no definition anywhere in the tree. The claim is true today by 3.5
    orders and nothing enforces it; S5 relies on the same discarded residual over a strictly larger
    surface. **S5 borrows an unenforced premise; S6 owes the fence.**
  - **The abstain rate on THE world's booted forest is UNMEASURED.** It could reintroduce the O(n) scan S4
    just removed, for exactly the fast subjects S9 creates. It is a measurement S5 owes, not a claim.
    **WHEN:** S6's index-quality gate, which already exists to measure this.
  - **The prior is re-judged against the edge in force for the CURRENT hysteresis side**, not the edge in
    force when the prior was sampled. True by accident of uniformity today (both band constants are
    global). **S6 makes every band tick-varying and breaks the premise.**
  - **The arrival flush guard's guarantee shrank, and its comment was rewritten to say so.** It holds ONE
    instant, so it asks the point half. It still catches every mislanding where the arriving pose ITSELF
    sits outside the band — the measured 12.15 / 23.00 / 20.72 m class it was built for — but an arrival
    fast enough to pass THROUGH the destination region within one tick would be refused there and
    accepted by the destination's scan. There is no prior to fix it with: the arriving pose is walked
    through a multi-link conversion chain, and the stored prior carries neither a stamp to convert it
    through that chain nor a frame tag to say what it was relative to.
  - **It is exact only while every placement is identity-oriented.** The day a rotating placement ships,
    each endpoint's fold costs up to one cell per axis, so the segment gains up to two cells per axis of
    purely spurious displacement — enough to manufacture a false dip in exactly the 1.95 mm regime where
    the real minimum divergence lives, and enough to break the one-cell equality proof.

### D-S6 🟩 BANDS SIZED FROM REAL CLOSING SPEED — LANDED (slice S6). The law the world's own radii were already solved against

> ⚠️ **THESE NUMBERS ARE NOT SAFE ACROSS THE MOVEMENT LAW (D-MOVE-1).** Every band below is sized FROM the
> approach governor's guarantee — *"the fastest a subject may lawfully be moving at a boundary is that
> boundary's own cap"*. The owner's movement law REMOVES that governor. This row's own measurement of the
> alternative is the warning: sizing against the parent's ceiling instead would ask for **bands thousands
> of times larger than the bodies they wrap**. The band solve must be re-opened with the movement law, not
> after it.

- **THE LAW WAS NOT INVENTED HERE.** The galaxy's radius has always been `R_universe − outset(v_cap)`, an
  expression whose `outset` term is a containment band that did not exist. Every band shipped as the same
  three metres — one value built once and copied onto every region in the universe, from a five-metre
  station to a two-and-a-quarter-petametre shell. This slice applies the formula the geometry already
  assumed, so the shells and the bands can no longer disagree about what a band costs.
- **EACH BOUNDARY IS SIZED FROM ITS OWN CEILING, and the reason is measured rather than argued.** The
  approach governor lowers a subject's ceiling onto the body it is nearing, and it is enforced on BOTH
  arms — approaching a child, and leaving your own realm. At either surface the distance term is zero, so
  **the fastest a subject may lawfully be moving at a boundary is that boundary's own cap.** Sizing
  against the PARENT's ceiling — which is what the S2 sweep did, correctly, before anything bounded the
  arrival speed — would size every band for a speed no lawful subject can hold there, and measured, would
  ask for bands thousands of times larger than the bodies they wrap.
- **BOTH INPUTS ARE CONSTANTS OF THE SOLVE, NEVER THE CLUSTER'S.** The tick already was
  (`GEOMETRY_TICK_DT_S`); the foot speed now is too (`GEOMETRY_V_FOOT_MPS`). A band is part of the world's
  geometry, and two clusters must boot the identical world however fast they tick and however fast they
  let a person walk. MEASURED: the foot speed binds on **0 of 13,144** boundaries of THE world — every
  generated realm is large enough that its own geometric ceiling dominates — so that constant does not
  size a single band that ships; it exists so the small-realm arm has a lawful value instead of a
  deployment's.
- **WHAT IT COSTS, over the derived 284-seed sweep:**

  | Reading | Result |
  |---|---|
  | Band as a share of its own body | **0.133 %** (the affordability line was 50 %) |
  | Nesting margin consumed | ~600 m of a **4.3285e8 m** margin |
  | Sibling-separation margin consumed | ~4.8e9 m of a **1.2072e14 m** margin |
  | Boundaries crossed unseen, before | **13,144 of 13,144** |
  | Boundaries crossed unseen, after | **0** |
  | Thinnest crossing | **5.59 ticks** at the shipped approach constant |

- **★ THE METRIC WAS RE-POINTED, VISIBLY, AND BOTH READINGS ARE STILL ASSERTED.** The S2 sweep counted
  boundaries unobservable at the PARENT's ceiling. That was the honest upper bound when nothing bounded
  arrival speed; it is now a speed the governor forbids. The ungoverned count is still computed and still
  pinned (12,860, down from 13,144 — S6 makes one boundary per world observable even against a speed that
  cannot happen), and the GOVERNED count is asserted at zero. A re-pointing, not a weakening: the old
  number did not quietly disappear.
- **THE QUANTUM FENCE IS WRITTEN.** `region_verdict` has cited `guard_quantum_band` by name since it was
  written, to justify discarding the sub-cell residual, and a grep of the tree returned that one comment
  and no definition. The claim held by three and a half orders at the millimetre grid and NOTHING enforced
  it, so it would have gone silently false the day a coarser grid lit up — or, as it turns out, the day a
  band was CLAMPED DOWN to fit a small body, which this slice does. It is now an arm of the boot fence,
  both sides driven, with the inclusive edge driven exactly.

- **★★ THE ARM I LEFT OUT, AND WHAT IT COST — the entry worth reading.** The design's band law has a clamp
  (`band ≤ inscribed extent × BAND_EXTENT_CLAMP`) and I implemented everything except it. **Seven re-home
  tests went from one crossing to none.** The acquire edge sits one third of the band INSIDE the surface,
  so a five-metre station handed a sixty-metre band has its acquire edge past its own centre and becomes
  impossible to enter AT ANY SPEED — the exact property this slice exists to guarantee, destroyed by the
  slice that guarantees it. The clamp is bound on the INSCRIBED extent, because that is the shape's
  narrowest direction and a band that fits the widest can still swallow the narrowest.
- **THREE FIXTURES CARRIED A DISTANCE THAT ONLY WORKED AT THREE METRES**, and all three now derive it:
  a departure staged "five metres past the shell", and two round-trip waypoints at "x = 50" whose system's
  release edge moved from 42 m to 53 m. Each looked like a physical fact and was really a band constant.

- **THREE OF MY OWN MEASUREMENTS WERE WRONG BEFORE THEY WERE RIGHT, and each was caught by a control
  rather than by luck. They are recorded because the discipline is the point.**
  1. **The fence check refused THE world by 1.5e15 m** — the size of the universe, from a band worth
     600 m. It subtracted a child's centre (in its parent's frame) from its parent's centre (in the
     GRANDPARENT's frame). Caught only because the test asks the same question with TODAY'S band first: a
     measurement that refuses the shipped world is measuring itself.
  2. **It reported 83,542 overlapping sibling pairs in THE world. There are none.** A moving planet's
     region carries a ZERO centre by design — its real placement is authored into the parent's book every
     tick — so every orbiting pair read as sitting exactly on its star. The test now skips moving pairs
     and says why, so the scare is not re-derived.
  3. **It reported a 3.33-tick crossing using an approach constant no cluster runs** (no demand cadence,
     no boot latency — the floor of the derivation). The shipped cadence gives **5.59 ticks**, which
     clears the cooldown. Both are now computed and both asserted, so neither can quietly become the
     convenient one.
- **WHAT IS STILL OWED, named rather than implied.**
  - **The ceiling fence is a measured gate, not a boot refusal.** On THE world it CANNOT fail, because the
    band is defined from the ceiling — the ratio is one sixth by construction, so a boot fence there would
    be decoration. It can only fail where the band is floored or clamped, i.e. on small hand-placed
    fixtures, where it would refuse our own test forest. Promoting it to a boot refusal belongs with A4 at
    P8, where a realm's ceiling is derived from its band rather than compared to it.
  - **At the approach constant's FLOOR the cooldown is not cleared** (10/3 ticks against 5). Reaching it
    needs a band factor of 15 rather than 6 — still a third of a percent of each body — but that factor
    also sets the galaxy's own radius, so moving it is S7's subject. Pinned, so a cluster configured into
    that corner turns a test red instead of thrashing quietly.
  - **★ THE PARKED PROCESS TEST WAS RE-RUN, AND IT MOVED THE STORY.** Fence **22 → 14** against a bound
    of 12, and **legs C, D, A and E now cross exactly once each** — four of five boundaries clean, where
    before only three were. The whole residual is leg F: nine crossings of the galaxy/ring-sibling
    boundary, down from eighteen.
    - **THE BAND IS NOT WHAT IS LEFT, and that is a measurement rather than a hope.** The occupant
      traverses the WHOLE band each way — released just outside the release edge, re-taken just inside
      the acquire edge, **6.6e8 m apart against a 4.7e8 m band**. That is what a real crossing looks
      like. My first reading of those two numbers was that the two shards DISAGREED about the position
      by 6.6e8 m; they do not, and the check that killed it was noticing I had compared a *release*
      event against a *re-acquire* event, which must lie on opposite band edges.
    - **THE CAUSE IS THE TEST'S OWN AIM.** Leg F names a FIXED vector — two solved shells down the
      SIBLING's pole — and the instant the occupant re-homes the server reads the same three numbers in
      the GALAXY's frame, where that point is essentially the galactic origin 1.5e15 m away. So it turns
      round, flies back into the sibling, and is handed down again. **Proved** by its distance from the
      galactic centre falling monotonically at every hand-down (1.499035431e15 → 1.499031347e15 →
      1.499026399e15 → 1.499021151e15 m).
    - **AN ATTEMPTED CURE WAS MEASURED AND REVERTED.** Aiming relative to the occupant's own delivered
      position made it **worse** (9 crossings → 11). The delivered pose changes frame at the DELIVERY
      LAG while the `WalkTo` target changes at the SERVER's re-home, so any aim mixing them is wrong in
      exactly the window the defect lives in. Reverted whole; the finding is kept, the speculation is
      not. **Owed:** dev-control must state WHICH realm frame a delivered pose is in.
    - **A PER-LEG FENCE PRINT LANDED** and is what turned "the fence ballooned" into "leg F does all of
      it". The original measurement had to be reconstructed by hand from shard logs that do not survive
      the run; this one falls out of it.
    - **The band factor question is therefore NOT what this test is waiting on.** Raising it would slow
      the oscillation and hide the cause. It stays an open S7 question on its own merits (the approach
      constant's floor), not as this test's cure.
  - **The index-quality gate is not yet taken.** The index radius is `extent + outset`, so the band
    inflates it by 0.089 % and can cost at most one octave of grid edge. Bounded, but UNMEASURED.

### D-S7 🟩 THE HEAVIEST STAR STOPS DEPENDING ON THE GALAXY'S SIZE — LANDED (slice S7), and the fence it lands REFUSES THE WORLD ON PURPOSE

- **THE INVERSION, which is the whole slice.** The heaviest star used to be *whatever this galaxy could
  afford*: `solve_mass_cap` read the galaxy's radius out of its own module and handed the answer to the
  star draw as the initial mass function's upper bound. So the biggest star in the universe was a
  consequence of a COORDINATE CHOICE, and every star in every system was drawn through it — the last time
  that number moved, **99 of 99 planet rows and 6 of 12 system rows moved with it**. Physics now states
  the top and the galaxy has to be big enough for it, which is the owner's Q9 ruling in its own words. A
  coordinate step is then JUDGED rather than obeyed.
- **THE ENABLING CHANGE IS ONE ARGUMENT, and its whole risk is that it must move nothing.** The budget is
  a parameter instead of a module read, so the one question the solve exists to answer — *what would the
  cap be if the galaxy were a different size?* — can finally be asked of it. Gated by re-solving at the
  shipped budget and comparing **all three** derived fields by exact equality (the mass, the reservation
  it implies, and the star's own photosphere; comparing only the mass would let the other two drift), plus
  a determinism arm so the equality is a property of the function and not of one lazily-cached value.
- **★ WHAT THE SOLVER SAYS, ON THE WORLD, AT EACH CANDIDATE STEP.** The design estimated ~1,048 solar
  masses at the ruled two-metre step by propagating a two-point fit and said in as many words that this
  was *"arithmetic through a FITTED exponent, not a measurement"*. It was **23 % low**:

  | Coordinate step | Galaxy radius (m) | Heaviest star (M☉) | Its shell (m) |
  |---|---|---|---|
  | 0.0009765625 (today) | 2.248797e15 | **16.360035** | 7.494898e14 |
  | 1 | 2.302769e18 | 866.013 | 7.675885e17 |
  | **2 (ruled)** | 4.605537e18 | **1287.288** | 1.535178e18 |
  | 16 | 3.684430e19 | 4226.907 | 1.228143e19 |
  | 1024 | 2.358035e21 | 45543.212 | 7.860117e20 |

  Today's row reproduces the shipped golden exactly, which is what makes the other four trustworthy. The
  harness also asserts the solve is MONOTONE in its budget — a larger galaxy can never afford a smaller
  star — because without that property every number in the table would be noise.
- **THE FENCE, AND IT IS RED ON THE WORLD WE SHIP.** `guard_galaxy_affords_its_stars` asks whether a
  galaxy can pay for a star at the stated physical top. **Today it cannot: that star costs 32.37× the
  whole galaxy**, which affords 16.36 M☉ against a stated 120. The fence **passes from a ONE-METRE step
  upward**, found by search rather than assumed, so the ruled two-metre step clears it with margin.
  - **THAT REFUSAL IS THE ARGUMENT FOR THE STEP CHANGE**, written as something that fails instead of as a
    paragraph. It is the reason the slice exists before the one-way door.
  - **IT IS NOT ARMED AT BOOT**, and the reason is the same shape as S6's ceiling fence: arming it today
    would refuse the world we ship, and the cure is the step, not a weaker fence. Both arms are driven by
    tests — the refusal on today's world, the pass at the ruled step — so neither is a green light nothing
    could turn red.
- **THE PHYSICAL TOP IS A NAMED FACT AGAIN.** `IMF_MASS_HI_PHYSICAL_MSUN = 120.0` is the literal that was
  deleted when the derived cap replaced it. Deleting it was right at the time and for the stated reason,
  but it lost something real: **how big a star the universe actually makes stopped being written down
  anywhere**, and "the heaviest star" silently came to mean "whatever this radius affords". The two are
  different KINDS of fact — one is physics, one is a fact about a coordinate step — and they now sit side
  by side, so which of them BINDS is a question that can be asked and answered.
- **THE EDGE THE PLAN FEARED WOULD BE DROPPED AS SOFT, DISCHARGED PRECISELY.** The plan warned that
  *"reconciling S2's two tick counts moves that radius and therefore the cap"*. It does not, and this is
  by inspection rather than by hope: the galaxy radius and the containment band both read
  `GEOMETRY_TICK_DT_S` (0.02) — a constant of the solve, deliberately not the cluster's tick — while
  `AOI_TICK_DT_S` (0.05) has separate readers in `InterestConfig` and never enters a radius. **Reconciling
  the area-of-interest tick cannot move the cap.** The two-home owe on the AoI tick stands on its own.
- **A STRUCTURAL FINDING, RECORDED NOT ACTED ON.** The shipped galaxy radius is derived as `R_universe`
  MINUS a band. That is a **single-lattice artefact**: it exists only because the galaxy and everything
  inside it share one coordinate grid, so the galaxy has to leave room inside the root's budget. Once the
  galaxy has its own lattice its radius becomes its own fence equality (2⁶¹ cells × its own step) and this
  subtraction dies. Do not harden anything against the subtraction's exact form.
- **WHAT IS STILL OWED.** Choosing the step is S9's, not this slice's — S7 exists to make that choice
  readable off a number. The fence's boot arming waits on the step. And the table above is the authority
  for that decision: **do not build against 1,287 any more than against the design's 1,048** — re-run the
  harness at whatever step is actually ruled.
- **GATES:** workspace suite green; Tier-A coverage 100 % region and branch; fmt and clippy clean at
  `-D warnings` (verified by exit code — an earlier report of "clippy ok" in this slice was a shell line
  that printed unconditionally, and clippy had in fact failed on a doc comment my own insertion orphaned).

### D-S8 🟩 THE LADDER, BUILT BUT NOT YET CLIMBED — LANDED (slice S8). Three rungs, every ratio a bit shift, and three false claims retired

- **WHY A LADDER AT ALL.** Every position in the universe counted in millimetres. A millimetre-counting
  ruler runs out at about a quarter of a light year, which is why a hundred and fifty thousand star systems
  would not fit in a galaxy — not by a little, by **974×**. Each level now counts in a step that suits it:
  **2⁻¹⁰ m** below a star system (UNCHANGED, bit for bit), **2 m** for a galaxy, **2¹⁵ m = 32,768 m** for
  the universe. Only the two levels nothing has ever been drawn from get a new one.
  - **TWO METRES, NOT THE DESIGN'S ONE** (owner ruling Q1, 2026-08-24, explicitly overriding the document):
    one metre left five percent of headroom against what 150,000 systems need and content grows; two costs
    nothing visible at light-year distances and buys eight times the room.
- **THE EXPONENT IS THE DATUM AND THE STEP IS DERIVED.** `Tier::step_exponent()` is stored; the edge is
  built from it through the IEEE bit pattern (`f64::powi` is not `const`). Two things fall out that a table
  of decimals could not give: every step is a power of two BY CONSTRUCTION — so re-bucketing is exactly
  idempotent at every rung and no decimal can drift — and every ratio is a DIFFERENCE OF EXPONENTS, i.e. a
  bit shift. `FINE_CELLS_PER_LY: i128`, the double-width integer the light-year forced, is deleted.
- **THE LIGHT-YEAR WAS DOING TWO JOBS UNDER ONE NAME** and now has its own module. It WAS the coarse step,
  so one constant meant both "how far apart stars are" (a physical fact) and "how a galaxy counts" (which
  must be a power of two, and a light-year is not: `9_460_730_472_580_800 = 2⁶ × 147_823_913_634_075`).
  That non-idempotence was inert only because nothing produced a coarse position. Re-valuing the rung
  retires the hazard rather than documenting it. `vd_core::units::LIGHT_YEAR_M` is stated as its IAU
  derivation, not its digits, and kills a duplicate in `vd-seedsearch.rs`.

- **★ THREE FALSE CLAIMS FOUND IN THE TREE, each load-bearing where it sat.**
  1. **`pose.rs`: *"the edge is a compile-time constant, never serialized, so this moves zero bytes."***
     False the moment a step moves: the set of steps is folded into the saved-data label AND the protocol
     version, precisely so a build counting in different units cannot silently read another's positions.
     Re-valuing a step is a flag day BY CONSTRUCTION. Retired with the reason written out.
  2. **`frame.rs`: *"a tier change still routes through the ONE existing re-quantizer."*** It routes the
     OUTPUT, not the arithmetic. `transfer_frame` does its ENTIRE add-and-subtract at the SOURCE rung while
     the placement book's anchors are authored at a fixed rung (`FramePlacement::moving` hard-codes it), so
     the moment those differ a distance counted in one unit is added onto an anchor counted in another —
     and re-stating the finished result cannot repair arithmetic that already happened. Now a LOUD refusal
     (`FrameError::CrossTierCrossingNotBuilt`), and the final re-statement is DELETED as provably the
     identity. `PLACEMENT_TIER` makes the constructor and the guard agree by construction instead of by a
     doc comment.
  3. **The plan AND `galaxy_sky.md`: *"crossing up is exact in the whole-number part."*** **IT IS NOT**, and
     the counterexample is reproduced in the test: `2047 × 2⁻¹⁰` plus the largest sub-cell offset has an
     exact value a hair BELOW two metres and rounds to EXACTLY two metres — one whole galaxy step — so the
     mandatory re-bucketing carries and the whole number moves by one. **Density: 1,600 hits in a
     3,200-case directed sweep, 0 in 200,000 random draws.** A property test passes while the gate lies.
     The corrected gate asserts the POSITION, never the cell.
- **THE RESIDUAL BOUND: FOUR SOURCES, THREE WRONG — INCLUDING MINE.** `galaxy_sky.md` says 2⁻⁶³ (it applied
  the fine rung's granularity to a coarse rung's residual). The plan says 2⁻⁵¹ for everything, which is
  unsatisfiable at the universe rung by 4096×. **I told the owner 2⁻⁵¹ as "verified" — I had used the
  binade above the value instead of the one containing it.** The bound is PER RUNG: half an ulp of the
  destination step, `step · ε / 4`. MEASURED tight and attained: Fine→Galaxy 1.110223e-16 = 2⁻⁵³,
  Galaxy→Universe 1.818989e-12 = 2⁻³⁹.
- **THE OVERFLOW, AND WHY THE REFUSAL IS THE NORMAL CASE.** Scaling an absolute galaxy cell down to
  millimetres reaches `2⁷³` against an `i64::MAX` of `2⁶³` — over by **1024×**. Only the innermost
  **0.098%** of the domain is refinable at all, so `BeyondReach` is not a corner case, and the bound is
  DERIVED (`(CELL_DOMAIN_MAX − (ratio − 1)) / ratio`) and driven one cell either side.
- **THE FENCE ASKS EACH LEVEL IN ITS OWN STEP**, and judges the galaxy's shell as well as the root's
  (1.00134× of slack — thin enough to be a real question). Applied to the universe's own `2⁷⁶ m` shell with
  the old fixed step it would have refused by **2²⁵ = 33 million times** a world that fits its own lattice
  EXACTLY; a fence that refuses a lawful world is worse than none, because the refusal looks authoritative.
  It also **failed OPEN on a shell that is not a length** — `NaN` is neither greater than nor less than
  anything, so it sailed through the one comparison and was reported as a representable world with a `NaN`
  occupancy. Both arms now refuse.
- **THE ROOT RADIUS IS ITS OWN DERIVATION**, not the literal `2_251_799_813_685_248` with a comment
  explaining where it came from. A comment cannot follow a change. It is the fence solved at equality —
  `2⁶¹` cells at whatever the step is — so occupancy is exactly one half and headroom exactly one octave at
  EVERY rung by construction rather than by three coincidences. It reproduces the old literal TO THE BIT.
- **TWO PLACES WOULD HAVE TURNED "I CANNOT SAY" INTO "DEFINITELY NOT."** The containment scan safe-degrades
  every frame refusal to "not a member", which is right for the refusals it was written for. A CROSS-UNIT
  refusal is a different thing, and degrading it to a definite no would hand an occupant to the wrong realm
  silently — the one failure this whole mechanism exists to prevent. The degrade is UNCHANGED (a wrong
  answer is not improved by a panic); the reason is now COUNTED (`cross_tier_refused`) instead of invisible.

- **★ THE FLAG DAY PROVED ITSELF MID-SLICE, on the first real unit change.** A leftover store refused to
  open: *"this file's positions are counted in different units than this build uses (coordinate generation
  2786608213895069291 against 16303737421466688120)."* Built in S1 for exactly this moment; it fired
  correctly, before a single position was read. **It said no without saying how to proceed** — now it names
  both options (delete, or allow genesis) and states that everything held is discarded either way, matching
  the owner's "refuse, don't convert" ruling.
  - My first three theories for that failure were all wrong (the machine, my own guard, the transport tag).
    It failed at load 1.50 on a quiet box, and the process output named the cause in one line. **Read the
    output before reasoning from the symptom.**
- **WHAT THE COVERAGE GATE CAUGHT IN MY OWN TESTS, all five fixed rather than exempted:** an
  `assert!(a && b)` short-circuit; a failure message computed lazily so its region never ran on a green
  test; a `match` with an unreachable `panic!` arm; a `const fn` that only ever executes in the compiler
  (invisible to coverage is not the same as covered); and an "exact" comparison built in floating point on
  BOTH sides, so it measured **zero error everywhere and passed** — a bound nothing reaches passes for any
  implementation, however loose.
- **DELIBERATELY NOT DONE — S9's, each for a reason.** The galaxy and universe cannot own anything yet: no
  new realm-id arms, no galaxy seed on the frame, no `UniverseSpace` frame. `convert_tier` therefore has
  **zero production callers** after S8, which is correct — the ladder is built, not climbed. The radii do
  not move (moving `ROOT_TIER` and moving the radius are the SAME change and must land together, or a 50%
  occupancy silently becomes 0.0015% and the fence stops saying anything).
- **GATES:** workspace **2265 passed / 0 failed**; Tier-A coverage **100 % region and branch, 0 real
  misses**; fmt and clippy clean at `-D warnings` (verified by exit code).

### D-S9 🟩 THE CLIMB — LANDED (slice S9). The galaxy and the universe become real places, and the limit on a star moves house

- **WHAT S8 BUILT AND S9 USED.** S8 landed a three-rung ladder and deliberately did not climb it: every
  realm still sat at the fine rung, so the world was the same size it had always been. S9 moves the two
  ambient levels onto their own rungs. **Nothing below a star system moved, bit for bit.**
  - The **UNIVERSE**: 2⁵¹ m → **2⁷⁶ m ≈ 7.99 million light years** (×33,554,432). Same fence, same
    construction — 2⁶¹ cells either side of the origin — read with a 32,768 m ruler instead of a millimetre one.
  - The **GALAXY**: `R_universe − a band` → **2⁶² m = 487.46 light years** (×2,051). The subtraction is gone.
    It existed only because the galaxy had no lattice of its own and had to leave room inside the ROOT's
    storage budget; S7 recorded it as a single-lattice artefact due to die here, and it has. The galaxy's
    radius is now its OWN fence solved at equality, exactly as the root's is.

- **★ THE RESULT WORTH QUOTING, AND IT IS A MEASUREMENT.** At the target census of 150,000 systems the
  galaxy affords a mean spacing of **4.4595 light years** against a real mean stellar separation of
  **3.8926** — the world sits at **0.8729× true stellar density, with no compression factor at all.**
  Before the climb the same census afforded 0.00145 ly, 690× inside the owner's one-light-year floor.
  Pinned in `the_target_census_lands_near_true_stellar_density`, with the pre-climb radius kept as a
  driven control so the claim can fail.

- **★ THE LIMIT ON A STAR'S MASS MOVED HOUSE, and this is the finding, not a number.** Two things can stop
  a star existing: the galaxy may have no room to place the system around it, or that system may be too
  wide to state its own positions on the millimetre lattice it counts in. Until S9 only the first could
  ever bind, so only the first was ever asked. The climb reverses it:
  - the galaxy's purse went from **7× SHORT** of a 120 M☉ star to **63.4× headroom**;
  - the system's own lattice is **10.775× over** at that mass, and now binds;
  - the derived cap therefore rises **16.360 → 30.745 M☉** and sits EXACTLY on the fine rung's fence
    (`target_system_bound_max_m() == root_radius_at(Tier::Fine)`, to the last bit).
  - **EVERY star class is kept** (O from 16 M☉ down to M) — solved with our own bisection, not the design
    run's fitted exponent, and cross-checked by a SECOND solver written independently inside
    `the_star_cap_after_the_climb_solved_with_our_own_solver`. The two agree bit for bit.

- **THE FENCE AND THE SOLVE HAD DRIFTED APART, and the climb exposed it.** `affordable_at` learned the
  lattice constraint; `guard_galaxy_affords_its_stars` did not. After the climb the fence would have PASSED
  a 120 M☉ star the solve refuses at 30.745 — a fence guarding a door the world no longer uses, whose
  silence reads as approval. Both now read ONE definition (`binding_limit`), which also names WHICH limit
  bound (`StarLimit::GalaxyPurse` / `SystemLattice`); both arms are driven.
  - **AND THE LATTICE BECAME AN ARGUMENT, for the same reason the budget did in S7.** With the budget
    alone the fence's Ok branch was unreachable — no galaxy radius makes a 120 M☉ system fit a millimetre
    lattice — so it would have been dead code wearing a green light (HR5). Parameterised, the question
    "what would it take?" can be asked, and the answer is pinned: **four octaves of the fine rung**
    (a millimetre step becoming 16 mm). That number is P10's size.

- **★ THE 2048× CONFUSION HID IN A DOZEN PLACES.** A region's `center` is its position in its PARENT's
  frame; its `frame` is its OWN. While every realm shared one rung the two were interchangeable, and a
  dozen readers had quietly assumed so. The body generator's centres were fixed first (a 3.07e18 m vs
  2.25e15 m nesting refusal named it); the rest surfaced one at a time as each suite was brought up:
  the walk band-crossing measurement, the child-index lookup measurement, the pre-generator golden,
  the sim story's `centre` and its metre helper, the union-over-draw oracle, the window-hop round trip,
  the cross-crate frame fixture, and three test writers that stamped every pose in millimetres whatever
  realm it stood in.
  - **★ ONE OF THEM WRITES THE GOLDEN VECTOR.** `one_containment_answer` flattened each placement with
    the CHILD's step. Left alone, this slice would have blessed the galaxy's rows into the world's own
    pinned bits 2048× wrong — not a failing test, a wrong fact recorded as the reference.
  - **THE LESSON:** a test that reads the wrong ruler does not fail. It measures a world that does not
    exist and stays green — and when the ORACLE and the FIXTURE use the same wrong ruler they agree
    with each other perfectly, which is what the union-over-draw case looked like right up until the
    real fold disagreed. Every one of these was invisible until the rungs actually differed.
  - **★ THE DURABLE CURE IS BUILT: `ParentCentre` (S9, after the pins).** Fixing sites by hand does not
    stop the next one, so `RealmRegion.center` is no longer a bare `LatticePos`. The bare `delta_m` is
    gone from that path; the only way to metres is `metres_in(parent)`, which takes the PARENT REGION and
    reads the step off it, so the child's cannot be passed. `metres_in_forest` /
    `RealmRegion::centre_m` do the lookup and answer `None` when the parent is absent — **the step is
    then unknowable, and the `map_or(child's own tier)` fallback that used to stand there was the defect
    written down as a default.**
  - **★ THE TYPE FOUND THREE MORE THE MOMENT IT COMPILED — 14, 15 and 16.** One was in SHIPPED code
    (`vd_bins::world_roster`, flattening a galaxy-authored centre with the sibling's own step). The
    sixteenth is the most instructive: it compared a pose against a centre with BOTH sides read at the
    wrong step, so two errors of the same size cancelled and the comparison gave the right answer for the
    wrong reason. **Sixteen sites, and only one of them ever failed a test.**
  - **THE WIRE COSTS NOTHING, AND IT IS MEASURED:** `#[serde(transparent)]`, with a test that encodes
    both forms and asserts equal bytes AND that the wrapper decodes from what the bare form wrote — the
    direction that matters for a `regions.json` already on disk. An attribute quietly removed is a silent
    on-disk break, so it is checked rather than trusted.
  - **THE FENCE ITSELF IS MEASURED, not claimed from the compiler:** a FINE child inside a GALAXY parent
    reads 100 m through `metres_in`, and the same bits read at the child's step give 100/2048 — the exact
    ratio that hid in sixteen places, asserted as a control.
  - **What it does NOT cover:** a pose written in the wrong step (three test writers did that this
    slice). A pose carries its frame, so the honest cure there is to stamp from the frame, which those
    writers now do — but nothing yet stops a fourth.

- **★ THE MASS CAP IS A LAUNCH-BLOCKING INPUT, NOT A TUNING KNOB — stated here because S9 is the third
  time it has re-rolled the whole world.** `sample_imf_mass` inverts a BOUNDED power law and the cap is
  its bound, so moving it moves EVERY star in EVERY world: 120.0 (literal) → 16.360 (galaxy-derived) →
  30.745 (lattice-derived). Each move re-pins the photometrics golden, the earth-like candidate, seed 0's
  home shell, and — because the nest sweep is sized from the IMF tail above the cap — the sweep itself
  (284 → 664 worlds), and with it every count derived from it. **The P10 lattice lift moves it a fourth
  time, and that lift must land BEFORE the seed freezes.** After the freeze, moving it re-draws every star
  every player has seen.

- **WHAT S9 DELIBERATELY DID NOT DO.**
  - **THE SHAPED PLACEMENT IS S12's** (owner agreed 2026-08-26). The census is still three systems on a
    ring. The owner's shape rulings — a real galaxy shape drawn from the seed, never evenly spaced,
    seed-permanent placement — are recorded in `owner_decisions_2026-08-24.md` and bind when it lands.
    ⚠ The S9 design run's headline numbers ("the world caps at 12.57 systems", "28× inside the floor")
    are DISPROVED; the cap is ~674,000 and the spacing was never the problem.
  - **χ (the compression ratio) IS RETIRED IN MEANING, not just re-pinned.** It divides the real stellar
    separation by the PLACEMENT RADIUS, which measures compression only while every system sits on one
    ring at that radius. It reads 0.00799 now, i.e. "stretched 125×", which is true of the ring and says
    nothing about the world. Its successor is the density measurement above. Delete χ at S12.
- **★ THE CROSS-RUNG CROSSING HAD TO BE BUILT, AND WAS — the largest single piece of S9, and it was not
  on the plan.** S8 refused every crossing between two rungs and named S9 as the slice that would make the
  arithmetic rung-aware. I first ledgered that as P10 work on the reasoning that "nothing in the game does
  that yet". **That was wrong, and the dual-cluster crossing test proved it in one line**: a dot leaving
  its home system for the galaxy IS a cross-rung crossing, and it is the most ordinary journey in the
  game. While it was refused, nothing could enter or leave the galaxy at all.
  - **THE FIX IS ONE SENTENCE:** the arithmetic happens at the BOOK's own rung. A placement book's rows
    are placements its ANCHOR authored in the anchor's frame, so the anchor's rung is the unit they count
    in — not a constant, and not the source pose's. The pose converts INTO it on the way in and OUT of it
    into the destination's rung at the end; between those points every quantity is in one unit, which is
    the property S8's blanket guard was standing in for.
  - **`Separation::convert_tier` is the new primitive**, and a difference needed its own rather than
    borrowing the position converter: a difference is routinely negative and is not sub-cell, so the
    position converter would have refused it as "not a lawful position" for the ordinary case of pointing
    backwards. Coarsening is total (the remainder folds back into the residual); refining is exact and
    refuses past the finer rung's reach.
  - **SAME-RUNG IS THE BIT-FOR-BIT IDENTITY**, measured over every rung × cell × residual case including
    negative, non-normalized and larger-than-a-cell residuals — so the shipped path pays nothing.
  - **WHAT STILL REFUSES, AND SHOULD:** a distance too wide to count in the finer unit. A star system
    counts in millimetres, which reach a quarter of a light year, and its galaxy's centre is 0.7 away.
    So a system CANNOT state its parent's position in its own unit — measured, and pinned as a count:
    four of THE world's window-hop inversions refuse, and they are exactly the two RING systems at both
    sampled instants. The HOME system is the exception, because it sits at the galaxy's origin and zero
    has a count in every unit. **The galaxy's origin is the only place in it a millimetre-counting realm
    can name its parent from.** That is P10's trigger, and it is now a number rather than an argument.

- **★ THE STAND-INS ARE GONE, AND THE LAST ONE WAS ON THE LIVE PATH.** `RealmLevel::to_realm_id` still
  lowered a Universe to `System(0)` and a Galaxy to `System(1)` — its own doc said "Slice 4 flips them to
  dedicated arms", and S9 is that slice. It was the last reader handing them out, and it did so where a
  dot leaving home asks where it is going: the answer was `System(1)`, no shard held `System(1)`, and the
  dual-cluster crossing looped `CROSSING UNRESOLVED … dest_head_missing=true` until it timed out.
  - **THE LOWERING IS NOW LOSSLESS FOR THOSE TWO:** a galaxy keeps its seed, a universe is fieldless.
    Two collisions that used to be pinned as known-lossy are pinned as impossible instead.
  - **AND IT REMOVED A REAL FOOT-GUN:** a shard asked to host `Galaxy(2)` used to silently boot THE
    world's galaxy, because every galaxy lowered to the same id whatever its seed. It now finds an empty
    forest and refuses.
  - **ONE COLLISION REMAINS AND IS UNCHANGED:** the same system seed in two different galaxies still
    lowers identically. `RealmCoord::path` is still the collision-free key at multi-galaxy scale.

- **HOW A HAND-TYPED TOLERANCE FAILS.** Two identity checks carried a literal `1.0` m chosen as "a few ulp
  at 2.25e15 magnitudes". Those identities cancel two GALAXY-scale numbers (spacing 1024 m) to leave a
  system-scale result, so after the climb they missed by 512 m and 286 m — exactly half an ulp of the
  largest term, and not faults. Both now derive the bound from the largest magnitude in the chain. A
  literal cannot follow a change of scale; it can only be re-typed after each one breaks it.

### D-S10 🟩 A PARENT'S COST STOPS GROWING WITH ITS CHILDREN — LANDED in part (slice S10)

**Five mechanisms were named. Three landed, one is deferred by owner ruling, one is recommended closed as
not-a-defect. Plus the file split the slice named as its first task, and one deliverable that turned out to
belong to S11.**

- **THE TEST FILE SPLIT (the slice's stated first task).** One 16,139-line module became a 2,388-line
  fixture file plus TEN subject modules. **Only the test FUNCTIONS moved** — the mapping found 89 helpers
  that would have had to stay behind, and caught one a reader had confidently called local that a
  whole-file search showed used in three other subjects. Moving no helper means no judgement about which
  fixture belongs where, and that error class cannot arise. The cut was proved LOSSLESS by reassembling
  the original file from the pieces before anything was written; the 296 test NAMES are diff-identical.
- **THE LINEAGE LIST LEFT THE HOT PATH.** Both AoI latches were keyed by a child's full `RealmPath` — a
  heap list CLONED TWICE PER (observer, child) PAIR every tick, 300,000 allocations per tick at the target
  census, to answer a question about identity. Keyed by `RealmId` now. The proof it changed nothing: 499
  tests untouched, and `RealmPath` became an UNUSED IMPORT in both files.
- **A DISTANT LOOKER WAKES NOTHING (M7).** The interest signal carried a yes/no, so a realm receiving it
  woke EVERY child — 150,000 realms and 150,000 messages per tick, by construction, unfixable by any loop
  shape. It carries the looker's DISTANCE now (no direction: SL2 forbids a pose crossing, and proximity
  needs a direction while SIZE needs only distance). Each child's own visibility band decides. Gate proved
  RED under the old rule.
- **THE KEEP-ALIVE BASELINE IS A DIGEST.** The body baseline kept the WHOLE bag per child — a heap vector
  per direct child per subscriber, compared byte-by-byte every tick, all freed and re-allocated on every
  keep-alive beat by design. The relay baseline one field below already solved this exact problem this
  exact way; there is now ONE digest definition in the crate. ⚠ This trades EXACT comparison for
  probabilistic: a 64-bit collision defers one re-send to the next keep-alive beat, which is the same
  argument the relay lane already relies on. The half nothing pinned — that a genuine change still ships —
  is now gated and proved red.
- **★ THE PER-TICK SKY IS GONE — and this deliverable belongs to S11, not S10.** See below.
- **THE FOLD INVERSION IS DEFERRED TO S12** (owner ruling; see D-MOVE-1 / M8). MEASURED: only 8 of 499 sim
  tests catch a fold that silently skips a child, and the widest AoI fixture has TWO children.
- **THE CLIENT'S PER-DELTA CLONE: ★ CLOSED AS NOT-A-DEFECT (owner, 2026-08-27)** — *"we can close it for
  now, we can re-review when the real game will be tested if we see problems."*
  - **WHY IT IS NOT AN SL9 DEFECT:** it is bounded by the DRAWN set, not the roster. The update feeding it
    already carries only the realms entering and leaving the view, so the cost grows with the SKY, never
    with the galaxy's child count.
  - **THE COPY IS LOAD-BEARING, TWICE OVER, and this is why it must not be "tidied away" later.** (1) The
    renderer may be reading the scene at the instant an update arrives — the scene is shared behind a
    reference count, so copy-then-swap is what makes a concurrent read safe. (2) A malformed delta (a
    cycle in the parent links, say) must leave the LIVE picture untouched rather than half-applied; the
    new copy is simply discarded. Mutating in place forfeits both.
  - **★ THE REOPEN TRIGGER, owner-stated:** live play of the real game showing a problem. **What to measure
    when that happens — and measure it BEFORE changing anything:** how large the DRAWN box map actually
    gets with a full sky, and the per-update copy cost against it. If that number is genuinely large the
    answer is a persistent map with structural sharing (two changed entries cost two, and the previous
    version stays valid for whoever is mid-read) — which is a NEW DEPENDENCY and therefore an owner
    decision, never adopted alone.
  - **Until then: do not remove the copy.** Removing it without the sharing map silently forfeits both jobs
    above, and neither failure shows up in a test — one needs a concurrent read, the other a malformed
    delta.

---

### D-S11-SKY 🟧 THE SKY IS STATED ONCE — the split LANDED early (in S10); the rest of S11 is owed

**LANDED (2026-08-27, owner-approved under SL6):** the author's direct children are split by **whether they
move**. Static rows ride a new reliable arm (`ShardToGateway::WindowStaticRows`, wire index 11) send-on-
change — which for a realm whose children do not move is exactly ONCE plus the keep-alive repair. Movers
keep riding the per-tick frame, where level-triggered repetition is still the right loss story. The frame
still arrives every tick because its STAMP is what the composer aligns chains on; the saving is the rows,
never the beat.

**MEASURED, on the wire (an earlier 31.2 MB figure was `size_of`, not the encoded size — retired):**

| | |
|---|---|
| one `RealmSnap`, postcard-encoded | 95 bytes |
| the full roster at the census | **14.2 MB** |
| at 20 Hz | **285 MB/s per subscriber** |
| a star system's 9 planets | 855 bytes/tick = **17.1 KB/s** |

**The star system was never the problem. The galaxy was, and only because it repeated something that never
moves** — pinned by the world's own `placement_rows.golden` across ticks 0, 1,000 and 50,000.

**THE TWO LANES RE-UNITE AT THE GATEWAY INGEST, deliberately.** Three consumers read a level's rows and
expect the FULL roster (the compose fold, the marker admission's roster vouch, the confirmed gate). Teaching
each about a second source would be three chances to get it wrong — and ONE OF THEM FAILS SILENTLY: the
marker vouch would refuse every static child's marker as unvouched, only at real scale, because every
current fixture contains a mover. Re-uniting at ingest keeps the saving on the WIRE, where the problem was.

- **★ THE GATE AS WRITTEN IS THE ONE S11 WARNS ABOUT.** S11's own text: *"Both gate on 'per-tick sky bytes
  are ZERO after the first tick' — which is also exactly what a dropped oversize datagram produces. The gate
  as written passes on the failure it exists to prevent."* **That is precisely the shape of
  `a_static_roster_is_stated_once_and_the_frame_keeps_only_its_stamp`.** It is sound TODAY (two rows cannot
  be oversized) and it becomes a false green the moment the census makes the message droppable. **The
  replacement S11 names: gate on the RECEIVED catalogue — the client's rendered star count equals the
  census — never on the absence of bytes.**
- **★ THERE IS NO STAR-MAP MODE — corrected by the owner 2026-08-27:** *"we don't have any mode, we should
  just point our ship to the star and engage warp."* SL8 forbids a map screen, a toggle or a separate view.
  The stars are drawn IN THE SKY, always, and aiming is a PHYSICAL act — you rotate the hull. **The
  catalogue exists to DRAW the sky, and for nothing else.** Any S11 work that grows a selection UI, a
  destination list or a map view is building the seam the law exists to prevent.
  - ⚠ **AND THIS MAY MOOT PART OF Q4's CONDITION 4.** That condition guards against an edited cache
    becoming *"warp anywhere"*, which assumes a STATED destination the server could be tricked into
    accepting. Under physical flight there is no stated destination — you point, you burn, and the demand
    loop warms whatever you actually approach from your own velocity. An edited cache would then draw a
    star that is not there and fly you to empty space: a lie told to that player's own screen, not an
    exploit. **Whether any destination is stated at all is not settled here — settle it before building
    the validation half.**
- **★ THE CATALOGUE ROW'S SHAPE, measured 2026-08-27 before building:**
  - **A star's position is EXACTLY CELL-ALIGNED** — every generated system's centre carries `offset =
    (0,0,0)` at the galaxy's own 2 m step. So a row needs the integer cell and NO sub-cell part.
  - Costed against the current row: **~47 bytes vs 95**, i.e. **7.0 MB vs 14.2 MB** at the census.
  - **THE ROW MUST CARRY WHICH SYSTEM IT IS, and the reason is a SEAM (SL8), not tidiness.** Fly toward a
    star and it wakes: the client then holds a catalogue POINT and a LIVE REALM for the same star. Unable
    to tell they are one thing, it either draws both (a double image) or swaps them (a POP at arrival).
    Both are seams. The identity is what lets the point be suppressed the instant the real thing arrives.
- **★ THE LIVENESS BEAT — a new wire arm, APPROVED under SL6 and LANDED (owner, 2026-08-27).**
  `ShardToGateway::StarSkyAlive` (wire index 13) and `ServerControlMsg::SkyAlive`, carrying one number:
  the generation the server believes is current, restated on the AoI cadence whether or not it changed.

  **THE PROBLEM IT REMOVES.** Every other statement on this lane is send-on-change, and a galaxy does
  not change — so a perfectly healthy server is SILENT there essentially forever. Silence therefore
  carried no information at all, because it was the signature of two opposite states at once:

  ```text
    the sky did not change   ──┐
                               ├──► ...both look EXACTLY like this: nothing arrives.
    the emitter is broken    ──┘
  ```

  The per-tick frame did not settle it: a frame proves the SHARD is running, never WHICH SKY it holds,
  so a shard whose catalogue emitter died would go on framing forever. The beat makes the healthy case
  audible, and the three readings separate:

  ```text
    beat arrives, generation == mine   ──►  Current   "nothing changed"   (healthy, and PROVEN)
    beat arrives, generation != mine   ──►  Stale     "I hold the wrong sky"
    no beat at all                     ──►            "nobody is working"
  ```

  Eight bytes on a cadence against a 7.0 MB catalogue — the cheapest possible statement that the
  expensive one is still true. **UNCONDITIONAL**, alone on this lane: it is the one statement that must
  NOT be suppressed when it repeats, because the repetition IS the message. Gated only by the same
  emptiness guard as the catalogue (a shard parenting no stars has no sky to vouch for). Per GATEWAY,
  never per window — the sky is not window-scoped.

  **STILL OWED (with the renderer):** the third arm is an ABSENCE, and absence is not an event. The
  client counts `beats_current`, and a caller holding a clock reads "stopped climbing" as *"nobody is
  working"* — but the watchdog POLICY (what the player is shown when the sky lane goes quiet) is a
  renderer decision and is not built. Deliberately, the beat also does NOT act on `Stale`: knowing the
  held sky is wrong is a different job from replacing it, and replacing it is the
  receiver-states-its-generation exchange below, not this.
- **★ A CLIENT DOES NOT RECEIVE THE WHOLE GALAXY FROM A SYSTEM SHARD — MEASURED 2026-08-27, and it is
  an S12 question, not a defect in this lane.** A shard folds its sky from the realms IT booted, so a
  cluster hosting one star system states that system's stars and no others. The real-process gate went
  red on its first run with the WORLD's census written into it: **1 star held against 3 in the world.**

  ★ **CORRECTED, AND CLOSED, 2026-08-27.** The entry above said three answers were possible and NONE
  was chosen. That was wrong: one had already been ruled the same day — *"we're passing the Galaxy just
  once over reliable lane"*. The assistant re-opened a settled question because it reasoned from the
  code in front of it and found a gap, instead of reading the ruling. That is the second time in one
  session; see the movement law for the first.

  **THE RULING RULES OUT TWO OF THE THREE.** A client also subscribing to the galaxy shard is the galaxy
  crossing PER SUBSCRIPTION, not once. A gateway composing the sky per observer is the per-observer
  product the lane exists to remove. The third — stated once by a party holding the whole galaxy — is
  the one that matches, and is what LANDED.

  **WHAT LANDED.** The GATEWAY folds the galaxy at boot, over the whole world, and states it to a client
  that has not confirmed that generation. No shard states a sky. The shard's catalogue, request set and
  two emitters are deleted, and so is the gateway's forwarding of them — dead code that tests keep
  exercising is a false green. `ShardToGateway::StarCatalogue`/`StarSkyAlive` and
  `GatewayToShard::SkyRequest` stay on the wire (deleting a variant renumbers every later one) and are
  COUNTED, so a binary that has not caught up is visible.

  **HR4:** the sky is no longer a shard feature, so there is no shard kind to vary and the rule is
  satisfied by construction. Owner-confirmed 2026-08-27. The three-profile gate went with the shard
  lane; the behaviours it protected are re-tested at the gateway.

- **★ HR4 G-IDENTICAL FOR THE SKY — LANDED 2026-08-27 (owner-confirmed).**
  `assert_sky_feature_anywhere` drives the identical sky lane on THREE profiles — galaxy (relay, no
  voxel realm), system (hull host, no voxel realm) and planet (a SPHERICAL voxel realm with functional
  blocks, block edit, surfaces and seats) — and asserts the parts, the generation, the beat and all
  three counters are byte-equal.

  **THE HONEST LIMIT, stated in the test itself:** nothing in the sim READS a capability yet, so this
  gate cannot go red today. It is a TRIPWIRE armed for the day capabilities start gating behaviour —
  which is the day the sky could quietly acquire a dependency on one. The 2026-08-14 audit found the
  previous G-IDENTICAL gate green on a fixture that ran the SAME kind twice (D-38), so this one asserts
  the three profiles differ, and that the planet run really does host a voxel realm, before it compares
  anything.

- **★ THE CLIENT'S CACHE AND THE GENERATION EXCHANGE — LANDED 2026-08-27.** The two arrived together
  because they are one design: the cache is what gives the client something to state.

  **THE CACHE** is a postcard file split into a header and the rows. A header that reads alone can be
  DISBELIEVED alone, so the claimed star count is tested against the cap before one row is decoded —
  decoding is exactly what a hostile file wants to buy. It carries `StoreRole::ClientCatalogue`'s stamp
  (seed, epoch, coordinate generation, world generation, all reused from `store_stamp::verify` rather
  than re-stated) and the catalogue generation. Refusals are named and COUNTED: `TooLarge`,
  `Undecodable`, `ForeignWorld`, `TooManyStars`, `CountMismatch`, `Unrepresentable`, `DigestMismatch`.

  ★ **THE FOLD IS THE ONLY TEST THAT CATCHES AN EDIT**, and this is MEASURED rather than argued: every
  single byte flip in the row payload reaches the fold, and NOT ONE is turned away by a test above it.
  postcard is positional, so a flipped byte does not fail to decode — it decodes into a different,
  perfectly well-formed sky, with the stamp intact, the count unchanged and every row drawable. A
  refused cache is counted and leaves a good sky alone: a player already flying does not lose the
  galaxy because a stale file on disk was bad.

  **THE EXCHANGE** deletes the sender's memory at both hops rather than patching it:

  ```text
    client  --SkyHeld{generation}-->  gateway     "this is the sky I have"
    gateway --SkyRequest---------->  shard        "somebody here needs one"
    shard   --StarCatalogue-------->  gateway     answered, and NOT remembered
    gateway --StarCatalogue-------->  only the clients whose held generation differs
  ```

  The shard now keeps NO record of who holds what — `SkyStatedTo` is gone, replaced by `SkyRequests`,
  a set that is answered and cleared in the same tick. The holder's identity lives on the SESSION,
  which is the only place it is ever one-to-one: one session holds exactly one sky.

  **THE BEAT DRIVES THE ASK**, which is why the two slices had to land in this order. The beat is what
  tells the gateway which generation is current, so it is the only moment the gateway can know a client
  holds the wrong one. That rate-limits asking to the beat's cadence with no timer and no
  acknowledgement: a lost catalogue is re-asked at the next beat, and the exchange repairs itself. It
  stops the moment every client confirms.

  Answering also moved OUT of the window guard. Under the old memory the emitter ran only while a
  window was open, and that coupling was load-bearing by accident — see the finding below.

  **STILL OWED:** the file I/O itself. The codec is pure and the decision is fully driven, but nothing
  reads or writes the file yet, because the client has no persistence seam. Also owed: the watchdog
  policy for a sky lane that goes quiet, and one instanced point cloud instead of 150,000 objects.

- **★ THE SENDER'S MEMORY CANNOT BE MADE CORRECT — MEASURED 2026-08-27. FIXED the same day by the
  exchange above; kept because the REASON is what binds future work.** The emitter remembers which sky it last stated to each gateway. That memory is
  wrong in BOTH directions and the sender cannot tell the two cases apart:
  - **FORGET TOO EAGERLY** ⇒ a client returning from a crossing is re-sent a sky it already holds. A
    warp leg is TWO crossings, so one journey re-transmits the whole galaxy twice — 14 MB at the census.
  - **REMEMBER TOO LONG** ⇒ a NEW subscriber behind the same gateway is told nothing and has NO SKY AT
    ALL, because some previous session once held one.

  Both arms were driven by a test that RECORDED the broken behaviour: the emit ran only while some
  window was open, so a shard whose LAST window closed never reached the line that forgot the gateway —
  **but with a second gateway present it did, and the return leg re-stated the whole sky.** Measured,
  not argued. That test was written to flip when the cure landed, and it did: it is now
  `a_closed_and_re_opened_window_does_not_re_state_the_sky`, and it keeps the second gateway precisely
  because that is the case which broke the memory.

  **No length of memory fixes this, because the sender is not the party that knows.** S11's own wording
  is the cure: *"the receiver stating its generation and the sender answering from that rather than from
  its own memory of what it sent."* It needs the client's on-disk cache to exist first. **Do not patch
  the retain — it would trade one wrong direction for the other.**
- **STILL OWED FROM S11:** the compact catalogue to the CLIENT (nothing sent to realms — a ship holds the
  same generator and asks it); encoded-vs-seed-folded byte identity, one truth from two producers; the
  generation derived from content, never hand-incremented; the client's on-disk cache with a content digest
  and a bounded decode (a byte-flipped cache must produce a counted refusal and a re-request, never a drawn
  frame); the login level CHUNKED under the frame cap — **the paging deliberately not built in S10, because
  at today's 189-byte roster there is nothing to page and transport machinery that cannot be exercised must
  not be written**; the liveness digest as an assertion; and the catalogue NOT re-sent when a crossing bumps
  the scene epoch (a warp leg is two crossings — getting it wrong re-transmits the whole sky twice a
  journey).
- **AND ONE MEMORY HALF THIS DID NOT FIX:** the gateway still holds one MERGED copy of the roster per
  retained tick in its level ring. That is memory, not bandwidth; the wire was the stated problem.

### D-SL9-WALKS 🟥 FOUR PER-TICK WALKS OVER EVERY CHILD — two removed, two owed to S12

**SL9 says a cost that grows with the child count is a defect, and must be MEASURED on a realm with many.**
Four jobs walk every direct child on every tick. Named here with what each costs at the target census of
150,000 children at the 20 Hz cadence, so none of them can look handled when it is not.

| # | The job | Touches | Per second at census |
|---|---|---|---|
| 1 | Which children are worth waking | every child **×** every observer | 3,000,000 |
| 2 | Where each child is (the placement book) | every child | 3,000,000 |
| 3 | What each child looks like (the picture rows) | every child | 3,000,000 |
| 4 | How fast an occupant may fly (the governor) | every child **×** every occupant | 3,000,000 |

**JOB 4 IS NOT A DEFECT TO FIX — it is deleted by the movement law (D-MOVE-1).** Do not optimise it. Do
not delete it in S10 either: the realm ceiling IS the speed today, so removing it now pins every occupant
at foot pace. It goes with the force phase, together with the band re-solve.

**JOB 1's WALK is deferred to S12** (M8 of the 2026-08-26 movement file): the owner refused a hand-planted
wide field, and MEASURED, only 8 of 499 sim tests catch a fold that silently skips a child. What S10 did
land on job 1 is two costs that are NOT the walk: the per-pair lineage clones (300,000 heap lists per tick
at census, now keyed by realm id) and the wake-everything interest proxy (M7).

---

#### ★ JOBS 2 AND 3: THE ANALYSIS, DONE, SO S12 INHERITS IT RATHER THAN A HUNCH

**THE OBSERVATION.** A STATIC child's placement is the same at every tick. THE WORLD'S OWN GOLDEN PROVES
IT: `crates/bins/tests/goldens/placement_rows.golden` carries each static child's row at tick 0, tick 1,000
and tick 50,000, and the bytes are identical. Rebuilding 150,000 identical rows 20 times a second is not
an algorithm that needs improving — it is arithmetic whose inputs did not change.

**WHY "JUST CACHE IT" DOES NOT WORK, and this is the part worth inheriting.** The placement book is not one
table that gets overwritten. **It is a TIME SERIES.** `PlacementLedger` retains a deque of books spanning
`span_back` ticks, because a message can arrive late carrying its own instant and must be measured against
the book that was true THEN — reading the newest one instead would move the subject by however far the
world swept in between. So a book cannot be advanced in place, and each tick genuinely creates a new one
that the old ones must survive alongside.

**THE SHAPE THE FIX HAS TO TAKE:** each tick's book SHARES its unchanged rows with the previous tick
instead of copying them, and holds only its own movers. That is a change to the internal representation of
a core type (`PlacementBook`) that nearly every lane reads.

**THE ASYMMETRY THAT MAKES IT WORTH DOING — and worth doing ONLY for one case:**

| Parent | Children | Movers | Rows written per tick today | With sharing |
|---|---|---|---|---|
| A galaxy | 150,000 systems | **none** | 150,000 | **0** |
| A star system | 9 planets | **all** | 9 | 9 |

A galaxy's children do not move; a star system's all do. So the change buys everything at the level where
the count is large and nothing where it is small — which is the right shape, and also means a measurement
taken inside a star system will show no gain and must not be read as the change failing.

**JOB 3 IS HALF-COVERED ALREADY.** The picture walk feeds two consumers: the window frames, and the relay
FINGERPRINT — which serialises the WHOLE picture every tick purely to compare it with last tick's and
decide whether to re-send. That fingerprint IS S10 mechanism 2. The walk that fills it is not.

**WHY NEITHER LANDS IN S10 (recommended by the assistant, owner agreed 2026-08-26):**
1. The gain is UNMEASURABLE until the census rises — the galaxy holds three systems today.
2. It changes a core type's representation, under every lane in the tree.
3. Neither is on S10's list, and together they are larger than everything on it.
4. ★ The blindness measured this same day — 8 of 499 — came from building what could not be measured.
   Repeating that pattern in the placement path would be worse, not better.

**WHEN:** S12, with the real wide world, where the change can be proved rather than argued.

### D-MOVE-1 🟥 THE MOVEMENT CONTRACT — ruled 2026-08-26, and NEITHER HALF EXISTS IN THE TREE

- **THE RULING** (`owner_decisions_2026-08-26_movement.md`, extending A4 of 2026-08-24). A parent NEVER
  sets an occupant's speed. Two lanes upward from a child:
  - **PER TICK: acceleration + torque ONLY, in the CHILD'S OWN frame** — *what I am doing*. The child
    divides by its own mass before it speaks, because it knows its own mass best. Six numbers, and the
    same six whatever the child is: a ship, a person, a rock.
  - **ON CHANGE ONLY: mass, cross-section, drag coefficient** — *what I am*. A declared property, like
    extent. Never per tick.
  - **DOWNWARD:** the placement the parent authored, stamped, read-only, ONE HOP (SL1 clause 2/4).
  - **VELOCITY NEVER CROSSES UPWARD.** It is half a placement and only the parent writes those (SL1
    clause 3). Velocity is the answer, never the question.
- **WHY MASS CROSSES AT ALL, since gravity does not need it.** The child's mass CANCELS out of gravity
  (`G·M·m/r²` divided by `m`), so a 50,000 kg ship and a 1 kg drone fall identically. It does NOT cancel
  out of drag, which is an outside push: identical hulls at identical speed slow at 0.6 vs 6.0 m/s² for a
  ten-to-one mass difference. Collisions cannot be done in accelerations at all. The rejected alternative
  — the parent states its medium downward and the child computes its own drag — was refused because the
  child would use a velocity one tick old, and lagged velocity under strong drag oscillates.
- **WHAT IS MISSING — BOTH HALVES, measured by reading the tree, nothing executed:**
  - No physics engine. 708 dependencies, no rigid-body solver.
  - Nothing that moves carries a mass, an engine, a thrust figure or a hull size. The only masses describe
    stars and feed closed-form orbit sums.
  - The PARENT side is missing too: every occupant advance passes zero acceleration explicitly.
  - The lane a child's numbers would ride is sealed with no port in it.
  - **The traverse constant is not a ceiling above a speed — it IS the speed.** A stick position is a
    fraction of the realm's own number, and velocity is read back OUT of the step just taken; it never
    accumulates. Remove it with nothing in its place and every occupant falls to foot speed. MEASURED:
    crossing the galaxy (9.2234e18 m) at 15 m/s takes 1.95e10 years — longer than the universe has existed.
- **THEREFORE THIS IS A PHASE, NOT A SLICE.** Hulls, engines, mass, the up-lane, and integration on the
  parent side. An interim was discussed and is NOT ruled: a ship states its own rated cruise speed and
  acceleration and the parent applies them — lawful (an engine rating is not a placement), stops the
  parent choosing immediately, and is not a second mechanism because the parent's side is the same shape
  either way.
- **WARP IS A DECLARED STATE (M6), NOT AN ENGINE.** Reaching a galaxy-crossing speed by acceleration takes
  1.66 million years at 10 g; doing it in 60 s needs 8.71e12 g AND travels 16.2 light years during the ramp.
  So warp changes the MEDIUM's relationship to the ship and rides the same slow lane as mass. **The set of
  declared states is CLOSED and reviewed like a wire arm**, behind a five-test gate whose two sharp edges
  are: it must survive an EMPTY SHIP (kills autopilot destinations), and it must NOT MOVE YOU BY ITSELF
  (kills a velocity in disguise). An autopilot is software pressing the stick — it computes an acceleration
  and uses the ordinary per-tick lane.
- **WHEN:** P5 (physics) at the earliest. It cannot land before the band solve is settled — see the
  warning on D-S6 and D-WORLD-4b.
- **★ STILL OPEN, the owner has NOT ruled:**
  1. **THE ROCK.** The 2026-08-19 ruling — *"the realm's ceiling governs everything the realm contains,
     piloted or not"* — is recorded as landed. The movement law neither names ungoverned things nor
     carries a reversal note, and the SAME file writes an explicit reversal for the placement law, so the
     silence is meaningful. Rocks and shells have no engines to compute an acceleration from and no
     autopilot to arrive them.
  2. **WAKING UP IN TIME.** The safety solve names one test (one tick fits inside the thinnest band).
     A realm also needs time to START, and one that is not running cannot be drawn (SL3) — so at high
     closing speed you arrive at a dark realm, which is a seam (SL8). Either the top speed is the stricter
     of the two, or the interest radius grows with the observer's measured closing speed.

### D-MOVE-2 🟥 THE TEMPORARY CONTROL SEAM — approved 2026-08-31, and it is INTERIM BY CONSTRUCTION

- **THE RULING** (`owner_decisions_2026-08-27_movement_answers.md` M-C, and the owner's approval of the
  vector-only shape on 2026-08-31). There is no character to control, no signal bus and no functional
  blocks, and we still want the whole movement path testable NOW. So the controls attach to the SHIP
  REALM directly: the controls make forces in the ship's own shard, and those forces ride the lane the
  movement contract (D-MOVE-1) already defines. Nothing here is a special case for a ship.

- **★ THE LANE NAMES NO MOVEMENT TYPE, and that is the whole point.** The owner approved the vector-only
  shape by name: *"Agree with vector only, it's way better approach."* Forward, reverse, strafe, climb and
  dive are DIRECTIONS of one push vector; roll, pitch and yaw are directions of one turn vector. There is
  never a new arm for a new manoeuvre, and the hot path stays six numbers however many thrusters a hull
  grows. **DRAG NEVER TRAVELS UPWARD** — the realm holds its own medium and does the arithmetic from what
  the child declared it IS. The rejected alternative (the parent states its medium downward, the child
  computes its own drag) is recorded under D-MOVE-1: the child would use a velocity one tick old, and
  lagged velocity under strong drag oscillates.

- **THE TWO OPENINGS, approved 2026-08-31 under SL6** (default NO; this is the grant):
  - **A — the per-tick lane.** Push (three numbers, m/s²) and turn (three numbers, rad/s²), both in the
    CHILD'S OWN frame, plus whose and which tick. Fire-and-forget: it is effect-free, carries no transfer
    trigger and no authority-gating discrete state.
  - **B — the on-change lane.** Mass, cross-section, drag coefficient, and the CLOSED set of declared
    states (warp). *What I AM*, like extent — never per tick.
  - ★ **B RIDES THE RELIABLE LANE — RESOLVED 2026-08-31, and NO NEW DATA CROSSES.** The worry was real:
    "on change only" plus a lost message leaves the parent holding a stale mass forever, and a stale mass
    is a wrong drag and a wrong impact. A version tag on the per-tick lane was proposed and WITHDRAWN —
    the transport already answers it, and the answer is a CLASS CHOICE, not a new field.
    - `MsgClass::reliability()` already states each class's guarantee, and `Durability` already states
      whether a send survives a restart. The reliable mesh lane is at-least-once across a blip (per-lane
      sequence + retry-buffer replay + a receiver contiguity ledger).
    - **The project already uses this exact split one level over**: the composed scene's membership and
      body changes ride the RELIABLE lane, and the per-tick positions ride unreliable datagrams
      (`crates/wire/src/channels.rs`). The movement lanes are the same problem, one boundary up.
    - **The industry answer agrees.** Unreal replicates changed properties and re-sends until the far side
      confirms, while telling you NOT to send reliable messages every tick. The networked-physics
      literature says the same from the other side: on the hot lane send STATE redundantly rather than
      EVENTS, because a repeated statement heals its own loss and an event sent once does not.
    - ⚠ **THE ONE REAL GAP, and it is a restart, not a loss.** The receiver's dedup ledger is RAM-only, so
      a parent that restarts has forgotten everything. **A child therefore restates its facts ONCE per
      connection**, never periodically. That is redundancy applied where it is needed.

- **WHAT REPLACES IT, and when.** The signal system and functional blocks (P9). A thruster block will then
  publish the force and the seam disappears. **The per-tick lane itself does NOT disappear** — only the
  direct attachment of controls to the ship realm does. What is interim here is the SOURCE of the numbers,
  never their shape.

- **THE ORDER THIS MUST LAND IN, and the reason it is not negotiable.** Today the realm HANDS OUT a speed:
  the stick is a fraction of the realm's own number and velocity is read back out of the step just taken,
  so it never accumulates. Deleting that path with nothing in its place drops every occupant to walking
  pace, and crossing the galaxy at walking pace takes longer than the universe has existed (D-MOVE-1's own
  measurement). **The new lane and the deletion must land in ONE change.**

- ★ **WHO MAY SPEAK ON THIS LANE — settled 2026-08-31 by the owner's question about player realms.**
  The lane carries a child REALM speaking to its parent, and nothing else. **A PLAYER IS AN OCCUPANT,
  NOT A REALM** (SL2: *"CONTACTS ARE REALMS, PEOPLE ARE SEEN"*; `EntityKind::Player` is the first entry
  of the transfer registry). An occupant's push never crosses a boundary — the realm holding it applies
  the push in its own process — so a space walk and a parachute jump need NO new machinery: they are an
  occupant crossing a realm boundary, which the transfer arc already proves under kill-9.
  - **The rule that decides realm-hood: you are a realm when something can be INSIDE you.** A ship is a
    realm to its crew and an occupant of its star system at the same time; that dual role is what makes
    nesting work. A person contains nothing. The day a mech suit carries a passenger, the SUIT is a
    realm and its pilot is the occupant.
  - **The plain cost, beside the law:** a realm is a shard. The port band holds roughly five hundred, so
    a realm per player would consume the world at five hundred players.

- ⚠ **A WEAK ARGUMENT, CORRECTED BEFORE IT SET (2026-08-31).** The first draft justified the generic lane
  with *"a ship speaks upward and listens to its crew, so the lane must serve both directions"*. The
  listening half happens IN THE SAME PROCESS and never touches the wire, so it proved nothing about the
  wire lane. **The real proof is HR4's own gate**: the identical fixture must pass on TWO REALM KINDS —
  a ship inside a star system and a ship inside a planet — and a ship-specific lane cannot pass it,
  because the second parent would need ship-specific code and the ship WOULD know it moved house.
  Recorded because the same shape caused today's galaxy defect: a justification that stayed in the file
  after the ground under it moved.

- **THE ACCEPTANCE TEST** is the owner's own last line in the worked example: a ship crosses from a star
  system into a planet and **sends the identical six numbers to its new parent**. It does not know it moved
  house. One mechanism; any realm can be the parent.

- **STATUS 2026-09-02 — THE KEYS BECAME THE STICK, after the owner flew the hull.** The owner flew the
  first hull in the window and reported three things: the mouse turned the hull about the wrong axes,
  the camera flew around the hull while forces were applied, and the push was too weak to travel inside
  the star system. What changed, all of it INTERIM and all of it deleted with this entry:
  - **The mouse turns nothing on a hull** (`stick_from_input`, `sim/stub/dot.rs`). The look rode the
    movement-shaped turn stick in the wrong slots, so the mouse's yaw pitched the hull and its pitch
    rolled it; and the same look also turned the local camera, so one motion made two turns. Now the
    look is the camera's alone. `A`/`D` are the TURN of the nose (the strafe axis, read as a yaw about
    the hull's up axis; `D` is a negative rotation), `E`/`Q` raise and lower the nose (two action bits
    named in `vd_core::controls`, bound in the shard, never on the client), `W`/`S` and `Space`/`Ctrl`
    push. Example: `D` and `E` together make the turn `[0, +1, -1]` — nose up and to the right.
  - **The strafe axis is not throttled** (`input_map::MovementKeys::axes`): a turn runs at the rated
    rate while the key is held and stops when it is released. An eased throttle would have kept the
    hull turning for seconds after the key was up.
  - **The throttle ladder is geometric** (`input_map::throttle_magnitude`, floor `1e-13`): tier 1 is the
    floor times the rating, tier 10 is the rating, each tier is about twenty-eight times the one below.
    One ladder must serve a docking nudge and a run between stars.
  - **The test hull's rating is on its row, stated by the berth tool** (`vd-build-ship
    --max-push-micro-mps2`, `examples/berth_hull.rs`): ten million million metres per second per second,
    three radians per second per second. FLOWN FIRST at a hundred million (owner, 2026-09-02): the hull
    gained a thousand million metres per second every ten seconds, exactly its rating; the crossing out
    of the star system went through at speed; the planets and the star left the picture and their
    shards went to sleep behind it; and the star field did not move, because the nearest stars sit
    thirty thousand million million metres away. The rating rose five orders of magnitude for the
    between-stars run, and the ladder's floor followed it so tier 1 stays one metre per second per
    second. MEASURED against the world before choosing it: the home
    system is a shell 5.2e12 m in radius, the spawn sits 1.08e10 m from its star, the planets' shells
    are 1.5e8 to 4.7e10 m in radius. Ten seconds at the top tier reaches 1e9 m/s. The number is one
    hull's fact, printed by the tool at berth time; no default shares it.
  - **The pilot's panel** (`draw_pilot_panel`, windowed only): the realm's placement in the galaxy and
    its speed come from the sky anchor the gateway already ships for the star cloud (its velocity is
    the same fold); the stick and the throttle are what the window last sent. NOTHING new crosses the
    wire. The hull's rated push does not reach a screen, so the panel shows the fraction commanded.
  - **The camera lock** (`L`): the eye holds the realm's own axes, in first or third person.
  - **What "test the warp" means today.** Warp is a DECLARED STATE with a five-test gate (D-MOVE-1;
    the movement ruling) and it is not built. What the same machinery gives now is an extreme push at
    the top tiers, which the parent integrates without a ceiling (owner ruling: the ceiling leaves the
    flight path). Containment is SWEPT (slice S5, `candidates_segment`), so a hull that crosses a
    planet's shell inside one tick is caught. UNMEASURED at 1e9 m/s: whether a planet's shard boots
    before the hull arrives — the interest lead is `vel × boot_ticks_p99 × tick`, and the wake-up
    constant is the ruling's own open item.

- **★ MEASURED 2026-09-02, second owner flight, at ~2e14 m/s: A HULL NEVER LEAVES ITS PARENT.** The
  hull left the home system's shell (5.2e12 m) within seconds and no crossing followed: the star
  system went on authoring its placement at 1e16 m from its star, two thousand shells out, and the
  directory still listed the hull under the star system. The containment scan re-homes OCCUPANTS; a
  driven child REALM has no exit crossing. The ledger owes it under D-37 ("Realm/Ship re-home owed"),
  and it is this entry's own acceptance test — *a ship crosses and sends the identical six numbers to
  its new parent* — so it is recorded here too. CONSEQUENCE for the between-stars run: without the
  crossing a second star system never wakes for a hull (a parent demands only its OWN children), and
  the hull flies through it as a ghost. The sky stays right meanwhile: the anchor lifts the hull's
  placement through the star system into the galaxy's frame, and that fold is exact whoever the
  parent is — which is why the parallax test could run before the crossing exists.
  - **NEXT:** the child-realm exit and entry crossing — out of a star system into the galaxy, and
    down from the galaxy into the next star system — on the SAME transfer machinery (HR2/HR3: a ship,
    a station, a moon and a rock cross by identical code). The hull's own shard keeps running through
    it; what changes hands is the AUTHORSHIP of its placement (SL1) and the lane its drive rides up.
  - **PLANNED 2026-09-02:** `docs/design/realm_crossing_plan_2026-09-02.md` — the exterior rides the
    `Ship(id)` key the design reserved for it (transfer_protocol §8), the parent's swept verdict over
    its DRIVEN CHILDREN is the trigger, the old parent converts at the destination's rung, the new
    parent adopts in one tick, the hull is TOLD its lineage once (the one new datum, SL6 ask), six
    slices with gates. Awaiting the owner's answers to the plan's §7 asks before slice 0.
  - **BUILT 2026-09-03, slices 0 and 1** (owner: *"Yes, add the arm and continue"*): the exterior key
    (`DirectoryKey::exterior_of`, `ExteriorAuthority`, `resolve_exterior_head`; MEASURED live: the
    directory holds `ship-of-…` at System 7's node); the driven-child lane in the scan
    (`SubjectLane::Exterior`, the early lead over the request ttl, the self-exclusion); the third
    request arm `ExteriorCrossingRequest` (minor 25); the exterior saga walk (`SagaCtx::exterior`:
    flush → CAS → demote + envelope → promote → done, no gateway). **OWED (slice 2):** the source's
    exterior flush (`on_flush_source` still no-ops on a `Ship` subject), the destination's adoption
    (region insert, driven state seed, berth row), the source's release, the runtime region mutator,
    the store handle in the sim; and the exterior blob (spin) on `SourceFlushed.state` — spin is zero
    across the hand-over until it lands, which is a seam only for a hull turning at the crossing.
  - **BUILT 2026-09-03, slice 2 (unit level):** `stub/exterior.rs` — the source's `flush_exterior`
    (pose + `ExteriorState` blob on `SourceFlushed.state`, drive frozen ⇒ both sides coast), the
    destination's `adopt_exterior` (`RealmRegions::adopt_child` with an incremental `ChildIndex::insert`,
    driven state seeded, exterior lease at the crossing's fence, berth row into the realm's own store —
    `RealmStore`, kept open by the shard bin from now on — and a head read for the hull's node), the
    source's `release_exterior` at the demote, the promote ack, the abort thaw. **STILL OWED:** slice 3
    (the hull is told its lineage, so its drive and facts reach the NEW parent — until then an adopted
    hull sits still, its facts unknown to its new parent), slice 4 (the launch ledger), slice 5 (the
    gateway's chains), the G-IDENTICAL process-level fixture, and the galaxy-scale insert measurement.
  - **BUILT 2026-09-03, slice 3 (unit level):** `stub/lineage.rs` and `InterShardFlow::LineageStated`
    (disc 39, minor 25) — the parent owes an adopted child its lineage from the adoption, states it
    once the child's node is known and again on every head read until the child's facts arrive; the
    child applies it only from the node the directory names as its exterior's holder (held and the
    head re-read otherwise), replacing its coord, re-pointing its up-lanes and its own row, forgetting
    its stated facts so they reach the new parent. STILL OWED: slices 4 and 5, the process-level
    acceptance test.
  - **BUILT 2026-09-03, slice 4 (unit level):** the exterior CAS win notes `(child, to_realm)` on the
    saga runtime; the lifecycle reconciler re-keys the child's demand cell (`DemandLedger::rekey`) and
    the process launcher rewrites the persisted launch intent (`RealmSpawner::reparent_realm`), so a
    restart rehydrates the hull under its true parent. OWED: the child's PEER BOOK (a hull can reach a
    new parent only if it was booked with it — the galaxy is on every hull's ancestor closure, a
    sibling star system is not), slice 5, the process-level acceptance test.
  - **BUILT 2026-09-03, slice 5 (owner: *"implement all"*, the arm asked under SL6 and approved):**
    `InterShardFlow::ExteriorMoved` (disc 40, minor 26; orchestrator → every session gateway; a
    producer-less one-shot pushed `Retained`, the FIFTH such arm). The reconciler states the moved child's
    new coord when it re-keys the cell; `gateway/home.rs::lineage_splice` replaces the ancestry above the
    child on every chain that holds it and keeps the session's own descent; the composer bumps the origin
    epoch on the changed authors and the window derivation moves the child's window. Per-child tick guard
    against a redelivered older statement; three counters on the admin views.
  - **BUILT 2026-09-03, the release cleanup:** `release_exterior` also drops the hull's rows from the
    area-of-interest ledger (watched or watching), the interest latch, the in-band verdict, the luma and
    the lineage owed (`exterior::ChildTables`, threaded through the reply arm's `ReleaseSide`).
  - **★ THE ACCEPTANCE FIXTURE IS GREEN, 2026-09-03** — `tests/tests/hull_crossing_e2e.rs`: six shards on
    the fabric (orchestrator, gateway, System 7, the galaxy, System 8, the hull's own shard, each with
    its REAL profile via `vd_tests::hull_crossing_cluster`). The hull leaves System 7 for the galaxy and
    enters System 8; System 7, the galaxy and System 8 hold the SAME six drive integers in turn
    (G-IDENTICAL: a galaxy and a star system adopt alike); no tick leaves the hull unheld by any parent
    (the ghost-row concern, plan §7 ask 7, measured at the authors); the hull's own coord follows it
    twice (berth 7 steps, out 56, told 2, down 528, told again 2). **Two defects it found and fixed:**
    the descending flush re-validated the destination as an occupant's would and refused the
    early-started exterior as a stale entry (`flush_stale_entry` — the galaxy's hand-down aborted
    silently); and `ship_flight_e2e` had been red since the "at the controls" ruling (a hull with no body
    states no drive) — it now carries a body. Two plant traps recorded in memory (a planted driven child's
    `pos_m` is travel, not the berth; inert interest bands stop the parent head read).
  - **★ SL9 MEASURED AND FIXED, 2026-09-03** (`tests/tests/galaxy_adopt_measure.rs`, in the gate): one
    adopt-and-release of a hull on THE world's galaxy (233 220 systems) cost 23 291 µs against 7.7 µs on
    ten children — 3 028×, a cost that grew with the children. The child index removed a child by
    sweeping every cell; the parent's children list was searched linearly at the release and at the
    swap fix-up; the adoption folded over every child for the widest extent. Now: the index keeps each
    child's span, `children_of` is a set, the radial list is an ordered map keyed by the distance's bit
    pattern, the widest extent is a kept number — 18.6 µs against 5.4 µs, 3.4×, asserted ≤ 25×.
  - **★ THE FIFTH FLIGHT, 2026-09-03 (the owner in the window, two runs) — two defects found on the real
    cluster and fixed:** (1) the reparent note was taken only from the saga step's OWN event, and the
    in-process directory CAS feeds its win back INSIDE the quiescence run — so on the shipped path no note
    was taken, the reconciler never re-keyed the hull, no `ExteriorMoved` went out, the pilot's window on
    the old parent closed at the release and none opened on the new one (the planet vanished, the drawn
    hull froze). `run_to_quiescence` now reports the inside win; the walk test asserts the reconciler's
    count. (2) `ChildIndex::candidates_along` stepped the observer's LEAD LINE one grid cell at a time —
    a hull at warp leads by ~3e11 m, ~7e7 steps per observer per tick — the star's shard ran nine seconds
    per tick, its window went stale, the chain's fresh prefix stopped at the hull and the sky went black.
    A lead longer than the index's occupied cells now asks the occupied cells against the line (slab
    test), the same answer at the index's cost. Also observed: the hand-DOWN into the star's realm (the
    home system's star is a realm the hull can enter) ran end to end on real sockets, the peer book
    included — the star's shard was launched after the hull and was never booked with it. And a login
    placed before the gateway's clock sync takes the static route and retries an absent shard per tick;
    with the peer book that is one `PeerLocate` per tick and one orchestrator warn per ask — a rate limit
    on the ask and a once-per-peer warn are OWED.
  - **★ THE GALAXY'S TICK, MEASURED AND CUT (2026-09-03, owner: *"why do we need to look into all
    systems every tick?"*, hierarchy refused, two ideas approved):** the galaxy's own pace line now
    prints the fold's counters. Nobody under it: 0.13 ms, 1 candidate. One pilot in System 7: 60 ms,
    **9640 candidates per tick** — the interest index is one grid whose cell is sized by the widest
    child, and one cell holds ~9640 star systems; every query returned the whole cell and the fold
    built a row for each. (1) `ChildIndex` keeps each child's bounding sphere and every query
    (`candidates`, `candidates_segment`, `candidates_along`) keeps only the children whose sphere
    meets the asked point or line — a subtraction and a compare per child, no row built. (2) The
    fold remembers each observer's index answer (`AoiQueryMemo`: where it stood, its lead, its reach,
    the roster generation) and re-asks only when the observer moved farther than the lead the answer
    already looked ahead, its reach changed, or the roster changed (`RealmRegions::roster_generation`,
    one process-wide counter). A star system standing in the galaxy asks once; a hull at warp re-asks
    when it has flown its own lead. Counters `aoi_queries_run/reused` on the pace line. Also on the
    pace line now: `mean_ms`, `aoi_candidates`, the ledger sizes, windows, observers.
  - **★ THE FOLD'S MEMORY UNDER ACCELERATION, AND THREE MORE SCANS (2026-09-04):** the memo re-asked
    every tick for a hull at warp because its lead changes every tick; it now remembers the LINE it
    asked (twice the lead ahead) and re-uses the answer while the line needed lies on it — a hull re-asks
    once it has flown its lead, or when it turns. MEASURED with a probe pilot whose home sits late in the
    galaxy's child list: `emit_window_frames` found a window's child row by a linear `find` over the
    galaxy's 233 220 children, once per window per tick — 90 % of a 26 ms tick; the two directory reply
    arms asked "is this my child" the same way. All three are lookups now (`direct_child`, `parent_of`).
    What the client receives, from the code: the star catalogue ONCE at login (then a small "sky alive"
    beat), one composed picture per tick (the chain's rows only), a window's bodies and static roster on
    change. The galaxy never sends its systems; it only searched them, and no longer does.
  - **OWED, in order:** the PEER BOOK (plan slice 7 — D-RLM-6 mechanism C with its two address arms, the
    MISS as the trigger, `Transport::book_peer`, the mesh's lazy dial of a booked-but-laneless peer);
    THEN the flight on the cluster (plan slice 6: out of the home system AND into the next); the
    no-flicker gate on a pilot aboard across the swap tick; the wake-up constant re-measured from the
    orchestrator's `rlm.boot_ticks_observed_max` on that flight (the 6000-tick value is the S12 census
    measurement, deliberately generous, and shrinks with the boot cost).

- **STATUS 2026-09-02, third owner flight — THE FACING CROSSES WITH THE BODY, AND THE VIEW FOLLOWS
  IT.** The owner boarded looking along x and the hull left at ninety degrees to the view: the push
  rode the NOSE, the mouse turned nothing, and looking and moving never met again (*"the direction of
  the movement suddenly changes after rehome, and obviously the controls are pointing in wrong
  directions all the time"*). Owner-approved rule, generic for every realm kind, no new wire data:
  - **The look turns the body whoever holds the stick** (`dot::face`, split out of the old
    integrator; a walker runs `face` then `walk` in the old order and is byte-identical). A pilot
    turns without walking.
  - **The push follows the pilot's facing** (`stick_from_input(input, facing)`): `W`/`S` along where
    the pilot looks, `Space`/`Ctrl` along the facing's up, in the hull's own frame; `A`/`D`/`E`/`Q`
    turn the hull, and the pilot — whose facing lives in the hull's frame — swings with it. The nose
    no longer states the flight direction: point the mouse at the nose to fly along it. Example: the
    pilot looks along x and holds `W`; the push is `facing × (0, 0, -1)` = x.
  - **A scene swap re-expresses the view** (`place_camera`, `CameraState::last_origin`): the local
    yaw and pitch are numbers in the frame the picture is drawn in; at a swap the camera takes the
    DELIVERED facing — the server's answer in the new frame — and the mouse continues from there.
    This closes a latent seam the owner did not see only because the hull sat at its berth: boarding
    a hull that has turned made the view jump by the hull's rotation.
  - **`L` = face the nose:** sends the one look that turns the delivered facing onto the realm's own
    forward, holds the eye there, and sends no mouse look while locked, so the facing and the eye stay
    together.
  - The server side already converted a facing at a crossing and derived yaw/pitch from it on
    arrival (`saga_arms.rs`, the walker's earlier fix); this round reused it, changed nothing there.

### D-MOVE-5 🟥 A SIZE IS A PLAIN DECIMAL WHILE A POSITION IS ON A RULER — raised 2026-09-01 by the owner

- **THE OBSERVATION (owner, 2026-09-01).** Reviewing why a built-realm record cannot promise total
  equality, the owner asked whether a box's `20.0` depends on the ruler's smallest step. **It does not,
  and that is the finding.** The two halves of the same geometry are measured differently:
  - a POSITION is `LatticePos { cell: I64Vec3, offset: DVec3 }` — a whole-number cell on a TIERED ruler
    (millimetres inside a star system, ~2 m across a galaxy, kilometres at the root) plus a leftover;
  - a SIZE is `Boundary::Shell { r: f64 }` / `Aabb { half: DVec3 }` — a plain decimal on no ruler at all.

- **WHAT IT COSTS TODAY.** A size can hold `NaN`, so it cannot promise to equal itself, so no type
  containing one may claim total equality (`Eq`) — which is why `BuiltBody` and `RealmInterest` both
  carry only `PartialEq`. It also means two sizes that should be identical can differ in their last
  bits, and no comparison of boundaries is ever exact.

- **THE CURE, and why it is not small.** A size on the SAME ruler as a position would be an exact whole
  number: a 20 m hull is exactly 20 000 millimetre steps, equal to itself for ever, with no `NaN`
  reachable. But it needs the TIER for the same reason a position does — one ruler cannot span a 20 m
  hull and a galaxy at once: at millimetre steps a galaxy needs ~9.2e21, and the whole-number type
  stops at ~9.2e18. So a size would have to state which ruler it is on, exactly as a position does.

- **THIS IS AN OWED CHANGE TO THE ONE WORLD, NOT A VARIANT (SL5).** The law names "true astronomical
  scale + generated stations/areas + the galaxy cell lattice" as owed changes to this world; this is
  the same species. Numbers change on THE world.

- **BLAST RADIUS:** every boundary in the world — the containment fence, the interest radius, the
  drawing, the nest fence, and every stored extent. It is a design, not an edit.

- **WHEN:** after the first ship flies (owner, 2026-09-01: *"add it to the ledger to chase after the
  test flight is done"*).

### D-MOVE-4 🟥 A PLANET HAS NO DAY, AND THE SECOND FRAME THAT ANSWERS IT — raised 2026-09-01

- **MEASURED, and it is the fact the whole question turns on:** nothing in the tree spins. `FrameRef`
  has no rotating arm, the generator draws no rotation rate, and `celestial.rs` names no sidereal day.
  So a planet does not turn, and "fixed above a point on the surface" works today only because the
  surface does not move either.

- **THE OWNER'S QUESTION (2026-09-01).** A player builds a station inside a planet's sphere of
  influence — static, or on rails? And if a space elevator reaches it, what does the planet's spin do
  to it, and how hard is docking?

- **THE ANALYSIS, recorded so it is not re-derived.**
  - A frame that SPINS with its planet makes building trivial and every orbit inside it wrong (a
    rotating frame invents forces that are not there).
  - A frame that stays STILL keeps orbits correct and makes a "static" station hang over a moving
    surface — which is what would tear a space elevator off.
  - ⇒ **RAILS FOR ANYTHING IN SPACE.** A space elevator's station is not motionless: it goes round
    once a day, exactly as the ground does. Same mechanism as a low station, different number.
  - ⇒ **A SECOND, TURNING FRAME FOR ANYTHING ON THE GROUND** — a city, a pad, a mine. Real spaceflight
    uses exactly these two frames and no more.
  - ⇒ **A STATION IN A STAR SYSTEM IS ALSO ON RAILS.** One that does not orbit falls into the star the
    moment the system applies its pull.

- ★ **DOCKING IS EASY *BECAUSE* OF THE NESTING, which is the strongest argument for rails.** The hard
  part — matching a speed — exists only while you are OUTSIDE. Cross into the station's own realm and
  the station reads as still; what is left is the metres per second you actually have relative to it.
  A 7 km/s rendezvous becomes a 2 m/s drift, and the boundary crossing does the sum. That machinery
  exists and is proven under kill-9.

- **WHAT IS OWED:** a rotation rate per body (seed-derived, like every other property), a turning
  frame beside `PlanetCentered`, and the day/night lighting that is the owner's stated reason for
  wanting it. **WHEN:** after the ship flies (owner, 2026-09-01: *"let's discuss that after your
  planned work with the ship is done"*).

### D-MOVE-3 🟥 THE BAND RE-SOLVE — approved 2026-08-31 as THREE PIECES OF WORK, none of them a number

- **THE RULING** (`owner_decisions_2026-08-27_movement_answers.md` M-D, approved 2026-08-31). The bands
  were never the defect. **The band protected a SNAPSHOT, not a player** — sizing it by speed was a
  workaround for looking too rarely, which is why deleting the governor broke every band at once: the two
  were one mechanism wearing two names.

- **MEASURED, and it changes the job completely.** Containment is ALREADY swept — slice S5 built it, and
  `region_verdict` reads the LINE from the previous pose to this one on exact integer arithmetic. So a fast
  ship is still SEEN. The real constraint is TIME: a hand-over is a saga needing `k_dwell = 5` ticks, and
  the thinnest governed crossing in the world measures **5.59 ticks**. The machinery sees you; it cannot
  hand you over in time.

- **THE THREE PIECES:**
  1. **CLOSE THE SHAPE HOLE.** `Boundary::Obb` is the one shape that still tests an instant
     (`geometry.rs:1521-1523`), falling back to the point verdict. No shipped realm uses it today, so this
     is a hole rather than a live defect — but it is the one arm where the swept law is not true.
  2. **START THE HAND-OVER EARLY.** The parent authors velocity and the test already reads the travelled
     line, so a system can SEE a crossing coming and begin the saga BEFORE the edge is reached. Same shape
     as the wake rule: warm ahead, never slow down.
  3. **THEN THE BANDS SHRINK ON THEIR OWN.** With an early start a band keeps only two jobs — stay wider
     than the coordinate step, and give hysteresis. **Neither job scales with speed.**

- **⚠ THE THREE RED TESTS MEASURE THE GOVERNOR, WHICH THE RULING DELETES.** `g_flight_table_the_governed_legs…`,
  `g_governed_bands_bracket_every_ambient_boundary…` and the dual-shard cruise leg all read
  `realm_speed_cap_mps` / `leg_time_s` — the closed form of the ceiling that M-D removes from the flight
  path completely. They are tests of a mechanism being deleted, not failures to repair. **The owner has not
  yet said whether they retire with it or stay red until the replacement lands.**

- **WHEN:** with D-MOVE-2, and piece 2 is what makes high speed survivable at all.

- **STATUS 2026-09-04 — THE HAND-OVER NAMES ITS NODE, THE INDEX IS A TREE, A FLIGHT RUNS RELEASE.**
  Three items the owner approved on 2026-09-04 (*"Yes, I agree we need to have 4, and 5 … Also agree
  that we need to implement rstar"*):
  - **Item 4, the hand-over names its node** (wire minor 28, a mesh flag day). `ExteriorMoved` now
    carries `parent_node`, the commit's destination node. The reconciler fills it from the saga
    context; the gateway's lineage splice writes it into the realm-head map for the new parent, so the
    window derivation opens the new parent's window ON THE SPLICE TICK instead of after the next
    half-second head poll. Example: the hull leaves System 7 for the galaxy at tick 500; the galaxy's
    window used to open at tick 512, now it opens at tick 502. Unit-pinned in the gateway
    (`the head is known at once`). The blink on the cluster is OWED a measurement on the release
    flight below.
  - **The child index is an R*-tree** (`rstar` 0.13, `crates/core/src/child_index.rs`). WHY: the grid's
    cell was sized by the widest child, so the galaxy's cell held 9 640 systems and every hull's query
    filtered them all (10 ms per hull, a ceiling of about two flying hulls). The tree stores every
    child under its own padded f64 box and descends only the boxes a point or a line meets; the exact
    sphere test on the lattice decides, so the answer is the grid's answer. There is NO abstain any
    more: a step of any width is answered, so the containment fold's "the index abstained, ask
    everything" arm is deleted (`worth_asking` has four arms). The grid-edge octave pin in the physics
    index gate is gone with the grid. ★ SL9 MEASURED on this build (`galaxy_adopt_measure`, the ratio
    ≤ 25 gate): an adopt-and-release round on the galaxy's 233 220 children costs 45.8 µs, on a
    ten-child realm 47.7 µs — ratio 1.0 (the grid measured 3.1–3.4 on 2026-09-03). The physics index
    gate answers the same before and after the band sizing (max 13, mean 7.52 candidates per query
    over 123 990 queries). The per-hull cost on a galaxy with a pilot flying is still to be read off
    the release flight's pace line.
  - **Item 5, release builds for the window** — `VD_BUILD=release scripts/dev-cluster.sh up …` and
    `VD_BUILD=release scripts/client.sh …` build and launch from `target/release`; the default stays
    debug. The berth example follows the profile it is run with (`cargo run --release --example
    berth_hull`).
  - **The markers the owner saw (2026-09-04, *"parents should not draw children"*)** are the R9 step 5
    deletion, which the ruling binds to run LAST, after step 4 (reach) is green — see D-REACH-1. Today
    `current_bodies` still ships one marker per child in range and the client draws it while the child
    is silent. Not forgotten; sequenced.

- **STATUS 2026-09-04, the sixth owner flight (release build) — "still blinks and stops for a second
  on transitions".** Owner: *"Let's do all planned optimisations first."* MEASURED from the gateway's
  own admin counters (slot port `admin_gateway`, 7561 on slot 17) and the shard logs:
  - every shard's tick mean is under 0.12 ms on release, the galaxy asks the tree for 3 candidates —
    the server is not the slowness;
  - three hand-overs (System 7 → the star realm → System 7 → the galaxy), each with the same shape:
    ~120 ms from the crossing decision to the new parent's window (item 4 opened it 20 ms after the
    adopt), `window_compose_hold_ticks` 11 over the three (≈ 4 ticks, 80 ms each), and on the
    CLIENT a full refill of the 120 ms interpolation buffer, because the scene swap forgot every
    track although the origin (the hull) had not changed. The hull itself never stopped on the
    server: it coasted at its flushed velocity;
  - `window_misauthored_body` 2: the new parent's relay named the hull's body one tick before its
    level rostered it (the row waits for the book; the body did not) — dropped, counted;
  - ★ `window_rotated_refused` 510 904 — twelve rows per fold, every fold, for the whole flight.
    `FrameError::RotationBeyondExactReach` ("unreachable in production today (every placement in the
    tree is identity-oriented)") is REACHED: the hull's placement carries the pilot's turn, so every
    row beyond the FINE rotation reach (2^42 m ≈ 29 AU) mapped down through the hull's hop is refused.
    Today those are the galaxy's in-range systems, which the star cloud draws anyway, so the picture
    does not show it. It is the named P10 trigger firing on every tick, not a bug in the refusal;
    ★ BUILT 2026-09-04 (owner: "proceed with all the changes"): THE FAR ROW. A row the exact descent
    refuses for rotation, whose author's frame is the SKY's (`compose_in(.., sky_frame)`), rides in
    that frame as stated (`Composed::far_rows`, never refused); the client's realm view admits a row
    in the sky anchor's frame as the one lawful foreign frame, and the renderer places such a box
    from the sky anchor with the star cloud's own transform (`sky_cloud_transform`: one f64 turn,
    narrowed once — the precedent the owner approved for the stars, R1). No wire change: the row's
    `pose.frame` already says which frame it is stated in. Example: a neighbouring system two
    light-years out, seen from a hull turned thirty degrees, is exact in the galaxy's step and turned
    where the star field already is. Pinned: gateway (refused without a sky frame, a far row with it,
    not admitted for another sky), client (the anchor's frame folds, another foreign frame is still
    skipped). Flight OWED: `window_rotated_refused` should read zero under the galaxy.
  BUILT on this status (unit-green, flight OWED):
  1. **The same-origin swap keeps its motion** (`RealmView::swap_epoch(_, keep_tracks)`,
     `retain_placements`; `net.rs`'s level arm): the epoch bumps for a chain change under the same
     origin, and the client keeps every track, the feed counter and the sky anchor, dropping only the
     tracks the new level does not draw. A NEW origin still forgets everything. No wire change.
  2. **A relayed body rides only beside its row** (`window.rs` relay emitter): bodies are filtered to
     the relay's own level rows, so the gateway's roster vouch never sees a subject the rows lack.
  3. **The picture rule at a hand-over** — the hold is 4 ticks per hand-over, measured; it is the
     fresh-prefix rule doing what it says while the new parent's first level arrives. Left as is: a
     row composed from two instants is refused by law, and 80 ms is inside the interpolation buffer
     once item 1 no longer empties it. Re-measured on the next flight; revisited only if it shows.
  4. **The peer book's ask rate** (`PeerBook::pending`, doubling backoff, one warning per peer): a
     peer that stays unknown is asked at its 1st, 2nd, 4th, 8th … missed tick; the answer clears it.
  5. **The wake-up constant** (`DEV.boot_ticks_p99` 6 000 → 500): `rlm.boot_ticks_observed_max` read
     216 ticks on the release flight; doubled and rounded to ten seconds. The AoI's F7 horizon (the
     wake-ahead lead) shrinks with it from two minutes of travel to ten seconds.
  ★ THE SEVENTH FLIGHT (2026-09-04, release, items 1–5 aboard) — owner: *"All works … the rest were
  quite seamless."* Gateway counters over the same three hand-overs: hold ticks 11 → 0, mis-authored
  bodies 2 → 0, no standing realm 5 → 8 (all eight at boarding, none at a hand-over). Two things
  left: (a) *"only star blinked when I passed by"* — the hull entered the star's own realm and left
  it again; the swap level is composed from a partial chain while the new hop's first level is in
  flight, and the client pruned the star's track against that level (built in item 1, now
  corrected: a track is forgotten only when a DELTA removes its realm, never at a swap level);
  (b) *"in the Galaxy couple of times I saw moving stars froze"* — the star cloud is placed by the
  sky anchor, which rides the composed datagram; UNMEASURED which beat it missed. Next flight: poll
  the client's own state at the tick rate and log every tick the anchor did not advance.
  ★ THE COVERAGE GATE ON THE ARC (2026-09-04, `just coverage-fast` on a clean debug tree): the R-tree
  index (`vd_core::child_index`) has NO miss. The gate is RED on the arc as a whole: 241 real misses
  (192 lines, 49 branches) in 31 files — `drive.rs` 61, `forest_query.rs` 26, `orchestrator.rs` 19,
  `containment.rs` 14, `realm_head.rs` 13, `register.rs` 11, `regions.rs` 10, `exterior.rs` 10,
  `generate.rs` 10, `placement.rs` 9, `aoi.rs` 9, `guards.rs` 8, the rest ≤ 7 each. This is the debt
  of everything uncommitted since 9639c73 (the movement contract, the ruler switch, the reach slice),
  not of one change; 47 of it was paid today (a roster edge suite, two window-lane arms, the
  source-side demote, the promote redelivery, the inbox facts arm). The full list is kept beside this
  ledger (`docs/audit/coverage_gate_2026-09-04.txt`). HR5 says the phase is not done until it reads
  zero; it is a dedicated pass, scheduled before the commit.
  ★ THE EIGHTH FLIGHT (2026-09-04, after the wake-up constant 6 000 → 500): the same three hand-overs
  read 9 hold ticks (all at the first, into the star realm; the star's shard had booted 48 s before, so
  a late boot is NOT the cause — UNMEASURED which hop's ring lacked the tick), `scene_deltas_sent`
  370 against 18 on the seventh, and `window_instant_mismatch` 4 955 against 0, rising ~12 a second
  from the first hand-over on. READ FROM THE CODE: the ten-second horizon makes the in-range set
  churn as the pilot flies (each change is a delta), and `merge_static` folds the static roster's
  rows — stamped at the tick they were sent — into every later level for a child the level no longer
  carries (one that just left the range); `transfer_frame` refuses that row for its instant, counted,
  never drawn. So the counter is a stale-roster refusal, not a lost picture; the churn is the
  horizon's nature at warp. OWED: drop a static row the moment its child leaves the admitted set
  (then the counter reads zero again), and the hold's hop on a fixture.
  ★ THE NINTH FLIGHT (2026-09-04, steps 4 + 5 + the far row aboard): owner *"Feel better … I still
  see the dots (at least on the star)."* Counters: holds 7 (the first hand-over again), rotation
  refusals 7.6 k and rising (the far-row arm sat on the direct path only — the refusals come through
  the RELAY path; now on both, and `window_far_rows` is a stats row), instant mismatches 2.5 k and
  rising, star-cloud stalls 0, the galaxy shard's boot 38 s (was 4 s: the reach re-banded 233 220
  children one leaf insert each, three times — now re-banded in place with ONE bulk rebuild, and the
  own realm is named last). ★ THE MISMATCH, PINNED (`a_static_row_composes_through_a_hop_at_a_later_
  tick`): a static roster row kept the tick its roster was sent at, and a hop at any later tick
  refused it for its instant — so from inside a hull a parent's STATIC children (the star, a station)
  were composed only on the roster's own tick. The dot the owner saw on the star was the star field's
  point with no body over it. FIXED: `merge_static` re-stamps a static row at the level's tick (a
  static placement is time-invariant). Owed: the tenth flight.
  ★ THE TENTH FLIGHT (2026-09-04, the three fixes aboard): mismatches 0 (was 10 280), scene deltas
  14 (was 714), far rows 797 ridden, rotation refusals 1 601 (was 65 932; the rest are the universe's
  rows about the galaxy — not sky-authored, not drawable), holds 5, the galaxy's boot 217 ticks again.
  The star: its BODY now composes at every tick, and the owner's screenshot shows the star field's
  point through it — the realm box is a translucent volume (`AlphaMode::Blend`, the honest-translucency
  rule for containers seen from inside) and the cloud adds over it. FIX: a body that states its own
  LIGHT (its self-look carries `TAG_LUMA`) is drawn OPAQUE — a glowing body has no interior to see
  through — so its depth hides the point behind it; decided by data presence, never by kind. The
  anchor logger cannot tell "stationary" from "frozen"; it reads with the pilot's speed next.
  ★ THE ELEVENTH FLIGHT (2026-09-04, the opaque lit body): the star is a solid disc, no point over
  it. Owner: *"galaxy, sun and all other objects are not moving together at the same time — that
  creates that bouncing effect"* (when the hull turns; steady when stopped). READ FROM THE CODE and
  MEASURED at the client (the star's body and System 7's coincident body never disagreed by more than
  half a metre over 1 079 samples, so it is not two bodies): the SKY ANCHOR placed the star cloud the
  instant each datagram arrived, while every body is drawn from its track 120 ms behind — so a turn
  reached the sky a buffer ahead of the bodies. FIXED: the anchor rides an `EntityTrack` like any
  realm's placement and is sampled at the same render cursor as the scene (`sky_anchor_now`), in
  both readers (the cloud and the far rows); pinned: two anchors at ticks 10 and 12 give the halfway
  pose at cursor 11, and the newest past it.
  ★ THE BRIGHT-BODY COUNT (R8 item 4, MEASURED 2026-09-04 on the eleventh flight, read from the
  slot's realm stores after the pilot logged out): one pilot in System 7 ran 17 shards — the
  universe, the galaxy, System 7, its star, TWELVE planets and moons, and the hull; every planet and
  moon woke by its light (their light reach, ~50 AU for an Earth, covers the whole system). On the
  size-only reach (the seventh flight) it was seven planets and moons, plus five far systems that
  the two-minute horizon woke and the ten-second one does not. So the price of R2 today: +5 shards
  per occupied system, −5 far systems per warp. Both counts are what the rulings ask for; nothing to
  tune. Owner (the twelfth flight): *"Now all waaay smoother."*
  ★ THE HAND-OVER'S GAP, MEASURED AND CLOSED (2026-09-04, the gateway fixture
  `a_hand_over_holds_the_departed_strata_until_the_new_hops_first_level_lands`): at a splice the new
  hop's window is UNCONFIRMED until its first level lands, so the chain shrinks to the hops below it
  and every stratum the departed author drew was DROPPED — System 7's planets vanished for the new
  window's round trip (the fixture read: chain = the leaf alone, hold ticks 0, drawn rows empty). The
  flights' 5–9 hold ticks were the ticks after confirmation while the new hop's ring and the leaf's
  found a common stamp. FIXED: while the chain does not cover the lineage, a departed author's rows
  from the last fold become a HELD stratum (drawn at their last composed poses, bounded by the hold
  TTL) and leave the tick the chain is whole again; a promoted stratum takes only its own author's
  rows, never another author's that sat at that index before the splice re-numbered the chain. The
  stall logger now reads the pilot's own motion beside the anchor (a stationary pilot is not a stall).
  ★ THE TWELFTH AND THIRTEENTH FLIGHTS (2026-09-04, everything above aboard): owner *"Now all waaay
  smoother"*, then *"All works"*. Counters on the last: hold ticks 4 (post-confirmation stamp
  alignment only), instant mismatches 0, far rows ridden, rotation refusals 9 200 (the universe's rows
  about the galaxy), anchor stalls with the pilot moving 0. The flight list of the owner's
  2026-09-04 "proceed with all the changes" is CLOSED.
  ★ THE COVERAGE PASS (2026-09-04, evening): the gate read 269 real misses in 33 files over the arc.
  Six agents closed them in parallel, each on disjoint files, with tests only — a few provably dead
  arms removed and said so (the containment fold's `worth_asking` guard, made redundant by the ask
  list; an IPv6 slot arm dead by type; a poisoned-mutex arm nobody can reach; a static-row arm no
  production caller reaches, kept and pinned). `just coverage-fast` now reads **0 real misses —
  PASS** on the whole Tier-A set. Also found on the way: two mesh pins still expected "queue full"
  for an unknown destination (the peer book refuses it as unknown since 2026-09-03) — re-pinned; and
  a node must never lazily dial ITSELF (its own id sits in the topology) — refused before the dial.
  Known: `frame_conversion_e2e`'s `shipped_movers` regenerates the world per region per seed with two
  configs alternating on a one-slot forest cache — hours in debug; unchanged since Step 7, not run
  to completion here. Committed as 9f2b426 (Step 9).
  ★ THE SIXTEENTH FLIGHT (2026-09-04, after the commit): *"when I'm flying close to the star, game
  freeze for 3-4 seconds, star disappears, then all unfroze and star appear like we passed it some
  time ago … after warp speed is frozen and all is frozen in general."* MEASURED at the client
  (10 Hz poll of its own state): the client ran at 32 % CPU with no sampling gap, `decode_errors`
  2, its scene origin at epoch 9 while the gateway had moved on, `realm_feed_newest_tick` 9 067
  against universe tick 14 529, `stale_epoch_rows` 105, the sky anchor a constant — the server
  showed no slow tick and no warning. ROOT CAUSE: the hand-over hold (this evening) keeps a departed
  author's rows, INCLUDING that author's own body row, while the new hop's level rosters the same
  realm as a child row — one level named System 7 twice, the client refused it as malformed
  (`SceneError::DuplicateRealm`), kept its old epoch, and held every later datagram: the picture
  froze, then jumped when a later level got through. FIXED: `ShadowScene::drawn_rows` lists each
  realm ONCE — the fresh fold's rows first, a chain author's hold next, a departed author's last —
  pinned by the hand-over fixture with the universe hop unconfirmed and the galaxy rostering
  System 7. OWED (hardening): a refused level leaves the client behind until the next epoch bump;
  the gateway should restate the level on the keep-alive beat so one refusal cannot freeze a session.
  ★ THE SEVENTEENTH FLIGHT (2026-09-04, on the deduplicated gateway): the freeze is gone (owner:
  *"Good, fixed"*); *"the sun still blinks (is it when we are passed from System to Sun Realm)"*.
  MEASURED at the client (10 Hz): the drawn boxes went 12 → 1 → 12 across one sample at 21:17:38,
  the exact second of the orchestrator's hand-over into the star realm; again at 21:18:38 on the
  next hand-over. Every drawn realm but the hull blanked, not only the star — the star is the one
  the eye keeps. ROOT CAUSE: a look lives in the INGEST of the window that carried it, and a
  hand-over closes the departed hop's window and opens the new hop's; at the splice the old
  window's looks are dropped and the new windows hold none until the shards re-serve their bodies.
  The rows survived (the hand-over hold keeps the poses) but `scene_bag` answered empty for each,
  the delta shipped the empty bags, and the client stopped drawing them for the ticks the churn
  lasted. FIXED: `window::LookShelf` — per session, the last look emitted for each realm with the
  level tick it was live at; a drawn row whose chain windows state no look reads its shelved look
  while the entry is younger than `hold_ttl_ticks` (the SAME 2-beats-+-1 window the held strata
  live on: the picture survives the churn exactly as long as the poses do). A row that left the
  drawn set is never filled; a look older than the TTL is pruned, so a realm that truly stops
  stating its look goes dark one hold later, never sooner. Counter `window_looks_carried`; unit
  fixture `the_look_shelf_carries_a_look_through_a_window_churn_for_one_hold`. Gateway local, no
  new data crosses a realm boundary. VERIFY IN FLIGHT: the star's box count must stay 1 across
  the hand-over second and `window_looks_carried` must be non-zero after it.
  ★ THE EIGHTEENTH FLIGHT (2026-09-04, owner: *"Works"*), committed as 460c150 (Step 10). MEASURED:
  three hand-overs, no freeze, the star's box stayed drawn across each hand-over, 41 looks carried,
  no decode error. RESIDUAL, not a hand-over: one 0.17 s sample with only the hull drawn at 21:37:41,
  the second the pilot WALKED ABOARD the hull — `window_unresolved_standing` rose 0 → 5 there, so
  the composed feed was withheld for five ticks while the session's standing realm switched from
  System 7 to the hull (the login-race arm, §2.6.6). RE-READ (the same evening): the withheld ticks
  were NOT the blank — the client keeps drawing through them. The blank was the SWAP: the new
  origin's own window landed before the parent hop's window, the gateway swapped onto that partial
  chain, the swap level named the hull alone, and the client forgot every track of the old origin
  (positions in another frame) until the hop's first level came. BUILT: while an old picture
  exists, the new chain does not cover the lineage, AND the first uncovered hop's window is open
  and unconfirmed (`window::hop_pending`), the origin swap is DEFERRED — the old scene stays at
  its last composed poses, exactly as a held stratum does, for at most `hold_ttl_ticks`; then it
  is forced, counted (`window_origin_swap_deferred` per tick, `window_origin_swap_forced`). A
  lineage no window can cover (no head, a refused window) swaps at once, as before, which is what
  keeps the seven crossing fixtures green. Unit fixtures
  `an_origin_swap_waits_for_the_chain_to_cover_the_lineage_for_at_most_one_hold` and
  `hop_pending_reads_the_first_uncovered_lineage_hops_window`. The look shelf now copies a look
  only when it changed. VERIFY IN FLIGHT: no one-box sample at the walk-aboard second, the
  deferred counter non-zero, the forced counter zero. The lane proof is the two crossing fixtures
  that had pinned the old behaviour: `the_composer_folds_the_crossing_chain_at_one_tick_and_tears_
  down_to_nothing` (the swap waits two ticks for `Child(Planet 7)`, then lands whole at one tick:
  two epochs, not three) and `capstone_two_sub_overlap_...` (no swap level on the ready tick).
  ★ THE NINETEENTH FLIGHT (2026-09-04, owner: *"Works great"*). MEASURED at the client (10 Hz):
  boarding at 21:58:20–25 ramped 5 → 9 → 12 boxes with NO one-box sample; `window_origin_swap_
  deferred` 1, `window_origin_swap_forced` 0, `window_looks_carried` 24, three hand-overs, the star
  drawn across each. RESIDUAL (sub-sample): at the 21:58:48 hand-over the box count dipped 12 → 9 → 12
  inside one second — three rows whose parent is not the departed author (a relayed interior's
  grandchildren) are not held by the departed stratum's filter. BUILT (2026-09-05, Step 12):
  `window::author_subtree` — a held stratum admits the author's body, its children and, to a
  fixpoint bounded by the relay depth, the rows whose parent chain inside the stratum leads to the
  author (or that state no parent, the root); a foreign parent's row and another stratum's rows are
  refused. Used at both hold sites (the departed author and the fresh boundary). Pinned by
  `a_held_stratum_keeps_the_authors_whole_subtree_and_nothing_foreign`.
  ★ THE LEVEL IS RESTATED ON THE KEEP-ALIVE BEAT (2026-09-05, Step 12; the sixteenth flight's owed
  hardening). On each beat (`window_keepalive_cadence`, off the cost-discipline block so a quiet tick
  restates too), a session with a composed scene gets its whole level again at the CURRENT epoch,
  through the look shelf, and the delta baseline resets with it; a tick that shipped a swap level
  does not restate. A client behind (it refused one level) catches up within two beats; a client
  current replaces its scene in place and keeps every track (a same-epoch level is a restatement,
  never a swap — `swap_epoch` returns None at the same epoch). Counter `scene_levels_restated`.
  Cost: one level per session per beat, ~12 rows today. Pinned by the gateway's
  `the_level_is_restated_on_the_keep_alive_beat_at_the_current_epoch` and the client's
  `a_same_epoch_level_restates_the_scene_and_keeps_the_tracks`. VERIFY IN FLIGHT: no 12 → 9 → 12 dip
  at a hand-over; `scene_levels_restated` ≈ two per second per session; no decode error.
  ★ THE TWENTIETH FLIGHT (2026-09-05, owner: *"feels better"*). MEASURED at the client (10 Hz): the
  two hand-overs at 08:52:03 and 08:52:12 show NO box transition at all (12 boxes held through both;
  the 12 → 9 → 12 dip is gone); `scene_levels_restated` 464 over ~3.5 min ≈ 2.2/s; deferred 1,
  forced 0, looks carried 45; no decode error.
  ★ ITEM 3 — A MOVING RELAYED ROW IS ADVANCED OVER ITS SKEW (2026-09-05, Step 12). The relay fold
  resolves a relayed level at-or-before the fold's tick and re-stamps its rows to that tick under the
  fingerprint identity (a stale stamp ⇒ byte-identical rows). That identity holds for a row at rest;
  a mover relayed one tick behind would draw velocity × skew behind its true place. Now the row is
  advanced by its own stated velocity over the skew with the closed-form ballistic advance
  (`StampedPose::advanced_ballistic`, zero acceleration — Category A, the one lawful cross-host
  re-advance); for a row at rest or a zero skew it is the re-stamp exactly. The gateway's tick rate
  reaches the composer as a parameter (`compose_in(.., tick_hz)`; the tests' wrapper `compose` is
  `#[cfg(test)]` with a fixture rate). Gauge `relay_moving_skew_ticks` / `window_relay_moving_skew_
  ticks`: the widest skew a mover was advanced over. Pinned in
  `a_relayed_interior_is_membership_gated_and_each_refusal_class_counts_apart` (a 10 m/s sibling
  planet one tick behind lands 0.5 m ahead at 20 Hz). MEASURE ON THE CLUSTER: the mover gauge with a
  pilot standing in System 7 (the static gauge reads 25 = one beat; the mover gauge must read the
  relay's real lag, 0–2 ticks). MEASURED (2026-09-05, a headless pilot standing in System 7 for
  90 s): `window_relay_stamp_skew_ticks` 26 (one beat — the static rows), `window_relay_moving_
  skew_ticks` 2 (the movers relay near per-tick; the advance covers the two ticks), 17,530 relay
  rows composed, `window_instant_mismatch` 0. Item 3 CLOSED.
  ★ THE TWENTY-FIRST FLIGHT (2026-09-05, owner: *"way smoother, but when I approached the Star, and
  then flew away a bit from it, it jumped to another position for a sec and then turned back"*).
  ROOT CAUSE, from the code (the poll recorded no star position — it does now, `star_center`): the
  hand-over back into System 7 makes the star a DEPARTED author; its stratum is held at its last
  composed pose, which is a pose RELATIVE TO THE ORIGIN — the pilot's hull, which keeps flying at
  warp through the hop's round trip. A held row frozen at its old relative pose rides WITH the hull,
  and the client freezes a track at its newest sample (it never coasts), so the star moved with the
  ship for the hold and snapped back when the fresh row landed. Near the star the hull's displacement
  over the hold is comparable to the distance, which is why it was a jump here and invisible at the
  galaxy hand-over. FIXED: a held row is advanced to every later fold's tick by its own composed
  relative velocity (the hop composed it: `transfer_frame` carries velocity) with the closed-form
  ballistic advance, zero acceleration — Category A, the one lawful re-advance; `WindowTuning.tick_hz`
  carries the rate. A row at rest relative to the origin only moves its stamp. The held rows' stamp is
  now the fold's tick, not the old one (§2.6.4's "old stamps declared" clause is superseded by the
  advance). Pinned by `a_held_row_advances_by_its_relative_velocity_while_it_is_held` (100 m/s, 5 m
  per tick at 20 Hz over three held ticks); the two fixtures that pinned the old stamp re-pinned.
  Error bound: the relative ACCELERATION over the hold (a turn, a thrust change), never the speed.
  VERIFY IN FLIGHT: approach the star, fly away; `star_center` in the poll must move continuously
  through the hand-over second.
  ★ THE TWENTY-SECOND AND TWENTY-THIRD FLIGHTS (2026-09-05). The star did not jump (owner). *"seems
  like the movement of the star was not that smooth anymore"* — the owner asked to measure. THE
  PER-TICK TRACE (`VD_TRACE_REALM=Star` on the gateway, `trace_drawn_rows`, counter `window_trace_
  rows`; reader `trace_star.py`) logged 5,088 ticks of the star's composed row. FINDINGS, exact:
  (1) THE ARRIVAL STALL: after each hand-over the star's distance did not change for 22 ticks (in,
  6790–6811) and 6 ticks (out, 7507–7512) while the row's velocity read 5.7e8 m/s — the new
  parent authored a STILL hull. ROOT CAUSE (code): the parent seeds an adopted child with `facts:
  None` ("the hull restates what it IS to its new parent") and `advance_all` skipped a child
  without facts entirely; the facts arrived only after the peer lookup (*"no lane and no address
  for a peer — its frames wait while the orchestrator is asked"*, 0.3 s). The hull STOPPED for
  0.44 s at warp on every hand-over — the movement ruling says a transfer never changes a speed.
  FIXED: `advance_driven` takes the facts as an option; a child without them coasts and is pushed
  (the push is an acceleration, the pull cancels mass) and feels drag the tick its facts land.
  (2) THE PERIODIC STUTTER: every ~70 ticks the hull's placement lagged one tick then caught up two
  (t=7072/7073, with the gateway folding no 7071). ROOT CAUSE (code): a follower's universe tick
  moves only when the orchestrator's sync lands (`FollowerClock::now` is the last observed sync),
  so a local tick sees +0 or +2 now and then; `advance_all` integrated once per LOCAL tick, so the
  motion per universe stamp was uneven. FIXED: `DrivenChildren` remembers the tick it advanced to
  and integrates exactly the universe ticks elapsed (none on +0, two on +2). Pinned by
  `a_child_that_never_stated_its_facts_coasts_and_is_pushed_without_drag` and
  `the_step_count_follows_the_universe_tick_not_the_call_count`. (3) The gateway folded no row on
  152 of 5,244 ticks (a 2-tick gap; one 6-tick gap): the same clock jitter seen from the gateway —
  benign under the client's six-tick buffer, noted. VERIFY IN FLIGHT: the trace's held runs stay
  one tick, no zero-rate run after a hand-over, no lag-then-catch-up pairs.
  ★ THE TWENTY-FOURTH FLIGHT (2026-09-05, owner: *"Sun jumped again on rehome"*). THE TRACE, exact:
  the hand-over IN is clean now (held 2 ticks, then fresh, no zero-rate run — the arrival stall is
  gone); at the hand-over OUT the star's row at t=15040 read (−40, 0, −1.0822e10) for ONE tick —
  the negative of the hull's ORIGINAL BERTH — then the true 4.7e9 at 15042. ROOT CAUSE (code):
  `release_child` never rebuilt the parent's STATIC placement layer (an `Arc<Vec>` built once at
  boot), so after the hull left System 7 its berth row stayed in System 7's book for the whole star
  leg; at re-adoption the tick's book was authored BEFORE the adoption landed (the schedule's order)
  and still carried the berth row, the window's first level took its hop from it, and the star
  drew at minus the berth. `adopt_child` never rebuilt it either. FIXED: `rebuild_static_rows_for
  (parent)` — that parent's rows and movers only, O(its children), once per release or adoption,
  never per tick — called from both. Pinned by `a_released_static_child_leaves_its_parents_book_
  and_an_adopted_one_joins_it`. COST NOTE: on the galaxy (279,380 children) one release or adoption
  copies its whole static row vector once (milliseconds, once per hand-over through the galaxy);
  a per-row edit of the shared vector is the cheaper shape if that ever measures. VERIFY IN FLIGHT:
  no row at the berth's negative after the hand-over out.
  ★ THE TWENTY-FIFTH FLIGHT (2026-09-05, owner: *"It's smooth now"*, committed 60c179a): both
  hand-overs held two ticks with the row moving at the true rate; no standstill, no berth row, no
  stutter. MEASURED WITH IT: the galaxy shard's slowest tick was 60.0 ms (mean 0.32 ms) in the pace
  window right after the hull's hand-over INTO the galaxy — the static-layer copy of 279,380 rows at
  adoption, as the cost note above predicted (likely, not proven: unmeasured before the rebuild).
  FIXED (item 1 of the owner's list): A SHIP IS NEVER A STATIC ROW — a `Ship` child is a mover for
  the placement layer whether or not it moves yet; its row is the overlay's (`placement_row`: the
  driven state, else the authored centre), so adopting or releasing a hull touches `movers_of` only
  and copies nothing. A planet's adoption still rebuilds its parent's layer. Pinned by
  `a_ships_adoption_or_release_never_copies_the_parents_static_layer` (the layer's `Arc` identity
  unchanged across a ship's adopt + release; changed by a planet's). VERIFY: the galaxy's pace window
  after a hand-over reads no tick over 20 ms. Item 2: the flight instruments moved into the repo —
  `scripts/flight/{trace_star.py,star_poll.sh,flight_counters.sh}`.
  ★ THE TWENTY-SIXTH FLIGHT (2026-09-05, owner: *"Works"*, committed 7ac5e9c, Step 14): three
  hand-overs; the galaxy shard's slowest tick 4.5 ms (was 60 ms), mean 0.1 ms, nothing over budget;
  the trace's held runs one to two ticks each.
  ★ THE COVERAGE GATE AFTER STEPS 12–14 (2026-09-05): first run FAIL with 2 real misses, both one
  arm — a chain change under the same origin BEFORE any fold (an empty ring: nothing to hold);
  pinned by `a_chain_change_before_the_first_fold_holds_nothing_and_still_bumps_the_epoch`. Second
  run PASS: 100 % of merged source lines and branch sides (69 tool-missed rows, all artifact rows).
  ★ THE PROCESS GATES (2026-09-05, the five unmeasured since the arc started + the pre-existing
  reds), first run on the debug binaries as committed: `two_ships` RED at login — the gateway closed
  both clients at +14 s (*"the dynamic home realm did not become routable inside the bounded
  bootstrap TTL"*) while the debug shards were still planting (the last announced at +58 s): the
  500-tick `boot_ticks_p99` was measured on RELEASE (216) and the process gates spawn DEBUG
  binaries (43–56 s boots, as its own comment said). FIXED: the bound covers the binary that runs —
  `if cfg!(debug_assertions) { 3_500 } else { 500 }` (70 s at 50 Hz; a fact about the binary's
  speed, not a second world). `render_crossing_smoke`: 1 ignored (PARKED on D-LOOK-4, not a red).
  `look_pixels`: its derived-budget test plants THE world in debug — stopped after 31 min at 100 %
  CPU and 5 GB; UNMEASURED in debug (the hours class, like `frame_conversion_e2e`); run it in
  release or park it by name. `rlm_demand_login` on the old bound: the inner-planet chase never
  crossed within 150 s (the ledgered no-cruise leg), the rest waited on logins the old bound closed;
  stopped at 30 min and re-run below on the rebuilt binaries.
  SECOND RUN (debug, the widened bound): `two_ships` got PAST the login (the bound fix holds) and
  then sat in its fly-in at 2.5–4.4 km/s against a planet 6.1e10 m away at 4.9e4 m/s — 230 chunks of
  TIMEOUT in 21 min: the ledgered no-cruise leg (the governor is deleted; the band re-solve is the
  owner's open item), stopped by hand. `rlm_demand_login`: three of four tests failed inside their
  OWN 60 s deadline — in debug the gateway folds the sky at +35 s and the shards announce at +85 s
  (`DEADLINE = 60 s` in the test) — and the fourth reached the same no-cruise leg. So in DEBUG the
  gates' own deadlines are shorter than the boot they wait for. THIRD RUN, in RELEASE (`cargo test
  --release`, the profile the owner ruled flights run in, 2026-09-04): results below.
  THIRD RUN, RELEASE: `two_ships` PASS in 1 311 s (both hulls fly in properly — the debug fly-in
  never accelerated past 4 km/s, the release one reached 9.8e8 m/s and crossed; mutual visibility,
  the same separation from two chains of different depth to the metre, 610,752 relay statements, 0
  undecodable). `render_crossing_smoke` 1 ignored (parked). `look_pixels` STOPPED again at 20 min,
  100 % CPU, 4.8 GB: its `derived()` fixture calls `WorldView::generated(seed)` — it generates THE
  WHOLE WORLD (3.5 M bodies) inside the test, in release as in debug: the hours class. UNMEASURED;
  owed: derive the budgets from the planted neighbourhood (the shard's own forest), never the
  generated world. The remaining gates: below.
  `rlm_demand_login` (release): the three login tests ran; the inner-planet chase is the ledgered
  no-cruise red; the fourth flight (a bare dot into the between-space) WEDGED THE GALAXY SHARD — its
  ticks doubled, 45 s → 91 s → 183 s, at 100 % CPU, after the dot's crossing into the galaxy whose
  adopt never established (*"Promote found NO granted crossing-target dot … the saga re-drives"*
  every 0.5 s; *"ARRIVAL SHIELD EXPIRED … wedged"*). NEW RED, cause unnamed: the release binary
  is stripped (`strip = true`), so a stack sample names no frame. Owed: reproduce with
  `CARGO_PROFILE_RELEASE_STRIP=false` and sample the shard; the doubling means the tick's work grows
  with the queue that piles up during the tick. `window_shadow_parity`: INVALID this run — my own
  kill of the wedged cluster removed its cluster mid-flight; re-run below. `flight_table`: the two
  ledgered reds (the governed legs vs the deleted governor; the boundary roster 3,500,645 vs the
  pinned 51). `dual_cluster_crossing_smoke`: PASS in release (its debug red was the no-cruise leg).
  `star_sky_pixels`: RED on one star, `System(26526667488941648)`, "IN VIEW" at x = −10.5 px — off
  the frame's left edge, so no pixel can paint within 12 px: the same class as the ledgered
  edge-of-frame red, a different star. `ship_flight_e2e`: did not BUILD — `vd-tests` holds two
  `GatewayConfig` literals that Step 13's `trace_realm_kind` field broke (`cargo test --workspace`
  was not re-run after Step 13; the crate-level suites were): FIXED, both literals.
  RE-RUNS (release): `ship_flight_e2e` PASS (the ledgered "plants no OwnBody" red is gone);
  `window_shadow_parity` PASS in 675 s — 393,139 rows composed across 19,693 folds, 19,691 full-chain
  folds, dedup max deviation 0, fold divergence 0, shear mismatches 0. TALLY of the five owed gates
  in release: two_ships PASS, window_parity PASS, rlm_demand_login RED (one ledgered no-cruise leg +
  the NEW galaxy wedge), render_crossing parked, look_pixels unmeasured (the hours class). The
  pre-existing reds: dual_cluster now PASS, ship_flight_e2e now PASS, flight_table's two stay,
  star_sky_pixels one edge-of-frame star stays. OWED: the `just` gate targets must run release for
  the process flights (they spawn debug binaries today), and the bare-dot galaxy wedge.
  ★ THE GALAXY WEDGE, NAMED (2026-09-05). Rebuilt release without stripping, re-flew the exact
  between-space test, sampled the galaxy shard at a 5.8 s tick: 3,779 of 3,779 samples on
  `on_gateway_msg → dot::apply_input → dot::walk → dot::governed_ceiling_for_frame`, 3,272 of
  them in `Vec::from_iter` and 633 in `PlacementBook::of` — the governed ceiling walked EVERY
  child of the realm (`regions.child_rows`) on every walk input, 279,380 on the galaxy, and the
  same walk ran for every held transient every tick (`readvance_transients`). SL9 by name ("never
  a per-tick walk of all of them"), and the ruling's "the ceiling leaves the flight path
  completely". Inputs queued during the long tick made the next tick longer: the doubling.
  FIXED: (1) `ChildIndex::candidates_within(point, radius)` — the R*-tree answers the children
  whose padded sphere comes within a radius of a point; `governed_ceiling_in_book` asks it with
  the approach horizon `v × tau` (a child farther than that cannot lower the ceiling: `child_cap +
  dist/tau < v ⇒ dist < v·tau`) plus the movers, then tests exactly — O(nearby), never the roster.
  (2) the per-tick re-clamp of held transients is DELETED (owner ruling 2026-08-27: a transfer
  never changes a speed; a re-clamp is a jump) — a piece of debris keeps the speed it arrived
  with; `readvance_transients` no longer reads the regions or the placements at all. Pinned by
  `candidates_within_returns_the_children_the_radius_can_reach_and_no_others` (core) and
  `a_transient_keeps_the_speed_it_arrived_with_and_is_never_re_clamped`; the ceiling's own fixture
  stands unchanged (the 100 m child at 600 m still binds through the index). The walk's fence stays:
  a walk states a speed (a tool), which the ruling allows to be fenced. RE-FLOWN (unstripped
  release, the galaxy watched for 31 min): no doubling, slowest tick 97 ms, mean ~25 ms at 45 % CPU
  with the dot inside, and NO "Promote found NO granted crossing-target dot" — the adoption
  established: it had only been queued behind the 45-second ticks. (3) The horizon is capped by
  the WALKER's own ramp (`cap_mps`): the galaxy's own ceiling is ~170 million c, so `v × tau` would
  name every child again; `governed_ceiling_in_book(.., cap)` takes the ceiling no higher than
  the caller can use this tick, which bounds the query to `ramp × tau` — a walker at 3e8 m/s asks
  for the star systems within a few light-seconds. Exact: the walk used `min(ceiling, ramp)`
  already. The re-fly's own verdict was INVALID (my sampler killed the cluster at its 30-minute
  loop end: *"fly approach: no walk response"*); re-flown plainly below.
  RE-FLOWN PLAINLY (unstripped release, 55 min, no kill): the dot's crossing INTO the galaxy
  established at once (*"label flipped to Galaxy 1"*, no promote re-drive), the galaxy shard held
  1–2 % CPU with the dot inside and a worst tick of 20.0 ms (the budget), against 183 s before.
  THE WEDGE IS CLOSED. The test itself did not finish: its GOVERNED APPROACH leg — a 2.4e18 m ring
  gap it derives to fly at the galaxy's ceiling (5.1e16 m/s) — moved the walker at 8e13–1.2e14 m/s
  (measured twice, 20 s apart: not ramping), an eight-hour leg; stopped by hand at 55 min. That
  leg is the ledgered NO-CRUISE class (the governor is deleted; the band re-solve is the owner's
  open item — the same red as `flight_table`'s legs and the inner-planet chase), NOT the wedge.
  The bare-dot crossing INTO the galaxy — the generic re-home the owner asked for — works.
  ★ THE REACH INCLUDES THE SHELL (owner 2026-09-05, *"Sounds good"* — the band re-solve's first
  half). The owner's question: the visibility radius wakes a realm and the re-home shell is way
  smaller, so why a second mechanism? True for a planet (light reach 50 AU against a 1e6 km
  sphere of influence); false for a star system, whose star the galaxy draws itself, so its reach by
  look (its outermost planet, ~80 AU) fell 190× short of its 0.24 ly shell and a hull at warp
  crossed into it asleep. Ruled: ONE radius, and it includes the shell — a realm's reach is never
  smaller than its bound (`size_reach_of` and `own_reach` floor by `shape.circumscribed_extent()`).
  No second mechanism; the index query is unchanged. Pinned by
  `a_reach_is_never_smaller_than_the_shell` (a system whose look reaches 76 km inside a 1e9 m
  shell bands at the shell; a planet keeps its look reach; a realm states its own shell as its
  reach when its look is smaller). MEASURE: the galaxy probe with systems banded at their shells
  (the candidate count for one occupant at the origin), and a flight. The second half of the
  re-solve — the SWEPT containment test (the line from the last tick to this tick against the
  shells) — is still owed. CORRECTION (the same evening, from the code): the swept containment
  test IS built — `evaluate_realm_boundaries` reads the line from the swept prior to the current
  position (`candidates_segment(prev, cur)`, containment.rs: *"THROUGH within one tick holds
  neither endpoint, so a point lookup would skip it before the swept verdict could ever be asked"*)
  and the G-IDENTICAL swept-crossing proof pins it on two shard kinds. The band re-solve is
  therefore CLOSED by the shell floor; what stays red is the WALK tool's governed approach (the
  flight helpers), not containment.
  ★ THE MOTION REDS, ROOT-CAUSED (2026-09-05): the walk law's ramp time constant is the WAKE
  HORIZON (`FlightTuning::derive`: two beats + `boot_ticks_p99` + the pipeline, in ticks — ~12 s in
  release, 72 s in debug), and the approach ceiling near a shell uses the same constant: a player
  in a spacesuit may not accelerate faster than a shard can boot — the cap the 2026-08-27 ruling
  struck out. Measured on the release chase: 0 → 1.1 km/s → 6.8 km/s over 5,000 ticks; 280 s to
  cross 6e10 m against a 150 s deadline. RULED (`owner_decisions_2026-09-05_suit.md`): a character
  WEARS a suit that states its rating like a hull; without a suit no movement in space; the suit
  comes AFTER the character, which comes AFTER blocks and terrain. DEFERRED BY THE OWNER: the walk
  keeps the boot-time ramp as a placeholder until the suit. NOT deferred: the gates fly HULLS (S4).
  ★ THE TWENTY-NINTH FLIGHT (2026-09-05, owner: *"all good, no regression"*; *"during rehome in
  3rd-person view I think for a second I saw first person view, but I might be mistaken"*).
  UNVERIFIED: no instrument records the camera mode; if it recurs, add `camera_mode` to the
  client's dev state and poll it through a hand-over.
  ★ A VANISHED CLIENT BLOCKS THE NEXT LOGIN (2026-09-05, seen three times today): after a client is
  killed without a Close, the gateway keeps its session, sheds reliable frames to the unreachable
  peer (*"OutboundBox staging over cap: a RELIABLE frame was SHED"*) and a new login for the same
  node/account sits in `awaiting_welcome` / "authenticating" until the cluster is restarted. The
  bounded bootstrap TTL closes a client whose HOME never comes; nothing closes a client whose PEER
  never answers. OWED (foundation): the session's lease must end when the peer's lane is dead for
  a derived bound, and a fresh login for the same identity must replace the dead session, not wait
  behind it. Example: a pilot's window crashes; they start the client again and are in within the
  ★ ROOT-CAUSED AND BUILT (2026-09-05, foundation slice 1). FOUR facts, read from the code, made the
  wedge: (1) the gateway consumed only `Inbound::Wire`; a transport notice about a dead client was
  skipped, so nothing ever closed the session; (2) a second `Hello` from the same node id was ignored
  as "a duplicate on a live connection" — but a client sends `Hello` exactly once, so a second one is
  always a NEW PROCESS, and the new client waited for a `Welcome` that could never come; (3) the
  node's outbound staging had ONE cap for all peers, so the dead client's backlog filled it and the
  shed took every LIVE client's reliable frames ("a RELIABLE frame was SHED"); (4) the dev cluster and
  the process gates BOOKED the client at the gateway, so a restarted client at the same address was
  re-dialed and the dead process's retained frames (its Welcome, its subscriptions) REPLAYED into the
  new process; and the client stamped incarnation 0 on every launch, so the transport could not tell
  a restart from a replay. THE CURE, one piece per fact: (a) a new seam notice
  `Inbound::PeerReset { node, cause: ConnectionLost | Reincarnated }` — produced only by the mesh
  (a dialed-in peer's accepted connection died; or a peer's frames carry a strictly higher process
  incarnation, once per restart, pushed BEFORE the first frame of the new process); the gateway ends
  the session of any node it names, exactly as a `Bye` ends it (`end_session_of`, the ONE exit), and
  a `NodeUnreachable` toward a client does the same; a notice about a shard is not a session fact
  and the saga runtime counts it and never feeds it to liveness; (b) a `Hello` from a node that holds
  a session ENDS that session and proceeds as a fresh login (`sessions_replaced_by_relogin`), ordered
  before the capacity check so a re-login is never refused because its own dead session holds a slot;
  (c) the staging cap is PER PEER (`shed_over_cap`), and `OutboundBox::forget_peer` drops a closed
  session's staged frames at the next flush (`TickReport::forgotten`) instead of the cap shedding
  them; (d) a client is a LEARNED peer only — the client book is deleted from the env builders, the
  dev cluster and every gate — and the mesh's learned table is incarnation-aware: a newer process of
  the same node id supersedes a live older connection (the old one is closed, `learned_peers_superseded`),
  and a learned lane spawned for a superseded incarnation is a miss; the client stamps
  `launch_incarnation()`. Counters: `sessions_closed_peer_lost`, `sessions_closed_peer_reincarnated`,
  `sessions_replaced_by_relogin` (admin 60–62). Gate: `crates/bins/tests/login_after_kill.rs` kills a
  logged-in client with SIGKILL and logs the same node in again; it must be Active within the derived
  bound with no cluster restart. NOT a hold, NOT a resume: a killed client's session is OVER; a
  resume ticket (P3) is a different promise and is still owed.
  TWO STAND-IN CLIENTS CORRECTED BY THE RULE (measured on the workspace suite): `process_parity`
  and `dual_cluster_crossing_smoke` re-sent `Hello` every 500 ms "until welcomed (idempotent)" —
  the real client never does that (it sends once the transport accepts), and under the re-login
  rule every re-sent Hello ended the pending session and started over (parity: 43 levels in 60 s,
  and it never settled). Both doubles now send once, as `net.rs` does. And `process_parity`'s
  "exactly one level per login" pin predated Step 12's keep-alive restatement (a full level at the
  SAME epoch every beat, by design); it now pins ZERO re-origins (a level at a NEW epoch) and at
  least one level. Nobody had run the workspace suite since Step 12, which is how it hid.
  usual login time, not after an operator restart.
  ★ FOUNDATION ITEMS 2 AND 3 (2026-09-05, owner: *"do 1 to 4"*). (2) `look_pixels`: sampled with
  symbols — every sample sat in its own `oracle()`, which took EVERY parent in the forest (233,220
  star systems) and generated the world once per parent for its movers. The fixture now reads the
  planted neighbourhood (`realm_regions_for_config`, what the shard boots with) and the oracle
  rosters the HOME SUBTREE only. MEASURED: the derived-budget test went from unmeasurable (stopped
  at 31 min debug / 20 min release) to 8.4 s in release, PASS. (3) Every `just` process gate that
  spawns a cluster now runs `cargo test --release` (14 targets): the profile the owner ruled flights
  run in, and the one whose boots fit the gates' own deadlines.
  ★ ITEM 1, THE GATES FLY HULLS — IN PROGRESS. `flight::berth_test_hull` (the same `vd-build-ship`
  tool the dev cluster uses, at a rating the test chooses), `board_hull` (a 2 % creep: at the dev
  cluster's 1 km/s foot speed a full-speed walk passes THROUGH a hull inside one tick and a
  closed-loop walk-to overshoots and runs away under the walk ramp — measured 115 km, then 9.6 km
  braked), and `fly_hull_to` (body-frame guidance off the drawn box's centre: aim by a
  stopping-angle controller on the hull's turn rating, push, brake on the measured acceleration).
  Probed on the dev cluster through `examples/hull_flight_probe`. FOUND ON THE WAY: a hull rated at
  9e12 m/s² takes 2 % of stick as 1.8e11 m/s² — the boarding creep's residual stick shot it to
  warp before the stop arrived; the test hull is berthed at 1e8 m/s² / 0.3 rad/s².
  MEASURED ON THE PROBE (six flights, 2026-09-05 evening): boarding by creep is reliable; the
  fly controller is NOT. (a) The hull's SPIN NEVER DAMPS — there is no angular drag in space and
  no attitude hold, so every turn command ever given accumulates; the berthed hull was measured
  turning ~1 rad/s at rest after a probe, and a 10 Hz test-process loop with 100–300 ms of pose lag
  cannot stabilise a double integrator at that rate (the aim oscillated between ±1 and ±2.5 rad
  for minutes on every gain set tried). A human pilot in the window damps it by eye. (b) The drawn
  boxes' centres in the hull's frame ROTATE with that spin (the same fact seen from the client),
  so a body-frame bearing is exact but useless at that rate. (c) The pilot's own row inside the
  hull did not move on a strafe or a push in the manual test while the hull did turn — UNRESOLVED
  (the dev state's own row may be the stale entity feed; not chased). DECISION: the closed loop
  belongs on the CLIENT at tick rate, next to `walk_to`/`look_at` (`client-harness::nav`): a
  `FlyTo` dev-control request that reads the render snapshot's own orientation and the target
  box at 50 Hz with no round trip — the shape every other nav helper already has. OWED with item 1.
  ALSO A GAME QUESTION, for the owner: a hull's attitude hold. The 2026-08-27 ruling names *"a
  ship's own safety block slows it (and a player may switch that off)"* — the same block should
  null the spin when the stick is released; today a released stick leaves the hull turning forever.

### D-REACH-1 🟧 REACH: one radius per realm, tested by its parent — steps 1–3 LANDED, 4–6 OWED (owner ruling 2026-09-02, `owner_decisions_2026-09-02_reach.md`)
- **LANDED 2026-09-02 (measured green on the cluster: `world_from_inside`, the hull subject).** A player crosses into a player-built hull forty metres from the spawn and sees the stars and the star system's own children. What landed, each pinned:
  - **The instrument** — `DevState::stars_drawn` (the drawn count beside the held count) and `DevState::sky_anchor`; the gate `world-from-inside` in the pre-merge list. Before it, every gate read the HELD count and certified a black sky.
  - **R9 step 2, the chain reaches the root** — the hop is the authored placement (D-WINDOW-4 resolved above; wire minor 24, a mesh flag day).
  - **R9 step 3, the galaxy is always in the picture** — decision 2: the gateway lifts the observer's origin up the chain into the galaxy's frame and states it as `RealmSnapshotDatagram.sky_anchor` (~130 B/tick, on every chunk; a tick with an anchor and no row still ships one datagram); the client builds ONE star cloud from the catalogue (`SkyDraw::points_from`, every star, nobody left out) and places it per frame (`sky_cloud_transform`, f64 folded and narrowed once; the shader applies the model transform it used to ignore); the re-anchor and its `anchor_of`/`points_around` are DELETED; a rebase bound is derived (`sky_rebase_bound_m` ≈ 7.9e18 m, wider than the galaxy).
  - **Decision 3, a window ships only the children in range** — `emit_window_rosters` runs AFTER the range fold and ships, per window, the children in that window's range plus the hop child (movers always; everything with no live band); the relay ships the same set. MEASURED before: the galaxy shipped 233,220 rows + 233,220 markers per keep-alive (22 MB) and shed for minutes.
  - **A datagram over the budget is split** — the window frame is partitioned exactly as the entity snapshot is, the hop on every chunk, and the gateway MERGES chunks of one stamp. MEASURED: the home system's hull-window frame was 1,420 B against 1,200 and EVERY one was dropped by the transport (counted where no gate reads).
  - **R8 item 1, the first half** — the placement book is LAYERED (statics authored once and shared by `Arc`, movers per tick); the fold visits CANDIDATES by lookup (`ChildIndex::candidates_along` the looker's lead, `children_reachable_from_outside` for the interest proxy) plus the latched; `aoi_candidates_visited` and `aoi_full_walks` are gauges. MEASURED on the galaxy (debug, seed 0 world, one occupant at the centre): 1.9 s → 250 ms per tick; on the cluster (seed 2298): 61 ms mean, still OVER the 20 ms budget.
- **OWED:**
  - **R9 step 4, REACH itself — BUILT 2026-09-04 (owner: "1. agree" to the brightness rule; unit-green,
    flight OWED).** What landed, against the plan below: the arm is `ReachStated { child, child_fence,
    at, size_reach_m, light_reach_m }` (mesh minor 29, discriminant 43, producer-less reliable) — TWO
    whole-metre numbers, not one, because the owner's rule needs the parent to drop a child's LIGHT
    while keeping its SIZE, and a folded single number cannot be un-folded; the parent
    (`RealmRegions`) keeps `reach_stated`, re-bands a stating child with `AoiConfig::with_spin_up`
    and re-inserts its leaf (`set_child_reach`, O(log n)); a child that never states (a dormant
    planet) is banded by its planted light (`with_child_light`, the same datum the marker roster
    carries) — so a bright planet wakes at ~50 AU, not at 76 radii; a realm that draws its
    children's light (`with_lights_children`, the galaxy — its star field is shipped once by the
    gateway) tests and folds SIZE only; a realm's own reach is `max(own look reach, max over
    children of distance + reach)` per kind, kept as two sorted multisets so a statement costs
    O(log n) on the galaxy (`own_reach`); the emitter (`reach::emit_own_reach`) states upward once
    per change, `Retained`; the consumer (`reach::on_reach_stated`) carries the three up-lane guards
    and counts (`reach_*`). Constants: the dot angle moved to `vd_core::geometry::
    VISIBILITY_THETA_MIN_RAD` (1.5°, the world solve now reads the same one); the limiting
    magnitude `vd_core::look::LIMITING_MAGNITUDE` = 6.5 beside the Sun's absolute magnitude and the
    parsec; `light_reach_m` pins a Sun at ~70 ly and an Earth at ~50 AU. OWED (R8): the galaxy with
    one looker on this build, and the bright-body count — how many shards wake per pilot now that a
    planet reaches by its light.
  - **R9 step 4, REACH itself — PLAN 2026-09-04 (shown to the owner, one decision pending):** ONE new
    arm `ReachStated { child, child_fence, at, reach_m }` (every realm, on change, producer-less
    reliable beside `ChildFacts`; a planet has no drive facts to ride with, so the reach gets its own
    arm — R6's "beside mass" read as the same lane class); the parent keeps `reach_of: child → m`,
    rebuilds that child's band with `AoiConfig::with_spin_up(reach)` (the tear-down keeps the wider of
    its old ratio and its old gap, so a velocity-safe band stays safe) and re-inserts the child's leaf
    in the tree — no rebuild, no walk; a realm's own reach = max(own look reach, running max over
    children of distance + stated reach), kept as a sorted multiset so a child's statement costs
    O(log n) on the galaxy; the own look reach = max(size: `visibility_reach_m(extent, θ = 1.5°)`,
    brightness: the distance at which luma_lsun reaches the naked-eye limit, magnitude 6.5 — a new
    perception constant beside θ, cited). THE DECISION ASKED: a star's brightness would wake every
    system within ~70 ly (thousands per pilot); proposed rule — brightness counts toward a realm's
    reach only when no ancestor already draws that light, and the galaxy's star field draws every
    star's, so a star system's reach is its disc plus its planets' reaches. Measurements owed with it
    (R8): the galaxy with one looker; shards awake per pilot.
  - **R9 step 4, REACH itself (status before the plan)** — no realm states a reach yet; the band is still the wake band (bound × cot(θ/2)) and the angle is still 1.5°; the child→parent reach datum (approved, R6) is not on the wire. The dot angle re-solve and its cost measurement are owed with it.
  - **R9 step 5 — BUILT 2026-09-04 (owner: "7. approved"; after step 4 landed, as the ruling binds):** the
    parent's marker is DELETED at its source — `current_bodies` states the realm's own look and nothing
    about its children (direct windows and the relay alike) — and at its two reads in the gateway:
    `body_tag` no longer answers `Marker` (the variant is gone) and `scene_bag` no longer falls back to
    a parent's luma bag. A realm that states no look is tracked and not drawn. The wire arm
    `BodyStmt::Marker` stays reserved and a marker a shard still stated is admitted, stored and never
    read (the store is a tombstone that leaves with the arm's reservation); the client's marker body
    kind stays as dead decode. The occupancy bit still builds the proxy observer (lawful under R5).
    Also from the eighth flight: `merge_static` no longer folds a static row of a realm the shipped
    membership does not name, so a child that left the range stops riding every later level with its
    stale stamp (the ~12/s `window_instant_mismatch`). Flight OWED: the ninth.
  - **R9 step 6 / R8 item 1, the second half** — the galaxy's tick was 61 ms mean against 20 ms; the lookup fold, the sphere filter, the query memo and (2026-09-04) the R*-tree index each took a lever — see D-MOVE-2's 2026-09-04 status for the tree. The per-candidate cost (~22 µs each: a `RealmCoord::child` allocation, three `BTreeMap` inserts, the interest emitter) and the 11,275 candidates one centre occupant collects on the seed-0 world (index cell 61 ly, because a system's wake band is its gravitational bound × 76 — the widest is 18 ly) are the two levers. Neither is a number to argue: the probe `measure_the_galaxy_shards_tick_against_its_census` (ignored, hand-run) prints both.
  - **The band re-solve (D-MOVE-3)** is what makes the index cell small; it is the owner's open item and is NOT touched here.
  - ★ **RE-MEASURED 2026-09-05 (item 5 of the owner's list) — the 61 ms / 11,275 figures above are STALE.** They
    predate the lookup fold, the sphere filter, the query memo, the R*-tree index and the reach re-band (the
    galaxy excludes its children's LIGHT reach — it draws every star itself — so a system's band is its disc
    plus its planets' reaches, not 76 × its gravitational bound). The probe
    `measure_the_galaxy_shards_tick_against_its_census` on this tree, release: the galaxy boots 279,380 direct
    children in 1.0 s; a tick with one occupant at the origin is 20 µs and visits 5 candidates; the widest
    child extent is 2.24e15 m (0.24 ly); an empty tick is 2 µs. LIVE, the dev cluster: the galaxy shard's
    pace line reads mean 0.10–0.15 ms with 2 candidates (a headless pilot standing in System 7, 90 s), and the
    sixth owner flight's status (2026-09-04, warp through the galaxy) read every shard under 0.12 ms mean with
    the galaxy asking the tree for 3 candidates. Budget 20 ms. Item 5 is CLOSED as measured; the two levers
    named above stay valid as levers but nothing binds today. UNMEASURED today: a warp leg's pace line on this
    exact tree (the agent flight script did not board the hull — its `walk_to` returned a state dump instead
    of walking, script drift, not a server fault); the owner's next flight supplies it (read the galaxy's
    `tick pace` lines in `realm-logs/realm-<galaxy node>.log`, `OVER BUDGET` must stay 0).
  - **The second subject of the gate** — the planet subject is `#[ignore]`d with its measurement (a walking occupant at 1.0e3 m/s cannot reach a planet 6.1e10 m away; the governor is deleted). It runs when a hull flies there (M-C) or the spawn moves (G10). The look/acceptance gates that fly the same leg are in the same state and were NOT re-run in this pass — UNMEASURED.
  - **A static attach onto a BUILT realm** still gets a one-realm chain, counted (`attach_lineage_unresolved`). A static attach onto a SEEDED realm now derives the whole chain from the gateway's own forest (2026-09-02, found by the star-sky gate on the dual cluster: 233,220 stars held, none drawn, `sky_anchor = None`); the flown path (login + crossing) never needed it.
- **PRE-EXISTING REDS, measured on the previous commit (41b0ba0), NOT introduced here:** `flight_table` (two tests: a leg that holds no cruise under the deleted governor, and a boundary count pinned at 51 on a three-system world that now counts 3,500,645 bodies), `dual_cluster_crossing_smoke` (the same no-cruise leg), `ship_flight_e2e::a_ship_states_a_push...` (the test never plants the hull's `OwnBody`, which the last commit made a gate), and `frame_conversion_e2e::no_level_of_the_chain_can_place_itself...` (the test plants a second planet under the star system on that very shard and expects the system not to place it — identical left/right on the previous commit, built and run in a scratch checkout 2026-09-02). Also: `frame_conversion_e2e::the_linear_carry_chord_error_sizes_the_placement_skew_cap` regenerates the whole 3.5M-body forest inside its own body and ran 18+ minutes at full CPU without finishing — skipped in this pass, the quadratic shape this arc has removed elsewhere.
- **`star_sky_pixels` RE-RUN, and its red is PRE-EXISTING (measured 2026-09-02).** On this tree the gate reaches its pixel probe (the anchor arrives, the stars are drawn) and fails on ONE star of 77 in view: `System(35151578967932295)`, a faint red dwarf (5.3e-3 L_sun, drawable under the shipped law at crop 1.4 px) at the frame's bottom edge, with no lit pixel within 12 px. The SAME star fails the SAME probe on the previous commit (built and run in a scratch checkout). The kept capture, analysed: 65,120 lit pixels; over all 77 expected positions the nearest lit pixel is at median 2.2 px, p90 6.9 px, with no radial trend (inner third 2.3 px, outer 3.6) — the one cloud is placed and turned exactly where the seed puts the sky, and this one star's absence is an OPEN item of its own (why a drawable near star paints nothing there is unexplained by reading). The gate now keeps its capture when `VD_KEEP_CAPTURE=<dir>` is set.
- **Process gates NOT re-run after this pass (UNMEASURED):** `two_ships`, `render_crossing_smoke`, `look_pixels`, `rlm_demand_login`, `window_parity`. The deterministic `cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` are green.

### D-SL1-2 🟥 SL1 REWRITTEN — a realm MAY be told where it is (owner reversal 2026-08-24), and the three fences that make it safe are UNBUILT

- **WHAT CHANGED, and it is a law change, not a feature.** SL1 read "ONLY THE PARENT KNOWS POSITIONS —
  a realm never knows, stores, derives or is told its own position … not even as a zero field, because a
  field's PRESENCE is the leak" (and this row's own parent clause records the 2026-08-05 deletion that
  text caused). The owner rewrote it on 2026-08-24: **a realm is TOLD where it is; it never DECIDES where
  it is.** The parent stays the ONLY writer; the child holds a stamped, read-only reading; the child never
  derives, adjusts or states its own placement; ONE HOP ONLY (what you are told about yourself you never
  pass on, and you are never told your parent's placement, so no realm can chain hops into an absolute).
  The full text is in `CLAUDE.md` SL1 with the owner's stated reasoning.
- **WHY (the owner's reasoning, recorded so it is not re-litigated).** The old law forced TWO mechanisms
  for one job — "contents plus your own eye" to a renderer, "a view composed for you" to a realm — a fork
  by RECIPIENT KIND. Worse, the composed-per-observer form costs contacts × observers, and the sky arc
  measured that product fatal (150,000 rows per player per tick, an un-split datagram the gateway simply
  never receives). Being told where you are turns the product into a SUM for every consumer.
- **STATUS 2026-09-05 — fences 1 and 2 BUILT, before the datum, as the ledger demands; fence 3 waits
  for the datum's first consumer.** The vocabulary the datum will use is ONE constant,
  `vd_core::pose::SL1_TOLD_PLACEMENT_VOCABULARY` (`ToldPlacement`, `PlacementTold`, `told_placement`);
  both fences read it, so a new spelling must be added there first or both go blind together.
  (1) The wire pin `no_living_inter_shard_payload_carries_a_realm_placement_or_a_centre` is
  RE-POINTED at clause 3: a living arm may carry a placement only if it is a DOWNWARD statement
  named in the vocabulary (`DOWNWARD_STATEMENTS`, empty until `PlacementTold` lands); its
  observed-failing control is the five tombstoned lanes the scanner must still catch. (2) The
  module fence `sl1_the_placement_containment_and_crossing_machinery_cannot_name_a_realms_own_
  position` (`tests/tests/crate_isolation.rs`) scans nine modules — core's `frame.rs` and
  `placement.rs`; sim's `containment.rs`, `placement.rs`, `conversion.rs`, `crossing_receive.rs`,
  `exterior.rs`, `saga_arms.rs`, `transient.rs` — for the vocabulary, with a built-in control (a
  planted line the scanner must catch, and a child's `FramePlacement` it must not) AND an observed
  real failure: a `ToldPlacement` line planted at the end of `containment.rs` failed the fence with
  the message "SL1 VIOLATION in crates/sim/src/stub/containment.rs" (2026-09-05), and passed again
  once removed. (3) The staleness rule needs the reading and its consumer; it lands in that slice.
- **WHAT WAS OWED — three fences (the original text, kept):**
  1. **The outbound vocabulary fence.** Nothing a realm sends may carry its own placement. A wire-vocabulary
     pin of exactly this shape already exists (the Slice-C2 deletion planted one to replace the retired SL1
     self-placement filter); it must be re-pointed at the new clause 3 and given an observed-failing control.
  2. **The dependency fence.** The placement, containment and crossing machinery may not NAME a realm's own
     position — a crate/module rule with a control that fails if it is re-opened, the same shape as
     `sl4_the_crossing_path_cannot_name_a_motion` (`tests/tests/crate_isolation.rs`), which is the ONLY
     enforcement in this codebase that has actually held.
  3. **The staleness rule.** Every self-placement reading carries its instant and is REFUSED past a DERIVED
     bound. Absent data stops a consumer; stale data makes it act confidently on a wrong number — the worse
     failure, and the likeliest one to reach production.
- **DO NOT BUILD THE READING BEFORE THE FENCES.** Landing the datum first and the fences later is precisely
  how the 2026-08-05 defect happened the first time.
- **WHEN:** with the first consumer that needs it (the ship's sensor/contact view, the autopilot, and the
  client eye row of the sky slice). Its own slice, with the three fences in the SAME slice.
- **THE SL2 QUESTION — RAISED AND ANSWERED (owner, 2026-08-24), recorded so it is not re-opened.** I first
  flagged this as unresolved and was WRONG about why. SL2 governs REALM-TO-REALM; the gateway is not a realm,
  and a shard handing its own occupants to it was never a crossing (it is how any player has ever seen
  anything). So SEEING another realm's people is lawful with no ruling: each shard streams its OWN occupants
  to its OWN subscribed clients and the GATEWAY composes one ready-to-draw picture per observer. That is the
  lane [[D-RLM-18]] already specifies and owes. The owner's line for the SIMULATION side: **CONTACTS ARE
  REALMS, PEOPLE ARE SEEN** — a ship's systems track ships, stations and bodies (placements the parent
  authors), never individual people; acting on a person is a projectile CROSSING, i.e. the transfer
  machinery. SL2 is clarified in `CLAUDE.md`, not amended. **What this makes urgent rather than optional:**
  [[D-9]] (snapshot emit is a whole-realm broadcast with NO per-entity interest filter, double-encoded) is
  the enabling defect for a crowded scene, and the read-sub lifecycle, the range/line-of-sight bound on
  streaming a hull's INTERIOR to outsiders (an exploit surface, not a rendering detail) and the mixed-tick
  join are all owed with [[D-RLM-18]].

### D-SCALE-1 🟥 THE COORDINATE-SCALE MODEL — owner's rulings 2026-08-05 (supersede the universe-wide-absolute assumption)
- **THE MODEL (owner, binding).** Every realm has its OWN coordinate frame; nothing anywhere holds a universe-wide absolute (that follows from the "a realm is never told where it is" reversal, realm-unification entry above). **Compression is applied BETWEEN realms only — a star system keeps its TRUE size and true internal distances; the GAPS between systems are shortened.** The galaxy frame is therefore a CHART, deliberately not a scale model. Rationale: a uniform compression would make crossing a system boundary a jump in apparent scale (outside, the system is drawn compressed; inside, full size — the camera sees a zoom at the seam), which is the opposite of seamless.
- **WHY THIS DISSOLVES THE "REAL INTERSTELLAR DISTANCE" SLICE (arc slice 13) AS ORIGINALLY SCOPED.** The precision problem was never about distance, it was about MAGNITUDE. Per-realm frames keep every number small. Worked numbers (f64 relative precision ≈ 2.2e-16, so the representable spacing at magnitude M is M × 2.2e-16): out to ~1e13 m (past Pluto — a whole planetary system) spacing is ~2 mm, fine; at 1e14 m ~2 cm, marginal; at 1e15 m ~0.2 m, walking becomes a staircase. So REAL coordinates inside a star system are sound out to roughly the Kuiper belt, which is exactly what the owner requires. An Oort cloud would need the same treatment as the galaxy.
- **SEPARATION IS THE BINDING CONSTRAINT, and it is a CORRECTNESS one, not a visual one.** The compressed gap between two systems MUST exceed a system's own diameter, or their authority spheres overlap and "which realm am I in" is ambiguous. A planetary system is ~1.2e13 m across, so gaps of ~1e14 m give clean separation — a compression of ~400× from the real 4.0e16 m to the nearest star. At that separation a Sun-sized star subtends ~3 arcsec (the eye resolves ~60), so systems ARE dots; and travelling one gap swings a star three gaps away through ~18°, so PARALLAX is strong — stronger than reality, because the travel distance is comparable to the spacing. **Owed:** a boot fence that SIBLING regions do not overlap (the existing fence only checks that a child fits inside its parent).
- **RULING — the system boundary is NEVER DRAWN as an object.** A system's authority sphere is far too large to ever be a dot; rendered as an object, systems would always read as overlapping bubbles. It is an authority construct, not a visible one. (Subsumes the cosmetic ticket "stop system boundaries reading as stars".)
- **RULING — star brightness is a KNOB, not a derived value.** Compressing ~400× makes a neighbouring star ~1.6e5× brighter than reality; physically it would be blinding. The owner has NOT decided the curve and wants to see it in game first. Whatever lands must expose it as a live knob, never bake a chosen curve in.
- **RULING — ALL MECHANICS WORK IN DEEP SPACE, and it is slow by design.** With warp disengaged, manoeuvring, repositioning and space-walking must all work between systems; it simply takes ages, "almost like standing still in the galaxy". You reposition and re-engage warp to reach a system. **THIS IS THE RULING THAT FORCES THE NEXT ONE**, and it reverses an earlier (wrong) claim of mine that no coarse-coordinate machinery is needed anywhere: that claim held only while everything far from an origin moves FAST (precision requirements scale with step size, and at warp a tick covers ~1e9 m so hundreds of metres of rounding is irrelevant). Slow movement out there breaks it — a smooth space-walk needs the quantum well under a centimetre per tick, i.e. magnitude under ~9e12 m, which is less than ONE inter-system gap.
- **RULING — THE GALAXY IS A LATTICE OF CELL-REALMS, decided 2026-08-05, NOT deferred.** Each cell is an ordinary realm with its own origin, so coordinates are always local and slow movement stays smooth anywhere; crossing between cells is an ORDINARY crossing on the existing machinery — no new concept, no warp code path. This also answers the occupancy problem (a single galaxy realm would put every inter-system traveller on one server) — both reasons converge on the same answer.
  - **THE SIZING TENSION, unresolved and owed to the design pass:** a cell must be large enough to CONTAIN a star system (needs a half-extent ≳ 1e13 m, since the system's own sphere must nest inside it per the existing boot fence) yet small enough that local magnitudes stay under ~9e12 m for a smooth space-walk. Those two bounds nearly touch. Centring each system in its own cell makes it work with a few millimetres of quantisation at the cell corners (~3.5 mm against a ~20 mm per-tick walk step) — tight but probably acceptable. **Do not improvise this; it wants a design pass with the numbers in front of it.** The planted coarse+fine coordinate type STAYS in the wire as the cheaper alternative fix if subdivision proves awkward — it costs nothing to leave in place and removing it forecloses an option.
  - **WHEN:** the decision binds NOW (nothing may assume the galaxy is singular; addressing stays on the full lineage path). The BUILD waits for something to fly — there is no ship, no EVA and no warp yet, so building it now would be machinery with no user, which is precisely the mistake the deleted "received seat" slice made.

### D-RLM-17 🟩 A CO-HOSTED occupied realm draws LIVE at one point — closed by the cross-lane space alignment (crossing-render slice, 2026-08-12)
- **WHAT was owed, briefly:** a realm co-hosted on its parent's shard (an area on its planet's shard) had no drawn-position lane while occupied — the entity lift hid the delivered flip, and the realm lane (grouped by the dot's STORED frame) either starved the observer or spoke a different space than the lifted pose.
- **HOW it closed, same slice:** the realm lane's direct-emit observers are now grouped by the SAME space the entity lift delivers (own frame for a dot standing in a placeable child), so a co-hosted stander receives its realm's row live, in the one space its own pose arrives in — the row is no longer 'the recipient's own row' in that group, so SL3's own-row drop does not starve it. Measured: `the_box_and_the_thing_standing_in_it_draw_at_one_point` half (2) — the OCCUPIED area's box keeps turning while stood upon and draws at the occupant, gap 0.
- **What remains (the OWN-shard case):** a realm with its OWN shard flips the delivered frame at the crossing; the client's one-space rule forgets the old space and the realm draws from its streamed outline at the occupant's origin — pinned in vd-client's crossing-flip tests. The flush now converts along the full hosted chain (ascent + descent, each link in its authoring parent's context, authorship-constrained), which also cured the pre-existing co-host strand the §4i anchor fix left (`crossing_same_node_e2e`, 3/3 green again).

### D-RLM-16 🟩 The spin-up-ahead scenario REWRITTEN against the multi-star ring — green (Step 5 slices B+C, 2026-08-13)
- **WHAT was owed, briefly:** a truthful spin-up-AHEAD gate. The old scenario presumed an outer planet CULLED at login; 32fcc5c's wider visibility (8°→1.5°) made every planet of the home system visible at login, so the premise was unsatisfiable inside one system of THE world and the test sat expected-red.
- **HOW it closed:** rewritten (never deleted — the standing scenario law) as `a_flying_occupant_streams_a_neighbour_system_in_ahead_then_the_vacated_realm_is_reaped` (`crates/bins/tests/rlm_demand_login.rs`): the genuinely culled realm is the NEIGHBOUR STAR ~12 km down the ring (asleep by construction — the ring is 1.05× the wake radius). The occupant exits the home system, parks INSIDE the neighbour's wake band ~10 km outside its shell, and the gate asserts (1) the neighbour spins up AHEAD (spins climb above the settled login baseline), (2) its planet BOXES stream into the drawn scene (slice C's up-shape mirror), (3) those boxes MOVE (the up-rows lane animating them at full orbit speed) — all while `location` never leaves the between-space; the return leg asserts the vacated neighbour reaps BEHIND (above the settled reap baseline, so the home system's own outbound reaps cannot satisfy it). First green 2026-08-13, ~31 s. It doubles as the owner-flown gap's gate ("approaching another star system, its planets were not loading").
- **The load-bearing fix it measured:** the parent-resolution HeadRead sat BELOW `aoi_decide`'s zero-occupant early-return, so a realm nobody had ever entered could never resolve its parent — its whole up-observation lane (rows + outlines) stayed structurally mute. Moved above the Empty return; the exited-system case only ever worked because the parent node had been resolved while the realm was occupied and stayed cached.

### D-RLM-18 🟥 TRUE remote avatars — the lawful read-sub lane (owed by Step 5 slice E's accepted loss; owner-gated)
- **WHAT is deferred:** rendering another realm's occupants as FIGURES (not just their realm's box) at steady state. Slice E (2026-08-13, minor 13) deleted the entity relay because it shipped occupant poses across realm boundaries (SL2); the owner accepted the loss (design §3/§8.2): a sibling realm's occupants are sub-child-resolution at parent scale, and the occupied realm's box is their proxy (SL7). What is OWED is the lawful replacement when the product wants real figures again: the CLIENT holds read subs on visible realms' shards, and each shard streams its OWN occupants to its OWN subscribed clients — no pose ever crosses realm-to-realm; the client-side frame relation ties into the server-authoritative composition rework ([[project_floating_origin_plan]]: the client must never compose two shards' frames itself, so the SERVER side of this lane must compose ready-to-draw rows per subscriber).
- **WHY deferred:** designed-but-unbuilt in the Step 5 design (§3); it is new client-facing machinery with its own AoI/sub lifecycle, and SL6 gates it on the owner asking for it. Nothing in the game breaks without it for the LEAVER (their own avatar rides the dual subs cleanly); a BYSTANDER's view of the leaver is the frozen ghost fill and then, until [[D-4]](a) lands, a stranded frozen track — the D-4 slice E escalation.
- **WHERE it lands / WHEN:** owner-gated, earliest after slice F (SpawnV2 cutover) and the floating-origin server-authoritative rework it composes with. **Pin:** the `EntityInterest` tombstone note (`crates/wire/src/intershard.rs`) names the future lane; `two_players_in_two_realms_are_each_drawn_only_by_their_own_realm` (`tests/tests/frame_conversion_e2e.rs`) is the scenario that flips when it lands (the absence asserts become presence-via-read-sub asserts).

### D-RLM-15 🟥 A REALM-keyed transfer lock has no producer — the retired teardown clause must return WITH one (arrival-race slice 9c, 2026-08-05; design `scripts/arrival_race_8_9_design.md`)
- **WHAT is deferred:** a directory `in_transfer` lock taken on a **Realm** key, and the teardown clause that reads it. The clause existed in `teardown_ready` from RLM Step 3 and was believed to shield a realm somebody was being handed into. It never could: `lock_transfer` is only ever called on the transfer SUBJECT's key (an Entity), `locks_directory_key(Transient) == false` takes no lock at all, and `commit_cas` clears even the subject's lock at the commit point — so on a Realm head the flag is permanently `None`, the clause is permanently true, and it shielded nothing. The unit test that proved it load-bearing hand-built a `TeardownFacts { in_transfer: true }` the running system cannot produce. **DELETED in slice 9c** rather than left as a comment correction, so nobody reasons from it again.
- **WHY it is safe to delete NOW:** the protection it was believed to provide is real and now actually exists, by a different and stronger mechanism — the ARRIVAL SHIELD (`vd_sim::rlm::desired_alive`'s third arm, fed by `SagaRuntimeRes::arriving_dest_realms`). That set is derived each sweep from the orchestrator's own live hand-offs, in the same process on the same tick, so it has no lock to set, no message to lose and no window to race. It also covers what a realm-keyed lock never would: transient batches (which take no directory lock) and the whole post-commit tail (where the subject's lock is already cleared).
- **WHEN a realm-keyed lock genuinely becomes necessary:** realm MOBILITY — a realm that is itself the SUBJECT of a transfer (a ship's interior moving hosts, P8; a system handed between clusters, P10). Today the only realm-keyed saga context is the transient batch, which is not a realm transfer. **Binding condition for its return:** it lands WITH its producer (the `lock_transfer` call on a Realm key) AND an end-to-end test that drives a real realm-subject transfer through it — never as a clause with no writer.
- **Where:** `crates/sim/src/rlm.rs` (`TeardownFacts`, `teardown_ready` — see the RETIRED comment), `crates/node/src/saga_runtime.rs` (`locks_directory_key`, `commit_cas`).

### D-45 🟧 Containment realm-membership — owed slices + P4/P5/P8/P10 (task #135; C-1/C-2 `8ef619c`, C-3 landed 2026-07-16, C-5/C-6/C-6b/C-6c landed 2026-07-16; design `scripts/containment_realm_membership_design.md`)
- **LANDED:** the pure containment layer (C-1/C-2) + the sim swap (C-3: detector rewrite, `RealmRegions`/`ContainmentProgress`/`RegionMembership`, deleted portal fns, ~40-test migration) — green + 100% region/branch, −764 lines. INERT in prod (`RealmRegions` default-empty) UNTIL C-6b flips the seed boot LIVE (below).
- **C-4** the moving-container reverse-cross bound: `k_dwell ≥ commit-latency` is enforceable only at P3 (in-proc 1-tick commit); the cross-host bound (the redelivering transport's ack window) is LEDGERED to P6/P7 — do NOT claim a `debug_assert` forecloses it. Reverse-cross stability rests on (band dead-zone)+(per-entity `Occupied` serialization)+(fresh-state dest self-heal), NOT the cooldown (which does not survive the ownership handoff).
- **C-5 🟩 LANDED:** `vd_core::geometry::guard_regions_nest(regions, max)` boot validator (exactly one `parent:None` root, unique `.realm`, parents resolve, no cycles, count ≤ `MAX_REGIONS`=64), each reject arm `expect_err`-tested. The detector's defensive rootless no-op stays covered by `a_rootless_region_forest_is_a_safe_no_op`.
- **C-6 🟩 LANDED (worldgen forest):** `vd_core::worldgen::{realm_regions_for(seed), realm_neighbourhood_for(seed, hosted_realm), MAX_RENDERABLE_EXTENT_M}` — the WALK-scale Universe⊃Galaxy⊃{System 7⊃Planet 7, System 8} forest; `realm_neighbourhood_for` returns own+ancestors+owned-children, **NEVER siblings** (System 7 shard → {Universe,Galaxy,System 7,Planet 7}; Galaxy shard → {Universe,Galaxy,System 7,System 8}; System 8 shard → {Universe,Galaxy,System 8}). The CORRECTED sibling-routing model: a sibling crossing routes THROUGH the shared Galaxy parent (leave System 7 → land in the Galaxy ancestor; the Galaxy shard, owning both systems as children, sees the entry into System 8), so no shard needs a sibling in its scan.
- **C-6b 🟩 LANDED (bins BOOT LIVE + client render single-source):** `shard.rs` now computes the seed neighbourhood from `VD_UNIVERSE_SEED` (default 0) via `realm_neighbourhood_for`, fences it with `guard_regions_nest` (fail-LOUD at boot), and plants it — **the containment detector is LIVE in prod** (no longer inert). ~~The INTERIM `regions_for_source_plant` adapter is REMOVED (renamed `override_regions_for_boundaries` — see the RETAINED-OVERRIDE note); `resolve_realm_boundaries`/`VD_REALM_BOUNDARIES` are RETAINED as an OPTIONAL OVERRIDE that plants an authored born-inside CHILD crossing shell, so the process-tier `dual_cluster_crossing_smoke` / `render_crossing_smoke` still prove a DIRECT source→dest re-home over real binaries WITHOUT standing up a Galaxy shard.~~ **⚠ AMENDED (2026-08-14, SL5 sweep — D-WORLD-4):** the RETAINED-OVERRIDE decision is REVERSED and the whole override family is DELETED — the boundary-file env substitution in `shard.rs`, `resolve_realm_boundaries`/`RealmBoundariesError`/`parse_realm_boundaries`/`guard_boundaries_in_realm`, `override_regions_for_boundaries`/`override_containment_band`, `source_crossing_boundaries`/`write_source_boundaries`/`resolve_source_boundaries` + the launcher injection, the `crossing_playground` module (box consts/trigger/scene/`write_fixtures`) + `emit-crossing-fixtures`, `write_seed_regions` + `emit-seed-fixtures`, and `write_visual_regions` + `emit-visual-fixtures`. Its "single-sourced with the client's `--realm-boxes`" claim was the exact mechanism by which a Dual cluster simulated a 150 m star system while drawing an authored 40 m one. The crossing smokes now fly THE world's own home shell (Dual, zero injection), and the ONE emitter is `write_world_regions` (`emit-world-scene`). Client render: `RealmScene::from_regions(&[RealmRegion])` + `from_regions_json` project the FINITE renderable regions (`Boundary::finite_extent() <= MAX_RENDERABLE_EXTENT_M` — SKIP the ambient Galaxy/Universe shells), single-sourced with the shard via `vd_bins::write_world_regions`; the client `--realm-boxes` tries `from_regions_json` first, falls back to the legacy `from_boxes_json` reader (still exercised by unit fixtures).
- **C-6c 🟩 LANDED (3-shard round-trip gate):** `tests/tests/three_shard_round_trip_e2e.rs` — a REAL 3-shard harness cluster (orchestrator + gateway + System 7 + **Galaxy** (`System(1)`, granted so `head(Realm(Galaxy))` resolves) + System 8), each shard planted with its seed neighbourhood, drives a subject origin→50→100→50→origin and asserts the AUTHORITATIVE HOLDER flips through the FULL chain **BOTH WAYS**: System 7 → Galaxy → System 8 → Galaxy → System 7 (the RETURN legs 8→Galaxy→7 are the reverse-cross proof). Each leg is a REAL autonomous saga-driven re-home (zero `trigger_transfer`). Plus the unit tier that ACTUALLY pins the container behaviour at HEAD — corrected 2026-08-14 (Stage-C audit :161: this clause used to name `stub::tests::{symmetric_recross_resolves_the_full_container_sequence_both_ways, escape_soi_lands_in_the_immediate_parent_not_a_skipped_ancestor}`, two tests DELETED with their `seed_container_at` descent oracle in the rehome arc (rehome_one_mechanism §4b) while this 🟩 clause stood stale): the `container()` fold itself is pinned in vd-core (`geometry::tests::container_folds_the_deepest_member_from_the_root_identity`), and the sim tier pins the RESOLVED DESTINATIONS via the emitted `to_realm` — the observable the crossing actually ships — in `stub::tests::containment::slice4b_an_undock_re_homes_outward_to_the_parent_realm` (an escape lands in the IMMEDIATE parent; the escape forest plants the root as a skippable ancestor) and `stub::tests::containment::slice4b_a_dock_and_undock_resolve_contrasting_destinations` (the symmetric dock/undock contrast). **TIER + SUBJECT honesty:** the gate runs at the HARNESS tier (the process cluster machinery is fixed source/dest → N-shard process roster is D-44) with a TRANSIENT subject asserting `owned_transients` (the HR2 ownership head — a transient has no directory `OwnerRecord`). A DURABLE dot's DIRECTORY Entity-head multi-hop was BLOCKED at the harness tier (a `Cutting`→`CutTimeout` on the 2nd hop) — **RETIRED by S3 (server-timed cut, see below)**; NO session-route migration was needed (the gateway `store_commit` already re-routes on the current authority per hop — the only stall was the saga's dependency on a CLIENT `CUT_MARKER`). A FRESH transient per leg avoids the static-`Fence(1)` batch-id collision (`crossing_transfer_id(entity, src_realm_fence, 0)` re-mints a colliding id when the SAME entity re-visits a shard; production fences ADVANCE per re-home). **OWED at the process tier (D-44):** the 3-shard PROCESS cluster (roster generalization).
- **S3 🟩 LANDED (server-timed cut — durable MULTI-HOP round-trip):** the transfer cut is now driven ENTIRELY server-side — the gateway `apply_request_cut` SELF-ACKS `CutConfirmed` (no client wait) and `apply_freeze` derives the input-cut seq from its OWN `last_input_seq` high-water at cut-INSTALL time (the leak-free partition point; the client `CUT_MARKER` / `on_cut_marker` observer is DELETED as inert). This retires the durable-multi-hop `CutTimeout`: on hop 2+ the client's session stays bound to the FIRST shard's port, so its marker never reached the current authority's saga — the old marker-driven `Cutting` starved into abort. GATE: `tests/tests/three_shard_round_trip_durable_e2e.rs` — a REAL DURABLE player (a logged-in avatar whose directory `Entity` head + fence ADVANCE per commit, NOT a transient) driven origin→50→100→50→origin, asserting `head(Entity(subject))` flips System 7 → Galaxy → System 8 → Galaxy → System 7 (both ways) with ZERO CutTimeouts / aborts and the fence advancing across four commits. Anti-vacuity PROVEN: reverting the self-ack fails the gate (the saga parks pre-`Freezing`, head stuck on System 7). Input-conservation preserved: D-28 (`p2_transfer_gates`) now checks the SERVER-derived cut = the source's actual last-applied seq (no leak, no double-apply), for ANY client behaviour (no client pause relied on).
- **S2+S4+S6 🟩 LANDED (2026-07-18) — PURE-RENDERER, NODE-AGNOSTIC CLIENT:** the client no longer learns which node owns any entity. **S2 (wire, minor 1→2):** appended `ServerControlMsg::OwnEntity{entity}` (the own-avatar cue by `EntityId` ALONE — no sub/node) + the reliable `EventMsg::EntityRemoved{entity}` de-owned-copy eviction arm; additive roundtrip tests, `PROTO_MINOR=2`. **S4 (gateway):** `announce_own_entity` emits `OwnEntity` on `negotiated_minor >= 2` ALONGSIDE `AuthorityChanged` (kept for old minor<2 clients) at session-attach + the dest re-point; a minor-2 client reads `OwnEntity` and IGNORES `AuthorityChanged`/`RequestCut`. **S6 (client collapse):** `DeliveredView.tracks` re-keyed `(SubId,EntityId)`→`EntityId` (ONE track per entity, latest-wins via `EntityTrack::observe`'s frame-collapse — a cross-realm re-home flips the frame node-agnostically); DELETED `authoritative_sub`/`set_authority`/`chosen_subs` + the client `cut_marker_armed` stamp + the `RequestCut`/`AuthorityChanged` handlers; ADDED `set_own_entity`/`remove_entity` + the `OwnEntity` handler. `held_subs` STAYS a SET (multi-realm AoI). Three consumers migrated in lockstep (real client `net.rs`, harness `ScriptedClient`, `bins/tests/process_parity.rs`); `devproto DevEntityRow.authoritative_sub` kept as an INERT constant (`vd_client::view::RENDERED_SUB`=0) for decode compat. `p2_transfer_gates` crossing gate re-proves the crossing as the rendered REALM-FRAME flip (System 7→8), the old two-holder sub-overlap anti-vacuity deleted (the client folds both subs into one EntityId track); `client-harness::crossing_was_real` repurposed to "a delivered track exists".
  - **🟩 SUPERSEDED / LANDED (status corrected 2026-08-14 — Stage-C audit :626 found this bullet still 🟥 after Step 5 slice F + D-4 minor 14 landed both halves; no "S0" arc exists or is owed):** (1) ~~the `stub.rs::emit_frames` SEND-ONCE filter to `simulates()` only~~ — SUPERSEDED by slice F's hold-gated emit: `emits()` = `simulates() | (is_retained_ghost & Source-hold-open)`, so the leaver stops emitting AT HOLD CLOSURE (the pose-free `SpawnV2` take-over proof + the remove message — it vanishes from bystanders then, never at a later send-once pass), and the fed-ghost emit died with slice F (`refresh_source_ghost`/`SourceGhostMirror`/`GhostFlow::Spawn|Delta` deleted/tombstoned). The stated blocker — "the sim has NO server-timed no-vanish hold" — is dead: `HandoffHolds` + `prune_holds`/`handing_over` ARE that hold, TTL server-derived (`derive_arrival_shield_ticks`). (2) the `EntityRemoved` transfer-subject emit LANDED shard-side: `push_entity_removed` fans at hold closure + TTL expiry (the band-exit re-emit is idempotent on clients that already evicted), `fan_entity_removed` fans it to clients — see D-4 (consumer (a) LANDED, minor 14). (3) NO dedicated reliable `MsgClass::Event` was added and none is owed — the landed decision went the other way: `EventMsg` rides `ServerControlMsg::Event` over the EXISTING reliable Control class (D-4(a)); `grep MsgClass::Event crates/` is empty by design.
- **#133 STATION/AREA FIRST-CLASS 🟩 LANDED (2026-07-17):** `worldgen::realm_regions_for` now plants `RealmId::Station(7)` (an `Aabb` box at (-25,0,0) half 5 under System 7, depth 3) and `RealmId::Area(7)` (an `Aabb` box at (25,0,0) half 3 inside Planet 7, depth 4 — the DEEPEST region) as FIRST-CLASS realms in the seed forest, so the SAME kind-agnostic containment detector re-homes into/out of them with ZERO station/area-specific code (HR3 upheld — no realm-kind fork in any live path). Client render is FREE (`from_regions` → `BoxShape::Box`, scene now 5 finite realms). GATE: `stub::tests::containment::{a_dot_moving_into_the_station_box_re_homes_into_the_first_class_station_realm, a_station_owning_shard_re_homes_a_dot_that_leaves_the_station_back_to_system_7, a_dot_moving_into_the_area_box_re_homes_into_the_first_class_area_realm}` prove Station IN + Station OUT (from the Station's OWN authority — the reverse leg) + Area IN, each a REAL decoded `CrossingRequest` asserting `to_realm`/`from_realm`/`subject` (non-vacuous: the `==1` count and the `to_realm` both require the box to win the container fold). **DECISION (user, 2026-07-17): Area is FIRST-CLASS / authority-bearing, SUPERSEDING the original #133 "Tier-1 EffectFree interest-only, no directory record" locked decision** — the containment model has no interest-only tier; the AoI/interest READER stays deferred to P6 (below). This closes the pre-P4 half of task #133; the moving-container Station semantics (a station is a MOVING container) ride P4/P5 (a/b below), dynamic reparent P8 (e), transient batch-coalescing D-43 #9-SCALE. Adversarially reviewed SOUND (wf a6d8d2dd: gate non-vacuous, geometry Station⊆System7 / Area⊆Planet7 by-hand, collision-safe, HR3 clean).
- **INTEREST / AoI path DEFERRED (write-only, no reader):** the old `InterestZones` was WRITE-ONLY with no reader; multi-mesh client rendering uses the ghost-collider feed (`feed_source_ghosts`/`GhostFlow`), architecturally DISJOINT from containment — removing the interest stub breaks nothing. Re-add the interest/AoI subscription path (which neighbour meshes a client subscribes to) as a dedicated slice WITH its reader, layered on the containment machinery (its per-region hysteretic membership was entangled with the deleted `CrossingState` dwell — re-add cleanly). **KEPT-UNUSED pending this slice:** `RealmBoundary` + its `shell`/`aabb`/`boxed` constructors + `CrossEffect` + `Direction` stay in `geometry.rs` (covered only by their own shape/band tests + the bins adapter's input type) — scaffolding for the Interest path + the client `--realm-boxes`, NOT live containment machinery; delete them if the AoI slice does not need them.
- **SWEPT tunnel-guard DEFERRED:** C-3 relies on the velocity-safe `ContainmentBand` (`inset+outset ≥ v_rel·dt·K_SAFETY`) so a region cannot be tunneled at expected speeds; the additive swept-segment DEBUG tripwire is owed as a diagnostic, not correctness-load-bearing.
- **P4/P5/P8/P10:** (a) moving regions `f(seed,tick)` + seed→celestial-parameter generator + SPIKE determinism gate — P4/P5; (b) non-identity input-side `transfer_frame` — **FA-1 (D-45(a), visual-universe arc) SWAPPED the detector's input seam `IdentityFrames`→`LocalFrames`**: `evaluate_realm_boundaries` now re-expresses each pose through the shard's OWN per-shard ephemeris `FrameContext` (`RealmRegions::frame_context` → `vd_core::frame::LocalFrames`), byte-IDENTICAL at static walk scale (every region registered at the identity placement; `node_per_realm_walk` + `process_parity` unchanged) with an `Err`→`f64::MAX` SAFE-DEGRADE so a frame the shard cannot name folds to non-member, never a spurious container. **Still owed:** the NON-IDENTITY MOVING direct-child placements `f(seed,tick)` (which make a cross-realm rebind actually MOVE the pose) ride FA-2/FA-4 on this same frozen seam — P4/P5; (c) D-41 non-zero-cell arithmetic in the containment subtract — P4/P5; (d) geometric child⊆parent subset check in `guard_regions_nest` — LANDED EARLY (Stage-C batch 1, 2026-08-14): the boot hands `guard_regions_nest` every child's WORST-INSTANT reach (`ChildReach`, movers judged at apoapsis — D-PLACE-1) and refuses a child that can leave its parent, at every boot; (e) moving-container reparent straddler guard + dynamic-registry O(N·M²) re-arm — P8; (f) the coarse-tier warp-map (galaxy/universe ly-cells) + warp-as-position-driver + exit-warp-void — P10 (§1.4 coordinate regimes).
- **REALM-UNIFICATION (D-45(a), vetted `wf_f4caf34a-ada` 2026-07-21, HOLDS_WITH_CAVEATS + adversary-survives):** everything spatial is a REALM (an authority domain: a shard owns it, it authors its DIRECT children's poses from `{input signals, last-wins}` + `{its own ambient physics}` and SHIPS them; **it is NEVER told where it is** — see the 2026-08-05 reversal below) vs an OCCUPANT (a shipped pose a realm simulates). System/Planet/Ship/Station/Area differ ONLY in parameters, never machinery (HR3). A planet is a MOVING realm = the zero-signal degenerate case; a ship is the signal-driven case. CONSEQUENCE: **the `CelestialBody` EntityKind idea is DELETED** (frozen tag space untouched) and **NO `ContinuityModel::Orbital`** (the parent ships every tick; nothing re-advances a stale realm pose; no client prediction). A realm reparent rides the EXISTING fence-CAS on its OWN `DirectoryKey::Ship/Realm` (already transfer subjects, `transfer_subject_entity`→None); `fan_out_crossing` splits on `durability_of`, never a realm-kind match. The old FA-2/FA-4b/FA-4c COLLAPSE into ONE verb — the parent authors its moving children's placements each tick and ships them — to ONE recipient on ONE lane: OBSERVERS (render, FireAndForget latest-wins). ~~+ the CHILD SHARD (receive-own-position, an EffectFree Coupling — a SEPARATE authority lane…)~~ **STRUCK 2026-08-05 — see the reversal immediately below.**

- **★ REVERSAL (owner, 2026-08-05): A REALM IS NEVER TOLD WHERE IT IS. The receive-own-position lane is DELETED, not deferred.** The owner's reasoning, which supersedes the FA-4b clause above and every document that repeated it: *a hull's pose inside its own realm never changes — the ship frame IS the ship. What changes is where the ship REALM sits inside its parent. So nothing needs to go back.* A child realm is therefore **completely ignorant of the universe outside itself**, which is the cleanest possible reading of HR1: it holds no coordinate it did not author.
  - **WHERE THE WORK MOVES INSTEAD — the parent expresses a child's SURROUNDINGS relative to that child.** The question this reversal has to answer is: a player stands inside a ship and looks out of the window at a planet — who computes where that planet appears? Not the ship shard (it must not know where it is) and not the client (it composes nothing — the hard law). **The PARENT does**, because it already holds both positions, already runs collisions for its children, and already computes their area-of-interest. It ships each child's surroundings already expressed in that child's frame.
  - **THE CLIENT'S RENDER ORIGIN IS THE INNERMOST REALM IT OCCUPIES.** Standing inside a ship, a player's ship-frame position IS the drawable position — nothing is composed anywhere. Moving through a realm boundary re-pins the origin, which is the SAME re-anchor machinery the warp re-anchor slice builds; there is no second mechanism.
  - **WHAT THIS DELETES:** the entire "received seat" slice (arc slice 11) — a realm holding a position it was told is machinery for a requirement that no longer exists. The `Placement::Dynamic` / `OriginLink::Dynamic` widening, the self-pose resource, the staleness bound on a told-position, and the dedicated told-position wire arm are all struck. **Do not resurrect them without re-opening this reversal.**
  - **WHAT SURVIVES AND IS STILL OWED:** (i) the parent authoring a child's surroundings in the child's frame (the real FA-4b successor — design owed, no producer today); (ii) the render-origin re-pin on a realm crossing (the warp re-anchor slice); (iii) **an OPEN question this reversal does NOT settle** — the ship-thrust port's `HullPose` feedback exists for the interior "down" reference (which way is down for someone walking inside a ship under 3g, or landed on a planet). That is an ACCELERATION/ORIENTATION quantity, not a position, and it is a legitimate ambient-physics input for the child's own interior. It must be decided on its own merits, and if it survives it must be named as felt-acceleration, never as a pose (see `sealed_shards.md` and `PLAN.md`).
  - **FA-2a 🟩 LANDED (RealmSnap carrier PLANT, this commit):** `vd_wire::channels::{RealmSnap{realm: RealmId, pose}, RealmSnapshotDatagram{sub, frame_id, source_tick, universe_tick, realms}}` — the `RealmId`-keyed observer carrier (a realm is NOT an entity: `RealmScene` is `RealmId`-keyed, `FrameRef::realm()` is LOSSY, so a moving realm box is unrecoverable from an `EntitySnap`). Behaviour-identical PLANT (the OwnEntity/D-41 discipline): a shard emits it ONLY when it parents ≥1 moving/renderable child, so at walk/static scale (no moving child) NOTHING is sent — zero bytes, `node_per_realm_walk`/`process_parity` unmoved; 100% region+branch. FA-2c wires the emit (+ the shared MTU `partition_realms`).
  - **FA-2b 🟩 LANDED (moving-child data source, this commit):** the vet's "add `with_authored_child`" was a MISREAD — `with_placed` already takes an arbitrary `FramePlacement`, so a per-tick signal-computed ship pose rides it as-is (no redundant method). The REAL gap was the DATA: `to_regions` bakes an `Orbital` body's tick-0 anchor into `center` and DISCARDS its `OrbitalElements`. Added `worldgen::moving_children_for(seed, hosted_realm)` (the shard's DIRECT `Orbital` children as `(realm, elements)`; branchless-shim `orbital_of`; EMPTY at walk scale) + `RealmRegions.moving` (a builder `with_moving_children`, `new` call sites unchanged) + `frame_context` now registers a roster realm `with_moving_child` (authored live from its ephemeris) else `with_placed(identity)`. EMPTY at walk scale ⇒ every region takes the identity arm ⇒ byte-identical (`node_per_realm_walk` 142.9s + `process_parity` + workspace unmoved; 100% region+branch). The two placement sources the arc needs — orbital (`with_moving_child`) + authored/supplied (`with_placed`) — BOTH exist.
  - **FA-5 PRE-MOVER HARDENING 🟩 LANDED (holistic /goal audit `wf_c9444997`, HEAD `8437c89`→this commit — fixes the audit's ONE HIGH + its MEDIUM before movers turn on):** the realm-observer gate now matches the entity feed's per-source discipline. **(1) Per-`RealmId` high-water (the HIGH):** `RealmView.high_water` was ONE feed-global `Option<u64>`; because each shard runs an INDEPENDENT `RealmFrameCounter` from 0 and authors only its DISJOINT direct children, the moment two mover shards share one client's AoI (the node-per-realm Forest, e.g. a durable player's System 7→Galaxy→System 8 warp) the higher-counter shard ratcheted the scalar past the lower and FROZE its boxes forever (`is_stale(Some(500),30)` for every one of B's frames). Re-keyed to `BTreeMap<RealmId, u64>` gated PER-ROW (a realm is single-owner ⇒ its `frame_id` stream is monotone ⇒ per-realm keying decouples the independent counters; the render was ALREADY per-`RealmId` latest-wins, so the key matches the render). Regression `realm_view::tests::two_mover_shards_with_divergent_frame_ids_both_stay_live_no_cross_shard_freeze`. **(2) DevState realm fault counters (the MEDIUM):** `net.rs::devstate()` surfaced only the ENTITY view's `stale_frames_dropped`/`nonfinite_poses`; the realm feed's own (computed + exposed) faults were invisible on the HR6 surface — now SUMMED across both feeds (`net::tests::devstate_surfaces_the_realm_feeds_own_fault_counters...`). BYTE-IDENTICAL at walk scale (empty roster ⇒ no realm frame ⇒ both counters 0, per-realm map empty; `process_parity` unmoved, 100% region+branch). **STILL OWED (ledgered, NOT fixed — each needs the `realm_fence`/FA-6, INERT until a realm reparents):** (i) **FA-6 realm re-home split-brain** — when a realm RE-HOMES across shards (SAME `RealmId`, old owner's counter high, new owner fresh) per-`RealmId` keying rejects the NEW authority's fresh-but-lower `frame_id`, AND the gateway `on_shard_realm_frame` still drops `realm_fence` via `..` — so the client would render the DEMOTED owner's stale placement. FIX AT FA-6 (with the region-vs-region reparent detector): the gateway forwards `realm_fence`; the client gates on `(realm_fence, frame_id)` — a higher fence supersedes and RESETS that realm's high-water. NOT planted now: an inert fence-gate branch would be HR5-uncoverable (production-unreachable until P8 ships / P10 warp physically move a container — the 4b-STATUS(c) HIGH-2 rule). (ii) **`EventMsg::RealmRemoved` eviction** — `RealmView.placements`+`high_water` grow monotonically (no removal path); correct through P3 (boot-fixed renderable roster), owed when a realm leaves a live observer's AoI. (iii) **Realm AoI (the D-9 twin)** — the gateway fan-out of `RealmFrame`s is O(observers×movers) unfiltered; ledger WITH the entity AoI (D-9) before ships become movers at scale. (iv) **HR6 live-QUIC loop at FA-5** — the 2-capture render smoke + a process-tier/vdctl `wait-until RealmFramesApplied >= 1` over the REAL binaries (the server-emit→gateway-forward→client-fold path is unit-proven only, never composed over live QUIC — the D-18/D-36 class on the Unreliable `RealmSnapshot` datagram).
  - **FA-2c server→gateway pipeline 🟩 LANDED (byte-identical):** **FA-2c-1** appended `MsgClass::RealmSnapshot` (WIRE-FROZEN, byte 7, Unreliable; the client routes a realm datagram to its own consumer — postcard is non-self-describing) + updated the 3 golden pins. **FA-2c-2a** added `ShardToGateway::RealmFrame` + a DRY `partition_realms` (both partitioners refactored to branchless shims over a monomorphic `chunk_boundaries`). **FA-2c-2b** the sim producer: `RealmRegions::authored_realm_snaps` (moving children as `RealmSnap` rows, orbital pose in the OWN frame) + the `emit_realm_frames` system (ships a `RealmSnapshotDatagram` to observers, gated on authority + a moving child + a present observer). **FA-2c-2c** the gateway `on_shard_realm_frame` forward (FireAndForget, no authority gating per the vetted observer design). **FA-2c-3 (client consumer — PIPELINE COMPLETE, vetted `wf_894145a5-ac2`):** 3.0+3.1 a shared `wire::channels::is_stale` scalar (DRY between the entity + realm gates) + `client::realm_view::RealmView` (the `RealmId`-keyed twin of `DeliveredView`, reusing `EntityTrack` VERBATIM); 3.2+3.3 the net.rs `MsgClass::RealmSnapshot => on_realm_snapshot` route (converts the former silent `ignored`) + `RealmScene::overlaid(&RealmView)` streamable-scene overlay at `render_snapshot()` publish (empty ⇒ boot `Arc` pointer-bump; non-empty ⇒ fresh immutable overlaid `Arc`). ZERO client-render change (`sync_realm_boxes` re-reads the scene each frame). Vet must-fixes folded: the overlay writes BOTH `box.frame` (the authored PARENT frame) AND `center_offset`; `on_realm_snapshot` ALSO anchors `render_clock` (no spectator freeze). ALL byte-identical: at walk/static scale the moving roster is empty ⇒ nothing authored/emitted/forwarded/consumed (`process_parity` + all vd-tests + the render-boxes GPU smoke unmoved; 100% region+branch each). **Still owed:** ~~FA-4b child-shard receive-own-position (EffectFree Coupling)~~ **DELETED 2026-08-05 by the owner's reversal above — a realm is never told where it is; the parent expresses a child's surroundings in the CHILD's frame instead, and that successor design is owed;** ★★ **RE-OPENED 2026-08-24 BY THE OWNER'S SECOND REVERSAL — SEE [[D-SL1-2]] BELOW.** The 2026-08-05 deletion no longer stands: SL1 was rewritten so that a parent MAY state to a child the placement it authored for it, read-only and one hop. Do NOT read this struck clause as current; it is kept because the ledger keeps its history, not because it is true. FA-5 turn on moving placements + make `view.rs::world_pos` RECURSE for N-level nesting + the moving-container straddler guard (=D-45(e)); FA-6 the REGION-vs-REGION reparent DETECTOR (genuinely NEW — the occupant detector emits `DirectoryKey::Entity` subjects only) + the populated-ship atomic N+1-key directory CAS (=D-33). LEDGER-NOW owed on the containment side: (i) a voxel-built ship/station realm keeps an ANALYTIC bounding `Boundary` (Aabb/Obb) the blocks INSCRIBE for containment membership (blocks drive only render+rapier), keeping `region_signed_distance` cheap + HR3-clean; (ii) a RUNTIME register-a-realm-region path (`RealmRegions` is boot-only `::new(Vec)`, no mutator) for emergent hulls (the dynamic-registry half of (e)); (iii) D-35 `parent_version` on the snapshot to version-match a hull against its `ShipLocal` interior under datagram loss. **FA-2c-3 deferrals (→ FA-5):** the real-proportions CANONICAL generation that PRODUCES `Orbital` movers + populates `RealmRegions.moving` at boot (the visible orbiting pixels; needs D-41 non-zero cells); the 2-capture pixel-displacement render smoke (needs a mover); DevState realm counters + a `WaitField` arm for the vdctl HR6 closed-loop e2e (S0 🟩 LANDED `8437c89`: `realm_frames_applied` + the `RealmFramesApplied` `WaitField`; realm decode errors DRY-fold into `decode_errors`); `EventMsg::RealmRemoved` (a realm leaving a live observer's AoI — persist-last is correct through P3 since the renderable roster is boot-fixed); render-side cursor interpolation for a moving box (`current_render_pose` gives the latest per-step; smoothness for a FAST mover is P6+).
- **RAYON (task #133 decision 4, later):** the per-entity loop is `par_iter`-ready (the `RegionMembership` bitset is contention-free); only the shared `outbox`/`stats`/`in_flight` sinks block it (map-reduce into thread-locals when it lands).

### D-46 🟥 Gateway input-cut partition atomicity — REQUIRED before the "(future) threaded 20 Hz forwarder" (S3 server-timed cut, verify `wf a9946a1c`)
- **Now (SOUND):** S3's server-timed cut (`gateway.rs::apply_freeze`) derives the input-cut `marker_seq = last_input_seq.load()` at install then `store_cut`. Input-conservation holds BY SERIALIZATION: `route_input` (from `on_client_input`) and `apply_freeze` (from `on_transfer_control`) both run inside the ONE `process_gateway_inbound` Bevy system on the single sim thread, one inbound msg at a time — they never overlap, so the `load(marker)`→`store_cut` pair is atomic w.r.t. the router. No double-apply is reachable today; the durable multi-hop `three_shard_round_trip_durable_e2e` + the D-28 `p2_transfer_gates` "cut == source last-applied" gate pin it.
- **The tripwire:** the moment the router/forwarder is made THREADED (concurrent `route_input` off the sim thread — flagged "(future, threaded)" at `gateway.rs` ~178/243), the `load(marker)`→`store_cut` becomes a live TOCTOU: a concurrent `route_input` can `fetch_max(X+1)` and Forward X+1 to the SOURCE in the window before the cut installs (marker=X), and the DEST re-applies X+1 (it seeds `last_applied` to `marker_seq`, NOT the source's true last-applied — `stub.rs::OpenInputSlot` — so it does NOT dedup this cross-shard double). A cross-shard input DOUBLE-APPLY = input-conservation violation.
- **The fix (WHEN threading):** make marker-capture + cut-install atomic w.r.t. the forwarder — install the cut FIRST with a sentinel `marker_seq` (so any racing input reads the installed cut and Buffers), then settle the marker to the observed high-water; OR seed the dest resume watermark from the SOURCE's acked last-applied (a real drain oracle), not the gateway high-water. Do NOT thread the forwarder before this lands.
- **Source:** S3 adversarial review `wf a9946a1c` (2026-07-17) — the one flagged item; correctness-sound today, latent-for-threaded.

### D-43 🟧 Geometric transfer-TRIGGER owed items — the crossing PRODUCER + consumer + abort DURABILITY + unresolved-dest re-drive landed (50dc45e/42d7b73/978b59d/3f-D4); SCALE hardening (#4-#8, #9-SCALE) + the process-tier crossing-SIGKILL owed before task #133 plants the first real `RealmBoundaries` band (holistic audit `wf_3916069b`, 2026-07-13)
- **C-3 CONTAINMENT SUPERSEDES THE PORTAL TRIGGER (task #135, 2026-07-16):** the user rejected authored "trigger zones" — a crossing must fall out of "which realm's REGION contains me", symmetric, no authored destination. The DIRECTIONAL portal machinery is DELETED: `RealmBoundaries`, `should_commit`, `resolve_winner_ix`/`candidate_beats_ix`/`resolve_winner`, `authority_dest`, `crossing_candidates`, `candidate_direction`, `boundary_depth`, the `winner_ix`/`was_member`/`inward_ticks` dwell, `InterestZones`, `crossing_outward_no_parent`/`crossing_interest_updates`. REPLACED by CONTAINMENT (`vd_core::geometry::{RealmRegion, ContainmentBand, container, depth_beats, region_depth, region_signed_distance, should_rehome}` + `vd_core::worldgen::realm_regions_for` + `vd_sim::stub::{RealmRegions, ContainmentProgress, RegionMembership}`): each tick full-scan the shard's regions → deepest containing realm → `should_rehome(own, container)` → ONE derived-`to_realm` re-home. Net −764 lines, 100% region+branch. The fence-CAS transfer/abort/redrive machinery (3f-D) is UNCHANGED (only the destination-DECISION changed). **Resolves-by-containment:** #4 (per-(entity,boundary) cooldown) — MOOT (a container change is one symmetric membership event, not a boundary-crossing sequence); #5 (`winner_ix` stable id) — MOOT (the per-region `RegionMembership` bitset has no single-winner slot); #6 (`boundary_depth` O(N·M²)) — RETIRED for the static registry (depth cached at boot in `RealmRegions`, re-armed only for a P8 dynamic registry). **The 4a/4b directional slices below are SUPERSEDED** (kept for the audit trail). **Still owed by containment → see D-45.**
- **Status:** the trigger is INERT in P3 (`RealmBoundaries` default-empty; `evaluate_realm_boundaries` authority-gated + empty-candidate early-return), so EVERY item below is latent — reachable only when task #133 / P4 plants a real band. NONE is a running-tree defect (the 2026-07-13 audit confirmed 0 CRITICAL, base AAA-sound + not cornered). Sequence these WITH 3f-D/3f-E, never after the first band.
- **The owed items (from the audit, each with its fix):**
  1. **Abort-reply durability (3f-D, the big one; audit M1/H1/L6):** a durable pre-CAS crossing abort must POSITIVELY clear the source `RequestInFlight` latch — only the source *consumer* (`on_crossing_aborted`, `stub.rs`) + the wire `CrossingAborted`/`CrossingAbortedAck` arms landed; the orchestrator EGRESS is unbuilt. Land Mechanism-Y: `pending_abort_replies` map + `scan_deadlines` re-emit + **dead-source self-reap** + the `CrossingAbortedAck` consumer + source unconditional ack, crash-durable via the saga store/rehydrate. **Discriminator = STATELESS TAG-CHECK** `(ctx.transfer.0 >> 120) == 0x39` (verified exclusive: rehome=0x37, regular=small/high-byte-0) — NOT the plan's earlier Option-3A recompute (H1: recompute against `ctx.expected_fence`=HEAD would false-negative the wire-vs-head fence split; the tag-check is fence-independent, dodges the persisted-`SagaSnapshot` change entirely). **H2 same-fence aliasing = GENERATION-STAMP the id** (user-ratified): `crossing_transfer_id(subject, fence, attempt)` — the attempt (a per-entity `CrossingState.crossing_attempt`, sent in a new `CrossingRequest.attempt` field) folds into the id so each attempt is unique; NO separate `generation` wire field.
  1-STATUS 🟩 **ABORT-REPLY DURABILITY LANDED (3f-D, commit 978b59d):** Mechanism-Y `pending_abort_replies` + persist/rehydrate + `scan_deadlines` re-emit + **ownership-reap** (`dir.head(subject)!=source` — the adversary-review fix, NOT the `is_confirmed_dead` pulse) + `CrossingAbortedAck` consumer + source unconditional ack + the `0x39` tag-check + the attempt-stamp. All 3 files 100%. The crash-durable round-trip is DONE; the process-tier SIGKILL crossing-crash e2e is the only owed piece (below).
  1-CLUSTER-STATUS 🟩 **ABORT LEG + TIER-1 RESTART CRASH LEG COMPOSE AT CLUSTER TIER (plan `scripts/abort_leg_crashleg_plan.md`, `tests/tests/crossing_e2e.rs`):** the crash-durable crossing-abort (Mechanism-Y) is now proven to COMPOSE end-to-end over the real cluster fabric. (a) **Abort leg** (`crossing_e2e_pre_cas_abort_clears_the_source_latch`): a crossing-origin durable saga forced to REJECT at Prepare (a new INERT one-shot `GatewayConfig.reject_next_prepare` lever — the gateway is the durable Prepare decider, so `apply_prepare` emits `Prepared{ Rejected(Spatial(Obstructed)) }` → `abort_from_pre_freeze`, no `ThawSource`/`IssueCommitCas`) stages a persisted latch-clear reply (`pending_abort_replies >= 1` observed mid-flight — the Mechanism-Y non-vacuity discriminator, since a non-crossing abort stages NOTHING), the source acks + clears its latch, and the directory head **NEVER moves off the SOURCE** (the pre-CAS proof: head-stayed-`Shard(SOURCE)` necessary half + DEST-never-adopted sufficient half). (b) **Tier-1 restart crash leg** (`crossing_e2e_abort_survives_orchestrator_restart`): the ack is PARKED (fabric `set_policy` on the SHARD→ORCH Saga link, block+redeliver) between the abort-emit and the ack, the orchestrator is REBUILT from the RETAINED WAL, and the persisted reply is RESTORED (`pending_abort_replies >= 1` captured pre-heal, BEFORE any healed step), then the parked ack redelivers and reaps it — head never moved. (c) **Negative control** (`crossing_e2e_abort_reply_absent_from_empty_store_rebuild`): the SAME fixture rebuilt against a fresh `MemStore::new()` recovers `== 0` — proving the restore reads the WAL, not an in-process survivor. INERT knob = behaviour-identical (every real cluster passes `None`); HR5 two-arm coverage OWNED by 3 gateway unit tests (`apply_prepare_{inert,armed,not_active}_*`); the `scan_deadlines`/`rehydrate` arms cited from their existing owners. **HONEST TIER: it is a World-REBUILD (`rebuild_orchestrator`), NOT a process `kill -9`** — the process-tier crossing-SIGKILL stays owed below.
  1-SIGKILL-OWED 🟧 **Process-tier geometric-crossing SIGKILL** — a real `kill -9` of the orchestrator through the `orchestrator` bin MID geometric-crossing-abort. Blocked on a boundary-plant knob in the `shard` bin (it boots empty `RealmBoundaries`, no in-band way to plant a crossing shell across the process seam). The abort-reply crash durability is already process-proven *generically* by `orchestrator_crash.rs`, and *WAL-restore-specifically* by the `rehydrate_restores_a_persisted_abort_reply_and_reemits_on_first_scan` unit test — so the crossing-SIGKILL is a **composition gap, not a durability gap**. The Tier-1 test is deliberately named `..._survives_orchestrator_restart` (never `..._kill9`) to hold the distinction.
  2-STATUS 🟩 **UNRESOLVED-DEST RE-DRIVE LANDED as a LATCH-SCAN (3f-D4, `stub.rs` `redrive_stranded_crossings`):** the durable `CrossingRequest` is emitted ONCE (`Entry::Vacant`); a DELIVERED-but-unresolved dest (`head(Realm(to))` transiently absent at P4/P5 — a realm shard mid-lease/partitioned) STRANDS the entity. **The `Entry::Occupied` re-drive the prior text prescribed was UNREACHABLE** — `should_commit` (`geometry.rs:642`) is a STRICT one-tick rising edge, so `fan_out_crossing` never re-runs for a statically-DWELLING stranded dot; the re-drive therefore MOVED to a separate per-tick latch-scan. FIX (landed): a NEW system `redrive_stranded_crossings` over `RequestInFlight`, scheduled `.after(evaluate_realm_boundaries)` in the same chained group, re-emitting the SAME latched request (same `(subject, subject_fence, attempt)` → byte-identical `crossing_transfer_id`; the orchestrator's `contains_key` guard absorbs a dup that started, an unresolved one re-tries the head reads) once `local_tick - last_commit_tick >= request_ttl_ticks`, then re-arming the ttl timer. **INERT by default** (`request_ttl_ticks == 0` → an early return before any iteration — every current rig). Authority-gated exactly like `evaluate_realm_boundaries`; deterministic (BTreeMap iteration, `.min(u32::MAX)` cast-guard, no rng/wall-clock/default-hasher). Design decision Q-A (user-resolved, do NOT change `RequestInFlight`'s value type): the latch's re-emit fields `{to_realm, subject_fence, session}` ride a new `Option<LatchedCrossing>` on `CrossingState` (all `Copy` → `CrossingState` stays `Copy`+`Default`), set alongside the latch in the durable `Vacant` arm — the scan cannot recover them from `RequestInFlight`'s `EntityId→TransferId` alone. New `StubStats.crossings_redriven` counter. `request_ttl_ticks` was already WIRED (audit L8). The `Occupied` arm stays a PURE suppress (untouched). Tests: `a_stranded_durable_latch_redrives_after_the_ttl` (headline — a static dweller's latch strands, the scan re-emits the SAME id past the ttl, re-arms the timer, no storm), `the_ttl_redrive_is_inert_when_request_ttl_ticks_is_zero` (the `ttl==0` early-return); the `.expect()` invariants are straight-line (100% HR5). **CORRECTION (D-WORLD-2 cure, 2026-08-15): the "INERT by default / every current rig" posture is RETIRED** — `request_ttl_ticks` is now ARMED in every launcher and fixture via the `derive_request_ttl_ticks` derivation, the re-drive is BOUNDED by the derived `crossing_redrive_budget`, and exhaustion takes the source-local pre-CAS abort that clears the latch (see D-WORLD-2 for the full cure); `ttl == 0` remains only as the disarmed unit-rig posture.
  3-STATUS 🟩 **CrossingProgress RE-ARM LANDED (3f-D, `stub.rs` on_crossing_aborted):** the id-match abort-clear resets the dwell (inward_ticks/was_member/winner_ix/last_commit_tick) + bumps the attempt, so a still-in-band entity re-requests without physically re-crossing (the latch-scan #2 makes the ttl path reachable too). Regression landed.
  4. **Cooldown per-(entity,boundary) (audit L5):** the anti-thrash cooldown is per-ENTITY while the dwell is per-BOUNDARY — a legit back-to-back station→bay Authority crossing within `k_dwell` is dropped (the landed regression uses `Interest`, which arms no cooldown, so this is untested). Make the cooldown per-(entity,boundary) to match the dwell; add an Authority A-then-B regression. **DEFERRED to Slice 4b** (dock/undock nesting) with #5 — it is entangled with the direction-keyed latch redesign.
  5. **`winner_ix` stable identity (audit L9):** `CrossingState.winner_ix` is a RAW slice index into `RealmBoundaries`; the dwell reset compares indices. Stable only while the registry is immutable — a P4/P5 dynamic reorder aliases dwell to a different boundary. Key the reset on a STABLE per-boundary id (NOT `RealmId` alone — coincident dock mouths share a realm). **DEFERRED to Slice 4b** (dock/undock nesting).
  4a-STATUS 🟩 **SLICE 4a LANDED (task #133, the CLEAN subset — dock/undock NESTING split out to 4b):** the clean, INWARD-only crossing surface hardened + proven. (a) `RealmBoundary::aabb(...)` + `OverlapBand::for_box_velocity_safe(...)` (`crates/core/src/geometry.rs`, 100% Tier-A) — the NORMALIZED-Chebyshev velocity-safe box boundary that mirrors `shell(...)`/`for_soi_velocity_safe(...)` arm-for-arm (unit = 1.0 = the box surface); + `LatticePos::at(cell, offset)` (`crates/core/src/pose.rs`) — the single non-zero-cell construction point the P4/P5 rebase needs. (b) **D-38 discharged** for the crossing feature — the `assert_feature_anywhere` test (`crates/sim/src/stub.rs`) runs ONE fixture on a Spherical Shell + a Cartesian Aabb (see [[D-38]] 🟩). (c) **Ledger-eviction-on-leave** proven by `slice4a_all_three_ledgers_evict_when_the_subject_leaves` (all three of CrossingProgress/RequestInFlight/InterestZones → `None` on subject removal, the InputLog-leak class; the `retain_live` machinery was already landed). (d) **Cell-carry SHAPE** proven by `slice4a_nonzero_cell_is_carried_unchanged_and_the_offset_crossing_still_fires` — a DISTINCT non-zero `LatticePos.cell` is carried UNCHANGED through the trigger and the offset-based crossing still fires (SHAPE only; the "integer in/out DECISION flips across a cell boundary" test is BLOCKED on the P4/P5 rebase math — `should_commit`/`evaluate_one_subject` read ONLY `offset()`, never comparing the cell — and is deliberately NOT claimed). NO parent-mutation, NO `RequestInFlight`-keying change, NO `crossing_transfer_id`/`on_saga_demote` change, NO frozen-wire change (all Slice 4b). The `Occupied` suppress arm is untouched.
  4b-DEFERRED ✅ **SUPERSEDED by `4b-STATUS 🟩` below** (kept for the audit trail — do NOT read the "strands" claim as live). This entry recorded the PRE-adversary understanding: that the OUTWARD/undock crossing's direction-BLIND single latch (`RequestInFlight: BTreeMap<EntityId, TransferId>` keyed by entity only) would let a dock→undock interleave suppress the undock and STRAND the entity (adversary **CRITICAL-1**, `scripts/slice4_design_adversary.md`). **That "strands" framing was DISPROVEN** (`scripts/slice4b_design_adversary.md` FINDING-1 + the `crossing_nesting_e2e` self-heal proof): the shard serializes crossings PER-ENTITY (the `Occupied` suppress), so there is never a concurrent undock to strand, and the interleave SELF-HEALS — the dest shard re-evaluates the adopted entity and fires the undock. So the direction-keyed latch (`(Direction,TransferId)` re-key, `crossing_transfer_id` gaining direction, `LatchedCrossing`/`on_saga_demote` direction-matching) was NOT built — it is unnecessary AND would have INTRODUCED a `self_fence` direction-mismatch (adversary FINDING-1's real hazard). The fence-keyed re-parent moved to P8 (`4b-STATUS (c)`). ONLY the per-(entity,boundary) cooldown (#4) + stable `winner_ix` (#5) remain 🟧 (orthogonal to the direction axis — see `4b-STATUS` "Still-deferred").
  4b-STATUS 🟩 **SLICE 4b LANDED (RE-SCOPED — the SIMPLER design a design workflow + adversary + a verification spike proved):** the `outward_dest` re-home + the self-heal composition proof landed; the direction-keyed latch was DROPPED as unnecessary and the dynamic-parent reparent was DEFERRED to P8.
    - **(a) `outward_dest` — an undock re-homes to the PARENT.** `fan_out_crossing` (`crates/sim/src/stub.rs`) now resolves its AUTHORITY destination through a new monomorphic `authority_dest(dir, winner) -> Option<RealmId>` = `match dir { Inward => Some(winner.to_realm), Outward => winner.parent }` — a dock docks INWARD to the boundary interior, an undock re-homes OUTWARD to the enclosing `parent`. Used ONLY in the two Authority arms (durable `CrossingRequest` + transient `TransientCrossingRequest`) via `let Some(to_realm) = authority_dest(dir, winner) else { stats.crossing_outward_no_parent += 1; return; };`; the Interest arm stays direction-agnostic (reads `winner.to_realm` directly — adversary LOW-1: the degrade is INSIDE the Authority arms so it never short-circuits Interest). A `None` outward parent on an Authority-Outward boundary is a COMPOSER/authoring error (every real realm has an enclosing parent) — COUNTED in the new `StubStats.crossing_outward_no_parent` (adversary MEDIUM-1: loud tripwire a valid fixture asserts stays 0, NOT a per-tick silent drop), never panicked. HR5: `authority_dest` is a branchless 2-arm total map (both arms covered + CONTRASTED by the dock/undock tests); the `let-else` degrade covered by the top-level-outward (parent=None) test. Determinism SACRED: `Direction` is Copy, `winner.parent` a pure read, no rng/hasher/clock. Tests (all in `stub.rs` mod tests, 100% Tier-A): `slice4b_an_undock_re_homes_outward_to_the_parent_realm` (asserts `to_realm == PARENT`), `slice4b_a_dock_and_undock_resolve_contrasting_destinations` (both arms, destinations asserted to DIFFER), `slice4b_a_top_level_outward_undock_with_no_parent_degrades_loud` (parent=None → zero request + counter==1 + no orphan latch), `slice4b_two_same_depth_outward_boundaries_peel_deterministically` (adversary FINDING-3 — exactly ONE winner/tick, permutation-invariant; DOC: the Outward tiebreak sorts on `to_realm` (interior) by construction while the destination is `winner.parent`).
    - **(b) Self-heal composition PROOF (the direction-keyed latch DROPPED).** The shard serializes crossings PER-ENTITY (`RequestInFlight` is `BTreeMap<EntityId, TransferId>` + `fan_out_crossing`'s `Entry::Occupied` arm suppresses a 2nd crossing per entity) → there is NEVER two concurrent sagas per entity → the dock/undock interleave SELF-HEALS: after a crossing commits, the DEST shard's `evaluate_realm_boundaries` re-evaluates the ADOPTED entity and fires the next crossing on its own. So the adversary's CRITICAL-1 (a direction-blind latch strands the undock) DISSOLVES — no `(Direction, TransferId)` re-key, no `crossing_transfer_id` direction field, no `LatchedCrossing`/`on_saga_demote` direction-matching. Proven by `tests/tests/crossing_nesting_e2e.rs::crossing_e2e_dest_reevaluates_adopted_entity_self_heal` (DEST.crossings_requested 0→≥1 for the SAME adopted entity, over the real cluster fabric, nothing hand-fed) + `crossing_e2e_no_dest_boundary_means_dest_fires_nothing` (the negative control). Cite `scripts/slice4b_design_adversary.md` FINDING-1 (the redrive keys off `in_flight` membership; the per-entity serialization + self-heal make the direction key unnecessary). The throwaway `spike_self_heal.rs` was promoted into this file and deleted. **NO change to** `RequestInFlight`'s type/keying, `crossing_transfer_id`, the `Occupied` suppress arm, `DemoteCmd`, or any frozen wire.
    - **(c) DYNAMIC-PARENT REPARENT DEFERRED to P8.** `DemoteCmd.new_parent` + `apply_reparent` (a container-realm — a Ship — physically moving between parents, so its boundary re-nests at commit) is NOT built this slice: it is only exercisable when ships-as-realms physically move (P8), and planting inert `Some`-arm wire now would create an HR5-uncoverable production-unreachable branch (adversary HIGH-2). It rides the P8 ships-as-realms work. Slice-4b's `outward_dest` re-homes to the boundary's STATIC authored `parent`, which is all the pre-P8 nesting (SOI/Station/Bay boundaries authored under a fixed parent) needs.
    - **Still-deferred (unchanged):** the per-(entity,boundary) cooldown (#4) and stable `winner_ix` (#5) stay 🟧 — orthogonal to the direction axis (the self-heal design touches neither the per-entity cooldown nor the raw-index dwell reset). This slice does NOT regress them.
  6. **Trigger scale O(N·M²)→O(N·M) (audit L3):** `evaluate_realm_boundaries` is O(N_subjects · M_boundaries²) via `boundary_depth`'s per-candidate O(M) parent-chain walk (+ a per-subject `Vec` alloc). Precompute each boundary's depth ONCE per tick into a `Vec<u32>`; the proper scale answer is a per-cell boundary index on the D-41 grid (shared with D-9 AoI). Inert at M=0; land before a dense-boundary soak.
  7. **DRY id primitive (audit L2):** `crossing_transfer_id` (vd-wire, 0x39) and `rehome_transfer_id` (vd-node, 0x37) are byte-identical FNV-1a bodies differing only by tag. Hoist `namespaced_transfer_id(tag, seed_bytes)` into vd-wire (callers pass their own postcard seed — handles crossing's `attempt` arity); both call it. Fold into 3f-D (where `crossing_transfer_id` changes signature anyway).
  8. **Counter honesty (audit L10):** `crossing_unresolved`/`crossing_subject_gone` increment per REDELIVERED request (unlike the guarded `crossings_started`) — a stuck entity reads as a 20 Hz storm. Guard symmetrically OR rename to a `_requests` rate gauge.
  9. **TRANSIENT crossing batch-saga-start (goal-audit-e2e wf_6b221e10, empirically caught by the 3g Test 4):** `handle_transient_crossing_request` (`saga_runtime.rs:1460`) GRANTS the dest (`TransientCrossingGrant`) but — unlike the durable `handle_crossing_request` (`:1440` `start_transfer`) — never STARTS a `BatchHandoff` saga. So a crossing-triggered transient's downstream handoff has no live saga: when DEST acks `BatchAdopted`, `deliver` (`:1110`) early-returns on the unknown transfer → the item never promotes `Arriving→Held`, no go-token is written (`orch.batch_goes=[]`, `dest.owned_transients=[]`). The 3g durable e2e (Test 1) proves the DURABLE crossing composes end-to-end autonomously; the TRANSIENT crossing proves only Leg-1 + auto-grant (Test 4) — the batch completion is OWED. FIX: wire a Transient `start_transfer` (a `BatchHandoff` ctx keyed on the `batch` id, source=`from`, dest=`rec.authority.node()`) into `handle_transient_crossing_request` alongside the grant, mirroring the durable path (HR2 — transient composes through ONE machinery). Then extend Test 4 to assert `dest.owned_transients` + `batch_goes` non-empty. Land before task #133 closes (or ledger to P4 with the Station/Bay transient work).
  9-STATUS 🟩 **TRANSIENT BATCH-SAGA-START LANDED (D-43 #9, plan `scripts/transient_batch_saga_plan.md`):** `handle_transient_crossing_request`'s resolved arm now appends a `contains_key(&batch)`-guarded Transient `start_transfer` (the HR2/HR3 twin of the durable path — same `BatchHandoff` FSM, `class: Transient`, inert `session=SessionId::NONE`/`subject=Realm(to)`/`gateway=from`), keyed on the SAME `batch` the dest acks `BatchAdopted` under. No double-emit: the source ships its batch on the GRANT flip (`on_transient_crossing_grant`), NOT on the wire-silent go-token, so the grant/emit leg is UNCHANGED (`saga.rs`/`stub.rs`/frozen wire untouched). New `SessionId::NONE` sentinel (`core/src/ids.rs`, `u128::MAX` — never aliases a real session id). Tests: A (unit — grant starts `BatchHandoff{AwaitAdopt}` + redelivery idempotency via the `!contains_key` FALSE arm), B (harness 3-node composition — `orch.batch_goes` + `dest.owned_transients` non-empty + `verify_transient_authority_held` passes, the exact before/after symptom cured with a REAL dest reacting to a REAL envelope), C (crash-safety — the newly-reachable transient `BatchHandoff{AwaitRelease}` `SagaSnapshot` rehydrates across a kill-9 with `SessionId::NONE` intact + re-drives). Per-entity keying is CORRECT (one saga per crossing entity, adopt-before-drop lands) but NOT coalesced — see 9-SCALE below.
  9-SCALE 🟧 **Per-entity batch keying is CORRECT but not coalesced (the G-TIER promise unmet at entity granularity):** N distinct transients crossing the same boundary in the SAME tick = N distinct `batch` ids (`crossing_transfer_id(Entity(e), src_realm_fence, 0)`) = N sagas / N go-tokens / N fsyncs; `emit_transient_batch`'s group-by collapses to singletons. Re-key the batch on the **realm-pair + `src_realm_fence`** (`crossing_transfer_id(Realm(to_realm), src_realm_fence, 0)` or a dedicated `(from,to,src_fence)` seed) so the existing group-by collapses N→1 (one envelope / one saga / one go-token / one fsync). NOT mechanical — it ripples the FROZEN wire (`TransientCrossingRequest`/`TransientCrossingGrant` in `intershard.rs:630`, each currently carrying a single `subject`) + the source-side `on_transient_crossing_grant` flip (`stub.rs:1622`, which flips exactly one item) + `fan_out_crossing`'s per-entity request emission — a wire-versioned redesign, NOT foldable into the correctness fix. **OPEN Q3 (gates the SCALE priority):** does the density soak (`p1_volume_dense_hundreds`) route N DISTINCT entities across a boundary in one tick, or same-entity redelivery (already coalesced to one saga by the `contains_key` guard)? Read `p1_volume_dense_hundreds` + its crossing driver before scheduling — if the realistic dense corner is same-entity redelivery, the SCALE slice is genuinely deferrable to P4/P5 physics-debris (where true multi-entity debris storms first exist). Do NOT implement the re-keying yet.
- **Landed already (goal-audit L4 fix, this pass):** `fan_out_crossing`'s durable arm no longer `.expect()`s the session — a Durable-TAGGED entity in the held-transient set (a mis-tagged batch item) now DEGRADES (counts `crossing_durable_no_session`, emits nothing) instead of panicking the shard (regression `a_durable_tagged_transient_degrades_instead_of_panicking`).
- **Pin (exists-to-be-flipped):** `stub.rs` `fan_out_crossing` + `RequestInFlight`/`CrossingProgress` docs, `saga_runtime.rs` `handle_crossing_request` unresolved arm, `scripts/slice3f_optionB_plan.md` + memory [[project_spatial_boundary_trigger_plan]]. **3f-D abort durability + the CLUSTER-tier abort/restart legs (1-CLUSTER-STATUS) have landed 🟩** (the pre-CAS abort composes end-to-end + survives an orchestrator World-rebuild + the empty-store negative control). This whole entry flips 🟩 when the first real band is crossed with a proven latch-clear + re-drive AND the remaining owed items above land — the process-tier crossing-SIGKILL (1-SIGKILL-OWED) is the only crash-leg gap, a composition gap (not a durability gap), blocked on a shard-bin boundary-plant knob.
- **Source:** holistic audit `wf_3916069b` (2026-07-13, post-42d7b73 — 0 CRITICAL, 2 HIGH + 12 MED/LOW, all inert; this entry closes the audit's L7 registry-gap + tracks H1/H2/M1/L1/L2/L3/L5/L9/L10).

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
  end-to-end instead of re-allocating per input/client/tick;
  (d) **SCALE-FANOUT-COPY** — the read-plane snapshot fan `on_shard_frame` opens with
  `for session_id in sessions.subscribers_of(from)`, and `subscribers_of` does `set.iter().copied().collect()`
  into a FRESH `Vec<SessionId>` every shard frame (an S-element heap alloc 20×/sec per shard, S→1000 on a popular
  planet) on the path otherwise hardened to "a fence check + a refcount bump". Fix: borrow the `BTreeSet` (split the
  borrow / stage outbound NodeIds once) so the hot fan allocates no owned Vec.
- **Where:** `crates/connection-plane/src/gateway.rs` (`drive_pending_sessions`; `on_client_input`'s
  `bytes.to_vec()`; `subscribers_of`'s per-frame `collect`); `crates/sim/src/io/mod.rs` (`BoundedInbox::push`);
  `crates/node/src/orchestrator.rs` + `saga_runtime.rs` (`serve_directory` / `drive_sagas` inbound loops).
- **When / proper:** SCALE-INBOX / SCALE-5 / SCALE-INPUT-COPY
  / SCALE-CLOUD-1 → the **P1/P3 load-test + observability slice** (the load-tests-when-applicable standard): an
  index for active vs pending sessions + a ring/heap for the inbox + an Arc-carried input body + publish the
  admin snapshot on-change or at a sub-tick rate (SCALE-CLOUD-1: the orchestrator bin currently rebuilds+clones
  the FULL directory + saga views every tick at `tick_hz` — `crates/bins/src/bin/orchestrator.rs`), each sized
  by a real bench, not before.
- **Source:** whole-codebase audits `wwk1uh5k9` / `wo2gkj7t7` / `wg9gc765s` / `wdznr0x6i` / `wxwinv5no`
  (SCALE-1C2-1 / SCALE-3-inbox / SCALE-5 / SCALE-CLOUD-1 / SCALE-3-input-copy) + `wf_b82d1a67` (SCALE-FANOUT-COPY).

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

### D-27 🟧 Dest `OpenInputSlot`: the AUTHORITY adopt (front half) landed (1c.8); pose/render/ghost/multi-shard-routing (back half) is 1d
- **Landed (1c.5):** to let the transfer-DESTINATION shard ACCEPT (not drop as `UnknownSession`) the post-
  marker input the gateway drains at commit, `OpenInputSlot` minted a provisional `Dot` on the dest:
  `input_active: true` (applies input via the existing `apply_input` + `last_applied_seq` dedup, seeded to
  `marker_seq`) but not yet authority-held.
- **Landed (1c.8 — AUTHORITY adopt, the FRONT half):** `OpenInputSlot` now also carries the transfer `subject`
  (verbatim from `CommitAuthority`); the dest extracts the `Entity` and ADOPTS it — its provisional dot's
  `entity` becomes the SUBJECT id (not a fresh mint), `adopting: true`, and `request_pending_grants`'s 3-way
  emits `HeadRead{Entity}` (NOT a `LeaseGrant`, which the post-genesis CAS fence would Refuse). The existing
  grant-flip then sets `granted: true` + `entity_fence = record.fence (== new_fence)` matched by `dot.entity ==
  subject`, so `held_entities == [(SUBJECT, new_fence)]` matches the directory's `Entity(SUBJECT)@Shard(DEST)@
  new_fence` (FENCE-9 passes). The load-bearing `granted` flag was SPLIT: `granted` is now AUTHORITY-held
  (oracle/held_entities/directory-record-owner — flipped on adopt) while a NEW `render_ready` gates
  `emit_frames`; an adopt sets `granted` WITHOUT `render_ready` (renders NOTHING — no origin-teleport) and
  WITHOUT `SessionAttached` (the source still owns the client — R2). A non-`Entity` subject is a counted no-op
  (`StubStats.input_slots_malformed`), never an extraction panic. This GRADUATES `verify_authority_settled`/
  `verify_authority_unique` after a full transfer (D-28(b) below).
- **Landed (1d.1 — the POSE crossing, the first BACK-half piece):** the entity STATE now crosses. The saga
  flushes the source pose (`FlushSource`→`SourceFlushed`, behind the pose-before-promote FSM gate so a lost flush
  blocks the commit, never ships a poseless crossing), emits a `StubCrossing` envelope at commit, and the dest
  STORES the sanitized pose on the adopted dot — journaled exactly-once by `(transfer, STUB_CROSSING_STEP)` (1d.0
  `AppliedSteps`, D-21/D-22). The held-poses crossing gate proves the crossing via the dest pose's FRAME (the
  SOURCE realm frame seed-7, unreachable by the dest's own input — RED-on-drop verified by mutation; see D-21).
  `render_ready` STAYS false (the dot still renders nothing) — the VISIBLE flip is 1d.3.
- **Landed (Track R / 1d.2 — DEST FRAME ROUTING + the avatar re-point):** the gateway now routes N shards into N
  per-session subs (a lock-free `SubTable` on `Arc<SessionHot>` + the `subscribed_shards` reverse index; node-class
  dispatch is the STABLE `GatewayConfig.known_shards`, fan-out is the mutable per-session index). At adopt the dest
  emits the ADDITIVE `ShardToGateway::SubscriptionReady`; the gateway opens a SECOND per-session sub on the dest at
  the dest realm fence and RE-POINTS the avatar's render authority to it (`AuthorityChanged{entity, dest_sub}` —
  FORK 0a, the read-plane analog of the write-plane `CommitAuthority`), so the client renders the avatar EXACTLY
  ONCE (the source copy is composited-suppressed by `DeliveredView`, double-vision-safe). `ReleaseSubscribe` closes
  the source sub (one-tick `Draining` grace, then swept); `AbortTransfer` defensively closes any dest sub. The
  client admits the held-sub SET (1d.2d). The 2-shard capstone (`gateway::tests::capstone_two_sub_overlap_…`)
  proves both subs route at their OWN realm fences and the avatar resolves to one sub. `render_ready` STAYS false —
  the dest sub is OPEN + ROUTABLE but emits no frames; the VISIBLE flip is the only remaining gating piece (1d.3).
- **Landed (1d.3 — the VISIBLE flip):** the dest renders the crossed seed-7 pose; the capstone
  `p2_dod_the_cross_shard_crossing_renders_at_the_dest_at_the_crossed_pose` proves the source→dest flip
  (overlap_ticks==0 + 0<vanish_gap<=budget, the interim vanish pinned exists-to-be-flipped → [[D-2]]).
- **Landed (1d.4b — the per-entity Authority FSM ATTACHED):** `authority.rs`'s `Authority` (Owned/Frozen/Ghost) is
  now the per-entity TRUTH on the stub `Dot` (`simulates()` is the `emit_frames` gate, half the `apply_input` gate,
  and the oracle held-set). `render_ready` is REMOVED entirely. Login AND the transfer-dest both mint
  `Ghost{GENESIS}` and Promote `Ghost→Owned` via the IDENTICAL machinery (kind-generic — HR2). The source
  self-fence demotes `Owned→Frozen→Ghost` and RETAINS the dot (the FIRST ghost; no more `dots.remove`); the oracle
  excludes the Ghost so it cannot false-trip `verify_authority_unique`. FG-2 honest single-truth: `granted`/
  `entity_fence` stay the directory-record PREDICATE/poll bookkeeping until 1d.5b's poll tear-out.
- **Still owed (1d.5b+ — the rest of the BACK half):** the real ghost-as-collider FEED + registration (`GhostFlow`
  Spawn/Delta/Despawn + the dest `GhostColliderRegistration`; the retained source Ghost EXISTS now but emits no
  collider feed yet — collision RESPONSE is P5, [[D-2]]/audit `wf_b82d1a67`); the `AbortTransfer`-tears-down-the-
  dest-INPUT-SLOT teardown on the STUB side (1d.2b closed the gateway dest SUB on abort, but the dest's provisional
  `input_active` `Dot` is not yet despawned on abort). The dest spatial admissibility check stays PRE-commit at
  `PrepareSubscribe` (D-21).
  **FRAME-REBINDING (audit finding 5) — 🟩 AUTHORITY REBOUND (2026-07-16):** the crossed pose is now re-expressed
  into the DEST realm's frame at the ONE machinery every cross-realm hand-off funnels through —
  `vd_core::frame::rebind_pose_to_dest`, called by the durable crossing (`build_crossing`), the D-37 forward re-home
  (`build_rehome`), AND the D-7 transient batch (`emit_transient_batch`) (HR3 — one rebind, no per-path fork). The
  SOURCE computes the dest pose and ships it verbatim (`transfer_protocol.md` §6), so authority never depends on
  cross-binary float reproducibility. Through P1-P3 the `IdentityFrames` context makes this a pure FRAME-field rebind
  (position/velocity/orientation UNCHANGED, only the frame label flips `SystemSpace{7}`→`SystemSpace{8}`), so the HUD
  reads the dest realm the instant authority commits while the dot stays put. The pinned RED-to-flip gates flipped
  seed-7→seed-8 across `frame.rs`, `saga_runtime.rs`, `stub.rs`, `p2_transfer_gates`, `crossing_e2e`, and
  `render_crossing_smoke`.
  **[⚠ CORRECTED 2026-08-14 (Stage-C audit :812 — the three named callers had become fictional at HEAD while this
  🟩 stood):** `rebind_pose_to_dest`, `IdentityFrames` and the safe-degrade test named below were later DELETED
  (the placement arc, D-PLACE-1). The ONE-rebind law lives on split by SL1 direction in `vd_sim::stub`:
  `flush_pose_for_dest` (the SOURCE converts only into a realm it authors — descending — and otherwise ships
  VERBATIM) + `place_arriving_pose` (the RECEIVER accepts its own frame, ADDS a direct child's placement, and
  REFUSES anything else, counted + loud). All three ingresses run the receiver guard — the durable crossing, the
  D-37 re-home, and (Stage-C fix, audit :105/:374/:384) the D-7 transient adopt. STILL-OWED item (1) below is
  superseded the same way: the P4/P5/P10 ephemeris conversion lands inside these two helpers, not as an
  `IdentityFrames`→`FrameContext` swap. The `to_parent` threading this row's sibling paragraph describes is a
  ★DEAD wire field (D-WIRE-1).]**
  **STILL OWED (P4/P5/P10):** (1) the NON-IDENTITY transform — swap `IdentityFrames` for the closed-form ephemeris
  `FrameContext` so a cross-realm rebind actually MOVES the pose (a P10 warp between star systems light-years apart);
  the `transfer_frame`/`FrameContext` signature is frozen, so this lands the function body with NO caller reshape.
  (2) the CLIENT-COMPOSITE re-expression of NEIGHBOUR realms — rendering every realm around the player in the player's
  own coordinates (the pure-renderer redesign, `project_greenfield_client_tech`); today the client observes the single
  authoritative realm. (3) `frame_for_realm`'s `Area` arm threads its planet-parent provenance — **now EXERCISED by
  the task #133 Station/Area plant (2026-07-17): `realm_regions_for` builds an `Area(7)` region whose frame resolves
  via `frame_for_realm(Area(7), Some(Planet(7)))` → `AreaLocal{planet_seed:7, area_seed:7}`**; a genuinely
  un-nameable dest ~~currently SAFE-DEGRADES to the source-frame pose (label lags, NEVER a dropped hand-off) —
  covered by `rebind_pose_to_dest_safe_degrades_an_unnameable_dest_to_the_source_pose`~~ **[corrected 2026-08-14:
  that test died with the function (D-PLACE-1); the receiver guard now REFUSES an un-nameable dest — counted, loud
  (`UnplaceableArrival::UnnameableOwnFrame`), covered by `place_arriving_pose`'s own unit tests — the silent
  safe-degrade was itself the defect the guard replaced]**.
  **EMPIRICALLY CONFIRMED (2026-07-16, `render_crossing_smoke`):** over the live dual-shard process tier a dot
  re-homes System(7)→System(8), the client observes it (own `authoritative_sub` flips 0→1, DEST owns the entity), the
  dot renders at the correct WORLD position (box B), AND `DevState.location` now reads `"System 8"` post-crossing — the
  frame-rebinding re-expresses the delivered pose into `SystemSpace{8}`. The smoke now gates on BOTH the delivered
  render truth (`expected_box`==box B + the dot's pixels + DEST-owns + auth_sub-flip) AND the flipped `location` label.
  GENERAL: diagnose the delivered state, not a derived label — this fix is the label CATCHING UP to an
  already-correct delivered state (the earlier "System 7" misread was the label lagging, not the crossing failing).
- **Also owed:** the EARLY (prepare-time) `OpenInputSlot` for the gateway-adoption-mid-cut race is a **P3**
  resilience item (the gateway re-drives the slot before the buffer drain); inert in 1c (single process, commit
  emits the slot in the same handler before the drain).
- **Where:** `crates/wire/src/session_flow.rs` (`OpenInputSlot.subject`; the additive
  `ShardToGateway::SubscriptionReady` arm — 1d.2b); `crates/connection-plane/src/gateway.rs` (`apply_commit`
  forwards `subject`; the `SubTable`/`SubEntry`/`SubRecord` registry + `subscribed_shards` reverse index +
  `open_sub`/`close_sub`/`publish_subs`/`sweep_draining`, the `SubscriptionReady` handler, `apply_release`/
  `apply_abort` sub-close — 1d.2a/b/c); `crates/sim/src/stub.rs` (the adopt mint, `subject_entity`, the 3-way
  `pending_grant_op`, `flip_grant` → `GrantFlip::Adopted` carrying the `SubscriptionReady` egress — 1d.2c,
  `Dot.adopting`/`Dot.render_ready`, `emit_frames` on `render_ready`); `crates/harness/src/client.rs`
  (`SubscriptionClosing` handling; the held-sub SET — 1d.2d); `crates/client/src/view.rs` +
  `crates/wire/src/channels.rs` (`classify_snapshot(held_subs)` — 1d.2d).
- **Source:** Slice 1c.5 design `wf_9679f7c7`; the front-half adopt landed in Slice 1c.8; DEST FRAME ROUTING +
  the avatar re-point + client multi-sub admission landed in Track R / Slice 1d.2 (a/b/c/d).

### D-28 🟧 Cross-cut INPUT-CONSERVATION: the durable-player zero-fault instance is CLOSED (1c.7); the tail forms are owed
- **Landed (1c.7):** the end-to-end gate `tests/tests/p2_transfer_gates.rs ::
  p2_dod_cross_cut_input_is_conserved_exactly_once` drives a REAL transfer through the REAL saga producer
  (`SagaRuntimeRes::start_transfer` → `PrepareSubscribe → RequestCut → FreezeSource → commit_cas →
  CommitAuthority`) over a two-stub fabric topology, the client stamping the CUT_MARKER in response to the real
  `RequestCut` (then pausing across the freeze window so no `seq > marker` leaks to the source) and resuming so
  post-marker inputs BUFFER and DRAIN to the dest. It asserts `verify_input_conservation == Ok` (the gateway's
  drained output fed into a REAL stub `apply_input` — the two crate halves JOINED for the first time), a REAL
  source/dest partition (`source <= M`, `dest > M`, `dest.min == M+1`, non-empty drained batch), the marker
  THREADED (the gateway-emitted `resume_from_seq` latched on the dest == the client's own marker seq, no
  hardcoded literal on either side — via `StubStats.last_input_slot_resume` → `InspectReport.dest_resume_seq`),
  zero `input_window_evictions`, and the directory CAS landed authority at the dest. Five negative controls are
  checked-in + locally verified (lost-drain ⇒ Unaccounted, mis-thread ⇒ dest-empty, no-pause ⇒ source-leak, all
  confirmed RED; double-apply + out-of-order-drain covered at the unit level). A determinism sibling proves the
  bounded-poll choreography is byte-identical under one seed.
- **Landed (1c.8 — the DONE/settle tail, closes (a)+(b)):** the same gate now drives the saga to `live() == 0`
  (the bandless interim + the gateway `Released` ack close `Demoting → Releasing → Done`; D-2), the dest ADOPTS
  the subject and the source SELF-FENCES its dot (D-27 front half), and — after a two-window precondition poll
  (SOURCE reports nothing for the subject AND DEST holds it, resolving the transient promote-then-demote windows)
  — it asserts `verify_authority_unique` AND `verify_authority_settled` Ok, the directory holding exactly one
  `Entity(SUBJECT)@Shard(DEST)@new_fence`, DEST `held_entities == [(SUBJECT, new_fence)]`, SOURCE holding
  nothing. A stability-soak sibling proves the settle is idempotent under continued operation. This is the
  D-28(b) graduation — the kind-generic HR2 backstop is now assertable after a FULL transfer.
- **Still owed (tail forms):** (c) the EPOCH-stamped ghost-misapply form (`test_harness.md`
  §8#9 CONFLICT-B: input carries `(cid, authority_epoch)` + a `stale-authority` discard reason) — the oracle/
  `InspectReport` key only on `(SessionId, seq)`, so this proves the epoch-LESS projection; lands with the real
  ghost (D-27); (d) the TRANSIENT-class re-run — BLOCKED on the P3 `TransientGo` go-token (D-7), not merely
  deferred; (e) tick-skew + N-concurrent-transfer siblings (the kind-generic scenario is parameterized to re-run
  for them once the lockstep gate + demote tail are in).
- **Where:** `tests/tests/p2_transfer_gates.rs`, `tests/src/lib.rs` (`p2_cluster`/`trigger_transfer`/
  `read_subject`/`saga_states`), `crates/harness/src/client.rs` (`RequestCut` → marker + pause/resume),
  `crates/harness/src/topology.rs` (`dest_resume_seq`/`input_window_evictions` scrape, `ShardNode::as_any_mut`),
  `crates/sim/src/stub.rs` (`StubStats.last_input_slot_resume`).
- **Source:** the 1c.5 audit `wf_e3397eb2` + Slice 1c.7 plan `wf_ab09d799` (the focused-scope decision: input
  conservation now, the authority-settled pairing with the demote tail).

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
- **1d.1 sibling (same class):** the entity-STATE crossing (`StubCrossing`) shares this no-re-drive posture — the
  saga emits it ONCE in the `CasWon` batch (`emit_crossing`), with re-emission ONLY on the (not-yet-runtime-fired)
  `Swapping`/post-commit `Timeout` retry. The dest `PendingCrossings` buffer handles the ADOPT-ORDERING race
  within a single delivery (decoupling arrival from the late adopt), but a genuinely LOST crossing delivery has no
  recovery in 1d.1 — its at-least-once rides the SAME Slice-2 adaptive-deadline machinery as every post-CAS
  command (`CommitAuthority`/`ReleaseSubscribe`), and the proper retain-source-until-crossing-acked gating is the
  D-2 (b) tear-out. Same recovery hook as this entry.
- **Source:** the 1c.5 audit `wf_e3397eb2` (openinputslot-deferred-loses-buffer-no-redrive); the 1d.1 crossing
  sibling added in Slice 1d.1.

### D-14 🟥 `rendered()` recomputed ~3×/display-frame on the client
- **Missing:** the windowed/headless render path recomputes the composited view 2–3× per frame.
- **Where:** `crates/client-render/src/lib.rs` (sync_world + draw_hud + location); `crates/client/src/view.rs`.
- **When / proper:** the **P2 client interest-cap work** — sample `rendered()` once/frame in `Update`, publish
  as a Bevy resource the egui pass reads. Pure perf; negligible at a handful of entities.
- **Source:** P1.5 foundation audit (`scalability-1`).

### D-15 🟩 `walk-to`/`look-at` wired through `vdctl` (landed early as S5b, Jul 2026)
- **Landed:** the closed-loop `walk_to`/`look_at` dev-control commands drive the own avatar to a WORLD
  position / facing by injecting ordinary Move/Look at the input seam. `DevRequest::WalkTo`/`LookAt` are handled
  in `client.rs` `serve_conn` (`drive_walk_to`/`drive_look_at`, sharing a `drive_closed_loop` core that mirrors
  `wait_until`), REUSING the pure `vd_client_harness::nav` math (`walk_to`/`look_at`, already 100% Tier-A — no
  new math). `DevEntityRow` gained an `orient: [f64;4]` field (the delivered `StampedPose.orient`, sanitized to
  IDENTITY-on-nonfinite) so look-at reads the current facing — **NO frozen-wire change** (orient already rode
  the wire). vdctl `walk_to`/`look_at` subcommands (defaults arrive_epsilon 0.5 m / align_epsilon 0.02 rad /
  400·200 tick budgets). **NO client prediction** — the loops steer on the DELIVERED (lagged) state; the server
  stays sole authority.
- **Adversary-caught CADENCE FIX:** the driver polls every 5 ms but the sim assembles input at 20 Hz and `Look`
  ACCUMULATES (`add_look`), so a naive per-poll emit would sum ~10 clamped deltas into ONE datagram (~10× turn →
  oscillation under lag). The drivers emit AT MOST ONE input per assembled sim step (gated on `step_seq`
  advancing) — one clamped delta per datagram, matching nav's convergence model.
- **Proof:** `crates/bins/tests/dev_control_nav.rs` — a REAL client logs into a REAL cluster; walk_to converges
  to a world target, look_at aligns (verified by reconstructing `orient·-Z` from the new field — the end-to-end
  orient wire-through), an unreachable target TIMES OUT bounded. Plus vdctl parse tests + vd-client
  `sanitize_quat` (both arms) + the devstate orient assertion. `just gate` GREEN (Tier-A 100%).
- **Frame note (does NOT corner P5):** the drivers consume `orient` as a full WORLD-SPACE rotation (correct
  through P3 — all live spawns are `FrameRef::SystemSpace`, up == +Y). When planet radial-up lands, the fix is
  the sim's ONE `orient_from_angles` composing `frame_from_up`, keeping the stored orient a true world rotation
  the drivers read UNCHANGED; the `forward_in_frame` seam stays deliberately inert (wiring it now would
  double-apply the frame).
- **Where:** `crates/bins/src/bin/{client.rs,vdctl.rs}`, `crates/devproto/src/state.rs`,
  `crates/client/src/net.rs`, `crates/bins/tests/dev_control_nav.rs`, `crates/bins/Cargo.toml`.
- **Source:** Slice-3 T6 scope decision; landed early as S5b (cloud-base finish). Design + adversary:
  `scripts/s5b_vetted_design.md`.

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

### D-30 🟥 The PAIRED wgpu visual (client2 watches client1 cross a boundary) — the HR6 consumer of the visible crossing
- **Landed (1d.3):** the dest now RENDERS the crossed entity (`render_ready` flipped on the dest crossing-apply;
  the dest emits the crossed seed-7 pose; the client composites it from the dest sub on the REAL `vd_client`
  `DeliveredView`, wired into the harness `ScriptedClient`). The 1d.3b WireMonitor render oracles
  (`verify_no_vanish` / `verify_pose_continuity`, `world_pos`-evaluated, derived ε/K, with their negative-control
  meta-test) and the per-tick observer refactor of `run_cut_transfer` (sampling DURING the window, byte-identical
  under one seed) are landed. Zero GPU, zero gateway/`view.rs`-render-logic/`emit_frames`/snapshot-fan change.
- **Missing — the SEAMLESS in-process gate (blocked on D-2, NOT on 1d.3):** the headline "renders exactly once
  DURING a two-holder overlap, NO-VANISH (K=0), pose-continuous through the source drop" gate is NOT assertable
  today: in the zero-fault path the SOURCE sub closes ~3 ticks BEFORE the DEST sub opens at the client (a real
  vanish, no overlap) because the source release fires on the **unconditional** bandless `interim_demote_complete`.
  The seamless overlap requires the proper observer-watermark-gated demote — **D-2(a)+(b)**. So this gate lands with
  the D-2 band/ghost slice (1d.4/1d.5), reusing the 1d.3 oracle + observer + REAL-view machinery already in place.
- **Missing — the wgpu paired-client visual:** the real two-client wgpu scenario (client2 screenshots client1
  crossing the source→dest boundary, per `PLAN.md:186/212`). Blocked on: (i) a 2-shard local-process cluster
  (`crates/bins/src/bin/vd-devcluster.rs` spawns ONE `vd-shard`); (ii) a second capture client + cross-boundary
  `vdctl` driving; (iii) ~~**D-15** (`WalkTo`/`LookAt` wired)~~ ✅ DONE (S5b, Jul 2026) — cross-boundary vdctl
  driving is now available. The render crate is real and `G-RENDER-SMOKE` is live, but single-shard/single-client;
  remaining D-30 blockers are (i) the dual-shard cluster + (ii) the second capture client.
- **Where:** `crates/bins/src/bin/vd-devcluster.rs` (dual-shard spawn), `crates/bins/tests/render_smoke.rs` (the
  paired scenario sibling), `crates/bins/src/bin/client.rs` (D-15), `justfile` (a new `render-cross` recipe);
  the seamless in-process gate in `tests/tests/p2_transfer_gates.rs` (reusing `vd_harness::oracle` render oracles).
- **When / proper:** the seamless in-process gate at the **D-2 band/ghost P2 slice**; the wgpu paired visual at the
  **P2 client-track slice** that lands D-15. Sibling to D-16/D-17 (the existing G-RENDER-SMOKE GPU/terrain limits).
- **Source:** Slice 1d.3 design + the 1d.3 implementation finding (the unmasked window is a vanish, not an overlap).

---

## BLOCKING SPIKES (must run before the phase they gate)

### D-18 🟩 snapshot-datagram loss over the k3d overlay — ROOT-CAUSED + FIXED + LIVE-PROVEN
- **✅ RESOLVED (Jul 2026): it was OUR CODE, not the environment.** Root cause: datagram reading was ASYMMETRIC —
  the `read_datagram` loop lived ONLY in `serve_connection` (the ACCEPT path); the DIAL path spawned
  `dispatch_streams` (reliable, both sides) but NO datagram reader. A peer that DIALS never reads the datagrams
  the far end sends back over that connection. A CLIENT dials the GATEWAY → login/Active works (reliable streams)
  but snapshots (unreliable datagrams) were silently dropped on the client, with `send_datagram` returning Ok on
  the gateway (no drop counter moves). Invisible in the fully-booked mesh (each receiver's peer dials a fresh
  connection to it, so datagrams always land on an ACCEPTED connection); fatal for reply-on-connection (the
  gateway can't dial an unbooked client, so it replies over the client's INBOUND/dialed connection). MTU / PMTUD /
  "Docker Desktop artifact" were ALL wrong guesses — a pure-loopback quinn repro (test
  `ca1_reply_on_connection_delivers_an_unreliable_snapshot_datagram`) nailed it with NO env at all. **Fix
  (mesh.rs):** extract the loop into `read_datagrams` and spawn it on the DIAL side too, symmetric with accept —
  universal, no profile gate. **Proofs:** loopback repro red→green + 62/62 mesh tests; LIVE on default-VXLAN
  Docker-Desktop k3d the agent went `snapshots_applied` 0 → **2227** (decode_errors=0). See
  [[project_d18_datagram_overlay]].
- **✅ input→movement leg RESOLVED (Jul 2026): it was a SCENARIO axis bug, not a system bug.** With snapshots
  flowing the agent E2E reached `move 1 0 0` → pos advance, which appeared stuck. Root cause: the shared movement
  convention (`vd_core::kinematics::local_axes_from_movement`) maps `move 1 0 0` (forward) to world **−Z**, but
  `scenario-boundary.sh` asserted on `pos[0]` (X) — so it checked the wrong axis. The avatar was moving all along.
  Fix: the scenario now checks the axis-agnostic **Euclidean displacement magnitude** (no hand-re-encoded axis
  drift — the sim owns the ONE convention). LIVE-proven on default-VXLAN Docker-Desktop k3d: the avatar moved
  `[0,0,0] → [0,0,-1.0]` (exactly −Z as predicted) and the **full agent-HR6 boundary-crossing E2E now PASSES**
  end-to-end (login → Active → own_entity → snapshots → forward input → shard authoritative sim → displacement
  threshold met).
- **✅ SPIKE-3a LANDED (Jul 2026): the 2nd hard latency gate.** `crates/io-prod/tests/mesh_snapshot_latency.rs`
  — ONE real-QUIC loopback sender→receiver connection on which a ~1.8 GB reliable `Saga` burst saturates the
  wire while a 20 Hz unreliable Snapshot datagram stream competes on the SAME per-peer FIFO lane. Hard gate
  (release-only): delivered-snapshot p99 < 15 ms (a named, headroom-justified const, ~20x the observed ~0.7 ms
  and under one 50 ms tick), plus an always-on ≥50% delivery-ratio floor, a `datagrams_dropped_too_large==0`
  honesty check, and a `reliable_acked ≥ bulk_pushed/2` competitor-real check. Result: the snapshot hot path
  stays **sub-millisecond and lossless** under massive bulk (the "hundreds in one location + a bulk transfer must
  not starve snapshots" property). DRY (HR3): `percentile_unstable` extracted to `vd-harness` (`latency.rs`),
  shared with the SPIKE-2a gate. Wired into `just spike3a` + `just gate`. Adversarially reviewed (no bugs; the
  debug-window/env-floor/jitter concerns folded in).
- **Source:** `PLAN.md` SPIKE list.
- **⚠️ CONFIRMED LIVE (S5a agent-HR6, Jul 2026, k3d cluster voxeldust-newsystem):** the in-cluster agent-client
  logs in over real QUIC/mTLS + reaches Active + gets its own_entity (the RELIABLE control plane works
  end-to-end over the flannel VXLAN overlay), but `snapshots_applied` stays **0** — the gateway logs a steady
  `OutboundBox staging over cap: shed oldest UNRELIABLE frames (Snapshot/Input) toward sustained-congested
  peer` (~every 20ms), and the client receives NOTHING (`decode_errors=0`, `stale_frames_dropped=0`). Root
  shape: the servers tolerate the overlay because their control plane RETRANSMITS (reliable streams); the
  20 Hz snapshot fan-out is FIRE-AND-FORGET unreliable QUIC datagrams, and those are lost on the overlay
  (path-MTU / datagram-loss), degrading the connection so the gateway can't drain → sheds. This is EXACTLY the
  datagram-over-overlay delivery gap SPIKE-3a exists to close.
- **⚠️ MTU-PIN HYPOTHESIS DESIGNED + IMPLEMENTED + DISPROVED LIVE, then REVERTED (Jul 2026, understand
  `wf_7bdd33d9` + design `wf_a06f687e`):** the vetted fix was a cloud-gated quinn MTU pin (`initial_mtu(1350)` +
  `min_mtu(1200)` + `mtu_discovery_config(None)`, plumbed via `resolve_mesh_mtu`). It was fully implemented +
  GATE-GREEN (Tier-A 100%, io-prod Tier-B floor held, workspace 59/0). **But the LIVE re-run on the pinned server
  image FALSIFIED it:** (a) the gateway STILL shed "sustained-congested peer" (the pin did NOT stop the snapshot
  loss); (b) the pin REGRESSED the client login (it reached Active+own_entity BEFORE the pin, and could NOT after
  — an asymmetric pinned-server / unpinned-client handshake break). **KEY INSIGHT (why MTU is a red herring):**
  the snapshot datagram is ~1115 B and the measured overlay MTU is 1450 (eth0/cni0/flannel.1 = 1450), so a 1115 B
  datagram ALWAYS fits — even UNPINNED, quinn discovers the 1450 path and 1115 < 1450, so `datagrams_dropped_too_
  large` should never trip. The real snapshot loss is a DEEPER unreliable-QUIC-datagram delivery failure over
  **Docker Desktop's k3d VXLAN / VM networking** (the LinuxKit/gVisor UDP stack drops fire-and-forget QUIC
  datagrams for reasons OTHER than payload size), which is very possibly a LOCAL-DEV-ONLY artifact that would NOT
  reproduce on a real cloud CNI + real nodes. The MTU-pin code was REVERTED (it regressed login without fixing the
  loss).
- **✅ FLANNEL-MTU-BUMP CONTROL RUN — MTU DEFINITIVELY RULED OUT (Jul 2026):** recreated k3d with the flannel
  `host-gw` backend (`--k3s-arg '--flannel-backend=host-gw@server:*'` → NO VXLAN encapsulation → pod MTU = the full
  host **1500**, up from the VXLAN 1450; VERIFIED live: `cni0=1500`, no `flannel.1` iface). Redeployed the
  (unpinned, current-code) server image + re-ran the S5a agent Job. **RESULT: the client STILL reaches Active +
  own_entity but `snapshots_applied` is STILL 0** (decode_errors=0) — raising the pod MTU 1450→1500 (where even
  quinn's 1452 DPLPMTUD probes fit) changed NOTHING. This is the DECISIVE bracket: MTU / PMTUD is conclusively
  **not** the cause (confirming 1115 B < any MTU always fits). D-18 is a DEEPER unreliable-QUIC-datagram delivery
  failure over Docker Desktop's k3d VM networking, independent of MTU/VXLAN — most likely a LOCAL-DEV-HARNESS
  artifact (LinuxKit UDP stack dropping fire-and-forget datagrams), NOT a product bug.
- **STILL OWED (re-scoped, MTU path now CLOSED):** (1) confirm on a REAL cloud CNI (a real k8s cluster, not
  Docker-Desktop k3d) whether the snapshot datagrams deliver — if they do, D-18 is fully a local test-harness
  artifact + the S5a live proof should run there; (2) only if it reproduces off-Docker-Desktop, dig the datagram
  send path (ECN/GSO/UDP behaviour), NOT MTU (disproved). **The S5a HARNESS + the login/Active/own_entity e2e are
  PROVEN correct on the overlay; only the unreliable-snapshot leg fails, and that leg is NOT confirmed to be a
  real-cloud problem.** See [[project_cloud_ready_k3d_plan]].

### D-19 🟥 SPIKE-6a (rapier snapshot/restore + cross-binary determinism) blocks P5; SPIKE-10a (dual-frame ship-interior physics) blocks P8.
- **Source:** `PLAN.md` SPIKE list.

### D-FO-7 🟧 The MOVERS-ONLY realm-feed filter is DELETED (SL4/owner Q3); the realm-lane SCALE decision is the owed half
- **What changed (the placement arc S3, 2026-08-14):** the movers-only filter on `authored_realm_snaps`
  was the last surviving rival "does this child have an orbit?" test (audit finding 25 / the owner's Q3
  ruling: *"'Movers only' is not available — it cannot be expressed without a motion test on the read
  path"*). It is DELETED: the feed ships EVERY direct child's authored row per tick, static and moving
  alike, level-triggered on the latest-wins lane (a lost datagram or a late joiner self-heals next
  tick — send-on-change over an UNRELIABLE lane would starve both, which is why the plan's
  per-observer-baseline variant did not land).
- **What's still owed (the SCALE half A4c named, unchanged):** the realm lane has no AoI cull, so a
  parent with MANY static children pays O(children×tick) fan-out — tolerable at THE world's tens of
  rows, a cliff at a planet with 1000 stations. At **P4** take the decision: a realm-lane AoI cull /
  send-on-change over a RELIABLE repair path, **vs** the child shard authoring its own box row (the
  DRY, SCALE-safe option). Send-on-change bandwidth at true scale is UNMEASURED.
- **★ THE DIRECTION IS RULED (owner, 2026-08-27):** *"Agree that some data (like this) should be sent
  through reliable lane."* The owner's own framing of the problem is the shape of the answer — *"why do
  you need to send it every tick? This should happen just once, when you load the game, you download the
  galaxy."* The lane splits by whether the thing CHANGES, not by what kind of thing it is:

  | | What | Lane | When |
  |---|---|---|---|
  | A galaxy's systems | never move | **RELIABLE**, retried until it lands | once on arrival |
  | A star system's planets | move every tick | ephemeral, repetition IS the loss story | every tick, as now |

  This is not a compromise between two options — it is the honest split. Repeating a thing that never
  changes is waste; retrying a thing that changes every tick is worse than useless, because next tick's
  value beats a resend of last tick's. **THE WORLD'S OWN GOLDEN DECIDES WHICH IS WHICH:** a static
  child's row is byte-identical at tick 0, 1,000 and 50,000 (`placement_rows.golden`).
- **★ MEASURED, so the cliff is a number and not a word:** one `RealmSnap` is **208 bytes**, so the
  per-tick frame at the target census is **31.2 MB — per subscriber, per tick**. That is not a slow
  message, it is an unsendable one: this lane STOPS WORKING at the census, it does not degrade.
- **★ RULED 2026-08-27: THE GALAXY CROSSES ONCE, OVER THE RELIABLE LANE.** Owner: *"we're passing the
  Galaxy just once over reliable lane, the rest in your proposal makes sense."* **Client-side derivation
  is REFUSED** — the client-only-renders law stands, and the saving comes from not repeating a thing that
  never changes. See `owner_decisions_2026-08-27_seed_and_secrecy.md` §S6.
- **★ THE NUMBERS, CORRECTED AND MEASURED ON THE WIRE** (an earlier figure of 31.2 MB was the IN-MEMORY
  `size_of`, not the encoded size — retired):

  | | |
  |---|---|
  | one `RealmSnap`, postcard-encoded | **95 bytes** |
  | 150,000 rows | **14.2 MB** |
  | per second at 20 Hz | **285 MB/s** |
  | ten subscribers | **2.85 GB/s** |
  | a star system's 9 planets | **855 bytes/tick = 17.1 KB/s** |

  **The star system is not the problem — 17 KB/s is nothing. The galaxy is,** and only because it repeats
  something that never moves. 14.2 MB does not fit a datagram: this lane does not degrade at the census,
  it STOPS.
- **THE SHAPE TO BUILD:** static rows ship on the RELIABLE session lane with the send-on-change baseline
  the body lane already uses (for a static child that means ONCE, plus the keep-alive re-assert as the
  heal); moving rows keep repeating on the ephemeral frame, where level-triggered repetition is still the
  right loss story. ⚠ **PAGING IS NOT IN THIS STEP:** at 14.2 MB the one-time transfer needs parts,
  acknowledgement and resumption, and none of that can be exercised until the census rises at S12. At
  today's world the whole static set is 189 bytes.
- **Where:** `authored_realm_snaps`, `crates/sim/src/stub.rs`; the moving-parent tripwire
  `d_fo_7_no_static_region_sits_under_a_varying_ancestor_chain` lives with the generator in
  `crates/physics/src/worldgen.rs`.
- **Source:** floating-origin A4c (`scratchpad/fo_server_plan.md`); superseded in part by the placement
  arc (Stage-C SL4 findings 24-28) and owner Q3.

### D-GATE-1 🟧 Process-tier port reservation is still TOCTOU outside the test binary, and the tier lock is void under a process-per-test runner
- **What's missing:** cluster-booting process tests are now serialized AT THE SOURCE by `vd_bins::cluster_tier()`
  (a process-local `Mutex` every such test holds for its whole body; `every_cluster_booting_test_holds_the_tier`
  in `crates/bins/src/lib.rs` proves none was missed, transitively — it closes over helper fns, so a test that
  boots via a fixture is caught too). That makes the plain `cargo test --workspace` honest, which a `justfile`
  flag could never do. TWO gaps remain, and both are stated in the guard's own doc so nobody mistakes it for
  complete: (1) **the lock is PROCESS-LOCAL.** It is sufficient only because `cargo test` runs test EXECUTABLES
  sequentially while parallelising WITHIN each one. Under `cargo-nextest` (a process per test), or two
  concurrent `cargo` invocations, or a second worktree building at the same time, it protects nothing.
  (2) **reservation stays TOCTOU.** `reserve_udp_addr`/`reserve_tcp_addr` bind `:0`, read the address, and drop
  the listener — so between the drop and the child's real bind, any process on the machine can take that port.
  Serialization removes the in-binary race, not the machine-wide one. `ClusterAddrs::reserve()` closes the
  weaker intra-struct aliasing hazard (all sockets held simultaneously, distinctness asserted PER PROTOCOL —
  UDP and TCP are separate namespaces, so a 17-element cross-protocol set assertion would itself flake).
- **Where:** `cluster_tier` / `CLUSTER_TIER` / `TIER_OWNER` and the `reserve_*_addr` helpers,
  `crates/bins/src/lib.rs`; the guard-first convention in every `crates/bins/tests/*.rs` cluster test.
- **When:** before the k3d/cloud CI work (task #123), where a process-per-test runner and a shared build host
  are both likely. Complete fix = deterministic per-fixture port BANDS — the `DevPortScheme` slot scheme
  (`crates/devproto/src/lib.rs`) and the `VD_RLM_FIRST_PORT` pattern already do this for the demand-spawned
  shards; retrofit the same onto the remaining ephemeral fixtures so no fixture ever races for a port at all.
  That subsumes gap (2) entirely and makes gap (1) harmless.
- **Source:** the five-item arc, gate-honesty slice (2026-08-03). RED control recorded before the fix:
  `probe_endpoints` at default parallelism gave 1 passed / 3 failed with shuffling identities.

### D-WORLD-1 🟥 Station and Area crossings are UNPROVEN
- **What's missing / lost:** the only process-tier exercise of the `AreaLocal{planet,area}` frame
  (`crates/core/src/pose.rs` — the one frame that REFUSES to form without a parent) and crossing depth
  5 → 4. The retired walk fixture claimed these legs; THE world cannot host them. (NOT missing: any
  `to_parent` "provenance conversion" — corrected 2026-08-14: that wire field is ★DEAD, D-WIRE-1; an
  Area frame forms at the RECEIVER from its own roster, `arrival_frame`, so the process-tier leg owed
  here is the roster-formed Area arrival, no wire provenance involved.)
- **Why:** `worldgen` has no Station/Area arm at all (`generate_system_forest` authors systems + planets
  only) — stations are PLAYER-BUILT. The old "proof" stood on dead shards that never booted (they failed
  `guard_regions_nest` with 0 ambient roots and exited), pinned green only because the walk gate never
  called `Cluster::first_exited` (D-WORLD-7).
- **Where pinned:** `world_roster`'s doc-comment (`crates/bins/src/lib.rs`) and the walk gate's header
  (`crates/bins/tests/node_per_realm_walk.rs`).
- **When:** the block/station slice that grows THE world; the legs return as `world_roster` entries and
  Chain legs, not as authored fixtures.

### D-WORLD-2 🟩 The unresolved-dest strand is CURED — bounded re-drives, then a source-local abort that clears the latch
- **What it was:** an occupant steered at a realm no shard hosts was dropped with no reply while its source
  per-entity latch stayed standing ("CROSSING UNRESOLVED", `crates/node/src/saga_runtime.rs`) — a
  permanent strand, not a soft failure (judge finding J-0). Observed LIVE 2026-08-15: `node_per_realm_walk`
  failed 2/4 runs when the fly-in grazed unhosted `Planet(1164718096683563219)`'s 3.95 m SOI mid-leg
  (wall-clock-dependent — the leg-start tick set the planet's phase); the drop was counted
  (`crossing_unresolved`) + WARNed at the orchestrator, the source latch suppressed every later crossing
  of that entity (`crossings_suppressed_in_flight`), and `request_ttl_ticks: 0` kept the 3f-D4 re-drive
  scan inert — nothing else fired, forever.
- **The cure (2026-08-15, owner-ordered):** the retry is ON and the strand latch CLEARS, production
  semantics end to end. (1) `request_ttl_ticks` is ARMED everywhere via
  `vd_sim::saga::derive_request_ttl_ticks` = `abort_deadline_ticks + POST_COMMIT_STEPS ·
  redrive_deadline_ticks + 1` (57 at the defaults) — strictly outlasting the worst HEALTHY saga resolve,
  the same saga term the arrival shield derives from, so a re-drive never races a live-but-slow saga.
  (2) Re-drives are BOUNDED by the new `StubConfig::crossing_redrive_budget`, derived
  (`derive_crossing_redrive_budget` = `abort / redrive`, 3 at the defaults — the saga's own patience
  ratio). (3) EXHAUSTION takes the SOURCE-LOCAL pre-CAS abort — `abort_crossing_latch` in
  `crates/sim/src/stub.rs`, the SAME body the wire `CrossingAborted` consumer runs (hold close, latch
  clear, attempt bump ⇒ a later crossing of the same entity mints a fresh id and FIRES); the entity
  stays simulated at the source (no saga ever started, nothing else to thaw), and NO new wire arm exists
  (SL6: the timeout is source-local). (4) The sit-inside refire is bounded by the EXISTING containment
  dwell: the exhaustion abort arms `last_commit_tick` at the abort tick, so `should_rehome`'s `k_dwell`
  cooldown plus the ttl cycle caps a parked dweller at `budget+1` requests per `(budget+1)·ttl + k_dwell`
  ticks (measured: `a_sit_inside_dweller_is_bounded_to_the_derived_refire_cycle`); a graze that already
  left simply continues (`a_graze_continues_unharmed_after_the_exhaustion_abort`). The pair reaches every
  shard as `VD_CROSSING_TTL_TICKS`/`VD_CROSSING_REDRIVE_BUDGET` (`crossing_redrive_env` in `vd-bins`:
  static clusters via `common_env`, demand spawns via the orchestrator's spawn anchors from its LIVE saga
  tuning); the in-proc fixtures run the same derivation (`vd-tests::armed_request_ttl_ticks`, harness
  `armed_crossing_redrive`). Headline arc pinned sim-tier:
  `an_exhausted_latch_aborts_locally_and_the_next_crossing_fires`.
- **Still true:** every static gate flies the ±Z polar corridor and `world_roster` ASSERTS the
  I-AXIS / I-POLE / I-RADIAL margins — the corridor is the reason a graze is RARE, the cure is the reason
  it is now harmless (a graze self-heals in ≤ `(budget+1)·ttl` ≈ 228 ticks ≈ 4.6 s at 50 Hz). The human
  playground (`scripts/crossing-playground.sh`) still runs `up --demand`, where approached realms spin up.

### D-WORLD-3 🟩 The one-hop sibling crossing is RETIRED — it never existed on THE world
- **What:** `System(7) → System(8)` in one hop came from an authored born-inside shell; THE world's
  sibling stars are ~12.8 km apart under a shared parent, so the honest path is home → galaxy → sibling
  (Chain legs A/E in `node_per_realm_walk`).
- **Do NOT re-add:** a one-hop sibling crossing needs authored geometry, which SL5 forbids.

### D-WORLD-4 🟩 The boundary-file override (`VD_REALM_BOUNDARIES` / `VD_DEVCLUSTER_BOUNDARIES`) is DELETED
- **What:** the shard's boundary-file substitution, its resolver/error/parser/in-realm guard, the
  source-shell writer + launcher injection, the two-box crossing playground, and the seed/visual scene
  emitters — the whole family by which one process could boot geometry the rest of the cluster does not
  have. Discharges the audit MAJOR on the retained override (:546-552 of the audit); the lib's "nothing
  can select a different one" claim is true for the first time. ONE emitter remains:
  `write_world_regions` (`vd-devcluster emit-world-scene`).
- **Guard:** a grep for the two env keys over `crates/` + `scripts/` returns nothing (live code/scripts;
  historical design notes under `scripts/*.md` keep their record), and the env-builder tests assert no
  shape's env carries ANY `*BOUNDARIES*` key.

### D-WORLD-4b 🟧 The shipped containment band is NOT speed-sized — and the only speed-sized band is now gone
- **What:** `worldgen`'s `BandConfig::build` constructs every shipped band with `v_rel = 0.0` (the
  velocity widening inert); the deleted `override_containment_band` was the only speed-sized band in the
  tree. `crates/sim/src/stub.rs`'s containment-site comment claimed the shipped band "is sized from the
  occupant's SPEED" — FIXED in the same sweep (the comment now states the static edges and names this
  row), so deleting the override did not launder the audit MINOR (:846).
- **When:** the velocity-safe widening mechanism is owed at P5 (physics), where relative speeds stop
  being a configured constant.

### D-WORLD-5 🟧 `generate_walk_forest` lost its last shipped-binary caller but still builds the in-process scenario tier
- **What:** the hand-placed walk forest no longer reaches any bin (the seed/visual emitters that shipped
  it to the client are deleted), and the SHIPPED-LIBRARY constructor is gone too: the gateway's
  `SeedInjectorConfig::default()` — which this row used to wave off as "the gateway's test fixtures"
  while it built a hand-placed walk world in production library code (batch review) — is DELETED
  (the injector takes an explicit lowered world; vd-connection-plane's vd-physics edge is dev-only).
  What still builds on the walk forest, named exactly: `tests/src/lib.rs` + the gateway's
  `#[cfg(test)]` fixtures (via `WorldView::hand_placed`), and the non-test `pub` constructors
  `WorldView::hand_placed` + `moving_children_for` in vd-physics itself.
- **Flips green when:** `grep generate_walk_forest crates/` returns nothing outside its definition.

### D-WORLD-6 🟧 Triple/Forest retired into Chain; NO launcher produces `VD_HELD_REALMS`
- **What:** co-hosting keeps unit coverage only (`parse_held_realms` + the shard-side parse); no shape
  emits the env. A shape with no consumer is exactly what rotted into the CRITICAL (a cluster naming
  realms the world does not contain), so the parse is covered but the co-hosting path is declared
  CONSUMERLESS here rather than left to look alive.

### D-WORLD-7 🟩 CORRECTION — `node_per_realm_walk` was structurally unrunnable AND inert
- **What it corrects:** the D-44 "PROVEN … System 7 → Planet 7 → Area 7 → Planet 7 → System 7 → Galaxy →
  System 8" claim (amended in place above). Four of the six Forest shards named realms THE world does not
  contain and DIED AT BOOT; the gate never polled `Cluster::first_exited`, so the deaths were invisible;
  and every leg target (x = 15/25/35/65) sat INSIDE THE world's 150 m home shell, so the green run
  crossed nothing.
- **The rewrite:** shape Chain (home + galaxy + inner planet + sibling star, every name off
  `world_roster`), legs C/D/A/E/F on the ±Z polar corridor, every leg asserting the REALM LABEL reached
  (never a coordinate), `first_exited` polled through bring-up, deadline 300 s, and the observed max
  entity fence PRINTED per run.
- **NEW STANDING REQUIREMENT:** every in-test cluster bring-up polls `Cluster::first_exited` — a dead
  child during bring-up is a loud failure, never a silent absence.

### D-WORLD-8 🟩 CORRECTION — the two pre-arc reds had ONE cause
- **What it corrects:** `dual_cluster_crossing_smoke` red and `rlm_proc_spawn_smoke` red were BOTH a
  cluster naming realms THE world does not contain, dying in `guard_regions_nest` with 0 ambient roots
  (`crates/bins/src/bin/shard.rs`). The D-44 present-tense "PROVEN by dual_cluster_crossing_smoke" and
  the D-45 C-6b "RETAINED as an OPTIONAL OVERRIDE" texts are amended in place above. The
  port-band-survivor diagnosis at `docs/design/rehome_one_mechanism.md` ("a SURVIVOR still holding this
  realm's slot in the RLM port band") is RETRACTED there: that smoke reserves fresh EPHEMERAL loopback
  ports, never the fixed band.
- **Standing rule:** a red gate is ledgered with a RED row and an owner decision — never explained away
  in commit prose.

### D-WORLD-9 🟧 The demand suite's exit gates were a FALSE GREEN, not only a flake
- **What:** both exit legs in `crates/bins/tests/rlm_demand_login.rs` flew `[220,0,0]` — straight
  through the orbital annulus (orbit radii ~17–142 m in the near-XY plane) — and broke on ANY label
  != "System 7", so an in-plane PLANET capture satisfied them.
- **Fixed (slice 5):** both legs re-aimed at the ±Z polar corridor `(0,0,−220)` (what I-AXIS licenses)
  through the shared `vd_bins::flight::cross_leg`, ASSERTING the destination label equals the GALAXY's.
  AMBER, not green: the polar corridor is a fixture discipline, not a product guarantee — an in-plane
  exit still captures a planet by design (containment doing its job); the row flips green when a gate
  exists that distinguishes "captured en route" from "arrived" for arbitrary player flight (the demand
  playground's own AoI/label surface).

### D-WORLD-10 🟥 `UniverseConfig::visual_scale()` — an authored geometry surviving in the test tier (finding 16 residue)
- **What:** the visual-scale preset family (`crates/physics/src/worldgen.rs::visual_scale` /
  `visual_geometry` / `visual_demand`) has ZERO non-test callers — production boots
  `UniverseConfig::world()`. Its consumers are worldgen's own unit tests, one gateway `#[cfg(test)]`
  rig, and `frame_conversion_e2e.rs`. SL5 says tests run THE world; an authored 5-planet preset in
  the test tier is the same class as the walk forest (D-WORLD-5), kept only because its consumers
  predate the multi-star world.
- **When:** rebase those consumers onto `UniverseConfig::world()` fixtures (player-built regions
  where a scenario needs shapes THE world lacks — the fixture-forest doctrine), then DELETE the
  preset family. Rides the D-WORLD-5 scenario-tier re-base.
- **Where:** `crates/physics/src/worldgen.rs`, `crates/connection-plane/src/gateway.rs` (test rig),
  `tests/tests/frame_conversion_e2e.rs`.

### D-PLACE-1 🟩 THE PLACEMENT ARC (Stage-C SL4 critical 24 + majors 25/26/27 + minor 28) — landed 2026-08-14
- **What landed:** ONE writer (`author_placements`, head of every synced tick) authors per-instant
  `PlacementBook`s into ONE `PlacementLedger`; every consumer (containment scan, crossing flush/arrival,
  AoI, scene lanes, realm feed) SELECTS a book by an instant it holds as data and reads rows with NO
  CLOCK (`vd_core::placement`; `transfer_frame`/`region_signed_distance` are monomorphic over
  `&PlacementBook`, `FrameError::InstantMismatch` refuses mixed times). The `FrameContext` trait,
  `LocalFrames`, `IdentityFrames`, `rebind_pose_to_dest`, the `_resolved` twins and `ChildFrame` are
  DELETED. The four rival has-orbit tests collapsed into the one writer; the fifth (the `v_peri` AoI
  width) IS `Motion::closing_speed_mps` — the one lowering (`to_regions`) reads the accessor for the
  AoI width (wired 2026-08-14: the batch review caught this row claiming the collapse while the
  accessor sat dead beside the still-live `orbital_of(..).map_or(0.0, v_peri)` test; the test is now
  the accessor's only expression, and `max_excursion_m` gained its stated consumers — the boot fence's
  `child_reaches` and the corridor-margin apoapsis terms). `vd-physics` (Tier-A) now holds
  ALL motion (celestial + `Motion` + the seed generator, split rule: reads-the-seed/mints-a-body moves,
  reads-`&[RealmRegion]` stays); the crossing path carries NO edge to it —
  `sl4_the_crossing_path_cannot_name_a_motion` in `tests/tests/crate_isolation.rs`, OBSERVED FAILING
  with the edge added (2026-08-14). Motion reaches the sim only as opaque injected
  `vd_core::placement::MotionFn` closures (the sim::io seam discipline) — a deviation from the plan's
  writer-in-node shape that preserves its structural properties (sim cannot NAME a motion; the reader
  cannot see time). The flush reads the world of NOW (the B-1 fix; latch→flush gap MEASURED at 1 tick
  in-process = 0.17 m @50 Hz production against the 1.0 m inset, pinned to 0 by `crossing_e2e`). The
  boot fence judges movers at APOAPSIS (`ChildReach`; the scalar-only planned form was REFUTED — walk
  Area A needs box-corner exactness) — on EVERY shard that plants a mover's row: the reach roster
  derives from THE world for every parent the neighbourhood names, never from the hosted shard's own
  moving subset, whose gaps judged a non-hosted mover at its zeroed centre, `Fixed(0,0,0)` — one
  forest, a different verdict per shard (the batch-review zero-reach MAJOR, closed 2026-08-14;
  `a_shard_that_does_not_host_a_movers_parent_still_judges_it_at_apoapsis`) — and THE WORLD'S NUMBERS MOVED (owner lever 1: `ECC_CAP_SIGMAS=4`,
  compression solved against apoapsis; planet SOI 4.1607→3.954 m with `VISUAL_SOI_GAP_FRACTION`
  0.35→0.372 preserving the pinned visible-from-anywhere property the plan's arithmetic missed).
  HR4/G-IDENTICAL: `a_kepler_child_and_a_thrusting_child_cross_by_identical_code` (one fixture, two
  motions, two shard kinds, identical crossings).
- **Golden instruments:** `crates/bins/tests/goldens/placement_rows.golden` (bit-for-bit rows through
  the shipped boot; regenerate ONLY with a stated world-numbers change) + the apoapsis tripwire + the
  flush-gap tripwire in `one_containment_answer.rs`/`crossing_e2e.rs`.

### D-PLACE-2 🟥 `RealmRegion.center` still stores a position (finding 27's residual; SL1 presence-leak)
- **What's missing:** the boot fence no longer reads a mover's zeroed centre (it takes the boot's
  `ChildReach`), but the FIELD survives as the static children's stored position and every shard is
  still planted with its own row's centre. The plan's full cure — positions living ONLY in authored
  placement books, statics riding `Motion::Fixed` in the boot roster, the field deleted (E0609) —
  is owed. Readers at HEAD: `placement_row`'s static arm and the boot roster. (Two readers this row
  used to name are GONE: `child_shape` died with the parent-authored outline — window lane Slice C2,
  minor 19 — and the `--realm-boxes` on-disk contract died at the Slice-C1 flag day. `VD_REALM_BOUNDARIES`
  was listed here too; the symbol was DELETED by the Stage-C batch-1 cluster rework and reads nothing.)
  **★ SLICE D (2026-08-16) TOUCHED NEITHER READER.** The window lane's capstone added drawing, pruning
  and pixel gates; it did not go near the boot roster or `placement_row`'s static arm, so this row's
  residual is exactly where Slice C2 left it — SHRUNK by two dead readers, and OPEN. Said plainly rather
  than quietly rolled into the slice's flips.
  **When:** with the motion-roster boot rework (the writer-in-node move below).
- **Where:** `crates/core/src/geometry.rs` (`RealmRegion.center`), `crates/sim/src/stub.rs`,
  `crates/bins/src/lib.rs`.

### D-PLACE-3 🟩 RESOLVED (window lane Slices C1+C2, 2026-08-16) — `RealmShape` carries no position, and the lane that restated outline centres is gone
- **WHAT was deferred:** owner Q2 ("the outline carries no position") was decided but not landed — the outline lane still restated outline centres at a pinned epoch instant, and the client still read `RealmShape.center`.
- **HOW it resolved:** the Slice-C1 FLAG DAY (client minor 18, floor → 18; owner-approved 2026-08-15/16 items 1/9/10) DELETED the `center` field from `RealmShape` in place — a shape is pure self-description, and there is nothing left to restate. Slice C2 then deleted the whole lane that carried it (`InterShardFlow::ChildSceneSet` tombstoned; `child_shape`, `SHAPE_LANE_TICK`, `restate_shapes_in_child_frame`, `lift_shapes_from_child_frame` and `shape_hop_to` all gone — the last four went in C1 with the up-shape half).
- **THE ORIGIN PROBLEM this row recorded as its blocker** ("how does the client learn which realm the scene is measured in") was answered by option (b)'s successor: the composed level carries an explicit ORIGIN MARKER `(origin: RealmId, origin_epoch: u64)` on the message itself (`ServerControlMsg::RealmRegistry`/`RealmSceneDelta` + `RealmSnapshotDatagram.origin_epoch`), which is `pin`'s lawful successor and cannot desync from the level it rides on.
- **Where the cure lives:** `crates/wire/src/channels.rs` (`RealmShape`, `SceneRow`, the origin marker), `crates/wire/src/version.rs` (minors 18/19), `crates/connection-plane/src/gateway.rs::compose_scenes`.

### D-PLACE-4 🟥 The placement WRITER lives in vd-sim behind the opaque seam; the plan's writer-in-node shape is owed
- **What's missing:** the plan puts `author_placements` in vd-node reading a `vd_physics::MotionRoster`
  so vd-sim holds NO motion data at all. Landed instead: the writer stays in vd-sim but every motion is
  an opaque injected `MotionFn` (sim runs it, cannot name it; the crate edge is banned + the `publish`
  mint is clippy-banned INSIDE vd-sim — measured 2026-08-14, an injected call produced the diagnostic —
  while `PlacementLedger::publish` stays `pub` and unbanned in vd-node/vd-bins/vd-tests: today no crate
  outside vd-sim names `Placements`, an observation, not a fence — batch review). Structurally
  equivalent for SL4's letter; the node shape additionally removes the closure seam AND is where the
  publish surface stops being public. **When:** with D-PLACE-2 (the same boot rework).
- **Where:** `crates/sim/src/stub.rs::author_placements`, `crates/node/src/lib.rs`.

### D-PLACE-5 🟥 `secs_since_epoch`'s per-shard `tick_hz` is an UNDEFENDED cross-shard dependency
- **What's missing:** cross-shard placement agreement (the anti-flap property) silently requires parent
  and child to share `tick_hz`; the per-shard knob is deliberate (`vd_core::kinematics`) and the local
  pair is validated (`validate_tick_pair`), but nothing checks AGREEMENT between two shards' clocks.
  A cross-shard fence would be new wire data (SL6: default NO — ask the owner before adding it, or
  derive it from the ClockSync lane's existing fields). An undefended dependency, not a defect.
- **Where:** `crates/core/src/kinematics.rs::secs_since_epoch`; every `author_book` caller.

### D-PLACE-6 🟥 Ledger `span_ahead` retention + `PlacementBook::project` are ledgered NON-features
- **What's missing (deliberately):** lanes DO present future instants — a sender's ClockSync arrival
  phase can lead the receiver's (the relay lane), and the arriving hand-off lane clamps BOTH
  directions — but every forward ask is answered by a HEAD CLAMP, counted and measured per direction
  (`placement_skew_clamped`, `placement_skew_ahead_max_ticks` / `placement_skew_behind_max_ticks`;
  the batch review caught this row claiming "no lane asks for a FUTURE instant" beside
  `book_at_or_head`, which exists for exactly that ask, and one |Δ| gauge conflating the two
  directions). What stays unbuilt is a forward RETENTION window and any projection: the clamp reads
  the world of NOW, which is the same one-instant rule the flush follows. If a lane ever must span
  more than the backward window, the ledgered fallback is `PlacementBook::project(row, dt)` —
  monomorphic, kind-blind, refused past a writer-authored horizon `Δt_max = √(2·budget/a_max)`
  (0.05 m budget at a_max 4.34 m/s² ⇒ 3 ticks @20 Hz / 7 @50 Hz; error ½aΔt² ≤ 48.8 mm / 7.8 mm).
- **Where:** `vd_core::placement::PlacementLedger`.

### D-LANE-1 🟥 Lane trust: the directory head is the ADMISSION authority, the learned frame sender stays the ROUTE — and NodeId itself stays self-asserted until P7 (Step-5 lane cure, findings 0/43, 2026-08-14)
- **★ SCOPE NOTE (window lane Slice C2, 2026-08-16):** FOUR of the five lanes this row cures are now
  TOMBSTONED and their receive gates deleted with them — `RealmCascade`, `ChildSceneSet`,
  `RealmObservation`, `RealmShapeObservation` — along with the counters named below
  (`cascade_unauthored`, `child_scene_unauthored`, `realm_observation_unattested`). The RULE is
  unchanged and still enforced on every LIVING realm-boundary lane: the SL7 `ChildLive` bit and the
  Q2 relay (`WindowRelay`) both admit only the node the directory names for that child, and the
  window lane attests every row against the `ShardRoster` head at the gateway. The paragraphs below
  are the rule's provenance; read the lane list as history.
- **WHAT LANDED:** every realm-boundary lane now attests its sender. Down (parent→child: `RealmCascade`, `ChildSceneSet`): accept iff the sender equals the cadence-resolved `ParentRealmNode`, else drop+count (`cascade_unauthored`/`child_scene_unauthored`), fail closed on an unresolved parent. Up (child→parent: `ChildLive`, `RealmObservation`, `RealmShapeObservation`): accept iff the sender equals the directory head for that child (`ChildRealmNodes`, written by the same realm-Head reply arm as the parent resolve, read eagerly on the same AoI cadence for demanded/held children; an unattested bit arms a lazy re-read — the [[D-RLM-6]] mechanism-C backstop, which is also load-bearing at BOOT: a parent holding nobody self-reports Empty before its demand loop, so the first bit's refusal is what pulls the head). **What this closes completely:** the ZOMBIE window — a deposed child still beating held the `home` route at its old fence until its own next beat; now its bit fails the head compare, the route is never refreshed, and the entry TTLs out deterministically, independent of the child.
- **THE TRANSPORT CONSTRAINT this row must keep saying (the H3 hole both Step-5 designs missed):** the mesh routes ONLY to statically-booked or CONNECTION-LEARNED peers (`crates/io-prod/src/mesh.rs` `unknown_destinations_are_loud_backpressure`, ~3606: an id absent from the address book is permanent, loud back-pressure). A directory-derived NodeId may therefore be UNROUTABLE — an RLM-spawned shard is in nobody's static book. So `ChildLiveEntry.home` (the learned return address, CA-1 reply-on-connection) is RETAINED as the down-lanes' route and the directory head admits it; the two are deliberately different jobs and folding them would hard-fail the down-lanes at the process/dev tier.
- **WHAT THIS IS NOT:** a security boundary. NodeId is self-asserted on the current trust model (see [[D-RLM-8]] — unchanged, not flipped); this is a live CORRECTNESS cure with a deterministic bound, not authentication. P7's connection-bound NodeId turns the same compares into a real boundary for free.
- **Where:** `crates/sim/src/stub.rs` (`ChildRealmNodes`, `update_child_node`, `head_reads_due`, the five receive gates), tests beside each.

### D-LANE-3 🟥 The observation lanes carry exactly TWO levels — depth 3 is an owner decision, not a default (Step-5 lane cure, finding 39, 2026-08-14)
- **★ SCOPE NOTE (window lane Slice C2, 2026-08-16):** the observation lanes this row bounds are
  TOMBSTONED. The two-level law SURVIVES on their successors and is pinned there: a parent's window
  statements name at most its DIRECT children, and the one level of interior comes from the live
  realm's OWN statements relayed VERBATIM one hop (`InterShardFlow::WindowRelay`, the Q2 ruling) —
  `stub::tests::window_lane::the_live_siblings_interior_is_one_level_out_and_never_deeper_q1` is the pin. The
  revisit trigger below is unchanged.
- **WHAT LANDED:** both up-relay recursions are DELETED (`emit_realm_frames`: the rows re-relay inside the observed-interior fan; the lifted-set flatten in the shape ship). The law: every lane carries exactly two levels — what I author about my children, and what my children authored about themselves; a level never relays what it was relayed. Volume before was O(the entire live subtree) per link per tick under a comment claiming visibility culls it (nothing did); the local interior fan is now also bounded by each observer's own AoI band on that child (`AoiMembership` in `emit_realm_frames` — one tick of lag on entry, grace-latched on exit).
- **WHAT DEPTH 3 WOULD HAVE SHOWN:** a neighbouring system's planets' MOONS, seen from the galaxy, before entering the system (both flown symptoms — "exiting the system freezes its planets", "approaching a system its planets never appear" — survive at depth 2: the level-1 ship is not the relay). **Revisit trigger:** the first world where a grandchild's angular size at the grandparent's typical viewing distance exceeds the eye's resolution (the moment P4 real-scale moons land), or an owner ruling on OD-5 look-ahead depth.
- **Where:** `crates/sim/src/stub.rs::emit_realm_frames`, `vd-wire::intershard` `RealmObservation`/`RealmShapeObservation` tombstone docs.

### D-LANE-4 🟩 RESOLVED (window lane Slices C1+C2, 2026-08-16) — SL3 reached: a realm authors HOW IT LOOKS, and no parent can state a child's outline at all
- **WHAT was deferred:** the outline a parent shipped for a child (`child_shape` — `shape: region.shape`, the parent's ROSTER's opinion) moving to the realm's OWN self-description, and the parent's per-child message shrinking to a placement and nothing else (SL3's own sentence).
- **HOW it resolved (owner-approved 2026-08-15/16, `docs/design/window_lane.md` §2.10 + §5 RULINGS):** a realm now states its own look on `ShardToGateway::WindowBody::SelfLook`, built from what it knows about ITSELF (its own boot extent via `RealmRegions::own_shape`), straight to the connection plane — and the two lanes a parent used to author outlines on are DELETED with their machinery: `InterShardFlow::ChildSceneSet` and `ShardToGateway::RealmSceneDelta` are tombstoned (minor 19) and `child_shape` no longer exists. What a parent may still say about a child is a PLACEMENT (`WindowFrame` rows) and, for a DORMANT child only, one photometric marker (§1.1 item 3b — the owner-ruled R4 datum, superseded the instant the child speaks for itself). Attestation makes the split structural, not careful: a `SelfLook` is legal only when `subject == the sender's own realm`.
- **THE OWNER RULING IT WAITED ON — answered by the marker model.** The question was: if a non-running realm draws nothing, is an empty star system invisible? The answer is that a dormant child is its PARENT's point of light, positioned by the parent's placement row before, during and after spin-up; only the LOOK payload upgrades (marker ⇒ self-look) when the child starts speaking, by data presence, never both and never zero. So the draw IS liveness-gated structurally, and nothing goes dark.
- **MEASURED, not argued:** `tests/tests/frame_conversion_e2e.rs::the_room_the_player_is_standing_in_has_exactly_one_author` (the room's look crosses the shard→gateway edge stated by its OWN shard and by no other host, and no host ever states a look about a realm it is not).
- **THE CLIENT-SIDE HALF LANDED AT SLICE D (2026-08-16), so this row owes nothing further:** markers draw as luma-driven point sprites (`vd_client::realm_scene::marker_look` — the MK class's own blackbody colour, radius `√L` off the one shared point-source base, then the shared apparent-size floor; one unit vertex buffer + one material per class, so a point of light never costs a mesh), and a self-look that stops being re-asserted is PRUNED after the derived roster-loss window so the parent's ever-present marker resumes (`WindowTuning::look_ttl_ticks` + `WindowIngest::prune_stale`). Both directions of the handover are measured in pixels by `just warp-pixels` — see [[D-WINDOW-1]]'s Slice-D entry for the numbers.
- **Where:** `crates/sim/src/stub.rs` (`emit_window_bodies`, `current_bodies`), `crates/wire/src/session_flow.rs` (`BodyStmt`, `window_body_admissible`), `docs/design/window_lane.md` §2.10.

### D-LANE-5 🟩 The SL7 bit beats on its contract's cadence; the retain TTL is derived in cadence beats (Step-5 lane cure, finding 41 — landed 2026-08-14)
- **LANDED:** the `ChildLive` emit is gated on the AoI cadence (`aoi_recheck_cadence` — the same expression the shape ship uses) plus the occupancy EDGE (`WasOccupied`: the transition itself, derived with no hook in any adopt path; going empty re-arms it) — the contract text (`intershard.rs`) and the code now say the same thing, ~25× less traffic at the shipped profile. `retain_ttl_ticks` = max(1 s loiter, 2 cadence beats + 1 tick slack) — the shipped 50 Hz/recheck-25 profile's exactly-two-beats-zero-slack equality was a coincidence, now a derivation. The beat RATE is asserted (no longer print-only) in `the_per_leg_liveness_cost_and_rate_are_measured` against the sender's own production expression; the freshness harness (stale-under-loss, never gone) is the TTL's experiment and stayed green unchanged.
- **The residual this bought (priced, accepted):** consumers of the bit read up to one cadence of staleness (the chain-latency gate's up budget is now sender-cadence-derived), and noticing a realm went empty takes up to one cadence longer.
- **Where:** `crates/sim/src/stub.rs` (`aoi_decide` bit emit, `WasOccupied`, `retain_ttl_ticks`), `tests/tests/frame_conversion_e2e.rs`.

### D-LANE-6 🟩 RESOLVED (Slice C1, 2026-08-16) — the `--realm-boxes` boot file is DELETED; the stream is the one scene source
- **WHAT was deferred:** the client's `--realm-boxes` boot path rendered realm outlines straight from an on-disk regions file — no running realm behind any box — and every pixel gate stood on it.
- **HOW it resolved (owner-approved 2026-08-15/16, decisions items 1/9/10 — THE DRAW LAW; docs/design/window_lane.md §2.11):** the flag day deleted the whole path — `--realm-boxes`/`debug_scene_arg` (client bin), `from_regions_json`/`from_boxes_json`/`load_scene` (vd-client), `write_world_regions`/`emit-world-scene` (vd-bins + devcluster), and the emit+pass in `scripts/visual-run.sh`/`client.sh`. The client draws ONLY the composed stream (the `RealmRegistry` level + deltas + composed datagrams); a realm that is not running appears only as its parent's marker datum. The pixel gates were re-based onto the streamed scene in the same cut (drawn set == the gateway-emitted level set; extents read streamed off `DevState.realm_boxes.extent_m`).
- **Where the cure lives:** `crates/wire/src/channels.rs::SceneRow`, `crates/connection-plane/src/gateway.rs::compose_scenes`, `crates/client/src/realm_scene.rs::from_scene_rows`, `crates/bins/tests/render_boxes_smoke.rs` / `render_crossing_smoke.rs` (re-based).

### D-WIRE-1 🟥 `to_parent` is a DEAD field on four frozen wire arms — flag-day removal owed (Stage-C audit :685/:812, 2026-08-14)
- **WHAT is deferred:** removing `to_parent: Option<RealmId>` from `CrossingRequest`, `TransientCrossingRequest`, `TransientCrossingGrant` and `FlushSource` (plus the `SagaCtx.to_parent` / `TransientStatus::Crossing.to_parent` / `LatchedCrossing.to_parent` carriers that mirror them). The field is produced and threaded end to end but READ BY NOTHING: the consumer it was appended for (`rebind_pose_to_dest`, wire minor 3 — the "Area label never flips" fix) was DELETED by the placement arc (D-PLACE-1). The receiver forms an `Area` dest's frame from its own ROSTER (`arrival_frame` → `RealmRegions::hosted_frame`, which carries the enclosing planet's seed losslessly), so the datum never needed to cross a boundary at all (SL6's own test: the receiver computes it from what it legitimately holds).
- **WHY it stays until a flag day:** postcard v1 is positional — deleting a field in place re-labels every later field on the wire while every same-build test stays green (the tombstone-discipline hazard `intershard_closed.rs` pins for ARMS, applied to a field). The frozen contract grows additively only. Until removal: the field docs are rewritten to ★DEAD tombstone tense (no doc may claim an Area frame "forms" from it), the three sim shape-pin tests keep the bytes pinned as SHAPE (their narration says so), and any NEW reader of `to_parent` is a defect — the roster is the one source.
- **WHEN:** the next flag-day wire MAJOR — the same bump that reaps the tombstoned arms (`OccupantInterest`/`ProxySceneSet`/`EntityInterest`/`EntityCascade`, `GhostFlow::Spawn`/`Delta`).
- **Where:** `crates/wire/src/intershard.rs` (the four arms), `crates/wire/src/version.rs` (minor 3), `crates/sim/src/saga.rs::SagaCtx`, `crates/sim/src/stub.rs` (`TransientStatus::Crossing`, `fan_out_crossing`), `crates/node/src/saga_runtime.rs` (threading).

### D-SHIP-1 🟥 Ship realms cannot be named by the lineage coordinate — every coord lane EXCLUDES them, counted, until P8 (Stage-C audit :713, 2026-08-14)
- **WHAT is deferred:** a lineage-coordinate arm for entity-backed `Ship` realms. `RealmKindTag` carries six seed-keyed tags and no Ship (`RealmLevel.seed` is a `u64`; `RealmId::Ship(EntityId)` keys on a `u128`), so `level_of(Ship) = None`, a Ship own-realm cannot build a `StubConfig` (`root_coord`'s expect), the shard bin refuses `VD_REALM_KIND=ship` at boot, and no Step-5 lane can NAME a ship child (demand/AoI, cascade targeting, scene reflect, interior fan). This is a first-class realm kind (PLAN.md P8: ship = own shard) structurally outside the realm lanes.
- **WHAT LANDED NOW (the interim):** `region_level` returns `Option` and every coord-needing lane excludes a Ship region GRACEFULLY — counted per lane pass (`StubStats::ship_child_regions_excluded`), never a panic (it was an `expect` that aborted the whole shard on one hosted ship region). A ship still counts where no coord is needed: its `ChildLive` bit is a child OBSERVER, so an occupied ship keeps its parent warm (SL7). Pinned by `stub::tests::aoi_demand::a_ship_child_region_is_excluded_from_every_coord_lane_counted_never_a_panic` and the `None` arm in `region_level_recovers_seed_lineage_kinds`. **The lane count fell FOUR → TWO at window lane Slice C2 (2026-08-16):** the cascade targeting, the interior fan and the scene reflect died with their messages, leaving the AoI/demand fold and a `Child`-scope window's hop row — the pin asserts `counter == 2` now. No producer plants a Ship region through P3, so the counter reads 0 in every shipped world.
- **WHEN:** P8 (the ship-realm work) — a new lineage arm PLUS a payload widening (the `u64` seed cannot hold an `EntityId`), i.e. a deliberate wire change on the frozen `RealmPath`, not an append; the counter and the exclusions retire with it.
- **Where:** `crates/core/src/realm_path.rs` (`RealmKindTag`), `crates/core/src/worldgen.rs::level_of`, `crates/sim/src/stub.rs::region_level` + the four lane guards, `crates/bins/src/lib.rs::realm_from_kind_seed`.

### D-WINDOW-1 🟩 RESOLVED (2026-08-16) — THE WINDOW LANE: the observer chain that replaced the bucket cascade, LADDER COMPLETE (slices 0/A/B/C1/C2/D all landed; owner-approved 2026-08-15/16, docs/design/window_lane.md)
- **WHAT this is:** the binding slice ladder for the window lane (`docs/design/window_lane.md`; the signed
  five-topic approval + Q1/Q2/Q3 rulings live in `docs/design/owner_decisions_2026-08-15.md`, 2026-08-16
  addendum). Each world level states only what it lawfully owns — placements, one hop row per occupied child,
  its own look, one marker per direct child, its SL7 membership verdict — straight to the observer's gateway;
  the connection plane stacks the statements per observer at ONE universe tick; the client only draws.
  Zero new realm→realm data; four inter-shard scenery arms end producer-less and are tombstoned.
- **The ladder (a slice is not done until its owner-visible outcome is real):**
  - **Slice 0 — the skeleton (🟩 landed).** The signed record in-repo; the wire
    types (`WindowOpen`/`WindowClose` + `WindowScope`, `WindowFrame`/`WindowBody` with typed
    `BodyStmt`/`WindowMembership` + `WindowId`/`HopRow`), attestation predicates, positional/roundtrip
    pins, mesh minor 16 with the owner citation; the per-system photometric draw in
    `generate_system_forest` (pinned f(seed) values). *Owner-visible outcome:* the signed ask + the wire
    contract. NOTHING MOVES — no producer, no consumer, no behavior change.
  - **Slice A — shard emits the window (🟧 THIS ROW'S STATUS: landed now;** old lanes still running,
    untouched; no client change). The shard-side window registry (`vd-sim` `OpenWindows`, keyed
    (opener, id)) + the DERIVED keep-alive TTL (2 beats + 1 of the ONE `aoi_recheck_cadence` — owner
    law 3(a)); per tick per open window ONE `WindowFrame` (the authored rows built once, the hop row
    inverted AT the author THROUGH `transfer_frame` itself — refusal semantics inherited, dropped +
    counted); `WindowBody` send-on-change + on-open (the look from the realm's OWN boot extent via the
    `vd_core::look` bag codec; markers from the boot-planted `ChildLuma` roster — `vd-bins`
    `child_luma_bags`, the boot/config path); `WindowMembership` diffed per window out of the ONE
    existing `aoi_decide` fold. Gateway side: the MINIMAL derivation (per Active sub: own-realm
    `Occupants`; `Child(c)` on the parent where the session's own subs + the login registry already
    name it — both chains held through a crossing overlap), keep-alive on the derived cadence, close
    on session end (zero sessions ⇒ zero windows, both sides), and fail-closed ingest (unknown window /
    forged sender / mis-authored body dropped + counted; admitted rows deliberately unconsumed until
    the Slice-B engine). Diagnosis: `windows_open` + the four egress meters on `StubStats`/the harness
    report; the gateway counters + `windows_open` gauge on `GatewayView`/`/admin/snapshot`.
    Gates LANDED: G-IDENTICAL (`assert_window_emission_feature_anywhere`, System/Planet profiles),
    attestation fail-closed (`a_window_row_before_its_engine_is_dropped_fail_closed_and_counted_apart`),
    TTL-expiry chaos (`a_dead_gateways_windows_die_by_the_derived_ttl_with_zero_leaked_emissions`),
    INV-BODY-AT-ORIGIN + inversion-inertness pins on THE world
    (`inv_body_at_origin_and_the_rotated_hop_inertness_are_pinned_on_the_world`), the Q1 two-level
    live-sibling pin (`the_live_siblings_interior_is_one_level_out_and_never_deeper_q1`), zero-state
    teardown both sides. **STOPPED SUB-ITEM (owner-gated, reported):** the Q2 PARENT-RELAY emission —
    the child→parent VERBATIM `WindowBody` ship has NO lawful carrier on the existing mesh:
    `RealmShapeObservation` is a typed `Vec<RealmShape>` with no fence and no bag field (it cannot
    carry the self-look TLV without a wire-shape change to a frozen struct), a parent-minted
    `WindowBody` would violate both the admission rule and Q2's "no re-state", and a NEW
    `InterShardFlow` arm is forbidden by the design itself (§2.5) and owner-gated (SL6). Costs
    nothing in Slice A (no gateway opens a live-sibling window before the Slice-B chain engine);
    the lawful shape of the relay leg goes back to the owner with Slice B. **SLICE B'S HANDLING
    (design-first, as ordered):** every enumerated carrier needs an owner ruling — (i) evolving
    the INTERIM `RealmShapeObservation` content into the verbatim relay is a frozen-struct wire
    change only the owner can authorize (his recorded "content evolves" intent, item 7,
    contemplates it — the postcard freeze still gates it); (ii) any new relay arm/variant is
    §2.5-forbidden + SL6-gated; (iii) the interim parent-authored restatement contradicts Q2's
    "no re-state" and would be unreachable dead code today (no gate topology holds a live sibling
    with interior children — measured: the exclusion class below counts 0). So Slice B EXCLUDES
    those rows from parity BY NAME (`parity_sibling_interior_excluded`, counted + printed by the
    gate) and the SL6-form ask rides the Slice-B report. The old lane keeps serving the pixels
    until the ruling.
  - **Slice B — gateway composition engine, SHADOW mode (🟩 landed).** `crates/connection-plane/src/window.rs`
    complete: level rings + newest-wins bodies + membership store per open window; the `transfer_frame`
    fold at ONE common tick (mixed ticks refused by `InstantMismatch`, per row, counted); per-stratum hold
    + the derived dead-hop exit; origin marker + epoch mechanics (bump on chain/origin change); the
    session/stream-only chain derivation (lineage recorded on the session from the login descent +
    crossings; parenthood confirmed by attested hop levels; realm→node from the gateway's own subs + the
    EXISTING `HeadRead{Realm}`/`Head` directory pair re-polled on the window keep-alive beat — the
    Slice-A seed-forest parent lookup AND the forest-read marker admission are DELETED, the roster now
    vouches markers). Output compared, never shipped (`shadow_compose` emits zero bytes). Gates landed:
    the process-tier parity MEASUREMENT with named mismatch classes (`window_shadow_parity.rs` /
    `just window-parity`, in `gate`) — **RAN GREEN and is RECORDED here; its comparator was RETIRED in
    Slice C2 with the lanes it measured** (§4.5 Topic 5's rule is "shadow parity BEFORE any client cut",
    and the client cut landed in C1; §4's Slice-C2 gate list carries no parity entry). The scenario file
    was NOT deleted: it is re-based onto composed self-consistency, because three §2.12 named invariants
    had no other process-tier home — the exact-cadence proof (`window_full_chain_folds > 0`), both dedup
    f64 agreements (`window_dedup_disagree == 0`, `window_fold_divergence == 0` with
    `window_fold_hits > 0`) and the live shear law (`window_instant_mismatch == 0`) — and it now also
    asserts the four deleted lanes stay SILENT in a real cluster. Its last parity reading:
    zero unexplained mismatches over hundreds of matched rows, sibling-interior exclusion class 0.
    G-SHEAR unit half incl. the deliberate mixed-tick FAILURE
    (`g_shear_the_deliberate_mixed_tick_compose_must_fail`); the exact-cadence boot pin
    (`window_full_chain_folds > 0` asserted in the parity run — a ≥2-level fold exists only at an
    identical cross-shard stamp); BOTH dedup f64 measured bounds (hop-vs-child-row
    `window_dedup_disagree == 0` + max-dev gauge; shared-vs-per-session `window_fold_divergence == 0`,
    non-vacuous via 2 co-located sessions); zero-state teardown (windows + realm-heads + scenes all die
    with the sessions); G-COMPOSE-LOAD both sides (`just window-compose-load`, p99 < one realm-lane
    tick, release); the shadow soak riding the `rlm-soak` recipe (state monotone-bounded, asserted).
    **THE NAMED PARITY EXCLUSION (recorded per the Slice-B plan):** old-lane rows lifted from a LIVE
    SIBLING's interior are EXCLUDED from parity as the counted, printed class
    `parity_sibling_interior_excluded` — their lawful carrier is the recorded Q2-relay gap (Slice A's
    stopped sub-item: the child→parent VERBATIM statement leg has no lawful mesh carrier; every
    candidate needs an owner ask — see the Slice-B report's SL6 ask text). The class counts 0 in the
    landed parity run (no live-sibling interior exists in its topology); it goes nonzero — loudly, by
    name — the day one does, and flips to composed rows when the owner rules the carrier.
  - **Slice C1 — THE FLAG DAY (🟩 landed 2026-08-16;** client minor 18, floor 8 → 18; owner item 9:
    no shims, no dual-decode). The client consumes the composed lane (`RealmRegistry` = the level
    `{origin, origin_epoch, rows: Vec<SceneRow>}`; the delta and the composed per-tick datagram beside
    it); `RealmShape.center` left the wire (a shape is pure self-description); the origin/epoch scene
    swap replaced `forget_space` (the one-space + echo guards stay ALIVE until C2 removes their cause —
    §4.5 Topic 4); `pin`/`render_pin` deleted; the `--realm-boxes` boot file + emitter + scripts deleted
    (D-LANE-6 flipped 🟩); picture gates re-based (§2.11: settle on the composed feed's own signal,
    streamed extents, DevState-driven dot detection, the origin-marker J1 + the crossing no-flicker
    gate). **THE SIBLING-INTERIOR CARRIER rode this slice by the owner's carrier ruling
    (owner-approved 2026-08-16 — docs/design/owner_decisions_2026-08-15.md addendum +
    docs/design/window_lane.md §5 RULINGS, recommendation 1 of the Slice-B ask):**
    `InterShardFlow::RealmShapeObservation` (disc 32) is TOMBSTONED — its interim content EVOLVED,
    exactly as its minor-11 entry promised — and its successor pair landed at mesh minor 17:
    `InterShardFlow::WindowRelay` (disc 34 — the child's VERBATIM self-authored statements as SEALED
    bytes + its own fence; the parent holds them UNOPENED and its whole lawful vocabulary is
    forward-or-drop) and `ShardToGateway::WindowRelayed` (disc 10 — the forward leg, admitted against
    the CHILD's identity by the existing predicates; admitted looks land in the one body store, so the
    §2.8 marker⇒look handover runs today; relayed interior LEVELS are held per child for Slice D's
    interior compose). The planets' REFLECTED marker draw (§1.1 item 3b "per direct child") landed in
    `vd-physics` (`reflected_photometrics`: L★·albedo·r²/(4d²), albedo seed-drawn over the canonical
    geometric-albedo table) — the flag day made markers load-bearing, so the owed planet datum could
    no longer wait. *Owner-visible outcome:* the game runs on the composed picture; boxes come only
    from the stream; dormant planets are their parent's points of light.
  - **Slice C2 — THE DELETION (🟩 landed 2026-08-16; mesh minor 19).** The §2.9 deletion map was
    RE-WALKED against the tree first, as the design orders (drift was proven, and found again: the SL1
    self-placement filter sat in the down-reflect emit loop, not at the design's quoted `stub.rs:8384`).
    **TOMBSTONED** (variants + payload structs remain, discriminants reserved forever, both exhaustive
    classifications frozen, received frames counted `undecodable` on their REAL carrier):
    `InterShardFlow::RealmCascade` (26), `RealmObservation` (31), `ChildSceneSet` (33) — beside
    `RealmShapeObservation` (32, minor 17). The golden tombstone set grew FIVE → EIGHT
    (`crates/wire/tests/intershard_closed.rs::the_tombstoned_discriminants_are_exactly_this_golden_set`
    pins {24,25,26,27,28,31,32,33} against a compile-forced exhaustive classifier).
    **TWO shard→gateway carriers went with them, producers deleted in the same commit:**
    `ShardToGateway::RealmFrame` (the old opaque per-tick realm datagram — the `Occupants`
    `WindowFrame` subsumes it per §2.9 step 3, and the composed feed has been the client's one scene
    author since the minor-18 flag day) and `ShardToGateway::RealmSceneDelta` (the per-dot OUTLINE
    push — §2.9 shrinks it to the ids-only `WindowMembership` verdict). Both are now counted-and-dropped
    at the gateway (`old_realm_frames_dropped`, `old_scene_deltas_dropped`), so a revived producer is
    loud rather than silent.
    **SHARD MACHINERY DELETED** (`crates/sim/src/stub.rs`): the cascade
    (`restate_rows_in_child_frame`/`restate_rows_in_frame`/`push_cascade`/`on_realm_cascade`, plus
    `hop_book`/`ActiveChild`/`active_children`/`home_of_active_child`/`any_active_child`/`placement_of`
    and the now-dead relay-instant selector `book_at_or_head`); the down-reflect block (`FromAboveScene`,
    `ChildSceneSent`, `on_child_scene_set` and the per-live-child emit loop); the observed-interior fan
    + TTL machinery (`ObservedInterior`, `ObservedInteriorShapes`, `interior_of`); the up-observation
    ship + `on_realm_observation`; the direct realm-datagram emit + its chunking and
    `RealmFrameCounter`; and the per-dot scene fold's shape-pushing (`child_shape`,
    `diff_scene_into_delta`, `RenderSent`, the own-outline lead) — which shrinks to the ids-only
    membership emit exactly as §2.9 says. FIFTEEN counters died with them: `cascade_rows_converted`,
    `cascade_rows_dropped`, `cascade_unauthored`, `misrouted_cascade`, `realm_cascade_received`,
    `realm_cascade_relayed`, `realm_observation_received`, `realm_observation_misrouted`,
    `realm_observation_unattested`, `realm_observation_unplaceable`, `child_scene_received`,
    `child_scene_misrouted`, `child_scene_unauthored`, `child_scene_unaddressable`,
    `realm_rows_unplaceable_observer`. THE ENTITY LANE IS UNTOUCHED (SL2 unchanged).
    **★ THE SL1 SELF-PLACEMENT FILTER RETIRED BY AMENDMENT, NOT BY EROSION** — owner ruling Q3,
    2026-08-16 (`docs/design/window_lane.md` §5 RULINGS; the owner had previously said "the filter
    stays" and then approved deleting it TOGETHER WITH THE LANE IT GUARDED, on the ground that the
    hazard ceases to exist). Its replacements are STRUCTURAL and both can fail:
    (i) COMPILE-LEVEL — `crates/wire/tests/intershard_closed.rs::no_living_inter_shard_payload_carries_a_realm_placement_or_a_centre`
    scans the wire crate's own vocabulary and asserts no LIVING inter-shard payload declares a realm
    placement, a centre, or the opaque pre-serialized datagram the dead lanes hid one inside — with the
    tombstoned payloads as the positive control proving the detector fires;
    (ii) END-TO-END — `tests/tests/frame_conversion_e2e.rs::no_message_into_a_realm_names_that_realm_or_carries_a_placement`
    sweeps every byte delivered into all three hosts of a live chain for 150 ticks and asserts no realm
    is ever told about itself (including inside the SEALED relay, opened by the test only).
    **SCENARIOS RE-BASED, NEVER DELETED:** the three chain descent scenarios became
    `the_authored_world_no_longer_descends_and_each_level_states_only_itself`,
    `no_message_into_a_realm_names_that_realm_or_carries_a_placement` and
    `the_room_the_player_is_standing_in_has_exactly_one_author`; the chain-latency and lossy-link gates
    re-based onto the Q2 relay leg + the leaf's own level at the gateway edge — and the picture's budget
    **no longer contains depth** (§2.13's structural claim, now a gate: one hop, not a sum over levels).
    **★ A REAL DEFECT THE DELETION EXPOSED, found and fixed:** the in-proc harness gateway ran at
    `tick_hz: 50` while its shards run 20 Hz, so the window keep-alive beat (25 ticks) was longer than
    the shard's window TTL (2 beats + 1 = 21) and EVERY WINDOW LAPSED between beats — measured as
    alive ~10 ticks, dead ~15, forever. The old scenery lanes had been masking it; with them gone it is
    a frozen picture. `tests/src/lib.rs` now states the cluster's one tick rate, and the leaf's
    own-level age at the gateway edge fell from p99 16 ticks to p99 1.
    *Owner-visible outcome:* four world-to-world scenery lanes gone forever; nothing changes on screen
    (the suite stays green).
  - **Slice D — THE WARP ACCEPTANCE (🟩 landed 2026-08-16; the capstone).**
    **MARKER POINT SPRITES.** A sleeping realm now DRAWS. `vd_core::look`'s `TAG_LUMA` datum
    `(class_code, luma_lsun)` reaches the client on the composed row and is turned into a point of
    light by ONE Tier-A expression (`vd_client::realm_scene::marker_look`): the colour is the
    Morgan-Keenan class's own blackbody sRGB (`MARKER_CLASS_SRGB`, the published sequence — a
    REFLECTOR carries its illuminator's class, so a planet reads in its star's colour), and the
    radius is `√L · POINT_SOURCE_BASE_RADIUS_M` — the equal-surface-brightness law, anchored on the
    ONE base radius the avatar dot is drawn at (`vd_client_render::DOT_RADIUS` now TAKES that
    constant rather than restating it). Every point then passes the shared apparent-size floor
    (`marker_world_radius`), so a marker is never sub-pixel and the pixel gates size their
    rectangles from the same two calls the renderer scales the sprite by. **A MARKER NEVER COSTS A
    MESH** (§2.14's tier-0 rung): the renderer builds ONE unit point-sprite vertex buffer and one
    unlit emissive material per spectral class at setup, and a sleeping star adds an entity and a
    transform, nothing more. **THE HANDOVER IS ATOMIC IN THE RENDERER:** the drawn body is keyed by
    `(entity, BodyKind)`, so a realm whose author flips (it woke and states its own outline, or its
    look was pruned and its parent's marker resumed) despawns and respawns IN THE SAME FRAME —
    exactly one of {marker, body} on screen, never zero, never both.
    **ROSTER-DRIVEN LOOK PRUNING** (§2.8's departure mirror). `WindowTuning` gained
    `look_ttl_ticks` (the same 2-beats-+-1 law off the same beat) and `WindowIngest::prune_stale`
    drops any SELF-LOOK — and any relayed interior LEVEL — not re-asserted inside it, counted
    `window_looks_pruned` / `window_relay_levels_pruned`. MARKERS ARE DELIBERATELY EXEMPT: a marker
    is the presence gate's FLOOR, and expiring it would blank a star rather than shrink a system to
    a dot. What makes the TTL sound is the other half of the change: a keep-alive `WindowOpen` now
    RE-ASSERTS every send-on-change lane shard-side (`OpenWindow::reset_baselines`), so a statement
    that stops arriving means the realm behind it stopped speaking rather than that nothing changed
    — and the beat is the SUBSCRIBER's own, needing no cross-process cadence agreement (the C2
    tick_hz lesson). Without the prune, a departed realm's last look would sit in the composer
    forever and the NEXT approach would draw a realm whose shard was not running — a body with no
    author, and a vacuous wake handover.
    **THE PILOT VIEW** (`client --capture-pilot`; `vd_client_harness::camera::pilot_capture_camera`).
    The default capture framing fits the whole drawn scene, and on a flight between two FIXED points
    of the star ring that frustum barely moves — so a system you fly toward could not grow on screen
    at all, and the warp acceptance is a statement about what the PILOT sees. The capture camera can
    now be placed at the avatar's eye along its DELIVERED facing (never a local view state — a
    headless run has no mouse), through the ONE Tier-A expression the pixel gates reconstruct it by.
    The scene-fitting framing stays the default, so every existing box gate is byte-identical.
    **★ A REAL DEFECT THE GATE EXPOSED, root-caused by measurement and fixed:**
    `vd_client_harness::nav::look_at` had NO brake against the delivered-pose feedback lag — the one
    `walk_to` has had since Stage B4. Measured on THE world: a 90° `LookAt` reported ALIGNED at an
    error of 0.0030 rad and then kept rotating for another **0.6030 rad (34.5°)** as the in-flight
    deltas landed; the aim froze a third of a right angle off target while the closed loop said it
    had arrived, and over the 12 031 m ring that is a 425 m miss — which is how the first warp
    flight sailed straight past its destination. Cure: `LOOK_FEEDBACK_STEPS = 4` (the same measured
    lag the walk brake is sized to), so the commanded delta is `error / 4` and the sum still in
    flight is strictly smaller than the error still to correct. Pinned by
    `the_look_brake_commands_a_fraction_of_the_error_so_the_turn_cannot_overshoot`.
    **THE GATES, with their measured numbers (2026-08-16, THE world, seed 0, 500 m/s, 50 Hz).**
    Budgets are counted in TICKS off the landed cadences and reported in metres by the occupant's
    own travel per tick (10.0 m) — measuring in ticks is what makes a handover honest whether the
    ship is flying or parked when it lands.
    - `just warp-pixels` (`crates/bins/tests/warp_pixels.rs`) — **G-WARP-PIXELS**: departure, the
      destination is its PARENT'S point of light at 3.00 px with 35 pixels painted at a LOCAL probe
      (asleep because the ring 12 031.398 m exceeds the wake radius 11 458.475 m by the world's own
      572.924 m margin); the growth curve holds over 2018 samples, 3.00 px → 90.21 px, worst
      sample-to-sample shrink 0.000 px; arrival 92.78 px, 25 150 pixels painted; the system behind
      hands back to its parent's marker and is drawn at the shared 3.00 px point-of-light floor; the
      ship then crosses in and STANDS in the sibling. Every drawn row's provenance is attested in
      the run manifest's state dumps (HR6).
    - **G-HANDOVER, both directions, symmetric.** WAKE budget = 2×25 AoI cadence + 1 reconcile +
      the cluster's OWN measured boot + 2 Q2 relay + 1 compose + 1 draw (+ the gate's own sampling
      resolution): **measured 14 ticks / 140 m against 61 ticks / 610 m**. DEPARTURE budget =
      51 grace-hold + 25 AoI cadence + 1 hop + 1 compose + 1 draw: **measured 47 ticks / 470 m
      against 83 ticks / 830 m**. THE Q2 RELAY HOP, asserted APART per the owner's ruling
      (everything after the shard is up): **measured 13 ticks / 130 m against 60 ticks / 600 m — it
      does NOT break the wake budget, so [[D-WINDOW-2]] stays closed and the direct window stays a
      fresh owner ask.** A companion test closes the whole derivation before any process runs (both
      budgets must FIT the geometry that provides for them), so a world-numbers change fails in
      milliseconds rather than only in a four-minute flight.
    - `just two-ships` (`crates/bins/tests/two_ships.rs`) — **G-TWO-SHIPS**, the owner-ordered gate,
      TWO real capture clients in realms of different depth: (a) the two chains agree on the
      star-to-planet separation to 0.0612 m against the planet's own 1.4646 m of travel over the
      8-tick sampling gap, each hull drawn by its OWN self-authored look and pixel-probed (50 px
      each); (b) the inner observer draws the Planet's own body around itself AT THE ORIGIN, in
      pixels — the hop row's other half; (c) 2139 relayed statements ingested, 0 undecodable — the
      hulls' looks rode the Q2 PARENT RELAY, attested in both manifests; (d) one crossing while both
      watch: the crossing client's epoch bumps EXACTLY once, the WATCHING client's does not move,
      and its picture is sampled continuously through the commit — 412 samples, 3288 body-to-body
      continuity checks, worst 1.088 m against its own 10.814 m allowance; (e) occupant figures are
      ABSENT from the window lane (every drawn row names a realm) and ABSENT out of a realm (the
      observer OUTSIDE the Planet is shown nobody standing inside it).
    - **G-SHEAR, the pixel half** (joining the Slice-B unit half): every captured frame's drawn rows
      are asserted inside the composer's declared retention — no row newer than the fold, none older
      than the window a held stratum or a relayed interior may lawfully carry, which those state per
      row by design. Measured spreads of 0–2 ticks across every capture of both gates.
    *Owner-visible outcome:* **the warp experience in pixels on THE world.**
    **DEFERRED flips:** D-LANE-4 🟩, D-LANE-6 🟩 and D-PLACE-3 🟩 all landed EARLY (C1/C2 discharged
    their substance — see those rows; Slice D's owed half of D-LANE-4, "the marker point-sprite
    RENDERING and roster-driven look pruning", is what landed here). **D-PLACE-2's residual SHRANK
    but STAYS OPEN** (🟥): `RealmRegion.center` still stores a position, still read by
    `placement_row`'s static arm and the boot roster — nothing in this slice touched either, and
    saying otherwise would be a claim rather than a measurement.
    **NEWLY OWED ROWS registered by this slice:** [[D-WINDOW-3]] (census-scale photometric
    derivation + the HDR exposure model the point sprite stands in for), [[D-WINDOW-4]] (P10
    rotated-hop composition), [[D-WINDOW-5]] (the §2.14 LOD tier seam for P4+), and
    [[D-WINDOW-6]] (two measurements this slice took and did not assert: a re-homed occupant is
    still shown its PARENT realm's occupants over the entity lane, and relayed interior LEVELS
    refuse at the fold in the two-ships topology).
- **Where:** `crates/wire/src/session_flow.rs` (the lane's types + attestation predicates),
  `crates/wire/src/version.rs` (minor 16), `crates/physics/src/worldgen.rs` (the photometric draw),
  `docs/design/window_lane.md` (the binding spec), `docs/design/owner_decisions_2026-08-15.md` (the
  2026-08-16 addendum).
- **THE LOOK-HORIZON EXTENSION (2026-08-17; slices 3 + 4 of `docs/design/look_horizon.md`, both
  owner-approved by the RULINGS ADDENDUM):** the lane now carries a grandchild's OWN picture two
  hops (mesh **minor 20** — THE SEALED INTERIOR FORWARD, §2 ASK A: `WindowRelay.statements`
  renamed `own` in place + `interior: Vec<InteriorRelay>` appended on both relay arms; the depth
  bound is the TYPE — `InteriorRelay` has no `interior` field; the §3.4.5 forward gate is the
  sender's own in-band verdict; the gateway vouches/fences and admits ONLY the author's own
  picture, with the lawful filter and the violation counted APART — G-VERBATIM and
  G-STRUCTURAL-SEAL pin it), and a parent may now tell ONE direct child that something outside
  may be looking in (mesh **minor 21** — THE INTEREST BIT, §2 ASK B, Q1 APPROVED:
  `InterShardFlow::RealmInterest`, disc 35, one byte, two lawful values, fail-closed admission
  mirroring the SL7 bit, decaying to 0 on silence; the down-proxy observer + the
  observer-origin flag that caps the cascade structurally — G-NO-CASCADE + G-INTEREST-BAND, the
  derived radii `444.104489631` / `469.104489631` m bracketing the 150 m shell). The Q2
  rationale clause ("am I observed from outside stays unrepresentable in every realm") ENDS by
  the owner's explicit amendment — recorded on the arm, the version ledger, and here.
- **LOOK-HORIZON SLICES 5 + 6 LANDED (2026-08-17; no wire change):** the FLOWN-SYMPTOM PIXEL
  GATE (`just look-pixels`, in `just gate` — `crates/bins/tests/look_pixels.rs`): one demand
  cluster on THE world PLUS the planted player-built station/area pair (the SL5 fixture-forest
  doctrine's process path — `vd_physics::worldgen::FixturePlant`/`station_area_plant`, ONE
  derivation every process boots via `vd_bins::process_world_config` + `VD_FIXTURE_PLANT`,
  measured by the same boot fences: both plants climb 2, admission-accepted at arity 2). At the
  derived park (outside the 150 m shell, inside the 444.104489631 m interior band, along the
  outer planet's instantaneous radius vector): every planet's presence is its OWN picture (RED
  for all five before slice 4), radii equal the camera model exactly (outer 20.48 px at
  169.5 m against the 3 px floor), all five DISTINCT, every planet row parenting on its star
  system throughout, marker handovers EXACTLY once each way with no blank frame and no radius
  step above the readback quantum, and the reverse handover inside a derived budget stating
  THE EXTRA RELAY HOP as its own term (measured 26–36 ticks against 123 + resolution).
  G-IDENTICAL (HR4) runs the same assertions on the planted STATION (same scene) and one level
  deeper on the planted AREA (19–20 ticks against 58 + resolution). G-NOTHING-OWED (THE LAW
  GATE) enumerates the generated forest OUT-OF-BAND at both parks: every subject above the
  minimum angle draws its OWN picture; oracle sets non-empty with depth-2 members (park A: 4
  owed / 3 at depth 2; park B: 7 owed, the area at depth 2). SLICE 6 pins: ROWS PER FOLD on
  the departure fixture (derived-set equality, 8 rows on THE plain world — the §5.4 promise
  that the design does not change rows per fold, in `warp_pixels.rs`); RELAY EGRESS measured
  == `W × C × blob × rate` with the gateway multiplier MEASURED (W=2 doubles exactly; blob
  PINNED at 1118 B against §5.2's ~1045 B model; `vd-sim
  slice6_relay_egress_measured_equals_w_times_c_times_blob_times_rate`); the UNION OVER-DRAW
  measured at interim scale (6 rows per fold per observer — recorded on [[D-LOOK-2]]).

### D-WINDOW-3 🟥 The marker's photometrics are a RENDERING convention, not an exposure model — and the census-scale roster bound is still owed (window lane Slice D, 2026-08-16)
- **WHAT is deferred, and stated honestly:** a point of light's DRAWN size today is
  `√(L/L☉) · POINT_SOURCE_BASE_RADIUS_M`, floored at the shared minimum apparent radius — the classic
  "size encodes magnitude" convention (which is also what the ambient backdrop starfield already draws),
  anchored on the ONE base radius the avatar dot uses so the two cannot drift. It is NOT an exposure
  model: apparent FLUX (`L/4πd²`) never reaches the pixels, so two markers of very different brightness
  at very different ranges can draw at the same floored size. The physically-exposed successor is an HDR
  apparent-magnitude/exposure pass — and **exposure is one of the eight rendering decisions already
  awaiting the owner** (`scripts/snowflow_technique_analysis.md`), so inventing one here would have been
  a guess dressed as a result.
- **THE OTHER HALF (ALL-2, promised at Slice D by the design's §2.10 and hole register):** the
  CENSUS-SCALE marker roster bound. Markers ship the FULL direct-child roster, unfiltered, so "all direct
  children in the sky" is a WIRE-VOLUME question, not a visibility one. At true galactic scale the bound
  holds only because the galaxy becomes the P10 cell-realm lattice, which caps any one author's direct
  children. That is an argument, not a measurement: the DERIVATION (rows/tick per author × the lattice's
  own fan-out, against `WindowTuning`'s budgets) is owed, and it is owed as a derivation, never a literal.
- **WHEN:** with the owner's rendering-decisions pass (exposure) and P10 (the lattice), whichever lands first.
- **SINCE LOOK-HORIZON SLICE 3 (2026-08-17) the census bound ALSO OWES THE INTERIOR FORWARD:**
  a relay now carries its held child batches (+63 % on the link per look_horizon.md §5.2, ×W
  gateways per §5.1), so the owed derivation must price `W × C × blob × rate` with the interior
  term included — §5 gives today's numbers; the census derivation stays owed.
- **Where:** `crates/client/src/realm_scene.rs` (`marker_look`, `MARKER_CLASS_SRGB`,
  `POINT_SOURCE_BASE_RADIUS_M`), `crates/physics/src/worldgen.rs` (`StarPhotometrics`,
  `reflected_photometrics`), `docs/design/window_lane.md` §2.8/§2.10;
  `docs/design/look_horizon.md` §5.1-5.2 (the interior term's today-numbers).

### D-WINDOW-4 🟩 RESOLVED (2026-09-02) — the hop is the AUTHORED placement, and nothing is inverted at the shard any more (owner ruling 2026-09-02 R1/R4; wire minor 24)

- **HOW IT RESOLVED.** The hop row used to carry the author's frame pre-inverted into the CHILD's frame, so a rotated cross-cell inversion was refused in the frame core and a galaxy-scale one had no lattice count at all — the galaxy shard refused its own hop every tick and the observer chain never reached the galaxy. The hop now carries the child's placement in the author's frame, at the author's step; the gateway inverts once in the same routine that maps every roster row, at the author's step, where the count exists. The rotated cross-cell child is SERVED (`a_child_window_serves_a_ship_and_still_guards_a_stranger_and_a_rotated_hop`), the world pin flipped from "measured refusals" to "every cross-rung hop in THE world is stated" (`inv_body_at_origin_and_the_rotated_hop_inertness_are_pinned_on_the_world`), and `window_hop_refused` is retired.

<details><summary>The original entry (history)</summary>

#### D-WINDOW-4 (as registered) Rotated cross-cell hop composition is REFUSED, counted, and owed with P10's cell math (window lane, re-registered at Slice D)
- **WHAT is deferred:** folding a hop whose frame is ROTATED across integer lattice cells. Today every
  placement of THE world carries the identity orientation at cell zero, so the composition is exact; a
  rotated cross-cell inversion is refused by the frame core itself and counted
  (`window_rotated_refused` at the gateway, `window_hop_refused` at the author) rather than shipped with
  folded numbers. The inertness is MEASURED, not argued — `inv_body_at_origin_and_the_rotated_hop_inertness_are_pinned_on_the_world`
  fails the day it stops being true.
- **WHEN:** P10, with `convert_tier` and the cell math the galaxy lattice needs.
- **Where:** `crates/core/src/frame.rs`, `crates/connection-plane/src/window.rs` (`count_refusal`),
  `crates/sim/src/stub.rs` (`emit_realm_frames`'s hop inversion).

</details>

### D-WINDOW-5 🟥 The LOD tier seam is DESIGNED and unbuilt — detail tiers are more tags in the same look bag (window lane §2.14, registered at Slice D for P4+)
- **WHAT is deferred:** the ladder's middle rungs. The model already carries its top and bottom: tier 0 is
  the parent-authored marker (~30 B, and since Slice D a real drawn point of light), tier 1 the
  self-authored outline (~100 B). Detail tiers become MORE TAGS IN THE SAME LOOK BAG, authored by the
  realm itself (SL3: a realm chooses its own detail; fractal terrain coarsens free by dropping octaves),
  and SELECTION lives where the knowledge lawfully lives — the gateway knows every observer's angular
  size because it composes the chain, so it forwards only the tier a band needs and asks for a higher one
  with a plain tier number on the window subscription ("serve tier 2" — nothing about who watches, or
  from where). Nothing of that is built. Collision and entities never coarsen (standing law, untouched:
  this lane is render-only).
- **WHEN:** P4+ (with real terrain), or earlier if a realm's drawn footprint outgrows its outline.
- **BETTER PLACED SINCE LOOK-HORIZON SLICE 3 (2026-08-17):** the sealed interior forward is the
  tier seam's NATURAL CARRIER — a deep subject's picture already travels as an opaque
  self-authored bag, so a coarser tier for distant children is more tags in the SAME bag riding
  the SAME forward, selected by a rare and coarse decision at the gateway (where the owner ruled
  it belongs — look_horizon.md §5.6). Nothing pre-built; recorded so the seam lands on the
  carrier instead of beside it.
- **Where:** `docs/design/window_lane.md` §2.14, `crates/wire/src/session_flow.rs` (`WindowScope`,
  `BodyStmt`), `crates/core/src/look.rs` (the bag codec the tags would join);
  `docs/design/look_horizon.md` §5.6 (the interior forward as the carrier).

### D-WINDOW-6 🟥(1) / 🟩(2) Two things Slice D MEASURED and did not assert — (2) DISCHARGED by look-horizon slice 0 (2026-08-17)
- **(2) addendum (look-horizon slices 3+4, 2026-08-17):** the slice-4 wake/teardown churn
  surfaced ONE residual `relay_stamp_missing` class the discharge run never hit: a FIRST-SERVICE
  transient at a freshly (re)opened window, where the re-served reliable relay outruns one lost
  level datagram and a single fold sees a future-only ring (§3.6's "bounded, counted"). It is
  structurally the ONLY reachable class — the C6 head-anchored prune makes an established ring
  unable to go future-only again — so G-RELAY-STAMP now bounds the counter by the windows opened
  (derived churn, run-5 measurement: 1 miss, 4 opens) instead of pinning an accidental zero; the
  storm this row was about (50-87 exact-tick refusals per run, zero opens) stays an order of
  magnitude outside the bound, and the one-beat skew gauge keeps measuring every real lag. The
  discharge STANDS.
- **(1) A RE-HOMED OCCUPANT IS STILL SHOWN ITS PARENT REALM'S OCCUPANTS.** Measured in `two_ships.rs`,
  stable over ten seconds of polling: after ship Y re-homes from the star System down into a Planet, Y's
  client keeps being delivered the avatar of ship X, which is still standing in the System (Y's
  `foreign_space_rows` sits at 2). The SEALING DIRECTION the law protects is intact and IS asserted —
  ship X, standing OUTSIDE the Planet, is shown nobody who stands inside it — and the WINDOW lane
  carries no occupant figure in either direction (every drawn row names a realm, asserted). This is the
  ENTITY/subscription lane, not the window lane: the question is whether a session should retain its
  source subscription after the crossing commits, which is a containment/hand-off matter and outside
  this slice's scope. Printed by the gate every run so a change is visible, never asserted, never
  claimed clean.
- **(2) 🟩 RESOLVED (look_horizon.md slice 0, 2026-08-17 — G-RELAY-STAMP).** Root cause per
  look_horizon.md §3.7: the two-ships zero was a TOPOLOGY fact (the relayed child is a leaf — its
  sealed level carries no rows, and the counter counts rows), and every refusal was ONE arm — the
  placement lookup demanding EXACT tick equality between stamps authored by two different follower
  clocks, while the main chain solves the identical problem with a tolerant prefix. The cure: the
  relayed level store is a per-child RING at the existing span; the fold resolves at the tick the
  chain composed at with an at-or-before fallback (zero positional error structurally — the relay
  ship trigger fingerprints row VALUES, so a stale stamp implies byte-identical rows); the middle
  realm's placement comes from the parent's own level at T, exact. The aggregate counter is SPLIT
  (`relay_descent_refused` / `relay_stamp_missing` / `relay_unrostered` + skew/depth gauges) so the
  ledgered question became answerable, and MEASURED in the non-leaf topology (`rlm_demand_login`
  fly, 60.0 s process flight): `relay_rows_composed=16955`, `stamp_missing=0`,
  `descent_refused=0`, `unrostered=0`, `skew_max=25 ticks` (= exactly one keep-alive beat, the
  bound), `depth_max=2` (the carrier arity). The gate stays in the suite.
- **WHEN:** with the next containment/hand-off pass (1). (2) is done.
- **Where:** `crates/bins/tests/two_ships.rs` (both measurements, printed), `crates/connection-plane/src/window.rs`
  (`descent_at`, the relay fold), `crates/client/src/net.rs` (`foreign_space_rows`).

### D-WINDOW-2 🟥 The DIRECT live-sibling window is the ledgered upgrade path — taken ONLY on a measured G-HANDOVER failure (owner Q2 ruling, 2026-08-16)
- **WHAT is deferred (deliberately, by ruling):** `WindowScope::Observed` — a gateway subscribing DIRECTLY
  to a live realm the observer is NOT inside. The owner ruled Q2 = PARENT RELAY: the parent forwards its
  live children's self-authored statements verbatim (fence + attestation intact; no store, no merge, no
  re-state, no read), so "am I observed from outside" stays UNREPRESENTABLE in every realm and the variant
  NEVER ships (it does not exist on the wire — `crates/wire/src/session_flow.rs::WindowScope` is
  `Occupants | Child` only, and its docs say why).
- **THE ONE TRIGGER that re-opens this — MEASURED 2026-08-16, and it did NOT fire.** The relay costs one
  hop of look latency at exactly the wake moment. G-HANDOVER (Slice D, `just warp-pixels`) measures that
  hop APART inside the symmetric wake-footprint budget — everything after the woken shard is up, so a slow
  relay fails BY NAME instead of hiding inside a boot number. On THE world, flying the ring at the shipped
  speed: **the Q2 relay share was 13 ticks (130 m of travel) against a derived 60 ticks (600 m)**, and the
  whole wake was 14 ticks (140 m) against 61 (610 m). The relay hop does not break the budget, so this row
  STAYS CLOSED. It re-opens only if that measurement fails on a future world, and then only as a fresh SL6
  ask (it tells a realm-adjacent process "some gateway watches me from outside", which is genuinely new
  information — default NO stands until then). Never adopted for convenience, never pre-built.
- **Where:** `docs/design/window_lane.md` §2.3/§2.8/§5-RULINGS; `docs/design/owner_decisions_2026-08-15.md`
  2026-08-16 addendum (Q2); G-HANDOVER lands in Slice D (see [[D-WINDOW-1]]).

### D-LOOK-1 🟥 The candidate build-admission predicate EXISTS; its policy at interim scale is the owner's Q3 ruling — (b)-first-then-measure (look_horizon.md slice 2, LANDED 2026-08-17)
- **WHAT is deferred, and the RULING verbatim (owner, 2026-08-17):** Q3 RULED = (b)-first-then-
  measure. The carrier arity STAYS 2 (`vd_wire::session_flow::LOOK_CARRIER_ARITY`); the
  build-admission predicate (`vd_physics::worldgen::guard_candidate_climb_bounded`) REFUSES at
  interim scale — the design's default (c): a refusal is a measurement, a wrong pixel is not. The
  NEAR-REAL-SCALE world re-solve is the scheduled cure, and its FIRST GATE RUN MUST INCLUDE
  `measure_visibility_climb`; the carrier goes to arity 3 only if that measurement demands it.
- **THE EVIDENCE, produced by a test rather than an argument**
  (`q3_a_twenty_metre_structure_at_interim_scale_is_refused_by_the_admission_fence`): a 20 m
  surface structure planted on a planet of THE world's origin system measures a required climb of
  **3** (visible from outside its star system; slack at the galaxy stop 10 914.699 m) against the
  carrier's 2 — the placement is refused, never the boot (THE world itself passes the same fence:
  max climb 2, worst planet stopping slack 4.000000000000455 m == the containment margin + 4.6e-13
  of float association, the §3.3.2 one-equation identity, pinned in G-CLIMB).
- **BLOCKING for P6/P8** (building lands only where the carrier can carry the picture).
- **Where:** `crates/physics/src/worldgen.rs` (`measure_visibility_climb`,
  `guard_visibility_climb_bounded`, `guard_candidate_climb_bounded`, `CandidateRegion`);
  `crates/wire/src/session_flow.rs` (`LOOK_CARRIER_ARITY`); the boot fences in
  `crates/bins/src/bin/shard.rs` + `crates/bins/src/bin/gateway.rs`;
  `docs/design/look_horizon.md` §3.3/§7 Q3 + RULINGS ADDENDUM.

### D-LOOK-2 🟥 The union coarsening accepted by shared verdicts is UNMEASURED at near-real scale (look_horizon.md, registered at slice 0, 2026-08-17)
- **WHAT is deferred:** the realm-side interest verdict is SHARED by every observer under a scope
  (the owner's 2026-08-16 ruling — one set, slow cadence, nothing per-observer at the gateway), so
  the drawn set for any one observer is the UNION over observers — rows drawn for you because
  somebody else is closer. Look-horizon slice 6 measures that over-draw at INTERIM scale and
  records it; at NEAR-REAL scale it stays unmeasured until the re-solved world exists. Recording
  it here is what keeps "the union is small" a measurement owed, never a silent claim.
- **THE INTERIM-SCALE MEASUREMENT LANDED with look-horizon slice 6 (2026-08-17,
  `vd-sim slice6_the_union_over_draw_is_measured_at_interim_scale`):** two occupants of THE
  world's galaxy at maximally-separated stations (one 300 m from the home star, one 300 m short
  of a ring sibling) — each observer's own out-of-band set is a DISJOINT singleton, the fold's
  published shared verdict is exactly their union (2 children), and each observer over-draws the
  OTHER's child: **6 drawn realm rows per fold per observer** (the child's own look + its
  5-planet interior) that its own position never asked for. At interim scale the union
  over-draw is therefore up to 6 of the fold's ~9 rows for a worst-placed pair — bounded by the
  carrier arity (a child's subtree contributes 1 + its direct-child count, never deeper). The
  NEAR-REAL-scale number stays owed with the world re-solve.
- **WHEN:** ~~slice 6 (interim measurement)~~ landed; the near-real-scale world re-solve (the
  full one) still owed.
- **Where:** `docs/design/look_horizon.md` §5.5/§6 slice 6; `crates/connection-plane/src/window.rs`
  (`scene_bag`'s membership disjunction — the union's mechanical site);
  `crates/sim/src/stub.rs` (the interim measurement).

### D-LOOK-3 🟥 The live child-to-parent interior reach (Ask D) — owed when player building lands (owner Q4 ruling 2026-08-17: DEFERRED, not asked)
- **WHAT is deferred (deliberately, by ruling):** a child telling its parent ONE number — "things
  inside me are visible from this far out" — so player-built interiors wake correctly. For the
  seed-generated world the interior reach is DERIVED AT BOOT from the forest the generator already
  holds, so nothing crosses a realm boundary today and no ask is made (SL6 default NO stands).
  Once player building lands, a realm whose interior GROWS will under-wake until its parent
  re-boots — that is the moment this becomes a real SL6 ask, with a disagreement counter owed
  beside it (the dual-source drift surface the design names in H-FP-6).
- **WHEN:** with player building (P6/P8) — as a fresh SL6 ask, never adopted silently.
- **THE BOOT-DERIVED HALF LANDED with look-horizon slice 4 (2026-08-17):** every region row now
  carries its child's interior band stamped by the generator from the FULL forest before scoping
  (`RealmRegion.interior_band`, built in `vd_physics::worldgen::to_regions` via
  `interior_reach_m` — the §3.4.4 CLAIM settled by
  `the_boot_roster_stamps_each_systems_interior_reach_no_message_crossing`: a galaxy shard's
  roster row for a star system carries `444.104489631` m with no message crossing). What stays
  owed here is ONLY the LIVE update once interiors can grow.
- **Where:** `docs/design/look_horizon.md` §2 Ask D / §7 Q4 + RULINGS ADDENDUM (2026-08-17);
  `crates/core/src/geometry.rs` (`RealmRegion.interior_band`),
  `crates/physics/src/worldgen.rs` (`interior_reach_m`/`interior_band`).

### D-REAL-1 🟩 REALM EXTENT = GRAVITY SOI (planets) — owner-ruled 2026-08-18; LANDED by the taxonomy arc's in-system true-size re-solve
- **THE RULING (owner, 2026-08-18, final):** verify the realm shell the generator emits EQUALS the
  derived gravitational SOI per planet and pin it by name on THE world. (Sun/Moon realms arrive with
  the taxonomy slice.)
- **WHY IT CANNOT LAND ON THE INTERIM IN-SYSTEM WORLD — a measurement, not an argument** (pinned by
  `realm_shell_equals_the_interim_soi_law_and_the_gravity_soi_blocker_is_measured`,
  `crates/physics/src/worldgen.rs`): the interim world has NO planet mass, and its synthetic central
  mass (`13 407 950 220 013.18 kg`, Kepler-3-tuned for a 300 s outer period) sits ~12 orders below one
  Mercury — the real-scale design §3.3.4 mass draw's own bounds INVERT on it
  (`0.01·M★/N_p = 2.7e10 kg < Mercury 3.301e23 kg`), so any "gravitational SOI" computed today would
  be a dressed-up literal (forbidden). Additionally the FULL in-system re-solve cannot land before the
  bound/look split: measured on the addendum's numbers, a real-scale planet SOI (8.6e9 m) climbing with
  its BOUND as its picture measures climb 3 and every boot refuses at the landed arity 2.
- **★LANDED (the taxonomy arc):** the in-system re-solve replaced the interim compressed world —
  real star mass as every planet's `central_mass`, the appended per-planet mass draw (log-uniform
  Mercury..min(Jupiter, disc budget)), `celestial::planet_soi` LIVE at the drawn mass, clamped by
  half the worst-instant inter-orbit gap (MEASURED never to bind: the equality pin prints the
  soi/shell ratio == 1.0 exactly), together with the bound/look split (`RealmRegion.look`). The
  blocker pin was REWRITTEN into the equality pin
  `realm_shell_equals_the_gravitational_soi_at_the_drawn_mass_d_real_1` — every one of THE world's
  27 planets' region shells `==` its gravitational SOI, bit-for-bit. The first gate run's ordered
  climb measurement reported **max climb 1 over all 30 bodies** (32 with the plant), the design's
  prediction measured true.
- **Where:** `crates/physics/src/worldgen.rs` (the pin + `visual_planet_soi_r_m`, the interim law);
  `crates/physics/src/celestial.rs` (`planet_soi`, still dead); the real-scale design §3.3.4 + addendum.

### D-REAL-2 🟩 The outer re-solve (Option C geometry + Q-B 3-D placements) — LANDED with the cell activation (2026-08-18/19)
- **THE RULINGS (owner, 2026-08-18, final):** Q-A = OPTION C — (b)-scale gaps (star gap 0.2376656 ly,
  compression 16.378×, universe radius 2⁵¹ m = 50 % of the position store, headroom 2.0×) WITH the FINE
  cross-cell math LIVE and GENERIC (ONE `Separation` subtraction type in vd-core on every authority-path
  distance; no realm-kind forks; the COARSE tier stays refused — that refusal, `guard_root_representable`,
  is THE NAMED P10 TRIGGER). Q-B = REAL 3-D SEEDED STAR PLACEMENTS — the collinear ring DELETED; seeded
  uniform-sphere directions at the derived placement radius, appended per-system draws 32–33; the
  separation fence re-derived closed-form over the point set (`guard_seeded_systems_disjoint`); star
  count stays the existing config parameter. ★DISCOVERY-PERMANENCE (standing): the generator is
  APPEND-ONLY in its draw stream — the law is written into `crates/physics/src/worldgen.rs`'s module doc
  and pinned per draw.
- **WHAT LANDED:** vd-core `Separation` (cells/residual/tier; guarded `cells_sq` with the Chebyshev
  pre-test + floor-acquire/ceil-release rounding — H-01 cured; `rotated` with the restated reach law
  `rotation_exact_reach_m = cell_edge/ε` — H-11 cured); the integer containment verdict
  (`region_verdict`, Shell/Aabb integer, Obb the declared decimal shape arm); every producer normalized
  (generator centres, placements, both Motion arms, `at_rest` — H-16/H-17/H-19 cured; `advanced_ballistic`
  NOT deleted, H-18); `transfer_frame` integer-first with normalized output; the four outer numbers
  (2⁵¹ / R_uni−outset / R_gal−clearance / χ 16.378) with in-code derivations + bit-exact pins.
- **CLIMB at this slice (measured, recorded):** max climb == 2 (the interim in-system content keeps the
  planet regime at D-LOOK-1's record; the ambient galaxy measures a named levels-2 root artifact —
  no look yet). The addendum's climb-1 prediction is conditioned on the bound/look split + the
  in-system re-solve, LATER slices; the world was NOT adjusted toward the prediction.
- **Where:** `crates/core/src/pose.rs` / `frame.rs` / `geometry.rs`; `crates/physics/src/worldgen.rs`;
  the boot fences in `crates/bins/src/bin/shard.rs` + `gateway.rs`; the real-scale design + addendum
  (scratchpad `real_scale_design.md`, to be moved under `docs/design/` with the arc's closure).

### D-REAL-3 🟩 THE SPEED LAW (S3) — the governed warp mechanism, LANDED (2026-08-18)
- **THE MECHANISM (real-scale design §4 as corrected by addendum §A3; the owner's warp definition —
  "the CONTAINING REALM controls occupant speed" — implemented literally, no mode, no new machinery):
  four closed-form parts in `crates/core/src/flight.rs`, consumed at the sim's ONE integrator seam
  (`crates/sim/src/stub.rs::integrate`) and at the transient re-advance:
  (1) the realm ceiling `max(v_foot, 2·bound/T_TRAVERSE)`; (2) the GEOMETRIC throttle
  (`throttle·v_foot·(v_cap/v_foot)^throttle` — half stick is the geometric middle, the owner clause
  that keeps slow deep-space flight commandable); (3) the proportional ramp `dv/dt = v/τ`
  (state = the pose's own carried velocity — nothing new crosses); (4) the approach governor
  `min over children (child_cap + dist_to_bound/τ)` — you always arrive slowly and can never fly
  through anything. Deceleration is the governor's falling ceiling; stick-release stop stays instant.**
- **THE ONE POLICY NUMBER:** `vd_core::flight::TRAVERSE_S = 180.0 s` — the owner's "journeys in
  MINUTES" bar written once; `vd-physics`' geometry solve now reads the SAME constant (its τ-free
  outset), so shells and law cannot drift. τ is NOT a number: `τ = T_WAKE =
  (2·aoi_cadence + boot_ticks_p99 + WAKE_PIPELINE_TICKS)·tick_dt` (`FlightTuning::derive` — the same
  wake budget the process gates count; DEV posture: 1.10 s at the reactive `boot_ticks_p99 = 0`).
  Every flight number moves WITH a measured boot p99; nothing is pinned to today's τ.
- **★OQ-2 RULED (owner 2026-08-19, verbatim):** *"the realm's ceiling governs everything the realm
  contains, piloted or not"* — the governor binds on the crossing path for EVERY subject kind. Landed
  as the `readvance_transients` speed clamp (debris/projectiles governed by the identical arithmetic;
  HR2 one machinery), closing addendum §A3.5's measured hole (an ungoverned transient crossed a
  system's band in 0.297 of a tick). The one measured human-scale behaviour change: an over-unit
  diagonal stick is normalized to the ceiling (the pre-law √2·v_foot diagonal exceeded the realm's
  stated ceiling — a cure, pinned in `strafe_and_vertical_axes_integrate`).
- **INERTNESS, MEASURED (not argued):** every sub-45 km realm's ceiling clamps to the 500 m/s floor
  (`v_foot·T/2` break-even) and the clamped-scale step is BIT-identical by construction
  (`x/x == 1.0`, `1^t == 1.0`, `1.0·k == k` — pinned by
  `the_speed_law_is_bit_inert_wherever_the_ceiling_clamps`, a bare-vs-planted bit-equality); the
  landed battery stays green (vd-sim 490; the process battery per the S3 run report). A shard with
  no region forest / no authored book states NO ceiling (`None`) — each consumer maps that to its
  own pre-law behaviour (throttle → foot speed; transient clamp → no clamp), byte-identical.
- **LAWFULNESS (SL6 answer = NONE):** the governor reads ONLY the realm's own region bound and the
  placement book the realm itself authored this tick (`Placements.head` + the ONE `child_rows` join —
  H2: no second position path; SL4: a book row cannot say how a child moves). No wire change, no new
  arm, nothing crosses.
- **THE FLIGHT TABLE:** `crates/bins/tests/flight_table.rs` — the three addendum legs simulated
  tick-by-tick with the shipped functions against the closed form (derived band), printed verbatim;
  plus `g_governed_bands_bracket` (every ambient boundary: governed wake dwell ≥ T_WAKE, governed
  band-edge latch ≥ n_entry). The addendum's 117.2/248.7/130.2 s were computed at the PLACEHOLDER
  τ = 2.44 s and are printed beside the shipped-τ measurements.
- **OWED ONWARD (unchanged owners):** the A3.4 two-arm containment-band law rides the in-system
  re-solve slice (S4 folds into S6 — it binds only at target in-system scale); the §4.5 AoI
  `v_cap(parent)·T_lead` spin-up term + arming `boot_ticks_p99` from a measured p99 ride the same
  arc (the reactive posture is measured sufficient at this world's numbers by
  `g_governed_bands_bracket` + the fly-ahead gate). ★S5 OWES ONE PROBE BACK: `warp_pixels`'
  ARRIVAL GPU-paint probe is parked in-gate with citation (measured 2026-08-18: the composed body
  is correct — author SelfLook, 98.16 px — but the client draws relative to NOTHING and the f32
  view collapses at the 2.25e15 m arrival eye; the render_crossing_smoke class, cured by S5's
  camera-relative flatten, §A5.4). `assert_painted(.. "arrival/destination")` returns verbatim
  with S5; the departure/behind captures keep full GPU truth today.
- **Where:** `crates/core/src/flight.rs`; `crates/sim/src/stub.rs` (`flight_tuning`,
  `governed_ceiling_for_frame`/`_in_book`, `integrate`, `readvance_transients`);
  `crates/physics/src/worldgen.rs` (the T alias + the cited `TARGET_*` leg targets);
  `crates/bins/tests/flight_table.rs`; the re-derived gates in `crates/bins/tests/`
  (`warp_pixels.rs`, `node_per_realm_walk.rs`, `rlm_demand_login.rs`).
### D-TAX-1 🟥 `p_surf` — a stripped retainer's surface pressure/density draw (taxonomy arc T1, deferred BY the consumer filter)
- **WHAT is deferred:** the per-planet surface-pressure draw (per-PLANET stream position 2). Surface
  pressure has NO derivation (Earth 1 bar, Venus 92 bar, Mars 6 mbar at comparable escape speeds) and
  currently NO reader: its only consumer is the drag model the physics charter's arc builds. A
  shoreline-surviving SECONDARY atmosphere therefore carries `reference_density_kgm3: None`
  (`vd_physics::taxonomy::Atmosphere`) — stated honestly, never a dressed-up literal.
- **WHEN:** the slice that READS it (the physics charter's drag model) appends the draw at per-planet
  stream position 2 — append-only, so nothing moves.
- **Where:** `crates/physics/src/taxonomy.rs` (`Atmosphere.reference_density_kgm3`,
  `derive_atmosphere`'s secondary arm); celestial_taxonomy_design §3.1/§3.6.

### D-TAX-2 🟥 The SL3 residual: a System states a look equal to its star child's (taxonomy arc T2, ledgered not cured)
- **WHAT:** the System realm's `look` = `star_radius_m(drawn mass)` — the SAME value its Star child's
  own look states (one function, two call sites, mutually exclusive by "a realm containing the eye is
  not a subject", so nothing ever draws twice). A container stating a look is an SL3 impurity; curing
  it needs a parent-authored per-child marker extent DISTINCT from the child's own look, whose
  introduction is itself an SL6 ask (celestial_taxonomy_design §5.2/§12 X-3) — not made here,
  default NO stands.
- **WHEN:** if/when the marker-vs-look split becomes a real SL6 ask (P4+ naming/refinement work).
- **Where:** `crates/physics/src/worldgen.rs` (`generate_system_forest`, the system look
  assignment); celestial_taxonomy_design §5.2.

### D-TAX-3 🟥 The Dyson-standoff half of the Star realm's extent — STOPPED per ruling E's own instruction (owner ruling 2026-08-19; options measured)
- **THE RULING (owner, 2026-08-19, ruling E):** extent = max(dust-sublimation radius, an HONEST
  buildable-shell standoff) — "if no honest derivation closes, STOP that sub-item and report options —
  never hand-pick." The dust radius LANDED; the standoff sub-item STOPPED, with the option set
  MEASURED:
  - (a) the material bound (structures survive where solids do — `DUST_SUBLIMATION_K`, the one
    non-habitability thermal constant the tree holds) applied to built structures DEGENERATES to the
    dust radius exactly — closes, but the max() is vacuous and construction just OUTSIDE the bound
    stays outside the Star realm (the owner's stated intent unmet);
  - (b) the habitable-shell bound (the same equilibrium law at the ruling-B Kopparapu inner flux
    limit, `flux_radius_m(L, S=1.10)`) is UNIMPLEMENTABLE at any seed: the standoff/orbit identity is
    scale-free — `standoff/a_n = √(1/1.10)/(0.4·1.7ⁿ)` = 2.38 / 1.40 / 0.83 for rungs 0/1/2 — so the
    boundary would CONTAIN the two inner planets' whole orbits on every world the generator can
    produce, breaking sibling single-answer containment (the one question the authority model rests
    on) and requiring a hierarchy re-ruling no ruling covers;
  - (c) any intermediate radius is a hand-pick, forbidden by the ruling itself.
- **WHAT SHIPPED:** extent = the dust-sublimation radius alone (ruling E's confirmed physical inner
  meaning), with `guard_star_bound_exceeds_photosphere` boot-wired and the swept minimum printed.
- **WHEN:** re-opens when a cited material/thermal constant for BUILT structures (habitat/equipment
  thermal limits) enters the tree with the megastructure/building arc (P6/P8) — or when the owner
  re-rules the inner-planet hierarchy question (b) poses.
- **Where:** `crates/physics/src/worldgen.rs` (`star_bound_m`); celestial_taxonomy_design §5.3 +
  ruling E.

### D-REAL-4 🟥 The A3.4 two-arm per-child containment band — DEFERRED at the in-system re-solve, basis MEASURED (taxonomy arc, 2026-08-18)
- **WHAT is deferred:** the corrected per-child containment-band law (real-scale addendum §A3.4:
  `max(v_cap(parent)·dt·K_SAFETY, v_cap(child)·dt·N/(1−dt·N/τ), 1024·quantum).min(child_inscribed ·
  BAND_EXTENT_CLAMP)`, inset:outset 1:2) — the speed-law slice's ledger routed it onto the
  in-system re-solve ("S4 folds into S6"). The re-solve LANDED with the static 1.0/2.0 m band kept.
- **WHY (measured, not argued):** (a) the S3 battery already crossed system boundaries AT WARP on
  the full-scale outer world with the static band and latched green (node_per_realm_walk,
  rlm_demand_login 11/11, warp_pixels 3/3 — the crossing latch is verdict persistence, not band
  dwell); (b) the owner's OQ-2 ruling ("the realm's ceiling governs everything the realm contains,
  piloted or not") landed as the universal transient clamp, closing the fly-through hole §A3.5
  sized arm 1 against; (c) the remaining exposure is boundary-loiter re-home thrash at governed
  speeds (damped by `k_dwell`, UNMEASURED — the honest open half); (d) folding A3.4 in moves EVERY
  walk-world band (the 0.5-extent clamp binds on every walk realm), re-baselining landed behaviour
  where geometry did not change; and (e) a τ-bearing band would make `RealmRegion` a function of a
  per-cluster measured latency, forking the one containment answer between processes — the lawful
  form is the τ-free `guard_band_solvable` upper bound (`·BAND_TAU_HEADROOM`), which is how the
  outer geometry already consumed it (§A2.2).
- **★ THE NAMED FALSIFIER HAS FIRED (measured 2026-08-21, the gate-pass arc — `node_per_realm_walk`
  on THE world at the home seed).** Exposure (c) above is no longer UNMEASURED. A governed occupant
  flying leg F out of the ring sibling produced **eighteen crossing sagas across ONE boundary in
  eighteen seconds** — nine `System(1) → sibling` and nine `sibling → System(1)`, strictly
  alternating, at subject fences 4 through 21, **every one of them `attempt = 0`**: eighteen FRESH
  containment decisions, not one decision re-driven. The other three boundaries of the same flight
  (C, D, A) each crossed exactly once, so the thrash is that boundary's and not the chain's. The
  arithmetic the numbers point at is this row's own: the shipped band is the static 1 m inset /
  2 m outset — three metres of hysteresis — while the speed law licenses that sibling's own ceiling
  of ≈ 3.9e9 m/s, i.e. ≈ 7.8e7 m of travel in one 0.02 s tick. Three metres of band against
  seventy-eight thousand kilometres of per-tick travel is no hysteresis at all, which is
  D-WORLD-4b's statement one boundary further out. The gate is PARKED on this row, not weakened:
  `MAX_ENTITY_FENCE` stays 12 and both arms of the thrash guard are untouched, so un-ignoring
  `a_durable_player_flies_the_chain_node_per_realm_without_freezing_or_fence_thrash` is the proof
  that A3.4 landed.
- **WHEN:** NOW ARMED by the measurement above — the trigger this row named has occurred. (It was:
  the boundary-loiter thrash measurement, a governed occupant loitering at a system shell measuring
  re-home saga rate vs `k_dwell` — or the first world where a crossing latch is measured to fail.)
  D-WORLD-4b (the shipped `3.0 ≥ v·dt·K` invariant is false at 500 m/s) stays open under this row.
- **Where:** `crates/physics/src/worldgen/config.rs` (`BandConfig::build`, the static band — the
  file split of 40b5ce2 turned `worldgen.rs` into a module tree); `crates/core/src/geometry.rs`
  (`ContainmentBand`); real_scale_design §4.5/§A3.4; the speed-law ledger (scratchpad
  `speedlaw_state.md` item 10); and the parked gate
  `crates/bins/tests/node_per_realm_walk.rs`, whose `#[ignore]` carries the full measurement.

### D-LOOK-3 🟩 THE TRUE-SCALE CAMERA — LANDED with the S5 render-scale slice (2026-08-19)

**WHAT WAS BROKEN, and what it actually was.** On THE world every drawable body is millions of
kilometres away, and the readback came back EMPTY at rectangles the composer had placed correctly.
The row's original diagnosis named `vd-client-render`'s `STAR_FAR_PLANE = 2 × STAR_SPHERE_RADIUS =
120 000 render-metres`. **That diagnosis was wrong, and the correction is itself a measurement:**
Bevy 0.18 builds a *reverse-Z INFINITE* perspective for a `PerspectiveProjection`
(`bevy_camera::projection` — `Mat4::perspective_infinite_reverse_rh(fov, aspect, near)`; `far` never
reaches the GPU) and its CPU visibility check passes `intersect_far = false`
(`bevy_camera::visibility` — `frustum.intersects_obb(aabb, &world_from_local, true, false)`). The
far plane clipped nothing at all.

**★ THE ROOT CAUSE, MEASURED.** `Transform::looking_at(target, up)` computes `target - translation`
in **f32**. A unit offset is lost the moment a coordinate passes `2^24 = 1.678e7` m, and THE world's
eye stands at `1.8e8`–`2.2e15` m from the composed origin — one f32 ulp there is 16 m to 1.34e8 m.
So a target one metre ahead of the eye **rounded onto the eye itself**, the direction came out ZERO,
and Bevy's `look_to` fell back to `Dir3::NEG_Z` (`bevy_transform::components::transform` —
`-direction.try_into().unwrap_or(Dir3::NEG_Z)`). The camera faced world −Z regardless of where the
pilot was facing, which is exactly why correctly-composed subjects came back as empty rectangles
while the pixel model — computed in f64 — said they were on screen. Pinned by
`vd_client_harness::camera::tests::the_render_frame_survives_the_world_scale_that_kills_an_f32_camera`,
which asserts the f32 direction IS zero at each measured eye magnitude.

**WHAT LANDED (renderer-side only — nothing about composition changed).**
- **THE CAMERA-RELATIVE FLATTEN.** The follow camera sits at the RENDER ORIGIN
  (`Transform.translation == Vec3::ZERO`) carrying only a rotation, and every drawn thing is placed
  at `world − eye`, subtracted in f64 and narrowed to f32 only at the end
  (`vd_client_harness::camera::eye_relative`; `RenderEye` + `place_camera` + `sync_world` +
  `sync_realm_boxes` + `place_reference_scaffold` in `vd-client-render`). The drawn error is now
  relative to the DISTANCE to the thing (~6e-8 of it) instead of to the absolute coordinate.
- **THE f64 CAMERA ROTATION.** `vd_client_harness::camera::look_rotation` builds the orientation
  from the direction the caller already holds, in f64, off the SAME `look_basis` the pixel model
  projects through (pinned against `glam::DMat4::look_at_rh`). `looking_at` is never called on a
  world-scale coordinate again.
- **DERIVED NEAR/FAR.** `vd_client_harness::camera::depth_planes` reads the pair off WHAT IS DRAWN,
  every frame (`derive_camera_planes`): `near` = the world size of ONE PIXEL at the nearest drawn
  surface (never closer than arm's length), so nothing the camera can RESOLVE is clipped; `far` =
  the farthest drawn surface. `STAR_FAR_PLANE` is DELETED, and so are `project_point`'s
  `NEAR_PLANE`/`FAR_PLANE` literals (a symmetric perspective's x/y/w rows contain neither).
- **THE FRAMING RULE.** A diagnostic scene fit unions the SELF-AUTHORED OUTLINES only
  (`framing_bounds`, called by BOTH the renderer's per-frame refit and `live_scene_camera`'s
  reconstruction): a point of light states no outline, so there is nothing OF IT to frame. Measured
  failure it cures: unioning a sibling star's bare centre 2.2485e15 m out pushed the fitted eye past
  6e15 m and collapsed the home shell to 4e-11 px.

**★ THE DEPTH BUDGET, MEASURED (why no logarithmic depth).** Under the reverse-Z infinite
projection the depth code is `near/z`, so ONE f32 code ulp is a CONSTANT **1.1920929e-7 of the
range at every distance**: 21.75 m at 1.8248e8 m, 3.81e4 m at 3.2e11 m, 2.68e8 m at 2.2485e15 m —
each orders under the thinnest thing THE world draws there. Reverse-Z already spans the world's full
dynamic range; a second depth scheme would buy nothing. Gated by name in
`crates/bins/tests/render_scale.rs`.

**THE ACCEPTANCE — the parked gates un-parked.** `render_smoke` (content_fraction 0.0036 → 0.0983
against a 0.05 floor), `render_boxes_smoke`,
`look_pixels::g_true_scale_star_body_and_depth_four_moon_draw_themselves` (the very measurement this
row was opened on — the outer planet at 43.75 px, camera model 43.75 px, **5 770 pixels painted** at
1.816e8 m, where the park had recorded the identical composition with an EMPTY readback rect) and
`warp_pixels`' two gates. `render_crossing_smoke` is the one exception: the S5 camera work landed
for it too and its blow-out is gone, but its own verdict shape predates the bound/look split — it is
re-parked as **D-LOOK-4** below, with the measurement. A NEW gate,
`crates/bins/tests/render_scale.rs` (`just render-scale`), measures the depth and position budgets
directly with no cluster and no GPU, and `crates/bins/tests/acceptance_flight.rs`
(`just acceptance-flight`) flies the whole arc in pixels.

### D-MASS-CAP 🟧 The stellar mass cap is DERIVED FROM THE GALAXY'S RADIUS — it lifts when the galaxy cell lattice (P10) makes the galaxy unbounded

- **WHAT SHIPPED (owner ruling 2026-08-20).** `IMF_MASS_HI_MSUN` was the literal `120.0` and
  `TARGET_SYSTEM_BOUND_MAX_M` was `296_703_425_982.0423` — a SAMPLE of seed 0's own heaviest star
  (an M dwarf of 0.1618 M☉) wearing the name of a reservation. Both are now readings of ONE
  derivation (`DERIVED_MASS_CAP`, `crates/physics/src/worldgen.rs`): the cap is the greatest mass
  whose demand on the galaxy — the §3.2 clearance it is owed plus the two system bounds on the line
  between the origin-anchored home system and any sibling — fits inside `REAL_GALAXY_R_M`, and the
  reservation is the system shell at that cap. **Cap = 16.360034882257757 M☉; reservation =
  749 489 793 576 937.9 m; star look = 4 238 711 986.6327586 m; placement radius =
  1 498 979 587 153 876 m; compression χ = 24.5678.**
- **WHY IT EXISTS AT ALL.** This galaxy is 2.2487974139336678e15 m in radius — 0.2376981 ly. A
  system's shell grows with its star's mass (the orbit ladder is anchored at `0.4·√L` AU), and the
  retired literal named a star whose system solves to 2.43e16 m — **ten times wider than the whole
  galaxy**. No constant reservation could ever have covered every star the world may draw, so the
  rule and the world contradicted each other; a star heavier than the cap was never possible here
  and the world now says so.
- **THE DEFECT THAT FORCED IT (measured on the live cluster, 2026-08-20).** The galaxy shard refused
  to boot on the owner's home seed: *"region System(10487570625701098367) is not geometrically
  inside its parent System(1): it reaches 2248843017364804.8 m … but the parent's usable interior
  ends at 2248797413933667.8 m — refusing to boot"*. No galaxy ⇒ an outbound crossing had nowhere to
  land ⇒ the pilot drifted outside their own system forever, and no real star ever streamed. A
  per-child FIT CLAMP was tried and REVERTED: placing a child where it fits disarms the very fence
  whose refusal is the world's negative control.
- **WHAT IS STILL INTERIM (why 🟧 and not 🟩).** The cap is a fact about a galaxy of THIS radius,
  and that radius is the FINE lattice's storage budget (`guard_root_representable`). It is a
  restriction on the world's star population that physics does not impose — real galaxies host
  O stars — and it costs a third of the star gap (the placement radius fell from
  2 248 490 503 621 178.5 m and χ rose from 16.3784).
- **WHEN IT LIFTS:** the **galaxy cell lattice (P10)** — the same named trigger as
  `guard_root_representable`'s refusal and the compression χ → 1 work
  (`scripts/galaxy_cell_lattice_design.md`). When the galaxy stops being one bounded shell, the
  demand inequality is evaluated against the lattice's reach instead of one shell's radius and the
  cap rises with it. Nothing else changes: the derivation is already written against
  `REAL_GALAXY_R_M`, so P10 replaces one input, not the law.
- **THE GUARANTEE, MEASURED:** `g_nest_sweep_every_swept_seed_generates_a_world_that_nests` sweeps
  284 seeds (a size DERIVED from the IMF — enough seeds to expect a star within one octave of the
  cap) through the identical fence that refused the owner's boot: 13 428 regions judged, heaviest
  star drawn 8.540365549779963 M☉ (52.2 % of the cap), tightest margin 416 779 130.87 m. Its
  refusal arm (`the_nest_sweep_refuses_the_sampled_reservation_that_grounded_the_owner`) restores
  the SAMPLED reservation and reproduces the live cluster's verdict.
- **Where:** `crates/physics/src/worldgen/scale.rs` (`DERIVED_MASS_CAP`, `solve_mass_cap`,
  `galaxy_child_demand_m`, `imf_mass_hi_msun`, `target_system_bound_max_m`) and
  `crates/physics/src/worldgen/guards.rs` (`derived_nest_sweep_seeds`, `guard_swept_seeds_nest`);
  `crates/physics/src/taxonomy.rs` (`imf_tail_fraction`). (The file split of commit 40b5ce2 turned
  `worldgen.rs` into a module tree; every public path still re-exports.)

### D-SKY-1 🟩 THE DECORATIVE STARFIELD IS DELETED — every point of light is a real streamed realm (owner ruling 2026-08-20)

- **THE RULING, VERBATIM:** *"No unreachable star sky spheres please!"*
- **WHAT WAS THERE.** `crates/client-render/src/lib.rs` carried an ambient backdrop sky the CLIENT
  invented: a fixed seed (`STARFIELD_SEED = 0x5644535441525300`), 1 600 uniform + 1 200 Milky-Way-band
  unlit emissive spheres on a 60 km follow-shell (`STAR_SPHERE_RADIUS`), re-centred on the camera
  every frame (`follow_starfield`), with its own spectral palette, angular-size ladder, galactic-plane
  tilt and depth-plane subject. It was visually indistinguishable from the real streamed star markers.
- **WHY IT HAD TO GO.** With the galaxy shard refusing to boot (see D-MASS-CAP) no real star streamed
  at all, and the owner flew at painted, unreachable stars for a whole session with nothing to say the
  sky was a lie. A backdrop that cannot be flown to breaks the seamless mandate at its root: what you
  see must be where you can go.
- **WHAT WAS DELETED:** `STARFIELD_SEED`, `STAR_SPHERE_RADIUS`, `STAR_COUNT_UNIFORM`,
  `STAR_COUNT_BAND`, `STAR_BAND_HALF_ANGLE`, `STAR_ANGULAR_MIN/MAX`, `STAR_SIZE_MIN/MAX`,
  `GALACTIC_NORMAL`, `STAR_TIERS`, `STAR_TIER_CUM`, `struct Star`, `struct StarfieldRoot`,
  `generate_starfield`, `make_star`, `setup_starfield`, `follow_starfield`, the `Update`-schedule
  entry, the `setup_scene` spawn and the backdrop's `derive_camera_planes` subject — about 8.5 KB.
  NO GATE asserted on it (grep: the headless capture path never had one), so nothing was re-based and
  nothing was weakened.
- **WHAT THE SKY IS NOW:** the streamed marker lane alone — `vd_client::realm_scene`'s point-source
  ladder, whose `base_radius_m` is `√L ·`
  `POINT_SOURCE_BASE_RADIUS_M`. **If the sky is empty, that is the truth**: an empty sky now means
  no realm is streaming, which is a defect to fix rather than a picture to paint over.
- **Where:** `crates/client-render/src/lib.rs`; `crates/client/src/realm_scene.rs` (`MarkerLook`).


### D-LOOK-4 🟥 `render_crossing_smoke`'s verdict tests CONTAINMENT against the DRAWN outline

**WHAT is wrong.** The gate's central verdict is *"the dot's pixels lie INSIDE the home box's
projected region while the occupant is inside that realm, and outside it after the crossing"*. That
was ONE statement when a realm's drawn outline WAS its containment boundary. The bound/look split
(real_scale_design §3.0) made them two numbers, and on THE world they differ by 2027×: the home
realm DRAWS its star's photosphere at `7.805661e7` m and is CONTAINED by `1.582262e11` m. An
occupant lawfully inside the realm is therefore nowhere near its drawn disc, and the verdict cannot
be true.

**MEASURED (2026-08-20, with the S5 camera work landed).** Framing that `7.8e7` m photosphere puts
the diagnostic camera `2.35e8` m out. The avatar's own login standoff — the T2 spawn, `1.03e10` m
from the star — is 44× farther and projects BEHIND the star's disc, so the gate fails one assert
earlier than the containment arm: *"the dot itself must be pixel-visible against the shell disc"*,
with the dot rect `x636-648 / y354-366` inside the star's disc `x475-809 / y193-527`.

**WHAT IS OWED.** Restate the containment arm against the realm's BOUND — the number containment
actually uses, read from the booted regions exactly as every other true-scale gate reads it — and
give the gate a framing that can hold both a photosphere and an occupant `1e10` m away (a
subject-named fit, or the pilot camera with a companion dot).

**⚠ EVERY NUMBER ABOVE IS PRE-MASS-CAP AND MUST BE RE-MEASURED (2026-08-21, the gate-pass arc).**
`7.805661e7`, `1.582262e11`, the ratio 2027, `2.35e8`, `1.03e10` and both pixel rects are all
`f(home star mass)`, and the derived mass cap re-drew every star while `HOME_SEED = 2298` became the
default seed the gate boots. The SHAPE of the finding is untouched — a system is CONTAINED at its
solved clearance shell and DRAWN at its star's photosphere, a far smaller sphere about the same
centre, so an occupant lawfully inside the realm is nowhere near its drawn disc — and that is what
the `#[ignore]` citation now states, without digits. Re-take the measurement before scoping the
verdict-shape work from it. The gate's parks, aims and deadlines were re-derived in the same pass
(they were interim-world literals: a 2 km "outside" park inside a 1e11 m system, a 10 m walk-fixture
SOI used for every planet annulus, a flat 60 s deadline for a minutes-long governed leg), so the
test can be un-parked on its merits rather than dying on arithmetic.

**Until then**, the crossing is gated in pixels by `acceptance_flight` (five legs, label-asserted
crossings, handover counts and growth curves), by `look_pixels`' star/moon gate and by
`warp_pixels`. **Where:** `crates/bins/tests/render_crossing_smoke.rs` (the `#[ignore]` citation).


### D-GATEPASS-1 🟩 THE GATE PASS ON THE WORLD AS IT NOW IS — the golden regenerated, every suite re-derived (2026-08-21)

**WHY IT EXISTS.** Two world changes landed in commit `e28c8f2` by the owner's explicit choice while
iterating, with the heavy suites deliberately NOT re-run: `HOME_SEED = 2298` became the default
universe seed for every world-deriving process, and the stellar mass cap plus the galaxy's reserved
clearance became DERIVED (D-MASS-CAP). Commit `40b5ce2` then split four large files into module
trees. The commit message said it plainly: *"The process-tier suites and coverage are NOT re-run —
they still describe the previous world's numbers and are owed a pass."* This row is that pass.

**THE GOLDEN, REGENERATED WITH ITS CAUSES.** `crates/bins/tests/goldens/placement_rows.golden` was
red. It is regenerated, and its header now carries the audit trail rather than a commit message:
- The header gained a derived `# cap:` line (`config.stellar.mass_hi_msun` +
  `target_system_bound_max_m()`), so a future cap move lands in the header's own diff hunk.
- A static `WHY THESE BITS LAST MOVED` block names BOTH world changes, including the one that
  measurably did NOT move these rows. **Cause 1, the mover:** cap `120 → 16.360034882257757` M☉,
  reservation `296703425982.0423 → 749489793576937.9` m, star placement radius
  `2248490503621178.5 → 1498979587153876` m (χ `16.378 → 24.568`). **Cause 2, not a mover:** the
  home-seed default — this vector pins seed 0 explicitly and `boot_world` is a pure `f(seed, config)`.
- MEASURED, not argued: the golden was last captured at `2ddeba0`; `6e442bc` added 192 lines to the
  generator and deleted none (the draw stream untouched); the `SEED` const is byte-unchanged across
  every commit since. The regeneration diff moved **99 of 99 Planet rows, 6 of 12 System rows** (the
  two siblings at all three ticks) **and 0 of 9 Star rows** (a star holds its own system's origin, so
  its row is zero) — a placement-radius change alone would have moved the six sibling rows and
  nothing else, so the 99 planet rows are the re-drawn masses.

**THE TWO CITED CONSTANTS — the decision.** `TARGET_SYSTEM_BOUND_HOME_M` and
`TARGET_PLANET_SOI_OUTER_HOME_M` STAY as seed-0 provenance markers, and their docs now say so
outright. Reasoning, from how they are consumed: the first is pinned EQUAL to seed 0's generated home
shell by `g_star_shell_unmoved_the_stars_clearance_arm_never_binds` — a fixed ruler doing real work;
the second has NO pin and no generated counterpart at all (it is a hand-derived §3.3 figure for a
planet at the maximum mass draw), and its doc now states that plainly instead of implying it is a
fact about the world. What changed is the CONSUMER: `flight_table` used to fly both as leg distances
while its warp leg flew the live world — one table, three legs, two different worlds. MEASURED cost:
the shipped default seed's home shell is `5.089e12` m against the cited `1.582e11` m, a factor of 32,
so the gate flew a home system nobody lives in and departed the warp leg at a ceiling 32× too low.
Both in-system legs are now read off `boot_world(DEV.universe_seed, …)`; the cited targets are
PRINTED beside them, the same discipline the file already applies to the addendum's placeholder-τ
seconds. The reason to cite was discharged when the taxonomy flag day made the generator solve those
shells itself.

**WHAT WAS RE-DERIVED** (never a pasted literal; each replaced a number with a name):
`flight_table` in-system legs + a derived `governed_rows == parented_rows` non-vacuity beside the
census pin · `render_scale`'s close-in eye magnitude (now `outer_look · spin_up_factor`, the law's own
worst case) · `process_parity`'s two movement checks (see D-HOME-SPAWN-1) · `boot_guard`'s planet
(seed 0 → `DEV.universe_seed`) · `rlm_kill9_spawn`'s spawned coord (`planet(2,7,7)` = the retired walk
fixture's `PLANET_A`, which THE world does not contain → `world_roster`, the D-WORLD-8 fix
`rlm_proc_spawn_smoke` already had) · `window_shadow_parity`'s flight budget · `rlm_demand_login`'s
flap margin, ring precondition, exit floor, leg budget, child count and 14 hard-coded realm labels ·
`node_per_realm_walk`'s rendezvous budget · `look_pixels`' star-class datum (a literal `code 6` = M
class, now false — the home star is G — replaced by `marker_datum` over the world's own photometrics)
and its leg budgets · `warp_pixels`' last leg and pre-wake park · `two_ships`' rendezvous budget ·
`render_boxes_smoke`'s expected body kinds · `render_crossing_smoke`'s parks/aims/deadlines and its
walk-fixture SOI · `frame_conversion_e2e`'s demand seed and its `1.0e6` ambient root above a `1.58e11`
child · `crossing_e2e`'s "7.68 m/s fastest planet" message.

**STALE CLAIMS CORRECTED IN PRODUCTION DOCS** (not tests): `HOME_SEED`'s own census paragraph (its
star mass, luminosity, moon count and sibling distance all moved with the cap, hours after the seed
was chosen — the planet itself is bit-unmoved, which is why the choice stands) · `interior_reach_m`
and `RealmRegion::interior_band`'s `444.104489631 m` · `flush_stamp_gap_ticks_max` and the flush
conversion's `7.68 m/s` · `LatticePos`' "star gap" label and two window-lane magnitude labels ·
`camera.rs`'s "THE WORLD'S OWN RANGE PAIR" (a fixed ruler, deliberately, and now said so).

**Where:** the gate list is `justfile`'s `gate` recipe; the classification table rides the arc's
report.


### D-HOME-SPAWN-1 🟩 A STATIC LOGIN LANDED IN TWO DIFFERENT PLACES — the gateway's two attach producers disagreed (found + fixed 2026-08-21)

**THE SYMPTOM.** `process_parity` failed on *"the idle dot never moved"*, and re-running it flipped
between two outcomes with no code change in between.

**THE MEASUREMENT (nine runs).** The login's FIRST delivered own-pose was the realm ORIGIN on six
runs and the T2 spawn standoff (`z = 1.0614053982415949e10` m) on three. A bystander reading was
added to the gate to tell *"the dot is at the origin"* apart from *"one session's fan is stale"*:
both sessions AGREED within every run, so it was the server's placement, not a fan.

**THE ROOT CAUSE.** `gateway/directory.rs`'s STATIC arm sent `AttachSession { spawn: None }` — *"the
shard births at its own origin"* — while the per-tick retry in `gateway/liveness.rs` sent
`session.spawn`, i.e. `Some(standoff)`, under a comment promising *"a re-attach must not be able to
place the avatar anywhere else than the attempt it repeats."* TWO producers of ONE value, disagreeing,
with whichever the shard saw first deciding where the player stood. Twelve lines above the `None`, the
same function's own comment explains why the pose is descended in EVERY mode: *"since the star became
a body at the home centre, an attach with no pose would drop the account inside the Star realm — the
static chain measured exactly that as a frozen unresolvable crossing"* (T2). And the unit suite's
`attach_spawn` helper has claimed since T2 that *"EVERY attach carries the registry's pose (static
included)"* while a second test pinned the first static attach at `None`.

**WHY IT HID.** On the hand-placed walk fixture `default_home_offset_m()` is ZERO, so `None` and
`Some(origin)` are the same place. It only became observable when the world became real.

**THE FIX.** The static arm sends `spawn: Some(spawn)`. Blast radius was exactly one unit test:
`a_static_login_is_byte_identical_with_no_home_phase_and_no_extra_head_read` now asserts
`attach_spawn(&rig)` and its "byte-identical" claim is restated as FLOW-identity (the flow is what it
pins; the attach's bytes did change, deliberately). `vd-connection-plane` 221 pass.

**Where:** `crates/connection-plane/src/gateway/directory.rs` (the static `AttachSession`),
`crates/connection-plane/src/gateway/liveness.rs` (the retry it now matches),
`crates/bins/tests/process_parity.rs` (which additionally waits for the login to SETTLE, so it can no
longer be sampled mid-login, and records a `first_own_pose` so a login that lands somewhere
unexpected names itself).


### D-LOOK-5 🟥 THE PRESENCE FLOOR AND A REALM'S OWN LOOK ARE DISCONTINUOUS — the marker⇒look hand-over POPS at true scale (measured 2026-08-21)

**THE SYMPTOM.** `warp_pixels`' `g_look_growth_a_planets_point_of_light_grows_strictly_on_approach`
— *"a planet's point of light grows monotonically as the camera closes, and it NEVER pops and NEVER
blanks"* — failed on its monotone arm, and the arm is right to fail. The point of light did not
merely stop growing; it collapsed.

**THE MEASUREMENT.** The inner planet's drawn footprint fell from a running maximum of **3.2105 px
to 0.2724 px in one sample step**, at tick 29500, at a range of **1.4188e10 m**. That is an 11.8×
step DOWN, between two consecutive samples, on an approach that was closing the whole time.

**THE MECHANISM, identified arithmetically from the two numbers themselves — no second run needed.**

* **3.2105 px is the PRESENCE FLOOR.** `DOT_MIN_APPARENT_RADIUS_PX` is 3.0 px, and
  `vd_client_harness::camera::marker_world_radius` applies it to *every* parent-authored MARKER:
  `base_radius_m.max(3 px worth of world at this range)`. The remaining 0.7 % is the off-axis
  projection term the gate's own curve comment already records and bounds.
* **0.2724 px is the body's TRUE angular size, unfloored.** `extent / range · (h/2) / tan(fov/2)` at
  the planet's own stated look (`4447120.8351` m) and that range is 0.2724 px to four figures — and
  the gate prints the floor-crossing range this is derived from (`1.288e9` m) on the same line.
  `vd_bins::pixel::subject` has exactly two arms: a `"marker"` row goes through
  `marker_world_radius` (floored); a `"look"` row is drawn at the literal `row.extent_m` (**no
  floor at all**). The renderer takes the same two paths through the same shared expression
  (`crates/client/src/realm_scene.rs` — `marker_base_radius_m` + the one apparent-size floor).
* **So the step IS the hand-over.** The planet's own shard woke mid-leg — the run log shows its
  realm demand and its shard announcing itself in the seconds before the sample — and the picture
  switched authors between two frames. A REALM THAT STARTS DRAWING ITSELF GETS SMALLER, by exactly
  the amount the floor was holding it up by.

**WHY IT IS A DEFECT AND NOT THE LAW BEING HONEST.** At true scale a body's real angular size at its
own wake radius is orders BELOW the floor (`warp_pixels` measures the same step in the other
direction on the star-gap flight and records it in its own comment as *"a STEP DOWN of exactly the
floor: MEASURED, 3.002 px"* — the marker's whole footprint). So this is not an edge case; it is what
every wake looks like from outside. The owner's standing seamless law is *no toggles, no loading, no
teleport — the same picture the whole way*, and an 11.8× jump in a point of light the instant a
server boots behind it is a visible pop with a server event as its cause. It is the last place in
the drawn picture where the machinery is visible from the pilot's seat.

**WHAT IS OWED.** A CONTINUOUS hand-off between the two authors. Either the floor fades out as the
look grows into it (the floored and unfloored radii blended over a band the camera model itself
derives), or a realm's own look takes over the drawing only once its true angular size already
exceeds the floor — the marker keeps the picture until then, which is what the floor is *for*.
Whichever is chosen, the invariant to gate on is the one the parked test already asserts: across the
author flip the drawn footprint may not fall by more than the readback's own quantum.

**WHAT IS PARKED ON IT.** `g_look_growth_a_planets_point_of_light_grows_strictly_on_approach` is
`#[ignore]`d with the full measurement above in its citation. **Not weakened and not deleted** —
every assert stands exactly as written, including the strict-growth arm; un-ignoring it is the proof
the hand-off landed. Its sibling `g_warp_pixels_…` in the same file stays live because its growth
curve deliberately reads only the samples on ONE side of the flip, which is itself a workaround for
this row and should be removed with it.

**Where:** `crates/client-harness/src/camera.rs` (`DOT_MIN_APPARENT_RADIUS_PX`,
`marker_world_radius` — the floor), `crates/client/src/realm_scene.rs` (`marker_base_radius_m` and
the renderer's two paths), `crates/bins/src/pixel.rs` (`subject` — the two arms the gates read),
`crates/bins/tests/warp_pixels.rs` (the `#[ignore]` citation and the one-sided growth curve).


### D-FLIGHT-BUDGET-1 🟥 A LEG'S DEADLINE IS A MULTIPLE OF THE PHYSICS TIME AND CARRIES NO INSTRUMENT TERM — it lost its margin when the star gap shrank (measured 2026-08-21)

**WHAT IS WRONG.** `vd_bins::flight::governed_leg_budget` is `60 s + 3 × leg_time_s(...)` — a fixed
slack plus a multiple of the speed law's own closed form. The `3×` is a MEASUREMENT, and its doc says
so: the walk gate on 2026-08-19 flew a home-system exit whose closed form was ~105 s in ~260 s (2.5×),
and 3× was chosen to cover that with margin. But the multiplier scales only the PHYSICS half of a
leg. A dev-control flight is flown in CHUNKS — a `WalkTo` round trip plus a `DevState` sample per
chunk — and that cost is a property of the instrument, not of the distance. It does not shrink when
the leg does.

**HOW IT SURFACED.** The derived mass cap (D-MASS-CAP) cut the star gap from 2 248 490 503 621 178.5 m
to 1 498 979 587 153 876.0 m. That cut the warp leg's closed form without cutting a single tick of the
instrument's cadence — so the budget, being a pure multiple of the closed form, lost precisely the
margin the 2.5×→3× choice had bought.

**THE MEASUREMENT (`acceptance_flight` leg 4, 2026-08-21, with the per-chunk trace added in the same
pass).** The leg's budget is `governed_leg_budget(gap, galaxy ceiling) + governed_leg_budget(run-in,
system ceiling)` = 663 s. The gap's closed form is 111.99 s ⇒ its term is 396 s; the gap portion
**actually took 490 s, a ratio of 4.4×**. The flight itself is correct: the range to the destination
fell monotonically 1.5016e15 m → 6.6022e10 m over 80 samples with no stall and no recede, the crossing
into the sibling landed at tick 66189, and the destination's drawn radius climbed 0.739 px → 2.037 px
against a 3.000 px stop condition — **68 % of the bar, still closing at ~0.33 px per sample** when the
clock ran out, i.e. roughly 15–40 s short of a 663 s leg. The same run measures the other legs at 2.3×
(leg 2: closed 102.17 s, flown 235.5 s) and 1.3× (leg 3: closed 218.60 s, flown 291.6 s), which is why
they still fit — the shorter the physics time, the worse the instrument term hurts.

**WHAT IS OWED.** A budget law with an INSTRUMENT term: the per-chunk cost (one `WalkTo` round trip
plus one sample) times the number of chunks the leg will take, added to the physics time — rather than
a bare multiple of the physics time. That form is scale-honest: it grows when the cadence grows and
does not silently evaporate when a world change shortens a leg. Until it lands, every chunked flight
gate is one world re-solve away from the same failure, and padding a deadline hides it.

**NOT DONE INSTEAD:** widening the deadline. The number would have gone green and the property would
have stayed broken.

**Where:** `crates/bins/src/flight.rs` (`governed_leg_budget`, and `planet_exit_budget` /
`watched_leg_budget` which are written over it); the parked gate
`crates/bins/tests/acceptance_flight.rs` (its `#[ignore]` carries the full trace); every other
chunked-flight consumer — `node_per_realm_walk`, `rlm_demand_login`, `window_shadow_parity`,
`two_ships`, `look_pixels`, `warp_pixels` — all of which pass today with margin and are the reason
this is 🟥 rather than urgent.

### D-S12-FIXTURE 🟥 THE FRAME FIXTURE BUILDS A REDUCED WORLD, WHICH SL5 FORBIDS — pinned to a measured fraction to unblock S12 (owner-directed, 2026-08-28)

**WHAT IS WRONG.** `vd-tests`' `frame_fixture` does not read THE world. It builds one: two star
systems, two planets per star, a circular in-plane orbit, its own outer radius, and its own orbital
elements written over the generated topology. SL5 says one world — no scale knob, no preset, no
reduced or test-only variant. This is a variant, and until today it was not on this register at all,
which is worse than a registered one.

**HOW IT SURFACED.** The galaxy shape (S12) moved every placement. The fixture's anti-drift guard
went red because its story distance was a knob the SHELL obeyed exactly and the SHAPE does not: the
knob now names the galaxy's rim, and the seed names where inside it the star sits.

**THE INTERIM, AND WHY IT IS THE SMALL ONE.** `NEIGHBOUR_FRACTION_OF_RIM = 2.711_063_548_671_444e-1`
— MEASURED, identical to the last bit at both story magnitudes (1.2031e4 m and 1e13 m), because every
term of the placement scales with the rim while the angles read `r / r_max` and do not. The guard now
compares the generator against that literal, so it still fails on a shape change, a draw-order change
or a re-seed. Nothing circular, and no production code was added to serve a test — the alternative
considered was a new seam letting a caller author a child's centre, which was refused as production
growth for a fixture's benefit.

**THE PROPER SOLUTION.** Delete the fixture, in two halves that are each stronger than what they
replace:
- the WORKED ARITHMETIC becomes a pure unit test on the conversion itself, with hand-written poses,
  frames and placements. No world, so no variant, and the sum stays checkable by eye.
- the CONVERSION END TO END runs on THE world, at real size. The world already supplies both regimes
  the fixture invented: a star inside its galaxy (~1e17 m, where a millimetre snaps to a whole step)
  and a person on a planet (metres, where it survives).

**WHY IT IS WORTH DOING (owner, 2026-08-28).** *"We will need to rewrite tests in case if we will
re-seed, but then reseed itself can't be accidental."* A test that reads a fixture can survive a
re-seed silently. A test that reads THE world cannot. Making the cost of a re-seed visible is the
point, not a side effect.

**SIZE, MEASURED.** 772 fixture lines, ~25 tests inside it, and exactly ONE outside consumer — the
round-trip test in `frame_conversion_e2e`. The one mention in `vd-sim` is a comment.

**WHEN.** Its own slice, after S12 lands. Owner-directed sequencing: unblock now, land S12, then plan
the rewrite.
