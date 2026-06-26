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
  lease-TTL reap, and TTL enforcement is itself unbuilt ([[D-3]]), so on a real cluster the entry is immortal. This is
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
- **Source:** Slice-2a design `wf_9f22c70d` (finding #3); boundary-documented, not silently wedged.

### D-37 🟥 Permanent participant-kill recovery: the forward re-home / abort-on-unreachable producer
- **Missing:** when a transfer PARTICIPANT is permanently KILLED mid-flight (not crash+resurrect — the fabric's
  at-least-once recovers that), the saga has no way to recover to a live, consistent owner. The Slice-2a Timeout
  producer re-drives the lost step toward the DEAD node forever (it cannot tell dead from slow — D-3) and the saga
  PARKS. Empirically pinned by the P3 Slice-1 crash matrix (`tests/tests/p3_crash_matrix.rs`):
  - **kill SOURCE in Demoting** (post-commit): the directory committed to the dest; the Demote never acks; the dest
    stays Ghost. `ParkedHalfOpen{authority_at: DEST}`.
  - **kill DEST pre-freeze** (the design's "aborts to the live source" claim was REFUTED by the matrix): the dest is
    NOT in the pre-freeze ack path (Prepared/CutConfirmed/SourceFrozen come from the gateway + source), so killing it
    does NOT trigger a pre-freeze abort — the saga sails to the orchestrator's LOCAL commit-CAS (commits to the dest
    regardless of liveness) and PARKS in Promoting (the dead dest cannot ack PromoteAck). `ParkedHalfOpen{DEST}`.
    **Consequence: there is NO clean permanent-kill-to-LIVE-source cell** — killing the source/gateway makes THEM
    dead (a `DeadOwnerOrphan`); the abort-to-live-source end state is reached only by a NON-kill abort (a spatial
    rejection / a transient-fault pre-freeze timeout that leaves the source alive — P3 Slice 1b).
  - **kill SOURCE pre-freeze** (Freezing): the freeze timeout aborts (`abort_with_thaw`, gateway-acked) + tombstones,
    but the directory is left at the now-DEAD source (fence-neutral abort — `abort_clear` no longer bumps).
    `DeadOwnerOrphan{SOURCE}` (a `HeldNowhere`: the dead owner holds nothing, so the fence value is moot here).
- **Where:** `crates/node/src/saga_runtime.rs` (the `scan_deadlines` producer re-drives toward the dead node; no
  abort-on-unreachable, no re-home). The harness makes these HONEST today: the dead-node-aware oracle
  (`FaultFabric::is_dead`, `Topology::inspect_live`/`dead_nodes`, `oracle::verify_authority_unique_excluding`)
  EXCLUDES the dead node's corpse claim, so a park/orphan surfaces as the true `HeldNowhere`/`RealmHeldNowhere`
  orphan rather than a false-passing held@dead-node (or a false-RED legit park). It is OBSERVABLE, never silently wedged.
- **When / proper:** a forward RE-HOME producer (re-drive the entity onto a LIVE shard / abort-to-a-live-owner),
  gated on **D-3** (lease-lapse liveness — the real dead-vs-slow discriminator) + **D-6** (the durable saga WAL to
  re-home from). Same root as the D-2 loss-of-autonomous-recovery + the killed-source-mid-Demoting case D-2 already
  named as P3/Slice-2 scope. The orchestrator-crash row of the crash matrix is separately gated on D-6.
- **Source:** P3 Slice-1 design `wf_540e3497` + the empirical crash matrix (which refuted the design's clean-abort
  claim for kill-DEST-pre-freeze).

### D-38 🟧 HR4's literal G-IDENTICAL gate (ONE fixture, ≥2 shard kinds) is unbuilt — only the capability-DAG FOUNDATION landed
- **LANDED (the foundation):** `crates/sim/src/capability.rs` has the validated `ShardProfile` capability DAG (private
  fields, `ShardProfile::build()` the only ctor), the 5 canonical kinds as DATA, the coherence test
  `canonical_profiles_are_coherent`, and the negative gate (incoherent profile fails loud, `capability.rs:227`). This is
  HR4's STRUCTURAL half: features are written against capability traits and a shard type is zero new code.
- **MISSING (the behavior half):** the literal **G-IDENTICAL** CI gate CLAUDE.md/PLAN.md mandate — a NAMED test
  (`assert_feature_anywhere`) that runs ONE identical feature fixture on a Spherical AND a Cartesian profile (with one
  fixture forcing a `reanchor()`), or it doesn't land. `grep -rn assert_feature_anywhere crates/ tests/` returns
  nothing. So HR4 is the ONE hard rule whose enforcing gate is owed-at-first-feature rather than already-green.
- **WHY this is an INTERIM, not a defect (the deferral is sound):** there is ZERO feature code to diverge yet — the only
  sim is the stub shard (points in empty space), and real capability-bearing features begin at **P4** (terrain) / **P6**
  (block edits, the first true G-IDENTICAL fixture — "block edits on planet AND ship"). A G-IDENTICAL harness now would
  assert sameness over an empty feature set: scaffolding with nothing to protect (the smallest-correct discipline forbids
  it). The `FrameSpace` seam it tests (`SphericalSpace`/`CartesianSpace` + `reanchor()`/`AnchorGen`) itself lands at P4/P5.
- **WHAT the audit caught (the honesty hole this entry closes):** CLAUDE.md HR4 + PLAN.md schedule G-IDENTICAL as a
  PERMANENT gate, yet it was absent from BOTH the code AND this ledger — invisible to the "a phase isn't done until its
  DEFERRED entries flip green" gate. The DEFECT would be the UNLEDGERED absence (the same class [[D-31]] closed for HR2's
  `TransferableKind` and [[D-32]] closed for the directory partition seam), not the deferral. This entry is the fix.
- **When / proper:** the `assert_feature_anywhere` harness + the first G-IDENTICAL fixture land with the first
  capability-bearing feature — **P6** (block edits on a planet `SphericalSpace` AND a ship `CartesianSpace`, one fixture
  forcing a reanchor), per the PLAN.md P6 delta. Every transition phase from P4 onward then adds its own fixture.
- **Pin (exists-to-be-flipped):** `crates/sim/src/capability.rs` module doc names the missing gate + this entry. Flips
  🟩 when `assert_feature_anywhere` runs one identical fixture green on ≥2 shard kinds in `tests/`.
- **Dependency:** `FrameSpace` (`SphericalSpace`/`CartesianSpace`, P4/P5); the first capability-bearing feature (P6).
- **Source:** the b0304ca holistic audit `wf_2e97cbe7` (the most substantive of four noted minors — the one hard rule
  whose literal gate is owed-at-first-feature rather than already-green).

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
  `core/geometry.rs`) — and emits `GhostFlow::Despawn` (RELIABLE carrier; a lost Despawn would leak the collider) +
  DEREGISTERS the feed (`GhostColliderRegistration`). The source `on_ghost_flow` Despawn arm now TEARS DOWN: removes
  the `SourceGhostMirror` entry AND the retained ghost DOT (the source stops self-emitting + being a collider) —
  IDEMPOTENT + counted (`ghost_despawns` / `ghost_despawn_no_host` / dest `ghost_band_exits`). Strictly POST-release
  (the destroy edge is many per-tick steps out, so the entity walks well past the demote→promote→release handoff before
  exiting), so the source ghost is removed only once the dest is the SOLE render source — the capstone
  `p2_dod_band_exit_tears_down_the_source_ghost_seamlessly` proves NO vanish across it (`max_absent_run==0`, on the
  REAL `DeliveredView`). The .3b feed-forever (dest feed-pass + source self-emit every tick per committed transfer) is
  CLOSED. Routing stays DIRECT shard↔shard mesh.
  **⚠️ INTERIM (.3c) — the motion-scaled stub band:** `OverlapBand::for_motion` sizes the band off per-tick travel
  (the stub has no realm-center/SOI geometry) and the anchor is the CROSSING pose (a local-boundary approximation).
  Production realm bands use `for_planet_soi`/`for_system_soi` anchored at the realm center (P4/P5 spatial geometry) —
  band-exit then has no anchor-at-crossing artifact. The teardown machinery is geometry-agnostic (only the band +
  anchor source change), so the swap is additive.
  **⚠️ OWED (.3c) — the orchestrator-side teardown gate:** the "refuse a Despawn that races a live saga" guard the spec
  named on the directory `in_transfer` field (DEAD — cleared at commit-CAS) CANNOT live on a `vd-sim` shard (it cannot
  see the orchestrator live-saga set; the dependency rule forbids sim→node — full-audit `wf_3fee0260` skeptics 4/4).
  The shard-LOCAL stand-in is STRUCTURAL: the source tears down ONLY a retained `Ghost` dot (a RE-OWNED `Owned` dot is
  refused — `remove_retained_ghost`), the destroy-edge sizing keeps band-exit post-release, and the orchestrator
  one-saga-per-key lock prevents a concurrent same-key saga. The proper orchestrator-side gate + ghost-lifecycle crash
  recovery are owed at **Slice-2** / **D-6**. **⚠️ saga `(Demoting, DestDelivered)` latch arm is
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
  the adopt-ordering race, not the permanent design. (iii) The crossing is NOT gated on its dest ack: the saga
  reaches `Done` whether or not the crossing landed (the source self-fence is independent of the crossing). The
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

### D-3 🟧 Lease lifecycle: slices 0–2 LANDED (config + heartbeat producer + flap fault); the reaper + CSCALE-1 tracker + self-fence owed (design wf_24c1ecc5, 6 slices)
- **✅ LANDED (slices 0–2 of 6, all gate-green 100% Tier-A):**
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
    stand-in for an io-prod blip. EXISTS but no scenario consumes it yet (Slice 3 wires the CSCALE-1 cells).
- **Still owed (slices 3–5):** **Slice 3 — the CSCALE-1 tracker:** rework `dead_participants` → an
  evidence-gated `LivenessTracker` (N-consecutive within `unreachable_window_ticks` + clear-on-ack), add a
  per-saga `dead_observed_since`, and move the DESTRUCTIVE dest-abandon behind the LARGE `abort_deadline_ticks`
  (the cure). **Slice 4 — the orchestrator expiry REAPER + CAP freeze** (lapsed-AND-confirmed-dead, inside the
  D-6 group-commit barrier; reaps `Session` keys, MARKS Realm/Entity/Ship for D-37). **Slice 5 —
  self-fence-before-grant** (the proactive owner timer on `local_tick`). Each gate-green; commit-ask each.
- **⚠️ CSCALE-1 still OPEN until Slice 3** (the tracker is not yet reworked): `dead_participants` is still
  insert-only / never-cleared, and the dest-abandon still rides the cheap `redrive_deadline_ticks`. Bounded +
  gate-invisible exactly as below (the flap fault exists but no cell uses it yet, so the blip path stays
  unexercised). Slice 3 closes it.
- **⚠️ CSCALE-1 (whole-codebase audit `wf_2de9063f`, HIGH — latent, fix before real-node chaos):** the
  D-7d dead-resolution uses a kill-only `Inbound::NodeUnreachable` as its dead-vs-slow STAND-IN: in io-prod
  a SINGLE recoverable write blip (a 20s idle-reap / VXLAN drop of a LIVE peer — `io-prod/.../mesh.rs`
  emits `NodeUnreachable`, the writer auto-redials next frame) inserts the peer into
  `SagaRuntimeRes.dead_participants` (`saga_runtime.rs` `drive_sagas`) which is **NEVER cleared** (no
  `remove`/`retain` anywhere — confirmed), and a due `BatchHandoff` saga then resolves to `DestUnreachable`
  → `EmitTransientAbandon` + Tombstone = irreversible accounted loss of an in-flight transient batch toward
  a HEALTHY dest. `deadline_for` puts `BatchHandoff` on the CHEAP `redrive_deadline_ticks`, so a blip + a
  few ticks is enough. NOT critical TODAY: the harness emits `NodeUnreachable` only via a PERMANENT kill
  (the blip→live-dest path is gate-invisible), there is NO real-network fault injection over QUIC, and the
  transient burst path runs only in-process — so it CANNOT fire on a live cluster until P3 real-node chaos
  (this entry's WHEN). Bounded to transient kinds (debris/projectiles) within loss-budget; durables
  unaffected.
- **Where:** `crates/sim/src/directory.rs` (`renew`/`lease_expires` written, never consumed);
  `crates/node/src/orchestrator.rs` (the only `LeaseRenew` consumer is the receive arm);
  `crates/node/src/saga_runtime.rs` (`dead_participants` insert in `drive_sagas`, never cleared; the
  `DestUnreachable`/`SourceUnreachable` resolution gate in `scan_deadlines`; `deadline_for`'s BatchHandoff →
  `redrive_deadline_ticks`). Pinned by the `OwnerRecord.lease_expires` doc in
  `crates/wire/src/seams/directory.rs` (TTL-ENFORCEMENT-UNIMPLEMENTED) + the kill-only-today doc on
  `dead_participants`.
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

### D-6 🟧 Durable saga WAL: S0–S5 LANDED (persist+recover ENGINE + e2e orchestrator kill-9 cells); the redb backend + 3 io-prod preconditions owed
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
- **⚠️ The prod orchestrator BINARY is NON-DURABLE** (`crates/bins/src/bin/orchestrator.rs` injects a fresh
  in-memory `MemStore`): a restart resets the clock + loses in-flight transfers. LOUD `tracing::warn` at boot
  (audit D6-ROB-1) — now also names the transport-redelivery dependency below — until the redb backend swaps in.
  No production deployment until then.
- **⚠️ Three io-prod / real-deploy PRECONDITIONS (audit `wf_7cc86404`, all HIGH, none break P2/P3 correctness —
  proven against the in-process MemStore + harness at-least-once model the S0–S5 cells run on):**
  1. **`AwaitAdopt` recovery depends on transport REDELIVERY the production `MeshTransport` does not provide.**
     `BatchHandoff::AwaitAdopt` (saga.rs:563-571,631-634) has NO orchestrator re-drive egress (the dest adopts off
     the source's envelope; the orchestrator only awaits `BatchAdopted`). The harness `crash` preserves the
     at-least-once ledger so the lost ack redelivers; the io-prod `MeshTransport` (mesh.rs:357-395) is at-most-once
     (`NodeUnreachable`-and-drop). **Owed cure (preferred): give `AwaitAdopt` a real re-solicit egress on Timeout**
     (orchestrator re-prompts the source to re-emit the `TransientBatch`, or the dest to re-ack) so saga recovery is
     SELF-SUFFICIENT and stops depending on an un-promised transport property — a designed slice (NEW saga action +
     shard handler, full architecture first per the no-hack rule). Alternative: a sender-side durable outbox /
     retry-until-acked transport layer (identity_persistence.md:122). **WHEN: before the redb backend / any real
     rolling deploy.** Until then, producer-less-phase orchestrator-crash recovery is proven ONLY vs the FaultFabric.
  2. **Directory RECONCILE is a per-TICK O(directory) delete-all + put-all + fsync, even on fully idle ticks**
     (saga_runtime.rs:1187-1203). Correctness-safe; a write-amplification cliff at MMO directory scale (per-Session/
     Entity/Realm/Ship rows). **Owed: io-prod replaces the full reconcile with INCREMENTAL per-mutation deletes
     (threaded from `serve_directory` + `commit_cas`, as the :1185 comment names) + a dirty-guard that skips the
     directory write (and the fsync if nothing else staged) on an unmutated tick + a SCALE soak guard (large stable
     directory, zero transfers). WHEN: the redb backend swap.**
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
  **FRAME-REBINDING (audit finding 5):** 1d.1 stores the crossed pose's `FrameRef` VERBATIM — it stays the SOURCE
  realm's frame (e.g. `SystemSpace{system_seed: 7}`) on a dot owned by the DEST realm. Inert in 1d.1 (render_ready
  false → nothing reads the frame; and the gate USES this as the crossing discriminator). But render (1d.3) /
  ghost / client compositing MUST re-express the pose into the dest realm's frame (or carry an explicit
  cross-frame transform) before it is rendered or fed to physics — owed with `render_ready` + the real frame seam
  (`FrameSpace`, P4/P5). Pinned RED-to-flip: the gate asserts seed-7 today and flips to the dest frame when
  rebinding lands.
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
  `vdctl` driving; (iii) **D-15** (`WalkTo`/`LookAt` wired). The render crate is real and `G-RENDER-SMOKE` is live,
  but single-shard/single-client.
- **Where:** `crates/bins/src/bin/vd-devcluster.rs` (dual-shard spawn), `crates/bins/tests/render_smoke.rs` (the
  paired scenario sibling), `crates/bins/src/bin/client.rs` (D-15), `justfile` (a new `render-cross` recipe);
  the seamless in-process gate in `tests/tests/p2_transfer_gates.rs` (reusing `vd_harness::oracle` render oracles).
- **When / proper:** the seamless in-process gate at the **D-2 band/ghost P2 slice**; the wgpu paired visual at the
  **P2 client-track slice** that lands D-15. Sibling to D-16/D-17 (the existing G-RENDER-SMOKE GPU/terrain limits).
- **Source:** Slice 1d.3 design + the 1d.3 implementation finding (the unmasked window is a vanish, not an overlap).

---

## BLOCKING SPIKES (must run before the phase they gate)

### D-18 🟥 SPIKE-3a — datagram delivery under a 4 MB BULK burst on the k3d overlay
- **Blocks:** **P3.** The hand-rolled latency-gate harness (`percentile_unstable`, established by SPIKE-2a)
  should be extracted to a shared `vd-harness` helper when SPIKE-3a adds the second hard latency gate.
- **Source:** `PLAN.md` SPIKE list.

### D-19 🟥 SPIKE-6a (rapier snapshot/restore + cross-binary determinism) blocks P5; SPIKE-10a (dual-frame ship-interior physics) blocks P8.
- **Source:** `PLAN.md` SPIKE list.
