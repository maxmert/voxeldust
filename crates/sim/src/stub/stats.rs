//! THE HONESTY COUNTERS: one field per condition this shard tolerates but must never suffer
//! silently.
//!
//! Owns: the counter surface itself, and nothing else. A drop, a refusal, a stale fence, a
//! degraded conversion — each has a name here, so "it worked" and "it silently did not" are
//! distinguishable from outside the process.
//!
//! Does NOT own: any of the code that increments it. Every lane bumps its own counters from its own
//! module, which is why a new tolerated condition costs one line here and one line there, and why
//! no lane can hide a discard behind a bare `continue`.

use bevy_ecs::prelude::Resource;
use vd_core::collections::DetHashMap;
use vd_core::entity_kind::EntityKind;

/// Counters for conditions that are tolerated but must never be silent.
#[derive(Resource, Debug, Default, PartialEq, Eq)]
pub struct StubStats {
    /// Attach requests that arrived before the realm lease was granted; the
    /// gateway retries attach until it sees `SessionAttached` (at-least-once).
    pub attaches_deferred: u64,
    /// Inter-shard frames that failed to decode at ingress (a malformed/garbage
    /// gateway→shard or directory-reply payload). Tolerated — the frame is dropped,
    /// never mis-applied — but counted so a decode regression is observable rather
    /// than log-only (ROB-E2E-1; mirrors the gateway's `undecodable`). 0 in any
    /// healthy run.
    pub undecodable: u64,
    /// `OpenInputSlot` arrived before this shard holds its realm lease — BUFFERED + counted
    /// (Stage B2; was a permanent drop, the reproduced fence-7 strand of rehome_one_mechanism
    /// §4v). The gateway emits `OpenInputSlot` exactly once (at `apply_commit`) and the saga is
    /// forward-only past `Committed`, so the buffered slot is drained at the lease affirm
    /// (`drain_pending_input_slots`) — the adopt completes late instead of never. The post-marker
    /// INPUT buffer the gateway take-drained in the same window can still shed (that half stays
    /// DEFERRED D-8/D-29). 0 in a run where the dest realm lease precedes the commit.
    pub input_slots_deferred: u64,
    /// Buffered `OpenInputSlot`s DRAINED at the lease affirm (Stage B2) — each one is an adopt
    /// that the pre-B2 posture would have stranded permanently. Pairs with
    /// `input_slots_deferred`: a deferred count with no matching drain is a lease that never
    /// landed (the realm never came up), which the RLM/SL7 work owns.
    pub input_slots_drained: u64,
    /// `OpenInputSlot` carrying a fence BELOW the dot's session fence — a replay or a
    /// partitioned old gateway; dropped + counted, input never re-armed (the day-one
    /// stale-gateway-drop rule, `wire::session_flow`). 0 in a healthy run.
    pub input_slots_stale: u64,
    /// `OpenInputSlot` carrying a NON-Entity transfer subject (1c.8): the dest only ADOPTS an
    /// `Entity` transfer, so a Realm/Session/Ship subject is a counted no-op (no extraction
    /// panic, no adopt). 0 in a healthy Entity-transfer run.
    pub input_slots_malformed: u64,
    /// The `resume_from_seq` of the most recent HONORED `OpenInputSlot` — the watermark this
    /// shard last opened a transfer-dest input slot at. Observability latch (NOT a counter): it
    /// makes the gateway-EMITTED resume value visible to the cross-cut conservation gate (the
    /// emitted value must equal the client's own CUT_MARKER seq, threaded through the real saga —
    /// D-28), and is the operational answer to "what seq did this shard resume a handed-off
    /// session at". `None` until the first honored slot.
    pub last_input_slot_resume: Option<u64>,
    /// 1d.1 entity-state crossings APPLIED (`StubCrossing` pose stored on the adopted dot, first
    /// delivery of a `(transfer, step)`). The headline 1d.1 counter.
    pub crossings_applied: u64,
    /// Crossings BUFFERED because the adopt grant had not flipped yet — drained + applied at the
    /// flip (the crossing, emitted at CAS, races ahead of the dest's adopt under the 1c.8
    /// promote-before-demote model). 0 only if every crossing arrived after adopt.
    pub crossings_buffered: u64,
    /// Crossings DROPPED for a fence below the dot's recorded authority fence (fence rule 1 — a
    /// stale leftover from a superseded transfer). 0 in a healthy single-transfer run.
    pub crossings_stale: u64,
    /// `Transfer` envelopes whose payload kind 1d.1 does not consume (`InitialSpawn` /
    /// `TransientBatch`) — a counted no-op, never a panic. 0 in a 1d.1 crossing run.
    pub crossings_unhandled: u64,
    /// FAULT (observability): subject evaluations whose pose frame named a realm this shard does not host,
    /// so the DERIVED containment prior resolved to nothing and the stored-only (pre-fix) prior was used.
    /// Nonzero means a rebind upstream safe-degraded and those subjects can still flap at a boundary.
    pub containment_prior_unhosted: u64,
    /// `Transfer` envelopes carrying a `universe_epoch` that does NOT match this shard's current
    /// clock epoch — REFUSED at ingress (never applied, never buffered, never acked). The
    /// transfer_protocol §3.3 fail-safe: "a leg whose epoch_id mismatches the current epoch is
    /// discarded, not resumed — no entity placed at a stale celestial position" (the R6/R7
    /// epoch-reset class the rebuild exists to kill). The source stamps `clock.epoch` on every
    /// envelope; pre-this-gate the field was carried-but-unread at the receiver. Mirrors the
    /// follower clock's `epoch_mismatches` counted-ignore. 0 in any single-epoch run (P3 runs one
    /// persisted orchestrator epoch that never resets mid-run); becomes load-bearing once a clean
    /// re-genesis or a delayed redelivery (the owed redelivering transport) can carry a prior-epoch
    /// envelope across a restart. `schema_version` version-floor validation rides the TLV-blob
    /// handshake owed at D-31 (a distinct, deliberately-deferred concern).
    pub crossings_epoch_mismatch: u64,
    /// SOURCE self-fence redeliveries that found the dot ALREADY demoted to a Ghost — a counted
    /// no-op (the idempotency guard's taken arm). 1d.4b retains the source as a Ghost instead of
    /// `dots.remove`, so a second foreign-owner reply must NOT re-demote; this proves the guard.
    pub self_fence_skipped: u64,
    /// SOURCE saga-pushed `Demote` (1d.5b.1) received for a NON-Entity subject (Realm/Session/Ship)
    /// — there is no local Entity dot to demote, so the flip is skipped (the `DemoteAck` is still
    /// sent). 0 in a healthy Entity-transfer run (the stub's transfer subject is always an Entity).
    pub saga_demote_no_entity: u64,
    /// DEST saga-pushed `Promote` (1d.5b.3b) received and APPLIED for the first time — the headline
    /// ordered-promote counter. 1d.5b.3b RELOCATED the real `Ghost→Owned` flip here (out of
    /// `apply_crossing`), so this handler does the flip + announces the dest sub + registers the
    /// ghost feed + acks `PromoteAck`.
    pub promotes_confirmed: u64,
    /// DEST saga-pushed `Promote` REDELIVERIES (already-journaled `(transfer, PROMOTE_STEP)`) — a
    /// counted re-ack-only no-op (at-least-once delivery). 0 in a healthy single-delivery run.
    pub promotes_redelivered: u64,
    /// DEST `Promote` (1d.5b.3b) that found NO dot for the subject entity — a counted no-op (still
    /// acks). 0 in a healthy run (the crossing creates the dest dot before the Promote round-trip).
    pub promote_no_dot: u64,
    /// DEST `Promote` (1d.5b.3b) whose crossing pose has NOT yet landed (`STUB_CROSSING_STEP` not
    /// journaled): the flip is SKIPPED (pose-before-promote — never a poseless origin frame); the
    /// saga `Promoting`-timeout re-emits. 0 in the happy path (the crossing precedes the Promote).
    pub promote_before_crossing: u64,
    /// DEST `Promote` that arrived while this shard does NOT hold its realm (a realm self-fence raced
    /// the Promote) — DROPPED as a counted no-op (degrade, never panic), the saga re-drives. 0 in P2
    /// (no realm-revoke producer); reachable only at P8/P10 realm mobility (the re-drive is owed).
    pub promote_without_realm: u64,
    /// D-37 forward re-home ADOPTS: a `ReHome` landed the subject Owned at the target — a fresh dot
    /// built from `ReHomeState`, or (see `re_home_flipped`) a held same-entity dot flipped in place.
    /// `> 0` proves CELL-2 recovery actually adopted the re-homed entity.
    pub re_home_adopted: u64,
    /// The SUBSET of `re_home_adopted` where the target ALREADY held a dot for the subject entity —
    /// the retained Ghost its own outward demote left behind (the forward re-home resolves its
    /// target from the flushed pose's frame realm, which on an UPWARD hand-off is the SOURCE realm,
    /// so a dead dest re-homes the entity straight back here). The held dot is FLIPPED in place;
    /// minting a second dot under the synthetic key was the batch-review MAJOR (the orphan Ghost's
    /// hold ran to TTL and the expiry fan broadcast `EntityRemoved` for an entity this shard OWNS).
    pub re_home_flipped: u64,
    /// `ReHome` REDELIVERIES (already-journaled `(transfer, RE_HOME_STEP)`) — a counted re-ack-only no-op
    /// (at-least-once). 0 in a healthy single-delivery run.
    pub re_home_redelivered: u64,
    /// `ReHome` for a non-Entity subject (no `transfer_subject_entity`) — a counted no-op (still acks).
    /// 0 in P3 (CELL-2 re-homes an Entity); reachable when Realm/Ship re-home lands (Slice 4).
    pub re_home_no_entity: u64,
    /// `ReHome` that arrived while this target does NOT hold its realm — DROPPED as a counted no-op
    /// (degrade, never panic; the re-home target normally holds its realm, committed by the orchestrator).
    pub re_home_without_realm: u64,
    /// `ReHome` adopt carrying a `universe_epoch` that does NOT match this shard's current clock epoch —
    /// REFUSED at ingress (never journaled/adopted/acked). The transfer_protocol §3.3 fail-safe applied
    /// UNIFORMLY to the SECOND pose-placing ingress (the first is the crossing →
    /// [`StubStats::crossings_epoch_mismatch`]); mirrors it exactly. 0 in any single-epoch run (P3 runs one
    /// persisted orchestrator epoch that never resets mid-run); load-bearing once a clean re-genesis or a
    /// delayed redelivery (the owed redelivering transport) can carry a prior-epoch `ReHomeCmd` across a
    /// restart — exactly the player/ship re-home + P7 checkpoint-reload paths that must never place an
    /// entity at a stale celestial position.
    pub re_home_epoch_mismatch: u64,
    /// SOURCE `GhostFlow::Despawn` (1d.5b.3c band-exit) received and TORN DOWN — the retained ghost
    /// `Dot` removed (the ghost lifecycle ENDS; the return-crossing target is gone). Its clients were
    /// told to evict at hold closure already (slice F), so nothing visible changes here. The headline
    /// band-exit teardown counter. (The per-Delta feed counters died with the pose feed, slice F.)
    pub ghost_despawns: u64,
    /// SOURCE `GhostFlow::Despawn` that tore down NOTHING (1d.5b.3c) — no mirror entry AND no retained
    /// ghost dot to remove: an at-least-once REDELIVERY after teardown, or a stale Despawn for an entity
    /// the source has since RE-OWNED (the `Owned` dot is structurally refused — only a retained Ghost is
    /// torn down). A counted idempotent no-op, never a panic. 0 in a healthy single-delivery run.
    pub ghost_despawn_no_host: u64,
    /// DEST band-exit (1d.5b.3c): the owned entity left the overlap band, so the dest emitted
    /// `GhostFlow::Despawn` to the ghost-host and DEREGISTERED the feed (`GhostColliderRegistration`).
    /// The headline band-exit DETECTION counter (the dest half of the source's `ghost_despawns`).
    pub ghost_band_exits: u64,
    /// DEST feed pass skipped a registration whose entity is NOT currently owned here (no dot, or a
    /// non-`simulates()` dot) — a counted no-op (only an Owned dot's live pose is fed). 0 steady-state.
    pub ghost_feed_skipped: u64,
    /// SOURCE: `TransientBatch` envelopes EMITTED (D-7) — one per (dest realm, tick) batch regardless
    /// of item count (G-TIER). The headline transient-egress counter.
    pub transients_emitted: u64,
    /// DEST: transient items ADOPTED as `Arriving` on the FIRST delivery of a `TransientBatch` (the
    /// uncounted mid-flight tier). The headline transient-adopt counter.
    pub transients_adopted: u64,
    /// DEST: transient items REFUSED at adopt — the arriving pose could not be placed in the dest
    /// realm's frame ([`place_arriving_pose`], the SAME receiver-side conversion the durable
    /// crossing and the D-37 re-home run; HR2 — one machinery, policy fan-out, audit :105/:374/:384).
    /// The item is NOT adopted: a counted, logged loss within the Transient class budget (the
    /// durable twin refuses the whole crossing instead, so the source keeps authority). 0 in every
    /// lawful hand-off — the source flushes through `flush_pose_for_dest`, so the pose arrives in
    /// the dest realm's own frame or a direct child's.
    pub transient_arrivals_unplaceable: u64,
    /// A hosted DIRECT-child region whose realm the seed-lineage coordinate cannot name — an
    /// entity-backed `Ship` realm (`level_of` = `None`; `RealmKindTag` has six seed-keyed tags and
    /// no Ship arm until P8, DEFERRED D-SHIP-1). The two remaining coord-needing lanes — the
    /// AoI/demand fold and a `Child`-scope window's hop row — each EXCLUDE such a region: a
    /// graceful typed exclusion, counted per lane pass, NEVER a panic (audit :713: the old
    /// `expect` aborted the whole shard). (It was FOUR lanes until window lane Slice C2 deleted
    /// the cascade targeting, the interior fan and the scene reflect.) Always 0 through P3 — no producer plants a Ship region; the P8 ship-realm
    /// work gives ships a lineage coordinate and retires this counter.
    pub ship_child_regions_excluded: u64,
    /// DEST: `TransientBatch` REDELIVERIES (already-journaled `(transfer, TRANSIENT_BATCH_STEP)`) — a
    /// counted re-ack-only no-op (at-least-once). 0 in a healthy single-delivery run.
    pub transients_adopt_redelivered: u64,
    /// DEST: `Arriving→Held` PROMOTIONS on a `TransientDrop` (D-7b: the transient twin of Ghost→Owned,
    /// reachable only after the source released). The headline transient-promote counter.
    pub transients_promoted: u64,
    /// SOURCE: `Held{outbound}` items RELEASED to `Departing` on a `TransientRelease` (D-7b: the source
    /// goes uncounted BEFORE the dest promotes — the clean hand-off, NOT a loss). The source half of
    /// the structural drop-before-promote.
    pub transients_handed_off: u64,
    /// `TransientDrop` (promote) REDELIVERIES (already-journaled `(transfer, TRANSIENT_DROP_STEP)`) — a
    /// counted idempotent no-op (at-least-once). 0 in a healthy single-delivery run.
    pub transient_drop_noop: u64,
    /// `TransientRelease` + `ReleaseComplete` REDELIVERIES (already-journaled
    /// `(transfer, TRANSIENT_RELEASE_STEP)`) — a counted idempotent no-op (at-least-once). 0 healthy.
    pub transient_release_noop: u64,
    /// GROSS transients DROPPED on a realm self-fence — EVERY tier (handover + settled-resident). Ops
    /// visibility only; 0 on the happy path. NOT the loss-budget gate (that reads
    /// `transients_lost_in_handover` — a 1000-resident burst eviction would dwarf a tiny budget here).
    pub transients_dropped: u64,
    /// D-3 Slice 5 — PROACTIVE self-fences: the holder dropped its realm authority because the lease
    /// went un-confirmed past `self_fence_grace_ticks` (a detected partition from the orchestrator). Ops
    /// visibility / partition signal; `0` on the happy path and inert (`self_fence_grace_ticks == 0`).
    pub realm_self_fenced_lapsed: u64,
    /// HANDOVER-attributable LOSS per kind (D-7b.3): on a realm self-fence, ONLY transients in a
    /// handover status (`is_in_handover()` — emitted/Departing/Crossing source items + Arriving dest
    /// items) are an in-flight transfer LOSS bucketed here; a settled `Held{outbound:None}` dropped on
    /// eviction is a resident-eviction event, OUT of transfer-budget scope. The `verify_transient_
    /// loss_budget` gate compares this per-kind count to the kind's `LossBudget` — so a mixed-kind
    /// burst can never spuriously trip a single-kind budget. `DetHashMap` (fixed-seed → deterministic).
    pub transients_lost_in_handover: DetHashMap<EntityKind, u64>,
    /// D-7d — transients dropped by `on_transient_abandon` (the dead-DEST resolution: the dest died
    /// mid-handoff, so the source's retained copy is dropped as an ACCOUNTED loss). 0 on the happy path
    /// and on a dead-SOURCE resolution (which self-promotes, no loss); `> 0` is the DEST-kill cell's
    /// anti-vacuity proof. Each increment ALSO feeds `transients_lost_in_handover` (the SAME budget the
    /// realm self-fence feeds, DRY) — so `verify_transient_loss_budget` sees a deterministic, named loss.
    pub transients_departure_cancelled: u64,
    /// R-6d3c — transients DISCARDED at the DEST by `on_transient_discard` (the source died in
    /// `BatchHandoff::AwaitAdopt` PRE-adopt, so a late-replayed `Arriving` copy is removed as an ACCOUNTED
    /// loss + the adopt is poisoned). 0 on every happy path and on a RESTART recovery (which adopts
    /// normally); `> 0` is the never-restart cell's anti-vacuity proof. Each increment WITH A DECODABLE
    /// kind also feeds `transients_lost_in_handover`; a corrupt-tag item is removed + counted HERE but NOT
    /// attributed (the `from_tag` Err arm — HR2, never decode-to-default).
    pub transients_discarded_source_crash: u64,
    /// CA-1 S3/S4 — DEST: `BatchAdopted` acks RE-DRIVEN by `redrive_pending_adoptions` (one per DISTINCT
    /// `Arriving` batch per tick). The LIVENESS half of the over-discard guarantee: it keeps re-presenting
    /// the adopt evidence to the orchestrator's `dest_adopted` latch so a transiently-lost single ack never
    /// strands the batch. `> 0` whenever a batch sits `Arriving` for more than the initial adopt tick.
    pub batch_adopts_redriven: u64,
    /// CA-1 S3 — SOURCE: `ReSolicitBatch` liveness probes RECEIVED (a counted NO-OP — the probe's signal is
    /// its SEND outcome at the orchestrator, not this handler; a LIVE source simply acknowledges receipt by
    /// existing). Ops visibility that the AwaitAdopt probe reached a live source.
    pub re_solicits_received: u64,
    /// Slice 3e — DURABLE geometric crossings REQUESTED: `evaluate_realm_boundaries` committed a durable
    /// entity across an `Authority` boundary and emitted ONE `CrossingRequest` (latched in
    /// `RequestInFlight`). The headline durable-trigger counter. `0` in prod through P3 (empty registry).
    pub crossings_requested: u64,
    /// Slice 3e — durable crossings SUPPRESSED because the subject was already latched in-flight (a second
    /// `should_commit` edge before the saga terminal cleared the latch). Proves exactly-one-request. `0`
    /// steady-state.
    pub crossings_suppressed_in_flight: u64,
    /// Slice 3f-D4 — a STRANDED durable latch RE-DRIVEN by `redrive_stranded_crossings`: a
    /// delivered-but-unresolved dest (`head(Realm(to))` absent — e.g. an UNHOSTED realm on a static
    /// cluster, D-WORLD-2) left the `RequestInFlight` latch standing with no rising edge to re-emit
    /// it, so the per-tick latch-scan re-emitted the SAME `CrossingRequest` once `local_tick -
    /// last_commit_tick >= request_ttl_ticks`. ARMED in every launcher/fixture since the D-WORLD-2
    /// cure (`0` only in disarmed unit rigs); `0` on every happy path (the saga terminal clears the
    /// latch before the ttl elapses).
    pub crossings_redriven: u64,
    /// D-WORLD-2 cure — stranded latches ABORTED LOCALLY on re-drive EXHAUSTION: the latch stood
    /// through `crossing_redrive_budget` re-drives with no saga terminal (the dest never resolved),
    /// so the source took the pre-CAS abort itself ([`abort_crossing_latch`]): latch cleared, attempt
    /// bumped, containment cooldown armed — the entity stays simulated at the source and its NEXT
    /// crossing fires. `0` on every happy path and while `request_ttl_ticks == 0`; a non-zero value
    /// is a realm the cluster genuinely could not resolve for a whole `(budget+1)·ttl` window.
    pub crossings_exhausted: u64,
    /// Slice 3e (robustness, goal-audit L4) — a Durable-TAGGED subject reached the durable crossing arm from
    /// the held-transient loop (a kind/loop mismatch — e.g. a mis-tagged batch item) and so carried no
    /// session: DEGRADED (counted, emitted nothing) instead of panicking. Must be `0` in a well-formed mesh;
    /// a non-zero value flags a mis-tagged transient batch from a peer.
    pub crossing_durable_no_session: u64,
    /// Slice 3e — TRANSIENT geometric crossings REQUESTED: a transient committed across an `Authority`
    /// boundary and emitted a `TransientCrossingRequest` (the batched-grant path). `0` in prod through P3.
    pub transient_crossings_requested: u64,
    /// Slice 3e — `RequestInFlight` latches CLEARED by a saga terminal at the SOURCE: `on_saga_demote` on a
    /// durable COMMIT, or a `CrossingAborted` demux on a pre-CAS abort. Proves the POSITIVE (never-TTL)
    /// clear. `0` until a triggered crossing resolves.
    pub crossing_latches_cleared: u64,
    /// Slice 3e — SOURCE `TransientCrossingGrant`s APPLIED: a granted transient flipped `Held → Crossing`
    /// so `emit_transient_batch` ships it. `0` in prod through P3 (no transient crossings triggered).
    pub transient_grants_applied: u64,
    /// Slice 3e — `TransientCrossingGrant` counted NO-OPS: an unknown transient OR one already flipped (a
    /// redelivery after the flip). `0` in a healthy single-delivery run.
    pub transient_grant_noop: u64,
    /// Slice 3e — `TransientCrossingGrant` for a NON-Entity subject (Realm/Session/Ship) — a counted no-op.
    /// `0` in a healthy run (a transient crossing subject is always an Entity).
    pub transient_grant_no_entity: u64,
    /// Slice 3e — `CrossingAborted` for a NON-Entity subject — a counted no-op. `0` in a healthy run.
    pub crossing_abort_no_entity: u64,
    /// Slice 3e — `CrossingAborted` whose transfer id did NOT match the subject's current latch (a stale
    /// abort for a superseded / re-latched crossing) — a counted no-op, the latch is preserved. `0` healthy.
    pub crossing_abort_stale: u64,
    /// Step 5 slice A — `ChildLive` bits RECEIVED and upserted (fresh by `(fence, at)`).
    pub child_live_received: u64,
    /// Step 5 slice A — `ChildLive` bits DROPPED as mis-routed (the sender's parent link does not
    /// lower to this realm). Never normal.
    pub child_live_misrouted: u64,
    /// Step 5 slice A — `ChildLive` bits DROPPED as stale by `(fence, at)` (a reorder or a deposed
    /// incarnation). Expected under datagram reorder; a flood means a zombie.
    pub child_live_stale: u64,
    /// Lane cure (findings 0/43, up half) — `ChildLive` bits DROPPED because the sender does not match
    /// the directory's record for that child in [`ChildRealmNodes`] (fail closed; each refusal arms a
    /// head re-read). Expected transiently across a child's spin-up or re-home (up to one cadence);
    /// a flood means a zombie incarnation still beating from its old node.
    pub child_live_unattested: u64,
    /// Lane cure (finding 37) — demands REFUSED by [`push_demand`]'s structural gate: a demand this
    /// shard emits names its OWN realm (the Empty self-report) or a DIRECT CHILD (the AoI union) —
    /// SL7's two allowed shapes — and anything else is refused here, counted, one place (HR3).
    /// Non-zero exactly while an OUTWARD crossing's keep-alive window stands (the dest is the parent,
    /// which stays alive through arm B + `ancestor_close` instead — measured by the return-crossing
    /// gate).
    pub demand_refused_not_own_or_child: u64,
    /// A pose ARRIVED here in a frame this shard cannot measure — neither its own nor one of its DIRECT
    /// CHILDREN — so the crossing / re-home was REFUSED and the entity was NOT placed
    /// (`place_arriving_pose`). The commonest cause is a mis-routed hand-off: a sibling handing an
    /// occupant straight to a sibling, which no parent authored and which therefore has no meaning here.
    /// It used to be silent — the old ingress swallowed the error and kept the number under a new label,
    /// so the entity landed at whatever that number happened to mean in the wrong space. `0` healthy.
    pub arrivals_unplaceable: u64,
    /// The region roster says the hand-off destination is one of MY DIRECT CHILDREN, but the ephemeris
    /// could not express the outgoing pose in it — a genuine internal contradiction (typically a dot
    /// carrying a stale frame that is neither mine nor a child's). The hand-off is REFUSED, so this shard
    /// keeps authority and the saga aborts. Distinct from the ordinary upward case, which is not an error
    /// at all and never reaches the conversion. `0` healthy.
    pub flush_unplaceable_child: u64,
    /// A LOGIN offered a spawn pose measured in a frame that is not this realm's, so the avatar was born at
    /// the origin instead of somewhere arbitrary. Two ways to earn this, both real: a login routed to a
    /// shard that is not the realm the account's stored position is in (a static cluster attaches every
    /// login to one fixed shard, so this is expected there and harmless), or a stored pose left over from
    /// when poses were universe-absolute. It used to be neither refused nor counted — the pose was
    /// relabelled into this shard's frame and the player appeared wherever that number happened to fall.
    pub spawn_poses_refused: u64,
    /// Hand-offs whose occupant pose named a realm this shard does not author. Never normal: it means a
    /// stale frame label reached the crossing path, where it would otherwise have chosen the frame the
    /// arithmetic happens in.
    pub flush_anchor_not_own: u64,
    /// Hand-offs REFUSED at the flush because the re-read pose is no longer inside the DESTINATION
    /// region (the entry decision went stale between the scan and the flush drain — a fast occupant
    /// left before the pose shipped). The saga aborts pre-commit and this shard keeps authority; a
    /// pass-through costs one aborted saga instead of a committed mislanding (Stage B1, §4v cure 1).
    pub flush_stale_entry: u64,
    /// Hand-offs REFUSED at the flush because this shard's OWN region would still hold the re-read
    /// pose (the departure decision went stale — the occupant came back inside before the pose
    /// shipped). Same abort-and-keep shape as `flush_stale_entry`, for the flap's mirrored half.
    pub flush_stale_exit: u64,
    /// Placement-book selections that MISSED: a consumer asked the ledger for an instant outside the
    /// retained window (or an anchor with no books). Every miss is a degrade the lane handles loudly —
    /// a skipped subject, a refused arrival, a dropped relay batch — never a silently substituted
    /// nearby instant. 0 in any healthy run (the placement arc's S2 gate).
    pub placement_book_miss: u64,
    /// Cross-shard clock skew CLAMPED to the head: a message-carried instant fell outside this shard's
    /// retained window — AHEAD (two followers observe the same ClockSync broadcasts at different
    /// arrival phases) on any lane, or EITHER direction on the ARRIVING hand-off lane (a retained,
    /// retried envelope's stamp never changes, so its age grows with every redelivery). The receiving
    /// lane read its world of NOW — the head book, the pose re-stamped to the head's instant, the same
    /// one-instant rule the flush follows and the same "an occupant rides its realm" answer the
    /// up-observation ride measurement pinned. MEASURED at process tier (2026-08-14): the demand
    /// round-trip's fly-out wedged on exactly this before the clamp (refused at `head−345` ticks, once
    /// per redelivery, forever). Counted, never silent; the widest skew rides the per-direction
    /// gauges below. A stale RELAY datagram still drops loudly (`placement_book_miss`).
    pub placement_skew_clamped: u64,
    /// The widest FORWARD (ahead-of-the-head) skew clamped, in ticks — the `span_ahead` measurement
    /// the placement arc owed: how far a sender's ClockSync arrival phase has led this receiver's.
    /// One direction per gauge (batch review: one two-directional magnitude destroyed exactly this
    /// measurement — the arrival lane's redelivery staleness dominated it).
    pub placement_skew_ahead_max_ticks: u64,
    /// The widest BACKWARD (behind-the-window) staleness clamped, in ticks — the ARRIVING hand-off
    /// lane only (a retained, retried envelope's stamp never changes, so its age grows per
    /// redelivery; the relay lane drops stale datagrams loudly instead of clamping them).
    pub placement_skew_behind_max_ticks: u64,
    /// The WIDEST latch→flush staleness observed on this shard, in ticks: at each pose flush, the gap
    /// between this shard's clock and the flushed pose's stamp. A latched dot's stamp is FROZEN at the
    /// latch (`readvance_dots` deliberately skips it), so this gap is exactly how far the world's moving
    /// placements have swept under the departure/entry decisions the flush re-validates — the error is
    /// `gap × tick_dt × v(fastest mover)` (7.68 m/s for THE world's inner planet, post the S4 re-solve) against a 1.0 m
    /// containment inset. The S0 measurement of the frozen-flush-instant defect; the crossing e2e pins
    /// it to 0 once the flush re-reads the world at the CURRENT placement book (the S2 fix).
    pub flush_stamp_gap_ticks_max: u64,
    /// A row in this shard's OWN client-edge emit whose pose label could not be restated into this
    /// shard's own frame — a FOREIGN-LABELLED pose inside a locally-emitted row, i.e. the §4u ghost-feed
    /// corruption made countable. (The entity RELAY lane died in Step 5 slice E; this counter outlives
    /// it because it measures the emit path, not the relay.) EXPECTED non-zero only while the ghost
    /// feed's foreign write survives; slice F deletes that writer and pins this to `0` forever.
    pub entity_rows_foreign_labelled: u64,
    /// A remove message SUPPRESSED because this shard held no realm lease at the permanent stop
    /// (a self-fenced shard tearing down a ghost, a lease race at a detach) — the emit discipline
    /// says an unowned shard is silent, so the removal is withheld, but NEVER silently: each
    /// suppression is a bystander who may keep a frozen figure until the realm's sub teardown or
    /// take-over re-stream reaches them. `0` healthy; non-zero next to a self-fence is the
    /// partition surfacing.
    pub entity_removals_suppressed_no_lease: u64,
    /// THE WINDOW LANE (Slice A, docs/design/window_lane.md §2.3/§2.9) — a window REGISTERED
    /// (a fresh open, or a live id re-used under a new scope). THROUGHPUT.
    pub windows_opened: u64,
    /// THE WINDOW LANE (Slice A, docs/design/window_lane.md §2.3/§2.9) — GAUGE, not a counter:
    /// how many windows this shard currently holds open (refreshed each tick by the emitter,
    /// post-TTL-prune). The diagnosis surface's "windows_open"; `0` at zero subscribers is the
    /// structural teardown truth (zero sessions ⇒ zero windows).
    pub windows_open: u64,
    /// EGRESS — `WindowFrame` messages shipped (one per tick per open window; the per-realm
    /// egress meter the owner-approved design names in §4.5 Topic 1).
    pub window_frames_sent: u64,
    /// EGRESS — authored child rows shipped inside `WindowFrame`s (the row-volume meter beside
    /// the message meter: rows/tick is O(direct children), bounded by the lattice — §2.13).
    pub window_frame_rows_sent: u64,
    /// EGRESS — `WindowBody` statements shipped (send-on-change + on-open, NEVER per-tick; a
    /// steady non-zero rate here means a look/marker source is flapping).
    pub window_bodies_sent: u64,
    /// EGRESS — `WindowMembership` verdicts shipped (diffs of the parent's SL7 fold, on the
    /// fold's own change rhythm).
    pub window_memberships_sent: u64,
    /// A duplicate `WindowOpen` re-asserting a live window (the derived keep-alive) — refreshed
    /// its TTL, idempotent, THROUGHPUT (this beats once per keep-alive cadence per subscriber).
    pub window_reasserted: u64,
    /// A window dropped by the DERIVED keep-alive TTL (2 beats + 1 — owner law 3(a)): its
    /// subscriber stopped re-asserting (a dead gateway). The crash backstop working as designed;
    /// steady growth WITHOUT a gateway death means the subscriber's cadence stopped clearing
    /// the TTL.
    pub window_ttl_expired: u64,
    /// A `WindowClose` naming a window this shard does not hold — a counted no-op (the polite
    /// fast path racing the TTL backstop, or a malformed/unknown id). BENIGN in small numbers.
    pub window_close_unknown: u64,
    /// A `Child(c)` window whose `c` is not one of this shard's DIRECT children (counted per
    /// emission pass, like the ship exclusion): the subscriber's routing state and this roster
    /// disagree — nothing is emitted for it, never guessed at. FAULT if it persists.
    pub window_child_unrostered: u64,
    /// A `Child(c)` hop-row inversion REFUSED by the frame core (rotated frame across integer
    /// cells — `transfer_frame`'s own refusal, owed P10 cell math): the frame is dropped +
    /// counted, never shipped with folded-precision numbers. Pinned unreachable on THE world
    /// today (the inversion-inertness measurement); non-zero means the world grew a spinning
    /// realm a cell-block out before P10 landed.
    pub window_hop_refused: u64,
    /// THE Q2 RELAY (Slice C1, mesh minor 17; owner-approved 2026-08-16 —
    /// docs/design/owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS) — EGRESS UP:
    /// sealed statement batches this realm shipped one hop up (send-on-change on the look/marker
    /// set + parent resolve, re-asserted on the AoI cadence — a restarted parent holds relays
    /// only in RAM).
    pub window_relays_sent: u64,
    /// Q2 relay — batches RECEIVED from a live direct child and HELD, unopened (forward-or-drop:
    /// the parent never decodes the seal).
    pub window_relays_received: u64,
    /// Q2 relay — EGRESS OUT: held batches FORWARDED verbatim to a window subscriber
    /// (send-on-change per (window, child); a fresh window is served everything held once).
    pub window_relays_forwarded: u64,
    /// Q2 relay — batches DROPPED as mis-routed (the sender's parent link does not lower to this
    /// realm). Never normal.
    pub window_relay_misrouted: u64,
    /// Q2 relay — batches DROPPED as unattested (sender ≠ the directory's record for that child in
    /// [`ChildRealmNodes`]; fail closed). Expected transiently across a child's spin-up/re-home.
    pub window_relay_unattested: u64,
    /// Q2 relay — batches DROPPED as stale by the child's own fence (a deposed incarnation still
    /// shipping). Never normal past a re-home window.
    pub window_relay_stale: u64,
    /// Look horizon slice 4 — `RealmInterest` bytes SHIPPED to direct children (both values; the
    /// rising edge + every AoI beat while a child's interior band holds, one explicit `0` on the
    /// falling edge).
    pub realm_interest_sent: u64,
    /// Look horizon slice 4 — interest bytes lawfully ADMITTED (either value; the down-proxy
    /// reads the held entry, never this counter).
    pub realm_interest_received: u64,
    /// Look horizon slice 4 — an interest byte whose routing coord does not lower to THIS realm:
    /// dropped + counted (fail-closed, the mirror of every up-lane's mis-route guard). 0 in a
    /// healthy run.
    pub realm_interest_misrouted: u64,
    /// Look horizon slice 4 — an interest byte from a sender that is not the resolved PARENT
    /// head: dropped fail-closed + counted, and a lazy parent head re-read armed (the re-home
    /// backstop, mirroring `retain_child_live`). 0 in a healthy run.
    pub realm_interest_unattested: u64,
    /// Look horizon slice 4 — a deposed parent incarnation's byte (fence/tick below the held
    /// entry): refused + counted. 0 outside a re-home window.
    pub realm_interest_stale: u64,
    /// Look horizon slice 4 — a byte that is neither `0` nor `1` ("nothing else is lawful" —
    /// §2 ASK B): refused + counted. 0 always; nonzero is a hostile or corrupted sender.
    pub realm_interest_unlawful: u64,
}
