//! THE HONESTY COUNTERS: everything the gateway tolerates but rejects is counted, never silent.
//!
//! Owns: the counter surface. A refused sender, a stale fence, a dropped datagram, a parked
//! statement that never drained — each has a name here, so a router that is quietly discarding is
//! distinguishable from a router with nothing to do.
//!
//! Does NOT own: any of the code that increments it. Each lane bumps its own from its own module.

use bevy_ecs::prelude::Resource;

/// Honesty counters: everything tolerated-but-rejected is counted, never silent.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GatewayStats {
    pub logins_rejected: u64,
    pub version_rejected: u64,
    pub sessions_refused_capacity: u64,
    pub session_mints_refused: u64,
    pub resumes_refused: u64,
    pub inputs_deduped: u64,
    pub inputs_unroutable: u64,
    pub inputs_malformed: u64,
    pub stale_frames_dropped: u64,
    /// A shard's `EntityRemoved` refused for ONE subscriber because it carried a fence stale
    /// against that session's per-shard accepted fence (rule 5 — a demoted old owner's removal
    /// must not evict what the live owner still streams). The same discipline as
    /// `stale_frames_dropped`, counted apart because a wrongly-dropped REMOVAL strands a frozen
    /// figure (the D-4(a) defect) while a wrongly-dropped frame costs one tick of motion.
    pub stale_removals_dropped: u64,
    pub undecodable: u64,
    /// A frame from a node this router knows as NEITHER a shard NOR a client, on a class a stranger may
    /// not send. It is DISCARDED, and until now it was discarded as `undecodable` — the same bucket as a
    /// client sending malformed bytes.
    ///
    /// THE TWO ARE NOT THE SAME FACT, and lumping them hid a whole running shard. Measured on the demand
    /// gate: 17,365 frames in one run, against a counter whose own comment says it must stay zero for a
    /// healthy login, and no gate noticed. A shard that is up, simulating a player, and not on the roster
    /// had its entire output thrown away in silence; the player it owned kept being drawn by the realm
    /// they had already left.
    pub refused_unknown_sender: u64,
    /// Rosters applied from the ownership record — the router being TOLD whose frames it may read, rather
    /// than inferring it. On a healthy demand cluster this moves a handful of times as realms spin up and
    /// down, and then stops.
    pub shard_rosters_applied: u64,
    /// Rosters refused as older than the one held. Not a fault by itself — a redelivery on a re-driven
    /// lane is expected — but a rising count next to a router that is missing shards means pushes are
    /// arriving out of order and the tick guard is the only thing holding the line.
    pub shard_roster_stale: u64,
    /// A session in the `subscribed_shards` reverse index for `from` had NO matching
    /// `SubEntry` in its hot `SubTable` (an index/table desync — an invariant breach the
    /// `publish_subs` co-republish makes impossible by construction). Counted, never a silent
    /// `continue` (C2 honesty floor). A straggler from a just-closed source is NOT this — that
    /// is the one-tick `Draining` grace, drained not dropped.
    pub frame_sub_desync: u64,
    /// A sub close was REFUSED because the shard is the session's CURRENT authority
    /// ([`GatewaySessions::close_sub`]'s authority-sub invariant). The reachable producer is a
    /// SAME-NODE re-home (`source == dest`): its `ReleaseSubscribe` names a `src` that is also the
    /// post-commit authority, and honouring that close would strand the client on its own owner
    /// (the total-freeze class). `> 0` simply means the cluster ran a same-node re-home — expected
    /// whenever an occupant re-enters the realm it just left; it is a health signal, not an error.
    pub sub_close_refused_authority: u64,
    /// A `TransferControl` command (or cut marker, or a `SubscriptionReady` read-plane
    /// notice) for an unknown/absent session, or a phase command whose `TransferProgress`
    /// prerequisite is missing — dropped + counted, never panicked (mirrors `inputs_unroutable`).
    pub transfer_unroutable: u64,
    /// A still-parked route phase (after 1c.4: `ReleaseSubscribe`, the demote tail).
    /// Counted + warned, no ack (the saga correctly pins until its handler exists),
    /// never journaled.
    pub transfer_control_parked: u64,
    /// `CommitAuthority` arrived for a BOUND transfer (session + matching `TransferProgress`
    /// present) but `route.cut` is `None` — a FreezeSource-precedes-commit ordered-control
    /// protocol violation. Counted + `tracing::error`-logged; the saga PINS (route untouched),
    /// never a garbage-dest swap. DISTINCT from `transfer_unroutable` (the session or the
    /// in-flight transfer is absent) so the WEDGE-1 pin signal stays unblurred.
    pub commit_without_cut: u64,
    /// A `seq > marker_seq` client input held in the cut buffer for the dest (the partition
    /// fired). Drained to the dest at `CommitAuthority`. 0 outside a transfer's cut window.
    pub inputs_buffered_for_dest: u64,
    /// A buffered input dropped because the cut buffer hit `max_buffered_inputs` (oldest-
    /// first, latest-wins) — the never-silent floor; nonzero only under a parked/stalled
    /// saga holding the cut open (D-23). The durable backstop is 1d/P3 (D-8).
    pub dest_inputs_dropped: u64,
    /// D-3 Slice 5b — PROACTIVE Session self-fences: an Active session whose lease went un-confirmed past
    /// `self_fence_grace_ticks` (a detected partition from the orchestrator) was hard-stopped. Ops
    /// visibility / partition signal; `0` on the happy path and inert (`self_fence_grace_ticks == 0`).
    pub sessions_self_fenced_lapsed: u64,
    /// RLM 5f-3d — bounded-TTL dynamic-home bootstrap FAILURES: a login whose demand-spawned home realm did
    /// not become routable (or whose resolved home never confirmed the attach) inside
    /// `bootstrap_ttl_ticks` was Closed LOUDLY. `0` on every happy path and inert for a static gateway; a
    /// nonzero value is the ops signal that the spawn path — not the session path — is broken.
    pub home_bootstrap_timeouts: u64,
    /// RLM 5f-3d — a `home_bootstraps` member referenced a session absent from `by_session` (an invariant
    /// breach the `begin_home_wait`/`end_home_wait` pairing makes impossible by construction). Counted,
    /// never a silent `continue` — the same C2 honesty floor as `frame_sub_desync`. (Plain backticks, not an
    /// intra-doc link: this is PUBLIC documentation naming a private index.)
    pub home_wait_desync: u64,
    /// RLM 5f-3d (MF3) — committed-lease logins HELD because the gateway is ARMED but its clock has not
    /// synced yet, so the home realm could not be demanded (a pre-sync demand carries `universe_tick` 0,
    /// which the reconciler reads as "never demanded"). The login stays in `AwaitingDirectory` and its
    /// idempotent `LeaseGrant` is re-driven, so this self-heals on the first `ClockSync`: a small count at
    /// boot is normal, a CLIMBING one says the clock broadcast never arrived (check the orchestrator's
    /// `clock_peers`). `0` on a static gateway — the static path never consults the clock.
    pub logins_held_pre_sync: u64,
    /// D-3 Slice 5b — REACTIVE Session self-fences: a `Session`-head recheck reply revealed the lease had
    /// been reassigned/revoked (no longer this gateway at its fence), so the session was hard-stopped
    /// promptly (the link-alive cure, vs the proactive timer's partition cure). `0` on the happy path.
    pub sessions_self_fenced_revoked: u64,
    /// RLM 5f RG-3 — reactive-greeting `ShardPresence` frames received from demand-spawned shards. Pure
    /// OBSERVABILITY: the connection-learning that makes the shard reachable already happened below the app
    /// seam (the mesh records the return connection on this same reliable frame), so the gateway does NOTHING
    /// with it but count + debug-log — never a `dynamic_shards` claim (a presence has no session role or
    /// lifetime, and the roster is authority-refcounted). `0` on a static gateway (no shard ever greets). Its
    /// existence keeps a greeting from being miscounted `undecodable` (the honesty floor).
    pub presence_announces: u64,
    /// THE STAR CATALOGUE's parts forwarded to clients (S11). Expected to reach a small number per
    /// session and STOP: a sky that does not change is stated once. A counter that keeps climbing on a
    /// static world means the catalogue is being re-issued for nothing — the exact cost S11 removes.
    pub star_catalogue_parts_sent: u64,
    /// THE SKY'S LIVENESS BEATS forwarded to clients (S11). EXPECTED to climb forever on a static
    /// world — this is the one counter here whose flat line is the defect, not its growth.
    pub sky_alive_beats_sent: u64,
    /// REQUESTS FOR THE SKY sent to shards (S11). The gateway asks at most once per beat, and only
    /// while some client behind it lacks the sky the beat named. Settles once every client is served.
    pub sky_requests_sent: u64,
    /// CLIENTS STATING THE SKY THEY HOLD (S11). Each statement is what lets the gateway skip sending a
    /// 7.0 MB catalogue to a client that already has it.
    pub sky_held_stated: u64,
    /// CATALOGUE PARTS NOT SENT because the client already held that sky (S11). This is the saving,
    /// counted: every skipped part is bytes that did not cross for a galaxy that did not move.
    pub sky_parts_skipped: u64,
    /// SKY STATEMENTS REFUSED FROM A SHARD (S11). Expected ZERO for ever: no shard states a sky since
    /// the galaxy moved to the gateway. A non-zero value means a shard binary has not caught up — which
    /// is worth seeing rather than dropping in silence.
    pub sky_from_shard_refused: u64,
    /// THE WINDOW LANE's admitted rows (mesh minor 16): a `WindowFrame`/`WindowBody`/
    /// `WindowMembership` row that PASSED attestation (a known window, the roster-head sender,
    /// an admissible body) and was INGESTED into the window's composer state (Slice B retired
    /// Slice A's deliberate `window_rows_unconsumed` — the engine now consumes every admitted
    /// row). Counted APART from `undecodable` (the honesty floor); grows at the realm-lane rate.
    pub window_rows_ingested: u64,
    /// THE WINDOW LANE, fail-closed (Slice A — docs/design/window_lane.md §2.2/§2.6.6): a window
    /// row whose sender is NOT the head this gateway resolved for the stating realm when it
    /// opened the window ([`window_sender_is_head`] driven with the held head). A forged or
    /// deposed sender's row is dropped + counted, never served on faith. `0` healthy.
    pub window_sender_mismatch: u64,
    /// THE WINDOW LANE, fail-closed: a `WindowBody` whose authorship the admission rule refuses
    /// ([`window_body_admissible`] — a look about anything but the author itself, a marker about
    /// a non-child). Dropped + counted, never patched (§2.2). `0` healthy.
    pub window_misauthored_body: u64,
    /// THE WINDOW LANE, fail-closed: a window row naming an id this gateway holds no window for —
    /// a straggler from a closed window (dropped by id mismatch, exactly the `WindowId` contract)
    /// or a forged id. BENIGN in small numbers around closes; steady growth is a fault.
    pub window_unknown_row: u64,
    /// THE WINDOW LANE's control egress — `WindowOpen`s sent for NEWLY derived windows
    /// (THROUGHPUT; one per window per derivation edge, never per tick).
    pub window_open_sent: u64,
    /// THE WINDOW LANE's control egress — `WindowClose`s sent when a window stopped being
    /// derivable (a session ended, crossed away, or its parent sub drained). The polite fast
    /// path; the shard's derived TTL is the crash backstop.
    pub window_close_sent: u64,
    /// THE WINDOW LANE's keep-alive egress — `WindowOpen` re-asserts on the derived cadence
    /// (idempotent per window; the shard's TTL is 2 beats + 1). THROUGHPUT.
    pub window_keepalives_sent: u64,
    /// Slice B — a `WindowFrame` stamped further behind its window's ring head than the derived
    /// span (`WindowTuning::ring_span_ticks`): refused, counted, never folded (§2.6.2).
    pub window_level_refused: u64,
    /// Slice B — a `WindowBody` older (by `authored_at`) than the statement already held for its
    /// subject: refused (newest wins on a re-driven lane), counted.
    pub window_body_stale: u64,
    /// Slice B — a `Marker` arriving before its author's FIRST placement level, so the
    /// stream-only child roster (the author's own attested full-roster rows — §2.6.2 deleted the
    /// seed-forest source) cannot vouch for the subject yet: refused fail-closed, counted apart
    /// from `window_misauthored_body` (a well-ordered peer racing its own lanes is not a forger).
    pub window_body_preroster: u64,
    /// Slice B — SHARED folds actually computed: one per (origin, common tick) per gateway tick,
    /// shared across every session standing in that origin (§2.14). THROUGHPUT.
    pub window_folds: u64,
    /// Slice B — sessions served from an already-computed shared fold this tick (§2.14's
    /// memoization). The parity gate requires this nonzero with ≥2 co-located sessions.
    pub window_fold_hits: u64,
    /// Slice B — a shared fold that did NOT equal the per-session recompute bit-for-bit (the
    /// §2.14 "dedup f64 agreement: shared fold == per-session fold" measurement, taken on every
    /// memo hit). MUST stay 0 — asserted by the parity gate.
    pub window_fold_divergence: u64,
    /// Slice B — folds whose fresh prefix covered the WHOLE derived chain of length ≥ 2: direct
    /// proof that two different shards stamped levels at IDENTICAL universe ticks (the
    /// exact-cadence boot pin, §2.6.3 — asserted nonzero by the parity gate).
    pub window_full_chain_folds: u64,
    /// Slice B — GAUGE: sessions holding a non-empty derived chain this tick.
    pub window_chains_held: u64,
    /// Slice B — a stratum HELD at its last composed poses this tick (per stratum per tick —
    /// §2.6.4 `compose_hold_ticks`): the sky above a lagging hop holds while the local world
    /// keeps moving.
    pub window_compose_hold_ticks: u64,
    /// Slice B — held strata removed by the DEAD-HOP EXIT (held past the derived TTL): clean
    /// removal, counted, never a silent truncation (§2.6.4).
    pub window_hop_dead: u64,
    /// Slice B — a would-be common-tick REWIND froze emission (§2.6.3 monotone-T). 0 healthy.
    pub window_t_monotone_stalled: u64,
    /// Slice B — a chain derivation met a cycle and truncated fail-closed (§2.6.2). 0 healthy.
    pub window_chain_cycle: u64,
    /// Slice B — rows refused by `FrameError::InstantMismatch` inside a fold: the shear law
    /// firing (§2.6.3). MUST stay 0 in a healthy run (asserted by the parity gate); driven
    /// nonzero on purpose by the G-SHEAR anti-vacuity unit.
    pub window_instant_mismatch: u64,
    /// Slice B — rows refused by `RotationBeyondExactReach` inside a fold (pre-P10 cell math;
    /// today reachable only at the author's inversion — pinned by the composer unit).
    pub window_rotated_refused: u64,
    /// Slice B — rows dropped because their stated tail frame was not their level's own frame
    /// (or an unknown-frame refusal from the fold): alien, dropped, counted (§2.6.6).
    pub window_alien_rows: u64,
    /// Slice B — a chain level whose hop was absent/mismatched (or whose roster was empty): the
    /// fold's fresh prefix capped there, fail-closed.
    pub window_hop_invalid: u64,
    /// Slice B — an Active session whose composed feed was WITHHELD because its own-level window
    /// (or its own realm) is not derivable yet — the login race of §2.6.6, held not guessed.
    pub window_unresolved_standing: u64,
    /// Slice B — the §2.12 hop-vs-child-row agreement: composed positions for the SAME realm from
    /// the two lawful sources that disagreed AT ALL (bit-level). 0 on THE world today (identity
    /// orientations cancel exactly) — asserted by the parity gate.
    pub window_dedup_disagree: u64,
    /// Slice B — GAUGE (max): the largest measured hop-vs-child-row deviation, in nanometres —
    /// the §2.12 "measured bound", printed by the parity gate.
    pub window_dedup_max_dev_cells: u64,
    /// Slice B — `HeadRead{Realm(ancestor)}` polls sent on the window keep-alive cadence to
    /// resolve lineage ancestors the session never subscribed to (the EXISTING directory pair —
    /// no new wire arm). THROUGHPUT.
    pub window_head_reads_sent: u64,
    /// Slice B — composed rows produced across all folds. THROUGHPUT; with
    /// `parity_rows_matched` it bounds the (explained) composed-surplus: the composer carries
    /// the FULL direct-child roster of every chain level, dormant children included.
    pub window_composed_rows: u64,
    /// §2.6.5 step 4 (Q2 = PARENT RELAY, Slice C1): relayed live-child interior rows composed
    /// into folds — the sibling-interior carrier's rows actually reaching drawn scenes.
    pub window_relay_rows_composed: u64,
    /// C5 split (look_horizon.md §3.5 — the old `window_relay_unplaceable` aggregated three
    /// classes, one structurally dead): the chain descent below a relaying stratum could not be
    /// rebuilt at the fold's tick — dropped, healed by the next fold.
    pub window_relay_descent_refused: u64,
    /// C5 split: a relayed child with NO ring level at-or-before the fold's tick (G-RELAY-STAMP
    /// asserts this stays 0 over a process flight — the D-WINDOW-6(2) discharge).
    pub window_relay_stamp_missing: u64,
    /// C5 split: a member child missing from the stratum author's own level at the fold's tick
    /// (a roster race) — healed by the author's next level.
    pub window_relay_unrostered: u64,
    /// C5 GAUGE (max): the at-or-before fallback's declared skew, in ticks (G-RELAY-STAMP
    /// bounds it by one keep-alive beat).
    pub window_relay_stamp_skew_ticks: u64,
    /// C5 GAUGE (max): the deepest relayed subject composed, in levels below its forwarding
    /// author — 2 is the carrier's whole arity; more is an implementation climb bug (§3.3.4).
    pub window_relay_depth_max: u64,
    /// ★TOMBSTONED LANES, counted so a revived producer is never silent (window lane Slice C2,
    /// minor 19). `old_realm_frames_dropped`: the old opaque per-tick realm datagram
    /// (`ShardToGateway::RealmFrame`), whose producer and whose last consumer (the Slice-B parity
    /// comparator) both died here. Non-zero means a shard is speaking a lane nobody serves.
    pub old_realm_frames_dropped: u64,
    /// ★TOMBSTONED (window lane Slice C2, minor 19): old-lane per-observer scene deltas received
    /// from a shard and DROPPED. The composed lane has been the one scene author since the
    /// minor-18 flag day, and the shard-side producer is now deleted too — the ids-only
    /// `WindowMembership` verdict replaced it. `0` on a healthy cluster; non-zero means a shard
    /// is speaking a lane nobody serves.
    pub old_scene_deltas_dropped: u64,
    /// Composed-scene EGRESS — full levels shipped (`ServerControlMsg::RealmRegistry`): one per
    /// epoch bump (login, crossing). The pixel gates' emitted-level provenance.
    pub scene_levels_sent: u64,
    /// Composed-scene EGRESS — reliable deltas shipped (`ServerControlMsg::RealmSceneDelta`) on
    /// membership/body change at a stable epoch.
    pub scene_deltas_sent: u64,
    /// Composed-scene EGRESS — per-tick composed realm datagrams shipped (chunks counted; one
    /// tick's chunks share a frame id).
    pub scene_datagrams_sent: u64,
    /// Q2 relay (mesh minor 17) — relayed statements ADMITTED into a window's ingest (levels held
    /// per child; bodies into the one body store — the marker⇒look handover path).
    pub window_relays_ingested: u64,
    /// Q2 relay — sealed blobs that did not decode (counted, dropped — fail-closed).
    pub window_relay_undecodable: u64,
    /// Q2 relay — a relayed child the window author's own attested roster does not vouch
    /// (dropped; the roster is the stream-only child set, same as the marker admission).
    pub window_relay_unvouched: u64,
    /// Q2 relay — a relay refused by the CHILD's fence order (a deposed incarnation still
    /// shipping) or a relayed level older than the held one. Never normal past a re-home window.
    pub window_relay_stale: u64,
    /// Slice D (§2.8 departure mirror): SELF-LOOKs dropped by the derived roster-loss window — the
    /// realm behind them stopped speaking, so its parent's marker resumes and the body it drew
    /// shrinks to a point of light.
    pub window_looks_pruned: u64,
    /// Slice D: relayed interior LEVELS dropped by the same window — a dead live-child's interior
    /// cannot keep composing rows after its statements stop.
    pub window_relay_levels_pruned: u64,
    /// Look horizon slice 3 (§3.2 admission rule 1) — a forwarded grandchild batch naming a
    /// child the RELAYING child's own attested roster does not vouch: refused + counted. A
    /// VIOLATION counter — gates assert it 0 on every lawful flight (a hostile or buggy
    /// forwarder is the only producer).
    pub window_relay_interior_unvouched: u64,
    /// Look horizon slice 3 (§3.2 admission rule 3) — statements inside an admitted grandchild
    /// batch that are NOT the author's own picture (its `Level`, its markers): depth-3 subjects
    /// no row can exist for, dropped + counted. The LAWFUL filter — expected NON-zero on any
    /// flight that forwards an interior, never asserted zero (deliberately a different counter
    /// from the violation above).
    pub window_relay_interior_filtered: u64,
}
