//! The ControlOracle invariants live in P1: AUTHORITY-UNIQUE and
//! INPUT-CONSERVATION (`docs/design/test_harness.md` §oracles). Ground truth comes
//! from [`crate::topology::InspectReport`]s — the directory, node-reported
//! held-sets, shard input logs, and client sent-logs. Oracles AUDIT; they never
//! trust a single node's claim about another.

use std::collections::{BTreeMap, BTreeSet};

use vd_core::glam::{DQuat, DVec3};
use vd_core::pose::FrameRef;
use vd_core::{EntityId, Fence, NodeId, SessionId, TickId};
use vd_node::saga_runtime::ActiveTransfer;
use vd_wire::channels::SubId;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, OwnerRecord};

use crate::topology::InspectReport;

/// AUTHORITY-UNIQUE failed: an entity without EXACTLY one owner.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AuthorityViolation {
    #[error("entity {entity} is held by {holders:?} — exactly one holder required")]
    WrongHolderCount {
        entity: EntityId,
        holders: Vec<NodeId>,
    },
    #[error("entity {entity} held by {holder} but the directory records {recorded}")]
    DirectoryDisagrees {
        entity: EntityId,
        holder: NodeId,
        recorded: String,
    },
    #[error("entity {entity} held by {holder} has NO directory record (zero-owner)")]
    Unrecorded { entity: EntityId, holder: NodeId },
    #[error("directory entity {entity} (owner {recorded}) is held by no live node")]
    HeldNowhere { entity: EntityId, recorded: String },
    #[error("entity {entity} is still pending at {node} after the run settled")]
    UnsettledPending { entity: EntityId, node: NodeId },
    #[error("transfer subject {subject} still has a live saga at {node} after the run settled")]
    UnsettledTransfer { subject: String, node: NodeId },
    #[error("entity {entity} held at {held} but the directory records {recorded} (FENCE-9)")]
    FenceMismatch {
        entity: EntityId,
        held: Fence,
        recorded: Fence,
    },
    #[error("realm {realm} is held by {holders:?} — exactly one holder required")]
    RealmWrongHolderCount { realm: String, holders: Vec<NodeId> },
    #[error("realm {realm} held by {holder} at {held} disagrees with the directory")]
    RealmDisagrees {
        realm: String,
        holder: NodeId,
        held: Fence,
    },
    #[error("directory realm {realm} (owner {recorded}) is held by no live shard")]
    RealmHeldNowhere { realm: String, recorded: String },
}

/// AUTHORITY-UNIQUE (binding P1, checked every committed tick): every entity in
/// any held-set or directory record has EXACTLY one holder, and the directory
/// agrees with it. `len == 1` exactly — `len > 1` is split-brain, `len == 0` is a
/// zero-owner orphan; both are violations (`<= 1` would hide orphans).
///
/// In P1 `reports` come from one orchestrator + one shard; the audit is written
/// for ANY number of each (P2 adds shards, nothing here changes).
///
/// # Errors
/// The first [`AuthorityViolation`] found, in deterministic entity order.
pub fn verify_authority_unique(
    reports: &[(NodeId, InspectReport)],
) -> Result<(), AuthorityViolation> {
    verify_authority_unique_excluding(reports, &BTreeSet::new())
}

/// AUTHORITY-UNIQUE, but EXCLUDING the held/pending/departing claims of nodes the caller knows are
/// DEAD (the P3 crash matrix: a killed/crashed participant's `Dot` is a CORPSE — its stale authority
/// claim must not be read as live). A directory record naming a dead node then surfaces as the honest
/// [`AuthorityViolation::HeldNowhere`] (the orphan), and a dead W1-transfer-source is excluded BEFORE
/// the W1 excuse so a live parked saga cannot mask a dead source. With an empty `dead` set this is
/// byte-identical to [`verify_authority_unique`] (the orchestrator + a live cluster are never dead).
///
/// # Errors
/// The first [`AuthorityViolation`] found, in deterministic entity order.
pub fn verify_authority_unique_excluding(
    reports: &[(NodeId, InspectReport)],
    dead: &BTreeSet<NodeId>,
) -> Result<(), AuthorityViolation> {
    // Who CLAIMS to hold each entity, and at what fence.
    let mut holders: BTreeMap<EntityId, Vec<NodeId>> = BTreeMap::new();
    let mut held_fence: BTreeMap<EntityId, Fence> = BTreeMap::new();
    // Who awaits a grant confirmation for each entity (the legal in-flight window).
    let mut pending: BTreeMap<EntityId, Vec<NodeId>> = BTreeMap::new();
    // What the directory RECORDS for each entity key (authority + fence).
    let mut recorded: BTreeMap<EntityId, OwnerRecord> = BTreeMap::new();
    let mut departing: BTreeMap<EntityId, Vec<NodeId>> = BTreeMap::new();
    // Live (in-flight) transfers, keyed by Entity subject (1d.5b.3d) — the W1 transfer-window
    // excuse ground truth. Non-Entity (Realm/Ship) subjects are not entity-key transfers, so they
    // never excuse an entity disagreement; their build arm is covered by a Realm-subject unit test.
    let mut active: BTreeMap<EntityId, ActiveTransfer> = BTreeMap::new();
    for (node, report) in reports {
        // A DEAD node's report is a corpse — its held/pending/departing claims are stale (it cannot
        // be a live holder). Skipped BEFORE the holder/W1 logic so a dead node's authority can never
        // false-pass uniqueness nor (as a W1 source) mask the orphan. (A dead shard/gateway carries no
        // directory/active anyway — those are orchestrator-only, and the orchestrator is never dead.)
        if dead.contains(node) {
            continue;
        }
        for (entity, fence) in &report.held_entities {
            holders.entry(*entity).or_default().push(*node);
            held_fence.insert(*entity, *fence);
        }
        for entity in &report.pending_entities {
            pending.entry(*entity).or_default().push(*node);
        }
        for entity in &report.departing_entities {
            departing.entry(*entity).or_default().push(*node);
        }
        for (key, record) in &report.directory {
            if let DirectoryKey::Entity(entity) = key {
                recorded.insert(*entity, *record);
            }
        }
        for at in &report.active_transfers {
            if let DirectoryKey::Entity(entity) = at.subject {
                active.insert(entity, *at);
            }
        }
    }

    // Every held entity: exactly one holder, the directory names it, AND the holder's
    // fence matches the directory's (FENCE-9 — a stale-fence holder is split-brain).
    for (entity, holding_nodes) in &holders {
        if holding_nodes.len() != 1 {
            return Err(AuthorityViolation::WrongHolderCount {
                entity: *entity,
                holders: holding_nodes.clone(),
            });
        }
        let holder = holding_nodes[0];
        match recorded.get(entity) {
            None => {
                // Legal ONLY while the holder is releasing it (revoke recorded
                // at the directory, confirmation in flight back).
                let releasing = departing
                    .get(entity)
                    .is_some_and(|nodes| nodes.contains(&holder));
                if !releasing {
                    return Err(AuthorityViolation::Unrecorded {
                        entity: *entity,
                        holder,
                    });
                }
            }
            Some(record) if record.authority == AuthorityRef::Shard(holder) => {
                let held = held_fence[entity];
                if held != record.fence {
                    return Err(AuthorityViolation::FenceMismatch {
                        entity: *entity,
                        held,
                        recorded: record.fence,
                    });
                }
            }
            Some(record) => {
                // W1 (1d.5b.3d): the LEGAL post-CAS, pre-demote transfer window — the source still
                // holds the subject Owned at the old fence while the directory already records the
                // dest. Excused ONLY for a live saga of the exact (source→dest) shape; this arm is
                // reached AFTER the `len == 1` guard, so the excuse can NEVER mask a split-brain.
                if excuse_w1(&active, *entity, holder, record, held_fence[entity]) {
                    continue;
                }
                return Err(AuthorityViolation::DirectoryDisagrees {
                    entity: *entity,
                    holder,
                    recorded: format!("{:?}", record.authority),
                });
            }
        }
    }

    // Every directory-recorded entity is held SOMEWHERE — or its grant
    // confirmation is still in flight TOWARD THE RECORDED OWNER (the directory
    // commit precedes the owner's knowledge by one delivery; an entity pending
    // anywhere else is NOT excused).
    for (entity, record) in &recorded {
        if holders.contains_key(entity) {
            continue;
        }
        let in_flight_to_owner = pending.get(entity).is_some_and(|nodes| {
            nodes
                .iter()
                .any(|node| record.authority == AuthorityRef::Shard(*node))
        });
        if !in_flight_to_owner {
            return Err(AuthorityViolation::HeldNowhere {
                entity: *entity,
                recorded: format!("{:?}", record.authority),
            });
        }
    }

    // REALM authority is checked the same way (FENCE-3): a realm transfer that
    // double-grants must not pass green just because the oracle only looked at
    // entity keys. (Ship keys extend identically at P8.)
    verify_realm_authority(reports, dead)
}

/// The 1d.5b.3d W1 transfer-window excuse: is this `DirectoryDisagrees` the LEGAL post-CAS,
/// pre-demote window — the source still holds the subject `Owned` at the OLD fence while the directory
/// already records the dest at the new fence? True ONLY when a LIVE saga for THIS entity has
/// (a) an entry at all, (b) `source == holder` (the source is the one still holding), (c) the record
/// names `Shard(dest)` (the directory moved to the saga's dest), and (d) the held fence is STALE
/// versus the record fence (the holder is at the old fence). Reached only AFTER the `len == 1` guard,
/// so it can NEVER excuse a split-brain (two holders); it never touches `FenceMismatch` (the
/// record-names-the-holder arm) or `WrongHolderCount`. Monomorphic: the `let-else` and the three
/// bitwise-`&` conjuncts keep every false arm coverable in THIS helper, not the audit loop (HR5).
#[must_use]
fn excuse_w1(
    active: &BTreeMap<EntityId, ActiveTransfer>,
    entity: EntityId,
    holder: NodeId,
    record: &OwnerRecord,
    held: Fence,
) -> bool {
    let Some(at) = active.get(&entity) else {
        return false;
    };
    (at.source == holder)
        & (record.authority == AuthorityRef::Shard(at.dest))
        & held.is_stale_against(record.fence)
}

/// The realm-key half of AUTHORITY-UNIQUE: every realm record has exactly one
/// holding shard at the matching fence, and every held realm is recorded.
fn verify_realm_authority(
    reports: &[(NodeId, InspectReport)],
    dead: &BTreeSet<NodeId>,
) -> Result<(), AuthorityViolation> {
    use vd_core::pose::RealmId;
    let mut holders: BTreeMap<RealmId, Vec<(NodeId, Fence)>> = BTreeMap::new();
    let mut pending: BTreeMap<RealmId, Vec<NodeId>> = BTreeMap::new();
    let mut recorded: BTreeMap<RealmId, vd_wire::seams::directory::OwnerRecord> = BTreeMap::new();
    for (node, report) in reports {
        // A dead shard's held realm is a corpse too (the P3 crash matrix): excluded, so its realm
        // record surfaces as the honest `RealmHeldNowhere` rather than a false-passing dead holder.
        if dead.contains(node) {
            continue;
        }
        for (realm, fence) in &report.held_realms {
            holders.entry(*realm).or_default().push((*node, *fence));
        }
        for realm in &report.pending_realms {
            pending.entry(*realm).or_default().push(*node);
        }
        for (key, record) in &report.directory {
            if let DirectoryKey::Realm(realm) = key {
                recorded.insert(*realm, *record);
            }
        }
    }
    for (realm, held) in &holders {
        if held.len() != 1 {
            return Err(AuthorityViolation::RealmWrongHolderCount {
                realm: realm.to_string(),
                holders: held.iter().map(|(n, _)| *n).collect(),
            });
        }
        let (holder, fence) = held[0];
        match recorded.get(realm) {
            Some(record)
                if record.authority == AuthorityRef::Shard(holder) && record.fence == fence => {}
            _ => {
                return Err(AuthorityViolation::RealmDisagrees {
                    realm: realm.to_string(),
                    holder,
                    held: fence,
                });
            }
        }
    }
    for (realm, record) in &recorded {
        if holders.contains_key(realm) {
            continue;
        }
        // Excused while the grant confirmation is in flight TO the recorded owner
        // (the directory commit precedes the shard's knowledge by one delivery).
        let in_flight_to_owner = pending.get(realm).is_some_and(|nodes| {
            nodes
                .iter()
                .any(|node| record.authority == AuthorityRef::Shard(*node))
        });
        if !in_flight_to_owner {
            return Err(AuthorityViolation::RealmHeldNowhere {
                realm: realm.to_string(),
                recorded: format!("{:?}", record.authority),
            });
        }
    }
    Ok(())
}

/// The settled form: after a quiesce window NOTHING may remain pending — every
/// requested grant has resolved, and AUTHORITY-UNIQUE holds exactly.
///
/// # Errors
/// [`AuthorityViolation::UnsettledPending`] for any lingering pending entity,
/// or whatever [`verify_authority_unique`] finds.
pub fn verify_authority_settled(
    reports: &[(NodeId, InspectReport)],
) -> Result<(), AuthorityViolation> {
    for (node, report) in reports {
        if let Some(entity) = report
            .pending_entities
            .first()
            .or(report.departing_entities.first())
        {
            return Err(AuthorityViolation::UnsettledPending {
                entity: *entity,
                node: *node,
            });
        }
        // 1d.5b.3d: a still-LIVE saga post-quiesce means the transfer never settled — the W1 excuse
        // is a MID-FLIGHT allowance only; once settled, NO transfer may be in flight.
        if let Some(at) = report.active_transfers.first() {
            return Err(AuthorityViolation::UnsettledTransfer {
                subject: format!("{:?}", at.subject),
                node: *node,
            });
        }
    }
    verify_authority_unique(reports)
}

/// INPUT-CONSERVATION failed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConservationViolation {
    #[error("input (session {session}, seq {seq}) applied {count} times — exactly once allowed")]
    AppliedTwice {
        session: SessionId,
        seq: u64,
        count: usize,
    },
    #[error("applied seqs for session {session} are not strictly increasing: {seqs:?}")]
    NonMonotonicApply { session: SessionId, seqs: Vec<u64> },
    #[error(
        "sent input (session {session}, seq {seq}) is unaccounted: neither applied \
         nor discarded-with-reason"
    )]
    Unaccounted { session: SessionId, seq: u64 },
    #[error("applied input (session {session}, seq {seq}) was never sent by any client")]
    Phantom { session: SessionId, seq: u64 },
}

/// INPUT-CONSERVATION (binding P1, zero-fault form): every input a client SENT is
/// accounted for exactly once across all shards — applied once, or discarded with
/// a typed reason — and per-session applied seqs are strictly increasing. Phantom
/// applies (never sent) are violations too. Call after a quiesce window so no
/// input is still in flight.
///
/// # Errors
/// The first [`ConservationViolation`] found, in deterministic order.
pub fn verify_input_conservation(
    reports: &[(NodeId, InspectReport)],
) -> Result<(), ConservationViolation> {
    let mut sent: BTreeMap<(SessionId, u64), usize> = BTreeMap::new();
    let mut applied: BTreeMap<(SessionId, u64), usize> = BTreeMap::new();
    let mut discarded_with_seq: BTreeMap<(SessionId, u64), usize> = BTreeMap::new();
    let mut applied_order: BTreeMap<SessionId, Vec<u64>> = BTreeMap::new();

    for (_, report) in reports {
        for key in &report.sent_inputs {
            *sent.entry(*key).or_default() += 1;
        }
        for key in &report.applied_inputs {
            *applied.entry(*key).or_default() += 1;
            applied_order.entry(key.0).or_default().push(key.1);
        }
        for (session, seq, _) in &report.discarded_inputs {
            if let Some(seq) = seq {
                *discarded_with_seq.entry((*session, *seq)).or_default() += 1;
            }
        }
    }

    // Exactly-once application + strict per-session monotonicity.
    for ((session, seq), count) in &applied {
        if *count != 1 {
            return Err(ConservationViolation::AppliedTwice {
                session: *session,
                seq: *seq,
                count: *count,
            });
        }
        if !sent.contains_key(&(*session, *seq)) {
            return Err(ConservationViolation::Phantom {
                session: *session,
                seq: *seq,
            });
        }
    }
    for (session, seqs) in &applied_order {
        let strictly_increasing = seqs.windows(2).all(|w| w[0] < w[1]);
        if !strictly_increasing {
            return Err(ConservationViolation::NonMonotonicApply {
                session: *session,
                seqs: seqs.clone(),
            });
        }
    }
    // Every sent input is applied or discarded-with-its-seq. (Reasonless discards
    // — malformed/unknown-session — carry no seq and cannot account for a sent
    // input; they cover injected garbage, not script traffic.)
    for (session, seq) in sent.keys() {
        let accounted = applied.contains_key(&(*session, *seq))
            || discarded_with_seq.contains_key(&(*session, *seq));
        if !accounted {
            return Err(ConservationViolation::Unaccounted {
                session: *session,
                seq: *seq,
            });
        }
    }
    Ok(())
}

// =================================================================================================
// THE RENDER-TRACE ORACLES (WireMonitor §5b): NO-VANISH + POSE-CONTINUITY — what a real client
// actually drew, tick-by-tick, NEVER a node's internal hope (`test_harness.md` §5b). They consume a
// captured per-tick RENDER trace (one composited sample per held entity, from the client's REAL
// `DeliveredView`), so a fabric that delays the dest's first frame past K produces a real
// client-visible vanish the oracle DOES see, and a wrong frame-eval comparator passes a teleport.
// =================================================================================================

/// One entity's composited render sample at one tick: the sub it was rendered FROM, its frame, its
/// WORLD-evaluated position (the `DeliveredView::world_pos` of its render pose — NOT the raw
/// frame-local `pos`, which silently diverges from world space for a `ShipLocal` interior), and its
/// orientation. POSE-CONTINUITY compares `world_pos` so it survives D-27 frame-rebind and P8
/// ship-interior frames; carrying `raw_pos` alongside lets the oracle's meta-test PROVE it ignores
/// the raw component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderSample {
    /// The authoritative sub the entity was composited from (the source→dest flip is a change here).
    pub sub: SubId,
    /// The delivered frame (the load-bearing crossing discriminator; never interpolated).
    pub frame: FrameRef,
    /// The frame-LOCAL render position, BEFORE world-evaluation (carried only so the meta-test can
    /// prove the oracle reads `world_pos`, not this). The oracles MUST NOT read this for continuity.
    pub raw_pos: DVec3,
    /// The world-space position (`DeliveredView::world_pos` of the render pose). The continuity basis.
    pub world_pos: DVec3,
    /// The render orientation (for the rotation-continuity arm).
    pub orient: DQuat,
}

/// A captured per-tick render trace for ONE subject entity: at each topology tick, the composited
/// sample the client rendered for the subject (`None` ⇒ the subject rendered NOTHING that tick — a
/// vanish candidate). Built from delivered bytes only (the `DeliveredView`), mirroring how
/// `verify_input_conservation` consumes captured logs — the oracle never touches node internals.
pub type RenderTrace = Vec<(TickId, Option<RenderSample>)>;

/// The tolerances NO-VANISH / POSE-CONTINUITY are checked against — every value DERIVED, never
/// magic (`test_harness.md` §5b: K and ε are derived, not magic). Built with [`RenderTolerances::derive`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderTolerances {
    /// NO-VANISH K: the max consecutive ticks the subject may be absent while a transfer is
    /// in-flight. Derived `K < band − delay`; in the zero-fault lockstep gate the dest's
    /// `SubscriptionReady`→`AuthorityChanged` and its first data frame land in the SAME client step
    /// (the §2 fabric-ordering chain), so `delay == 0`, `band` is unbounded (the source never
    /// drops before authority re-points), and the tight K=0 (present EVERY tick) is correct.
    pub max_absent_ticks: u64,
    /// POSE-CONTINUITY ε_pos (metres): `max_velocity·dt·(1 + stagger_offset) + slack`. The largest
    /// world-space step one tick of authoritative motion can produce at the source→dest flip, plus
    /// a coordinate-conversion slack — so genuine motion passes but a teleport (the failure mode) does
    /// not. Lockstep ⇒ `stagger_offset == 0`.
    pub epsilon_pos: f64,
    /// POSE-CONTINUITY ε_rot (radians): the max angular step in one tick. A look-rate × dt + slack;
    /// for a non-rotating-during-transfer avatar it is the slack alone (a teleport-rotation tripwire).
    pub epsilon_rot: f64,
}

impl RenderTolerances {
    /// Derive the tolerances from the world params + fabric timing — no magic literals.
    ///
    /// * `max_velocity_mps` / `tick_dt_s` — the shard's own motion params (`StubConfig`).
    /// * `stagger_offset_ticks` — the max per-tick stagger between the two shards' frames (0 in the
    ///   lockstep gate); it widens ε_pos because a staggered source/dest can be one extra tick apart.
    /// * `max_fabric_delay_ticks` — the max extra delivery delay on the saga/snapshot class; with the
    ///   §2 ordering chain it is 0 in the zero-fault gate, so `K = 0`. Under chaos it is `> 0` and `K`
    ///   relaxes to `band − delay` (the caller passes `band`).
    /// * `band_ticks` — the overlap-band length (the ticks BOTH subs are live); `K < band − delay`.
    /// * `pos_slack_m` / `rot_slack_rad` — the coordinate-conversion slack (derived from the f64 → wire
    ///   round-trip + sanitize, NOT zero per §5b); the caller supplies the measured round-trip bound.
    /// * `look_rate_rad_per_s` — the max angular rate the avatar can turn (0 if it does not rotate
    ///   during the transfer); ε_rot = `look_rate·dt + rot_slack`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn derive(
        max_velocity_mps: f64,
        tick_dt_s: f64,
        stagger_offset_ticks: u64,
        max_fabric_delay_ticks: u64,
        band_ticks: u64,
        pos_slack_m: f64,
        look_rate_rad_per_s: f64,
        rot_slack_rad: f64,
    ) -> RenderTolerances {
        // K < band − delay (a SATURATING subtraction: a band shorter than the delay floors K at 0,
        // never wraps). The strict `<` means we take `(band − delay) − 1` when positive, else 0.
        let band_minus_delay = band_ticks.saturating_sub(max_fabric_delay_ticks);
        let max_absent_ticks = band_minus_delay.saturating_sub(1);
        // ε_pos = max_velocity·dt·(1 + stagger) + slack. The stagger widens the window by one extra
        // tick per offset (the source and dest frames can be `stagger` ticks apart).
        let stagger_factor = 1.0 + stagger_offset_ticks as f64;
        let epsilon_pos = max_velocity_mps * tick_dt_s * stagger_factor + pos_slack_m;
        let epsilon_rot = look_rate_rad_per_s * tick_dt_s + rot_slack_rad;
        RenderTolerances {
            max_absent_ticks,
            epsilon_pos,
            epsilon_rot,
        }
    }
}

/// NO-VANISH failed: the subject rendered nowhere for too long across the transfer window.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "NO-VANISH: subject absent from the render for {absent_run} consecutive ticks ending at \
     {last_absent} — exceeds K={max_absent_ticks}"
)]
pub struct VanishViolation {
    pub last_absent: TickId,
    pub absent_run: u64,
    pub max_absent_ticks: u64,
}

/// POSE-CONTINUITY failed: at the source→dest authoritative-sub flip the world pose jumped.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum ContinuityViolation {
    #[error(
        "POSE-CONTINUITY: world position jumped {delta} m at the {from:?}→{to:?} sub flip \
         ({tick}) — exceeds ε_pos={epsilon_pos}"
    )]
    Position {
        tick: TickId,
        from: SubId,
        to: SubId,
        delta: f64,
        epsilon_pos: f64,
    },
    #[error(
        "POSE-CONTINUITY: orientation jumped {delta} rad at the {from:?}→{to:?} sub flip \
         ({tick}) — exceeds ε_rot={epsilon_rot}"
    )]
    Rotation {
        tick: TickId,
        from: SubId,
        to: SubId,
        delta: f64,
        epsilon_rot: f64,
    },
}

/// NO-VANISH (WireMonitor §5b): across the captured window the subject must render on SOME held sub
/// every tick (no run of `> K` absent ticks). A fabric that delays the dest's first frame past K, or
/// a source-track-drop before the authority re-points, produces a client-visible vanish this oracle
/// catches — because it reads DELIVERED render samples, not node internals.
///
/// # Errors
/// [`VanishViolation`] for the first absent run that exceeds `tol.max_absent_ticks`.
pub fn verify_no_vanish(trace: &RenderTrace, tol: RenderTolerances) -> Result<(), VanishViolation> {
    let mut absent_run = 0u64;
    for (tick, sample) in trace {
        match sample {
            Some(_) => absent_run = 0,
            None => {
                absent_run += 1;
                if absent_run > tol.max_absent_ticks {
                    return Err(VanishViolation {
                        last_absent: *tick,
                        absent_run,
                        max_absent_ticks: tol.max_absent_ticks,
                    });
                }
            }
        }
    }
    Ok(())
}

/// POSE-CONTINUITY (WireMonitor §5b): at every tick the subject's authoritative SUB changes (a
/// source→dest flip), the WORLD-space pose must move `< ε_pos` and rotate `< ε_rot` from the LAST
/// sample on the old sub to the FIRST on the new — genuine one-tick motion passes, a teleport does
/// not. Compares `world_pos` (NOT `raw_pos`), so it survives ship-interior / rebind frames; the
/// meta-test proves the `raw_pos` field is ignored.
///
/// # Errors
/// The first [`ContinuityViolation`] at a sub flip whose world Δpos or Δrot exceeds the tolerance.
pub fn verify_pose_continuity(
    trace: &RenderTrace,
    tol: RenderTolerances,
) -> Result<(), ContinuityViolation> {
    // Walk the rendered samples (skipping absent ticks — NO-VANISH owns those); compare each
    // adjacent rendered pair, but only ACT when the authoritative sub changed (the A→B flip §5b).
    let mut prev: Option<(TickId, RenderSample)> = None;
    for (tick, sample) in trace {
        let Some(curr) = sample else { continue };
        if let Some((_, last)) = prev
            && last.sub != curr.sub
        {
            check_flip(*tick, last, *curr, tol)?;
        }
        prev = Some((*tick, *curr));
    }
    Ok(())
}

/// The monomorphic continuity check at ONE sub flip (HR5: the `?`/comparisons live here, not in the
/// `verify_pose_continuity` walk). Reads `world_pos` (the continuity basis) and the orientation;
/// `last.raw_pos`/`curr.raw_pos` are deliberately UNUSED.
fn check_flip(
    tick: TickId,
    last: RenderSample,
    curr: RenderSample,
    tol: RenderTolerances,
) -> Result<(), ContinuityViolation> {
    let delta_pos = (curr.world_pos - last.world_pos).length();
    if delta_pos > tol.epsilon_pos {
        return Err(ContinuityViolation::Position {
            tick,
            from: last.sub,
            to: curr.sub,
            delta: delta_pos,
            epsilon_pos: tol.epsilon_pos,
        });
    }
    // Quaternion angular distance: 2·acos(|dot|), clamped (a tiny float overshoot past 1.0 must not
    // NaN the acos). `abs()` folds the double-cover (q and −q are the same rotation).
    let delta_rot = 2.0 * last.orient.dot(curr.orient).abs().clamp(-1.0, 1.0).acos();
    if delta_rot > tol.epsilon_rot {
        return Err(ContinuityViolation::Rotation {
            tick,
            from: last.sub,
            to: curr.sub,
            delta: delta_rot,
            epsilon_rot: tol.epsilon_rot,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::{Fence, UniverseTick};
    use vd_sim::stub::DiscardReason;
    use vd_wire::seams::directory::OwnerRecord;

    const ORCH: NodeId = NodeId(1);
    const SHARD: NodeId = NodeId(2);
    const CLIENT: NodeId = NodeId(3);
    const DEST: NodeId = NodeId(4);
    const SESSION: SessionId = SessionId(7);

    fn entity() -> EntityId {
        EntityId(42)
    }

    fn record(node: NodeId) -> OwnerRecord {
        record_at(node, Fence(1))
    }

    fn record_at(node: NodeId, fence: Fence) -> OwnerRecord {
        OwnerRecord {
            authority: AuthorityRef::Shard(node),
            fence,
            lease_expires: UniverseTick(100),
            in_transfer: None,
        }
    }

    fn realm() -> vd_core::pose::RealmId {
        vd_core::pose::RealmId::System(7)
    }

    /// An orchestrator + shard agreeing on one realm at fence 3.
    fn realm_healthy() -> Vec<(NodeId, InspectReport)> {
        vec![
            (
                ORCH,
                InspectReport {
                    directory: vec![(DirectoryKey::Realm(realm()), record_at(SHARD, Fence(3)))],
                    ..InspectReport::default()
                },
            ),
            (
                SHARD,
                InspectReport {
                    held_realms: vec![(realm(), Fence(3))],
                    ..InspectReport::default()
                },
            ),
        ]
    }

    #[test]
    fn dead_node_exclusion_reaches_the_realm_half_orphan() {
        // P3 crash matrix: a killed SHARD's held REALM is a corpse too — excluded by the realm-half
        // dead filter (the entity half is empty here, so the check reaches verify_realm_authority with
        // a non-empty dead set). The realm record naming the dead shard surfaces as RealmHeldNowhere.
        let reports = realm_healthy(); // ORCH dir realm@SHARD, SHARD held_realms=[(realm, 3)], no entities.
        assert_eq!(
            verify_authority_unique(&reports),
            Ok(()),
            "live: the realm is consistently held"
        );
        let dead: BTreeSet<NodeId> = [SHARD].into_iter().collect();
        assert_eq!(
            verify_authority_unique_excluding(&reports, &dead),
            Err(AuthorityViolation::RealmHeldNowhere {
                realm: realm().to_string(),
                recorded: format!("{:?}", AuthorityRef::Shard(SHARD)),
            })
        );
    }

    #[test]
    fn realm_authority_unique_passes_and_catches_split_brain() {
        assert_eq!(verify_authority_unique(&realm_healthy()), Ok(()));
        // Two shards claiming the same realm.
        let mut reports = realm_healthy();
        reports.push((
            NodeId(9),
            InspectReport {
                held_realms: vec![(realm(), Fence(3))],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::RealmWrongHolderCount {
                realm: realm().to_string(),
                holders: vec![SHARD, NodeId(9)],
            })
        );
    }

    #[test]
    fn realm_fence_or_owner_disagreement_is_caught() {
        // Shard holds at a STALE fence (directory moved to 4).
        let mut reports = realm_healthy();
        reports[0].1.directory = vec![(DirectoryKey::Realm(realm()), record_at(SHARD, Fence(4)))];
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::RealmDisagrees {
                realm: realm().to_string(),
                holder: SHARD,
                held: Fence(3),
            })
        );
        // Shard holds a realm the directory records to a DIFFERENT owner.
        let mut reports = realm_healthy();
        reports[0].1.directory =
            vec![(DirectoryKey::Realm(realm()), record_at(NodeId(9), Fence(3)))];
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::RealmDisagrees {
                realm: realm().to_string(),
                holder: SHARD,
                held: Fence(3),
            })
        );
    }

    #[test]
    fn realm_in_flight_grant_is_excused_but_an_orphan_is_caught() {
        // Recorded for SHARD, pending at SHARD (grant confirmation in flight): legal.
        let mut reports = realm_healthy();
        reports[1].1.held_realms.clear();
        reports[1].1.pending_realms = vec![realm()];
        assert_eq!(verify_authority_unique(&reports), Ok(()));
        // Pending at a DIFFERENT node excuses nothing: genuine orphan.
        let mut reports = realm_healthy();
        reports[1].1.held_realms.clear();
        reports.push((
            NodeId(9),
            InspectReport {
                pending_realms: vec![realm()],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::RealmHeldNowhere {
                realm: realm().to_string(),
                recorded: format!("{:?}", AuthorityRef::Shard(SHARD)),
            })
        );
    }

    #[test]
    fn an_entity_held_at_a_stale_fence_is_a_split_brain() {
        // The directory moved the entity to fence 2 but the holder still claims 1.
        let mut reports = healthy();
        reports[0].1.directory = vec![(DirectoryKey::Entity(entity()), record_at(SHARD, Fence(2)))];
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::FenceMismatch {
                entity: entity(),
                held: Fence(1),
                recorded: Fence(2),
            })
        );
    }

    fn healthy() -> Vec<(NodeId, InspectReport)> {
        vec![
            (
                ORCH,
                InspectReport {
                    directory: vec![(DirectoryKey::Entity(entity()), record(SHARD))],
                    ..InspectReport::default()
                },
            ),
            (
                SHARD,
                InspectReport {
                    held_entities: vec![(entity(), Fence(1))],
                    applied_inputs: vec![(SESSION, 1), (SESSION, 2)],
                    discarded_inputs: vec![
                        (SESSION, Some(3), DiscardReason::DuplicateSeq),
                        // A seq-less discard (injected garbage): accounts for nothing.
                        (SESSION, None, DiscardReason::MalformedInput),
                    ],
                    ..InspectReport::default()
                },
            ),
            (
                CLIENT,
                InspectReport {
                    sent_inputs: vec![(SESSION, 1), (SESSION, 2), (SESSION, 3)],
                    ..InspectReport::default()
                },
            ),
        ]
    }

    #[test]
    fn healthy_topology_passes_both_oracles() {
        let reports = healthy();
        assert_eq!(verify_authority_unique(&reports), Ok(()));
        assert_eq!(verify_input_conservation(&reports), Ok(()));
    }

    #[test]
    fn split_brain_is_caught() {
        let mut reports = healthy();
        reports.push((
            NodeId(9),
            InspectReport {
                held_entities: vec![(entity(), Fence(1))],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::WrongHolderCount {
                entity: entity(),
                holders: vec![SHARD, NodeId(9)],
            })
        );
    }

    #[test]
    fn zero_owner_is_caught_in_both_directions() {
        // Held but unrecorded.
        let mut reports = healthy();
        reports[0].1.directory.clear();
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::Unrecorded {
                entity: entity(),
                holder: SHARD,
            })
        );
        // Recorded but held nowhere.
        let mut reports = healthy();
        reports[1].1.held_entities.clear();
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::HeldNowhere {
                entity: entity(),
                recorded: format!("{:?}", AuthorityRef::Shard(SHARD)),
            })
        );
    }

    #[test]
    fn in_flight_grants_are_legal_but_misdirected_or_settled_pending_is_not() {
        // Recorded in the directory, pending at the RECORDED owner: legal window.
        let mut reports = healthy();
        reports[1].1.held_entities.clear();
        reports[1].1.pending_entities = vec![entity()];
        assert_eq!(verify_authority_unique(&reports), Ok(()));
        // The settled check refuses the same state after quiesce.
        assert_eq!(
            verify_authority_settled(&reports),
            Err(AuthorityViolation::UnsettledPending {
                entity: entity(),
                node: SHARD,
            })
        );
        // Pending at a node that is NOT the recorded owner excuses nothing.
        let mut reports = healthy();
        reports[1].1.held_entities.clear();
        reports.push((
            NodeId(9),
            InspectReport {
                pending_entities: vec![entity()],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::HeldNowhere {
                entity: entity(),
                recorded: format!("{:?}", AuthorityRef::Shard(SHARD)),
            })
        );
        // And the settled check passes a fully-held topology.
        assert_eq!(verify_authority_settled(&healthy()), Ok(()));
    }

    #[test]
    fn the_release_window_is_legal_until_settled() {
        // Directory already cleared (revoke committed), holder still finishing:
        // held + departing at the SAME node, no record — legal in flight.
        let mut reports = healthy();
        reports[0].1.directory.clear();
        reports[1].1.departing_entities = vec![entity()];
        assert_eq!(verify_authority_unique(&reports), Ok(()));
        // But it may not survive the settle.
        assert_eq!(
            verify_authority_settled(&reports),
            Err(AuthorityViolation::UnsettledPending {
                entity: entity(),
                node: SHARD,
            })
        );
        // Departing at a DIFFERENT node excuses nothing.
        let mut reports = healthy();
        reports[0].1.directory.clear();
        reports.push((
            NodeId(9),
            InspectReport {
                departing_entities: vec![entity()],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::Unrecorded {
                entity: entity(),
                holder: SHARD,
            })
        );
    }

    /// The W1 mid-flight transfer window: SHARD (the saga SOURCE) still holds the subject Owned at
    /// the OLD fence (1) while the directory already records the DEST at the NEW fence (2), with a
    /// live saga SHARD→DEST in flight. The base case is EXCUSED (Ok) — the legal post-CAS, pre-demote
    /// state the per-tick oracle must not RED on.
    fn w1_window() -> Vec<(NodeId, InspectReport)> {
        vec![
            (
                ORCH,
                InspectReport {
                    directory: vec![(DirectoryKey::Entity(entity()), record_at(DEST, Fence(2)))],
                    active_transfers: vec![ActiveTransfer {
                        subject: DirectoryKey::Entity(entity()),
                        source: SHARD,
                        dest: DEST,
                    }],
                    ..InspectReport::default()
                },
            ),
            (
                SHARD,
                InspectReport {
                    held_entities: vec![(entity(), Fence(1))],
                    ..InspectReport::default()
                },
            ),
        ]
    }

    #[test]
    fn dead_node_exclusion_turns_a_corpse_holder_into_the_honest_orphan() {
        // P3 crash matrix: a held entity at a node the caller knows is DEAD is a corpse — excluded,
        // so the directory record naming it surfaces as the honest HeldNowhere (not a false-pass).
        let reports = healthy(); // SHARD holds entity, ORCH directory records SHARD.
        // Empty dead set ⇒ byte-identical to the plain oracle (the holder is live, all agrees).
        assert_eq!(
            verify_authority_unique_excluding(&reports, &BTreeSet::new()),
            Ok(())
        );
        assert_eq!(verify_authority_unique(&reports), Ok(()));
        // SHARD dead ⇒ its held claim is a corpse; the directory still records SHARD ⇒ HeldNowhere.
        let dead: BTreeSet<NodeId> = [SHARD].into_iter().collect();
        assert_eq!(
            verify_authority_unique_excluding(&reports, &dead),
            Err(AuthorityViolation::HeldNowhere {
                entity: entity(),
                recorded: format!("{:?}", AuthorityRef::Shard(SHARD)),
            })
        );
    }

    #[test]
    fn a_dead_w1_source_is_excluded_before_the_transfer_window_excuse() {
        // P3 crash matrix (the load-bearing dead-aware distinction): the W1 window (source still
        // Owned@old, directory records DEST, a live saga) is normally EXCUSED — but if the source is
        // DEAD, the excuse must NOT mask it. The dead source is excluded BEFORE the W1 logic, so the
        // honest orphan (the recorded dest holds nothing, the source is a corpse) surfaces.
        let reports = w1_window(); // SHARD held@1, ORCH dir DEST@2, live saga SHARD→DEST.
        // Live source ⇒ the legal window is excused.
        assert_eq!(verify_authority_unique(&reports), Ok(()));
        // Dead source ⇒ NOT excused: HeldNowhere (directory DEST, no live holder, dest not pending).
        let dead: BTreeSet<NodeId> = [SHARD].into_iter().collect();
        assert_eq!(
            verify_authority_unique_excluding(&reports, &dead),
            Err(AuthorityViolation::HeldNowhere {
                entity: entity(),
                recorded: format!("{:?}", AuthorityRef::Shard(DEST)),
            })
        );
    }

    #[test]
    fn w1_excuses_the_legal_window_but_each_conjunct_failing_re_reds() {
        // (TRUE) the legal mid-flight window is excused.
        assert_eq!(verify_authority_unique(&w1_window()), Ok(()));

        let disagree = AuthorityViolation::DirectoryDisagrees {
            entity: entity(),
            holder: SHARD,
            recorded: format!("{:?}", AuthorityRef::Shard(DEST)),
        };

        // (a FALSE) NO live saga for the entity → not excused.
        let mut r = w1_window();
        r[0].1.active_transfers.clear();
        assert_eq!(verify_authority_unique(&r), Err(disagree.clone()));

        // (b FALSE) the saga's SOURCE is not the holder → not excused.
        let mut r = w1_window();
        r[0].1.active_transfers[0].source = NodeId(99);
        assert_eq!(verify_authority_unique(&r), Err(disagree.clone()));

        // (c FALSE) the directory records someone OTHER than the saga's dest → not excused.
        let mut r = w1_window();
        r[0].1.directory = vec![(
            DirectoryKey::Entity(entity()),
            record_at(NodeId(77), Fence(2)),
        )];
        assert_eq!(
            verify_authority_unique(&r),
            Err(AuthorityViolation::DirectoryDisagrees {
                entity: entity(),
                holder: SHARD,
                recorded: format!("{:?}", AuthorityRef::Shard(NodeId(77))),
            })
        );

        // (d FALSE) the held fence is NOT stale vs the record (equal fences) → not excused.
        let mut r = w1_window();
        r[0].1.directory = vec![(DirectoryKey::Entity(entity()), record_at(DEST, Fence(1)))];
        assert_eq!(verify_authority_unique(&r), Err(disagree));
    }

    #[test]
    fn a_split_brain_during_a_live_saga_is_still_caught_never_excused() {
        // MASKING-SAFETY: two shards hold the entity WHILE a saga is live — the `len == 1` guard fires
        // BEFORE W1, so the split-brain is RED. The excuse can never swallow a real double-hold.
        let mut r = w1_window();
        r.push((
            NodeId(5),
            InspectReport {
                held_entities: vec![(entity(), Fence(2))],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_authority_unique(&r),
            Err(AuthorityViolation::WrongHolderCount {
                entity: entity(),
                holders: vec![SHARD, NodeId(5)],
            })
        );
    }

    #[test]
    fn a_non_entity_saga_subject_does_not_excuse_an_entity_disagreement() {
        // A live saga whose subject is a REALM (not the entity) must NOT excuse the entity's
        // disagreement — covers the non-Entity build arm + the no-entry excuse arm.
        let mut r = w1_window();
        r[0].1.active_transfers = vec![ActiveTransfer {
            subject: DirectoryKey::Realm(realm()),
            source: SHARD,
            dest: DEST,
        }];
        assert_eq!(
            verify_authority_unique(&r),
            Err(AuthorityViolation::DirectoryDisagrees {
                entity: entity(),
                holder: SHARD,
                recorded: format!("{:?}", AuthorityRef::Shard(DEST)),
            })
        );
    }

    #[test]
    fn a_live_saga_post_quiesce_is_unsettled() {
        // The W1 excuse is MID-FLIGHT only: verify_authority_settled rejects ANY in-flight saga (no
        // pending/departing here, but a live transfer means the run has NOT settled).
        let reports = vec![(
            ORCH,
            InspectReport {
                active_transfers: vec![ActiveTransfer {
                    subject: DirectoryKey::Entity(entity()),
                    source: SHARD,
                    dest: DEST,
                }],
                ..InspectReport::default()
            },
        )];
        assert_eq!(
            verify_authority_settled(&reports),
            Err(AuthorityViolation::UnsettledTransfer {
                subject: format!("{:?}", DirectoryKey::Entity(entity())),
                node: ORCH,
            })
        );
    }

    #[test]
    fn directory_disagreement_is_caught() {
        let mut reports = healthy();
        reports[0].1.directory = vec![(DirectoryKey::Entity(entity()), record(NodeId(9)))];
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::DirectoryDisagrees {
                entity: entity(),
                holder: SHARD,
                recorded: format!("{:?}", AuthorityRef::Shard(NodeId(9))),
            })
        );
        // A gateway-recorded ENTITY is equally a disagreement for a shard holder.
        let mut reports = healthy();
        reports[0].1.directory = vec![(
            DirectoryKey::Entity(entity()),
            OwnerRecord {
                authority: AuthorityRef::Gateway(NodeId(9)),
                ..record(NodeId(9))
            },
        )];
        let result = verify_authority_unique(&reports);
        assert_eq!(
            result,
            Err(AuthorityViolation::DirectoryDisagrees {
                entity: entity(),
                holder: SHARD,
                recorded: format!("{:?}", AuthorityRef::Gateway(NodeId(9))),
            })
        );
    }

    #[test]
    fn non_entity_directory_keys_are_ignored_by_authority_unique() {
        let mut reports = healthy();
        reports[0].1.directory.push((
            DirectoryKey::Session(SESSION),
            OwnerRecord {
                authority: AuthorityRef::Gateway(NodeId(5)),
                ..record(NodeId(5))
            },
        ));
        assert_eq!(verify_authority_unique(&reports), Ok(()));
    }

    #[test]
    fn double_apply_and_phantom_and_unaccounted_are_caught() {
        // Applied twice across two shards.
        let mut reports = healthy();
        reports.push((
            NodeId(9),
            InspectReport {
                applied_inputs: vec![(SESSION, 2)],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_input_conservation(&reports),
            Err(ConservationViolation::AppliedTwice {
                session: SESSION,
                seq: 2,
                count: 2,
            })
        );
        // Phantom: applied but never sent.
        let mut reports = healthy();
        reports[2].1.sent_inputs = vec![(SESSION, 1), (SESSION, 3)];
        assert_eq!(
            verify_input_conservation(&reports),
            Err(ConservationViolation::Phantom {
                session: SESSION,
                seq: 2,
            })
        );
        // Unaccounted: sent but neither applied nor seq-discarded.
        let mut reports = healthy();
        reports[2].1.sent_inputs.push((SESSION, 4));
        assert_eq!(
            verify_input_conservation(&reports),
            Err(ConservationViolation::Unaccounted {
                session: SESSION,
                seq: 4,
            })
        );
    }

    #[test]
    fn non_monotonic_application_is_caught() {
        let mut reports = healthy();
        reports[1].1.applied_inputs = vec![(SESSION, 2), (SESSION, 1)];
        reports[2].1.sent_inputs = vec![(SESSION, 1), (SESSION, 2), (SESSION, 3)];
        reports[1].1.discarded_inputs = vec![(SESSION, Some(3), DiscardReason::DuplicateSeq)];
        assert_eq!(
            verify_input_conservation(&reports),
            Err(ConservationViolation::NonMonotonicApply {
                session: SESSION,
                seqs: vec![2, 1],
            })
        );
    }

    #[test]
    fn violations_display_for_failure_messages() {
        assert_eq!(
            AuthorityViolation::Unrecorded {
                entity: entity(),
                holder: SHARD,
            }
            .to_string(),
            format!(
                "entity {} held by {} has NO directory record (zero-owner)",
                entity(),
                SHARD
            )
        );
        assert_eq!(
            AuthorityViolation::WrongHolderCount {
                entity: entity(),
                holders: vec![SHARD, NodeId(9)],
            }
            .to_string(),
            format!(
                "entity {} is held by [NodeId(2), NodeId(9)] — exactly one holder required",
                entity()
            )
        );
        assert_eq!(
            AuthorityViolation::DirectoryDisagrees {
                entity: entity(),
                holder: SHARD,
                recorded: "X".to_owned(),
            }
            .to_string(),
            format!(
                "entity {} held by {} but the directory records X",
                entity(),
                SHARD
            )
        );
        assert_eq!(
            AuthorityViolation::HeldNowhere {
                entity: entity(),
                recorded: "X".to_owned(),
            }
            .to_string(),
            format!(
                "directory entity {} (owner X) is held by no live node",
                entity()
            )
        );
        assert_eq!(
            ConservationViolation::AppliedTwice {
                session: SESSION,
                seq: 2,
                count: 3,
            }
            .to_string(),
            format!("input (session {SESSION}, seq 2) applied 3 times — exactly once allowed")
        );
        assert_eq!(
            ConservationViolation::NonMonotonicApply {
                session: SESSION,
                seqs: vec![2, 1],
            }
            .to_string(),
            format!("applied seqs for session {SESSION} are not strictly increasing: [2, 1]")
        );
        assert_eq!(
            ConservationViolation::Phantom {
                session: SESSION,
                seq: 2,
            }
            .to_string(),
            format!("applied input (session {SESSION}, seq 2) was never sent by any client")
        );
        assert_eq!(
            AuthorityViolation::UnsettledPending {
                entity: entity(),
                node: SHARD,
            }
            .to_string(),
            format!(
                "entity {} is still pending at {} after the run settled",
                entity(),
                SHARD
            )
        );
        assert_eq!(
            ConservationViolation::Unaccounted {
                session: SESSION,
                seq: 4,
            }
            .to_string(),
            format!(
                "sent input (session {SESSION}, seq 4) is unaccounted: neither applied \
                 nor discarded-with-reason"
            )
        );
    }

    // ===================== THE RENDER-TRACE ORACLE META-TESTS (5b negative control) ==============
    // The oracle ships WITH its falsifier (test_harness.md §5b/§8.1): hand-built traces that FAIL
    // (a vanish, a teleport) prove the oracle catches the divergence class, and a frame-eval
    // divergence proves it reads `world_pos`, not the raw frame-local pos.

    const SRC: SubId = SubId(0);
    const DST: SubId = SubId(1);

    fn sys7() -> FrameRef {
        FrameRef::SystemSpace { system_seed: 7 }
    }

    /// A rendered sample whose world_pos == raw_pos (a world frame), no rotation.
    fn sample(sub: SubId, x: f64) -> RenderSample {
        RenderSample {
            sub,
            frame: sys7(),
            raw_pos: DVec3::new(x, 0.0, 0.0),
            world_pos: DVec3::new(x, 0.0, 0.0),
            orient: DQuat::IDENTITY,
        }
    }

    /// The zero-fault lockstep tolerances actually used by the capstone: move_speed 2 m/s, dt 0.05 s,
    /// lockstep (stagger 0), in-step delivery (delay 0), a generous band, derived slacks.
    fn lockstep_tol() -> RenderTolerances {
        RenderTolerances::derive(
            2.0,  // max_velocity_mps (StubConfig.move_speed_mps)
            0.05, // tick_dt_s
            0,    // stagger_offset_ticks (lockstep)
            0,    // max_fabric_delay_ticks (in-step ordering chain)
            8,    // band_ticks
            1e-9, // pos_slack_m (f64 round-trip)
            0.0,  // look_rate_rad_per_s (avatar does not turn during the transfer)
            1e-9, // rot_slack_rad
        )
    }

    #[test]
    fn tolerances_derive_from_params_with_no_magic_and_saturate_safely() {
        let tol = lockstep_tol();
        // K = (band − delay) − 1 = (8 − 0) − 1 = 7 with a band of 8; ε_pos = 2·0.05·1 + 1e-9.
        assert_eq!(tol.max_absent_ticks, 7);
        assert!((tol.epsilon_pos - (0.1 + 1e-9)).abs() < 1e-15);
        assert!((tol.epsilon_rot - 1e-9).abs() < 1e-15);
        // The capstone's tight gate: a band of exactly 1 with delay 0 ⇒ K = 0 (present every tick).
        let tight = RenderTolerances::derive(2.0, 0.05, 0, 0, 1, 1e-9, 0.0, 1e-9);
        assert_eq!(tight.max_absent_ticks, 0);
        // SATURATION: a band SHORTER than the delay floors K at 0 (never wraps). Stagger widens ε_pos.
        let starved = RenderTolerances::derive(2.0, 0.05, 1, 5, 2, 1e-9, 1.0, 1e-9);
        assert_eq!(
            starved.max_absent_ticks, 0,
            "band(2) − delay(5) saturates to 0, then −1 to 0"
        );
        assert!(
            (starved.epsilon_pos - (2.0 * 0.05 * 2.0 + 1e-9)).abs() < 1e-15,
            "stagger 1 widens ε_pos by the (1 + stagger) factor"
        );
        assert!(
            (starved.epsilon_rot - (1.0 * 0.05 + 1e-9)).abs() < 1e-15,
            "a non-zero look rate contributes to ε_rot"
        );
    }

    #[test]
    fn no_vanish_passes_a_continuous_trace_and_catches_a_one_tick_vanish() {
        // (a) PASS: present every tick across an overlap (the source then the dest sub).
        let good: RenderTrace = vec![
            (TickId(1), Some(sample(SRC, 0.0))),
            (TickId(2), Some(sample(SRC, 0.1))),
            (TickId(3), Some(sample(DST, 0.2))),
            (TickId(4), Some(sample(DST, 0.3))),
        ];
        // Tight K=0 gate (the capstone's): still passes (no absent tick at all).
        let tight = RenderTolerances::derive(2.0, 0.05, 0, 0, 1, 1e-9, 0.0, 1e-9);
        assert_eq!(verify_no_vanish(&good, tight), Ok(()));

        // (b) NO-VANISH RED: one absent tick in the middle, against K=0 ⇒ a vanish.
        let vanish: RenderTrace = vec![
            (TickId(1), Some(sample(SRC, 0.0))),
            (TickId(2), None), // the avatar rendered NOWHERE this tick
            (TickId(3), Some(sample(DST, 0.2))),
        ];
        assert_eq!(
            verify_no_vanish(&vanish, tight),
            Err(VanishViolation {
                last_absent: TickId(2),
                absent_run: 1,
                max_absent_ticks: 0,
            })
        );

        // (c) A SHORT absent run UNDER a relaxed K passes (proves the K bound, not absence==fail):
        // one absent tick against the band-8 K=7 is tolerated, and a later present tick resets the
        // run (exercises the Some-reset arm after a None).
        let blip: RenderTrace = vec![
            (TickId(1), Some(sample(SRC, 0.0))),
            (TickId(2), None),
            (TickId(3), Some(sample(DST, 0.2))),
        ];
        assert_eq!(verify_no_vanish(&blip, lockstep_tol()), Ok(()));
    }

    #[test]
    fn pose_continuity_passes_a_smooth_flip_and_catches_a_teleport() {
        // (a) PASS: a source→dest flip where the world pose moves < ε_pos (one tick of walk).
        let smooth: RenderTrace = vec![
            (TickId(1), Some(sample(SRC, 0.00))),
            (TickId(2), Some(sample(SRC, 0.05))),
            (TickId(3), Some(sample(DST, 0.09))), // Δ 0.04 m at the flip < ε_pos 0.1
            (TickId(4), Some(sample(DST, 0.13))),
        ];
        assert_eq!(verify_pose_continuity(&smooth, lockstep_tol()), Ok(()));

        // (b) POSE-CONTINUITY RED (position): the dest's first pose is a teleport away.
        let teleport: RenderTrace = vec![
            (TickId(1), Some(sample(SRC, 0.0))),
            (TickId(2), Some(sample(DST, 100.0))), // Δ 100 m at the flip ≫ ε_pos
        ];
        assert_eq!(
            verify_pose_continuity(&teleport, lockstep_tol()),
            Err(ContinuityViolation::Position {
                tick: TickId(2),
                from: SRC,
                to: DST,
                delta: 100.0,
                epsilon_pos: lockstep_tol().epsilon_pos,
            })
        );

        // (c) POSE-CONTINUITY RED (rotation): same world pos, but a 180° flip in orientation, against
        // the zero-look-rate ε_rot ⇒ a rotation teleport.
        use std::f64::consts::PI;
        let mut spun = sample(DST, 0.0);
        spun.orient = DQuat::from_rotation_y(PI);
        let rot: RenderTrace = vec![(TickId(1), Some(sample(SRC, 0.0))), (TickId(2), Some(spun))];
        assert_eq!(
            verify_pose_continuity(&rot, lockstep_tol()),
            Err(ContinuityViolation::Rotation {
                tick: TickId(2),
                from: SRC,
                to: DST,
                delta: PI,
                epsilon_rot: lockstep_tol().epsilon_rot,
            })
        );
    }

    #[test]
    fn pose_continuity_skips_absent_ticks_and_same_sub_pairs() {
        // An absent tick (None) between two rendered samples is SKIPPED (NO-VANISH owns it), and
        // adjacent SAME-sub samples are NOT a flip — a big same-sub jump (the dest's own input
        // integration) must NOT trip continuity (only the A→B sub change does). This exercises the
        // `continue` arm, the `prev == None` first-sample arm, and the same-sub no-check arm.
        let trace: RenderTrace = vec![
            (TickId(1), Some(sample(SRC, 0.0))),
            (TickId(2), None),                     // skipped
            (TickId(3), Some(sample(SRC, 50.0))),  // same sub, huge jump — NOT checked
            (TickId(4), Some(sample(DST, 50.04))), // the only flip: Δ 0.04 < ε
        ];
        assert_eq!(verify_pose_continuity(&trace, lockstep_tol()), Ok(()));
    }

    #[test]
    fn the_oracle_reads_world_pos_not_the_raw_frame_local_pos() {
        // THE C3 FRAME-EVAL DIVERGENCE control: a `ShipLocal` interior whose raw frame-local pos and
        // its world_pos DIVERGE. If the oracle (wrongly) compared `raw_pos`, the two cases below
        // would give the OPPOSITE verdict. Proving it reads `world_pos` catches the wrong comparator
        // NOW (P1.5), not at P8 when ship interiors land.
        let interior = FrameRef::ShipLocal {
            ship: EntityId(900),
        };

        // (a) raw_pos CONTINUOUS but world_pos a TELEPORT (the hull jumped) ⇒ must FAIL on world.
        let last = RenderSample {
            sub: SRC,
            frame: interior,
            raw_pos: DVec3::new(1.0, 0.0, 0.0), // same local seat both ticks
            world_pos: DVec3::new(1.0, 0.0, 0.0),
            orient: DQuat::IDENTITY,
        };
        let curr = RenderSample {
            sub: DST,
            frame: interior,
            raw_pos: DVec3::new(1.0, 0.0, 0.0), // raw is UNCHANGED — a raw comparator would PASS
            world_pos: DVec3::new(80.0, 0.0, 0.0), // world TELEPORTED (hull moved 79 m)
            orient: DQuat::IDENTITY,
        };
        let raw_continuous: RenderTrace = vec![(TickId(1), Some(last)), (TickId(2), Some(curr))];
        assert_eq!(
            verify_pose_continuity(&raw_continuous, lockstep_tol()),
            Err(ContinuityViolation::Position {
                tick: TickId(2),
                from: SRC,
                to: DST,
                delta: 79.0,
                epsilon_pos: lockstep_tol().epsilon_pos,
            }),
            "the oracle compared world_pos (Δ79 m), NOT the unchanged raw_pos"
        );

        // (b) the INVERSE: raw_pos a TELEPORT but world_pos CONTINUOUS (the interior offset changed
        // but the hull moved to cancel it) ⇒ must PASS on world. A raw comparator would FAIL here.
        let last2 = RenderSample {
            sub: SRC,
            frame: interior,
            raw_pos: DVec3::new(0.0, 0.0, 0.0),
            world_pos: DVec3::new(10.0, 0.0, 0.0),
            orient: DQuat::IDENTITY,
        };
        let curr2 = RenderSample {
            sub: DST,
            frame: interior,
            raw_pos: DVec3::new(60.0, 0.0, 0.0), // raw JUMPED 60 m — a raw comparator would FAIL
            world_pos: DVec3::new(10.04, 0.0, 0.0), // world moved 0.04 m < ε
            orient: DQuat::IDENTITY,
        };
        let world_continuous: RenderTrace =
            vec![(TickId(1), Some(last2)), (TickId(2), Some(curr2))];
        assert_eq!(
            verify_pose_continuity(&world_continuous, lockstep_tol()),
            Ok(()),
            "the oracle passed on the continuous world_pos despite the raw_pos teleport"
        );
    }

    #[test]
    fn render_oracle_violations_display_for_failure_messages() {
        assert_eq!(
            VanishViolation {
                last_absent: TickId(5),
                absent_run: 3,
                max_absent_ticks: 1,
            }
            .to_string(),
            "NO-VANISH: subject absent from the render for 3 consecutive ticks ending at \
             tick-5 — exceeds K=1"
        );
        assert_eq!(
            ContinuityViolation::Position {
                tick: TickId(2),
                from: SRC,
                to: DST,
                delta: 5.0,
                epsilon_pos: 0.1,
            }
            .to_string(),
            "POSE-CONTINUITY: world position jumped 5 m at the SubId(0)→SubId(1) sub flip \
             (tick-2) — exceeds ε_pos=0.1"
        );
        assert_eq!(
            ContinuityViolation::Rotation {
                tick: TickId(3),
                from: SRC,
                to: DST,
                delta: 3.0,
                epsilon_rot: 0.01,
            }
            .to_string(),
            "POSE-CONTINUITY: orientation jumped 3 rad at the SubId(0)→SubId(1) sub flip \
             (tick-3) — exceeds ε_rot=0.01"
        );
    }
}
