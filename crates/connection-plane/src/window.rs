//! THE WINDOW LANE's composition engine (`docs/design/window_lane.md` §2.6; owner-approved
//! 2026-08-15/16, five-topic walk §4.5). The gateway stacks each observer's chain of attested
//! window statements at ONE universe tick and produces the composed picture.
//!
//! It landed in Slice B measured against the old inter-realm lanes and shipped to nobody; the
//! Slice-C1 flag day made its output the client's scene feed; Slice C2 deleted the lanes it was
//! measured against, and the shadow-parity comparator retired with them (its measurement ran and
//! is recorded in D-WINDOW-1). This is now the ONLY place a picture is composed.
//!
//! STRUCTURAL GUARDS (§2.6.1), where each lives:
//! 1. Dependency gate — this crate carries NO normal `vd-physics` edge (pinned by
//!    `tests/tests/crate_isolation.rs::the_window_composer_cannot_name_a_motion`): the composer
//!    is structurally unable to evaluate a placement or generate a world. Everything it folds is
//!    an attested statement from the statement's lawful author.
//! 2. Provenance gate — the composer consumes ONLY rows the Slice-A ingest admitted
//!    (`on_window_row`: known window, roster-head sender, admissible body). No other write path
//!    into [`WindowIngest`] exists.
//! 3. No shard-bound sender — every function here takes rows and returns composed values; no
//!    outbox, no `InterShardFlow`, no handle that could produce one.
//! 4. Zero-state — all state lives on the gateway's windows ([`WindowIngest`] per open window)
//!    and sessions ([`ShadowScene`] per session); zero sessions close every window and drop every
//!    scene, so nothing global ever accumulates (the teardown test asserts it).
//! 5. AoI separation — the demand/AoI fold has no data path in here: membership arrives as the
//!    PARENT's shipped verdict ([`WindowIngest::members`]) and is never re-derived.
//!
//! THE SHEAR LAW (§2.6.3) is enforced by the type system this workspace already proves: every
//! fold runs through `vd_core::frame::transfer_frame` with per-hop [`PlacementBook`]s built at
//! exactly the common tick, so a mixed-tick fold is refused by `FrameError::InstantMismatch` —
//! never silently blended. The G-SHEAR unit half below includes the DELIBERATE mixed-tick compose
//! that MUST fail (anti-vacuity).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use vd_core::Fence;
use vd_core::frame::{FrameError, FramePlacement, transfer_frame};
use vd_core::ids::UniverseTick;
use vd_core::placement::PlacementBook;
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_wire::channels::{RealmSnap, SceneRow};
use vd_wire::session_flow::{BodyStmt, HopRow, WindowId, WindowScope};

/// The ONE config home for every derived bound the composer uses (§2.6.7: "One `WindowTuning`
/// struct holds every derived bound — no magic numbers"). Both bounds are the owner-law 3(a)
/// derivation — at least two cadence beats plus one tick, never a free literal — off the SAME
/// keep-alive beat the gateway already re-asserts windows on (`window_keepalive_cadence`), so the
/// subscriber, the holder's TTL and the composer's retention all breathe on one rhythm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowTuning {
    /// How many ticks behind its newest stamp a window's level ring retains (ring K = two
    /// beats + one): wide enough that one whole lost keep-alive of cross-shard arrival skew
    /// still leaves a common tick to fold at.
    pub ring_span_ticks: u64,
    /// How long a held stratum may hold before the dead-hop exit removes it (§2.6.4): the same
    /// 2-beats-+-1 confirmation window after which the shard side would have TTL-expired the fan.
    pub hold_ttl_ticks: u64,
    /// THE ROSTER-LOSS CONFIRMATION WINDOW (§2.8's departure mirror): how long a SELF-LOOK — or a
    /// relayed interior level — may sit un-re-asserted before it is pruned and the parent's
    /// ever-present marker resumes. Same 2-beats-+-1 law, off the same beat, and sound because the
    /// keep-alive re-assert is what makes a statement arrive again: a shard re-served its whole
    /// body/relay set on every keep-alive `WindowOpen` (`vd_sim` `on_window_open`), so a statement
    /// missing for two whole beats means the realm behind it stopped speaking — it tore down, or
    /// its relay holder TTL-expired.
    ///
    /// MARKERS ARE DELIBERATELY EXEMPT. A marker is the presence gate's FLOOR ("never zero"): it
    /// is the parent's own datum about a sleeping child, and expiring it would blank a star rather
    /// than shrink a system to a dot.
    pub look_ttl_ticks: u64,
}

impl WindowTuning {
    /// Derive every bound from the one keep-alive beat (ticks). `beat` is already itself derived
    /// (`session_recheck_interval`, else tick-rate/2) — this only applies the 2-beats-+-1 law.
    #[must_use]
    pub fn derive(beat_ticks: u64) -> WindowTuning {
        let two_beats_one = 2 * beat_ticks + 1;
        WindowTuning {
            ring_span_ticks: two_beats_one,
            hold_ttl_ticks: two_beats_one,
            look_ttl_ticks: two_beats_one,
        }
    }
}

/// ONE per-tick unit of one window level, decoded once (§2.6.2 `WindowLevel`): the stamp, the
/// pre-inverted hop (Child-scope windows only) and the author's typed child rows.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowLevel {
    /// The one universe-tick stamp for everything inside (intra-level same-tickness is by
    /// construction — one message, one `at`).
    pub at: UniverseTick,
    /// "The author's own frame expressed in the hop child's frame at `at`" — pre-inverted BY the
    /// author (SL1's conversion-in-the-parent, hop by hop). `None` on an own-level window.
    pub hop: Option<HopRow>,
    /// The author's FULL direct-child roster at `at`, in the author's own frame.
    pub rows: Vec<RealmSnap>,
}

/// Why an ingested statement was refused (returned to the caller so every refusal is COUNTED,
/// never silent — the failure-mode table of §2.6.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ingested {
    /// Stored (a level newer than, equal to — latest-wins re-delivery — or inside the ring).
    Applied,
    /// A frame stamped further behind the ring head than the derived span: refused
    /// (`window_level_refused`) — composing that far back would fold a past nobody else retains.
    BehindRing,
}

/// Per-window ingest state (§2.6.2 `WindowState`'s data half): the level ring, the latest bodies
/// per subject, and the parent's shipped membership verdict. Held INSIDE the gateway's per-window
/// record, so closing the window drops it — zero windows, zero composer state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WindowIngest {
    /// The level ring, ascending by stamp; retention = [`WindowTuning::ring_span_ticks`] behind
    /// the newest (K = 2 beats + 1, derived).
    levels: VecDeque<WindowLevel>,
    /// Latest `SelfLook` per subject (newest `authored_at` wins; only ever self-shipped — the
    /// Slice-A admission already refused everything else).
    look_of: BTreeMap<RealmId, (UniverseTick, Vec<u8>)>,
    /// Latest `Marker` per direct child (same newest-wins).
    marker_of: BTreeMap<RealmId, (UniverseTick, Vec<u8>)>,
    /// The parent's SL7 membership verdict, ids only (gates BODIES/interiors in the client cut —
    /// stored now, read by Slice C1; the composed placement rows are deliberately NOT filtered
    /// by it: markers/placements always ship the full roster, §2.2).
    members: BTreeSet<RealmId>,
    /// The Q2 relay's per-child holdings on the PARENT's window (mesh minor 17; owner-approved
    /// 2026-08-16 — owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS): the
    /// newest relayed interior LEVEL per live child (the child's own authored rows in its OWN
    /// frame — also the attested roster its relayed markers are vouched against). Stored for the
    /// Slice-D interior compose; the relayed BODY statements land in `look_of`/`marker_of` like
    /// every other admitted body (the §2.8 marker⇒look handover needs nothing new downstream).
    relay_levels: BTreeMap<RealmId, (UniverseTick, Vec<RealmSnap>)>,
    /// The newest CHILD fence seen per relayed child — the zombie guard (a deposed incarnation's
    /// relay is refused; carried, never authorizing).
    relay_fence: BTreeMap<RealmId, Fence>,
}

impl WindowIngest {
    /// Ingest one attested `WindowFrame` level. Latest-wins per tick (a re-delivered stamp
    /// replaces); an out-of-order tick still inside the ring is inserted in stamp order (the
    /// common-tick fold wants it); a tick behind the ring is refused.
    pub fn ingest_frame(&mut self, level: WindowLevel, tuning: &WindowTuning) -> Ingested {
        let head = self.levels.back().map(|l| l.at);
        let Some(head) = head else {
            self.levels.push_back(level);
            return Ingested::Applied;
        };
        if level.at.0 + tuning.ring_span_ticks < head.0 {
            return Ingested::BehindRing;
        }
        match self.levels.binary_search_by_key(&level.at, |l| l.at) {
            Ok(i) => self.levels[i] = level,
            Err(i) => self.levels.insert(i, level),
        }
        let new_head = self.levels.back().expect("a level was just stored").at.0;
        while self
            .levels
            .front()
            .is_some_and(|l| l.at.0 + tuning.ring_span_ticks < new_head)
        {
            self.levels.pop_front();
        }
        Ingested::Applied
    }

    /// Ingest one attested `WindowBody`. Newest `authored_at` wins; an older statement is
    /// refused (returned `false` so the caller counts `window_body_stale`).
    pub fn ingest_body(
        &mut self,
        subject: RealmId,
        stmt: &BodyStmt,
        authored_at: UniverseTick,
    ) -> bool {
        let slot = match stmt {
            BodyStmt::SelfLook { .. } => &mut self.look_of,
            BodyStmt::Marker { .. } => &mut self.marker_of,
        };
        if slot.get(&subject).is_some_and(|(at, _)| *at > authored_at) {
            return false;
        }
        let bag = match stmt {
            BodyStmt::SelfLook { bag } => bag.clone(),
            BodyStmt::Marker { luma } => luma.clone(),
        };
        slot.insert(subject, (authored_at, bag));
        true
    }

    /// THE ROSTER-DRIVEN LOOK PRUNE (`docs/design/window_lane.md` §2.8's departure mirror, Slice D)
    /// — the handover run backwards. A realm that tore down stops stating its own look: the child
    /// stops relaying, the parent's holder TTL-expires it, and nothing re-serves it on the next
    /// keep-alive re-assert. Two whole beats later (+1 tick) its stored look is dropped here, the
    /// presence gate falls through to its parent's ever-present marker, and the system it drew
    /// shrinks to a dot. Relayed interior LEVELS age out on the same window for the same reason.
    ///
    /// This is what keeps THE DRAW LAW honest across a round trip: without it, a realm's last look
    /// would sit in this store forever and the NEXT approach would draw the realm before its shard
    /// was running again — a body with no author, and a vacuous wake handover.
    ///
    /// Markers are never pruned (the floor — see [`WindowTuning::look_ttl_ticks`]). Returns
    /// `(looks pruned, relayed levels pruned)` for the two counters.
    pub fn prune_stale(&mut self, now: UniverseTick, tuning: &WindowTuning) -> (u64, u64) {
        let cutoff = now.0.saturating_sub(tuning.look_ttl_ticks);
        let looks = prune_older_than(&mut self.look_of, cutoff);
        let relays = prune_older_than(&mut self.relay_levels, cutoff);
        (looks, relays)
    }

    /// Apply one attested `WindowMembership` diff onto the held verdict.
    pub fn ingest_membership(&mut self, added: &[RealmId], removed: &[RealmId]) {
        for r in removed {
            self.members.remove(r);
        }
        for r in added {
            self.members.insert(*r);
        }
    }

    /// The level stamped exactly `at`, if the ring retains it.
    #[must_use]
    pub fn level_at(&self, at: UniverseTick) -> Option<&WindowLevel> {
        self.levels
            .binary_search_by_key(&at, |l| l.at)
            .ok()
            .and_then(|i| self.levels.get(i))
    }

    /// The newest retained stamp (`None` before the first frame).
    #[must_use]
    pub fn newest(&self) -> Option<UniverseTick> {
        self.levels.back().map(|l| l.at)
    }

    /// Has this window received at least one attested level? (A `Child(c)` window with a level
    /// IS the author's attested claim `author = parent(c)` — the chain extends only through
    /// confirmed hops, §2.6.2.)
    #[must_use]
    pub fn confirmed(&self) -> bool {
        !self.levels.is_empty()
    }

    /// Does the author's newest roster name `realm` as a direct child? THE stream-only child-set
    /// source for the marker admission (§2.6.2 deletes the seed-forest second source): the
    /// author's own attested full-roster rows, nothing else.
    #[must_use]
    pub fn rosters(&self, realm: RealmId) -> bool {
        self.levels
            .back()
            .is_some_and(|l| l.rows.iter().any(|r| r.realm == realm))
    }

    /// The author's newest attested direct-child roster as a set — THE stream-only child list
    /// the marker admission vouches against (empty before the first level: fail-closed).
    #[must_use]
    pub fn roster_set(&self) -> BTreeSet<RealmId> {
        self.levels
            .back()
            .map(|l| l.rows.iter().map(|r| r.realm).collect())
            .unwrap_or_default()
    }

    /// The parent-shipped SL7 verdict (ids only; bodies/interiors gate — Slice C1's read).
    #[must_use]
    pub fn members(&self) -> &BTreeSet<RealmId> {
        &self.members
    }

    /// Admit one relayed child fence (the Q2 zombie guard): `true` iff `fence` is not stale
    /// against the newest seen for `child` (records the new high-water on admit). A refusal is
    /// counted by the caller (`window_relay_stale`), never patched.
    pub fn admit_relay_fence(&mut self, child: RealmId, fence: Fence) -> bool {
        if self
            .relay_fence
            .get(&child)
            .is_some_and(|held| fence.is_stale_against(*held))
        {
            return false;
        }
        self.relay_fence.insert(child, fence);
        true
    }

    /// Store one relayed interior level for `child` (newest `at` wins — an older relayed level
    /// is refused, `false`, counted by the caller like `ingest_body`'s staleness).
    pub fn ingest_relay_level(
        &mut self,
        child: RealmId,
        at: UniverseTick,
        rows: Vec<RealmSnap>,
    ) -> bool {
        if self
            .relay_levels
            .get(&child)
            .is_some_and(|(held_at, _)| *held_at > at)
        {
            return false;
        }
        self.relay_levels.insert(child, (at, rows));
        true
    }

    /// Every relayed live-child interior this window currently holds — `(child, at, rows)`,
    /// newest level per child (§2.6.5 step 4's raw material; the compose gates each by the
    /// membership verdict, never here).
    pub fn relayed_levels(&self) -> impl Iterator<Item = (RealmId, UniverseTick, &[RealmSnap])> {
        self.relay_levels
            .iter()
            .map(|(child, (at, rows))| (*child, *at, rows.as_slice()))
    }

    /// The relayed child's own attested roster (from its newest relayed level) — what its
    /// relayed MARKER statements are vouched against (empty before the first level: fail-closed,
    /// exactly like [`WindowIngest::roster_set`]).
    #[must_use]
    pub fn relay_child_roster(&self, child: RealmId) -> BTreeSet<RealmId> {
        self.relay_levels
            .get(&child)
            .map(|(_, rows)| rows.iter().map(|r| r.realm).collect())
            .unwrap_or_default()
    }

    /// The newest look bag stated about `subject`, if any (presence IS the draw law's gate).
    #[must_use]
    pub fn look_of(&self, subject: RealmId) -> Option<&[u8]> {
        self.look_of.get(&subject).map(|(_, bag)| bag.as_slice())
    }

    /// The newest marker bag stated about `subject`, if any.
    #[must_use]
    pub fn marker_of(&self, subject: RealmId) -> Option<&[u8]> {
        self.marker_of.get(&subject).map(|(_, bag)| bag.as_slice())
    }

    /// Every tick the ring currently retains, ascending (the common-tick fold's raw material).
    fn ticks(&self) -> impl Iterator<Item = UniverseTick> + '_ {
        self.levels.iter().map(|l| l.at)
    }
}

/// One row of the window catalog the chain derivation reads: a held window as (id, scope,
/// author, confirmed). Built by the gateway from its live window map — session/stream state
/// only, no world model (§2.6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CatalogRow {
    pub window: WindowId,
    pub scope: WindowScope,
    pub author: RealmId,
    /// `true` once the window has ingested ≥1 attested level — a `Child(c)` window's levels ARE
    /// the author's attested parenthood claim; an unconfirmed hop never extends a chain.
    pub confirmed: bool,
}

/// One hop of a derived chain, leaf-first: `hops[0]` is the observer's own-level window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainHop {
    pub window: WindowId,
    pub author: RealmId,
}

/// A session's derived chain (§2.6.2): leaf→root, cycle-safe.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Chain {
    pub hops: Vec<ChainHop>,
    /// A cycle was met and the chain truncated there, fail-closed (`window_chain_cycle`).
    pub cycled: bool,
}

/// Derive one session's chain from SESSION LINEAGE + ATTESTED HOP ROWS ONLY: the leaf realm is
/// the session's own (its live sub names it), the own-level window is the `Occupants` window
/// authored by that realm, and each upward hop is a CONFIRMED `Child(current)` window — whose
/// received levels are the author's attested claim `author = parent(current)`. No seed table, no
/// world model. Cycle-safe: a repeated author truncates the chain and flags it.
#[must_use]
pub fn derive_chain(leaf: RealmId, catalog: &[CatalogRow]) -> Chain {
    let mut chain = Chain::default();
    let Some(own) = catalog
        .iter()
        .find(|c| (c.scope == WindowScope::Occupants) & (c.author == leaf))
    else {
        return chain;
    };
    chain.hops.push(ChainHop {
        window: own.window,
        author: own.author,
    });
    let mut visited: BTreeSet<RealmId> = BTreeSet::new();
    visited.insert(leaf);
    let mut current = leaf;
    while let Some(hop) = catalog
        .iter()
        .find(|c| (c.scope == WindowScope::Child(current)) & c.confirmed)
    {
        if !visited.insert(hop.author) {
            chain.cycled = true;
            return chain;
        }
        chain.hops.push(ChainHop {
            window: hop.window,
            author: hop.author,
        });
        current = hop.author;
    }
    chain
}

/// The fresh prefix of a chain and the tick it can fold at (§2.6.3/§2.6.4): the LONGEST leaf-first
/// prefix whose windows share a retained stamp, at the NEWEST such stamp. A window that cannot
/// serve any common tick starts the held strata (everything from its level upward holds).
#[must_use]
pub fn fresh_prefix(levels: &[&WindowIngest]) -> (usize, Option<UniverseTick>) {
    let Some(first) = levels.first() else {
        return (0, None);
    };
    let mut common: Vec<UniverseTick> = first.ticks().collect();
    if common.is_empty() {
        return (0, None);
    }
    let mut best = *common.last().expect("non-empty checked above");
    let mut prefix = 1;
    for ingest in &levels[1..] {
        let theirs: BTreeSet<UniverseTick> = ingest.ticks().collect();
        let next: Vec<UniverseTick> = common
            .iter()
            .copied()
            .filter(|t| theirs.contains(t))
            .collect();
        let Some(newest) = next.last().copied() else {
            return (prefix, Some(best));
        };
        best = newest;
        common = next;
        prefix += 1;
    }
    (prefix, Some(best))
}

/// What a composed row's look payload would come from — the presence gate of §2.6.5 step 6
/// (self-look if one was received, else the parent's marker; a third source is unrepresentable
/// in the wire types). Diagnostic in shadow mode; Slice C1 attaches the bags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyTag {
    Look,
    Marker,
    Placement,
}

/// One composed row: a realm's pose re-expressed in the ORIGIN frame at the fold's tick.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedRow {
    pub realm: RealmId,
    /// The row's HEAD — the realm's own frame (carried through from the authored row).
    pub frame: FrameRef,
    /// The TAIL — the origin frame; stamp explicit per row (§2.4 `SceneRow.pose`).
    pub pose: StampedPose,
    /// Which chain level authored this row (leaf = 0) — the per-stratum hold's partition key.
    pub stratum: usize,
    pub body: BodyTag,
}

/// One fold's outcome: the composed rows plus every counted refusal the failure-mode table
/// (§2.6.6) names on the compose path. All counters are returned, never globally accumulated —
/// the CALLER owns the stats surface (guard 3: no handle out of the composer).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Composed {
    pub at: UniverseTick,
    pub rows: Vec<ComposedRow>,
    /// Rows refused by `FrameError::InstantMismatch` — the shear law firing (asserted 0 in the
    /// parity gate; deliberately driven nonzero by the G-SHEAR anti-vacuity unit).
    pub instant_refused: u64,
    /// Rows refused by `FrameError::RotatedFrameAcrossCells` (pre-P10 cell math, §2.6.6).
    pub rotated_refused: u64,
    /// Rows whose stated tail frame is not their level's frame (or an unknown-frame refusal from
    /// the fold) — alien, dropped, counted.
    pub alien_rows: u64,
    /// A level whose hop was absent/mismatched (or whose roster was empty, leaving its frame
    /// unstatable): the chain's fresh prefix was capped there.
    pub hop_invalid: u64,
    /// The §2.12 agreement measurement: how many chain-realm rows disagreed AT ALL (bit-level,
    /// position) with the hop-derived body pose for the same realm.
    pub dedup_disagree: u64,
    /// The measured bound of that agreement, in nanometres (max over the fold) — printed by the
    /// parity gate as the §2.12 "measured bound", never argued.
    pub dedup_max_dev_nm: u64,
    /// How many chain levels actually folded fresh (≤ the requested prefix — an invalid hop caps
    /// it). The exact-cadence pin reads folds where this equals the full chain length.
    pub fresh_levels: usize,
    /// §2.6.5 step 4 (Q2 = PARENT RELAY): relayed live-child interior rows COMPOSED into this
    /// fold — each mapped through `X_k ∘ placement(c)` at the relay's own stamp.
    pub relay_rows: u64,
    /// A relayed interior refused because its stamp fell off the chain rings, its child had no
    /// placement row at that stamp, or a hop below was invalid there — dropped, counted, healed
    /// by the child's next relay (§2.6.6: every class its own row).
    pub relay_unplaceable: u64,
}

/// The position gap between two composed poses, in nanometres, saturating. The §2.12 agreement
/// metric: exact zero today (identity orientations — the subtraction cancels bit-for-bit), a
/// measured bound the day a rotated hop lands.
fn pos_dev_nm(a: &StampedPose, b: &StampedPose) -> u64 {
    let cell_edge = a.frame.tier().cell_edge_m();
    let pa = a.pos.offset() + a.pos.cell().as_dvec3() * cell_edge;
    let pb = b.pos.offset() + b.pos.cell().as_dvec3() * cell_edge;
    let metres = (pa - pb).length();
    if metres.is_finite() {
        (metres * 1.0e9).min(u64::MAX as f64) as u64
    } else {
        u64::MAX
    }
}

/// THE FOLD (§2.6.5): compose the fresh prefix of one chain at exactly `t`, walking leaf→root
/// and mapping every level's rows (and every ancestor's own body) into the ORIGIN frame through
/// per-hop [`PlacementBook`]s built from the levels' own hop rows at `t`. Every step is
/// `transfer_frame` — the gateway inverts nothing, subtracts nothing, evaluates no motion; every
/// minus already happened in the lawful parent. Mixed-tick input is REFUSED row-by-row by
/// `InstantMismatch` (counted, dropped, never blended).
///
/// `levels[k]` must be the level stamped `t` of chain hop `k`; `prefix` bounds how far up the
/// walk goes (the caller's fresh prefix — everything above holds, §2.6.4).
#[must_use]
pub fn compose(
    origin: RealmId,
    origin_frame: FrameRef,
    t: UniverseTick,
    authors: &[RealmId],
    levels: &[&WindowLevel],
    prefix: usize,
    ingests: &[&WindowIngest],
) -> Composed {
    let mut out = Composed {
        at: t,
        ..Composed::default()
    };
    let mut rows: BTreeMap<RealmId, ComposedRow> = BTreeMap::new();
    // The per-hop step books, index k mapping frame F_k -> F_{k-1}; frames[k] = level k's own
    // frame (F_0 = the origin's).
    let mut frames: Vec<FrameRef> = vec![origin_frame];
    let mut books: Vec<PlacementBook> = Vec::new();
    for (k, level) in levels.iter().enumerate().take(prefix) {
        if k > 0 {
            // The hop must exist, name the chain child below, and the level must state its own
            // frame (≥1 row — a Child-window author always rosters at least the hop child).
            let hop_ok = level.hop.as_ref().map(|h| h.child);
            let stated_frame = level.rows.first().map(|r| r.pose.frame);
            let (Some(hop), Some(frame)) = (level.hop.as_ref(), stated_frame) else {
                out.hop_invalid += 1;
                break;
            };
            if hop_ok != Some(authors[k - 1]) {
                out.hop_invalid += 1;
                break;
            }
            books.push(PlacementBook::new(frames[k - 1], t, vec![(frame, hop.inv)]));
            frames.push(frame);
        }
        // The ancestor's own body rides the SAME mapping path as the child rows (one code path,
        // one set of refusal arms — HR5): a synthetic first row at the author's frame origin
        // (INV-BODY-AT-ORIGIN). Level 0 has no body row — the origin never ships one (§2.4).
        let body_row = (k > 0).then(|| RealmSnap {
            realm: authors[k],
            frame: frames[k],
            pose: StampedPose::at_rest(frames[k], vd_core::glam::DVec3::ZERO, t),
        });
        for row in body_row.iter().chain(level.rows.iter()) {
            if row.pose.frame != frames[k] {
                out.alien_rows += 1;
                continue;
            }
            let pose = match map_down(&row.pose, &frames, &books, k) {
                Ok(pose) => pose,
                Err(e) => {
                    count_refusal(&mut out, e);
                    continue;
                }
            };
            if row.realm == origin {
                // The origin never ships a row (§2.4) — but the row IS the agreement
                // measurement: the parent's placement for the origin, folded through the
                // pre-inverted hop, must land at the origin's own zero.
                let zero = StampedPose::at_rest(origin_frame, vd_core::glam::DVec3::ZERO, t);
                measure_agreement(&mut out, &pose, &zero);
                continue;
            }
            let stratum = k;
            let composed = ComposedRow {
                realm: row.realm,
                frame: row.frame,
                pose,
                stratum,
                body: body_tag(row.realm, ingests),
            };
            if let Some(existing) = rows.get(&row.realm) {
                // A chain author stated by BOTH its own hop-derived body and its parent's child
                // row: measure the §2.12 agreement, keep the parent's authored placement row.
                measure_agreement(&mut out, &existing.pose, &composed.pose);
            }
            rows.insert(row.realm, composed);
        }
        out.fresh_levels = k + 1;
    }
    // §2.6.5 STEP 4 (Q1 = YES-generic, Q2 = PARENT RELAY — §5 RULINGS, owner 2026-08-16): each
    // fresh stratum's RELAYED live-child interiors join the fold. A relayed row is the child's
    // OWN authored statement in its OWN frame (verbatim, sealed — the parent never re-stated
    // it); the stratum author's level AT THE RELAY'S OWN STAMP supplies `placement(c)` and the
    // descent below re-runs at that same stamp, so every conversion still composes exactly-
    // stamped pairs (§2.6.3's instant law — nothing here mixes two times). The composed row
    // keeps the relay's stamp, declared per row like a held stratum's (§2.6.4); the one relay
    // hop of look latency this stamps in is exactly the Q2-budgeted cost. STEP 5 gates the
    // interior by membership: a child beyond the stratum's shipped SL7 verdict draws nothing
    // (the flag day closed the beyond-visibility interiors fan for good).
    for k in 0..out.fresh_levels {
        // `.get`, not an index: the pure-fold units drive `compose` with bare levels and no
        // ingest state (relays then simply don't exist); the live caller always aligns the two.
        let Some(ingest) = ingests.get(k) else {
            continue;
        };
        for (child, at, crows) in ingest.relayed_levels() {
            if authors.contains(&child) {
                continue; // a chain member's interior rides its own window, never a relay
            }
            if !ingest.members().contains(&child) {
                continue; // step 5: beyond visibility ⇒ tracked nowhere (lawful, uncounted)
            }
            let Some((rframes, rbooks)) = descent_at(origin_frame, authors, ingests, k, at) else {
                out.relay_unplaceable += 1;
                continue;
            };
            let Some(placement) = ingest
                .level_at(at)
                .and_then(|l| l.rows.iter().find(|r| r.realm == child))
            else {
                out.relay_unplaceable += 1;
                continue;
            };
            let c_frame = placement.frame;
            // The parent-authored placement of the relayed child, as the one hop book entry —
            // orientation/velocity carried from the authored row; angular velocity is not on the
            // level wire (identity through P3; the rotated-hop composition is the ledgered P10
            // owe, exactly like `HopRow::inv`'s note).
            let hop_book = PlacementBook::new(
                rframes[k],
                at,
                vec![(
                    c_frame,
                    FramePlacement {
                        origin_cell: placement.pose.pos.cell(),
                        origin: placement.pose.pos.offset(),
                        velocity: placement.pose.vel,
                        orientation: placement.pose.orient,
                        angular_velocity: vd_core::glam::DVec3::ZERO,
                    },
                )],
            );
            for row in crows {
                if row.realm == origin {
                    // The origin never ships a row (§2.4) — a relayed statement naming it is
                    // dropped like the chain's own origin roster row (uncounted, lawful).
                    continue;
                }
                if row.pose.frame != c_frame {
                    out.alien_rows += 1;
                    continue;
                }
                // ONE fallible pipeline, ONE refusal arm (HR5): the hop transfer feeds the chain
                // descent through `and_then`, so a refusal at either stage lands on the same
                // counted arm (the descent stage alone cannot fail until P10's rotated hops —
                // its books are anchored identities at the relay's own stamp).
                let mapped = transfer_frame(&row.pose, rframes[k], &hop_book)
                    .and_then(|p| map_down(&p, &rframes, &rbooks, k));
                let pose = match mapped {
                    Ok(pose) => pose,
                    Err(e) => {
                        count_refusal(&mut out, e);
                        continue;
                    }
                };
                let composed = ComposedRow {
                    realm: row.realm,
                    frame: row.frame,
                    pose,
                    stratum: k,
                    body: body_tag(row.realm, ingests),
                };
                if let Some(existing) = rows.get(&row.realm) {
                    // A realm the chain already states (its parent's own child row) keeps the
                    // chain's placement; the relayed copy is the §2.12 agreement measurement.
                    measure_agreement(&mut out, &existing.pose, &composed.pose);
                    continue;
                }
                out.relay_rows += 1;
                rows.insert(row.realm, composed);
            }
        }
    }
    out.rows = rows.into_values().collect();
    out
}

/// The descent frames+books for strata `0..=k`, rebuilt AT `at` (a relayed interior's own
/// stamp): the identical walk [`compose`]'s main loop runs at T, refusing (`None`) when any
/// ring no longer retains `at` or a hop is invalid there — the caller counts one
/// `relay_unplaceable` and the child's next relay self-heals.
fn descent_at(
    origin_frame: FrameRef,
    authors: &[RealmId],
    ingests: &[&WindowIngest],
    k: usize,
    at: UniverseTick,
) -> Option<(Vec<FrameRef>, Vec<PlacementBook>)> {
    let mut frames = vec![origin_frame];
    let mut books = Vec::new();
    for j in 1..=k {
        let level = ingests.get(j)?.level_at(at)?;
        let hop = level.hop.as_ref()?;
        if hop.child != authors[j - 1] {
            return None;
        }
        let frame = level.rows.first().map(|r| r.pose.frame)?;
        books.push(PlacementBook::new(
            frames[j - 1],
            at,
            vec![(frame, hop.inv)],
        ));
        frames.push(frame);
    }
    Some((frames, books))
}

/// Map one pose from level `k`'s frame down to the origin frame, one `transfer_frame` per hop —
/// the pure composition of parent-authored, parent-inverted transforms via the frame core.
fn map_down(
    pose: &StampedPose,
    frames: &[FrameRef],
    books: &[PlacementBook],
    k: usize,
) -> Result<StampedPose, FrameError> {
    let mut pose = *pose;
    for j in (1..=k).rev() {
        pose = transfer_frame(&pose, frames[j - 1], &books[j - 1])?;
    }
    Ok(pose)
}

/// Drop every stamped entry older than `cutoff`, returning how many went. Generic over the stored
/// payload (a look is a TLV bag, a relayed level is typed rows) and deliberately BRANCHLESS per
/// HR5's generic-code discipline: the retain predicate is ONE comparison expression, so there is
/// no arm inside a generic body to leave uncovered in some monomorphization.
fn prune_older_than<T>(store: &mut BTreeMap<RealmId, (UniverseTick, T)>, cutoff: u64) -> u64 {
    let before = store.len();
    store.retain(|_, (at, _)| at.0 >= cutoff);
    (before - store.len()) as u64
}

/// Route one fold refusal onto its named counter (§2.6.6 — every class its own row, never a
/// shared bucket).
fn count_refusal(out: &mut Composed, e: FrameError) {
    match e {
        FrameError::InstantMismatch { .. } => out.instant_refused += 1,
        FrameError::RotatedFrameAcrossCells => out.rotated_refused += 1,
        FrameError::UnknownSourceFrame | FrameError::UnknownDestFrame => out.alien_rows += 1,
    }
}

/// Fold one agreement measurement (bit-level position) into the fold's counters.
fn measure_agreement(out: &mut Composed, a: &StampedPose, b: &StampedPose) {
    let dev = pos_dev_nm(a, b);
    out.dedup_max_dev_nm = out.dedup_max_dev_nm.max(dev);
    if a.pos != b.pos {
        out.dedup_disagree += 1;
    }
}

/// The presence gate (§2.6.5 step 6): a self-look from ANY chain window (only the realm itself
/// can have shipped one — admission), else a parent's marker, else the bare placement.
fn body_tag(realm: RealmId, ingests: &[&WindowIngest]) -> BodyTag {
    if ingests.iter().any(|i| i.look_of(realm).is_some()) {
        return BodyTag::Look;
    }
    if ingests.iter().any(|i| i.marker_of(realm).is_some()) {
        return BodyTag::Marker;
    }
    BodyTag::Placement
}

/// THE BAG SELECTION (§2.6.5 steps 5–6, LIVE since Slice C1): the TLV bag a composed row ships
/// with. A SELF-LOOK attaches iff one was received (only the realm itself can have shipped one —
/// admission) AND the realm passes the membership gate (`members` ∪ the chain authors — §2.2:
/// membership gates BODIES; the origin and its ancestors are always drawn). Else the parent's
/// MARKER (never membership-filtered — stars stay in the sky by construction). Else EMPTY: the
/// realm is tracked but not drawn (a missing statement means the thing is not drawn — THE DRAW
/// LAW by absence of data; a third pixel source is unrepresentable in the wire types).
#[must_use]
pub fn scene_bag(realm: RealmId, authors: &[RealmId], ingests: &[&WindowIngest]) -> Vec<u8> {
    let member = authors.contains(&realm) || ingests.iter().any(|i| i.members().contains(&realm));
    if member && let Some(look) = ingests.iter().find_map(|i| i.look_of(realm)) {
        return look.to_vec();
    }
    if let Some(luma) = ingests.iter().find_map(|i| i.marker_of(realm)) {
        return luma.to_vec();
    }
    Vec::new()
}

/// The hierarchy parent of one composed row (§2.4 `SceneRow.parent` — identity only, never a
/// position): a child row's parent is its stratum's author; a chain BODY row (the realm IS its
/// stratum's author) parents one level up; the root's body has none.
#[must_use]
pub fn scene_parent(row: &ComposedRow, authors: &[RealmId]) -> Option<RealmId> {
    if authors.get(row.stratum) == Some(&row.realm) {
        authors.get(row.stratum + 1).copied()
    } else {
        authors.get(row.stratum).copied()
    }
}

/// THE COMPOSED LEVEL'S ROWS (§2.4/§2.6.5 step 8, LIVE since Slice C1): the ORIGIN's own row
/// first — pose ZERO in its own frame at `t`, its self-look attached (the observer stands inside
/// it and sees its shell around them; "it draws from its look at the origin marker") — then every
/// fresh and held composed row, each with its hierarchy parent and its bag. The origin never
/// rides the per-tick DATAGRAM (a `RealmSnap` whose head equals its tail is banned); it rides the
/// LEVEL alone, where a [`SceneRow`] states exactly one frame.
#[must_use]
pub fn scene_level_rows(
    origin: RealmId,
    origin_frame: FrameRef,
    t: UniverseTick,
    authors: &[RealmId],
    ingests: &[&WindowIngest],
    scene: &ShadowScene,
) -> Vec<SceneRow> {
    let mut rows = vec![SceneRow {
        realm: origin,
        parent: authors.get(1).copied(),
        pose: StampedPose::at_rest(origin_frame, vd_core::glam::DVec3::ZERO, t),
        bag: scene_bag(origin, authors, ingests),
    }];
    rows.extend(scene.drawn_rows().map(|row| SceneRow {
        realm: row.realm,
        parent: scene_parent(row, authors),
        pose: row.pose,
        bag: scene_bag(row.realm, authors, ingests),
    }));
    rows
}

/// One stratum held at its last composed poses (§2.6.4): sky above the fault holds; each row
/// keeps its OLD stamp explicitly; a stratum dead past the derived TTL exits cleanly.
#[derive(Clone, Debug, PartialEq)]
pub struct HeldStratum {
    pub author: RealmId,
    pub rows: Vec<ComposedRow>,
    /// The fresh tick at which this stratum STOPPED being fresh (the dead-hop clock's anchor).
    pub held_since: UniverseTick,
}

/// What one [`ShadowScene::advance`] did — every count returned to the caller's stats surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdvanceReport {
    pub epoch_bumped: bool,
    /// Strata that HELD this tick (counted per stratum per advance — `compose_hold_ticks`).
    pub holds: u64,
    /// Held strata removed by the dead-hop exit (`window_hop_dead`).
    pub dead_hops: u64,
    /// A would-be T rewind froze emission (`window_t_monotone_stalled`).
    pub stalled: bool,
}

/// Per-session shadow scene: the chain identity, the origin marker (§2.7's epoch mechanics —
/// the crossing swap itself is Slice C1's client work), the held strata, the fresh-fold ring the
/// parity comparator reads, and the comparator's pending queue. Dropped with its session —
/// zero sessions, zero scenes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShadowScene {
    /// The chain's author list, leaf→root — the chain identity whose change bumps the epoch.
    pub chain_authors: Vec<RealmId>,
    /// The origin marker: the realm the picture is composed in, and its epoch (bumps on every
    /// chain/origin change — §2.7).
    pub origin: Option<RealmId>,
    pub origin_epoch: u64,
    /// The monotone-T guard (§2.6.3): a would-be rewind freezes, counted, never re-emitted.
    pub last_t: Option<UniverseTick>,
    /// Strata above the fresh boundary, held at their last composed poses (old stamps explicit).
    pub held: Vec<HeldStratum>,
    /// The fresh folds, ascending by tick, retained one ring span.
    pub ring: VecDeque<Composed>,
    /// Did the chain cover the session's whole lineage when last derived? (Diagnostic: a chain
    /// that does not yet reach the root is composing a partial sky, and the composer says so.)
    pub chain_covers_lineage: bool,
}

impl ShadowScene {
    /// Advance the scene by one gateway tick: bump the epoch on a chain/origin change, apply the
    /// monotone-T guard, roll holds/dead-hops, and push the fresh fold (if any) onto the ring.
    pub fn advance(
        &mut self,
        origin: RealmId,
        authors: &[RealmId],
        fold: Option<Composed>,
        tuning: &WindowTuning,
    ) -> AdvanceReport {
        let mut report = AdvanceReport::default();
        if self.origin != Some(origin) {
            // A CROSSING: the picture's frame changed, so nothing previously composed is
            // expressible any more — reset whole (§2.7: the epoch bump is the client's
            // atomic-swap signal).
            self.origin = Some(origin);
            self.chain_authors = authors.to_vec();
            self.origin_epoch += 1;
            self.ring.clear();
            self.held.clear();
            self.last_t = None;
            report.epoch_bumped = true;
        } else if self.chain_authors != authors {
            // The chain changed under the SAME origin (a hop confirmed, or a dead level left):
            // the epoch still bumps (§2.7 — the composed level's shape changed), but every
            // already-composed row is still expressed in the SAME origin frame, so the ring
            // stays; only strata of departed authors leave the hold.
            self.chain_authors = authors.to_vec();
            self.origin_epoch += 1;
            self.held.retain(|h| authors.contains(&h.author));
            report.epoch_bumped = true;
        }
        let Some(fold) = fold else {
            // No common tick anywhere: the whole scene holds at its last composed picture —
            // counted per chain level, but ONLY once something was ever composed (a pre-first-
            // fold boot tick has nothing to hold; counting it would be noise, not a stall).
            if !self.ring.is_empty() {
                report.holds += authors.len() as u64;
            }
            return report;
        };
        if self.last_t.is_some_and(|last| fold.at < last) {
            report.stalled = true;
            return report;
        }
        if self.last_t == Some(fold.at) {
            // The same tick re-folded (no newer common stamp yet): nothing new to emit, not a
            // rewind — the ring already holds it.
            return report;
        }
        self.last_t = Some(fold.at);
        // Roll the held strata: freshly composed levels leave the hold; levels beyond the fresh
        // prefix (that were fresh before) enter it at their LAST composed rows.
        let fresh = fold.fresh_levels;
        self.held.retain(|h| {
            authors
                .iter()
                .position(|a| *a == h.author)
                .is_some_and(|i| i >= fresh)
        });
        if let Some(prev) = self.ring.back() {
            for (i, author) in authors.iter().enumerate().skip(fresh) {
                if self.held.iter().any(|h| h.author == *author) {
                    continue;
                }
                let rows: Vec<ComposedRow> = prev
                    .rows
                    .iter()
                    .filter(|r| r.stratum == i)
                    .cloned()
                    .collect();
                self.held.push(HeldStratum {
                    author: *author,
                    rows,
                    held_since: fold.at,
                });
            }
        }
        report.holds += (authors.len().saturating_sub(fresh)) as u64;
        // The dead-hop exit (§2.6.4): a stratum held past the derived TTL is removed cleanly.
        let ttl = tuning.hold_ttl_ticks;
        let before = self.held.len();
        self.held.retain(|h| fold.at.0 - h.held_since.0 <= ttl);
        report.dead_hops += (before - self.held.len()) as u64;
        self.ring.push_back(fold);
        let newest = self.ring.back().expect("a fold was just pushed").at.0;
        while self
            .ring
            .front()
            .is_some_and(|c| c.at.0 + tuning.ring_span_ticks < newest)
        {
            self.ring.pop_front();
        }
        report
    }

    /// THE DRAWN SET (LIVE since Slice C1): the newest fresh fold's rows plus every held
    /// stratum's rows — exactly what the composed level and the per-tick datagram carry (held
    /// rows keep their OLD stamps explicitly, §2.6.4: stale-by-declaration, never silently mixed).
    pub fn drawn_rows(&self) -> impl Iterator<Item = &ComposedRow> + '_ {
        self.ring
            .back()
            .into_iter()
            .flat_map(|c| c.rows.iter())
            .chain(self.held.iter().flat_map(|h| h.rows.iter()))
    }

    /// The fresh fold at exactly `t`, if the ring retains it.
    #[must_use]
    pub fn fold_at(&self, t: UniverseTick) -> Option<&Composed> {
        self.ring
            .binary_search_by_key(&t, |c| c.at)
            .ok()
            .and_then(|i| self.ring.get(i))
    }

    /// The newest fresh tick (`None` before the first fold).
    #[must_use]
    pub fn newest(&self) -> Option<UniverseTick> {
        self.ring.back().map(|c| c.at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::glam::{DQuat, DVec3};

    const T: UniverseTick = UniverseTick(100);

    fn sys() -> FrameRef {
        FrameRef::SystemSpace { system_seed: 7 }
    }
    fn planet() -> FrameRef {
        FrameRef::PlanetCentered { planet_seed: 7 }
    }
    /// The galaxy PARENT of the fixture chain — the galaxy shard hosts a System realm
    /// (`RealmId::System(GALAXY_SEED)`; `vd-bins` `world_roster`), so the fixture mirrors that.
    const GALAXY: RealmId = RealmId::System(0);
    fn galaxy() -> FrameRef {
        FrameRef::SystemSpace { system_seed: 0 }
    }
    fn tuning() -> WindowTuning {
        WindowTuning::derive(2) // beats of 2 ticks ⇒ span/ttl 5, cap 5×64
    }

    fn row(
        realm: RealmId,
        frame: FrameRef,
        tail: FrameRef,
        pos: DVec3,
        at: UniverseTick,
    ) -> RealmSnap {
        RealmSnap {
            realm,
            frame,
            pose: StampedPose::at_rest(tail, pos, at),
        }
    }

    fn level(at: UniverseTick, hop: Option<HopRow>, rows: Vec<RealmSnap>) -> WindowLevel {
        WindowLevel { at, hop, rows }
    }

    /// The identity-orientation pre-inverted hop for a child sitting at `child_pos` in the
    /// author's frame, moving at `child_vel`: the author's frame in the child's = the negation.
    fn hop(child: RealmId, child_pos: DVec3, child_vel: DVec3) -> HopRow {
        HopRow {
            child,
            inv: vd_core::frame::FramePlacement::moving(-child_pos, -child_vel),
        }
    }

    #[test]
    fn the_tuning_is_the_two_beats_plus_one_law() {
        let t = WindowTuning::derive(10);
        assert_eq!(t.ring_span_ticks, 21);
        assert_eq!(t.hold_ttl_ticks, 21);
        assert_eq!(t.look_ttl_ticks, 21);
    }

    #[test]
    fn a_look_that_stops_being_re_asserted_prunes_and_the_marker_is_the_floor() {
        // THE DEPARTURE MIRROR (§2.8), as a measurement. A live realm re-asserts its look on every
        // keep-alive beat; one that tore down stops. Past the derived roster-loss window its look
        // is dropped, the presence gate falls through to the parent's marker, and the body it drew
        // shrinks to a point of light.
        let t = tuning(); // look ttl 5
        let subject = RealmId::System(7);
        let child = RealmId::Planet(7);
        let mut ingest = WindowIngest::default();
        assert!(ingest.ingest_body(
            subject,
            &BodyStmt::SelfLook {
                bag: vd_core::look::look_bag(&vd_core::geometry::Boundary::Shell { r: 150.0 }),
            },
            UniverseTick(100),
        ));
        assert!(ingest.ingest_body(
            subject,
            &BodyStmt::Marker { luma: vec![9] },
            UniverseTick(100)
        ));
        assert!(ingest.ingest_relay_level(child, UniverseTick(100), Vec::new()));
        // INSIDE the window (age exactly the TTL): everything survives — a live realm re-asserting
        // one beat late is not a dead one.
        assert_eq!(ingest.prune_stale(UniverseTick(105), &t), (0, 0));
        assert!(ingest.look_of(subject).is_some());
        assert_eq!(ingest.relayed_levels().count(), 1);
        // ONE TICK PAST IT: the look and the relayed level go; the MARKER stays (the floor — the
        // presence law is "never zero", so a departed system becomes a dot, not a hole).
        assert_eq!(ingest.prune_stale(UniverseTick(106), &t), (1, 1));
        assert_eq!(ingest.look_of(subject), None);
        assert_eq!(ingest.relayed_levels().count(), 0);
        assert_eq!(ingest.marker_of(subject), Some(&[9u8][..]));
        // And the presence gate now answers MARKER for the same realm it answered LOOK for.
        assert_eq!(scene_bag(subject, &[subject], &[&ingest]), vec![9u8]);
        // A gateway whose universe clock has not passed the window yet prunes nothing (saturating).
        assert_eq!(ingest.prune_stale(UniverseTick(0), &t), (0, 0));
    }

    #[test]
    fn the_ring_applies_replaces_reorders_and_refuses_behind_the_span() {
        let mut ingest = WindowIngest::default();
        let t = tuning();
        assert_eq!(ingest.newest(), None);
        assert!(!ingest.confirmed());
        assert_eq!(
            ingest.ingest_frame(level(UniverseTick(10), None, vec![]), &t),
            Ingested::Applied
        );
        assert!(ingest.confirmed());
        // Newer appends; the head advances.
        assert_eq!(
            ingest.ingest_frame(level(UniverseTick(12), None, vec![]), &t),
            Ingested::Applied
        );
        assert_eq!(ingest.newest(), Some(UniverseTick(12)));
        // A re-delivered stamp REPLACES (latest-wins per tick).
        let replacement = level(
            UniverseTick(12),
            None,
            vec![row(
                RealmId::Planet(7),
                planet(),
                sys(),
                DVec3::X,
                UniverseTick(12),
            )],
        );
        assert_eq!(ingest.ingest_frame(replacement, &t), Ingested::Applied);
        assert!(ingest.rosters(RealmId::Planet(7)));
        assert!(!ingest.rosters(RealmId::Planet(9)));
        // An out-of-order tick inside the span is inserted in stamp order.
        assert_eq!(
            ingest.ingest_frame(level(UniverseTick(11), None, vec![]), &t),
            Ingested::Applied
        );
        assert_eq!(
            ingest.level_at(UniverseTick(11)).map(|l| l.at),
            Some(UniverseTick(11))
        );
        // Behind the span (head 12, span 5 ⇒ floor 7): refused.
        assert_eq!(
            ingest.ingest_frame(level(UniverseTick(6), None, vec![]), &t),
            Ingested::BehindRing
        );
        // Advancing the head trims the tail out of the span (head 18, span 5 ⇒ floor 13).
        assert_eq!(
            ingest.ingest_frame(level(UniverseTick(18), None, vec![]), &t),
            Ingested::Applied
        );
        assert_eq!(ingest.level_at(UniverseTick(10)), None);
        assert_eq!(ingest.level_at(UniverseTick(12)), None);
        assert_eq!(
            ingest.level_at(UniverseTick(18)).map(|l| l.at),
            Some(UniverseTick(18))
        );
    }

    #[test]
    fn bodies_are_newest_wins_and_membership_is_a_diff() {
        let mut ingest = WindowIngest::default();
        let subject = RealmId::System(7);
        assert!(ingest.ingest_body(
            subject,
            &BodyStmt::SelfLook { bag: vec![1] },
            UniverseTick(5)
        ));
        // Older refused (counted by the caller), newest kept.
        assert!(!ingest.ingest_body(
            subject,
            &BodyStmt::SelfLook { bag: vec![9] },
            UniverseTick(4)
        ));
        assert_eq!(ingest.look_of(subject), Some(&[1u8][..]));
        // Same-stamp replaces (a re-driven statement is idempotent).
        assert!(ingest.ingest_body(
            subject,
            &BodyStmt::SelfLook { bag: vec![2] },
            UniverseTick(5)
        ));
        assert_eq!(ingest.look_of(subject), Some(&[2u8][..]));
        let child = RealmId::Planet(7);
        assert!(ingest.ingest_body(child, &BodyStmt::Marker { luma: vec![3] }, UniverseTick(5)));
        assert_eq!(ingest.marker_of(child), Some(&[3u8][..]));
        assert_eq!(ingest.look_of(child), None);
        assert_eq!(ingest.marker_of(subject), None);

        ingest.ingest_membership(&[child], &[]);
        assert_eq!(ingest.members().len(), 1);
        ingest.ingest_membership(&[], &[child]);
        assert!(ingest.members().is_empty());
    }

    fn catalog() -> Vec<CatalogRow> {
        vec![
            CatalogRow {
                window: WindowId(1),
                scope: WindowScope::Occupants,
                author: RealmId::System(7),
                confirmed: true,
            },
            CatalogRow {
                window: WindowId(2),
                scope: WindowScope::Child(RealmId::System(7)),
                author: GALAXY,
                confirmed: true,
            },
        ]
    }

    #[test]
    fn the_chain_derives_leaf_first_from_confirmed_hops_only() {
        let chain = derive_chain(RealmId::System(7), &catalog());
        assert_eq!(
            chain,
            Chain {
                hops: vec![
                    ChainHop {
                        window: WindowId(1),
                        author: RealmId::System(7)
                    },
                    ChainHop {
                        window: WindowId(2),
                        author: GALAXY
                    },
                ],
                cycled: false,
            }
        );
        // An UNCONFIRMED hop never extends the chain (no attested claim yet).
        let mut rows = catalog();
        rows[1].confirmed = false;
        assert_eq!(derive_chain(RealmId::System(7), &rows).hops.len(), 1);
        // No own-level window ⇒ empty chain (the login race withholds the composed feed).
        assert_eq!(derive_chain(RealmId::Planet(9), &rows), Chain::default());
    }

    #[test]
    fn a_chain_cycle_truncates_fail_closed_and_is_flagged() {
        // Two windows each claiming to parent the other's author: A ⊃ B and B ⊃ A.
        let rows = vec![
            CatalogRow {
                window: WindowId(1),
                scope: WindowScope::Occupants,
                author: RealmId::System(1),
                confirmed: true,
            },
            CatalogRow {
                window: WindowId(2),
                scope: WindowScope::Child(RealmId::System(1)),
                author: RealmId::System(2),
                confirmed: true,
            },
            CatalogRow {
                window: WindowId(3),
                scope: WindowScope::Child(RealmId::System(2)),
                author: RealmId::System(1),
                confirmed: true,
            },
        ];
        let chain = derive_chain(RealmId::System(1), &rows);
        assert!(chain.cycled);
        assert_eq!(chain.hops.len(), 2, "truncated AT the repeated author");
    }

    #[test]
    fn the_fresh_prefix_is_the_longest_leaf_first_common_tick_run() {
        let t = tuning();
        let mut leaf = WindowIngest::default();
        let mut parent = WindowIngest::default();
        let mut grand = WindowIngest::default();
        for tick in [10, 11, 12] {
            let _ = leaf.ingest_frame(level(UniverseTick(tick), None, vec![]), &t);
        }
        for tick in [10, 11] {
            let _ = parent.ingest_frame(level(UniverseTick(tick), None, vec![]), &t);
        }
        let _ = grand.ingest_frame(level(UniverseTick(9), None, vec![]), &t);
        // All three: the grandparent shares no tick ⇒ prefix 2 at the newest leaf∩parent stamp.
        assert_eq!(
            fresh_prefix(&[&leaf, &parent, &grand]),
            (2, Some(UniverseTick(11)))
        );
        // Leaf alone: its own newest.
        assert_eq!(fresh_prefix(&[&leaf]), (1, Some(UniverseTick(12))));
        // Full agreement: the shared newest.
        assert_eq!(fresh_prefix(&[&leaf, &parent]), (2, Some(UniverseTick(11))));
        // An empty leaf (or an empty slice): nothing to fold.
        assert_eq!(fresh_prefix(&[]), (0, None));
        assert_eq!(fresh_prefix(&[&WindowIngest::default()]), (0, None));
    }

    /// THE two-level fixture: an observer in System(7) under Galaxy(0). The system sits at
    /// (1000, 0, 0) in the galaxy, moving +Y at 5; its planet orbits at (30, 0, 0); the galaxy
    /// rosters a sibling system at (1300, 0, 0).
    fn two_level_fixture() -> (Vec<RealmId>, WindowLevel, WindowLevel) {
        let leaf_level = level(
            T,
            None,
            vec![row(
                RealmId::Planet(7),
                planet(),
                sys(),
                DVec3::new(30.0, 0.0, 0.0),
                T,
            )],
        );
        let parent_level = level(
            T,
            Some(hop(
                RealmId::System(7),
                DVec3::new(1000.0, 0.0, 0.0),
                DVec3::new(0.0, 5.0, 0.0),
            )),
            vec![
                row(
                    RealmId::System(7),
                    sys(),
                    galaxy(),
                    DVec3::new(1000.0, 0.0, 0.0),
                    T,
                ),
                row(
                    RealmId::System(9),
                    FrameRef::SystemSpace { system_seed: 9 },
                    galaxy(),
                    DVec3::new(1300.0, 0.0, 0.0),
                    T,
                ),
            ],
        );
        (vec![RealmId::System(7), GALAXY], leaf_level, parent_level)
    }

    #[test]
    fn the_fold_stacks_two_levels_at_one_tick_into_the_origin_frame() {
        let (authors, leaf_level, parent_level) = two_level_fixture();
        let ingests: Vec<&WindowIngest> = vec![];
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&leaf_level, &parent_level],
            2,
            &ingests,
        );
        assert_eq!(out.fresh_levels, 2);
        assert_eq!(out.instant_refused, 0);
        assert_eq!(out.alien_rows, 0);
        assert_eq!(out.hop_invalid, 0);
        // Three rows: the planet (own level), the galaxy's body (chain), the sibling system.
        // The origin's own row (System 7 in the galaxy's roster) is dropped — and measured.
        assert_eq!(out.rows.len(), 3);
        let by_realm = |r: RealmId| out.rows.iter().find(|row| row.realm == r).expect("row");
        let planet_row = by_realm(RealmId::Planet(7));
        assert_eq!(planet_row.pose.pos.offset(), DVec3::new(30.0, 0.0, 0.0));
        assert_eq!(planet_row.stratum, 0);
        assert_eq!(planet_row.pose.universe_tick, T);
        // The sibling system: (1300 − 1000) in the origin frame, moving −Y 5 relative.
        let sibling = by_realm(RealmId::System(9));
        assert_eq!(sibling.pose.pos.offset(), DVec3::new(300.0, 0.0, 0.0));
        assert_eq!(sibling.pose.vel, DVec3::new(0.0, -5.0, 0.0));
        assert_eq!(sibling.stratum, 1);
        // The galaxy's own body: at −(system placement) — the hop origin (INV-BODY-AT-ORIGIN).
        let galaxy_body = by_realm(GALAXY);
        assert_eq!(galaxy_body.pose.pos.offset(), DVec3::new(-1000.0, 0.0, 0.0));
        // The §2.12 agreement: the origin's own roster row folded to EXACT zero (identity
        // orientations cancel bit-for-bit) — measured, not argued.
        assert_eq!(out.dedup_disagree, 0);
        assert_eq!(out.dedup_max_dev_nm, 0);
    }

    /// §2.6.5 STEP 4 (Q2 = PARENT RELAY, Slice C1): a live sibling's relayed interior joins the
    /// fold — mapped through the galaxy's placement of the sibling AT THE RELAY'S OWN STAMP and
    /// down the chain into the origin frame — while the step-5 membership gate, the placement
    /// gate and the alien-frame arm each refuse their own class, counted.
    #[test]
    fn a_relayed_sibling_interior_composes_through_the_parents_placement_at_its_own_stamp() {
        let (authors, leaf_level, parent_level) = two_level_fixture();
        let t = WindowTuning::derive(2);
        // Stratum 0 (the origin's own window): no relays.
        let mut leaf = WindowIngest::default();
        assert_eq!(leaf.ingest_frame(leaf_level.clone(), &t), Ingested::Applied);
        // Stratum 1 (the galaxy window): membership admits the live sibling System 9, whose
        // relayed level states its planet at (40, 0, 0) in ITS OWN frame — self-authored,
        // verbatim, never re-stated by the galaxy.
        let sibling = RealmId::System(9);
        let sibling_frame = FrameRef::SystemSpace { system_seed: 9 };
        let mut parent = WindowIngest::default();
        assert_eq!(
            parent.ingest_frame(parent_level.clone(), &t),
            Ingested::Applied
        );
        parent.ingest_membership(&[sibling], &[]);
        assert!(parent.ingest_relay_level(
            sibling,
            T,
            vec![
                row(
                    RealmId::Planet(9),
                    FrameRef::PlanetCentered { planet_seed: 9 },
                    sibling_frame,
                    DVec3::new(40.0, 0.0, 0.0),
                    T,
                ),
                // An alien row (stated off the sibling's own frame): dropped + counted.
                row(
                    RealmId::Planet(10),
                    FrameRef::PlanetCentered { planet_seed: 10 },
                    galaxy(),
                    DVec3::new(1.0, 0.0, 0.0),
                    T,
                ),
            ],
        ));
        let ingests: Vec<&WindowIngest> = vec![&leaf, &parent];
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&leaf_level, &parent_level],
            2,
            &ingests,
        );
        // The sibling's planet: (1300 + 40) − 1000 = 340 in the origin frame, stamped at the
        // relay's own tick, stratum = the galaxy's.
        let planet9 = out
            .rows
            .iter()
            .find(|r| r.realm == RealmId::Planet(9))
            .expect("the relayed interior row composes");
        assert_eq!(planet9.pose.pos.offset(), DVec3::new(340.0, 0.0, 0.0));
        assert_eq!(planet9.pose.universe_tick, T);
        assert_eq!(planet9.stratum, 1);
        assert_eq!(out.relay_rows, 1);
        assert_eq!(out.alien_rows, 1);
        assert_eq!(out.relay_unplaceable, 0);
    }

    /// Every remaining relay-fold arm, driven one by one (HR5 region+branch): the chain-author
    /// skip, the relayed-origin drop, the already-composed dedup (agreement measured, chain
    /// wins), and the off-stamp row's `InstantMismatch` refusal.
    #[test]
    fn the_relay_folds_edge_arms_each_refuse_their_own_class() {
        let (authors, leaf_level, parent_level) = two_level_fixture();
        let t = WindowTuning::derive(2);
        let mut leaf = WindowIngest::default();
        assert_eq!(leaf.ingest_frame(leaf_level.clone(), &t), Ingested::Applied);
        let sibling = RealmId::System(9);
        let sibling_frame = FrameRef::SystemSpace { system_seed: 9 };
        let mut parent = WindowIngest::default();
        assert_eq!(
            parent.ingest_frame(parent_level.clone(), &t),
            Ingested::Applied
        );
        parent.ingest_membership(&[sibling, RealmId::System(7)], &[]);
        // (a) A relay keyed by a CHAIN AUTHOR: skipped whole (its own window composes it).
        assert!(parent.ingest_relay_level(
            RealmId::System(7),
            T,
            vec![row(
                RealmId::Planet(7),
                planet(),
                sys(),
                DVec3::new(99.0, 0.0, 0.0),
                T,
            )],
        ));
        // (b) The member sibling's relay carries FOUR rows: the ORIGIN (dropped uncounted), a
        // row the chain ALREADY states (Planet 7 — agreement measured, the chain's pose wins),
        // an OFF-STAMP row (InstantMismatch, counted), and one lawful new interior row.
        assert!(parent.ingest_relay_level(
            sibling,
            T,
            vec![
                row(
                    RealmId::System(7),
                    sys(),
                    sibling_frame,
                    DVec3::new(1.0, 0.0, 0.0),
                    T,
                ),
                row(
                    RealmId::Planet(7),
                    planet(),
                    sibling_frame,
                    DVec3::new(2.0, 0.0, 0.0),
                    T,
                ),
                row(
                    RealmId::Planet(11),
                    FrameRef::PlanetCentered { planet_seed: 11 },
                    sibling_frame,
                    DVec3::new(3.0, 0.0, 0.0),
                    UniverseTick(T.0 + 1),
                ),
                row(
                    RealmId::Planet(9),
                    FrameRef::PlanetCentered { planet_seed: 9 },
                    sibling_frame,
                    DVec3::new(40.0, 0.0, 0.0),
                    T,
                ),
            ],
        ));
        let ingests: Vec<&WindowIngest> = vec![&leaf, &parent];
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&leaf_level, &parent_level],
            2,
            &ingests,
        );
        // Only the lawful interior row composed as a RELAY row.
        assert_eq!(out.relay_rows, 1);
        assert_eq!(out.instant_refused, 1);
        // The chain's Planet 7 pose won (its own level's 30 m, not the relay's re-statement) —
        // and the disagreement was MEASURED, never silently overwritten.
        let planet7 = out
            .rows
            .iter()
            .find(|r| r.realm == RealmId::Planet(7))
            .expect("the chain's own planet row");
        assert_eq!(planet7.pose.pos.offset(), DVec3::new(30.0, 0.0, 0.0));
        assert!(out.dedup_disagree >= 1);
        // The chain author's relayed copy composed NOTHING beyond the chain's own statement:
        // Planet 7 stayed the chain's, and no 99 m ghost row exists anywhere.
        assert!(
            !out.rows
                .iter()
                .any(|r| r.pose.pos.offset() == DVec3::new(99.0, 0.0, 0.0)),
            "a chain author's relay must never compose"
        );
        // No row for the origin, ever (§2.4).
        assert!(!out.rows.iter().any(|r| r.realm == RealmId::System(7)));
    }

    /// [`descent_at`]'s refusal arms, each driven directly (HR5): a chain window missing at the
    /// stamp, an own-level window where a hop is required, a hop naming the wrong child, and an
    /// empty roster leaving the frame unstatable.
    #[test]
    fn the_relayed_descent_refuses_missing_levels_hops_and_frames() {
        let (authors, _leaf_level, parent_level) = two_level_fixture();
        let t = WindowTuning::derive(2);
        let sysf = sys();
        // (a) ingests shorter than the walk: refused.
        assert!(descent_at(sysf, &authors, &[], 1, T).is_none());
        // (b) a window with NO level at the stamp: refused.
        let empty = WindowIngest::default();
        let leaf = WindowIngest::default();
        assert!(descent_at(sysf, &authors, &[&leaf, &empty], 1, T).is_none());
        // (c) an own-level window (no hop) where the walk needs one: refused.
        let mut hopless = WindowIngest::default();
        assert_eq!(
            hopless.ingest_frame(level(T, None, parent_level.rows.clone()), &t),
            Ingested::Applied
        );
        assert!(descent_at(sysf, &authors, &[&leaf, &hopless], 1, T).is_none());
        // (d) a hop naming the WRONG child: refused.
        let mut wrong_child = WindowIngest::default();
        assert_eq!(
            wrong_child.ingest_frame(
                level(
                    T,
                    Some(hop(RealmId::System(9), DVec3::ZERO, DVec3::ZERO)),
                    parent_level.rows.clone(),
                ),
                &t
            ),
            Ingested::Applied
        );
        assert!(descent_at(sysf, &authors, &[&leaf, &wrong_child], 1, T).is_none());
        // (e) a valid hop over an EMPTY roster (frame unstatable): refused.
        let mut bare = WindowIngest::default();
        assert_eq!(
            bare.ingest_frame(
                level(
                    T,
                    Some(hop(RealmId::System(7), DVec3::ZERO, DVec3::ZERO)),
                    vec![],
                ),
                &t
            ),
            Ingested::Applied
        );
        assert!(descent_at(sysf, &authors, &[&leaf, &bare], 1, T).is_none());
        // (f) the happy walk — the same fixture the compose tests fold — succeeds.
        let mut parent = WindowIngest::default();
        assert_eq!(parent.ingest_frame(parent_level, &t), Ingested::Applied);
        let (frames, books) =
            descent_at(sysf, &authors, &[&leaf, &parent], 1, T).expect("the happy walk succeeds");
        assert_eq!(frames.len(), 2);
        assert_eq!(books.len(), 1);
    }

    /// The relay fold's refusal arms, each driven separately: a NON-MEMBER sibling draws
    /// nothing (step 5 — lawful, uncounted), and a member whose placement row is missing at
    /// the relay's stamp counts `relay_unplaceable` (fail-closed, healed by the next relay).
    #[test]
    fn a_relayed_interior_is_membership_gated_and_unplaceable_is_counted() {
        let (authors, leaf_level, parent_level) = two_level_fixture();
        let t = WindowTuning::derive(2);
        let mut leaf = WindowIngest::default();
        assert_eq!(leaf.ingest_frame(leaf_level.clone(), &t), Ingested::Applied);
        let sibling = RealmId::System(9);
        let sibling_frame = FrameRef::SystemSpace { system_seed: 9 };
        let interior = vec![row(
            RealmId::Planet(9),
            FrameRef::PlanetCentered { planet_seed: 9 },
            sibling_frame,
            DVec3::new(40.0, 0.0, 0.0),
            T,
        )];
        // (a) NOT a member: the interior is skipped entirely — no row, no refusal count.
        let mut parent = WindowIngest::default();
        assert_eq!(
            parent.ingest_frame(parent_level.clone(), &t),
            Ingested::Applied
        );
        assert!(parent.ingest_relay_level(sibling, T, interior.clone()));
        let ingests: Vec<&WindowIngest> = vec![&leaf, &parent];
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&leaf_level, &parent_level],
            2,
            &ingests,
        );
        assert_eq!(out.relay_rows, 0);
        assert_eq!(out.relay_unplaceable, 0);
        assert!(!out.rows.iter().any(|r| r.realm == RealmId::Planet(9)));
        // (b) A member relayed at a stamp the parent's ring does not retain: counted.
        let mut parent2 = WindowIngest::default();
        assert_eq!(
            parent2.ingest_frame(parent_level.clone(), &t),
            Ingested::Applied
        );
        parent2.ingest_membership(&[sibling], &[]);
        assert!(parent2.ingest_relay_level(sibling, UniverseTick(T.0 + 1), interior.clone()));
        let ingests2: Vec<&WindowIngest> = vec![&leaf, &parent2];
        let out2 = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&leaf_level, &parent_level],
            2,
            &ingests2,
        );
        assert_eq!(out2.relay_rows, 0);
        assert_eq!(out2.relay_unplaceable, 1);
        // (c) The descent WALKS at the relay's stamp but the parent's level THERE has no
        // placement row for the child (a roster race): counted, healed by the next relay.
        let t_old = UniverseTick(T.0 - 1);
        let mut parent3 = WindowIngest::default();
        assert_eq!(
            parent3.ingest_frame(parent_level.clone(), &t),
            Ingested::Applied
        );
        assert_eq!(
            parent3.ingest_frame(
                level(
                    t_old,
                    Some(hop(
                        RealmId::System(7),
                        DVec3::new(1000.0, 0.0, 0.0),
                        DVec3::new(0.0, 5.0, 0.0),
                    )),
                    vec![row(
                        RealmId::System(7),
                        sys(),
                        galaxy(),
                        DVec3::new(1000.0, 0.0, 0.0),
                        t_old,
                    )],
                ),
                &t
            ),
            Ingested::Applied
        );
        parent3.ingest_membership(&[sibling], &[]);
        assert!(parent3.ingest_relay_level(sibling, t_old, interior));
        let ingests3: Vec<&WindowIngest> = vec![&leaf, &parent3];
        let out3 = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&leaf_level, &parent_level],
            2,
            &ingests3,
        );
        assert_eq!(out3.relay_rows, 0);
        assert_eq!(out3.relay_unplaceable, 1);
    }

    #[test]
    fn g_shear_the_deliberate_mixed_tick_compose_must_fail() {
        // ANTI-VACUITY (§2.6.3): a level whose rows are stamped OFF the fold tick is refused by
        // `InstantMismatch` — the row is dropped and counted, never blended. This drives the
        // refusal arm on purpose, proving the shear gate CAN fail.
        let stale = UniverseTick(99);
        let leaf_level = level(
            T,
            None,
            vec![row(RealmId::Planet(7), planet(), sys(), DVec3::X, stale)],
        );
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &[RealmId::System(7)],
            &[&leaf_level],
            1,
            &[],
        );
        assert_eq!(
            out.instant_refused, 0,
            "level 0 needs no book — same-frame rows pass"
        );
        // Level 0 rows ride untransformed... so drive the SAME stale row through a HOP level,
        // where the book at T MUST refuse the tick-99 pose.
        let parent_level = level(
            T,
            Some(hop(
                RealmId::System(7),
                DVec3::new(1000.0, 0.0, 0.0),
                DVec3::ZERO,
            )),
            vec![row(
                RealmId::System(9),
                FrameRef::SystemSpace { system_seed: 9 },
                galaxy(),
                DVec3::X,
                stale,
            )],
        );
        let fresh_leaf = level(T, None, vec![]);
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &[RealmId::System(7), GALAXY],
            &[&fresh_leaf, &parent_level],
            2,
            &[],
        );
        assert_eq!(out.instant_refused, 1, "the mixed-tick fold WAS refused");
        assert!(
            !out.rows.iter().any(|r| r.realm == RealmId::System(9)),
            "the refused row never composed"
        );
    }

    #[test]
    fn a_rotated_cross_cell_hop_still_folds_because_the_down_maps_dest_is_the_identity_anchor() {
        // MEASURED, not assumed (the never-assume law): the gateway's down-map can NEVER hit
        // `RotatedFrameAcrossCells` itself — every step book's DESTINATION is its anchor (the
        // child frame at the identity), and the refusal fires only for a ROTATED DESTINATION
        // across cells. The rotated-inversion refusal therefore lives AT THE AUTHOR
        // (`vd-sim`'s hop inversion, `window_hop_refused`, per §2.2), and a hop that the author
        // DID ship folds here with its integer cell anchor surviving exactly.
        let spun = vd_core::frame::FramePlacement {
            origin_cell: vd_core::glam::I64Vec3::new(1_000_000_000_000, 0, 0),
            origin: DVec3::ZERO,
            velocity: DVec3::ZERO,
            orientation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            angular_velocity: DVec3::ZERO,
        };
        let parent_level = level(
            T,
            Some(HopRow {
                child: RealmId::System(7),
                inv: spun,
            }),
            vec![row(
                RealmId::System(9),
                FrameRef::SystemSpace { system_seed: 9 },
                galaxy(),
                DVec3::X,
                T,
            )],
        );
        let fresh_leaf = level(T, None, vec![]);
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &[RealmId::System(7), GALAXY],
            &[&fresh_leaf, &parent_level],
            2,
            &[],
        );
        assert_eq!(out.rotated_refused, 0);
        assert_eq!(
            out.rows.len(),
            2,
            "the galaxy body + the sibling row both folded"
        );
        let sibling = out
            .rows
            .iter()
            .find(|r| r.realm == RealmId::System(9))
            .expect("sibling row");
        assert_eq!(
            sibling.pose.pos.cell(),
            vd_core::glam::I64Vec3::new(1_000_000_000_000, 0, 0),
            "the hop's integer anchor rides through the fold un-flattened"
        );

        // The gateway-side counter arms stay real code with a pinned contract regardless:
        // every `FrameError` variant lands on its named counter (§2.6.6), driven directly.
        let mut out = Composed::default();
        count_refusal(
            &mut out,
            FrameError::InstantMismatch {
                book_at: T,
                pose_at: UniverseTick(1),
            },
        );
        count_refusal(&mut out, FrameError::RotatedFrameAcrossCells);
        count_refusal(&mut out, FrameError::UnknownSourceFrame);
        count_refusal(&mut out, FrameError::UnknownDestFrame);
        assert_eq!(
            (out.instant_refused, out.rotated_refused, out.alien_rows),
            (1, 1, 2)
        );
    }

    #[test]
    fn alien_rows_and_invalid_hops_cap_the_fold_fail_closed() {
        // A row stating a tail frame other than its level's: alien, dropped, counted.
        let leaf_level = level(
            T,
            None,
            vec![row(RealmId::Planet(7), planet(), galaxy(), DVec3::X, T)],
        );
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &[RealmId::System(7)],
            &[&leaf_level],
            1,
            &[],
        );
        assert_eq!(out.alien_rows, 1);
        assert_eq!(out.rows.len(), 0);

        // A hop naming the WRONG child: the prefix caps there (nothing above folds).
        let (authors, fresh_leaf, _) = two_level_fixture();
        let wrong_hop = level(
            T,
            Some(hop(RealmId::System(99), DVec3::X, DVec3::ZERO)),
            vec![row(
                RealmId::System(9),
                FrameRef::SystemSpace { system_seed: 9 },
                galaxy(),
                DVec3::X,
                T,
            )],
        );
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&fresh_leaf, &wrong_hop],
            2,
            &[],
        );
        assert_eq!(out.hop_invalid, 1);
        assert_eq!(out.fresh_levels, 1, "capped below the invalid hop");

        // A hop level with NO rows cannot state its own frame: same cap, same counter.
        let rowless = level(
            T,
            Some(hop(RealmId::System(7), DVec3::X, DVec3::ZERO)),
            vec![],
        );
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&fresh_leaf, &rowless],
            2,
            &[],
        );
        assert_eq!(out.hop_invalid, 1);

        // A hop level with rows but NO hop at all: the same fail-closed cap.
        let hopless = level(
            T,
            None,
            vec![row(
                RealmId::System(9),
                FrameRef::SystemSpace { system_seed: 9 },
                galaxy(),
                DVec3::X,
                T,
            )],
        );
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&fresh_leaf, &hopless],
            2,
            &[],
        );
        assert_eq!(out.hop_invalid, 1);
    }

    #[test]
    fn a_three_level_chain_dedups_the_mid_author_measuring_the_agreement() {
        // §2.6.5 step 7's dedup arm proper: a realm that is BOTH a chain author (its hop-derived
        // body entered at level 1) and a child row (its parent's roster at level 2) resolves to
        // ONE entry — the parent's authored placement row — with the two sources' f64 agreement
        // MEASURED (exact zero under identity orientations, the §2.12 bound).
        let mid = GALAXY; // the mid author: leaf System(7) ⊂ System(0) ⊂ System(42)
        let grand = RealmId::System(42);
        let grand_frame = FrameRef::SystemSpace { system_seed: 42 };
        let leaf_level = level(T, None, vec![]);
        // System(0) sits at 1000 in System(42); System(7) sits at 100 in System(0).
        let mid_level = level(
            T,
            Some(hop(
                RealmId::System(7),
                DVec3::new(100.0, 0.0, 0.0),
                DVec3::ZERO,
            )),
            vec![row(
                RealmId::System(7),
                sys(),
                galaxy(),
                DVec3::new(100.0, 0.0, 0.0),
                T,
            )],
        );
        let grand_level = level(
            T,
            Some(hop(mid, DVec3::new(1000.0, 0.0, 0.0), DVec3::ZERO)),
            vec![row(
                mid,
                galaxy(),
                grand_frame,
                DVec3::new(1000.0, 0.0, 0.0),
                T,
            )],
        );
        let out = compose(
            RealmId::System(7),
            sys(),
            T,
            &[RealmId::System(7), mid, grand],
            &[&leaf_level, &mid_level, &grand_level],
            3,
            &[],
        );
        assert_eq!(out.fresh_levels, 3);
        // ONE entry for the mid author (hop body replaced by the grand's child row), plus the
        // grandparent's own body. Both at −100 / −1100 in the leaf frame.
        assert_eq!(out.rows.len(), 2);
        let mid_row = out.rows.iter().find(|r| r.realm == mid).expect("mid row");
        assert_eq!(mid_row.pose.pos.offset(), DVec3::new(-100.0, 0.0, 0.0));
        assert_eq!(
            mid_row.stratum, 2,
            "the surviving entry is the parent's child row"
        );
        let grand_row = out
            .rows
            .iter()
            .find(|r| r.realm == grand)
            .expect("grand body");
        assert_eq!(grand_row.pose.pos.offset(), DVec3::new(-1100.0, 0.0, 0.0));
        // The agreement was measured on BOTH overlaps (origin row + mid dedup): exact.
        assert_eq!(out.dedup_disagree, 0);
        assert_eq!(out.dedup_max_dev_nm, 0);
        // An empty ingest rosters nothing (the pre-first-level shape the admission refuses on).
        assert!(WindowIngest::default().roster_set().is_empty());
    }

    #[test]
    fn the_body_tag_is_a_presence_gate_never_a_flag() {
        let mut own = WindowIngest::default();
        assert!(own.ingest_body(RealmId::System(7), &BodyStmt::SelfLook { bag: vec![1] }, T));
        assert!(own.ingest_body(RealmId::Planet(7), &BodyStmt::Marker { luma: vec![2] }, T));
        let ingests: Vec<&WindowIngest> = vec![&own];
        assert_eq!(body_tag(RealmId::System(7), &ingests), BodyTag::Look);
        assert_eq!(body_tag(RealmId::Planet(7), &ingests), BodyTag::Marker);
        assert_eq!(body_tag(RealmId::Planet(9), &ingests), BodyTag::Placement);
    }

    fn folded(t: UniverseTick, fresh_levels: usize, rows: Vec<ComposedRow>) -> Composed {
        Composed {
            at: t,
            rows,
            fresh_levels,
            ..Composed::default()
        }
    }

    fn one_row(realm: RealmId, stratum: usize, t: UniverseTick) -> ComposedRow {
        ComposedRow {
            realm,
            frame: sys(),
            pose: StampedPose::at_rest(sys(), DVec3::X, t),
            stratum,
            body: BodyTag::Placement,
        }
    }

    #[test]
    fn the_scene_bumps_its_epoch_on_chain_change_and_holds_monotone_t() {
        let t = tuning();
        let mut scene = ShadowScene::default();
        let authors = vec![RealmId::System(7), GALAXY];
        // First derivation: origin set, epoch 0 → 1.
        let r = scene.advance(RealmId::System(7), &authors, Some(folded(T, 2, vec![])), &t);
        assert!(r.epoch_bumped);
        assert_eq!(scene.origin_epoch, 1);
        assert_eq!(scene.newest(), Some(T));
        // Same chain, newer fold: no bump.
        let r = scene.advance(
            RealmId::System(7),
            &authors,
            Some(folded(UniverseTick(101), 2, vec![])),
            &t,
        );
        assert!(!r.epoch_bumped);
        // The SAME tick re-folded: a quiet no-op, never a stall.
        let r = scene.advance(
            RealmId::System(7),
            &authors,
            Some(folded(UniverseTick(101), 2, vec![])),
            &t,
        );
        assert!(!r.stalled);
        // A would-be REWIND freezes, counted (§2.6.3 monotone-T).
        let r = scene.advance(
            RealmId::System(7),
            &authors,
            Some(folded(UniverseTick(90), 2, vec![])),
            &t,
        );
        assert!(r.stalled);
        assert_eq!(
            scene.newest(),
            Some(UniverseTick(101)),
            "the rewind never landed"
        );
        // No common tick at all: the whole scene holds, one count per chain level.
        let r = scene.advance(RealmId::System(7), &authors, None, &t);
        assert_eq!(r.holds, 2);
        // A chain EXTENSION under the same origin: the epoch bumps but the ring (same origin
        // frame — still comparable) is KEPT and only departed authors leave the hold.
        let wider = vec![RealmId::System(7), GALAXY, RealmId::System(3)];
        let r = scene.advance(
            RealmId::System(7),
            &wider,
            Some(folded(UniverseTick(102), 3, vec![])),
            &t,
        );
        assert!(r.epoch_bumped);
        assert_eq!(scene.origin_epoch, 2);
        assert_eq!(scene.newest(), Some(UniverseTick(102)), "the ring survived");
        // A crossing (new origin): epoch bumps, the ring resets whole.
        let r = scene.advance(RealmId::Planet(7), &[RealmId::Planet(7)], None, &t);
        assert!(r.epoch_bumped);
        assert_eq!(scene.origin_epoch, 3);
        assert_eq!(scene.newest(), None);
    }

    #[test]
    fn a_stale_stratum_holds_at_its_last_composed_poses_then_exits_by_the_derived_ttl() {
        let t = tuning(); // hold ttl 5
        let mut scene = ShadowScene::default();
        let authors = vec![RealmId::System(7), GALAXY];
        // Tick 100: both strata fresh; the galaxy stratum's rows are in the ring.
        let full = folded(
            T,
            2,
            vec![
                one_row(RealmId::Planet(7), 0, T),
                one_row(RealmId::System(9), 1, T),
            ],
        );
        let _ = scene.advance(RealmId::System(7), &authors, Some(full), &t);
        assert!(scene.held.is_empty());
        // Tick 101: the galaxy hop lags — stratum 1 HOLDS at its last composed rows (old stamp).
        let partial = folded(
            UniverseTick(101),
            1,
            vec![one_row(RealmId::Planet(7), 0, UniverseTick(101))],
        );
        let r = scene.advance(RealmId::System(7), &authors, Some(partial), &t);
        assert_eq!(r.holds, 1);
        assert_eq!(scene.held.len(), 1);
        assert_eq!(scene.held[0].author, GALAXY);
        assert_eq!(scene.held[0].rows, vec![one_row(RealmId::System(9), 1, T)]);
        assert_eq!(scene.held[0].held_since, UniverseTick(101));
        // Still held two ticks later (inside the TTL) — held_since anchored, not re-stamped.
        let partial = folded(
            UniverseTick(103),
            1,
            vec![one_row(RealmId::Planet(7), 0, UniverseTick(103))],
        );
        let r = scene.advance(RealmId::System(7), &authors, Some(partial), &t);
        assert_eq!(r.holds, 1);
        assert_eq!(r.dead_hops, 0);
        assert_eq!(scene.held[0].held_since, UniverseTick(101));
        // Past the derived TTL (101 + 5 < 107): the DEAD-HOP EXIT removes the stratum cleanly.
        let partial = folded(
            UniverseTick(107),
            1,
            vec![one_row(RealmId::Planet(7), 0, UniverseTick(107))],
        );
        let r = scene.advance(RealmId::System(7), &authors, Some(partial), &t);
        assert_eq!(r.dead_hops, 1);
        assert!(scene.held.is_empty());
        // HEAL: the hop resumes fresh — no stratum held any more.
        let full = folded(
            UniverseTick(108),
            2,
            vec![
                one_row(RealmId::Planet(7), 0, UniverseTick(108)),
                one_row(RealmId::System(9), 1, UniverseTick(108)),
            ],
        );
        let r = scene.advance(RealmId::System(7), &authors, Some(full), &t);
        assert_eq!(r.holds, 0);
        assert!(scene.held.is_empty());
    }

    #[test]
    fn the_ring_retains_one_span() {
        let t = tuning(); // span 5
        let mut scene = ShadowScene::default();
        let authors = vec![RealmId::System(7)];
        for tick in 100..=110 {
            let _ = scene.advance(
                RealmId::System(7),
                &authors,
                Some(folded(UniverseTick(tick), 1, vec![])),
                &t,
            );
        }
        assert_eq!(
            scene.fold_at(UniverseTick(104)),
            None,
            "aged out of the span"
        );
        assert!(scene.fold_at(UniverseTick(105)).is_some());
        assert!(scene.fold_at(UniverseTick(110)).is_some());
    }

    #[test]
    fn the_deviation_metric_is_exact_at_zero_and_saturates_on_garbage() {
        let a = StampedPose::at_rest(sys(), DVec3::new(1.0, 2.0, 3.0), T);
        assert_eq!(pos_dev_nm(&a, &a), 0);
        let b = StampedPose::at_rest(sys(), DVec3::new(1.0, 2.0, 3.5), T);
        assert_eq!(pos_dev_nm(&a, &b), 500_000_000);
        let nan = StampedPose::at_rest(sys(), DVec3::new(f64::NAN, 0.0, 0.0), T);
        assert_eq!(pos_dev_nm(&a, &nan), u64::MAX);
    }

    #[test]
    fn zero_windows_and_zero_sessions_mean_zero_composer_state() {
        // The structural teardown truth at the module level: a default ingest and a default
        // scene hold NOTHING — every byte of composer state lives inside a window record or a
        // session record, both of which the gateway drops at close (asserted end-to-end by the
        // gateway's teardown test).
        let ingest = WindowIngest::default();
        assert_eq!(ingest, WindowIngest::default());
        assert!(!ingest.confirmed());
        let scene = ShadowScene::default();
        assert_eq!(scene.origin, None);
        assert_eq!(scene.origin_epoch, 0);
        assert!(scene.ring.is_empty() & scene.held.is_empty());
    }

    // ===== Slice C1 — the LIVE emissions' helpers (§2.4/§2.6.5 steps 5–8) ====================

    /// The bag selection is THE DRAW LAW by data presence + the membership gate: a member's (or a
    /// chain author's) self-look wins; a non-member's look is WITHHELD and falls to its marker
    /// (markers are never membership-filtered — stars stay in the sky); neither statement means
    /// an EMPTY bag (tracked, not drawn).
    #[test]
    fn scene_bag_is_the_membership_gated_presence_selection() {
        let realm = RealmId::Planet(7);
        let mut ingest = WindowIngest::default();
        // Neither statement: empty — the realm is tracked, never drawn.
        assert_eq!(scene_bag(realm, &[], &[&ingest]), Vec::<u8>::new());
        // A marker alone ships regardless of membership (never filtered).
        assert!(ingest.ingest_body(realm, &BodyStmt::Marker { luma: vec![2] }, T));
        assert_eq!(scene_bag(realm, &[], &[&ingest]), vec![2]);
        // A look exists but the realm is NO member and NO chain author: the look is withheld —
        // the marker still serves (§2.6.5 step 5: membership gates BODIES).
        assert!(ingest.ingest_body(realm, &BodyStmt::SelfLook { bag: vec![1] }, T));
        assert_eq!(scene_bag(realm, &[], &[&ingest]), vec![2]);
        // Membership admits the look…
        ingest.ingest_membership(&[realm], &[]);
        assert_eq!(scene_bag(realm, &[], &[&ingest]), vec![1]);
        // …and a CHAIN AUTHOR's look is always admitted (the origin and its ancestors are drawn).
        ingest.ingest_membership(&[], &[realm]);
        assert_eq!(scene_bag(realm, &[realm], &[&ingest]), vec![1]);
    }

    /// The hierarchy parent of a composed row: a child row parents on its stratum's author; a
    /// chain BODY row parents one level up; the root's body has none.
    #[test]
    fn scene_parent_is_identity_only_hierarchy() {
        let authors = [RealmId::System(7), GALAXY];
        let child_row = ComposedRow {
            realm: RealmId::Planet(7),
            frame: planet(),
            pose: StampedPose::at_rest(sys(), DVec3::X, T),
            stratum: 0,
            body: BodyTag::Placement,
        };
        assert_eq!(scene_parent(&child_row, &authors), Some(RealmId::System(7)));
        let galaxy_body = ComposedRow {
            realm: GALAXY,
            frame: galaxy(),
            pose: StampedPose::at_rest(sys(), -DVec3::X, T),
            stratum: 1,
            body: BodyTag::Placement,
        };
        assert_eq!(
            scene_parent(&galaxy_body, &authors),
            None,
            "the root's body"
        );
        let mid_body = ComposedRow {
            realm: RealmId::System(7),
            frame: sys(),
            pose: StampedPose::at_rest(sys(), DVec3::ZERO, T),
            stratum: 0,
            body: BodyTag::Placement,
        };
        assert_eq!(
            scene_parent(&mid_body, &authors),
            Some(GALAXY),
            "a chain body parents one level up"
        );
    }

    /// The composed level's rows: the ORIGIN row FIRST (pose ZERO in its own frame, its bag
    /// through the same selection), then every fresh and held drawn row with its parent + bag —
    /// and the drawn set is the newest fold plus the held strata (old stamps kept, §2.6.4).
    #[test]
    fn scene_level_rows_lead_with_the_origin_and_carry_the_drawn_set() {
        let (authors, leaf_level, parent_level) = two_level_fixture();
        let fold = compose(
            RealmId::System(7),
            sys(),
            T,
            &authors,
            &[&leaf_level, &parent_level],
            2,
            &[],
        );
        let mut scene = ShadowScene::default();
        let tuning = tuning();
        let report = scene.advance(RealmId::System(7), &authors, Some(fold), &tuning);
        assert!(report.epoch_bumped);
        // Give the origin a look so the origin row's bag arm is non-empty.
        let mut ingest = WindowIngest::default();
        assert!(ingest.ingest_body(RealmId::System(7), &BodyStmt::SelfLook { bag: vec![7] }, T));
        let rows = scene_level_rows(RealmId::System(7), sys(), T, &authors, &[&ingest], &scene);
        assert_eq!(rows[0].realm, RealmId::System(7), "the origin row leads");
        assert_eq!(rows[0].parent, Some(GALAXY));
        assert_eq!(rows[0].pose.pos.offset(), DVec3::ZERO);
        assert_eq!(rows[0].bag, vec![7], "the origin's look, membership-free");
        let drawn: Vec<RealmId> = rows.iter().skip(1).map(|r| r.realm).collect();
        assert_eq!(
            drawn,
            vec![RealmId::Planet(7), GALAXY, RealmId::System(9)],
            "the newest fold's rows ride behind the origin (BTreeMap fold order — RealmId Ord)"
        );
        // drawn_rows unions the newest fold with HELD strata: hold the top stratum and re-read.
        let empty_leaf = level(UniverseTick(101), None, vec![]);
        let fold2 = compose(
            RealmId::System(7),
            sys(),
            UniverseTick(101),
            &authors,
            &[&empty_leaf],
            1,
            &[],
        );
        let _ = scene.advance(RealmId::System(7), &authors, Some(fold2), &tuning);
        let rows = scene_level_rows(
            RealmId::System(7),
            sys(),
            UniverseTick(101),
            &authors,
            &[&ingest],
            &scene,
        );
        let held: Vec<(RealmId, UniverseTick)> = rows
            .iter()
            .skip(1)
            .map(|r| (r.realm, r.pose.universe_tick))
            .collect();
        assert_eq!(
            held,
            vec![(GALAXY, T), (RealmId::System(9), T)],
            "the held stratum's rows keep their OLD stamp, declared per row (§2.6.4)"
        );
    }

    /// The Q2 relay's ingest storage: the fence guard orders incarnations; a relayed level is
    /// newest-wins per child; the relayed roster vouches the child's OWN markers (empty before
    /// the first level — fail-closed).
    #[test]
    fn relay_storage_orders_fences_and_levels_and_vouches_rosters() {
        let mut ingest = WindowIngest::default();
        let child = RealmId::Planet(7);
        // Fence order: first admit records; an older fence refuses; same-or-newer admits.
        assert!(ingest.admit_relay_fence(child, vd_core::Fence(3)));
        assert!(!ingest.admit_relay_fence(child, vd_core::Fence(2)));
        assert!(ingest.admit_relay_fence(child, vd_core::Fence(3)));
        assert!(ingest.admit_relay_fence(child, vd_core::Fence(4)));
        // Roster before any level: EMPTY (fail-closed — a marker about anything refuses).
        assert_eq!(ingest.relay_child_roster(child), BTreeSet::new());
        // A level stores; its rows are the child's attested roster.
        let rows = vec![row(
            RealmId::Area(3),
            FrameRef::AreaLocal {
                planet_seed: 7,
                area_seed: 3,
            },
            planet(),
            DVec3::X,
            T,
        )];
        assert!(ingest.ingest_relay_level(child, T, rows.clone()));
        assert_eq!(
            ingest.relay_child_roster(child),
            BTreeSet::from([RealmId::Area(3)])
        );
        // An OLDER relayed level refuses (newest wins); an equal/newer replaces.
        assert!(!ingest.ingest_relay_level(child, UniverseTick(99), Vec::new()));
        assert_eq!(
            ingest.relay_child_roster(child),
            BTreeSet::from([RealmId::Area(3)]),
            "the stale level replaced nothing"
        );
        assert!(ingest.ingest_relay_level(child, UniverseTick(101), Vec::new()));
        assert_eq!(ingest.relay_child_roster(child), BTreeSet::new());
    }
}
