//! THE REGION FOREST this shard evaluates containment against: its own realm, its ancestor chain,
//! and a bounded set of children.
//!
//! Owns: the boot-planted region set with its precomputed depth keys and ancestor bitmasks, so the
//! per-tick containment fold never re-walks a parent chain, and the width bound that fails LOUD at
//! boot rather than silently aliasing one region onto another.
//!
//! Does NOT own: the hundreds of sibling realms a crowded parent may have — those resolve through
//! the directory, never through a local bit. Nor does it own MOTION: a region's stored centre is
//! where a static child sits; where a MOVING child is comes from the placement ledger, and nothing
//! here can tell the two apart (SL4).

use super::{placement_row, pose_of_row};
use bevy_ecs::prelude::Resource;
use std::collections::BTreeMap;
use vd_core::UniverseTick;
use vd_core::geometry::{DepthKey, RealmRegion, region_depth};
use vd_core::kinematics::secs_since_epoch;
use vd_core::placement::{MotionFn, PlacementBook};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::RealmLevel;
use vd_core::worldgen::level_of;
use vd_wire::channels::{RealmShape, RealmSnap};

/// The bitset width bound: a shard's region set exceeding this fails LOUD at boot (`guard_regions_nest`,
/// C-5). The scale answer is NOT a wider bitset — it is the own-realm + ~4-ancestor scoping (children via
/// the directory, §2.3), so 64 is generous headroom, not a ceiling on how crowded a realm can be.
pub const MAX_REGIONS: usize = 64;

/// The seed-derived realm REGIONS this shard evaluates CONTAINMENT against (task #135): the shard's own
/// realm + its ancestor chain (+ a bounded child set). Holds the regions, their BOOT-COMPUTED depth keys
/// (so the per-tick `container` fold never re-walks parents — O(entities × regions), not O(N·M²)), and the
/// ambient-ROOT realm (the `parent: None` region — the `container` fold identity). DEFAULT EMPTY → the
/// detector early-returns (INERT in production through C-3; the seed boot-population + `guard_regions_nest`
/// land C-5/C-6). Replaces the directional `RealmBoundaries` portal registry.
#[derive(Resource, Debug, Default)]
pub struct RealmRegions {
    /// ★THROWAWAY (test instrument, owner-ordered 2026-08-20): a multiplier on the CRUISE ceiling
    /// only — see [`vd_core::flight::FlightTuning::cruise_overdrive`]. `1.0` is the law and is what
    /// [`RealmRegions::new`] plants on every rig; a shard only ever carries more because its bin read
    /// `VD_TEST_OVERDRIVE` and called [`RealmRegions::set_cruise_overdrive`].
    ///
    /// It lifts the realm's OWN ceiling and NOTHING else. The approach governor's arms keep their
    /// lawful values deliberately: the containment bands are sized against the LAWFUL child ceilings,
    /// so lifting those too would let a subject cross a boundary band in under one tick and fly
    /// straight through a realm — the exact hole the OQ-2 ruling closed. Cruise fast, arrive lawfully.
    ///
    /// Scheduled for deletion with the keyboard-throttle instrument, when the ship realm becomes the
    /// thing you fly and a throttle becomes a commanded force from a functional block.
    pub(crate) cruise_overdrive: f64,
    pub(crate) regions: Vec<RealmRegion>,
    pub(crate) depths: Vec<DepthKey>,
    pub(crate) root_realm: Option<RealmId>,
    /// The shard's DIRECT MOVING children (FA-2b → the placement arc S5): realm → the OPAQUE
    /// [`MotionFn`] the boot injected for it. A region in this map has its book row authored live each
    /// tick by RUNNING the closure at the book's instant; a region absent from it rides its static
    /// `center`. The simulation can RUN a motion, never NAME one: the closure's contents live in the
    /// motion crate, which this crate carries no edge to (SL4, `crate_isolation` law) — the same seam
    /// discipline as `sim::io`. EMPTY at walk/static scale, so the writer is byte-identical to FA-1.
    moving: BTreeMap<RealmId, MotionFn>,
    /// Region index of each realm — the inverse of `regions`, so a realm resolves to its bit without a scan.
    pub(crate) ix_of: BTreeMap<RealmId, usize>,
    /// SELF ∪ ANCESTORS as a region bitmask, per region index (task #177). Computed ONCE at boot by the same
    /// parent walk the depth cache already does, bounded by [`MAX_REGIONS`] ⇒ at most 64×64 steps.
    ///
    /// WHY THIS EXISTS. Containment hysteresis needs to know whether you were ALREADY inside a region. That
    /// used to be REMEMBERED per entity — and a hand-off leaves the two shards with OPPOSITE memories, so
    /// anything parked in the ambiguous zone between the acquire and release edges ping-pongs forever. It is
    /// also RAM-only, so a shard restart blanks every resident's memory at once and reproduces the bug at
    /// full population in a single tick.
    ///
    /// The prior is instead DERIVED: you are always a hysteretic member of the realm you are AUTHORITATIVELY
    /// in, and of every ancestor of it. That is a total function of committed state, so it survives a crash,
    /// a replay, a re-drive and a lease change identically — there is nothing to get out of sync.
    pub(crate) ancestor_mask: Vec<u64>,
}

/// SELF ∪ ANCESTORS of `realm` as a region bitmask. Mirrors [`region_depth`]'s parent walk exactly —
/// same hop cap, same dangling-parent and cycle stops — so the two caches can never disagree about the
/// shape of the forest. A realm not present in `ix_of` contributes no bit (it is not a region here).
///
/// Bits beyond [`MAX_REGIONS`] are unrepresentable and are DROPPED rather than wrapped: a forest that
/// large is already rejected by the boot guard, and silently aliasing bit 64 onto bit 0 would hand a
/// subject membership in an unrelated region.
fn ancestry_bits(regions: &[RealmRegion], ix_of: &BTreeMap<RealmId, usize>, realm: RealmId) -> u64 {
    let mut bits = 0u64;
    let mut cur = realm;
    for _ in 0..regions.len() {
        if let Some(ix) = ix_of.get(&cur)
            && *ix < MAX_REGIONS
        {
            bits |= 1u64 << ix;
        }
        let Some(region) = regions.iter().find(|r| r.realm == cur) else {
            return bits; // dangling parent — stop where region_depth stops
        };
        let Some(parent) = region.parent else {
            return bits; // reached the ambient root
        };
        cur = parent;
    }
    bits // hop cap hit — a cycle (boot-guard-rejected); a safe stop, never a hang
}

impl RealmRegions {
    /// Build the resource from a region forest, computing the depth-key cache + the ambient-root realm
    /// ONCE (regions are static at P3). The per-tick detector reads the cache; it never re-walks parents.
    /// The moving-child roster starts EMPTY — [`with_moving_children`](Self::with_moving_children) adds it.
    #[must_use]
    /// ★THROWAWAY (test instrument): plant the cruise-only overdrive read from `VD_TEST_OVERDRIVE`.
    /// Values below the lawful `1.0` are refused back to it — the instrument may only ever go faster.
    pub fn set_cruise_overdrive(&mut self, factor: f64) -> f64 {
        self.cruise_overdrive = factor.max(1.0);
        self.cruise_overdrive
    }

    pub fn new(regions: Vec<RealmRegion>) -> RealmRegions {
        let depths = regions
            .iter()
            .enumerate()
            .map(|(ix, r)| (region_depth(&regions, r.realm), r.realm, ix))
            .collect();
        let root_realm = regions.iter().find(|r| r.parent.is_none()).map(|r| r.realm);
        let ix_of: BTreeMap<RealmId, usize> = regions
            .iter()
            .enumerate()
            .map(|(ix, r)| (r.realm, ix))
            .collect();
        let ancestor_mask = regions
            .iter()
            .map(|r| ancestry_bits(&regions, &ix_of, r.realm))
            .collect();
        RealmRegions {
            // ★THROWAWAY: the lawful 1.0 on every rig — only a bin that read `VD_TEST_OVERDRIVE`
            // raises it, through `set_cruise_overdrive`.
            cruise_overdrive: 1.0,
            regions,
            depths,
            root_realm,
            moving: BTreeMap::new(),
            ix_of,
            ancestor_mask,
        }
    }

    /// SELF ∪ ANCESTORS as a region bitmask for the realm a subject is authoritatively in — the DERIVED
    /// hysteresis prior. Zero for a realm this shard does not host, which degrades to the old blank-prior
    /// behaviour rather than inventing membership; callers warn on that path rather than passing it off as
    /// normal (a pose naming an unhosted realm means a rebind safe-degrade happened upstream).
    #[must_use]
    pub fn ancestor_mask_for(&self, realm: RealmId) -> u64 {
        self.ix_of
            .get(&realm)
            .and_then(|ix| self.ancestor_mask.get(*ix))
            .copied()
            .unwrap_or(0)
    }

    /// Register the shard's DIRECT MOVING children (FA-2b): the `(realm, motion)` roster the BOOT
    /// composition derives for the hosted realm and injects as opaque [`MotionFn`]s. A builder (not a
    /// `new` arg) so the many `RealmRegions::new` call sites stay unchanged and byte-identical (the
    /// walk roster passes an empty map). Only registered realms' rows author live; every other region
    /// stays static.
    #[must_use]
    pub fn with_moving_children(mut self, moving: BTreeMap<RealmId, MotionFn>) -> RealmRegions {
        self.moving = moving;
        self
    }

    /// The detector short-circuits (inert) when no regions are planted — production through C-3.
    pub(crate) fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// VU AoI S2a-2b — is ANY planted region's interest band armed (a non-zero spin-up radius)? The
    /// walk/static `AoiConfig::inert()` bands are all `spin_up_r_m == 0`, so this is `false` there and the
    /// whole up-relay stays byte-identical; it flips `true` only once a demand-scale forest is planted.
    pub(crate) fn aoi_live(&self) -> bool {
        self.regions.iter().any(|r| r.aoi.spin_up_r_m() > 0.0)
    }

    /// The full root-rooted [`RealmCoord`] of `realm` within this shard's neighbourhood forest — the RLM
    /// ledger key a source shard needs to `KeepAlive`-demand a crossing DEST (and, via `ancestor_close`, its
    /// whole Universe→…→dest chain) so it cannot be reaped mid-crossing. A branchless delegate to the pure
    /// vd-core fold; `None` only for a ship (entity-backed, never a crossing dest).
    pub(crate) fn coord_of(&self, realm: RealmId) -> Option<RealmCoord> {
        vd_core::worldgen::coord_of_realm(&self.regions, realm)
    }

    /// The forest's ambient-root frame: the `parent.is_none()` region's frame, defaulting to `GalaxySpace`
    /// for an empty forest. ONLY a fallback for [`own_frame`](Self::own_frame) — a shard whose own realm is
    /// absent from the forest has no frame of its own to name, and the ambient root is the one frame every
    /// forest is guaranteed to have. Nothing measures distances in it: under the ground rule a shard does
    /// not know where the root is, so folding INTO the root frame cannot succeed and must not be attempted.
    pub(crate) fn root_frame(&self) -> FrameRef {
        self.regions
            .iter()
            .find(|r| r.parent.is_none())
            .map_or(FrameRef::GalaxySpace, |r| r.frame)
    }

    /// THE frame this shard measures everything in: its OWN realm's frame. It is the identity anchor of
    /// [`frame_context`](Self::frame_context) — the shard IS its own origin — and therefore the one frame
    /// its child placements are stamped in and the one frame its observer distances are commensurate in.
    ///
    /// ONE accessor, deliberately, used by BOTH the context builder and every fold that targets it. They
    /// were once two expressions naming two different frames (the context anchored on the shard's own
    /// realm, the proxy fold on the ambient root). While every region sat at the identity those were the
    /// same numbers, so the divergence was invisible; the moment children moved off their parent's origin
    /// the fold started answering from the wrong space, and a shard asking where its own ancestor sits got
    /// the typed refusal it should get — leaving every distant traveller unplaceable. Keeping it one
    /// function is what stops that pair drifting apart again.
    pub(crate) fn own_frame(&self, own_realm: RealmId) -> FrameRef {
        self.regions
            .iter()
            .find(|r| r.realm == own_realm)
            .map_or(self.root_frame(), |r| r.frame)
    }

    /// THE ONE GEOMETRIC FACT A REALM HOLDS ABOUT ITSELF, as a render outline: its own boundary, centred on
    /// its OWN origin, in its own frame. That is not an address — a realm sits at zero in its own frame by
    /// definition, and it goes on doing so wherever its parent has put it — so stating it breaks nothing:
    /// the shard still has no idea where it itself is.
    ///
    /// WHY IT EXISTS. The box a player is standing INSIDE used to be authored by the router, out of its own
    /// copy of the seed forest, and re-expressed into a space the router picked. Nobody else can state it at
    /// login: the shard's parent does reflect it down (the shape lane keeps the path-child's outline at the
    /// origin), but only once the up-relay has reached that parent and its reflect has come back — and on a
    /// realm that is the top of the LIVE chain, never. This is the shard answering for its own realm from
    /// the instant it has an occupant, with the identical value the parent would later send.
    ///
    /// `None` only if this shard's own realm is absent from the forest it was planted with — the same
    /// degenerate [`own_frame`](Self::own_frame) covers, and the honest answer is to state no outline rather
    /// than invent a boundary.
    pub(crate) fn own_shape(&self, own_realm: RealmId) -> Option<RealmShape> {
        self.regions
            .iter()
            .find(|r| r.realm == own_realm)
            .map(|r| RealmShape {
                realm: r.realm,
                frame: r.frame,
                shape: r.shape,
                parent: r.parent,
            })
    }

    /// This shard's own realm's LOOK (real-scale design §3.0 — the outline it DRAWS), read off
    /// the roster. `None` when the realm is absent OR when it is a look-less ambient: either
    /// way the realm states no picture. Off-wire (the region row is boot-derived, never sent).
    pub(crate) fn own_look(&self, own_realm: RealmId) -> Option<vd_core::geometry::Boundary> {
        self.regions
            .iter()
            .find(|r| r.realm == own_realm)
            .and_then(|r| r.look)
    }

    /// The frame of ANY realm this shard holds a region for — the label read straight off the roster.
    ///
    /// Asked about the realm a hand-off NAMES as its destination, which on a shard hosting one realm is
    /// that shard's own realm and on a shard co-hosting a chain (System ⊃ Planet ⊃ Area) may be any link
    /// in it. `None` means this shard carries no region for that realm and therefore cannot name the space
    /// an occupant of it is measured in.
    ///
    /// Read from the roster rather than rebuilt from the realm id deliberately: `frame_for_realm` is a
    /// LOSSY inverse — an `Area` frame carries its enclosing planet's seed as well as its own — and the
    /// region already holds the exact answer.
    #[must_use]
    pub fn hosted_frame(&self, realm: RealmId) -> Option<FrameRef> {
        self.regions
            .iter()
            .find(|r| r.realm == realm)
            .map(|r| r.frame)
    }

    /// The realm whose OWN frame is `frame`, if this shard carries a region for it — the inverse of
    /// [`hosted_frame`](Self::hosted_frame), asked wherever a pose has to say which realm it is measured
    /// in (a pose carries a frame, not a realm id).
    ///
    /// It exists BESIDE `FrameRef::realm()`, which looks like the same question and is not: that one is a
    /// pure label decode and answers even for a realm nobody here has heard of, which is exactly the case
    /// that must NOT be treated as "a realm I can do arithmetic about".
    #[must_use]
    pub fn realm_of_frame(&self, frame: FrameRef) -> Option<RealmId> {
        self.regions
            .iter()
            .find(|r| r.frame == frame)
            .map(|r| r.realm)
    }

    /// THE ONE WRITER's per-instant output for `anchor` (the placement arc): where this shard puts
    /// each of `anchor`'s DIRECT children at `at`, as a [`PlacementBook`] — the instant a PROPERTY OF
    /// THE TABLE, never a parameter of a read. Every consumer (containment, crossing, AoI, the scene
    /// lanes, realm-frame authoring) reads THESE rows; a book carries no clock and no elements, so
    /// nothing downstream can ask HOW a child moves (SL4).
    ///
    /// THE GROUND RULE, in code: only a parent knows where its children are, and a child has no idea
    /// of its own position. So a book holds EXACTLY two things and nothing else:
    ///   - this shard's OWN realm, at the identity — it IS its own origin, and it never learns where
    ///     that origin sits (the book's anchor; computed, never stored — SL1);
    ///   - each DIRECT CHILD, at its placement IN THIS SHARD'S FRAME — which this shard authors, so it
    ///     knows it by definition.
    ///
    /// A parent, grandparent or sibling is deliberately absent: nobody has told this shard where they
    /// are, so a conversion involving one must fail LOUD (the typed error) rather than quietly pretend
    /// the identity and answer confidently from the wrong numbers.
    #[must_use]
    pub fn author_book(&self, anchor: RealmId, tick_hz: f64, at: UniverseTick) -> PlacementBook {
        let secs = secs_since_epoch(at.0, tick_hz);
        PlacementBook::new(
            self.own_frame(anchor),
            at,
            self.direct_children(anchor)
                .map(|r| (r.frame, placement_row(&self.moving, r, secs)))
                .collect(),
        )
    }

    /// The shard's authored placements for EVERY direct child as [`RealmSnap`] observer rows — the
    /// book's rows verbatim, one per child, static and moving alike (owner Q3, the placement arc S3:
    /// "movers only" was a motion test deciding WHAT THE REALM FEED SHIPS — the last of the rival
    /// has-orbit tests, D-FO-7 — and it silently dropped any child placed by non-orbital means). A
    /// static child's row repeats its authored value each tick; the lane is latest-wins/unreliable,
    /// so level-triggered repetition IS its loss story (send-on-change over an unreliable lane would
    /// starve a joiner or a lost datagram forever — bandwidth at true scale is the ledgered P4 owe).
    ///
    /// THE SHARD COMPOSES NOTHING. Each row ships exactly as this shard authored it: the child's
    /// placement measured in THIS shard's own frame, which is the only frame it is entitled to speak
    /// in. It used to be folded here to a universe-root ABSOLUTE, out of a per-tick table this shard
    /// computed by walking its own ancestor chain from the seed — i.e. the shard worked out where IT
    /// was, which is precisely the thing a realm may never know. The conversion into whatever space a
    /// particular viewer draws in happens once, at the gateway, which is the only party holding both
    /// ends of it.
    #[must_use]
    pub fn authored_realm_snaps(&self, own_realm: RealmId, book: &PlacementBook) -> Vec<RealmSnap> {
        self.child_rows(own_realm, book)
            .into_iter()
            .map(|(r, pose)| RealmSnap {
                realm: r.realm,
                // proto_minor 8, the edge HEAD: the CHILD's OWN frame — the frame that realm's
                // occupants are measured in. `pose.frame` beside it is the TAIL, THIS shard's own
                // frame, and `pose.pos` the placement in it, so the row is a complete
                // (head, tail, value) edge that needs no join against the shape lane and no
                // ordering between the two lanes to be read. It is carried rather than derived
                // because `FrameRef::realm` is a LOSSY inverse: an `AreaLocal{planet, area}`
                // collapses to `RealmId::Area(area)`, which nobody can invert without already
                // holding the hierarchy the receiver is trying to build.
                frame: r.frame,
                pose,
            })
            .collect()
    }

    /// The UNIFIED per-tick placement of EVERY DIRECT child (RLM Step 2, H2): ONE authored book
    /// ([`author_book`](Self::author_book)) joined back onto the child regions as stamped poses. ONE
    /// position code-path both the observer feed AND the AoI loop consume — no third position path.
    /// DIRECT children only (`parent == own`); ancestor/self/root regions excluded. Poses stamped in
    /// the shard's OWN frame — the SAME frame `authored_realm_snaps` used, so the AoI distance and the
    /// feed measure one geometry (H-1). Returns `(&RealmRegion, StampedPose)` so callers read
    /// extent/aoi/parent without a re-scan.
    #[must_use]
    pub fn child_placements(
        &self,
        own_realm: RealmId,
        tick_hz: f64,
        tick: UniverseTick,
    ) -> Vec<(&RealmRegion, StampedPose)> {
        let book = self.author_book(own_realm, tick_hz, tick);
        self.child_rows(own_realm, &book)
    }

    /// Join an authored book's rows back onto this shard's DIRECT child regions as stamped poses.
    ///
    /// ONE anchor: THIS SHARD'S OWN frame (the book's anchor). A parent authors its children's
    /// placements in its own frame — it has no other frame to author them in, and under the ground
    /// rule it does not know where the ambient root is. The `expect` states an invariant, never a
    /// hope: the book was authored over this SAME child roster, so every direct child has a row.
    #[must_use]
    pub fn child_rows<'a>(
        &'a self,
        own_realm: RealmId,
        book: &PlacementBook,
    ) -> Vec<(&'a RealmRegion, StampedPose)> {
        self.direct_children(own_realm)
            .map(|r| {
                let at = book
                    .of(r.frame)
                    .expect("the book was authored over this same child roster");
                (r, pose_of_row(book, at))
            })
            .collect()
    }

    /// This shard's DIRECT child regions — the ROSTER half of [`RealmRegions::child_placements`], asked
    /// without placing anybody. Split out so a question that is only about WHICH children exist (is any of
    /// them active?) does not pay for every mover's orbit solve; `child_placements` is defined on top of it,
    /// so the two can never come to disagree about what a direct child is.
    pub(crate) fn direct_children(&self, own_realm: RealmId) -> impl Iterator<Item = &RealmRegion> {
        self.regions
            .iter()
            .filter(move |r| is_direct_child(r.parent, own_realm))
    }
}

/// A region is a DIRECT child iff its parent IS the shard's own realm. A branchless equality (HR5): the
/// `==` is covered true (a child) and false (the root's `None`, an ancestor, a sibling) by any nested
/// forest.
fn is_direct_child(region_parent: Option<RealmId>, own_realm: RealmId) -> bool {
    region_parent == Some(own_realm)
}

/// The `RealmId → RealmLevel` for a hosted child region — sourced un-lossily from the seed via
/// [`level_of`] (the `System(0)`/`System(1)` stand-ins recover to Universe/Galaxy; keyed kinds pass
/// through). Monomorphic (HR5: the kind match is covered once inside `level_of`).
///
/// `None` for an entity-backed `Ship` realm: the lineage coordinate cannot name one — `RealmKindTag`
/// carries six seed-keyed tags and no Ship arm until the P8 ship-realm work (DEFERRED D-SHIP-1). Every
/// lane that needs a child COORD (demand/AoI, cascade, scene reflect, interior fan) EXCLUDES such a
/// region gracefully — counted (`StubStats::ship_child_regions_excluded`), NEVER a panic: this used to
/// be an `expect` that aborted the whole shard on one unrepresentable region (audit :713), structurally
/// excluding a first-class realm kind by crashing instead of by a typed, visible skip.
pub(crate) fn region_level(region: &RealmRegion) -> Option<RealmLevel> {
    level_of(region.realm)
}
