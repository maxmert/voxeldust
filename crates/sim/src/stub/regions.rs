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
use std::collections::{BTreeMap, BTreeSet};
use vd_core::UniverseTick;
use vd_core::child_index::{ChildIndex, IndexedChild};
use vd_core::frame::FramePlacement;
use vd_core::geometry::{DepthKey, RealmRegion};
use vd_core::glam::{DQuat, DVec3};
use vd_core::kinematics::secs_since_epoch;
use vd_core::placement::{MotionFn, PlacementBook};
use vd_core::pose::{FrameRef, LatticePos, RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::RealmLevel;
use vd_core::worldgen::level_of;
use vd_wire::channels::{RealmShape, RealmSnap};

// THE RETIRED WIDTH BOUND (SL9, 2026-08-24). `MAX_REGIONS = 64` lived here: a shard whose region set
// exceeded one machine word failed LOUD at boot, because membership was a bitset with ONE BIT PER
// WATCHED REGION. It is DELETED with that bitset. A parent's child count is unbounded — a galaxy states
// a hundred and fifty thousand star systems as direct children — so no width may be reserved anywhere,
// and the boot fence (`vd_core::geometry::guard_regions_nest`) no longer takes a maximum. Membership is
// now a SHORT LIST of the realms an occupant is inside (its own chain, plus the edge of a crossing);
// see `RegionMembership` in `containment.rs`.

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
    /// ★ EVERY PARENT'S DIRECT CHILDREN, BY REGION INDEX (2026-08-30) — the inverse of `parent`.
    ///
    /// [`Self::direct_children`] used to FILTER the whole region list. That is a scan of everything to
    /// answer a question about one realm, and SL9 names it exactly: finding which realms a parent
    /// holds is a LOOKUP, never a scan, and a cost that grows with the child count is a defect.
    ///
    /// MEASURED (2026-08-30): a window-lane pin walks every anchor of THE world and asks each for its
    /// children. At 233 221 anchors over 233 220 regions the filter is of the order of 5e10 passes —
    /// the sim's test binary sat at 100% of a core and 12.8 GB without finishing.
    ///
    /// Built in index order, so an iteration over a parent's children yields exactly the order the
    /// filter yielded.
    /// ★ A SET, NOT A LIST (SL9, measured 2026-09-03): a release removed its index by a linear `retain`
    /// over the galaxy's 233 220 systems, twice; a set removes one entry in O(log n) and iterates in
    /// the same index order the list did.
    pub(crate) children_of: BTreeMap<RealmId, BTreeSet<usize>>,
    /// Region index of each realm — the inverse of `regions`, so a realm resolves to its bit without a scan.
    pub(crate) ix_of: BTreeMap<RealmId, usize>,
    /// SELF ∪ ANCESTORS as a SET OF REALMS, per region index (task #177). Computed ONCE at boot by the same
    /// parent walk the depth cache already does. Its size is the forest's DEPTH (about six), never its
    /// breadth — which is why removing the old one-bit-per-region form removed the child-count cap (SL9).
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
    pub(crate) ancestor_chain: Vec<BTreeSet<RealmId>>,
    /// THE CHILD INDEX (SL9) — "which of my direct children could hold this point", so the containment
    /// fold never walks all of them. Built over the STATIC direct children only; a moving child is
    /// omitted and stays an unconditional candidate, which is conservative (see
    /// [`vd_core::child_index::ChildIndex::build`]).
    ///
    /// EMPTY until [`RealmRegions::with_own_realm`] names the realm whose children these are — a shard
    /// cannot tell which regions are its own children from the forest alone. An empty index indexes
    /// nothing, so nothing is skipped and the fold behaves exactly as it did before this existed. That
    /// is the SAFE default on purpose: forgetting to build it costs speed, never correctness.
    pub(crate) child_index: ChildIndex,
    /// ★ THE STATIC ROWS, AUTHORED ONCE PER ANCHOR (owner ruling 2026-09-02 R8 item 1; SL9): each
    /// parent's non-moving children as placement rows, built when the forest is, shared by every
    /// tick's book through the `Arc`. A tick authors only the movers and the driven.
    static_rows: BTreeMap<RealmId, std::sync::Arc<Vec<(FrameRef, FramePlacement)>>>,
    /// Each parent's MOVING children (indices), so a tick's overlay walks the movers and not the
    /// roster.
    movers_of: BTreeMap<RealmId, Vec<usize>>,
    /// ★ THE RANGE INDEX (owner ruling 2026-09-02 R8 item 1): this realm's static direct children
    /// with a live band, spanned by their tear-down radius plus the widest child extent (a child
    /// proxy reaches out to its own surface). A looker finds the children whose band can hold it
    /// by walking its lead through this grid — one lookup per cell of length, never a walk of the
    /// roster. EMPTY until `with_own_realm`, like `child_index`.
    aoi_index: ChildIndex,
    // ★ THE REACH (owner ruling 2026-09-02 R3/R6; built 2026-09-04 under the owner's rule *"brightness
    // counts only when no ancestor already draws that light"*).
    /// Each direct child's STATED reach — (by size, by light), whole metres — the datum the child
    /// sends up on change. Absent until the child states; a dormant child never states, so the
    /// parent's default (its look and its planted light) serves.
    reach_stated: BTreeMap<RealmId, (u64, u64)>,
    /// The (fence, tick) of each child's newest reach statement — the staleness order.
    reach_at: BTreeMap<RealmId, (vd_core::fence::Fence, vd_core::ids::UniverseTick)>,
    /// Each direct child's light, in Suns, as the boot planted it — the parent's own default for a
    /// child that has not stated, and the light the child's band is widened by.
    child_light: BTreeMap<RealmId, f64>,
    /// Does THIS realm draw its children's light itself (the galaxy's star field)? Then a child's
    /// light never widens its band and never folds into this realm's own reach by light.
    lights_children: bool,
    /// The terms this realm's OWN reach folds over its direct children — `distance + reach`, by
    /// size and by light — as sorted multisets (value bits → count) plus each child's own pair, so
    /// a statement costs O(log n) on the galaxy and never a walk (SL9).
    reach_size_terms: BTreeMap<u64, u32>,
    reach_light_terms: BTreeMap<u64, u32>,
    reach_terms_of: BTreeMap<RealmId, (u64, u64)>,
    /// The same children sorted by distance from this realm's centre, in metres of its own frame,
    /// for the OUTSIDE looker: a looker `d` metres out can only reach children at least
    /// `d − widest band` from the centre, which is a suffix of this list.
    /// Keyed by the distance's bit pattern — monotone for a non-negative float — so an insert, a
    /// removal and the suffix are each one ordered-map operation, never a shift of every entry after
    /// it (SL9, measured 2026-09-03).
    radial: BTreeMap<u64, Vec<RealmId>>,
    /// The widest tear-down band among the indexed children, in metres.
    widest_band_m: f64,
    /// The widest circumscribed extent among the indexed children, in metres — the pad every
    /// interest-index entry carries. Kept as a number so an adoption never folds over every child to
    /// learn it (SL9, measured: that fold was one of the three walks). It only grows: a released
    /// child leaves the pad conservative, which changes no verdict.
    widest_child_extent_m: f64,
    /// The rows the child index does NOT answer for — the ancestors, this realm, the movers — the
    /// only rows a subject evaluation must always ask. ★ SL9, MEASURED on the fifth flight
    /// (2026-09-03): the evaluation asked every row and asked the index whether it answered for each,
    /// 233 220 times per subject per tick on the galaxy — 325 ms a tick, the window went stale, the
    /// pilot's picture froze. Kept as a list so the evaluation never walks the children.
    unindexed: Vec<RealmId>,
    /// The BUILT rows of this forest — the ones with an exterior key (a hull), whose lease their
    /// parent requests and renews; the reader filters by parent with one lookup each. ★ SL9, MEASURED on the fifth flight (2026-09-03): the lease loop
    /// filtered ALL 233 220 of the galaxy's children every tick to find the one hull — a fifth of a
    /// 172 ms tick. A hull is rare among children, so the list is short and the walk is gone.
    built_children: Vec<RealmId>,
    /// Bumped on every change to the child roster (an adoption, a release, a re-parenting, the
    /// movers named), so a reader that remembers an answer about the children knows when it is stale.
    /// Drawn from ONE process-wide counter, so a table built later never repeats an earlier table's
    /// number — a memo taken against a forest that was then replanted is stale too.
    roster_generation: u64,
    /// The realm this shard hosts, once stated. `None` on a rig that never named one.
    own_realm: Option<RealmId>,
}

/// SELF ∪ ANCESTORS of `realm`, as the SET OF REALMS ITSELF. Mirrors [`region_depth`]'s parent walk
/// exactly — same hop cap, same dangling-parent and cycle stops — so the two caches can never disagree
/// about the shape of the forest. A realm not present in `ix_of` contributes nothing (it is not a region
/// here).
///
/// **This used to be a `u64` bitmask, one bit per region index, and that is what capped a parent at 64
/// children (SL9).** The chain is a property of DEPTH, never of breadth: it holds the realm and its
/// ancestors, about six entries in the deepest world we generate, whether the parent has one sibling or
/// a hundred and fifty thousand. Naming the realms directly is the same information with no width to
/// exceed — and it is the SAME key the membership set uses, so the two can be compared without an index.
fn ancestry_chain(
    regions: &[RealmRegion],
    ix_of: &BTreeMap<RealmId, usize>,
    realm: RealmId,
) -> BTreeSet<RealmId> {
    let mut chain = BTreeSet::new();
    let mut cur = realm;
    for _ in 0..regions.len() {
        // ★ A LOOKUP, NOT A SCAN (2026-08-29). This walked the WHOLE region list on every hop, to
        // find a realm whose index `ix_of` was already holding — the index is this function's own
        // argument, one line above the scan. SL9 names the shape: finding which realm a point or a
        // name belongs to is a LOOKUP, never a scan, and a cost that grows with the child count is a
        // defect. MEASURED: with a galaxy shard's 233 220 children the sim's own test binary ran
        // twenty minutes at 199% of two cores and had not finished.
        let Some(&ix) = ix_of.get(&cur) else {
            return chain; // dangling parent — stop where region_depth stops
        };
        chain.insert(cur);
        let Some(parent) = regions[ix].parent else {
            return chain; // reached the ambient root
        };
        cur = parent;
    }
    chain // hop cap hit — a cycle (boot-guard-rejected); a safe stop, never a hang
}

/// EVERY REGION'S DEPTH — [`region_depth`]'s exact walk, with a LOOKUP where it had a scan.
///
/// ★ WHY IT IS A PLAIN WALK AND NOT A MEMOISED ONE (2026-08-30). The first version of this shared
/// work between regions by remembering solved depths. It was WRONG in two branches — it credited a
/// remembered ancestor's depth to that ancestor's child, and it lost a hop on a dangling parent —
/// and depth drives the containment argmax, so every re-home in the suite silently went to zero.
///
/// A chain is about six realms deep and that is a property of nesting, not of census. So the walk
/// costs the forest's DEPTH per region and nothing more, and it is worth no cleverness at all. The
/// defect this replaces was the scan inside it, never the walk around it.
fn region_depths(
    regions: &[RealmRegion],
    ix_of: &BTreeMap<RealmId, usize>,
) -> Vec<(u32, RealmId, usize)> {
    regions
        .iter()
        .enumerate()
        .map(|(ix, r)| (depth_of(regions, ix_of, r.realm), r.realm, ix))
        .collect()
}

/// ONE realm's depth — the hop count `region_depth` returns, monomorphic so both stops are covered
/// once (HR5): a realm the forest does not name, and the ambient root's absent parent.
/// The process-wide roster change counter (see `RealmRegions::roster_generation`): every table's
/// every change draws the next number, so no two rosters ever share one. A counter, never a clock.
fn next_roster_generation() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn depth_of(regions: &[RealmRegion], ix_of: &BTreeMap<RealmId, usize>, realm: RealmId) -> u32 {
    let mut depth = 0u32;
    let mut cur = realm;
    for _ in 0..regions.len() {
        let Some(&ix) = ix_of.get(&cur) else {
            return depth; // dangling parent — exactly where `region_depth` stops
        };
        let Some(parent) = regions[ix].parent else {
            return depth; // the ambient root
        };
        depth += 1;
        cur = parent;
    }
    depth // hop cap hit — a cycle the boot guard rejects; a safe stop, never a hang
}

/// What a child's reach statement did to this realm's table (the reach, 2026-09-04).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReachOutcome {
    /// Not one of this realm's direct children — the caller's misroute guard should have said so.
    NotMine,
    /// Older than the statement already held.
    Stale,
    /// Held and re-banded; `own_changed` says this realm's own reach moved and is owed upward.
    Applied { own_changed: bool },
}

impl RealmRegions {
    /// ★THROWAWAY (test instrument): plant the cruise-only overdrive read from `VD_TEST_OVERDRIVE`.
    /// Values below the lawful `1.0` are refused back to it — the instrument may only ever go faster,
    /// and the RETURNED value is what was planted, not what was asked for. Both arms are pinned by
    /// `the_cruise_overdrive_instrument_may_only_ever_go_faster`; the shard binary is its only other
    /// caller, and a binary is outside the coverage domain.
    pub fn set_cruise_overdrive(&mut self, factor: f64) -> f64 {
        self.cruise_overdrive = factor.max(1.0);
        self.cruise_overdrive
    }

    /// Build the resource from a region forest, computing the depth-key cache + the ambient-root realm
    /// ONCE (regions are static at P3). The per-tick detector reads the cache; it never re-walks parents.
    /// The moving-child roster starts EMPTY — [`with_moving_children`](Self::with_moving_children) adds it.
    //
    // (The instrument above was inserted between this doc and its own function, so `new` had lost
    // both its documentation and its `#[must_use]` to a setter that wants neither.)
    #[must_use]
    pub fn new(regions: Vec<RealmRegion>) -> RealmRegions {
        let ix_of: BTreeMap<RealmId, usize> = regions
            .iter()
            .enumerate()
            .map(|(ix, r)| (r.realm, ix))
            .collect();
        // ★ EVERY DEPTH IN ONE PASS (2026-08-29). This called `region_depth` per region, and that
        // function walks the parent chain by SCANNING the whole region list at every hop. Two nested
        // walks over the same list: at a galaxy shard's 233 220 regions it is of the order of 5e10
        // comparisons before a shard answers anything.
        //
        // A parent's depth is its child's depth minus one, so the answers share their work. This
        // memoises the chain it is already walking and touches each region a bounded number of times.
        let mut children_of: BTreeMap<RealmId, BTreeSet<usize>> = BTreeMap::new();
        for (ix, r) in regions.iter().enumerate() {
            if let Some(parent) = r.parent {
                children_of.entry(parent).or_default().insert(ix);
            }
        }
        let depths = region_depths(&regions, &ix_of);
        let root_realm = regions.iter().find(|r| r.parent.is_none()).map(|r| r.realm);
        let ancestor_chain = regions
            .iter()
            .map(|r| ancestry_chain(&regions, &ix_of, r.realm))
            .collect();
        let unindexed_at_start: Vec<RealmId> = regions.iter().map(|r| r.realm).collect();
        let built_at_start: Vec<RealmId> = regions
            .iter()
            .filter(|r| matches!(r.realm, RealmId::Ship(_)))
            .map(|r| r.realm)
            .collect();
        let mut built = RealmRegions {
            // ★THROWAWAY: the lawful 1.0 on every rig — only a bin that read `VD_TEST_OVERDRIVE`
            // raises it, through `set_cruise_overdrive`.
            cruise_overdrive: 1.0,
            regions,
            children_of,
            depths,
            root_realm,
            moving: BTreeMap::new(),
            ix_of,
            ancestor_chain,
            child_index: ChildIndex::default(),
            static_rows: BTreeMap::new(),
            movers_of: BTreeMap::new(),
            aoi_index: ChildIndex::default(),
            reach_stated: BTreeMap::new(),
            reach_at: BTreeMap::new(),
            child_light: BTreeMap::new(),
            lights_children: false,
            reach_size_terms: BTreeMap::new(),
            reach_light_terms: BTreeMap::new(),
            reach_terms_of: BTreeMap::new(),
            radial: BTreeMap::new(),
            widest_band_m: 0.0,
            widest_child_extent_m: 0.0,
            // Nothing is indexed until a realm is named, so EVERY row is asked (the fixtures that
            // plant a forest without naming a realm are exactly the pre-index behaviour).
            unindexed: unindexed_at_start,
            built_children: built_at_start,
            roster_generation: next_roster_generation(),
            own_realm: None,
        };
        // The static rows exist from construction: a store with no movers named and no realm named
        // still authors every child's row (the fixtures build exactly that store).
        built.rebuild_static_rows();
        built
    }

    /// SELF ∪ ANCESTORS as a set of realms, for the realm a subject is authoritatively in — the DERIVED
    /// hysteresis prior. EMPTY for a realm this shard does not host, which degrades to the old blank-prior
    /// behaviour rather than inventing membership; callers warn on that path rather than passing it off as
    /// normal (a pose naming an unhosted realm means a rebind safe-degrade happened upstream).
    #[must_use]
    pub fn ancestor_chain_for(&self, realm: RealmId) -> &BTreeSet<RealmId> {
        const EMPTY: &BTreeSet<RealmId> = &BTreeSet::new();
        self.ix_of
            .get(&realm)
            .and_then(|ix| self.ancestor_chain.get(*ix))
            .unwrap_or(EMPTY)
    }

    /// Register the shard's DIRECT MOVING children (FA-2b): the `(realm, motion)` roster the BOOT
    /// composition derives for the hosted realm and injects as opaque [`MotionFn`]s. A builder (not a
    /// `new` arg) so the many `RealmRegions::new` call sites stay unchanged and byte-identical (the
    /// walk roster passes an empty map). Only registered realms' rows author live; every other region
    /// stays static.
    /// The roster's change counter (see the field).
    #[must_use]
    pub(crate) fn roster_generation(&self) -> u64 {
        self.roster_generation
    }

    #[must_use]
    pub fn with_moving_children(mut self, moving: BTreeMap<RealmId, MotionFn>) -> RealmRegions {
        self.roster_generation = next_roster_generation();
        self.moving = moving;
        self.rebuild_child_index();
        self
    }

    /// Name the realm this shard hosts, which is what lets the forest be split into "my direct children"
    /// (indexable) and "everything else" (always evaluated). Builds the child index (SL9).
    ///
    /// A builder rather than a `new` argument so the many existing `RealmRegions::new` call sites stay
    /// unchanged and byte-identical: without it the index is empty, nothing is skipped, and the fold
    /// walks the forest exactly as it always did.
    #[must_use]
    pub fn with_own_realm(mut self, own_realm: RealmId) -> RealmRegions {
        self.own_realm = Some(own_realm);
        self.rebuild_child_index();
        self.refresh_reach();
        self
    }

    /// ★ THE CHILDREN'S LIGHT (the reach): the per-child luminosity the boot planted — the same
    /// datum the marker roster carries. A parent widens a child's wake band by its light and folds
    /// it into its own reach by light, unless the parent draws that light itself.
    #[must_use]
    pub fn with_child_light(mut self, luma: &BTreeMap<RealmId, (u8, f64)>) -> RealmRegions {
        self.child_light = luma.iter().map(|(r, (_, l))| (*r, *l)).collect();
        self.refresh_reach();
        self
    }

    /// ★ THIS REALM DRAWS ITS CHILDREN'S LIGHT (the reach): the galaxy, whose star field is every
    /// star's point of light, shipped once. Its children's light then widens no band and folds into
    /// no reach — a star system wakes for its disc or its planets, never for the glow the sky shows.
    #[must_use]
    pub fn with_lights_children(mut self, lights: bool) -> RealmRegions {
        self.lights_children = lights;
        self.refresh_reach();
        self
    }

    /// Re-band every live direct child by its default reach and rebuild the reach terms. Called by
    /// the builders so their order cannot matter. A child's band is re-inserted in the range index
    /// only when it changed — for the galaxy (which lights its children) nothing changes and nothing
    /// is touched.
    fn refresh_reach(&mut self) {
        let Some(own) = self.own_realm else {
            return;
        };
        let ixs: Vec<usize> = self
            .children_of
            .get(&own)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        // Re-band in place (the band alone), then rebuild the range index ONCE in bulk: on the
        // galaxy this is 233 220 children, and a leaf insert per child took the boot from four
        // seconds to thirty-eight (MEASURED on the ninth flight, 2026-09-04).
        for &ix in &ixs {
            let realm = self.regions[ix].realm;
            let radius = self.test_radius_of(realm);
            let next = self.regions[ix].aoi.with_spin_up(radius);
            self.regions[ix].aoi = next;
        }
        self.rebuild_aoi_index(own);
        self.reach_size_terms.clear();
        self.reach_light_terms.clear();
        self.reach_terms_of.clear();
        for ix in ixs {
            self.insert_reach_terms(own, ix);
        }
    }

    /// A child's reach by size: stated, else the dot-angle reach of its own look.
    fn size_reach_of(&self, realm: RealmId) -> f64 {
        let by_look = match self.reach_stated.get(&realm) {
            Some((size, _)) => *size as f64,
            None => self
                .region_of(realm)
                .and_then(|r| r.look)
                .map_or(0.0, |look| {
                    vd_core::geometry::visibility_reach_m(
                        look.circumscribed_extent(),
                        vd_core::geometry::VISIBILITY_THETA_MIN_RAD,
                    )
                }),
        };
        // ★ THE REACH INCLUDES THE SHELL (owner 2026-09-05, "sounds good"): a realm you can enter must
        // be awake, so its reach is never smaller than its bound. For a planet the light reach (50 AU)
        // already dwarfs its sphere of influence; for a star system the galaxy draws its star, so its
        // reach by look (80 AU) fell 190× short of its 0.24 ly shell and a hull crossed in asleep.
        // One radius, one datum, the same index — the shell is the floor. Example: a hull at warp
        // closes on a star system; at its shell plus the closing lead the galaxy wakes it.
        let shell = self
            .region_of(realm)
            .map_or(0.0, |r| r.shape.circumscribed_extent());
        by_look.max(shell)
    }

    /// A child's reach by light: stated, else the limiting-magnitude reach of its planted light.
    fn light_reach_of(&self, realm: RealmId) -> f64 {
        match self.reach_stated.get(&realm) {
            Some((_, light)) => *light as f64,
            None => vd_core::look::light_reach_m(
                self.child_light.get(&realm).copied().unwrap_or(0.0),
                vd_core::look::LIMITING_MAGNITUDE,
            ),
        }
    }

    /// The radius this realm TESTS a child at: the larger of its two reaches — size alone when this
    /// realm draws the child's light itself.
    fn test_radius_of(&self, realm: RealmId) -> f64 {
        let light = if self.lights_children {
            0.0
        } else {
            self.light_reach_of(realm)
        };
        self.size_reach_of(realm).max(light)
    }

    /// Re-band one child at `radius` and, when the band changed, re-insert its leaf in the range
    /// index. A mover or an inert child keeps its band as it is.
    fn reband(&mut self, own: RealmId, ix: usize, radius: f64) {
        let realm = self.regions[ix].realm;
        let next = self.regions[ix].aoi.with_spin_up(radius);
        if next == self.regions[ix].aoi {
            return;
        }
        self.regions[ix].aoi = next;
        if self.moving.contains_key(&realm) {
            return;
        }
        let tier = self.own_frame(own).tier();
        let centre = self.regions[ix].center.in_parents_frame();
        self.widest_band_m = self.widest_band_m.max(next.tear_down_r_m());
        self.aoi_index.insert(
            &IndexedChild {
                realm,
                centre,
                radius_m: next.tear_down_r_m() + self.widest_child_extent_m,
            },
            tier,
        );
    }

    /// The child's `distance + reach` pair, from its placement in this realm's frame. `own` is this
    /// realm and `ix` one of its direct children's rows — every caller holds both already, so nothing
    /// here re-tests whose child this is.
    fn reach_terms_for(&self, own: RealmId, ix: usize) -> (u64, u64) {
        let r = &self.regions[ix];
        let realm = r.realm;
        let tier = self.own_frame(own).tier();
        let d = r
            .center
            .in_parents_frame()
            .delta_m(LatticePos::ORIGIN, tier)
            .length();
        let light = if self.lights_children {
            0.0
        } else {
            self.light_reach_of(realm)
        };
        let size = self.size_reach_of(realm);
        // A child with no reach of a kind contributes nothing of that kind: a dark moon does not
        // make its planet visible by light from the moon's distance.
        let term = |reach: f64| {
            if reach > 0.0 {
                (d + reach).round() as u64
            } else {
                0
            }
        };
        (term(size), term(light))
    }

    fn insert_reach_terms(&mut self, own: RealmId, ix: usize) {
        let realm = self.regions[ix].realm;
        let (size, light) = self.reach_terms_for(own, ix);
        *self.reach_size_terms.entry(size).or_insert(0) += 1;
        *self.reach_light_terms.entry(light).or_insert(0) += 1;
        self.reach_terms_of.insert(realm, (size, light));
    }

    fn remove_reach_terms(&mut self, realm: RealmId) {
        let Some((size, light)) = self.reach_terms_of.remove(&realm) else {
            return;
        };
        for (terms, key) in [
            (&mut self.reach_size_terms, size),
            (&mut self.reach_light_terms, light),
        ] {
            // The count is there by construction: `insert_reach_terms` writes the name and both
            // counts in one go, and only this function takes them out again. A last one leaves.
            let left = terms.get(&key).copied().unwrap_or(0).saturating_sub(1);
            if left == 0 {
                terms.remove(&key);
            } else {
                terms.insert(key, left);
            }
        }
    }

    /// The top of both multisets — what this realm's own reach folds from its children.
    fn children_reach(&self) -> (u64, u64) {
        (
            self.reach_size_terms
                .last_key_value()
                .map_or(0, |(k, _)| *k),
            self.reach_light_terms
                .last_key_value()
                .map_or(0, |(k, _)| *k),
        )
    }

    /// ★ THIS REALM'S OWN REACH — (by size, by light), whole metres: the larger of its own look's
    /// reach and the farthest `distance + reach` over its direct children. `own_luma` is this realm's
    /// own light, as its parent planted it. Example: a planet 6 400 km across with a station 40 000 km
    /// out that reaches 200 000 km states size 490 000 km (its own disc) and light 50 AU (its own
    /// reflected sunlight).
    #[must_use]
    pub fn own_reach(&self, own: RealmId, own_luma: Option<f64>) -> (u64, u64) {
        let (kids_size, kids_light) = self.children_reach();
        // The own reach by look, floored by the own shell (the reach includes the shell, 2026-09-05).
        let own_shell = self
            .region_of(own)
            .map_or(0.0, |r| r.shape.circumscribed_extent());
        let size = self
            .own_look(own)
            .map_or(0.0, |look| {
                vd_core::geometry::visibility_reach_m(
                    look.circumscribed_extent(),
                    vd_core::geometry::VISIBILITY_THETA_MIN_RAD,
                )
            })
            .max(own_shell)
            .round() as u64;
        let light = vd_core::look::light_reach_m(
            own_luma.unwrap_or(0.0),
            vd_core::look::LIMITING_MAGNITUDE,
        )
        .round() as u64;
        (size.max(kids_size), light.max(kids_light))
    }

    /// ★ A CHILD STATES ITS REACH: re-band it, re-insert its leaf, refold this realm's own terms.
    /// Returns the outcome; `Applied { own_changed }` says whether this realm's own reach moved and
    /// must be restated upward.
    pub fn set_child_reach(
        &mut self,
        child: RealmId,
        at: (vd_core::fence::Fence, vd_core::ids::UniverseTick),
        size_reach_m: u64,
        light_reach_m: u64,
    ) -> ReachOutcome {
        let Some(own) = self.own_realm else {
            return ReachOutcome::NotMine;
        };
        let Some(&ix) = self.ix_of.get(&child) else {
            return ReachOutcome::NotMine;
        };
        if self.regions[ix].parent != Some(own) {
            return ReachOutcome::NotMine;
        }
        if self.reach_at.get(&child).is_some_and(|held| at < *held) {
            return ReachOutcome::Stale;
        }
        let before = self.children_reach();
        self.remove_reach_terms(child);
        self.reach_stated
            .insert(child, (size_reach_m, light_reach_m));
        self.reach_at.insert(child, at);
        let radius = self.test_radius_of(child);
        self.reband(own, ix, radius);
        self.insert_reach_terms(own, ix);
        ReachOutcome::Applied {
            own_changed: self.children_reach() != before,
        }
    }

    /// Rebuild the child index from the current own-realm and moving-child statements. Called by both
    /// builders so their ORDER cannot matter — a caller that names the realm first and the movers second
    /// gets the same index as one that does it the other way round, which is the kind of ordering trap
    /// that is invisible until a mover is wrongly indexed.
    /// ★ A CHILD ARRIVES AT RUNTIME (the ruler switch, slice 2; owner-approved 2026-09-03): a hull the
    /// crossing just handed to this realm becomes a direct child of `own_realm` WITHOUT a rebuild of
    /// anything that grows with the child count (SL9). The region row is appended; its index, its
    /// depth and its ancestor chain are one entry each; the containment and area-of-interest grids
    /// take one insert; the radial list takes one sorted insert. The static rows are NOT touched: a
    /// driven child's row is the per-tick overlay `author_book_driven` writes, and it never needed a
    /// static row to be placed. A realm already on the roster is replaced in place (a re-driven adopt).
    ///
    /// Example: the galaxy adopts a hull that left System 7. Its forest grows by one row among
    /// 233,220, the hull's berth lands in one grid cell, and the galaxy's next tick authors the hull's
    /// placement from its driven state.
    pub fn adopt_child(&mut self, region: RealmRegion) {
        let Some(own) = self.own_realm else {
            return; // no realm named ⇒ this shard authors nobody: refuse silently, like the index
        };
        if region.parent != Some(own) {
            return; // only MY direct children are mine to adopt
        }
        if self.ix_of.contains_key(&region.realm) {
            self.release_child(region.realm);
        }
        self.roster_generation = next_roster_generation();
        let ix = self.regions.len();
        let realm = region.realm;
        self.regions.push(region);
        self.ix_of.insert(realm, ix);
        self.children_of.entry(own).or_default().insert(ix);
        let depth = depth_of(&self.regions, &self.ix_of, realm);
        self.depths.push((depth, realm, ix));
        self.ancestor_chain
            .push(ancestry_chain(&self.regions, &self.ix_of, realm));
        let tier = self.own_frame(own).tier();
        let r = &self.regions[ix];
        if self.moving.contains_key(&realm) {
            self.unindexed.push(realm); // a mover is never indexed: always asked
        }
        // The mover list the book's overlay and the moving set read — `release_child` already
        // keeps it across a swap-remove, so an adopted mover belongs in it too. A SHIP is a mover
        // for the placement layer whether or not it moves yet (2026-09-05): its row is the
        // overlay's, never a static row, so adopting or releasing it copies nothing.
        if self.moving.contains_key(&realm) | matches!(realm, RealmId::Ship(_)) {
            self.movers_of.entry(own).or_default().push(ix);
        }
        if matches!(realm, RealmId::Ship(_)) {
            self.built_children.push(realm);
        }
        if !self.moving.contains_key(&realm) {
            self.child_index.insert(
                &IndexedChild {
                    realm,
                    centre: r.center.in_parents_frame(),
                    radius_m: r.shape.circumscribed_extent() + r.band.outset(),
                },
                tier,
            );
            if r.aoi.spin_up_r_m() > 0.0 {
                // The aoi grid's radius carries the widest extent among the live children, as the
                // boot folded it; a newcomer wider than that widest grows the number, and no fold
                // over the children happens here (SL9, measured 2026-09-03).
                self.widest_child_extent_m = self
                    .widest_child_extent_m
                    .max(r.shape.circumscribed_extent());
                let widest = self.widest_child_extent_m;
                self.aoi_index.insert(
                    &IndexedChild {
                        realm,
                        centre: r.center.in_parents_frame(),
                        radius_m: r.aoi.tear_down_r_m() + widest,
                    },
                    tier,
                );
                self.widest_band_m = self.widest_band_m.max(r.aoi.tear_down_r_m());
                let d = r
                    .center
                    .in_parents_frame()
                    .delta_m(LatticePos::ORIGIN, tier)
                    .length();
                let names = self.radial.entry(d.to_bits()).or_default();
                names.push(realm);
                names.sort_unstable();
            }
        }
        // The reach (2026-09-04): the newcomer's `distance + reach` joins this realm's own fold.
        self.insert_reach_terms(own, ix);
        // The newcomer joins its parent's static layer the same tick (2026-09-05) — unless it is
        // a ship, whose row is the overlay's: nothing to copy.
        if !matches!(realm, RealmId::Ship(_)) {
            self.rebuild_static_rows_for(own);
        }
    }

    /// ★ THIS REALM MOVED HOUSE (the ruler switch, slice 3): its parent told it a new lineage. The own
    /// row's parent is re-pointed and every derived per-row table recomputed (a hull's forest is a few
    /// rows: itself, its ancestors as it booted, its children). The old ancestors' rows STAY: they are
    /// the ambient root the container fold seeds from, and a stale ancestor can only ever refuse to
    /// claim a point it cannot measure (a cross-unit refusal is a non-member). Nothing here says
    /// where this realm IS — only whose it is.
    pub(crate) fn reparent_own(&mut self, new_parent: RealmId) {
        let Some(own) = self.own_realm else {
            return;
        };
        let Some(&ix) = self.ix_of.get(&own) else {
            return;
        };
        let old_parent = self.regions[ix].parent;
        if old_parent == Some(new_parent) {
            return;
        }
        if let Some(list) = old_parent.and_then(|op| self.children_of.get_mut(&op)) {
            list.remove(&ix);
        }
        self.roster_generation = next_roster_generation();
        self.regions[ix].parent = Some(new_parent);
        self.children_of.entry(new_parent).or_default().insert(ix);
        self.depths = region_depths(&self.regions, &self.ix_of);
        self.ancestor_chain = self
            .regions
            .iter()
            .map(|r| ancestry_chain(&self.regions, &self.ix_of, r.realm))
            .collect();
    }

    /// ★ A CHILD LEAVES AT RUNTIME (the ruler switch, slice 2): the mirror of [`Self::adopt_child`].
    /// The row is swap-removed and the row that took its slot is re-indexed; every per-row table
    /// follows the same swap. A realm not on the roster is a no-op, so a release can be re-driven.
    pub fn release_child(&mut self, realm: RealmId) {
        let Some(&ix) = self.ix_of.get(&realm) else {
            return;
        };
        self.roster_generation = next_roster_generation();
        let last = self.regions.len() - 1;
        let parent = self.regions[ix].parent;
        // Drop the leaving row from its parent's child list, the grids and the radial list.
        if let Some(list) = parent.and_then(|p| self.children_of.get_mut(&p)) {
            list.remove(&ix);
        }
        self.child_index.remove(realm);
        self.aoi_index.remove(realm);
        self.remove_reach_terms(realm);
        self.reach_stated.remove(&realm);
        self.reach_at.remove(&realm);
        self.unindexed.retain(|r| *r != realm);
        self.built_children.retain(|r| *r != realm);
        // The leaving row's own radial entry, by its own distance — never a walk of the list (SL9).
        let tier = parent
            .map_or(self.root_frame(), |p| self.own_frame(p))
            .tier();
        let d = self.regions[ix]
            .center
            .in_parents_frame()
            .delta_m(LatticePos::ORIGIN, tier)
            .length();
        if let Some(names) = self.radial.get_mut(&d.to_bits()) {
            names.retain(|r| *r != realm);
            if names.is_empty() {
                self.radial.remove(&d.to_bits());
            }
        }
        self.ix_of.remove(&realm);
        // Swap-remove, then point every index-keyed table at the moved row's new slot.
        self.regions.swap_remove(ix);
        self.depths.swap_remove(ix);
        self.ancestor_chain.swap_remove(ix);
        if ix != last {
            let moved = self.regions[ix].realm;
            self.ix_of.insert(moved, ix);
            self.depths[ix].2 = ix;
            let moved_parent = self.regions[ix].parent;
            if let Some(list) = moved_parent.and_then(|mp| self.children_of.get_mut(&mp)) {
                list.remove(&last);
                list.insert(ix);
            }
            if let Some(movers) = parent.and_then(|p| self.movers_of.get_mut(&p)) {
                movers.retain(|i| *i != ix);
            }
            for movers in self.movers_of.values_mut() {
                for i in movers.iter_mut() {
                    if *i == last {
                        *i = ix;
                    }
                }
            }
        } else if let Some(movers) = parent.and_then(|p| self.movers_of.get_mut(&p)) {
            movers.retain(|i| *i != ix);
        }
        // The released child leaves its parent's static layer the same tick (2026-09-05) — a ship
        // was never in it (its row is the overlay's), so its release copies nothing.
        if let Some(p) = parent
            && !matches!(realm, RealmId::Ship(_))
        {
            self.rebuild_static_rows_for(p);
        }
    }

    /// The shared static layer of one parent, as the book holds it (a test reads its identity to
    /// prove a ship's adoption or release copies nothing).
    #[cfg(test)]
    pub(crate) fn static_rows_of(
        &self,
        parent: RealmId,
    ) -> Option<&std::sync::Arc<Vec<(FrameRef, FramePlacement)>>> {
        self.static_rows.get(&parent)
    }

    /// ★ ONE PARENT'S STATIC LAYER, REBUILT (2026-09-05, the twenty-fourth flight: the star jumped
    /// to the hull's OLD BERTH for one tick at re-adoption). The static rows are authored once and
    /// shared by `Arc`, and a release or an adoption never touched them — so a released hull's berth
    /// row stayed in its old parent's book, and the first level after re-adoption (authored before
    /// the adoption landed that tick) placed the hull at the berth. This rebuilds ONLY the named
    /// parent's rows and movers (O(that parent's children), once per event — never per tick), so
    /// a static child that leaves is gone from the book the same tick, and one that arrives is in it.
    fn rebuild_static_rows_for(&mut self, parent: RealmId) {
        let anchor = self.own_frame(parent);
        let mut rows: Vec<(FrameRef, FramePlacement)> = Vec::new();
        let mut movers: Vec<usize> = Vec::new();
        for &ix in self.children_of.get(&parent).into_iter().flatten() {
            let r = &self.regions[ix];
            if self.moving.contains_key(&r.realm) | matches!(r.realm, RealmId::Ship(_)) {
                movers.push(ix);
                continue;
            }
            rows.push((
                r.frame,
                FramePlacement {
                    origin_cell: r.center.in_parents_frame().cell(),
                    origin: r.center.in_parents_frame().offset(),
                    velocity: DVec3::ZERO,
                    orientation: DQuat::IDENTITY,
                    angular_velocity: DVec3::ZERO,
                },
            ));
        }
        self.static_rows
            .insert(parent, PlacementBook::static_rows(anchor, rows));
        if movers.is_empty() {
            self.movers_of.remove(&parent);
        } else {
            self.movers_of.insert(parent, movers);
        }
    }

    fn rebuild_child_index(&mut self) {
        self.rebuild_static_rows();
        let Some(own) = self.own_realm else {
            return; // no realm named ⇒ nothing is known to be a child ⇒ index nothing
        };
        self.rebuild_aoi_index(own);
        let tier = self.own_frame(own).tier();
        let children: Vec<IndexedChild> = self
            .regions
            .iter()
            .filter(|r| r.parent == Some(own) && !self.moving.contains_key(&r.realm))
            .map(|r| IndexedChild {
                realm: r.realm,
                // The index is BUILT AT THE PARENT'S STEP (`tier`, above, is the anchor's own), and
                // these centres are counted in that same step — which is the pairing this type exists
                // to keep honest.
                centre: r.center.in_parents_frame(),
                // THE CONSERVATIVE RADIUS: the shape's own circumscribed reach PLUS the band's release
                // edge, because membership extends past the surface by the outset. A point outside this
                // cannot be a member by any edge, so dropping the child for it cannot change a verdict.
                radius_m: r.shape.circumscribed_extent() + r.band.outset(),
            })
            .collect();
        self.child_index = ChildIndex::build(&children, tier);
        self.unindexed = self
            .regions
            .iter()
            .filter(|r| !self.child_index.answers_for(r.realm))
            .map(|r| r.realm)
            .collect();
    }

    /// The rows the child index does not answer for (see the field) — always asked, never walked.
    #[must_use]
    pub(crate) fn unindexed_rows(&self) -> &[RealmId] {
        &self.unindexed
    }

    /// This realm's built direct children (see the field) — the hulls whose exterior it leases.
    #[must_use]
    pub(crate) fn built_children(&self) -> &[RealmId] {
        &self.built_children
    }

    /// The child index this shard queries — see the field.
    #[must_use]
    pub fn child_index(&self) -> &ChildIndex {
        &self.child_index
    }

    /// The static rows of every parent, authored once (see the field). A mover's row is never
    /// here; a child that is neither moving nor driven is here at its authored centre.
    fn rebuild_static_rows(&mut self) {
        self.static_rows.clear();
        self.movers_of.clear();
        for (parent, ixs) in &self.children_of {
            let anchor = self.own_frame(*parent);
            let mut rows: Vec<(FrameRef, FramePlacement)> = Vec::new();
            let mut movers: Vec<usize> = Vec::new();
            for &ix in ixs {
                let r = &self.regions[ix];
                // ★ A SHIP IS NEVER A STATIC ROW (2026-09-05): it is driven by nature, so its row
                // is the overlay's, and its adoption or release never copies the static vector —
                // the galaxy's 279,380 rows were copied once per hand-over (a 60 ms tick).
                if self.moving.contains_key(&r.realm) | matches!(r.realm, RealmId::Ship(_)) {
                    movers.push(ix);
                    continue;
                }
                rows.push((
                    r.frame,
                    FramePlacement {
                        origin_cell: r.center.in_parents_frame().cell(),
                        origin: r.center.in_parents_frame().offset(),
                        velocity: DVec3::ZERO,
                        orientation: DQuat::IDENTITY,
                        angular_velocity: DVec3::ZERO,
                    },
                ));
            }
            self.static_rows
                .insert(*parent, PlacementBook::static_rows(anchor, rows));
            if !movers.is_empty() {
                self.movers_of.insert(*parent, movers);
            }
        }
    }

    /// The range index and the radial list over this realm's static, live-band direct children.
    fn rebuild_aoi_index(&mut self, own: RealmId) {
        let tier = self.own_frame(own).tier();
        let children: Vec<RealmRegion> = self
            .direct_children(own)
            .filter(|r| !self.moving.contains_key(&r.realm) && r.aoi.spin_up_r_m() > 0.0)
            .copied()
            .collect();
        let widest_extent = children
            .iter()
            .fold(0.0_f64, |acc, r| acc.max(r.shape.circumscribed_extent()));
        self.widest_child_extent_m = widest_extent;
        self.widest_band_m = children
            .iter()
            .fold(0.0_f64, |acc, r| acc.max(r.aoi.tear_down_r_m()));
        let indexed: Vec<IndexedChild> = children
            .iter()
            .map(|r| IndexedChild {
                realm: r.realm,
                centre: r.center.in_parents_frame(),
                radius_m: r.aoi.tear_down_r_m() + widest_extent,
            })
            .collect();
        self.aoi_index = ChildIndex::build(&indexed, tier);
        let mut radial: BTreeMap<u64, Vec<RealmId>> = BTreeMap::new();
        for r in &children {
            let d = r
                .center
                .in_parents_frame()
                .delta_m(LatticePos::ORIGIN, tier)
                .length();
            radial.entry(d.to_bits()).or_default().push(r.realm);
        }
        for names in radial.values_mut() {
            names.sort_unstable();
        }
        self.radial = radial;
    }

    /// The range index over this realm's static children (see the field).
    #[must_use]
    pub(crate) fn aoi_index(&self) -> &ChildIndex {
        &self.aoi_index
    }

    /// Is the range index built for `realm` — did `with_own_realm` name it? A store that never
    /// named its realm has no index, and a fold over it must say so rather than visit nobody.
    #[must_use]
    pub(crate) fn indexes_children_of(&self, realm: RealmId) -> bool {
        self.own_realm == Some(realm)
    }

    /// The static children an OUTSIDE looker `d_out_m` from this realm's centre could reach: every
    /// child at least `d_out_m − widest band` from the centre — a suffix of the radial list, found
    /// by one partition search.
    pub(crate) fn children_reachable_from_outside(
        &self,
        d_out_m: f64,
    ) -> impl Iterator<Item = RealmId> + '_ {
        // A distance is never negative, so its bit pattern orders like the number; a floor below
        // zero starts at zero (every child is reachable), exactly as the old partition did.
        let floor = (d_out_m - self.widest_band_m).max(0.0);
        self.radial
            .range(floor.to_bits()..)
            .flat_map(|(_, names)| names.iter().copied())
    }

    /// This realm's direct children that MOVE this tick — the orbiting and the driven — as a set.
    pub(crate) fn moving_children_of(
        &self,
        own: RealmId,
        driven: &crate::stub::drive::DrivenChildren,
    ) -> BTreeSet<RealmId> {
        let mut set: BTreeSet<RealmId> = self
            .movers_of
            .get(&own)
            .map(|ixs| ixs.iter().map(|&ix| self.regions[ix].realm).collect())
            .unwrap_or_default();
        set.extend(
            driven
                .0
                .keys()
                .copied()
                .filter(|c| self.parent_of(*c) == Some(own)),
        );
        set
    }

    /// ONE direct child of `own` by name — a lookup, never a scan.
    pub fn direct_child(&self, own: RealmId, realm: RealmId) -> Option<&RealmRegion> {
        let r = &self.regions[*self.ix_of.get(&realm)?];
        (r.parent == Some(own)).then_some(r)
    }

    /// The parent of a rostered realm, if the roster names it.
    pub(crate) fn parent_of(&self, realm: RealmId) -> Option<RealmId> {
        self.ix_of
            .get(&realm)
            .and_then(|ix| self.regions[*ix].parent)
    }

    /// ★ THE ROWS OF A NAMED SET (owner ruling 2026-09-02 R8 item 1): the placements of the given
    /// direct children of `own`, each one lookup — never a walk of the roster. A realm that is not
    /// a direct child of `own`, or not rostered, is skipped.
    pub(crate) fn child_rows_for<'a>(
        &'a self,
        own: RealmId,
        book: &PlacementBook,
        realms: impl IntoIterator<Item = RealmId>,
    ) -> Vec<(&'a RealmRegion, StampedPose)> {
        realms
            .into_iter()
            .filter_map(|realm| {
                let r = &self.regions[*self.ix_of.get(&realm)?];
                (r.parent == Some(own)).then_some(r)
            })
            .filter_map(|r| book.of(r.frame).map(|at| (r, pose_of_row(book, at))))
            .collect()
    }

    /// [`RealmRegions::child_rows_for`], as wire rows.
    pub(crate) fn snaps_for(
        &self,
        own: RealmId,
        book: &PlacementBook,
        realms: impl IntoIterator<Item = RealmId>,
    ) -> Vec<RealmSnap> {
        self.child_rows_for(own, book, realms)
            .into_iter()
            .map(|(r, pose)| RealmSnap {
                realm: r.realm,
                frame: r.frame,
                pose,
            })
            .collect()
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
        // THE INDEX THIS STRUCT ALREADY HOLDS — the wake loop asks this once per direct child,
        // every tick, and the un-indexed spelling scans the whole forest on every hop.
        vd_core::worldgen::coord_of_realm_indexed(&self.regions, &self.ix_of, realm)
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
            // ★ THE AMBIENT ROOT, AND IT CAN NOW SAY SO (slice S9). This used to fall back to a GALAXY
            // frame, which was the nearest thing to "outermost" available while the universe had no frame
            // of its own — a stand-in whose meaning had to be remembered. The universe has one now, so the
            // fallback names the thing it always meant.
            .map_or(FrameRef::UniverseSpace, |r| r.frame)
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
    /// ★ THE REGION FOR A REALM, BY LOOKUP (2026-08-30). Four accessors below asked this question by
    /// SCANNING the region list. `ix_of` has held the answer since the constructor.
    ///
    /// MEASURED: a whole-world pin calls `own_frame` once per anchor. At 233 221 anchors over 233 220
    /// regions that alone is of the order of 5e10 passes — the sim's test binary sat at 100% of a core
    /// and 10.5 GB without finishing. SL9 states the rule this restores: finding which realm a name
    /// belongs to is a LOOKUP, never a scan.
    fn region_of(&self, realm: RealmId) -> Option<&RealmRegion> {
        self.ix_of.get(&realm).map(|&ix| &self.regions[ix])
    }

    pub(crate) fn own_frame(&self, own_realm: RealmId) -> FrameRef {
        self.region_of(own_realm)
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
        self.region_of(own_realm).map(|r| RealmShape {
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
        self.region_of(own_realm).and_then(|r| r.look)
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
        self.region_of(realm).map(|r| r.frame)
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
        self.author_book_driven(
            anchor,
            tick_hz,
            at,
            &crate::stub::drive::DrivenChildren::default(),
        )
    }

    /// ★ THE SAME BOOK FOR A REALM THAT HOLDS DRIVEN CHILDREN (D-MOVE-2) — THE one implementation;
    /// [`RealmRegions::author_book`] is this with an empty driven book, which is what a realm that
    /// integrates nothing has.
    ///
    /// The pair exists so the thirty call sites that hold no driven child stay as they were, and so
    /// there is still only ONE place a row is filled. A second copy of the row logic is exactly the
    /// fork these rules exist to prevent.
    #[must_use]
    pub fn author_book_driven(
        &self,
        anchor: RealmId,
        tick_hz: f64,
        at: UniverseTick,
        driven: &crate::stub::drive::DrivenChildren,
    ) -> PlacementBook {
        let secs = secs_since_epoch(at.0, tick_hz);
        // ★ STATICS ONCE, MOVERS PER TICK (owner ruling 2026-09-02 R8 item 1; SL9): the overlay
        // walks the children that move — orbiting, or driven by a pilot — and the static layer is
        // shared. A driven child that was static at boot is in both; the overlay wins.
        let own_frame = self.own_frame(anchor);
        let statics = self
            .static_rows
            .get(&anchor)
            .cloned()
            .unwrap_or_else(|| std::sync::Arc::new(Vec::new()));
        let mut overlay: Vec<(FrameRef, FramePlacement)> = self
            .movers_of
            .get(&anchor)
            .map(|ixs| {
                ixs.iter()
                    .map(|&ix| {
                        let r = &self.regions[ix];
                        (r.frame, placement_row(&self.moving, driven, r, secs))
                    })
                    .collect()
            })
            .unwrap_or_default();
        for child in driven.0.keys() {
            if let Some(ix) = self.ix_of.get(child) {
                let r = &self.regions[*ix];
                if r.parent == Some(anchor) && !self.moving.contains_key(child) {
                    overlay.push((r.frame, placement_row(&self.moving, driven, r, secs)));
                }
            }
        }
        PlacementBook::layered(own_frame, at, statics, overlay)
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
    /// Does this child MOVE? The split the window lane's two lanes are chosen by (owner ruling
    /// 2026-08-27): a mover's row is worth repeating cheaply on the lossy per-tick frame and never
    /// worth retrying, because next tick's value beats a resend of last tick's; a static child's row is
    /// worth sending once on the reliable lane and never worth repeating.
    #[must_use]
    pub fn child_moves(&self, realm: RealmId) -> bool {
        self.moving.contains_key(&realm)
    }

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

    /// Every realm in this forest that has at least one direct child — the parent index's own keys.
    ///
    /// Exists so a caller cannot spell the question as a scan again: the answer is a map lookup, and
    /// the map is built once. See `children_of` for the measurement that forced it.
    pub(crate) fn parents_with_children(&self) -> impl Iterator<Item = RealmId> + '_ {
        self.children_of.keys().copied()
    }

    /// This shard's DIRECT child regions — the ROSTER half of [`RealmRegions::child_placements`], asked
    /// without placing anybody. Split out so a question that is only about WHICH children exist (is any of
    /// them active?) does not pay for every mover's orbit solve; `child_placements` is defined on top of it,
    /// so the two can never come to disagree about what a direct child is.
    pub fn direct_children(&self, own_realm: RealmId) -> impl Iterator<Item = &RealmRegion> {
        // A LOOKUP, NOT A SCAN — see `children_of` for the measurement that forced it. A realm with no
        // children is absent from the map and yields nothing, which is what the filter did.
        self.children_of
            .get(&own_realm)
            .into_iter()
            .flatten()
            .map(move |&ix| &self.regions[ix])
    }
}

/// The `RealmId → RealmLevel` for a hosted child region — sourced un-lossily from the seed via
/// [`level_of`] (the `System(0)`/`System(1)` stand-ins recover to Universe/Galaxy; keyed kinds pass
/// through). Monomorphic (HR5: the kind match is covered once inside `level_of`).
///
/// ★ TOTAL SINCE 2026-09-01 — IT USED TO REFUSE A SHIP. The old doc read: *"`None` for an
/// entity-backed `Ship` realm: the lineage coordinate cannot name one… every lane that needs a child
/// coord EXCLUDES such a region gracefully — counted, never a panic."*
///
/// That was true and is not any more. A ship has a lineage level, so no region can fail to produce
/// one, and every lane's exclusion branch became unreachable. Those branches are deleted rather than
/// left: an unreachable guard reads like a live protection, protects nothing, and cannot be covered.
///
/// The history is worth keeping, because the exclusion replaced something worse — an `expect` that
/// aborted a whole shard on one unrepresentable region, which excluded a first-class realm kind by
/// crashing rather than by a visible skip. The skip was the right cure then; the cure now is that
/// nothing is unrepresentable.
pub(crate) fn region_level(region: &RealmRegion) -> RealmLevel {
    level_of(region.realm)
}
