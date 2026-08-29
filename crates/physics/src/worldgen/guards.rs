//! THE BOOT FENCES: the refusals a world must survive before anything runs on it.
//!
//! Owns: the storage fence (is the root representable at all), the sibling-separation fences (two
//! realms whose boundaries intersect make "which realm contains this point" unanswerable), the nest
//! fence over a lowered forest, and the star-bound fence. Each returns a TYPED refusal naming the
//! offending pair, never a bare bool.
//!
//! Does NOT own: a blessed exception. A fence that a known-bad seed can be excused from is not a
//! fence; the sweep that proves the mass cap holds is sized to expect the hard case, not to avoid
//! it.

#[cfg(test)]
use super::Placement;
use super::{
    GeneratedBody, IMF_MASS_LO_MSUN, IMF_SLOPE, K_SPAN, UniverseConfig, WorldView,
    generate_system_forest, imf_mass_hi_msun, moving_children_for_config, placement_offset,
};
use crate::celestial::OrbitalElements;
use crate::motion::Motion;
use glam::DVec3;
use std::collections::BTreeMap;
use vd_core::geometry::RealmRegion;
use vd_core::pose::RealmId;
use vd_core::worldgen::GALAXY;

/// A world whose root outgrows the FINE lattice's representable budget — [`guard_root_representable`]'s
/// loud refusal, carrying every number of the verdict. **THIS REFUSAL IS THE NAMED P10 TRIGGER
/// (R3):** the day a world needs more than the FINE tier can hold exactly is the day the galaxy
/// cell lattice (the COARSE tier's activation) is built — compression → 1 is not a wish, it is
/// this fence's condition.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "the root shell ({root_r_m} m = {root_cells} FINE cells) with K_SPAN {k_span} outgrows the \
     FINE lattice's sanitized domain (CELL_DOMAIN_MAX = {domain_max} cells): occupancy \
     {occupancy_pct}% > 100%/K_SPAN — the world has outgrown the millimetre tier; the cure is \
     the galaxy cell lattice (P10), never a widened clamp"
)]
pub struct RootNotRepresentable {
    pub root_r_m: f64,
    pub root_cells: f64,
    pub k_span: f64,
    pub domain_max: i64,
    pub occupancy_pct: f64,
    /// The rung the shell was judged at — see [`RootBudget::tier`].
    pub tier: vd_core::pose::Tier,
}

/// The measured storage budget a representable root prints (occupancy + headroom, §A2.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootBudget {
    /// The shell radius in the cells of the rung it was judged at.
    pub root_cells: f64,
    /// Domain occupancy: root_cells / CELL_DOMAIN_MAX (0.5 exactly at the fence's own equality).
    pub occupancy: f64,
    /// Headroom against the clamp: CELL_DOMAIN_MAX / root_cells (2.0 exactly at that equality).
    pub headroom: f64,
    /// WHICH RUNG THIS VERDICT IS ABOUT. Carried because the same radius passes at one step and is
    /// refused at another, so a budget without its unit is a number nobody can check.
    pub tier: vd_core::pose::Tier,
    /// That rung's step, so a printed verdict needs no lookup.
    pub cell_edge_m: f64,
}

/// THE STORAGE FENCE (real-scale addendum §A2.1 F1 / §A4.9 R3 — a boot fence beside
/// `guard_visibility_climb_bounded` in every world-deriving process): a shell, with its `K_SPAN`
/// headroom octave, must fit its own lattice's sanitized domain, so the wire-ingress clamp is
/// unreachable from any lawful position. Refusing is THE NAMED P10 TRIGGER. Measured on THE world:
/// occupancy exactly 50.0000 %, headroom exactly 2.0000× (`2⁵¹ m = 2⁶¹ cells; 2 × 2⁶¹ =
/// CELL_DOMAIN_MAX + 1` — the equality is the construction).
///
/// ★ IT ASKS EACH LEVEL IN THAT LEVEL'S OWN STEP (slice S8). It used to divide by the millimetre step
/// unconditionally, which was right while every level counted in millimetres and becomes badly wrong the
/// moment they do not. Applied to the universe's own `2⁷⁶ m` shell it would have refused by a factor of
/// `2²⁵` — thirty-three million — for a world that fits its own lattice EXACTLY. A fence that refuses a
/// lawful world is worse than no fence, because the refusal looks authoritative.
///
/// # Errors
/// [`RootNotRepresentable`] with every number of the verdict, naming P10 as the cure.
pub fn guard_root_representable(
    config: &UniverseConfig,
) -> Result<RootBudget, RootNotRepresentable> {
    // ★ EVERY LEVEL IN ITS OWN STEP (slice S9). S8 judged the galaxy at the ROOT's step, which was
    // honest while the galaxy was not a realm and its shell was genuinely measured in root cells. It has
    // its own frame and its own step now, so it is judged in its own — which is what "runs per level"
    // was always supposed to mean.
    //
    // The root's verdict is the one returned, because the root is what a boot log reports and what the
    // named refusal is about. A galaxy that does not fit refuses the boot just the same.
    guard_shell_representable(config.scale.galaxy_r_m, vd_core::pose::Tier::Galaxy)?;
    guard_shell_representable(config.scale.universe_r_m, ROOT_TIER)
}

/// THE RUNG THE ROOT SHELL IS COUNTED IN TODAY. Named here rather than passed, because the root's own
/// step is a fact about the ladder and not a choice a caller makes — a caller that could pass the wrong
/// one would be a caller that could disable the fence.
///
/// ★ THE LADDER IS CLIMBED (slice S9). S8 built the rungs and left the root counting in millimetres,
/// because no frame produced a universe position yet. The universe has its own frame now, so it counts in
/// its own step — and its radius moves with it in the same change, which was the whole condition: the
/// fence's content is the EQUALITY between a step and a radius, and moving one without the other would
/// turn a 50 % occupancy into 0.0015 % and the fence would stop saying anything.
pub(crate) const ROOT_TIER: vd_core::pose::Tier = vd_core::pose::Tier::Universe;

/// The fence itself, for ANY shell at ANY rung — the shape [`guard_root_representable`] is one call of.
///
/// # Errors
/// [`RootNotRepresentable`] when the shell plus its headroom octave outgrows that rung's domain.
pub fn guard_shell_representable(
    shell_r_m: f64,
    tier: vd_core::pose::Tier,
) -> Result<RootBudget, RootNotRepresentable> {
    let edge = tier.cell_edge_m();
    let domain_max = vd_core::pose::CELL_DOMAIN_MAX;
    let root_cells = shell_r_m / edge;
    let budget = K_SPAN * root_cells;
    // `+ 1.0` exactly as the ▲ 1 derivation states: 2·2⁶¹ equals CELL_DOMAIN_MAX + 1, so THE
    // world passes with exact equality — the headroom octave is the construction, not slack.
    //
    // ★ FAIL CLOSED ON A SHELL THAT IS NOT A LENGTH. `NaN > x` is false and so is `NaN <= x`, so a
    // non-finite radius used to sail through this comparison and be reported as a representable world
    // with a NaN occupancy. A negative radius did the same. Neither is a shell, and a fence that answers
    // "fine" to a question that makes no sense is worse than no fence. Bitwise `|`: one branch, both
    // regions driven.
    if !(shell_r_m.is_finite() & (shell_r_m > 0.0)) | (budget > domain_max as f64 + 1.0) {
        return Err(RootNotRepresentable {
            root_r_m: shell_r_m,
            root_cells,
            k_span: K_SPAN,
            domain_max,
            occupancy_pct: 100.0 * root_cells / domain_max as f64,
            tier,
        });
    }
    Ok(RootBudget {
        root_cells,
        occupancy: root_cells / domain_max as f64,
        headroom: domain_max as f64 / root_cells,
        tier,
        cell_edge_m: edge,
    })
}

/// THE SEEDED-SYSTEM SEPARATION FENCE as a boot guard (Q-B's re-derived fence over the general
/// point set): build THE world's seeded system placements and judge every pair. Run beside the
/// climb fence at every world-deriving boot.
///
/// # Errors
/// [`SiblingsOverlap`] naming the first overlapping pair.
pub fn guard_seeded_systems_disjoint(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Result<(), SiblingsOverlap> {
    let centres: Vec<(RealmId, DVec3, f64)> = generate_system_forest(seed_universe, config)
        .iter()
        .filter(|b| b.parent == Some(GALAXY))
        .map(|b| {
            (
                b.realm,
                placement_offset(b.placement),
                b.shape.circumscribed_extent(),
            )
        })
        .collect();
    seeded_systems_disjoint_3d(&centres)
}

/// EVERY CHILD'S WORST-INSTANT REACH, for a forest already lowered to regions — the roster the
/// nest fence ([`vd_core::geometry::guard_regions_nest`]) consumes. THE ONE IMPLEMENTATION (HR3):
/// `vd_bins::child_reaches` is this with the process's own world config threaded in, so the boot
/// and every sweep below judge children by the identical law.
///
/// A mover states an `Excursion` (its apoapsis bound, from the ONE motion crate — the fence itself
/// may never ask HOW anything moves, SL4); anything else states the `Fixed` offset it was authored
/// at. A missing row is unrepresentable rather than defaulted: the regions and this roster derive
/// from the SAME `(seed, config)` forest.
#[must_use]
pub fn child_reaches_for_config(
    seed_universe: u64,
    regions: &[RealmRegion],
    config: &UniverseConfig,
) -> std::collections::BTreeMap<RealmId, vd_core::geometry::ChildReach> {
    let parents: std::collections::BTreeSet<RealmId> =
        regions.iter().filter_map(|r| r.parent).collect();
    let movers: std::collections::BTreeMap<RealmId, OrbitalElements> = parents
        .iter()
        .flat_map(|p| moving_children_for_config(seed_universe, config, *p))
        .collect();
    // The parent ROW for every realm that is one, looked up once instead of re-scanned per child.
    let by_realm: std::collections::BTreeMap<RealmId, &RealmRegion> =
        regions.iter().map(|r| (r.realm, r)).collect();
    regions
        .iter()
        .filter_map(|r| r.parent.map(|p| (r, p)))
        .map(|(r, p)| {
            let parent = by_realm
                .get(&p)
                .expect("a child's parent is a row of the SAME (seed, config) forest");
            (r.realm, one_child_reach(parent, r, movers.get(&r.realm)))
        })
        .collect()
}

/// One child's reach — the monomorphic body the shim above stays branchless over (HR5).
fn one_child_reach(
    parent: &RealmRegion,
    child: &RealmRegion,
    mover: Option<&OrbitalElements>,
) -> vd_core::geometry::ChildReach {
    use vd_core::geometry::ChildReach;
    match mover {
        Some(e) => ChildReach::Excursion(Motion::Kepler(*e).max_excursion_m(child.frame.tier())),
        // The stored offset is measured in the PARENT's frame, and `metres_in` reads the step off
        // the parent itself.
        //
        // ★ THE PARENT IS AN ARGUMENT NOW (slice S9). This used to scan `regions` for the parent and
        // FALL BACK TO THE CHILD'S OWN STEP when it was not found — the 2048× defect written down as
        // a `map_or` default. The caller's own contract says a missing row is unrepresentable, so the
        // honest shape is to demand the parent rather than to invent a unit for it. The scan is gone
        // with it, which also stops this being linear in the sibling count (SL9).
        None => ChildReach::Fixed(child.center.metres_in(parent)),
    }
}

/// A seed whose world does not nest — the named refusal of [`guard_swept_seeds_nest`], carrying
/// the seed, the reservation that failed to cover it, and the nest fence's own verdict verbatim.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "seed {seed} generates a world that does not nest under a reservation of {reservation_m} m: \
     {source} — the mass cap and the reservation must be two readings of ONE derivation"
)]
pub struct SeedWorldDoesNotNest {
    /// The seed whose world refused.
    pub seed: u64,
    /// The reservation in force when it refused (the placement radius's complement).
    pub reservation_m: f64,
    /// The nest fence's verdict, unaltered.
    pub source: vd_core::geometry::RegionNestError,
}

/// ONE seed's world, judged by the SAME fence every shard boot runs.
///
/// The `max` handed to the nest fence is this forest's own region count: the membership-bitset
/// width is `vd_sim`'s number and is fenced in `vd_core` by `guard_regions_nest_rejects_too_many_regions`.
/// What is measured HERE is the geometric half — the exact arm that refused the owner's boot.
///
/// # Errors
/// [`SeedWorldDoesNotNest`] carrying the fence's verdict.
pub fn guard_world_nests(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Result<usize, SeedWorldDoesNotNest> {
    let world = WorldView::generated(seed_universe, config);
    let regions = world.regions();
    let reaches = child_reaches_for_config(seed_universe, regions, config);
    vd_core::geometry::guard_regions_nest(regions, &reaches).map_err(|source| {
        SeedWorldDoesNotNest {
            seed: seed_universe,
            reservation_m: config.scale.galaxy_r_m - config.stellar.galaxy_rim_r_m,
            source,
        }
    })?;
    Ok(regions.len())
}

/// ★ THE MASS-CAP GUARANTEE, AS A MEASUREMENT THAT COULD HAVE FAILED (owner ruling 2026-08-20).
///
/// The structural claim is that the derived reservation covers EVERY star this world may draw, so
/// every seed's world nests. A claim is not a measurement, so this sweeps seeds `0..sweep` and puts
/// each generated world through the identical fence that refused the owner's galaxy shard — the one
/// whose refusal read *"is not geometrically inside its parent … refusing to boot"*.
///
/// Returns the number of REGIONS judged across the sweep, so a caller can prove the walk was not
/// vacuous.
///
/// # Errors
/// [`SeedWorldDoesNotNest`] naming the FIRST seed whose world does not nest.
pub fn guard_swept_seeds_nest(
    config: &UniverseConfig,
    sweep: u64,
) -> Result<usize, SeedWorldDoesNotNest> {
    let mut judged = 0usize;
    for seed in 0..sweep {
        judged += guard_world_nests(seed, config)?;
    }
    Ok(judged)
}

/// One octave: the sweep is sized to expect a star within a FACTOR OF TWO of the cap, which is the
/// only band in which a system's shell comes anywhere near the reservation (the shell grows about
/// as `M^1.75`, so half the cap is already a shell three times smaller than the reservation).
pub(crate) const NEST_SWEEP_TAIL_OCTAVE: f64 = 0.5;

/// THE SWEEP SIZE, DERIVED — never chosen. A sweep proves nothing if the stars it draws are all
/// tiny, and under a Salpeter IMF nearly all of them are. So the sweep is sized from the IMF ITSELF:
/// enough seeds that, IN EXPECTATION, at least one star lands within an octave of the cap —
///
/// `seeds = ceil( 1 / ( systems_per_world · P(M > cap/2) ) )`
///
/// with `P` the exact tail of the bounded power law [`crate::taxonomy::sample_imf_mass`] inverts.
/// Both inputs move with the world: change the cap, the system census or the slope and the sweep
/// resizes itself. The gate PRINTS the size and the heaviest star it actually drew, so a sweep that
/// silently stopped exercising the tail is visible rather than green.
/// ★ THE POPULATION IS AN ARGUMENT NOW, NOT A CONSTANT (S12/G8, 2026-08-28). It read
/// `WORLD_SYSTEM_COUNT`, a stated census the owner's ruling deleted. A galaxy's population is drawn,
/// so the caller — which holds a world — states the world's own count and this sizes the sweep from
/// it. The relationship is unchanged: more stars per world, fewer worlds needed to meet the tail.
#[must_use]
pub fn derived_nest_sweep_seeds(systems_per_world: u32) -> u64 {
    let cap = imf_mass_hi_msun();
    let tail = crate::taxonomy::imf_tail_fraction(
        NEST_SWEEP_TAIL_OCTAVE * cap,
        IMF_SLOPE,
        IMF_MASS_LO_MSUN,
        cap,
    );
    (1.0 / (f64::from(systems_per_world.max(1)) * tail)).ceil() as u64
}

/// THE 3-D SEPARATION FENCE — the ring's closed-form fence RE-DERIVED for the seeded point set
/// (Q-B: a proof rewrite, never a weakening). For every pair of seeded systems the pairwise
/// centre distance must exceed the sum of their circumscribed extents (containment stays
/// single-answer: no position may be inside two sibling authorities), and — the wake law's half —
/// every pair must be separated by more than one system's AoI spin-up reach, so a system is ASLEEP
/// at departure from any sibling. Closed form per pair (an exact subtraction and two sums), loud
/// ★ THE OVERLAP LOOKUP — the one place two sibling realms are tested for overlap (SL9: *"Finding
/// which child holds a point is a LOOKUP, never a scan"*).
///
/// Returns the FIRST overlapping pair `(i, j)`, `i < j`, in the same order an all-pairs walk would
/// have found it: the smallest `i` that overlaps anything, then its smallest `j`. That is not a
/// convenience — it is what makes this a drop-in for the walk it replaces, so the refusal names the
/// same pair it always did and no test has to be re-pinned to a different one.
///
/// ★ WHY A GRID IS ENOUGH, AND WHY THE CELL IS TWICE THE LARGEST REACH. Two siblings overlap only if
/// they are closer than the sum of their extents, and no sum exceeds twice the largest. So a cell of
/// that width puts every possible partner in the caller's own cell or one of the twenty-six touching
/// it. Anything farther cannot reach, and is never looked at.
///
/// The map is a `BTreeMap` and the candidates are sorted, so the answer does not depend on hashing or
/// on iteration order — the determinism rule, and the reason a hash map is refused here.
///
/// ★ WHY IT HAD TO CHANGE. The walk was every pair. MEASURED on the shipped generator at 1 000, 2 000,
/// 4 000 and 8 000 systems: 10.4 ms, 43.1 ms, 191.7 ms and 1 005.4 ms — clean `n²`, which is about six
/// minutes at the world's ~150 000 target, at every boot. SL9 forbids exactly that: *"a cost that
/// grows with the number of children is a defect, and it must be measured on a realm with many, not
/// argued."*
fn first_overlapping_pair(items: &[(RealmId, DVec3, f64)]) -> Option<(usize, usize)> {
    let reach_max = items.iter().map(|(_, _, e)| *e).fold(0.0_f64, f64::max);
    // With no reach there is no overlap: the test is `distance < a + b`, and a distance is never
    // below zero. Returning early also keeps the cell width off zero.
    if reach_max <= 0.0 {
        return None;
    }
    let cell = 2.0 * reach_max;
    let key = |v: DVec3| {
        (
            (v.x / cell).floor() as i64,
            (v.y / cell).floor() as i64,
            (v.z / cell).floor() as i64,
        )
    };
    let mut grid: BTreeMap<(i64, i64, i64), Vec<usize>> = BTreeMap::new();
    for (ix, (_, at, _)) in items.iter().enumerate() {
        grid.entry(key(*at)).or_default().push(ix);
    }
    for (i, (_, a_at, a_ext)) in items.iter().enumerate() {
        let (cx, cy, cz) = key(*a_at);
        let mut near: Vec<usize> = Vec::new();
        for dx in -1..=1_i64 {
            for dy in -1..=1_i64 {
                for dz in -1..=1_i64 {
                    if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        near.extend(bucket.iter().copied().filter(|j| *j > i));
                    }
                }
            }
        }
        // Ascending, so the pair this reports is the pair the all-pairs walk reported.
        near.sort_unstable();
        for j in near {
            let (_, b_at, b_ext) = items[j];
            if (b_at - *a_at).length() < a_ext + b_ext {
                return Some((i, j));
            }
        }
    }
    None
}

/// on refusal with both names and the measured gap.
pub(crate) fn seeded_systems_disjoint_3d(
    centres: &[(RealmId, DVec3, f64)],
) -> Result<(), SiblingsOverlap> {
    // ★ A LOOKUP, NOT A WALK (SL9) — see `first_overlapping_pair` for the measurement that forced it.
    first_overlapping_pair(centres).map_or(Ok(()), |(i, j)| {
        Err(SiblingsOverlap {
            a: centres[i].0,
            b: centres[j].0,
            parent: GALAXY,
        })
    })
}

/// Two sibling realms whose boundaries INTERSECT — the authoring mistake that makes "which realm contains
/// this position" have two answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiblingsOverlap {
    /// The two siblings, ordered as authored.
    pub a: RealmId,
    pub b: RealmId,
    /// Their shared parent.
    pub parent: RealmId,
}

/// Refuse a forest in which two STATICALLY-placed siblings intersect.
///
/// WHY THIS IS A CORRECTNESS FENCE, not tidiness. Authority is decided by descending into the deepest
/// child whose boundary contains you. If two siblings overlap, a position inside both has two equally
/// valid answers, and which shard owns you depends on iteration order — a coin flip that decides where
/// your input is applied and who simulates your collisions. It cannot be repaired downstream, because by
/// then the ambiguity is already a routing decision.
///
/// STATIC SIBLINGS ONLY, and that limit is real rather than convenient: an orbiting body's lowered region
/// sits at its frame ORIGIN (center zero — its position is authored live through its frame each tick), so
/// two planets are indistinguishable from co-located by any static comparison. Judging orbits needs their
/// SHELLS compared — two orbits are disjoint iff their radii differ by more than the sum of their
/// boundaries, at every eccentricity — which is a separate check over the moving roster. Recorded rather
/// than silently skipped: an unchecked orbital overlap is the same defect one level down.
///
/// Conservative on both sides: it compares FARTHEST-surface-point distances, so a doubtful placement is
/// refused rather than waved through. Cross-frame siblings are not judged (their numbers are not
/// comparable) — the same honest decline the parent-fit check makes.
/// RUN AT TEST TIME, not at boot, and that is a decision rather than an omission: the generator is
/// deterministic, so proving it over the shipped presets across a sweep of seeds proves every world that can
/// actually be booted, while a per-boot pass would be quadratic in a galaxy's population for an answer that
/// cannot change between runs. If worlds ever stop being purely seed-derived — the moment players place
/// structures the generator did not — this moves to the placement path, where the new body is the only thing
/// that needs judging.
#[cfg(test)]
pub(crate) fn siblings_disjoint(bodies: &[GeneratedBody]) -> Result<(), SiblingsOverlap> {
    // ★ GROUPED BY PARENT, THEN LOOKED UP (SL9, 2026-08-28). This walked EVERY PAIR OF BODIES and
    // filtered each one by parent inside the inner loop, so a forest of a hundred thousand bodies
    // paid for every pair of them to learn that almost none were siblings at all.
    //
    // MEASURED on the shipped generator at 1 000 / 2 000 / 4 000 / 8 000 systems: 10.4 ms, 43.1 ms,
    // 191.7 ms, 1 005.4 ms — `n²`, about six minutes at the world's ~150 000 target, at every boot.
    //
    // Siblings share a parent by definition, so the parent is the first key and the pairs never
    // cross a family. Within one family the reach test is the grid lookup, which cannot be worse
    // than the walk and is far better wherever the children are spread out.
    //
    // A body with an ORBIT is skipped, exactly as before: an orbit is judged on its shell, not on
    // where it happens to sit at the epoch. So is the ambient root, which has no siblings.
    let mut families: BTreeMap<RealmId, Vec<(RealmId, DVec3, f64)>> = BTreeMap::new();
    let mut order: Vec<RealmId> = Vec::new();
    for b in bodies {
        let Placement::StaticOffset(at) = b.placement else {
            continue;
        };
        let Some(parent) = b.parent else {
            continue;
        };
        let family = families.entry(parent).or_default();
        if family.is_empty() {
            order.push(parent);
        }
        family.push((b.realm, at, b.shape.circumscribed_extent()));
    }
    // FAMILIES IN THE ORDER THEY FIRST APPEAR, so the refusal names the pair the walk named: the
    // walk found the smallest body index that overlapped anything, and the family holding it is the
    // first family any such body belongs to.
    for parent in order {
        let family = &families[&parent];
        if let Some((i, j)) = first_overlapping_pair(family) {
            return Err(SiblingsOverlap {
                a: family[i].0,
                b: family[j].0,
                parent,
            });
        }
    }
    Ok(())
}

/// A star whose dust-sublimation bound fails to clear its own photosphere — the T2 boot fence's
/// loud refusal (`bound/R★ = 0.5·(T_eff/T_sub)²` → 1.993 at the hydrogen-burning limit, so the
/// fence holds across the whole IMF domain but is a FENCE, not an assumption — the swept gate
/// prints the minimum).
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "the star of {system:?} has a dust-sublimation bound ({bound_m} m) at or inside its own \
     photosphere ({photosphere_m} m) — the Star realm would be drawn wider than its authority; \
     the mass-radius or luminosity law changed under the extent derivation"
)]
pub struct StarBoundInsidePhotosphere {
    pub system: RealmId,
    pub bound_m: f64,
    pub photosphere_m: f64,
}

/// THE T2 BOOT FENCE (`guard_star_bound_exceeds_photosphere`): every generated star's bound
/// strictly exceeds its photosphere. Wired beside the climb fence in every world-deriving boot.
///
/// # Errors
/// [`StarBoundInsidePhotosphere`] naming the first offending system with both radii.
pub fn guard_star_bound_exceeds_photosphere(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Result<(), StarBoundInsidePhotosphere> {
    guard_star_bounds(&generate_system_forest(seed_universe, config))
}

/// The fence's ARITHMETIC over a forest already in hand — split from the generate-and-check shell
/// so the REFUSAL arm is reachable from a unit test (HR5: the branching lives in a monomorphic
/// helper a test can hand a hostile forest; THE world only ever produces the green arm).
pub(crate) fn guard_star_bounds(
    bodies: &[GeneratedBody],
) -> Result<(), StarBoundInsidePhotosphere> {
    for b in bodies {
        if let (RealmId::Star(_), Some(p)) = (b.realm, b.photometrics.as_ref()) {
            let bound_m = b.shape.finite_extent();
            let photosphere_m = crate::taxonomy::star_radius_m(p.mass_msun);
            if bound_m <= photosphere_m {
                return Err(StarBoundInsidePhotosphere {
                    system: b.parent.expect("a star nests in its system"),
                    bound_m,
                    photosphere_m,
                });
            }
        }
    }
    Ok(())
}
