//! # The worked example — the fixture the frame-conversion arc is checked against
//!
//! THE GROUND RULE this fixture exists to pin, in the owner's words: *only the parent knows where the
//! children are; a child has no idea about its own position; we do not leak unnecessary data from one
//! realm to another.* Every realm is centred on ITSELF. It holds its own boundary at its own origin and
//! the placements of its DIRECT CHILDREN in its own frame. It never knows, derives or stores where IT
//! ITSELF sits.
//!
//! THE STORY, which is what every test in the arc re-tells with different machinery:
//!
//! > A galaxy holds two star systems; the neighbour sits 12031 m out. Inside that neighbour a planet is
//! > authored 145 m from its star. A player flies 3 m above that planet.
//! >
//! > - the planet knows only "an occupant at 3 from my centre";
//! > - the system knows only "I put that planet at 145";
//! > - the galaxy knows only "I put that system at 12031".
//! >
//! > GOING UP the planet ships "3, in my frame"; the SYSTEM adds 145 → 148; the system ships "148, in my
//! > frame"; the GALAXY adds 12031 → 12179. Three separate additions, each by the one party that holds
//! > that number, and nobody ever learns its own address.
//! >
//! > GOING DOWN the galaxy computes 12179 − 12031 = 148 and hands it to the system; the system computes
//! > 148 − 145 = 3 and hands it to the planet; the planet ACCEPTS 3 and does no arithmetic at all.
//!
//! The expected results are LITERALS here ([`WorkedExample::up_1_m`] / [`up_2_m`](WorkedExample::up_2_m)),
//! never recomputed by the code under test. A fixture that computes its own expectation with the same
//! expression the subject uses cannot fail when the subject is wrong.
//!
//! ## Where the numbers come from: the PRODUCTION generator, not a hand-built region vector
//!
//! The forest is built by [`WorldView::generated`] — the same `f(seed, UniverseConfig)` a real shard boots
//! through (`vd_bins::boot_regions_and_movers` → `worldgen::realm_neighbourhood_for_config` →
//! `realm_regions_for_config` → the same `generate_system_forest` + `to_regions` lowering). The story's
//! distances are chosen by CONFIG — the same knobs the shipped presets set — so every band, every frame
//! label, every parent link and every region centre in this fixture is whatever production would produce
//! for that config, and the fixture cannot drift away from what a shard actually boots:
//!
//! - `stellar.system_ring_r_m` puts the NEIGHBOUR system on the galaxy's star ring. With a two-star
//!   galaxy the ring angle for index 1 is exactly zero, so its authored centre is exactly
//!   `(system_from_galaxy_m, 0, 0)` — an exact-integer-friendly number on one axis, not a rounded one.
//! - `scale.au_to_render_m` with `planet.orbital_a0_au == 1` makes the inner planet's semi-major axis
//!   EXACTLY `planet_from_star_m` (`orbital_axis_au(0, a0, ratio) == a0`, so the product is `1.0 * a0_m`).
//! - `planet.ecc_sigma == 0` and `planet.incl_sigma == 0` make that orbit exactly circular and exactly
//!   in-plane, so the planet's distance from its star is its semi-major axis at every tick.
//!
//! ONE thing is not left to the seed: the three PHASE angles of the story planet's orbit
//! (`raan`, `arg_periapsis`, `mean_anomaly_epoch`) are pinned to zero — see [`WorkedExample::elements`].
//! That chooses WHERE ON ITS CIRCLE the planet sits at tick 0, nothing else; sma, eccentricity,
//! inclination and central mass all still come from the config. Without it the planet sits at a
//! seed-drawn angle and the story's second addition is `12031 + 145·(some direction)`, which is a true
//! statement about a rotated triangle and a useless one to read in a test.
//!
//! ## What differs from the story's nouns, and why
//!
//! - There is no `RealmId::Galaxy` at P3: `worldgen` names the galaxy `RealmId::System(1)` under a
//!   `RealmId::System(0)` universe, so the galaxy's FRAME is a `SystemSpace`, not `FrameRef::GalaxySpace`.
//!   [`WorkedExample::galaxy_frame`] is the real one; `FrameRef::GalaxySpace` is, in this forest, a frame
//!   no region carries — which makes it a useful *unknown* frame to demand a refusal for.
//! - The production generator emits NO static planet: every planet under a generated star is an
//!   `OrbitalElements` child. So all three variants have an orbiting planet. What [`WorkedExample::moving`]
//!   adds is a TICK at which the orbit has actually carried the planet off its epoch point — the NEAR and
//!   FAR variants read the world at tick 0, where the pinned phase puts the planet exactly on `+x`.

use std::collections::{BTreeMap, BTreeSet};

use vd_core::celestial::{G, OrbitalElements};
use vd_core::geometry::RealmRegion;
use vd_core::glam::DVec3;
use vd_core::pose::{FrameRef, LatticePos, RealmId, StampedPose};
use vd_core::worldgen::{UniverseConfig, WorldView};
use vd_core::{UniverseTick, frame::LocalFrames};
use vd_sim::stub::RealmRegions;
use vd_wire::channels::RealmSnap;

/// The occupant's height above the planet in the NEAR and MOVING variants (m).
pub const NEAR_OCCUPANT_FROM_PLANET_M: f64 = 3.0;
/// The planet's distance from its star (m) — the number only the SYSTEM holds.
pub const NEAR_PLANET_FROM_STAR_M: f64 = 145.0;
/// The neighbour system's distance from the galaxy centre (m) — the number only the GALAXY holds.
pub const NEAR_SYSTEM_FROM_GALAXY_M: f64 = 12031.0;
/// The occupant re-measured in the STAR's frame after the SYSTEM adds its child's placement.
/// A LITERAL: `145 + 3`, written out, never computed by the code under test.
pub const WORKED_UP_1: f64 = 148.0;
/// The occupant re-measured in the GALAXY's frame after the GALAXY adds its child's placement.
/// A LITERAL: `12031 + 148`, written out.
pub const WORKED_UP_2: f64 = 12179.0;

/// The FAR variant's system distance. Chosen so the story's millimetre is BELOW the representable step:
/// 1e13 lies between 2^43 and 2^44, so consecutive doubles there are 2^-9 m = 1.953125 mm apart
/// ([`FAR_ULP_M`]). An occupant at 3.001 m cannot survive a trip through a value of this size — which is
/// the whole reason the fold is being removed, and the reason slice 7's round-trip gate uses this number.
pub const FAR_SYSTEM_FROM_GALAXY_M: f64 = 1.0e13;
/// The FAR variant's occupant height — a millimetre off a round number, deliberately.
pub const FAR_OCCUPANT_FROM_PLANET_M: f64 = 3.001;
/// The FAR variant's first addition, a LITERAL. Exact in binary at this magnitude.
pub const FAR_UP_1: f64 = 148.001;
/// The spacing between neighbouring doubles at [`FAR_SYSTEM_FROM_GALAXY_M`] — `2^-9` m. Any quantity that
/// has been through a value that large carries at least this much uncertainty, so it is the floor on what
/// a tolerance at that scale may claim.
pub const FAR_ULP_M: f64 = 1.0 / 512.0;

/// The orbital period (s) the story planet is given, via the star's mass. Named rather than inline: the
/// planet has to move fast enough for [`WorkedExample::moving`] to read a DIFFERENT placement a few ticks
/// on, and slowly enough that a few ticks is not most of an orbit.
const STORY_ORBIT_PERIOD_S: f64 = 600.0;
/// The universe tick rate the fixture samples its ephemeris at (Hz). The ephemeris is `f(elements, secs)`,
/// so a rate is needed to turn a tick into seconds; this is the shipped shard rate.
pub const FIXTURE_TICK_HZ: f64 = 20.0;
/// The tick [`WorkedExample::moving`] reads the world at — far enough into the orbit that the planet's
/// authored placement is nowhere near its epoch point, so a test that accidentally reads the epoch fails.
pub const MOVING_TICK: UniverseTick = UniverseTick(1_000);

/// How much bigger than the thing it must contain each shell is made. Every boundary in the fixture is
/// derived from the story's distances through this one factor, so there is no shell radius anywhere below
/// that a later edit to a distance can leave behind.
const SHELL_HEADROOM: f64 = 4.0;

/// The three-level worked example, planted through the production world generator.
///
/// Build one with [`near`](WorkedExample::near), [`far`](WorkedExample::far) or
/// [`moving`](WorkedExample::moving), then ask it for the pieces a shard boots with —
/// [`realm_regions`](WorkedExample::realm_regions) for the containment forest a given realm's shard holds,
/// [`frame_context`](WorkedExample::frame_context) for the ephemeris it converts through.
#[derive(Clone, Debug)]
pub struct WorkedExample {
    /// Which variant this is, for test failure messages.
    pub name: &'static str,
    /// The universe seed the forest was generated from.
    pub seed: u64,
    /// The generator config that produced it — the same type the shipped presets are.
    pub config: UniverseConfig,
    /// The tick every placement in this variant is read at.
    pub tick: UniverseTick,
    /// The tick rate the ephemeris is sampled at.
    pub tick_hz: f64,

    /// The ambient root. Present as a REGION in the galaxy's neighbourhood and deliberately absent from
    /// its frame context — the galaxy is not told where the universe is either.
    pub universe: RealmId,
    /// The top level of the story: it authors where the two star systems sit.
    pub galaxy: RealmId,
    /// The middle level: the neighbour star system, authored by the galaxy at
    /// [`system_from_galaxy_m`](Self::system_from_galaxy_m).
    pub system: RealmId,
    /// The bottom level: a planet, authored by its star at
    /// [`planet_from_star_m`](Self::planet_from_star_m).
    pub planet: RealmId,
    /// The OTHER star system under the same galaxy — the sibling no shard may place.
    pub sibling_system: RealmId,
    /// The OTHER planet under the same star — the sibling no planet shard may place.
    pub sibling_planet: RealmId,

    /// `universe`'s frame.
    pub universe_frame: FrameRef,
    /// `galaxy`'s frame. NOT `FrameRef::GalaxySpace` — see the module docs.
    pub galaxy_frame: FrameRef,
    /// `system`'s frame.
    pub system_frame: FrameRef,
    /// `planet`'s frame.
    pub planet_frame: FrameRef,
    /// `sibling_system`'s frame.
    pub sibling_system_frame: FrameRef,
    /// `sibling_planet`'s frame.
    pub sibling_planet_frame: FrameRef,

    /// The number only the GALAXY holds: where it put the neighbour system, in its own frame.
    pub system_from_galaxy_m: f64,
    /// The number only the SYSTEM holds: where it put the planet, in its own frame.
    pub planet_from_star_m: f64,
    /// The number only the PLANET holds: where the occupant is, in its own frame.
    pub occupant_from_planet_m: f64,
    /// The occupant in the SYSTEM's frame after one upward hop — a literal.
    pub up_1_m: f64,
    /// The occupant in the GALAXY's frame after the second upward hop — a literal. At FAR scale the
    /// decimal literal and the double addition need not be the same double; that gap is
    /// [`tolerance_m`](Self::tolerance_m), and it is the defect this arc removes, stated as a number.
    pub up_2_m: f64,
    /// How far a value at this variant's scale may legitimately be off. ZERO for NEAR: every number in
    /// the near story is exact in binary, so an approximate assertion there would be hiding something.
    pub tolerance_m: f64,

    /// The story planet's orbital elements as the fixture registers them: the config's semi-major axis,
    /// eccentricity, inclination and central mass, with the three PHASE angles pinned to zero so the
    /// planet sits exactly on `+x` at tick 0. See the module docs for why the phase is pinned.
    pub elements: OrbitalElements,

    /// The generated world. Private: everything a caller needs comes out of the accessors below, and
    /// handing out the whole world invites a test to fold a chain of ancestors out of it, which is the
    /// construction this arc exists to remove.
    world: WorldView,
}

impl WorkedExample {
    /// The readable variant: 12031 / 145 / 3, every number exact in binary, read at tick 0.
    #[must_use]
    pub fn near() -> WorkedExample {
        WorkedExample::build(
            "NEAR",
            NEAR_SYSTEM_FROM_GALAXY_M,
            NEAR_PLANET_FROM_STAR_M,
            NEAR_OCCUPANT_FROM_PLANET_M,
            WORKED_UP_1,
            WORKED_UP_2,
            0.0,
            UniverseTick(0),
        )
    }

    /// The precision variant: the same story with the system 1e13 m out and the occupant a millimetre off
    /// a round number. The millimetre is below the representable step at that magnitude, so any code that
    /// inflates the occupant's position to universe scale and back loses it.
    #[must_use]
    pub fn far() -> WorkedExample {
        WorkedExample::build(
            "FAR",
            FAR_SYSTEM_FROM_GALAXY_M,
            NEAR_PLANET_FROM_STAR_M,
            FAR_OCCUPANT_FROM_PLANET_M,
            FAR_UP_1,
            // Written out rather than summed: `1e13 + 148.001` is not exactly this decimal, and the gap
            // between them is exactly the precision this fixture is here to measure.
            10_000_000_000_148.001,
            FAR_ULP_M,
            UniverseTick(0),
        )
    }

    /// The FAR world read at a tick where the planet's orbit has carried it OFF the round number — which is
    /// what makes it the variant that can tell a nearest-common-ancestor walk apart from a fold to the root.
    ///
    /// At tick 0 the pinned phase puts the planet at exactly 145 m on `+x`, and 145 happens to be an exact
    /// multiple of the double spacing at 1e13 m (145 / 2^-9 = 74240). So at tick 0 inflating that placement
    /// to galaxy scale and back returns it UNHARMED, and the two constructions are indistinguishable — a
    /// fixture that only ever reads tick 0 cannot fail when the fold comes back. A few hundred ticks in, the
    /// placement is a generic `(145·cos θ, 145·sin θ, 0)` that sits nowhere near that grid, and the
    /// difference is about a millimetre. The up/down literals do NOT apply here, for the same reason they do
    /// not apply to [`moving`](Self::moving).
    #[must_use]
    pub fn far_moving() -> WorkedExample {
        WorkedExample::build(
            "FAR-MOVING",
            FAR_SYSTEM_FROM_GALAXY_M,
            NEAR_PLANET_FROM_STAR_M,
            FAR_OCCUPANT_FROM_PLANET_M,
            FAR_UP_1,
            10_000_000_000_148.001,
            FAR_ULP_M,
            MOVING_TICK,
        )
    }

    /// The moving variant: the NEAR world read at a tick where the planet's orbit has carried it well off
    /// its epoch point, so the system's authored placement for it is a live number rather than a stored
    /// one. The up/down literals do NOT apply here (the planet is no longer on `+x`); this variant exists
    /// for the tests whose subject is that a placement MOVES.
    #[must_use]
    pub fn moving() -> WorkedExample {
        WorkedExample::build(
            "MOVING",
            NEAR_SYSTEM_FROM_GALAXY_M,
            NEAR_PLANET_FROM_STAR_M,
            NEAR_OCCUPANT_FROM_PLANET_M,
            WORKED_UP_1,
            WORKED_UP_2,
            0.0,
            MOVING_TICK,
        )
    }

    #[allow(clippy::too_many_arguments)] // one private constructor for three variants; every argument is a named public const at the call sites
    fn build(
        name: &'static str,
        system_from_galaxy_m: f64,
        planet_from_star_m: f64,
        occupant_from_planet_m: f64,
        up_1_m: f64,
        up_2_m: f64,
        tolerance_m: f64,
        tick: UniverseTick,
    ) -> WorkedExample {
        let seed = 0;
        let config = story_config(system_from_galaxy_m, planet_from_star_m);
        let world = WorldView::generated(seed, &config);

        // Read the story's realms OUT of the generated forest rather than naming their seeds: the seeds
        // are `child_seed` avalanches of (galaxy, salt, index), so writing them down would be copying a
        // hash into a test and would go stale the moment the salt or the index changes.
        let universe = parent_of(&world, galaxy_of(&world)).expect("the galaxy has a parent");
        let galaxy = galaxy_of(&world);
        // The two star systems, in forest order: index 0 sits at the galaxy's own centre, index 1 out on
        // the ring. The story's system is the one that is actually somewhere — a hop of zero would prove
        // nothing about a parent adding its child's placement.
        let systems: Vec<RealmId> = children_of(&world, galaxy);
        let system = *systems
            .iter()
            .find(|r| centre_of(&world, **r).length_squared() > 0.0)
            .expect("a two-star galaxy has one system off its centre");
        let sibling_system = *systems
            .iter()
            .find(|r| *r != &system)
            .expect("a two-star galaxy has a second system");
        let planets = children_of(&world, system);
        let planet = planets[0];
        let sibling_planet = planets[1];

        // The story planet's elements: the config's, with the phase pinned. See the module docs.
        let generated = vd_core::worldgen::moving_children_for_config(seed, &config, system);
        let mut elements = generated
            .iter()
            .find(|(r, _)| *r == planet)
            .map(|(_, e)| *e)
            .expect("the story planet is an orbital child of its star");
        elements.raan = 0.0;
        elements.arg_periapsis = 0.0;
        elements.mean_anomaly_epoch = 0.0;

        WorkedExample {
            name,
            seed,
            tick,
            tick_hz: FIXTURE_TICK_HZ,
            universe_frame: frame_of(&world, universe),
            galaxy_frame: frame_of(&world, galaxy),
            system_frame: frame_of(&world, system),
            planet_frame: frame_of(&world, planet),
            sibling_system_frame: frame_of(&world, sibling_system),
            sibling_planet_frame: frame_of(&world, sibling_planet),
            universe,
            galaxy,
            system,
            planet,
            sibling_system,
            sibling_planet,
            system_from_galaxy_m,
            planet_from_star_m,
            occupant_from_planet_m,
            up_1_m,
            up_2_m,
            tolerance_m,
            elements,
            config,
            world,
        }
    }

    /// The containment forest the shard hosting `held` boots with — the PRODUCTION neighbourhood scope
    /// (its own realm, its ancestors, its direct children; never a sibling).
    #[must_use]
    pub fn regions_for(&self, held: RealmId) -> Vec<RealmRegion> {
        self.world.neighbourhood(&BTreeSet::from([held]))
    }

    /// The moving-child roster that shard authors: the production roster for `held`, with the story
    /// planet's phase pinned (only the star's roster contains it, so every other shard's roster is the
    /// generator's verbatim).
    #[must_use]
    pub fn moving_for(&self, held: RealmId) -> BTreeMap<RealmId, OrbitalElements> {
        vd_core::worldgen::moving_children_for_config(self.seed, &self.config, held)
            .into_iter()
            .map(|(realm, e)| {
                if realm == self.planet {
                    (realm, self.elements)
                } else {
                    (realm, e)
                }
            })
            .collect()
    }

    /// The `RealmRegions` resource the shard hosting `held` boots with, built through the same two
    /// production builders the bins use (`RealmRegions::new` + `with_moving_children`).
    #[must_use]
    pub fn realm_regions(&self, held: RealmId) -> RealmRegions {
        RealmRegions::new(self.regions_for(held)).with_moving_children(self.moving_for(held))
    }

    /// The ephemeris that shard converts through — the production `frame_context`, at this variant's tick.
    #[must_use]
    pub fn frame_context(&self, held: RealmId) -> LocalFrames {
        self.realm_regions(held)
            .frame_context(held, self.tick_hz, self.tick)
    }

    /// The realm rows the shard hosting `held` AUTHORS this tick — the production
    /// `RealmRegions::authored_realm_snaps`, i.e. the live realm lane verbatim.
    ///
    /// Each row is a complete `(child frame, parent frame, placement)` edge in that shard's own frame. An
    /// ORBITING child's region carries a ZERO centre — its placement exists only here, on the live lane —
    /// so a receiver that ignores these rows draws every moving body at its parent's origin.
    #[must_use]
    pub fn authored_realm_rows(&self, held: RealmId) -> Vec<RealmSnap> {
        self.realm_regions(held)
            .authored_realm_snaps(held, self.tick_hz, self.tick)
    }

    /// THE WHOLE FOREST — the GATEWAY's view, and only the gateway's.
    ///
    /// A shard gets [`regions_for`](Self::regions_for), which is its own realm, its ancestors and its
    /// direct children and nothing else. The gateway is a different party with a different entitlement: it
    /// holds both ends of every conversion it performs, which is precisely why the conversion was moved
    /// there. Reading this from a test that is pretending to be a SHARD would be re-planting the leak this
    /// arc removed, so the two accessors are named differently on purpose.
    #[must_use]
    pub fn forest(&self) -> &[RealmRegion] {
        self.world.regions()
    }

    /// A pose of `x` metres along `+x` in `frame`, at rest, stamped at this variant's tick — the shape
    /// every hop in the story carries.
    #[must_use]
    pub fn pose_in(&self, frame: FrameRef, x: f64) -> StampedPose {
        StampedPose::at_rest(frame, DVec3::new(x, 0.0, 0.0), self.tick)
    }

    /// The occupant as its own PLANET holds it: `occupant_from_planet_m` from the planet's centre, in the
    /// planet's frame. The planet knows this and nothing else about where the occupant is.
    #[must_use]
    pub fn occupant_pose(&self) -> StampedPose {
        self.pose_in(self.planet_frame, self.occupant_from_planet_m)
    }
}

/// The generator config the story is expressed in. Every geometry number is DERIVED from the two story
/// distances — nothing here is a loose literal that a later change to a distance could leave stranded
/// inside a shell that no longer contains what it should.
fn story_config(system_from_galaxy_m: f64, planet_from_star_m: f64) -> UniverseConfig {
    let mut config = UniverseConfig::walk_scale();
    // Exactly two stars: with `n_systems == 2` the ring angle for index 1 is `TAU * 0 / 1 == 0`, so the
    // neighbour's authored centre is exactly `(system_from_galaxy_m, 0, 0)` — one axis, no rounding.
    config.galaxy.system_count_lo = 2;
    config.galaxy.system_count_hi = 2;
    config.stellar.system_ring_r_m = system_from_galaxy_m;
    // Two planets per star: the second one is the SIBLING the negative gate demands a refusal for.
    config.planet.n_planets = 2;
    // `orbital_axis_au(0, a0, ratio) == a0`, so with `a0 == 1 AU` the inner planet's semi-major axis is
    // exactly the AU-to-metres factor — i.e. exactly the story's distance, with no product to round.
    config.planet.orbital_a0_au = 1.0;
    config.scale.au_to_render_m = planet_from_star_m;
    // A circular, in-plane orbit: the planet's distance from its star is its semi-major axis at EVERY
    // tick, so the story's "145" is true of the moving variant too and not only of the epoch.
    config.planet.ecc_sigma = 0.0;
    config.planet.incl_sigma = 0.0;
    // The star's mass, chosen through the period the story planet should take to go round — the same
    // move `visual_scale` makes (a synthetic mass so the demo is watchable), stated as the thing that is
    // actually being chosen rather than as a mass constant nobody can check.
    // `T = 2π√(a³/μ)` ⇒ `μ = 4π²a³/T²`, and `μ = G·M`.
    let mu = 4.0 * core::f64::consts::PI * core::f64::consts::PI * planet_from_star_m.powi(3)
        / (STORY_ORBIT_PERIOD_S * STORY_ORBIT_PERIOD_S);
    config.stellar.central_mass_kg = mu / G;
    // Every shell derived from what it has to hold, outermost last.
    config.planet.planet_soi_r_m = planet_from_star_m / SHELL_HEADROOM;
    config.stellar.system_soi_r_m = (planet_from_star_m * config.planet.orbital_ratio
        + config.planet.planet_soi_r_m)
        * SHELL_HEADROOM;
    config.scale.galaxy_r_m =
        (system_from_galaxy_m + config.stellar.system_soi_r_m) * SHELL_HEADROOM;
    config.scale.universe_r_m = config.scale.galaxy_r_m * SHELL_HEADROOM;
    config
}

/// The galaxy of a generated world: the single child of the ambient root.
fn galaxy_of(world: &WorldView) -> RealmId {
    let root = world
        .regions()
        .iter()
        .find(|r| r.parent.is_none())
        .expect("a generated world has one ambient root");
    children_of(world, root.realm)[0]
}

/// `realm`'s parent in the forest.
fn parent_of(world: &WorldView, realm: RealmId) -> Option<RealmId> {
    world
        .regions()
        .iter()
        .find(|r| r.realm == realm)
        .and_then(|r| r.parent)
}

/// `realm`'s direct children, in forest order.
fn children_of(world: &WorldView, realm: RealmId) -> Vec<RealmId> {
    world
        .regions()
        .iter()
        .filter(|r| r.parent == Some(realm))
        .map(|r| r.realm)
        .collect()
}

/// The frame `realm`'s region is expressed in.
fn frame_of(world: &WorldView, realm: RealmId) -> FrameRef {
    world
        .regions()
        .iter()
        .find(|r| r.realm == realm)
        .expect("the fixture only names realms the generator produced")
        .frame
}

/// The centre the PARENT authored for `realm`, as the region carries it. Zero for an orbiting body (its
/// position is authored live through its frame); the static offset otherwise.
fn centre_of(world: &WorldView, realm: RealmId) -> DVec3 {
    let region = world
        .regions()
        .iter()
        .find(|r| r.realm == realm)
        .expect("the fixture only names realms the generator produced");
    debug_assert_eq!(
        region.center.cell(),
        vd_core::glam::I64Vec3::ZERO,
        "every region in the P3 forest is at cell zero",
    );
    region.center.offset()
}

/// The lattice position of a pose, for tests that want the whole anchored value rather than the offset.
#[must_use]
pub fn lattice_of(pose: &StampedPose) -> LatticePos {
    pose.pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::frame::{FrameContext, FrameError, transfer_frame};

    /// THE ANTI-DRIFT GUARD. The story's three distances are asserted against what the PRODUCTION
    /// generator actually planted, so the literals above cannot quietly stop describing the world the
    /// fixture builds. Without it a later config change would leave a fixture that still compiles, still
    /// passes its own arithmetic, and no longer tests the thing its name claims.
    #[test]
    fn the_generator_plants_exactly_the_stories_distances() {
        for fx in [
            WorkedExample::near(),
            WorkedExample::far(),
            WorkedExample::far_moving(),
            WorkedExample::moving(),
        ] {
            assert_eq!(
                centre_of(&fx.world, fx.system),
                DVec3::new(fx.system_from_galaxy_m, 0.0, 0.0),
                "{}: the galaxy authors the neighbour system on +x at the story distance",
                fx.name,
            );
            assert_eq!(
                centre_of(&fx.world, fx.sibling_system),
                DVec3::ZERO,
                "{}: the other star is the one at the galaxy's own centre",
                fx.name,
            );
            assert_eq!(
                fx.elements.sma, fx.planet_from_star_m,
                "{}: the story planet's orbit radius IS the story distance",
                fx.name,
            );
            assert_eq!(fx.elements.ecc, 0.0, "{}: the orbit is circular", fx.name);
            assert_eq!(
                fx.elements.inclination, 0.0,
                "{}: the orbit is in-plane",
                fx.name,
            );
        }
    }

    /// The tick-0 placement the pinned phase buys: the planet sits EXACTLY on `+x` at the story distance,
    /// which is what makes the NEAR literals exact rather than approximately right.
    #[test]
    fn the_pinned_phase_puts_the_planet_exactly_on_plus_x_at_tick_zero() {
        let fx = WorkedExample::near();
        let ctx = fx.frame_context(fx.system);
        let placement = ctx
            .placement(fx.planet_frame, fx.tick)
            .expect("the star authors its own planet");
        assert_eq!(
            placement.origin,
            DVec3::new(fx.planet_from_star_m, 0.0, 0.0),
            "the star authors the planet exactly on +x at tick 0",
        );
        // And the MOVING variant is genuinely somewhere else — a test that accidentally read the epoch
        // instead of the live placement would be caught by this.
        let mv = WorkedExample::moving();
        let moved = mv
            .frame_context(mv.system)
            .placement(mv.planet_frame, mv.tick)
            .expect("the star authors its own planet")
            .origin;
        assert_ne!(
            moved,
            DVec3::new(mv.planet_from_star_m, 0.0, 0.0),
            "the moving variant reads a tick where the orbit has carried the planet off its epoch point",
        );
    }

    /// The same three additions the vd-sim gate makes, made here against the fixture's own accessors —
    /// so the fixture is proven to deliver a world the arc's story is actually true of, and does not
    /// depend on vd-sim's private test module staying in step with it.
    #[test]
    fn the_worked_example_walks_up_and_back_down_through_production_conversions() {
        let fx = WorkedExample::near();

        let at_system = transfer_frame(
            &fx.occupant_pose(),
            fx.system_frame,
            &fx.frame_context(fx.system),
        )
        .expect("the star can place its own planet");
        assert_eq!(at_system.frame, fx.system_frame);
        assert_eq!(at_system.pos.offset(), DVec3::new(fx.up_1_m, 0.0, 0.0));

        let at_galaxy = transfer_frame(&at_system, fx.galaxy_frame, &fx.frame_context(fx.galaxy))
            .expect("the galaxy can place its own system");
        assert_eq!(at_galaxy.frame, fx.galaxy_frame);
        assert_eq!(at_galaxy.pos.offset(), DVec3::new(fx.up_2_m, 0.0, 0.0));

        let back_to_system =
            transfer_frame(&at_galaxy, fx.system_frame, &fx.frame_context(fx.galaxy))
                .expect("the galaxy can place its own system");
        assert_eq!(back_to_system.pos.offset(), DVec3::new(fx.up_1_m, 0.0, 0.0));

        let back_to_planet = transfer_frame(
            &back_to_system,
            fx.planet_frame,
            &fx.frame_context(fx.system),
        )
        .expect("the star can place its own planet");
        assert_eq!(
            back_to_planet.pos.offset(),
            DVec3::new(fx.occupant_from_planet_m, 0.0, 0.0),
        );
    }

    /// The negative half, at the fixture tier: no level of the story can place its own parent or a
    /// sibling, and asking is a typed refusal rather than a number.
    #[test]
    fn no_level_of_the_worked_example_can_place_its_own_parent_or_a_sibling() {
        let fx = WorkedExample::near();
        for (own, parent_frame, sibling_frame) in [
            (fx.galaxy, fx.universe_frame, fx.planet_frame),
            (fx.system, fx.galaxy_frame, fx.sibling_system_frame),
            (fx.planet, fx.system_frame, fx.sibling_planet_frame),
        ] {
            let ctx = fx.frame_context(own);
            assert_eq!(ctx.placement(parent_frame, fx.tick), None);
            assert_eq!(ctx.placement(sibling_frame, fx.tick), None);
        }
        assert_eq!(
            transfer_frame(
                &fx.occupant_pose(),
                fx.galaxy_frame,
                &fx.frame_context(fx.planet),
            )
            .expect_err("a planet must not be able to answer where it is in its galaxy"),
            FrameError::UnknownDestFrame,
        );
    }

    /// The FAR-MOVING variant's reason to exist, stated as a measurement: at tick 0 the planet's placement
    /// sits exactly on the double grid at the far distance and survives a fold unharmed, so a gate that only
    /// reads tick 0 cannot tell a nearest-common-ancestor walk from a fold to the root. A thousand ticks in
    /// it does not, and the fold costs most of a millimetre.
    #[test]
    fn the_far_moving_variant_puts_the_placement_off_the_galaxy_scale_grid() {
        use vd_core::frame::FrameContext;

        let at_epoch = WorkedExample::far();
        let moved = WorkedExample::far_moving();
        for (fx, expect_lossless) in [(&at_epoch, true), (&moved, false)] {
            let placement = fx
                .frame_context(fx.system)
                .placement(fx.planet_frame, fx.tick)
                .expect("the star authors its own planet")
                .origin;
            let far = DVec3::new(fx.system_from_galaxy_m, 0.0, 0.0);
            let folded = (far + placement) - far;
            assert_eq!(
                folded == placement,
                expect_lossless,
                "{}: placement {placement:?} folded through {} m returns {folded:?}",
                fx.name,
                fx.system_from_galaxy_m,
            );
        }
    }

    /// The FAR variant's reason to exist, stated as a measurement: the millimetre the story carries is
    /// smaller than the gap between neighbouring doubles at the far system's distance, so a value that
    /// has been inflated to that magnitude cannot carry it back.
    #[test]
    fn the_far_variant_sits_below_the_representable_step() {
        let fx = WorkedExample::far();
        let step = (fx.system_from_galaxy_m + FAR_ULP_M) - fx.system_from_galaxy_m;
        assert_eq!(
            step, FAR_ULP_M,
            "one ulp at the far distance is exactly the documented step",
        );
        // MEASURED: a millimetre does not vanish at this distance, it SNAPS — 0.001 m goes in and
        // 0.001953125 m comes back, a 95% error on the quantity. That is worse than losing it, because
        // the number that comes out still looks like a plausible displacement.
        let below = (fx.system_from_galaxy_m + 0.001) - fx.system_from_galaxy_m;
        assert_eq!(
            below, FAR_ULP_M,
            "a millimetre put through the far distance comes back as a whole representable step",
        );
        // And the story's own occupant height is destroyed the same way: 3.001 in, 3.001953125 out.
        let occupant =
            (fx.system_from_galaxy_m + FAR_OCCUPANT_FROM_PLANET_M) - fx.system_from_galaxy_m;
        assert_ne!(
            occupant, FAR_OCCUPANT_FROM_PLANET_M,
            "the occupant's height cannot survive a trip through the far distance",
        );
        assert_eq!(fx.tolerance_m, FAR_ULP_M);
    }
}
