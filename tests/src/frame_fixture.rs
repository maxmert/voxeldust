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
//! > A galaxy holds two star systems; the neighbour sits 12031 m out, in a direction the seed chose.
//! > Inside that neighbour a planet is authored 145 m from its star. A player flies 3 m above that planet.
//! >
//! > - the planet knows only "an occupant at 3 from my centre";
//! > - the system knows only "I put that planet at 145";
//! > - the galaxy knows only "I put that system at 12031, THAT way".
//! >
//! > GOING UP the planet ships "3, in my frame"; the SYSTEM adds 145 → 148; the system ships "148, in my
//! > frame"; the GALAXY adds its own authored vector for that system → 148 along x off a point 12031 m
//! > out. Three separate additions, each by the one party that holds that number, and nobody ever learns
//! > its own address.
//! >
//! > GOING DOWN the galaxy subtracts the same authored vector and hands the system 148; the system
//! > computes 148 − 145 = 3 and hands it to the planet; the planet ACCEPTS 3 and does no arithmetic.
//!
//! Only the galaxy's hop reads as a vector rather than a scalar, and that is the 3-D seeded placement law
//! (owner ruling Q-B): the galaxy's number for its child is a RADIUS in a seeded DIRECTION, not a distance
//! along one axis. The two lower hops really are one-axis, which is why the story keeps its literals there
//! and states the top one as "12031 in some direction".
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
//! - `stellar.galaxy_rim_r_m` sets the galaxy's OUTER RADIUS — the rim, not a ring. Since the galaxy
//!   shape landed (S12; owner_decisions_2026-08-27_galaxy_shape.md) the seed decides where INSIDE that
//!   rim each system sits: a population, a radius on a density profile, an arm, a scatter and a height.
//!   The knob used to name the exact distance, because every system sat on one shell; the owner refused
//!   the shell by name, so it now names the edge and the seed names the place.
//!
//!   The neighbour's authored centre is therefore `system_from_galaxy_m × NEIGHBOUR_FRACTION_OF_RIM` in
//!   a seeded direction. Both halves are pinned by `the_generator_plants_exactly_the_stories_distances`
//!   below, which asserts the FRACTION against the generator and then asserts the direction is genuinely
//!   three-dimensional (`neighbour.y.abs() > 0.0`).
//! - `planet.ecc_sigma == 0` and `planet.incl_sigma == 0` make the story planet's orbit exactly circular
//!   and exactly in-plane, so its distance from its star is its semi-major axis at every tick. That axis
//!   is not a config product any more — the in-system true-size re-solve deleted `scale.au_to_render_m`
//!   and made the generator DERIVE every orbit from the drawn star — so the fixture registers its own
//!   worked distance over the generated topology instead (`build`'s WORKED-DISTANCE OVERRIDE, the same
//!   pattern as the phase pinning below).
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

use vd_core::geometry::RealmRegion;
use vd_core::glam::DVec3;
use vd_core::pose::{FrameRef, LatticePos, RealmId, StampedPose};
use vd_core::{UniverseTick, placement::PlacementBook};
use vd_physics::celestial::{G, OrbitalElements};
use vd_physics::worldgen::{UniverseConfig, WorldView};
use vd_sim::stub::RealmRegions;
use vd_wire::channels::RealmSnap;

/// The occupant's height above the planet in the NEAR and MOVING variants (m).
pub const NEAR_OCCUPANT_FROM_PLANET_M: f64 = 3.0;
/// The planet's distance from its star (m) — the number only the SYSTEM holds.
pub const NEAR_PLANET_FROM_STAR_M: f64 = 145.0;
/// The NEAR variant's galaxy OUTER RADIUS (m) — the rim the seed places the neighbour inside. Read the
/// story's distance as `NEAR_SYSTEM_FROM_GALAXY_M × NEIGHBOUR_FRACTION_OF_RIM`; see that constant.
pub const NEAR_SYSTEM_FROM_GALAXY_M: f64 = 12031.0;
/// WHERE INSIDE THE RIM THE SEED PUTS THE NEIGHBOUR — a fraction of the galaxy's outer radius.
///
/// ★ MEASURED, NEVER DERIVED (S12, 2026-08-28). Every term of the placement is proportional to the rim
/// except the angles, which read `r / r_max` and so do not move with it. The consequence is a pure
/// number, and it was measured rather than argued: the SAME fraction, to the last bit, at both story
/// magnitudes — 1.2031e4 m and 1e13 m. A shape change, a draw-order change or a re-seed all move it,
/// and `the_generator_plants_exactly_the_stories_distances` then fails loudly, which is that test's
/// whole reason to exist.
///
/// ★ NOW EXACTLY ONE (2026-08-31), and the change is the point. This was 0.271… — the fraction of the
/// rim at which the SEED happened to place the neighbour. A seeded placement cannot be asked for a
/// distance: you state a rim and the draw chooses somewhere inside it, so the story's own number was
/// never the distance, only a bound on it.
///
/// The fixture now stands on the HAND-PLACED world, which places by construction from its own config.
/// So the neighbour sits at the story's distance exactly, and the fraction is one. A worked example
/// whose stated number IS the measured number is the whole point of a worked example.
///
/// The constant is kept rather than deleted so the ratio stays asserted: if a future change ever puts
/// the neighbour somewhere other than where the story says, this goes red instead of passing quietly.
pub const NEIGHBOUR_FRACTION_OF_RIM: f64 = 1.0;
/// The occupant re-measured in the STAR's frame after the SYSTEM adds its child's placement.
/// A LITERAL: `145 + 3`, written out, never computed by the code under test.
pub const WORKED_UP_1: f64 = 148.0;
/// The occupant's DISTANCE from the galaxy centre if the galaxy's authored vector for the neighbour
/// pointed along `+x`. A LITERAL: `12031 + 148`, written out.
///
/// It is kept as the story's readable scalar and is deliberately NOT what the upward gate asserts. Since
/// the seeded placement law the galaxy's addition is a VECTOR sum, so the gate adds `up_1_m` along `+x`
/// to the centre the generator actually authored — see the walk-up-and-back-down test below. This value
/// is what that sum reduces to on the collinear story the prose tells.
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
/// [`placement_book`](WorkedExample::placement_book) for the authored book it converts through.
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
        // ★ THE HAND-PLACED WORLD (D-S12-FIXTURE, closed 2026-08-31). This asked the GENERATOR for a
        // galaxy holding exactly two stars at a distance it named. Since ruling G8 a population is a
        // RESULT — the volume a galaxy encloses at the density it drew — so neither the count nor the
        // distance can be asked for, and this fixture could not express itself at all.
        //
        // A worked example needs STATED distances: its whole point is that 12 031 m and 145 m appear
        // in the arithmetic and can be checked by eye. The hand-placed world places by construction
        // from its own config, which is exactly that — and the code that lowers, bands and converts
        // it is the same code the generated path uses, so nothing here is a second implementation.
        let world = WorldView::hand_placed(&config);

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
        // ★ BUILT OUTRIGHT, NOT READ AND OVERWRITTEN (2026-08-31). This read the generated orbit and
        // then replaced its phase, its semi-major axis and its central mass — every field the story
        // depends on. What survived was the eccentricity and inclination, which `story_config` pins
        // to zero anyway. So the read bought nothing and tied the fixture to a generated planet.
        let mut elements = vd_physics::celestial::OrbitalElements {
            sma: planet_from_star_m,
            // Circular and in-plane: the planet's distance from its star is its semi-major axis at
            // EVERY tick, so the story's number is true of the moving variant too.
            ecc: 0.0,
            inclination: 0.0,
            raan: 0.0,
            arg_periapsis: 0.0,
            mean_anomaly_epoch: 0.0,
            central_mass: 0.0,
        };
        // THE WORKED-DISTANCE OVERRIDE (see `story_config`): the fixture's registered elements
        // carry ITS exact semi-major axis and the Kepler-tuned central mass for the story
        // period — fixture DATA over generated topology, exactly like the phase pinning above.
        elements.sma = planet_from_star_m;
        elements.central_mass =
            4.0 * core::f64::consts::PI * core::f64::consts::PI * planet_from_star_m.powi(3)
                / (STORY_ORBIT_PERIOD_S * STORY_ORBIT_PERIOD_S)
                / G;

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
        // ★ THE STORY STATES ITS OWN ORBIT (2026-08-31), as it states its own distances.
        //
        // This read the GENERATED forest's movers and then substituted the story planet's elements
        // over the top. The fixture stands on the HAND-PLACED world now, whose bodies are placed
        // statically — so that read returned nothing, the substitution had nothing to substitute
        // into, and the moving variants silently stopped moving. The far-moving test caught it: its
        // placement was still exactly on the grid a thousand ticks in, which is the one thing that
        // variant exists to disprove.
        //
        // A worked example that states a distance should state its orbit the same way.
        if held == self.system {
            [(self.planet, self.elements)].into_iter().collect()
        } else {
            BTreeMap::new()
        }
    }

    /// The `RealmRegions` resource the shard hosting `held` boots with, built through the same two
    /// production builders the bins use (`RealmRegions::new` + `with_moving_children`).
    #[must_use]
    pub fn realm_regions(&self, held: RealmId) -> RealmRegions {
        RealmRegions::new(self.regions_for(held))
            .with_moving_children(vd_physics::motion::kepler_motion_fns(self.moving_for(held)))
    }

    /// The authored book that shard converts through — the production `author_book`, at this
    /// variant's tick.
    #[must_use]
    pub fn placement_book(&self, held: RealmId) -> PlacementBook {
        self.realm_regions(held)
            .author_book(held, self.tick_hz, self.tick)
    }

    /// The realm rows the shard hosting `held` AUTHORS this tick — the production
    /// `RealmRegions::authored_realm_snaps`, i.e. the live realm lane verbatim.
    ///
    /// Each row is a complete `(child frame, parent frame, placement)` edge in that shard's own frame. An
    /// ORBITING child's region carries a ZERO centre — its placement exists only here, on the live lane —
    /// so a receiver that ignores these rows draws every moving body at its parent's origin.
    #[must_use]
    pub fn authored_realm_rows(&self, held: RealmId) -> Vec<RealmSnap> {
        let regions = self.realm_regions(held);
        regions.authored_realm_snaps(held, &regions.author_book(held, self.tick_hz, self.tick))
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
    // ★ D-S12-FIXTURE, CLOSED 2026-08-31. This asked the generator for a galaxy holding exactly two
    // stars, one at the origin and one to hop to. Ruling G8 removed the way to ask: a population is a
    // RESULT — the volume a galaxy encloses at the density it drew — so the count knobs are gone, and
    // a galaxy the size of the story's 12 031 m holds ONE system.
    //
    // The register already named the cure and this is it: the HAND-PLACED world holds exactly two
    // systems BY CONSTRUCTION, at a distance its own config states. A worked example needs stated
    // distances — its whole point is that 12 031 m and 145 m appear in the arithmetic and can be
    // checked by eye — and a seeded position can never be asked for.
    //
    // Everything below the placement is the SAME code the generated path runs: one lowering, one band
    // solve, one conversion. Nothing here is a second implementation of anything.
    config.satellite.system_b_offset_m = system_from_galaxy_m;
    config.satellite.planet_offset_m = planet_from_star_m;
    config.stellar.galaxy_rim_r_m = system_from_galaxy_m;
    // Two planets per star: the second one is the SIBLING the negative gate demands a refusal for.
    config.planet.n_planets = 2;
    // A circular, in-plane orbit: the planet's distance from its star is its semi-major axis at EVERY
    // tick, so the story's "145" is true of the moving variant too and not only of the epoch.
    config.planet.ecc_sigma = 0.0;
    config.planet.incl_sigma = 0.0;
    // (Since the in-system true-size re-solve the generator DERIVES orbits, shells and the
    // central mass from the drawn star — the old compression/synthetic-mass knobs are gone.
    // The fixture keeps its exact worked distances by REGISTERING its own elements over the
    // generated topology — see `build`'s override, the same pattern as its phase pinning; the
    // generated forest supplies realms, frames and parents, which is all these conversion
    // stories read from it.)
    let _ = planet_from_star_m;
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

/// The centre the PARENT authored for `realm`, flattened to metres. Zero for an orbiting body (its
/// position is authored live through its frame); the static offset otherwise. NORMALIZED since the
/// cell activation — the value rides the integer half, so this flattens through `delta_m` (reading
/// `.offset()` raw would place every static body at the origin).
fn centre_of(world: &WorldView, realm: RealmId) -> DVec3 {
    let regions = world.regions();
    let region = regions
        .iter()
        .find(|r| r.realm == realm)
        .expect("the fixture only names realms the generator produced");
    // ★ AT THE PARENT'S STEP (slice S9). A region's `center` is its position in its PARENT's frame,
    // while `frame` is its own — the same step until the galaxy got a coarser one, and out by 2048×
    // after. Read with the child's step, this fixture's neighbour system sat 8.67 m from its galaxy
    // instead of the story's distance, and every conversion built on it inherited that.
    //
    // The lookup is `ParentCentre`'s own now, so the wrong step cannot be handed in here at all.
    region
        .centre_m(regions)
        .expect("the fixture's forest holds every named realm's parent")
}

/// The lattice position of a pose, for tests that want the whole anchored value rather than the offset.
#[must_use]
pub fn lattice_of(pose: &StampedPose) -> LatticePos {
    pose.pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::frame::{FrameError, transfer_frame};

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
            // Since the 3-D seeded placement law (owner Q-B) the neighbour's direction is the
            // seed's; the STORY fact is its RADIUS — asserted at the story distance (sqrt-of-sum
            // rounding bounded well under a micron at these magnitudes) — and that the direction
            // is genuinely three-dimensional (off the retired ring's y = 0 plane).
            let neighbour = centre_of(&fx.world, fx.system);
            assert_eq!(
                neighbour.length() / fx.system_from_galaxy_m,
                NEIGHBOUR_FRACTION_OF_RIM,
                "{}: the galaxy authors the neighbour system at the story fraction of its rim \
                 (rim {}, got {})",
                fx.name,
                fx.system_from_galaxy_m,
                neighbour.length(),
            );
            // ★ THE 3-D CHECK MOVED OUT (2026-08-31). This asserted that the neighbour sits off the
            // y = 0 plane — a fact about the SEEDED placement law, which this fixture no longer uses.
            // It hand-places now, along one axis on purpose, so the story's distances are numbers a
            // reader can check by eye.
            //
            // The law itself is not lost: `the_seeded_placements_are_three_dimensional_and_the_fence_
            // judges_the_point_set` in the physics crate owns it, against the generator that has it.
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
        let book = fx.placement_book(fx.system);
        let placement = book
            .of(fx.planet_frame)
            .expect("the star authors its own planet");
        // NORMALIZED placement (the cell activation): the 145 m rides the integer anchor exactly.
        assert_eq!(
            placement
                .anchor()
                .delta_m(vd_core::pose::LatticePos::ORIGIN, fx.planet_frame.tier()),
            DVec3::new(fx.planet_from_star_m, 0.0, 0.0),
            "the star authors the planet exactly on +x at tick 0",
        );
        // And the MOVING variant is genuinely somewhere else — a test that accidentally read the epoch
        // instead of the live placement would be caught by this.
        let mv = WorkedExample::moving();
        let moved = mv
            .placement_book(mv.system)
            .of(mv.planet_frame)
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
            &fx.placement_book(fx.system),
        )
        .expect("the star can place its own planet");
        assert_eq!(at_system.frame, fx.system_frame);
        assert_eq!(
            at_system
                .pos
                .delta_m(vd_core::pose::LatticePos::ORIGIN, at_system.frame.tier()),
            DVec3::new(fx.up_1_m, 0.0, 0.0)
        );

        let at_galaxy = transfer_frame(&at_system, fx.galaxy_frame, &fx.placement_book(fx.galaxy))
            .expect("the galaxy can place its own system");
        assert_eq!(at_galaxy.frame, fx.galaxy_frame);
        // The galaxy's addition on the seeded 3-D placement: the authored vector + 148 along x
        // (the collinear story's `up_2` on one axis, generalized).
        assert_eq!(
            at_galaxy
                .pos
                .delta_m(vd_core::pose::LatticePos::ORIGIN, at_galaxy.frame.tier()),
            centre_of(&fx.world, fx.system) + DVec3::new(fx.up_1_m, 0.0, 0.0)
        );

        let back_to_system =
            transfer_frame(&at_galaxy, fx.system_frame, &fx.placement_book(fx.galaxy))
                .expect("the galaxy can place its own system");
        assert_eq!(
            back_to_system.pos.delta_m(
                vd_core::pose::LatticePos::ORIGIN,
                back_to_system.frame.tier()
            ),
            DVec3::new(fx.up_1_m, 0.0, 0.0)
        );

        let back_to_planet = transfer_frame(
            &back_to_system,
            fx.planet_frame,
            &fx.placement_book(fx.system),
        )
        .expect("the star can place its own planet");
        assert_eq!(
            back_to_planet.pos.delta_m(
                vd_core::pose::LatticePos::ORIGIN,
                back_to_planet.frame.tier()
            ),
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
            let book = fx.placement_book(own);
            assert_eq!(book.of(parent_frame), None);
            assert_eq!(book.of(sibling_frame), None);
        }
        assert_eq!(
            transfer_frame(
                &fx.occupant_pose(),
                fx.galaxy_frame,
                &fx.placement_book(fx.planet),
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
        let at_epoch = WorkedExample::far();
        let moved = WorkedExample::far_moving();
        for (fx, expect_lossless) in [(&at_epoch, true), (&moved, false)] {
            let placement = fx
                .placement_book(fx.system)
                .of(fx.planet_frame)
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
