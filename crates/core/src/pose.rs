//! `StampedPose` / `FrameRef` — the ONE coordinate-frame transfer type
//! (integration resolution: unifies the designs' `ReferenceFrameDef`/`StampedPose`/
//! `FrameRef` into a single shared shape).
//!
//! Frames are time-parameterized: a pose is meaningful only together with the
//! `universe_tick` it was stamped at, and consumers re-evaluate frame functions at
//! their own render/sim instant — never by comparing wall clocks (R6: ~110 km/s
//! orbital velocity makes 1 ms of clock skew ≈ 110 m of error).
//!
//! The SOURCE shard computes a transfer's destination pose and ships it; the dest
//! sanity-bounds and uses it (authority never depends on cross-binary float
//! reproducibility).

use glam::{DQuat, DVec3, I64Vec3};
use serde::{Deserialize, Serialize};

use crate::frame::FrameError;
use crate::ids::{EntityId, UniverseTick};

/// A persistence/ownership realm: the unit of single-writer durable state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RealmId {
    /// A planet's surface world, keyed by its deterministic seed.
    Planet(u64),
    /// A star system's space, keyed by its deterministic seed.
    System(u64),
    /// A ship's interior world, keyed by the ship entity.
    Ship(EntityId),
    /// A space station's interior world, keyed by its deterministic seed — a first-class
    /// realm (like a ship) so a station's functional blocks can be a cross-shard signal
    /// source/sink. APPENDED (discriminant 3) so the wire stays additive.
    Station(u64),
    /// A sub-planet AREA (city district / spaceport / crowd zone) split to its own shard,
    /// keyed by its deterministic seed. APPENDED (discriminant 4).
    Area(u64),
    /// A STAR as its own realm inside its star system (the celestial taxonomy arc T2, owner
    /// ruling 2026-08-19): the near-star volume whose crossing turns on near-star physics —
    /// bounded by the dust-sublimation radius, drawn at the star's own photosphere. Keyed by
    /// its deterministic seed (`child_seed(system_seed, STAR_SALT, 0)`). APPENDED
    /// (discriminant 5) so the wire stays additive (`PROTO_MINOR` 22).
    Star(u64),
    /// ★ A GALAXY'S OWN SPACE, keyed by its deterministic seed. APPENDED (discriminant 6).
    ///
    /// Until S9 a galaxy had no identity and borrowed one: `RealmPath`'s `GALAXY_STANDIN` filed it under
    /// `System(1)`. That worked only while a galaxy OWNED NOTHING and nobody asked whose it was. S9 gives
    /// a galaxy star systems to own, so the question starts being asked — and `System(1)` cannot answer
    /// it, because a star system whose seed is 1 is a different thing with the same name.
    Galaxy(u64),
    /// ★ THE UNIVERSE'S OWN SPACE. APPENDED (discriminant 7), and deliberately FIELDLESS.
    ///
    /// There is exactly one universe. A seed field would always hold the same value, and a field that
    /// always holds one value is a field that eventually holds a different one by accident. It borrowed
    /// `System(0)` for the same reason the galaxy borrowed `System(1)`, with the same collision.
    Universe,
}

impl core::fmt::Display for RealmId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RealmId::Planet(seed) => write!(f, "planet-{seed:016x}"),
            RealmId::System(seed) => write!(f, "system-{seed:016x}"),
            RealmId::Ship(id) => write!(f, "ship-{id}"),
            RealmId::Station(seed) => write!(f, "station-{seed:016x}"),
            RealmId::Area(seed) => write!(f, "area-{seed:016x}"),
            RealmId::Star(seed) => write!(f, "star-{seed:016x}"),
            RealmId::Galaxy(seed) => write!(f, "galaxy-{seed:016x}"),
            // No seed to print, and none omitted: the universe is one thing.
            RealmId::Universe => write!(f, "universe"),
        }
    }
}

/// The reference frame a pose is expressed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FrameRef {
    /// Planet-centered Cartesian, planet center at origin (surface worlds).
    PlanetCentered { planet_seed: u64 },
    /// A ship's interior grid frame (moves with the hull).
    ShipLocal { ship: EntityId },
    /// Star-system space, star at origin.
    SystemSpace { system_seed: u64 },
    /// ★ A GALAXY'S OWN SPACE, and it now says WHICH galaxy (slice S9).
    ///
    /// It was fieldless while there was only ever one and it owned nothing. The universe holds
    /// sixty-one, and a frame that cannot name its own galaxy cannot be a frame two galaxies both use.
    /// Its cells are the galaxy's own two-metre step ([`Tier::Galaxy`]).
    GalaxySpace { galaxy_seed: u64 },
    /// A station's interior grid frame (moves with the hull, like a ship). APPENDED
    /// (discriminant 4) so the wire stays additive.
    StationLocal { station_seed: u64 },
    /// A sub-planet AREA frame: a zone WITHIN a planet, carrying the parent planet's seed
    /// (the fixed parent, no lookup) plus the area's own seed. APPENDED (discriminant 5).
    AreaLocal { planet_seed: u64, area_seed: u64 },
    /// The near-star frame: star centre at origin (the taxonomy arc T2). APPENDED
    /// (discriminant 6) so the wire stays additive.
    StarCentered { star_seed: u64 },
    /// ★ THE UNIVERSE'S OWN SPACE (slice S9). APPENDED (discriminant 7) so the wire stays additive, and
    /// fieldless because there is exactly one. Its cells are the universe's own thirty-two-kilometre
    /// step ([`Tier::Universe`]).
    UniverseSpace,
}

impl FrameRef {
    /// The realm whose owner is authoritative for entities expressed in this frame.
    ///
    /// ★ IT IS INFALLIBLE SINCE S9, AND THAT IS A CONSEQUENCE RATHER THAN A TIDY-UP. It used to return an
    /// `Option` for exactly one reason: galaxy space answered `None`, documented as *"no single realm
    /// owner"*, because a galaxy was not a realm and had nothing to own. A galaxy owns its star systems
    /// now and the universe owns its galaxies, so every arm names one.
    ///
    /// Leaving the `Option` would have left every caller with a `None` arm that nothing could produce —
    /// a branch no test can drive, which this project counts as a defect rather than as safety. The type
    /// now says what is true: a frame always names its realm.
    #[must_use]
    pub fn realm(self) -> RealmId {
        match self {
            FrameRef::PlanetCentered { planet_seed } => RealmId::Planet(planet_seed),
            FrameRef::ShipLocal { ship } => RealmId::Ship(ship),
            FrameRef::SystemSpace { system_seed } => RealmId::System(system_seed),
            FrameRef::GalaxySpace { galaxy_seed } => RealmId::Galaxy(galaxy_seed),
            FrameRef::UniverseSpace => RealmId::Universe,
            FrameRef::StationLocal { station_seed } => RealmId::Station(station_seed),
            FrameRef::AreaLocal { area_seed, .. } => RealmId::Area(area_seed),
            FrameRef::StarCentered { star_seed } => RealmId::Star(star_seed),
        }
    }

    /// A short human-readable label for the player-stats HUD and the `vdctl` location
    /// readout — the player's "where am I", derived from their authoritative frame, NOT a
    /// raw shard id (the client never sees shard processes; this is fence-validated and
    /// changes only on a real cross-realm move). Stub realms have no name yet, so it is
    /// the realm KIND + its seed/id; named planets/ships (P4/P8) refine the text here
    /// without changing the seam.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            FrameRef::PlanetCentered { planet_seed } => format!("Planet {planet_seed}"),
            FrameRef::ShipLocal { ship } => format!("Ship {ship}"),
            FrameRef::SystemSpace { system_seed } => format!("System {system_seed}"),
            FrameRef::GalaxySpace { galaxy_seed } => format!("Galaxy {galaxy_seed}"),
            FrameRef::UniverseSpace => "The universe".to_owned(),
            FrameRef::StationLocal { station_seed } => format!("Station {station_seed}"),
            FrameRef::AreaLocal {
                planet_seed,
                area_seed,
            } => format!("Area {area_seed} on Planet {planet_seed}"),
            FrameRef::StarCentered { star_seed } => format!("Star {star_seed}"),
        }
    }

    /// Which coordinate [`Tier`] this frame's lattice cells are measured in. Star-system space
    /// and everything nested inside it (planets, ships, stations, sub-planet areas) share the
    /// FINE tier (millimetre cells — sub-micron f64 offsets anywhere inside a system); galaxy
    /// space and out is COARSE (light-year cells, P10). A pure geometric property of the frame,
    /// NOT a feature fork on realm/shard KIND (HR3) — it selects a coordinate UNIT, nothing else.
    /// Through P3 only `SystemSpace` is live, so every live frame is [`Tier::Fine`].
    #[must_use]
    pub fn tier(self) -> Tier {
        match self {
            FrameRef::GalaxySpace { .. } => Tier::Galaxy,
            FrameRef::UniverseSpace => Tier::Universe,
            FrameRef::PlanetCentered { .. }
            | FrameRef::ShipLocal { .. }
            | FrameRef::SystemSpace { .. }
            | FrameRef::StationLocal { .. }
            | FrameRef::AreaLocal { .. }
            | FrameRef::StarCentered { .. } => Tier::Fine,
        }
    }
}

/// The forward realm→frame map: the [`FrameRef`] an entity OWNED by `realm` is expressed in (the dest
/// frame a crossing re-expresses into). It is the inverse of [`FrameRef::realm`] but NOT recoverable
/// from it — `realm()` is LOSSY in the `Area` arm (it drops the parent planet seed), so an `Area` frame
/// needs its enclosing planet as `parent` PROVENANCE (a sub-planet district lives ON a planet; the
/// crossing carries the parent via the boundary's `parent`). Every other kind is a one-field lift.
/// Returns `None` ONLY for an `Area` given without a `Planet` parent — a caller precondition failure,
/// never a silent wrong frame. This is THE one total mapping (HR3): a match over `RealmId`, not a
/// per-feature branch on realm KIND.
#[must_use]
pub fn frame_for_realm(realm: RealmId, parent: Option<RealmId>) -> Option<FrameRef> {
    match realm {
        RealmId::System(system_seed) => Some(FrameRef::SystemSpace { system_seed }),
        RealmId::Planet(planet_seed) => Some(FrameRef::PlanetCentered { planet_seed }),
        RealmId::Ship(ship) => Some(FrameRef::ShipLocal { ship }),
        RealmId::Station(station_seed) => Some(FrameRef::StationLocal { station_seed }),
        RealmId::Star(star_seed) => Some(FrameRef::StarCentered { star_seed }),
        RealmId::Galaxy(galaxy_seed) => Some(FrameRef::GalaxySpace { galaxy_seed }),
        RealmId::Universe => Some(FrameRef::UniverseSpace),
        RealmId::Area(area_seed) => match parent {
            Some(RealmId::Planet(planet_seed)) => Some(FrameRef::AreaLocal {
                planet_seed,
                area_seed,
            }),
            _ => None,
        },
    }
}

/// A POSITION, SAID OUT LOUD — the metres, the frame it is measured in, and the unit its integer cell
/// counted (owner ruling 2026-08-24, Q1 condition 3).
///
/// Every position that reaches a log goes through here. A bare number of metres is not a position: the
/// same integer means a millimetre in one frame and two metres in another, and the difference between
/// those two readings is the whole of an incident at three in the morning. Twelve diagnostics printed
/// the bare number and named neither.
///
/// A helper rather than twelve hand-written field lists, so a new diagnostic cannot print half of it.
#[must_use]
pub fn describe(pos: LatticePos, frame: FrameRef) -> String {
    let tier = frame.tier();
    let m = pos.delta_m(LatticePos::ORIGIN, tier);
    format!(
        "({:.6}, {:.6}, {:.6}) m in {frame:?} [cell = {} m, len {:.6} m]",
        m.x,
        m.y,
        m.z,
        tier.cell_edge_m(),
        m.length()
    )
}

/// FINE-tier cell edge: **2⁻¹⁰ m** (0.9765625 mm) — the largest power-of-two metre quantum ≤ 1 mm.
/// At the fine tier a [`LatticePos`] cell counts these ~mm quanta and the f64 `offset` is the sub-quantum
/// residual, so f64 precision stays at the ~10⁻¹⁹ m level ANYWHERE inside a star system. i64 quanta span
/// ≈ ±9.0×10¹⁵ m ≈ ±0.952 light-year before overflow — comfortably more than one system's active volume,
/// and beyond that you are in the COARSE tier by construction.
///
/// **Why a power of two (not 1e-3).** A power-of-two edge makes [`LatticePos::normalize`] EXACTLY
/// idempotent — `offset / edge` is a bit-shift-clean division with no rounding, so `normalize(normalize(x))
/// == normalize(x)` and the residual stays in `[0, edge)` bit-for-bit. At `1e-3` that fails: `17.9 / 1e-3`
/// is not exact, so `normalize(17.9)` lands a *negative* residual and is NOT idempotent — and idempotence is
/// the correctness basis of the cell-activation migration (a pre-activation `{cell 0, full offset}` and a
/// post-activation `{cell N, residual}` must denote the same point, `re_anchor` a no-op on the anchored
/// form). It ALSO makes every ratio in the ladder a bit shift — see [`Tier::step_exponent`].
///
/// ★ THE SENTENCE THAT USED TO END THIS DOC — *"the edge is a compile-time constant, never serialized, so
/// this moves zero bytes"* — WAS RETIRED WITH THE LADDER (slice S8). It is still true that no edge travels
/// on the wire. It is no longer true that changing one moves zero bytes: the set of steps is folded into
/// the saved-data label and into the protocol's own version
/// ([`crate::store_stamp::coordinate_generation`]), precisely so that a build which counts in different
/// units cannot silently read another's positions. Re-valuing a step is a flag day, by construction.
pub const FINE_CELL_EDGE_M: f64 = 1.0 / 1024.0;

/// A cross-rung re-statement that could not be done.
///
/// Its fields are integers and rungs only — deliberately no `f64`, so the type stays `Copy + Eq` and every
/// metre in the message is FORMATTED from a rung rather than stored. The offending offset rides as its bit
/// pattern for the same reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TierConversionError {
    /// The input is not a lawful position at its own rung. [`LatticePos::normalize`] is TOTAL, not SAFE: a
    /// finite-but-huge offset saturates its carry and can leave the offset non-finite, and a conversion
    /// fed that would produce a position with nothing to name it. Refused here instead.
    #[error(
        "not a lawful {at:?} position (step {} m): cell axis {cell_axis}, offset axis {} — it cannot be \
         re-stated in another unit",
        at.cell_edge_m(),
        f64::from_bits(*offset_bits)
    )]
    NotAdmissible {
        /// The rung the position claimed to be stated in. The step is FORMATTED from it rather than
        /// stored, so this type keeps `Eq` — an `f64` field would forfeit it, and a refusal that cannot be
        /// compared for equality cannot be asserted against in a test.
        at: Tier,
        /// The widest offending cell axis.
        cell_axis: i64,
        /// The offending offset axis, as bits (so the type keeps `Eq`).
        offset_bits: u64,
    },
    /// A position stated in one rung's cells cannot be re-stated in a FINER rung's cells: the finer
    /// lattice needs `ratio` times as many, and past its bounded domain the count is not a position any
    /// more. Refused LOUD, naming BOTH units and BOTH numbers.
    #[error(
        "a position {widest_cells} cells from its origin in {from:?} cells does not fit the {to:?} \
         lattice: the reach is {max_cells} {from:?} cells"
    )]
    BeyondReach {
        /// The rung the position is stated in.
        from: Tier,
        /// The finer rung it cannot be re-stated in.
        to: Tier,
        /// How far out it actually is, in `from` cells.
        widest_cells: u64,
        /// How far out a position may be and still be re-statable.
        max_cells: u64,
    },
}

/// The precondition BOTH directions need and neither used to have: a re-bucketed, in-domain, finite
/// position.
///
/// Bitwise `&`, never `&&`: every term is cheap and pure, and a short-circuit would leave the tail terms
/// uncoverable from a false left-hand side. A `NaN` offset fails BOTH comparisons (it is neither `>= 0`
/// nor `< step`), `+inf` fails the upper one, `-inf` the lower, and a saturated cell fails the domain.
fn admissible(p: LatticePos, at: Tier) -> Result<(), TierConversionError> {
    let step = at.cell_edge_m();
    let off = p.offset;
    let cell = p.cell;
    let ok = off.cmpge(DVec3::ZERO).all()
        & off.cmplt(DVec3::splat(step)).all()
        & (cell.max_element() <= CELL_DOMAIN_MAX)
        & (cell.min_element() >= -CELL_DOMAIN_MAX);
    if ok {
        Ok(())
    } else {
        Err(TierConversionError::NotAdmissible {
            at,
            cell_axis: worst_axis_i64(cell),
            offset_bits: worst_axis_f64(off, step).to_bits(),
        })
    }
}

/// The cell axis furthest from the origin — the one a domain refusal is about. Monomorphic so its
/// comparisons are covered once.
fn worst_axis_i64(c: I64Vec3) -> i64 {
    let pick = |a: i64, b: i64| {
        if a.unsigned_abs() >= b.unsigned_abs() {
            a
        } else {
            b
        }
    };
    pick(pick(c.x, c.y), c.z)
}

/// The offset axis that broke the `[0, step)` rule, or the widest one when none did (a domain refusal
/// still wants a number to print). `NaN` compares false against both bounds, so it is picked first.
fn worst_axis_f64(o: DVec3, step: f64) -> f64 {
    let bad = |v: f64| !(v >= 0.0 && v < step);
    let pick = |a: f64, b: f64| if bad(a) { a } else { b };
    pick(pick(o.x, o.y), o.z)
}

/// FINER → COARSER. TOTAL on an admissible input.
///
/// Dividing by a positive power of two only shrinks the magnitude, and the one signed-division overflow
/// (`i64::MIN / -1`) cannot arise because the divisor is positive — so there is nothing to refuse here. A
/// magnitude refusal on this path would be an arm no test could drive, which under HR5 is a defect rather
/// than coverage.
fn coarsen(src: LatticePos, from: Tier, to: Tier) -> LatticePos {
    let ratio = 1_i64 << (to.step_exponent() - from.step_exponent());
    let q = src.cell.div_euclid(I64Vec3::splat(ratio));
    let r = src.cell.rem_euclid(I64Vec3::splat(ratio));
    // THE ONE ROUNDING IN THE WHOLE CONVERSION. `r < ratio` is exact as an f64 and `from`'s step is a
    // power of two, so the multiply is exact; only this ADD rounds, and its exact value is strictly below
    // the destination step, so it costs at most half an ulp of that step.
    let offset = r.as_dvec3() * from.cell_edge_m() + src.offset;
    // MANDATORY, NOT DECORATION. That add can round the residual up to EXACTLY the destination step,
    // breaking the `[0, step)` invariant this type promises. Re-bucketing folds it into a carry of one and
    // is straight-line, so it adds no branch to cover — which is why it is preferable to an explicit test.
    LatticePos { cell: q, offset }.normalize(to)
}

/// A DIFFERENCE, finer → coarser. Total: the cell count only shrinks, so no bound can be crossed.
/// Monomorphic and straight-line (HR5) — every branch that could exist lives in the caller's match.
fn coarsen_separation(src: Separation, to: Tier) -> Separation {
    let ratio = 1_i64 << (to.step_exponent() - src.tier.step_exponent());
    let q = src.cells.div_euclid(I64Vec3::splat(ratio));
    let r = src.cells.rem_euclid(I64Vec3::splat(ratio));
    // The remainder returns to the residual AT THE SOURCE STEP, which is where it was measured. No
    // re-bucketing follows: a difference has no `[0, step)` invariant to restore, and forcing one
    // would move the answer.
    Separation {
        cells: q,
        residual: r.as_dvec3() * src.tier.cell_edge_m() + src.residual,
        tier: to,
    }
}

/// A DIFFERENCE, coarser → finer. Refuses above the finer rung's bound, then EXACT — the residual is
/// not touched at all, because `cells · ratio · step_to == cells · step_from` for powers of two.
fn refine_separation(src: Separation, to: Tier) -> Result<Separation, TierConversionError> {
    let ratio = 1_i64 << (src.tier.step_exponent() - to.step_exponent());
    // A DIFFERENCE has no carry from its residual (it keeps it), so the bound is the plain one: the
    // widest cell whose multiple still lands inside the domain a cell difference must stay within.
    let max_cells = (CELL_DOMAIN_MAX / ratio).unsigned_abs();
    let c = src.cells;
    let widest =
        c.x.unsigned_abs()
            .max(c.y.unsigned_abs())
            .max(c.z.unsigned_abs());
    if widest > max_cells {
        return Err(TierConversionError::BeyondReach {
            from: src.tier,
            to,
            widest_cells: widest,
            max_cells,
        });
    }
    Ok(Separation {
        cells: I64Vec3::new(c.x * ratio, c.y * ratio, c.z * ratio),
        residual: src.residual,
        tier: to,
    })
}

/// COARSER → FINER. Refuses above its bound, then EXACT — no float touches the integer half.
fn refine(src: LatticePos, from: Tier, to: Tier) -> Result<LatticePos, TierConversionError> {
    let ratio = 1_i64 << (from.step_exponent() - to.step_exponent());
    // THE BOUND, DERIVED — never a literal. A destination cell is `c*ratio + carry` with `carry` in
    // `[0, ratio)`, so the widest source cell that still lands inside the sanitized domain is
    // `(CELL_DOMAIN_MAX - (ratio - 1)) / ratio`. Bounded against CELL_DOMAIN_MAX and NOT `i64::MAX`,
    // because a result above it is clamped at wire ingress anyway and would break the one thing
    // CELL_DOMAIN_MAX exists for: a cell DIFFERENCE that cannot overflow.
    let max_cells = ((CELL_DOMAIN_MAX - (ratio - 1)) / ratio).unsigned_abs();
    let c = src.cell;
    // UNSIGNED magnitudes so `i64::MIN` cannot overflow an abs — the discipline the crossing path already
    // follows in `Separation::cells_chebyshev`.
    let widest =
        c.x.unsigned_abs()
            .max(c.y.unsigned_abs())
            .max(c.z.unsigned_abs());
    if widest > max_cells {
        return Err(TierConversionError::BeyondReach {
            from,
            to,
            widest_cells: widest,
            max_cells,
        });
    }
    let step_to = to.cell_edge_m();
    // Exact: dividing by a power of two is an exponent shift. Admissibility guarantees the offset is in
    // `[0, from_step)`, so this carry is in `[0, ratio)` and the sum below provably cannot overflow.
    let carry = (src.offset / step_to).floor();
    Ok(LatticePos {
        cell: I64Vec3::new(
            c.x * ratio + carry.x as i64,
            c.y * ratio + carry.y as i64,
            c.z * ratio + carry.z as i64,
        ),
        offset: src.offset - carry * step_to,
    })
}

/// The bounded cell domain enforced at wire ingress ([`StampedPose::sanitized`]): each `LatticePos.cell`
/// axis is clamped to `±CELL_DOMAIN_MAX`. Set to `i64::MAX / 2` so that a cell DIFFERENCE (`a.cell −
/// b.cell` in [`LatticePos::delta_m`] and the crossing rebase) can never overflow `i64` — the operation a
/// diverged/hostile sender would otherwise use to panic the receiver. Well beyond any real position: at the
/// FINE 2⁻¹⁰ edge it spans ±0.476 light-year (half the tier's overflow limit), and every in-domain pose
/// passes through BIT-FOR-BIT (`sanitized().cell() == self.cell()`), preserving determinism.
pub const CELL_DOMAIN_MAX: i64 = i64::MAX / 2;

/// Which coordinate RUNG a [`LatticePos`] cell is measured in — the unit one whole number of the lattice
/// counts. The unit differs by rung so the f64 in-cell `offset` keeps its precision at every scale:
/// millimetres inside a star system, two metres across a galaxy, thirty-two kilometres across the
/// universe. Selected by [`FrameRef::tier`] from a frame's KIND — a coordinate unit, never a feature
/// branch (HR3).
///
/// ★ THREE RUNGS, NOT TWO (slice S8, owner ruling Q1 of 2026-08-24). A millimetre-counting ruler runs out
/// at about a quarter of a light year, which is why a hundred and fifty thousand star systems would not
/// fit in a galaxy at all. Each level now counts in a step that suits it. **Every frame at or below a star
/// system keeps the millimetre step it has always had, bit for bit** — only the two levels nothing has
/// ever been drawn from get a new one.
///
/// The COARSE rung's old value was one light-year, which is **not a power of two**
/// (`9_460_730_472_580_800 = 2⁶ × 147_823_913_634_075`), so [`LatticePos::normalize`] was not exactly
/// idempotent there — the one property the fine edge's own doc says a power-of-two edge exists to
/// guarantee. That was inert only because nothing ever produced a coarse position. Re-valuing the rung to
/// two metres retires the hazard rather than documenting it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Star system and inward: `2⁻¹⁰ m` cells ([`FINE_CELL_EDGE_M`]). UNCHANGED, and every byte below a
    /// star system depends on it staying so.
    Fine,
    /// A galaxy's own space: `2¹ m` cells. **Two metres, not the design document's one** — the owner ruled
    /// the bigger step because one metre left five percent of headroom against what a hundred and fifty
    /// thousand systems need, and content grows; two costs nothing anybody can see at light-year distances
    /// and buys eight times the room.
    Galaxy,
    /// The universe's own space: `2¹⁵ m` cells (32,768 m).
    Universe,
}

impl Tier {
    /// EVERY rung, in ASCENDING step order — the totality list the saved-data stamp folds its coordinate
    /// generation over ([`crate::store_stamp::coordinate_generation`]).
    ///
    /// The ORDER is part of the meaning: the fold is order-sensitive, so a rung inserted in the middle
    /// changes the generation of every store, which is correct — a build that knows a rung the writer did
    /// not cannot be trusted to read that writer's positions.
    ///
    /// ★ A RUNG ADDED TO THE ENUM AND FORGOTTEN HERE IS THE ONE FAILURE THIS LADDER CANNOT SEE. This array
    /// has a single reader and its LENGTH is not compile-forced, so a fourth rung that never reaches it
    /// would leave the coordinate generation unmoved — and then stores, clients and the transport tag
    /// would all agree across a unit change, which is exactly the disagreement the generation exists to
    /// make loud. `every_rung_reaches_the_totality_list` is the witness.
    pub const ALL: [Tier; 3] = [Tier::Fine, Tier::Galaxy, Tier::Universe];

    /// THE DATUM OF THE LADDER: the base-two exponent of this rung's step, in metres.
    ///
    /// The exponent is stored and the edge is DERIVED from it, never the other way round. Two things fall
    /// out that a table of decimals could not give: every step is a power of two by construction (so
    /// [`LatticePos::normalize`] is exactly idempotent at every rung, and no decimal can drift), and every
    /// ratio between rungs is a DIFFERENCE OF EXPONENTS — an ordinary bit shift. The double-width integer
    /// the light-year forced (`FINE_CELLS_PER_LY: i128`) is gone with it.
    #[must_use]
    pub const fn step_exponent(self) -> i32 {
        match self {
            Tier::Fine => -10,
            Tier::Galaxy => 1,
            Tier::Universe => 15,
        }
    }

    /// Metres per cell edge at this rung — the exact quantum a [`LatticePos`] cell counts, `2^exponent`.
    ///
    /// `const` so the saved-data label's coordinate generation and the protocol's own contract can both be
    /// folded at compile time — a peer's unit must be comparable before a single byte is exchanged. Built
    /// from the exponent through the f64 bit pattern because `f64::powi` is not `const`: an IEEE-754
    /// double with a zero mantissa and a biased exponent IS the power of two, exactly.
    #[must_use]
    pub const fn cell_edge_m(self) -> f64 {
        f64::from_bits(((1023 + self.step_exponent()) as u64) << 52)
    }
}

/// A frame-local position as an integer CELL anchor + a bounded f64 local OFFSET — the
/// tiered-integer coordinate base (D-41). The `cell` is the authoritative, bit-deterministic
/// integer truth (an `i64` lattice; the FINE tier — star system and inward — is millimetres, the
/// COARSE tier — galaxy and out — a coarser unit, keyed by [`FrameRef`] tier); the `offset` is the
/// small f64 displacement WITHIN one cell, so float precision stays sub-micron locally regardless of
/// how far the frame sits from the universe origin. Cross-frame/cross-shard re-basing is exact
/// integer cell arithmetic (zero drift, bit-deterministic) — the cure for f64-absolute drift
/// (~131 km/ULP at galaxy scale).
///
/// **LIVE since the cell activation (real-scale addendum §A4).** Every producer fills the integer
/// half: the world generator, every placement, every spawn/home pose and the per-tick integrator
/// all NORMALIZE, and the one subtraction ([`LatticePos::separation`] → [`Separation`]) consumes
/// it exactly. Fields are private so all construction flows through the normalizing constructors
/// ([`LatticePos::from_metres`] publicly; `local` is crate-private) and the bounded-offset
/// invariant has a single home. The COARSE tier stays dormant (P10; its cross-tier conversion is
/// the reserved plant).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LatticePos {
    cell: I64Vec3,
    offset: DVec3,
}

impl LatticePos {
    /// The frame origin — cell `ZERO`, offset `ZERO`. THE one named origin every separation and
    /// flatten measures against (`==` [`LatticePos::default()`], named so call sites say what they
    /// mean instead of spelling a zero construction).
    pub const ORIGIN: LatticePos = LatticePos {
        cell: I64Vec3::ZERO,
        offset: DVec3::ZERO,
    };

    /// A position expressed purely as a frame-local offset, at the cell origin (`cell == ZERO`).
    ///
    /// **`pub(crate)` deliberately — the activation's structural cure (real-scale addendum §A4.5).**
    /// This constructor pins the integer half EMPTY, which was the load-bearing producer hole: the
    /// world generator, every placement and every spawn pose left the exact half unfilled and the
    /// f64 offset carrying the whole magnitude. Outside `vd-core` the only way to build a position
    /// from metres is [`LatticePos::from_metres`], which normalizes — so the mistake is unwritable,
    /// not discouraged. In-crate callers that genuinely mean "raw halves" (serde plumbing, tests)
    /// still reach it.
    #[must_use]
    pub(crate) fn local(offset: DVec3) -> LatticePos {
        LatticePos {
            cell: I64Vec3::ZERO,
            offset,
        }
    }

    /// THE public metre constructor: a frame-local position built from metres, NORMALIZED — the
    /// integer cell carries the whole-quantum part and the offset holds the sub-cell residual.
    /// Value-preserving to f64 (`cell·edge + offset` reproduces `v` exactly for every in-range
    /// input: the 2⁻¹⁰ edge makes the carry a bit-shift-clean division). There is no public way to
    /// build a position from metres that leaves the integer half empty (§A4.5's cure #1).
    #[must_use]
    pub fn from_metres(v: DVec3, tier: Tier) -> LatticePos {
        LatticePos::local(v).normalize(tier)
    }

    /// A position at an EXPLICIT integer cell anchor + a frame-local offset — the raw-halves
    /// constructor (serde plumbing, placement anchors, tests). It does NOT normalize: callers that
    /// hold metres use [`LatticePos::from_metres`]; callers that hold both halves (a wire decode, a
    /// placement row) are restating an already-normalized value.
    #[must_use]
    pub fn at(cell: I64Vec3, offset: DVec3) -> LatticePos {
        LatticePos { cell, offset }
    }

    /// The frame-local f64 offset — what physics / rendering / interpolation work in (small and
    /// cm-exact). Reads go through this accessor so a future integer-offset migration stays
    /// one-type-local.
    #[must_use]
    pub fn offset(self) -> DVec3 {
        self.offset
    }

    /// The integer CELL anchor — the authoritative, bit-deterministic coarse coordinate, LIVE on
    /// every producer since the cell activation. Mirrors [`LatticePos::offset`] — reads go through
    /// one accessor so a future integer-offset migration stays one-type-local. ⚠ This is a HALF of
    /// a position, never a position: subtract positions with [`LatticePos::separation`], flatten
    /// with [`LatticePos::delta_m`].
    #[must_use]
    pub fn cell(self) -> I64Vec3 {
        self.cell
    }

    /// This position displaced by `delta` metres, NORMALIZED — the one in-frame move. Replaces the
    /// deleted `map_offset` (real-scale addendum §A4.5 cure #2): `map_offset` preserved the cell but
    /// let the offset grow unboundedly, so every caller had to REMEMBER a trailing `.normalize()`.
    /// Here the normalize is internal, so the bounded-offset invariant cannot be forgotten. The one
    /// production mover (the per-tick integrator) used `map_offset(|o| o + step).normalize(tier)` —
    /// the identical two operations in the identical order, so this is byte-identical there by
    /// construction.
    #[must_use]
    pub fn translated(self, delta: DVec3, tier: Tier) -> LatticePos {
        LatticePos {
            cell: self.cell,
            offset: self.offset + delta,
        }
        .normalize(tier)
    }

    /// Re-bucket so the f64 `offset` lands back in `[0, cell_edge)` per axis, carrying the integer
    /// overflow into the `cell` anchor — the operation that keeps offsets small (and f64-precise) after
    /// any accumulation. Exact: the carry is the floored quotient `offset / edge`, subtracted back off in
    /// the same units, so `cell * edge + offset` is preserved to f64. Straight-line (no branches); an
    /// already-normalized pose is a fixed point. Through P3 no production code calls this (poses ride at
    /// `cell == ZERO` with `offset` carrying the full frame-local metres — [`LatticePos::local`]); it is
    /// the D-41 cell math planted for S5 activation, exercised now only by unit/proptests (byte-floor
    /// holds: nothing in the shipped path re-buckets, so the wire stays cell-0 identical).
    #[must_use]
    pub fn normalize(self, tier: Tier) -> LatticePos {
        let edge = tier.cell_edge_m();
        let carry = (self.offset / edge).floor();
        // SATURATING cell carry: a finite-but-huge `offset` (a diverged sender, a hostile wire pose)
        // makes `carry` exceed the i64 range; a plain `self.cell + carry.as_i64vec3()` PANICS in debug
        // (i.e. across the whole test + coverage suite) on that overflow. Saturating keeps `normalize`
        // TOTAL (never panics) while staying exact for every in-domain pose — the carry is the floored
        // quotient, subtracted back off in the same units, so `cell*edge + offset` is preserved to f64
        // wherever the cell does not saturate. Per-component so one saturated axis does not poison the
        // others; the `as i64` cast already saturates the f64→i64 conversion (NaN → 0). At the 2⁻¹⁰ edge
        // `offset / edge` is an exact bit-shift-scale, so `normalize` is exactly idempotent.
        LatticePos {
            cell: I64Vec3::new(
                self.cell.x.saturating_add(carry.x as i64),
                self.cell.y.saturating_add(carry.y as i64),
                self.cell.z.saturating_add(carry.z as i64),
            ),
            offset: self.offset - carry * edge,
        }
    }

    /// Re-state this position in another rung's cells.
    ///
    /// **SAME RUNG ⇒ `Ok(self)`, bit for bit, unnormalized** — and this arm is tested FIRST, before any
    /// admissibility check, on purpose. It is the only live path in the program today, and the poses it
    /// carries ride at cell ZERO with the whole frame-local distance in the offset
    /// ([`LatticePos::local`]). Those are not normalized, so an admissibility test placed ahead of this
    /// arm would refuse every position the shipped path has.
    ///
    /// **FINER → COARSER** is total once admissible. It is exact in the whole-number half *except for a
    /// single carry case* — see the note on the residual below, which is why what a round trip may assert
    /// is the POSITION and never the cell.
    ///
    /// **COARSER → FINER** is the one direction that can be refused, and it is refused loudly rather than
    /// wrapped. A finer lattice needs `ratio` times as many cells, and past its bounded domain the count
    /// is not a position any more: scaling an absolute galaxy cell down to millimetres reaches `2⁷³`
    /// against an `i64::MAX` of `2⁶³`, over by a factor of **1024**. Below its bound the direction is
    /// EXACT — no float touches the integer half.
    ///
    /// ★ WHAT A ROUND TRIP MAY ASSERT. Upward, the residual add can round UP to exactly the destination
    /// step (`2047 × 2⁻¹⁰` plus the largest sub-cell offset rounds to exactly `2.0`), and the mandatory
    /// re-bucketing then carries — moving the whole-number half by one cell. The resulting pair is the
    /// correctly-rounded representation of the same point, so the POSITION is right and the CELL is not
    /// what was asserted. A property test over random draws does not see this: it is 1,600 hits in a
    /// 3,200-case directed sweep and 0 in 200,000 random ones.
    ///
    /// # Errors
    /// [`TierConversionError::NotAdmissible`] when the input is not a lawful position at `from` — a
    /// non-finite offset, an offset outside `[0, step)` after re-bucketing, or a cell outside the
    /// sanitized domain. [`TierConversionError::BeyondReach`] when refining past the finer rung's reach.
    pub fn convert_tier(self, from: Tier, to: Tier) -> Result<LatticePos, TierConversionError> {
        if from.step_exponent() == to.step_exponent() {
            return Ok(self);
        }
        let src = self.normalize(from);
        admissible(src, from)?;
        if from.step_exponent() < to.step_exponent() {
            Ok(coarsen(src, from, to))
        } else {
            refine(src, from, to)
        }
    }

    /// THE subtraction (real-scale addendum §A4.3). Every position difference in the program is
    /// this call: an exact `i64` cell delta plus one f64 residual delta, as a [`Separation`].
    /// TOTAL over every sanitized-domain pair (`CELL_DOMAIN_MAX = i64::MAX/2` exists precisely so
    /// this difference cannot overflow), EXACT in the integer half, and branchless.
    #[must_use]
    pub fn separation(self, origin: LatticePos, tier: Tier) -> Separation {
        Separation {
            cells: self.cell - origin.cell,
            residual: self.offset - origin.offset,
            tier,
        }
    }

    /// This position MINUS `origin` (both same frame + tier) expressed in **metres** — the one-line
    /// spelling of `self.separation(origin, tier).metres()` kept for the ~40 sites that legitimately
    /// want metres (render, rapier contact, thrust, gauges), so their diff is zero. The honest
    /// exactness bound lives on [`Separation::metres`], where it belongs.
    #[must_use]
    pub fn delta_m(self, origin: LatticePos, tier: Tier) -> DVec3 {
        self.separation(origin, tier).metres()
    }
}

/// **THE ONE OPERATION** (real-scale addendum §A4.3): the exact difference of two positions in ONE
/// frame at ONE tier. This is not a position — a position says WHERE; a separation says HOW FAR
/// APART. The type distinction IS the foot-gun cure: a separation can be rotated, squared, compared
/// and flattened; a position can only be separated from another position or translated. Nothing
/// else is offered, so `.offset()` can no longer be mistaken for a position difference.
///
/// Never serialized — a local computation type only (nothing new crosses any boundary, SL6 = NONE).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Separation {
    cells: I64Vec3,
    residual: DVec3,
    tier: Tier,
}

/// The largest per-axis |cell delta| whose 3-axis square fits `i128` with headroom: squaring is safe
/// iff every axis is within ±`CELL_DOMAIN_MAX` (`3 · (i64::MAX/2)² ≈ 6.38e37 < i128::MAX ≈ 1.70e38`).
/// A hostile-but-in-domain pair at OPPOSITE domain ends measures `|Δ| = 2·CELL_DOMAIN_MAX` per axis,
/// whose square-sum OVERFLOWS `i128` (`≈ 2.55e38`) — the H-01 hole. [`Separation::cells_sq`] refuses
/// that case (`None`) instead of panicking; the containment verdict's Chebyshev pre-test makes the
/// refusal unreachable for every lawful shell (all shell thresholds are far inside this bound).
const SEPARATION_SQUARE_SAFE_MAX: i64 = CELL_DOMAIN_MAX;

impl Separation {
    /// The exact integer cell delta — the authoritative half of the answer.
    #[must_use]
    pub fn cells(self) -> I64Vec3 {
        self.cells
    }

    /// The sub-cell f64 residual delta (each operand `< 1` cell for normalized inputs, so its error
    /// is at most `ulp(0.98 mm) ≈ 2.2e-19 m`). The integer comparator DISCARDS it (§A5.3).
    #[must_use]
    pub fn residual(self) -> DVec3 {
        self.residual
    }

    /// The tier this separation's cells are counted in.
    #[must_use]
    pub fn tier(self) -> Tier {
        self.tier
    }

    /// (i) THE AUTHORITY COMPARATOR — the exact squared cell length as `i128`. No float, no square
    /// root: bit-identical on every host at every magnitude. GUARDED (§A4.4): `None` when any axis
    /// exceeds [`SEPARATION_SQUARE_SAFE_MAX`] — the hostile-but-in-domain overflow case (H-01) made
    /// a refusal instead of a debug panic on the containment hot path. Callers pre-test with
    /// [`Separation::cells_chebyshev`] against their threshold, which makes `None` unreachable for
    /// every lawful shell; a `None` that does surface reads as "outside" (a separation too large to
    /// square is out of every shippable region).
    #[must_use]
    pub fn cells_sq(self) -> Option<i128> {
        let safe = self.cells_chebyshev() <= SEPARATION_SQUARE_SAFE_MAX;
        let sq = |v: i64| i128::from(v) * i128::from(v);
        // The multiply only runs on the safe arm — the unsafe arm returns before any square.
        if safe {
            Some(sq(self.cells.x) + sq(self.cells.y) + sq(self.cells.z))
        } else {
            None
        }
    }

    /// (i′) Chebyshev on integers — the per-axis max |cell delta|, for box shapes and for the
    /// overflow pre-test. TOTAL BY CONSTRUCTION, not by domain argument: `i64::abs` PANICS in debug
    /// on `i64::MIN`, and `LatticePos::normalize` saturates to `i64::MIN`/`i64::MAX` rather than to
    /// `±CELL_DOMAIN_MAX`, so `i64::MIN` is reachable in principle and a panic here would land on the
    /// containment hot path — in debug, which is the whole test and coverage suite. `unsigned_abs`
    /// cannot overflow; the saturating cast back keeps `i64::MIN` reading as "further than any lawful
    /// threshold", which is what every caller already does with a too-large separation.
    #[must_use]
    pub fn cells_chebyshev(self) -> i64 {
        let m = self
            .cells
            .x
            .unsigned_abs()
            .max(self.cells.y.unsigned_abs())
            .max(self.cells.z.unsigned_abs());
        // `min` then cast: `2^63` (from `i64::MIN`) saturates to `i64::MAX`; every other value is
        // unchanged, so this is bit-identical to the old body over the whole in-domain range.
        m.min(i64::MAX.unsigned_abs()) as i64
    }

    /// (ii) THE FLATTEN — metres, for what consumes metres: render, rapier, thrust, gauges. THE
    /// ERROR IS RELATIVE TO THE ANSWER, NEVER TO THE WORLD (§A5.2): exact iff `|Δcell| ≤ 2⁵³` per
    /// axis (≈ 8.8×10¹² m ≈ 59 AU at FINE); beyond that the `as_dvec3` cast rounds and the error is
    /// ≤ 1 ulp of the answer (0.5 m per axis at the full 2.25e15 m GALAXY radius, the widest span
    /// the lattice ever holds — nine orders inside
    /// any band it could be compared to, and no authority commit compares metres there: those sites
    /// are integer).
    #[must_use]
    pub fn metres(self) -> DVec3 {
        self.cells.as_dvec3() * self.tier.cell_edge_m() + self.residual
    }

    /// Re-state this DIFFERENCE in another rung's cells (slice S9) — the primitive a cross-rung
    /// crossing is built from.
    ///
    /// ★ WHY A SEPARATION NEEDS ITS OWN CONVERSION AND CANNOT BORROW [`LatticePos::convert_tier`].
    /// That one converts a POSITION, and a position is admissible only with a residual inside
    /// `[0, step)` and a cell inside the sanitized domain. A difference obeys neither: its residual is
    /// routinely negative (it is a subtraction), and it is not required to be sub-cell. Feeding a
    /// difference to the position converter would be refused as "not a lawful position" for the
    /// ordinary case of pointing backwards.
    ///
    /// What a difference must preserve is its METRES, and this preserves them exactly:
    /// - **SAME RUNG ⇒ `Ok(self)`, bit for bit.** Every crossing in the world today, so the S9 climb
    ///   costs the shipped path nothing — proved by measurement, not by reading.
    /// - **FINER → COARSER** is total and needs no bound: fewer cells, never more. The whole-number
    ///   half divides exactly (`div_euclid`) and the remainder is folded back into the residual at the
    ///   SOURCE step, which is a power of two times an integer below the ratio — exact. Only the final
    ///   add rounds, by at most half an ulp of the destination step.
    /// - **COARSER → FINER** multiplies the cell count by the ratio, so it can leave the domain, and is
    ///   REFUSED there rather than wrapped. Below the bound it is EXACT and the residual is carried
    ///   through UNTOUCHED — `cells·ratio·step_to == cells·step_from` exactly, both being powers of two.
    ///
    /// # Errors
    /// [`TierConversionError::BeyondReach`] when refining a difference too wide for the finer rung.
    pub fn convert_tier(self, to: Tier) -> Result<Separation, TierConversionError> {
        if self.tier.step_exponent() == to.step_exponent() {
            return Ok(self);
        }
        if self.tier.step_exponent() < to.step_exponent() {
            Ok(coarsen_separation(self, to))
        } else {
            refine_separation(self, to)
        }
    }

    /// (iii) THE RE-ANCHOR — this difference re-expressed as a position relative to a new `origin`.
    /// Exact integer add + one f64 add, no normalize (the exact-rebase theorem §A5.1:
    /// `a.separation(b, t).from_origin(b) == a` bit-for-bit; a proptest pins it over the domain).
    #[must_use]
    pub fn from_origin(self, origin: LatticePos) -> LatticePos {
        LatticePos {
            cell: origin.cell + self.cells,
            offset: origin.offset + self.residual,
        }
    }

    /// The ONE operation that is not unconditional — rotation (§A4.6, THE RESTATED ROTATION LAW).
    ///
    /// A cell count is measured along one frame's axes; no rotation of an integer lattice vector is
    /// generally an integer lattice vector, so a rotated separation must FOLD through f64 metres —
    /// and rotating a lever of magnitude `L` costs `L · f64::EPSILON`. The rule, derived from that
    /// cost rather than restated from habit:
    ///
    /// - **identity quaternion ⇒ bit-exact passthrough** — no fold, no cost (EXACT equality, never
    ///   an epsilon: "nearly unrotated" is rotated, and a tolerance would silently resume folding
    ///   for every slowly-spinning realm);
    /// - **non-identity within [`rotation_exact_reach_m`] ⇒ fold, rotate, renormalize** — the fold
    ///   cost is at most one cell, so the millimetre grid survives;
    /// - **beyond the reach ⇒ refused LOUD** ([`crate::frame::FrameError::RotationBeyondExactReach`])
    ///   — a galaxy-scale spinning realm is precisely a COARSE-tier object, and the refusal is
    ///   P10's named trigger (R2).
    ///
    /// This replaces the pre-activation predicate ("rotated AND `cell != ZERO` ⇒ refuse"), which
    /// the activation made active-hostile: after cells go live essentially every hop is cross-cell,
    /// so the old rule would have refused every rotating realm the moment one existed (H-11). ONE
    /// rule, ONE call site — the `FrameXform` twin is deleted with that type.
    ///
    /// # Errors
    /// [`crate::frame::FrameError::RotationBeyondExactReach`] when `q` is not the identity and this
    /// separation's magnitude exceeds the tier's exact rotation reach.
    pub fn rotated(self, q: DQuat) -> Result<Separation, FrameError> {
        if q == DQuat::IDENTITY {
            return Ok(self);
        }
        let reach = rotation_exact_reach_m(self.tier);
        let metres = self.metres();
        if metres.length() > reach {
            return Err(FrameError::RotationBeyondExactReach);
        }
        let folded = LatticePos::from_metres(q * metres, self.tier);
        Ok(Separation {
            cells: folded.cell,
            residual: folded.offset,
            tier: self.tier,
        })
    }
}

/// The largest lever a rotation may fold through f64 metres while costing at most ONE cell
/// (§A4.6): `cell_edge / f64::EPSILON`. DERIVED from the tier — no literal, and it re-derives
/// itself the day the COARSE tier lights up. FINE: `2⁻¹⁰ / 2⁻⁵² = 2⁴² m ≈ 29.399 AU` (= 2⁵²
/// cells). A threshold of 2⁵³ cells would be one octave loose — the fold there costs two cells,
/// admitting a case that violates the millimetre grid this activation exists to restore (H-06).
#[must_use]
pub fn rotation_exact_reach_m(tier: Tier) -> f64 {
    tier.cell_edge_m() / f64::EPSILON
}

/// A pose + motion state bound to one frame at one analytic-clock instant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StampedPose {
    pub frame: FrameRef,
    pub pos: LatticePos,
    pub vel: DVec3,
    pub orient: DQuat,
    pub universe_tick: UniverseTick,
}

impl StampedPose {
    /// A rest pose at a position — the common spawn/home/test constructor, and PRODUCER #4 of the
    /// cell activation (real-scale addendum §A4.5/H-17): it sits on the live login/home path
    /// (`StoredHome::in_realm` → every account's spawn), so it NORMALIZES — the integer half is
    /// filled at birth exactly as the integrator fills it on the first step. Making `local`
    /// crate-private does not reach this constructor (same crate), which is why the normalize is
    /// stated here rather than inherited. Spawn-pose wire BYTES change with this (a declared golden
    /// refresh, addendum §A6.1 S1a); the VALUE is identical to f64.
    #[must_use]
    pub fn at_rest(frame: FrameRef, pos: DVec3, universe_tick: UniverseTick) -> StampedPose {
        StampedPose {
            frame,
            pos: LatticePos::from_metres(pos, frame.tier()),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick,
        }
    }

    /// A copy with every non-finite component replaced by a safe default (`pos`/`vel`
    /// per-component → 0, `orient` → identity). Delivered poses ride the wire, so a
    /// corrupt/diverged sender could carry `NaN`/`Inf`; the CLIENT must never feed one
    /// into its render transforms (it would poison the whole scene graph) — so the view
    /// sanitizes HERE, at the single decode-ingress chokepoint, and EVERY downstream
    /// consumer (interpolation, the render snapshot, the DevState rows) inherits finite
    /// values. The "never trust/panic on network input" mandate, applied once.
    #[must_use]
    pub fn sanitized(self) -> StampedPose {
        StampedPose {
            frame: self.frame,
            // Sanitize the local offset (an f64 can be NaN/Inf); the integer cell is finite by type but
            // a diverged/hostile sender can still ship one large enough to overflow a downstream cell
            // DIFFERENCE (`delta_m`, the crossing rebase) and panic the receiver — so CLAMP it to the
            // bounded `±CELL_DOMAIN_MAX` domain. Every in-domain pose (all of prod, cell ZERO) passes
            // through BIT-FOR-BIT, so `sanitized().cell() == self.cell()` and byte-identity holds.
            pos: LatticePos {
                cell: self.pos.cell.clamp(
                    I64Vec3::splat(-CELL_DOMAIN_MAX),
                    I64Vec3::splat(CELL_DOMAIN_MAX),
                ),
                offset: finite_or_zero(self.pos.offset),
            },
            vel: finite_or_zero(self.vel),
            orient: if self.orient.is_finite() {
                self.orient
            } else {
                DQuat::IDENTITY
            },
            universe_tick: self.universe_tick,
        }
    }

    /// Closed-form ballistic advance under constant acceleration for `dt_s` seconds
    /// (Category A: the ONLY way frozen/transient motion is re-advanced across hosts —
    /// never by re-stepping a physics engine, which is not cross-binary deterministic).
    #[must_use]
    pub fn advanced_ballistic(
        &self,
        accel: DVec3,
        dt_s: f64,
        new_tick: UniverseTick,
    ) -> StampedPose {
        StampedPose {
            frame: self.frame,
            // Closed-form advance of the frame-local offset; the cell anchor is preserved (ballistic
            // re-advance is within one frame — cell crossing is the deferred P4/P5 re-centering,
            // galaxy ly-cells at P10).
            pos: LatticePos {
                cell: self.pos.cell,
                offset: self.pos.offset + self.vel * dt_s + 0.5 * accel * dt_s * dt_s,
            },
            vel: self.vel + accel * dt_s,
            orient: self.orient,
            universe_tick: new_tick,
        }
    }
}

/// A finite scalar (`NaN`/`±Inf` → 0) — the per-component guard for [`StampedPose::sanitized`].
fn finite(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// A vector with each non-finite component zeroed (one bad component does not nuke the
/// other two).
fn finite_or_zero(v: DVec3) -> DVec3 {
    DVec3::new(finite(v.x), finite(v.y), finite(v.z))
}

#[cfg(test)]
mod tests {

    #[test]
    fn subtracting_before_flattening_is_exact_where_flattening_first_is_not() {
        // ★ SLICE S4's GATE, and it is red today on the world as it stands — no new world needed.
        //
        // Two objects near the star placement radius. Their separation is computed two ways:
        //   WRONG — flatten each from the frame origin, then subtract the two metre values;
        //   RIGHT — subtract in the integer lattice, then flatten the difference.
        //
        // ★ THE GATE ALL THREE DESIGNS PROPOSED CANNOT FAIL, and that is why this one is written
        // differently. They all chose "two things 100 m apart must draw 100 m apart". At this radius a
        // position is about 1.53e18 fine cells, whose f64 spacing is 256 cells — and 100 m is exactly
        // 102,400 cells, which is 400 × 256. Both endpoints round by the SAME amount, the errors
        // cancel, and the answer is exactly 100.000 m. Every WHOLE-METRE separation is exact here, for
        // the same reason. A gate built on one would have passed for ever while the defect stood.
        //
        // So this sweeps separations that are NOT multiples of the rounding quantum.
        const R_CELLS: i64 = 1_534_955_097_245_569_024; // the placement radius, in fine cells
        let tier = Tier::Fine;
        let base = LatticePos::at(I64Vec3::new(R_CELLS, 0, 0), DVec3::ZERO);

        let mut worst_flat_first = 0.0_f64;
        let mut worst_lattice_first = 0.0_f64;
        for sep_cells in [102_401_i64, 102_437, 102_501, 102_655, 103_000] {
            let other = LatticePos::at(I64Vec3::new(R_CELLS + sep_cells, 0, 0), DVec3::ZERO);
            let truth = sep_cells as f64 * tier.cell_edge_m();

            // WRONG: two flattens, then a subtraction. Each flatten rounds independently.
            let flat_first = (other.delta_m(LatticePos::ORIGIN, tier)
                - base.delta_m(LatticePos::ORIGIN, tier))
            .x;
            // RIGHT: the subtraction happens on the integers, which cannot round at all.
            let lattice_first = other.delta_m(base, tier).x;

            worst_flat_first = worst_flat_first.max((flat_first - truth).abs());
            worst_lattice_first = worst_lattice_first.max((lattice_first - truth).abs());
        }

        // THE DEFECT, MEASURED. Bounded by one step of the ABSOLUTE coordinate — a quarter of a metre
        // per axis at this radius. (The source's own doc claims half a metre; that bound is stale and
        // somebody will quote it.)
        assert!(
            worst_flat_first > 0.0,
            "flattening first must be measurably wrong, or this gate proves nothing"
        );
        assert!(
            worst_flat_first <= 0.25,
            "the error is bounded by one step of the absolute coordinate: {worst_flat_first} m"
        );

        // AND THE CURE IS EXACT — not smaller, EXACT. Two ships flying in convoy draw the distance
        // they actually are apart.
        assert_eq!(
            worst_lattice_first, 0.0,
            "subtracting in the lattice must be exact, not merely better"
        );
    }

    #[test]
    fn the_whole_metre_gate_the_designs_proposed_is_green_today() {
        // KEPT AS EVIDENCE, because a plan that everyone believed said otherwise. This asserts the
        // thing that made their gate useless: at this radius a whole-metre separation flattens
        // EXACTLY, so a test built on one could never have failed however wrong the code was.
        const R_CELLS: i64 = 1_534_955_097_245_569_024;
        let tier = Tier::Fine;
        let base = LatticePos::at(I64Vec3::new(R_CELLS, 0, 0), DVec3::ZERO);
        let other = LatticePos {
            cell: I64Vec3::new(R_CELLS + 102_400, 0, 0), // exactly 100 m
            offset: DVec3::ZERO,
        };
        let flat_first =
            (other.delta_m(LatticePos::ORIGIN, tier) - base.delta_m(LatticePos::ORIGIN, tier)).x;
        assert_eq!(
            flat_first, 100.0,
            "a whole-metre separation is exact even with the defect present — which is why the \
             proposed gate could not fail"
        );
    }

    #[test]
    fn a_described_position_names_its_metres_its_frame_and_its_unit() {
        // Q1 CONDITION 3, asserted rather than trusted to a reviewer. A bare number of metres is not a
        // position: the same integer counts millimetres in one frame and metres in another. If a
        // diagnostic can print one without the other, the rule is not enforced anywhere.
        let f = FrameRef::SystemSpace { system_seed: 7 };
        let text = describe(
            LatticePos::from_metres(DVec3::new(1.5, 0.0, 0.0), f.tier()),
            f,
        );
        assert!(text.contains("1.500000"), "the metres: {text}");
        assert!(text.contains("SystemSpace"), "the frame: {text}");
        assert!(
            text.contains(&f.tier().cell_edge_m().to_string()),
            "the unit its cell counted: {text}"
        );
    }
    use super::*;
    use crate::entity_kind::EntityKind;

    fn ship_id() -> EntityId {
        EntityId::pack(EntityKind::Ship, 3, 17, 0xABCDEF)
    }

    #[test]
    fn realm_mapping_per_frame() {
        assert_eq!(
            FrameRef::PlanetCentered { planet_seed: 5 }.realm(),
            RealmId::Planet(5)
        );
        assert_eq!(
            FrameRef::SystemSpace { system_seed: 9 }.realm(),
            RealmId::System(9)
        );
        assert_eq!(
            FrameRef::ShipLocal { ship: ship_id() }.realm(),
            RealmId::Ship(ship_id())
        );
        // ★ THE ARM THAT USED TO ASSERT `None`. Galaxy space named no realm because a galaxy owned
        // nothing; it owns its star systems now, so it names one like every other frame — and the
        // universe, which used to have no frame at all, names itself.
        assert_eq!(
            FrameRef::GalaxySpace { galaxy_seed: 3 }.realm(),
            RealmId::Galaxy(3)
        );
        assert_eq!(FrameRef::UniverseSpace.realm(), RealmId::Universe);
        assert_eq!(
            FrameRef::StationLocal { station_seed: 7 }.realm(),
            RealmId::Station(7)
        );
        assert_eq!(
            FrameRef::AreaLocal {
                planet_seed: 5,
                area_seed: 8
            }
            .realm(),
            RealmId::Area(8)
        );
    }

    #[test]
    fn frame_for_realm_is_the_total_forward_map() {
        // The one-field lifts.
        assert_eq!(
            frame_for_realm(RealmId::System(9), None),
            Some(FrameRef::SystemSpace { system_seed: 9 })
        );
        assert_eq!(
            frame_for_realm(RealmId::Planet(5), None),
            Some(FrameRef::PlanetCentered { planet_seed: 5 })
        );
        assert_eq!(
            frame_for_realm(RealmId::Ship(ship_id()), None),
            Some(FrameRef::ShipLocal { ship: ship_id() })
        );
        assert_eq!(
            frame_for_realm(RealmId::Station(7), None),
            Some(FrameRef::StationLocal { station_seed: 7 })
        );
        // The Area arm needs a PLANET parent as provenance (the parent seed is not in RealmId::Area).
        assert_eq!(
            frame_for_realm(RealmId::Area(8), Some(RealmId::Planet(5))),
            Some(FrameRef::AreaLocal {
                planet_seed: 5,
                area_seed: 8
            })
        );
        // An Area without a Planet parent (missing, or a non-Planet realm) is a loud None, never a
        // silent wrong frame.
        assert_eq!(frame_for_realm(RealmId::Area(8), None), None);
        assert_eq!(
            frame_for_realm(RealmId::Area(8), Some(RealmId::System(9))),
            None
        );
    }

    #[test]
    fn label_is_human_readable_per_frame_kind() {
        // The player-facing location string for every frame variant (the HUD / vdctl
        // readout) — kind + seed/id for stub realms; GalaxySpace has no seed.
        assert_eq!(
            FrameRef::PlanetCentered { planet_seed: 5 }.label(),
            "Planet 5"
        );
        assert_eq!(FrameRef::SystemSpace { system_seed: 9 }.label(), "System 9");
        assert_eq!(
            FrameRef::ShipLocal { ship: ship_id() }.label(),
            format!("Ship {}", ship_id())
        );
        // A galaxy says WHICH galaxy now — there are sixty-one, and "Galaxy" named none of them.
        assert_eq!(FrameRef::GalaxySpace { galaxy_seed: 4 }.label(), "Galaxy 4");
        assert_eq!(FrameRef::UniverseSpace.label(), "The universe");
        assert_eq!(
            FrameRef::StationLocal { station_seed: 7 }.label(),
            "Station 7"
        );
        assert_eq!(
            FrameRef::AreaLocal {
                planet_seed: 5,
                area_seed: 8
            }
            .label(),
            "Area 8 on Planet 5"
        );
    }

    #[test]
    fn realm_display_is_unambiguous() {
        assert_eq!(RealmId::Planet(0xAB).to_string(), "planet-00000000000000ab");
        assert_eq!(RealmId::System(0xCD).to_string(), "system-00000000000000cd");
        assert!(
            RealmId::Ship(ship_id())
                .to_string()
                .starts_with("ship-ent-01.")
        );
        assert_eq!(
            RealmId::Station(0xEF).to_string(),
            "station-00000000000000ef"
        );
        assert_eq!(RealmId::Area(0x12).to_string(), "area-0000000000000012");
        // ★ THE S9 ARMS. A galaxy prints its seed like every other keyed realm; the universe prints
        // no seed at all, because there is exactly one and a number after it would imply otherwise.
        assert_eq!(RealmId::Galaxy(0x34).to_string(), "galaxy-0000000000000034");
        assert_eq!(RealmId::Universe.to_string(), "universe");
        // …and every arm prints something DIFFERENT, which is what "unambiguous" means and what a
        // list of individual equalities does not actually check.
        let all = [
            RealmId::Planet(1).to_string(),
            RealmId::System(1).to_string(),
            RealmId::Ship(ship_id()).to_string(),
            RealmId::Station(1).to_string(),
            RealmId::Area(1).to_string(),
            RealmId::Star(1).to_string(),
            RealmId::Galaxy(1).to_string(),
            RealmId::Universe.to_string(),
        ];
        let unique: std::collections::BTreeSet<&String> = all.iter().collect();
        assert_eq!(
            unique.len(),
            all.len(),
            "two realms print the same label: {all:?}"
        );
    }

    #[test]
    fn station_and_area_arms_round_trip_over_the_wire() {
        // The APPENDED arms (RealmId 3/4, FrameRef 4/5) are NON-zero discriminants, so this
        // exercises the additive wire growth (existing arms keep their index) — the pose.rs
        // non-zero-cell precedent applied to the realm taxonomy.
        for r in [RealmId::Station(0xABCD), RealmId::Area(0x9876)] {
            let bytes = postcard::to_allocvec(&r).expect("encode realm");
            assert_eq!(postcard::from_bytes::<RealmId>(&bytes).expect("decode"), r);
        }
        for f in [
            FrameRef::StationLocal { station_seed: 42 },
            FrameRef::AreaLocal {
                planet_seed: 7,
                area_seed: 99,
            },
        ] {
            let bytes = postcard::to_allocvec(&f).expect("encode frame");
            assert_eq!(postcard::from_bytes::<FrameRef>(&bytes).expect("decode"), f);
        }
    }

    #[test]
    fn at_rest_has_zero_motion() {
        let p = StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::new(1.0, 2.0, 3.0),
            UniverseTick(40),
        );
        assert_eq!(p.vel, DVec3::ZERO);
        assert_eq!(p.orient, DQuat::IDENTITY);
        assert_eq!(p.universe_tick, UniverseTick(40));
    }

    #[test]
    fn sanitized_replaces_non_finite_components_with_safe_defaults() {
        // An all-finite pose passes through unchanged.
        let good = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            pos: LatticePos::local(DVec3::new(1.0, 2.0, 3.0)),
            vel: DVec3::new(-1.0, 0.0, 4.0),
            orient: DQuat::from_rotation_y(0.5),
            universe_tick: UniverseTick(7),
        };
        assert_eq!(good.sanitized(), good);
        // Non-finite pos/vel components are zeroed PER-COMPONENT; a non-finite orient
        // collapses to identity. Frame + tick are preserved.
        let bad = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            pos: LatticePos::local(DVec3::new(f64::NAN, 2.0, f64::INFINITY)),
            vel: DVec3::new(1.0, f64::NEG_INFINITY, 3.0),
            orient: DQuat::from_xyzw(f64::NAN, 0.0, 0.0, 1.0),
            universe_tick: UniverseTick(7),
        };
        let s = bad.sanitized();
        assert_eq!(s.pos.offset(), DVec3::new(0.0, 2.0, 0.0));
        assert_eq!(s.vel, DVec3::new(1.0, 0.0, 3.0));
        assert_eq!(s.orient, DQuat::IDENTITY);
        assert_eq!(s.frame, bad.frame);
        assert_eq!(s.universe_tick, UniverseTick(7));
    }

    #[test]
    fn ballistic_advance_is_exact_kinematics() {
        let p0 = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            pos: LatticePos::local(DVec3::new(0.0, 100.0, 0.0)),
            vel: DVec3::new(10.0, 0.0, 0.0),
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        };
        let g = DVec3::new(0.0, -2.0, 0.0);
        let p1 = p0.advanced_ballistic(g, 3.0, UniverseTick(60));
        // x = x0 + v*t; y = y0 + 0.5*a*t^2; v_y = a*t
        assert_eq!(p1.pos.offset(), DVec3::new(30.0, 100.0 - 9.0, 0.0));
        assert_eq!(p1.vel, DVec3::new(10.0, -6.0, 0.0));
        assert_eq!(p1.universe_tick, UniverseTick(60));
        assert_eq!(p1.frame, p0.frame);
    }

    #[test]
    fn ballistic_advance_is_composable() {
        // Advancing 2s then 3s equals advancing 5s (closed form, no accumulation drift).
        let p0 = StampedPose {
            frame: FrameRef::GalaxySpace { galaxy_seed: 0 },
            pos: LatticePos::local(DVec3::new(1.0, 2.0, 3.0)),
            vel: DVec3::new(-1.0, 0.5, 2.0),
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        };
        let a = DVec3::new(0.1, -0.2, 0.3);
        let split = p0
            .advanced_ballistic(a, 2.0, UniverseTick(40))
            .advanced_ballistic(a, 3.0, UniverseTick(100));
        let whole = p0.advanced_ballistic(a, 5.0, UniverseTick(100));
        assert!((split.pos.offset() - whole.pos.offset()).length() < 1e-9);
        assert!((split.vel - whole.vel).length() < 1e-12);
    }

    #[test]
    fn stamped_pose_serde_roundtrip() {
        let p = StampedPose {
            frame: FrameRef::ShipLocal { ship: ship_id() },
            pos: LatticePos::local(DVec3::new(4.0, 5.0, 6.0)),
            vel: DVec3::new(0.1, 0.2, 0.3),
            orient: DQuat::from_xyzw(0.0, 1.0, 0.0, 0.0),
            universe_tick: UniverseTick(77),
        };
        let bytes = postcard::to_allocvec(&p).expect("encode");
        let back: StampedPose = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, p);
    }

    #[test]
    fn lattice_local_is_cell_zero_with_offset_passthrough() {
        let v = DVec3::new(1.5, -2.5, 3.5);
        let lp = LatticePos::local(v);
        assert_eq!(lp.offset(), v);
        assert_eq!(lp.cell, I64Vec3::ZERO);
    }

    #[test]
    fn lattice_at_carries_the_explicit_cell_and_offset() {
        // The cell-aware ctor: the single construction point for a NON-ZERO cell (the P4/P5 rebase
        // precondition). Both the cell and the offset read back EXACTLY.
        let cell = I64Vec3::new(5, -7, 11);
        let offset = DVec3::new(0.25, -0.5, 0.75);
        let lp = LatticePos::at(cell, offset);
        assert_eq!(lp.cell(), cell);
        assert_eq!(lp.offset(), offset);
    }

    #[test]
    fn lattice_cell_accessor_reads_zero_for_local_and_the_exact_cell_otherwise() {
        // `local(..)` pins the cell to ZERO (the P3 invariant) — the accessor must read it back.
        assert_eq!(
            LatticePos::local(DVec3::new(1.5, -2.5, 3.5)).cell(),
            I64Vec3::ZERO
        );
        // A directly-constructed non-zero cell reads back EXACTLY (the P4/P5 rebase precondition;
        // mirrors the non-zero-cell serde precedent below).
        let lp = LatticePos {
            cell: I64Vec3::new(5, -7, 11),
            offset: DVec3::new(0.25, -0.5, 0.75),
        };
        assert_eq!(lp.cell(), I64Vec3::new(5, -7, 11));
    }

    #[test]
    fn lattice_translated_moves_in_frame_and_keeps_the_offset_bounded() {
        // The one in-frame move: displace, then re-bucket. The cell carries the whole-quantum part
        // of the move (the deleted `map_offset` left the offset growing unboundedly and every
        // caller remembering a `.normalize()` — the cure is internal now, §A4.5).
        let lp = LatticePos {
            cell: I64Vec3::new(5, -7, 11),
            offset: DVec3::new(0.25e-3, 0.5e-3, 0.75e-3),
        };
        let moved = lp.translated(DVec3::new(2.0, 0.0, 0.0), Tier::Fine);
        // 2 m = 2048 fine cells, carried into the integer half; the residual stays sub-cell and the
        // VALUE is preserved to f64 (the add `0.25e-3 + 2.0` rounds in its last ulp, so the residual
        // is compared by value, not bit).
        assert_eq!(moved.cell(), I64Vec3::new(5 + 2048, -7, 11));
        assert!((moved.offset() - DVec3::new(0.25e-3, 0.5e-3, 0.75e-3)).length() < 1e-12);
        // Byte-identity with the integrator's former spelling (map_offset + normalize): the same
        // two operations in the same order.
        let former = LatticePos {
            cell: lp.cell(),
            offset: lp.offset() + DVec3::new(2.0, 0.0, 0.0),
        }
        .normalize(Tier::Fine);
        assert_eq!(moved, former);
    }

    #[test]
    fn stamped_pose_serde_carries_a_nonzero_cell_exactly() {
        // The load-bearing plant: the i64 cell anchor rides the FROZEN wire bit-exact, so the future
        // non-zero-cell realization (P10) is additive — no wire rewrite. Through P3 cell is always
        // ZERO; this proves a non-zero cell would round-trip if/when it lands.
        let p = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 3 },
            pos: LatticePos {
                cell: I64Vec3::new(5, -7, 11),
                offset: DVec3::new(0.25, -0.5, 0.75),
            },
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(9),
        };
        let bytes = postcard::to_allocvec(&p).expect("encode");
        let back: StampedPose = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, p);
        assert_eq!(back.pos.cell, I64Vec3::new(5, -7, 11));
        assert_eq!(back.pos.offset(), DVec3::new(0.25, -0.5, 0.75));
    }

    // ---- S0 floating-origin lattice math (project_floating_origin_plan.md) ----

    #[test]
    fn tier_is_fine_for_a_system_and_every_nested_frame_coarse_only_for_galaxy() {
        // A pure coordinate-UNIT property, not a feature fork: everything inside a star system shares
        // the FINE (millimetre) tier; only GalaxySpace is COARSE (light-year cells, P10).
        assert_eq!(FrameRef::SystemSpace { system_seed: 9 }.tier(), Tier::Fine);
        assert_eq!(
            FrameRef::PlanetCentered { planet_seed: 5 }.tier(),
            Tier::Fine
        );
        assert_eq!(FrameRef::ShipLocal { ship: ship_id() }.tier(), Tier::Fine);
        assert_eq!(
            FrameRef::StationLocal { station_seed: 7 }.tier(),
            Tier::Fine
        );
        assert_eq!(
            FrameRef::AreaLocal {
                planet_seed: 5,
                area_seed: 8
            }
            .tier(),
            Tier::Fine
        );
        assert_eq!(
            FrameRef::GalaxySpace { galaxy_seed: 0 }.tier(),
            Tier::Galaxy
        );
    }

    #[test]
    fn every_step_is_the_power_of_two_its_exponent_names() {
        // The exponent is the datum and the step is DERIVED from it, so this asserts the derivation
        // itself rather than a table of decimals — a decimal table is what lets one entry drift.
        for t in Tier::ALL {
            assert_eq!(
                t.cell_edge_m(),
                (t.step_exponent() as f64).exp2(),
                "{t:?}'s step must be exactly two to its own exponent"
            );
        }
        // FINE = 2⁻¹⁰ m (0.9765625 mm) — the largest power-of-two metre quantum ≤ 1 mm, chosen so
        // `normalize` is exactly idempotent. UNCHANGED by the ladder, and everything at or below a star
        // system rests on that.
        assert_eq!(Tier::Fine.cell_edge_m(), 0.0009765625);
    }

    #[test]
    fn normalize_rebuckets_a_large_offset_into_the_cell_exactly() {
        // An un-normalized fine pose (cell 0, offset carrying full metres — the P3 shipping form) folds
        // its integer millimetres into the cell, leaving a sub-millimetre residual, preserving the total.
        let edge = FINE_CELL_EDGE_M; // 2⁻¹⁰ m ≈ 0.977 mm
        let lp = LatticePos::local(DVec3::new(1.2345, -0.0007, 2.0));
        let n = lp.normalize(Tier::Fine);
        // cell = floor(offset / edge) = floor(offset × 1024) per axis.
        assert_eq!(n.cell(), I64Vec3::new(1264, -1, 2048));
        // residual reconstructs the original total to f64 precision.
        let recon = n.cell().as_dvec3() * edge + n.offset();
        assert!((recon - lp.offset()).length() < 1e-9);
    }

    #[test]
    fn normalize_of_an_already_normalized_pose_is_a_fixed_point() {
        // Offset already within one cell (sub-mm) ⇒ zero carry ⇒ unchanged, cell and offset exact.
        let lp = LatticePos::at(I64Vec3::new(5, -7, 11), DVec3::new(0.0004, 0.0009, 0.0));
        let n = lp.normalize(Tier::Fine);
        assert_eq!(n.cell(), lp.cell());
        assert_eq!(n.offset(), lp.offset());
    }

    #[test]
    fn normalize_is_idempotent_at_the_power_of_two_edge() {
        // normalize(normalize(x)) == normalize(x), EXACT at the 2⁻¹⁰ edge (a bit-shift-clean division). At
        // the old 1e-3 edge this FAILS for 17.9 (a negative residual that re-carries) — the visual fixture's
        // orbital radius. Idempotence is the basis of the cell-activation migration being replay-safe.
        for &v in &[17.9_f64, -3.3, 1234.5, 0.0009765625, -0.0009765625, 1.0e9] {
            let n = LatticePos::local(DVec3::splat(v)).normalize(Tier::Fine);
            assert_eq!(
                n.normalize(Tier::Fine),
                n,
                "normalize not idempotent at {v}"
            );
        }
    }

    #[test]
    fn normalize_saturates_instead_of_panicking() {
        // A finite-but-astronomical offset makes the cell carry exceed i64 — normalize must SATURATE, never
        // panic (a panic here would take down the whole debug test + coverage suite). Both signs.
        let hi =
            LatticePos::at(I64Vec3::splat(i64::MAX), DVec3::splat(f64::MAX)).normalize(Tier::Fine);
        assert_eq!(hi.cell(), I64Vec3::splat(i64::MAX));
        let lo =
            LatticePos::at(I64Vec3::splat(i64::MIN), DVec3::splat(f64::MIN)).normalize(Tier::Fine);
        assert_eq!(lo.cell(), I64Vec3::splat(i64::MIN));
    }

    #[test]
    fn sanitized_clamps_a_hostile_cell_and_passes_an_in_domain_one_through() {
        // A hostile wire cell is CLAMPED to the bounded domain (so a downstream cell difference cannot
        // overflow); an in-domain cell passes through BIT-FOR-BIT (the determinism contract, byte-identity).
        let frame = FrameRef::SystemSpace { system_seed: 7 };
        let hostile = StampedPose {
            frame,
            pos: LatticePos::at(I64Vec3::splat(i64::MAX), DVec3::new(0.5, 0.5, 0.5)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        };
        assert_eq!(
            hostile.sanitized().pos.cell(),
            I64Vec3::splat(CELL_DOMAIN_MAX)
        );
        let ok = StampedPose {
            frame,
            pos: LatticePos::at(I64Vec3::new(1000, -2000, 3000), DVec3::new(0.5, 0.5, 0.5)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        };
        assert_eq!(ok.sanitized().pos.cell(), I64Vec3::new(1000, -2000, 3000));
    }

    #[test]
    fn delta_m_at_cell_zero_is_plain_subtraction() {
        // The byte-floor for every distance the server + client compute: at cell 0, delta_m is exactly the
        // offset difference (a plain vector subtract) — identical to the pre-lattice algebra.
        let a = LatticePos::local(DVec3::new(3.0, -4.0, 5.0));
        let b = LatticePos::local(DVec3::new(1.0, 1.0, 1.0));
        assert_eq!(a.delta_m(b, Tier::Fine), DVec3::new(2.0, -5.0, 4.0));
    }

    #[test]
    fn delta_m_is_exact_for_a_cell_difference_within_the_bound() {
        // A cell difference well inside the 2⁵³ exact bound: delta_m = integer-cell-diff × edge + offset
        // diff, bit-exact against the manual computation. (Beyond 2⁵³ the `.as_dvec3()` of the diff loses
        // low bits — the honestly-stated ceiling, never approached under the pin discipline.)
        let a = LatticePos::at(
            I64Vec3::new(1_000_000, 2_000_000, -3_000_000),
            DVec3::new(0.5, 0.0, 0.25),
        );
        let b = LatticePos::at(I64Vec3::new(4, 5, 6), DVec3::new(0.1, 0.0, 0.0));
        let e = FINE_CELL_EDGE_M;
        let want = DVec3::new(
            (1_000_000 - 4) as f64 * e + (0.5 - 0.1),
            (2_000_000 - 5) as f64 * e,
            (-3_000_000 - 6) as f64 * e + (0.25 - 0.0),
        );
        assert_eq!(a.delta_m(b, Tier::Fine), want);
    }

    #[test]
    fn delta_m_resolves_half_a_millimetre_at_interstellar_range() {
        // THE precision win: two positions ~4.9×10¹² m out, 0.5 mm apart (the offset carries it, the integer
        // cell is shared). delta_m keeps the 0.5 mm exactly; as plain f64 metres at that magnitude one ULP
        // is ~1 mm, so the naive difference quantizes the 0.5 mm away. Same test asserts BOTH.
        let cell = I64Vec3::new(5_000_000_000_000_000, 0, 0); // ~4.9e12 m at 2⁻¹⁰ edge, < 2⁵³ so f64-exact
        let a = LatticePos::at(cell, DVec3::new(0.0005, 0.0, 0.0));
        let b = LatticePos::at(cell, DVec3::ZERO);
        assert!((a.delta_m(b, Tier::Fine).x - 0.0005).abs() < 1e-12);
        let e = FINE_CELL_EDGE_M;
        let naive = (cell.x as f64 * e + 0.0005) - cell.x as f64 * e;
        assert!(
            (naive - 0.0005).abs() > 1.0e-4,
            "naive f64 metres lose the 0.5 mm at this magnitude (got {naive})"
        );
    }

    /// ★ GATE (a): DOWNWARD IS EXACT, AND OUT OF REACH IT REFUSES.
    ///
    /// Coarser → finer multiplies the whole-number half by the ratio. Inside the reach that is exact —
    /// no float touches the integer half — and a round trip back is the identity on BOTH halves.
    #[test]
    fn refining_within_reach_is_exact_and_a_round_trip_is_the_identity() {
        let cases = [
            (Tier::Galaxy, Tier::Fine),
            (Tier::Universe, Tier::Galaxy),
            (Tier::Universe, Tier::Fine),
        ];
        let mut checked = 0u32;
        for (from, to) in cases {
            let ratio = 1_i64 << (from.step_exponent() - to.step_exponent());
            let max_cells = (CELL_DOMAIN_MAX - (ratio - 1)) / ratio;
            for cell in [0, 1, -1, 7, -7, max_cells, -max_cells, max_cells / 3] {
                for off in [0.0, from.cell_edge_m() / 2.0] {
                    let src = LatticePos::at(I64Vec3::splat(cell), DVec3::splat(off));
                    let fine = src.convert_tier(from, to).expect("within reach");
                    // EXACT in the integer half: the whole number is scaled, not folded through metres.
                    assert_eq!(
                        fine.cell().x,
                        cell * ratio + (off / to.cell_edge_m()).floor() as i64,
                        "{from:?} -> {to:?} at cell {cell}"
                    );
                    // …and back again is the identity on BOTH halves.
                    let back = fine.convert_tier(to, from).expect("round trip");
                    assert_eq!(back.cell(), src.cell(), "{from:?} -> {to:?} -> {from:?}");
                    assert_eq!(
                        back.offset(),
                        src.offset(),
                        "{from:?} -> {to:?} -> {from:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 3 * 8 * 2, "every case must have run");
    }

    #[test]
    fn refining_past_the_reach_refuses_and_names_both_units() {
        // ★ THE REFUSAL IS THE NORMAL CASE, NOT A CORNER. Scaling an absolute galaxy cell down to
        // millimetres reaches 2⁷³ against an i64::MAX of 2⁶³ — over by a factor of 1024 — so only the
        // innermost 0.098% of the domain can be refined at all.
        let ratio = 1_i64 << (Tier::Galaxy.step_exponent() - Tier::Fine.step_exponent());
        let max_cells = (CELL_DOMAIN_MAX - (ratio - 1)) / ratio;
        // One cell inside the bound: admitted. One cell outside: refused. The bound is exact.
        let inside = LatticePos::at(I64Vec3::new(max_cells, 0, 0), DVec3::ZERO);
        assert!(inside.convert_tier(Tier::Galaxy, Tier::Fine).is_ok());
        let outside = LatticePos::at(I64Vec3::new(max_cells + 1, 0, 0), DVec3::ZERO);
        assert_eq!(
            outside.convert_tier(Tier::Galaxy, Tier::Fine),
            Err(TierConversionError::BeyondReach {
                from: Tier::Galaxy,
                to: Tier::Fine,
                widest_cells: (max_cells + 1).unsigned_abs(),
                max_cells: max_cells.unsigned_abs(),
            })
        );
        // NEGATIVE MAGNITUDES TOO — the bound is about distance from the origin, not about sign.
        let below = LatticePos::at(I64Vec3::new(-(max_cells + 1), 0, 0), DVec3::ZERO);
        assert!(below.convert_tier(Tier::Galaxy, Tier::Fine).is_err());
        // …and the reach really is a thousandth of the domain, stated as a measurement.
        let share = max_cells as f64 / CELL_DOMAIN_MAX as f64;
        assert!(share < 0.001, "the refinable share is {share}");
    }

    /// ★ GATE (b), CORRECTED. The plan and the design document both say upward is exact in the
    /// whole-number half. IT IS NOT, and this is the counterexample.
    ///
    /// `2047 × 2⁻¹⁰` plus the largest sub-cell offset has an exact value a hair BELOW two metres, and
    /// rounds to EXACTLY two metres — one whole galaxy step. The mandatory re-bucketing then carries, and
    /// the whole-number half moves by one. The resulting pair is the correctly-rounded representation of
    /// the same point, so what may be asserted is the POSITION and never the cell.
    ///
    /// The density is the lesson: 1,600 hits in a 3,200-case directed sweep, 0 in 200,000 random draws. A
    /// property test over random inputs passes while the gate lies.
    #[test]
    fn coarsening_carries_where_the_residual_rounds_up_to_a_whole_step() {
        let ratio = 1_i64 << (Tier::Galaxy.step_exponent() - Tier::Fine.step_exponent());
        let just_under = f64::from_bits(Tier::Fine.cell_edge_m().to_bits() - 1);
        let cell = 12_345 * ratio + (ratio - 1);
        let src = LatticePos::at(I64Vec3::new(cell, 0, 0), DVec3::new(just_under, 0.0, 0.0));
        let up = src
            .convert_tier(Tier::Fine, Tier::Galaxy)
            .expect("in domain");
        // THE CARRY: the naive whole-number answer is 12_345; the correct one is 12_346 with a zero
        // remainder, because the residual rounded up to a whole step.
        assert_eq!(cell.div_euclid(ratio), 12_345);
        assert_eq!(
            up.cell().x,
            12_346,
            "the residual rounded to a whole step and carried"
        );
        assert_eq!(up.offset().x, 0.0);
        // AND THE POSITION IS RIGHT, which is the thing that may be asserted. Reconstruct both and
        // compare within the derived per-rung bound.
        let exact = cell as f64 * Tier::Fine.cell_edge_m() + just_under;
        let got = up.cell().x as f64 * Tier::Galaxy.cell_edge_m() + up.offset().x;
        let bound = Tier::Galaxy.cell_edge_m() * f64::EPSILON / 4.0;
        // The difference is computed ONCE and asserted on. A lazily-evaluated format argument is a
        // region that only runs when the assertion fails, i.e. never on a green run.
        let off_by = (got - exact).abs();
        assert!(off_by <= bound);
    }

    #[test]
    fn the_upward_residual_bound_is_per_rung_and_is_attained() {
        // ★ THREE SOURCES DISAGREED AND ALL THREE WERE WRONG. The design document says 2⁻⁶³ (it applied
        // the FINE rung's granularity to a coarse rung's residual); the plan says 2⁻⁵¹ for everything,
        // which is unsatisfiable at the universe rung by a factor of 4096. The bound is PER RUNG:
        // half an ulp of the destination step, `step · ε / 4`.
        for (from, to) in [
            (Tier::Fine, Tier::Galaxy),
            (Tier::Galaxy, Tier::Universe),
            (Tier::Fine, Tier::Universe),
        ] {
            let bound = to.cell_edge_m() * f64::EPSILON / 4.0;
            let ratio = 1_i64 << (to.step_exponent() - from.step_exponent());
            let just_under = f64::from_bits(from.cell_edge_m().to_bits() - 1);
            let mut worst = 0.0_f64;
            for r in [0, 1, ratio / 2, ratio - 2, ratio - 1] {
                // THE OFFSETS MUST HAVE FULL MANTISSAS OR THE BOUND IS NEVER APPROACHED. A "round"
                // offset like half a step has almost no bits below the destination's own resolution, so
                // almost nothing is discarded and the error is tiny. The half-ulp maximum is reached
                // when exactly half a destination ulp is thrown away — which `bound` itself names — and
                // approached by any offset carrying bits all the way down.
                for off in [
                    0.0,
                    from.cell_edge_m() / 2.0,
                    just_under,
                    bound,
                    from.cell_edge_m() / 2.0 + bound,
                    from.cell_edge_m() * std::f64::consts::FRAC_1_SQRT_2,
                ] {
                    // ★ THE ERROR IS MEASURED EXACTLY, NOT BY COMPARING TWO SUMS. My first version of
                    // this rebuilt the "exact" value in f64 too, so both sides rounded the same way and
                    // it read ZERO everywhere — and a bound nothing reaches passes for any
                    // implementation, however loose. This uses the exact-error identity instead: for
                    // `|a| >= |b|`, `a + b` is exactly `fl(a+b) + (b - (fl(a+b) - a))`, and every step
                    // of that expression is itself exact. It measures the ONE rounding the conversion
                    // performs, which is the residual add.
                    let a = r as f64 * from.cell_edge_m(); // exact: a small integer times a power of two
                    let (hi, lo) = if a.abs() >= off.abs() {
                        (a, off)
                    } else {
                        (off, a)
                    };
                    let s = hi + lo;
                    let err = lo - (s - hi);
                    worst = worst.max(err.abs());
                    // …and the conversion really does answer within that error, so the identity above is
                    // measuring the thing the conversion actually does.
                    let cell = 7 * ratio + r;
                    let src = LatticePos::at(I64Vec3::new(cell, 0, 0), DVec3::new(off, 0.0, 0.0));
                    let up = src.convert_tier(from, to).expect("in domain");
                    // SPLIT, never `&&`: a short-circuit leaves the right-hand side uncoverable from a
                    // false left — this crate's own written rule, which the first draft of this test
                    // broke and the coverage gate caught.
                    assert!(up.offset().x >= 0.0);
                    assert!(up.offset().x < to.cell_edge_m());
                }
            }
            assert!(
                worst <= bound,
                "{from:?} -> {to:?}: worst {worst} exceeds {bound}"
            );
            // …AND THE BOUND IS ATTAINED, so it is tight rather than padded. A bound nothing reaches
            // would pass for any implementation, however loose.
            assert!(
                worst > bound / 4.0,
                "{from:?} -> {to:?}: worst {worst} never approaches {bound}"
            );
        }
    }

    #[test]
    fn a_refusal_names_the_axis_that_is_furthest_out_whichever_one_it_is() {
        // The refusal reports the WIDEST offending axis, so a diagnostic names the number that broke the
        // rule rather than whichever axis happens to be first. Driven on each axis in turn, because a
        // "pick the larger" fold that always returned its left argument would satisfy a single-axis test.
        for axis in 0..3 {
            let mut cell = I64Vec3::ZERO;
            cell[axis] = i64::MAX; // out of the sanitized domain on this axis only
            let src = LatticePos::at(cell, DVec3::ZERO);
            // EQUALITY ON THE WHOLE REFUSAL, not a match with a fallback arm. A `match` whose other
            // arm panics is a region no green run can reach — this crate's rule prefers an equality for
            // exactly that reason, and the coverage gate caught the first draft breaking it.
            //
            // The value reported is the one the position ACTUALLY holds: re-bucketing does not clamp a
            // cell — only the wire-ingress sanitizer does — so the refusal names the offending number
            // rather than a tidied version of it. The offset is lawful here, so its reported axis is the
            // last one inspected, which is zero.
            assert_eq!(
                src.convert_tier(Tier::Galaxy, Tier::Fine),
                Err(TierConversionError::NotAdmissible {
                    at: Tier::Galaxy,
                    cell_axis: i64::MAX,
                    offset_bits: 0.0_f64.to_bits(),
                }),
                "axis {axis} must be the one reported"
            );
        }
    }

    #[test]
    fn a_position_that_is_not_lawful_at_its_own_rung_is_refused() {
        // Every arm of the admissibility test, each driven by the thing it exists for.
        let bad_offset = LatticePos::at(I64Vec3::ZERO, DVec3::new(f64::NAN, 0.0, 0.0));
        assert!(bad_offset.convert_tier(Tier::Galaxy, Tier::Fine).is_err());
        // A cell outside the sanitized domain: re-bucketing saturates it and it is refused here.
        let far = LatticePos::at(I64Vec3::new(i64::MAX, 0, 0), DVec3::ZERO);
        assert!(far.convert_tier(Tier::Galaxy, Tier::Fine).is_err());
        let far_neg = LatticePos::at(I64Vec3::new(i64::MIN, 0, 0), DVec3::ZERO);
        assert!(far_neg.convert_tier(Tier::Galaxy, Tier::Fine).is_err());
        // …and a lawful one is not refused, so the arms above are about the position and not about the
        // conversion refusing everything.
        let ok = LatticePos::at(I64Vec3::new(3, -4, 5), DVec3::splat(0.5));
        assert!(ok.convert_tier(Tier::Galaxy, Tier::Fine).is_ok());
    }

    #[test]
    fn every_ratio_in_the_ladder_is_a_bit_shift() {
        // ★ REPLACES `fine_cells_per_ly_is_the_exact_integer_ratio` (slice S8). That test pinned a ratio
        // that exceeded a machine word and therefore needed a double-width integer — the whole reason
        // `FINE_CELLS_PER_LY: i128` existed. With every step a power of two, a ratio is a DIFFERENCE OF
        // EXPONENTS and the double-width integer is gone.
        for from in Tier::ALL {
            for to in Tier::ALL {
                let shift = to.step_exponent() - from.step_exponent();
                // The ratio is exactly 2^shift, and it is exact as an f64 in BOTH directions.
                let ratio = from.cell_edge_m() / to.cell_edge_m();
                assert_eq!(ratio, (-shift as f64).exp2(), "{from:?} -> {to:?}");
                // …and every step is a power of two, which is what makes re-bucketing exactly idempotent
                // at every rung. A non-power-of-two step has a non-zero mantissa.
                assert_eq!(
                    from.cell_edge_m().to_bits() & ((1 << 52) - 1),
                    0,
                    "{from:?}"
                );
            }
        }
        // The three ratios the ladder actually uses, by name, so a re-valued step is loud.
        assert_eq!(
            Tier::Galaxy.step_exponent() - Tier::Fine.step_exponent(),
            11
        ); // 2048
        assert_eq!(
            Tier::Universe.step_exponent() - Tier::Galaxy.step_exponent(),
            14
        ); // 16384
        assert_eq!(
            Tier::Universe.step_exponent() - Tier::Fine.step_exponent(),
            25
        );
    }

    #[test]
    fn the_steps_are_the_ruled_ones() {
        // The owner's Q1 ruling, as numbers: millimetres below a star system, TWO metres for a galaxy
        // (explicitly not the design document's one), 32,768 m for the universe.
        assert_eq!(Tier::Fine.cell_edge_m(), 1.0 / 1024.0);
        assert_eq!(Tier::Galaxy.cell_edge_m(), 2.0);
        assert_eq!(Tier::Universe.cell_edge_m(), 32_768.0);
        // …and the fine step is UNCHANGED, which is what everything at or below a star system rests on.
        assert_eq!(Tier::Fine.cell_edge_m(), FINE_CELL_EDGE_M);
    }

    /// ★ THE CLIMB COSTS THE SHIPPED PATH NOTHING, MEASURED (slice S9).
    ///
    /// `transfer_frame` now converts a distance into the book's rung on the way in and out of the
    /// destination's rung on the way out. Every crossing the world performs today has all three rungs
    /// equal, so both conversions must be the EXACT identity — not "close", not "within an ulp", but
    /// the same bits, including a residual that is negative, non-normalized, or larger than a cell
    /// (a difference has no sub-cell invariant, and a conversion that quietly tidied one would move
    /// the answer).
    ///
    /// Asserted on the primitive rather than argued from reading it, because "this is the identity"
    /// is exactly the sort of claim that is true when written and false two changes later.
    #[test]
    fn a_same_rung_conversion_is_the_identity_bit_for_bit() {
        let awkward = [
            DVec3::ZERO,
            DVec3::new(-0.5, 1.5, -2.5),
            DVec3::new(1.0e9, -1.0e9, 0.25),
            DVec3::splat(f64::MIN_POSITIVE),
        ];
        let cells = [
            I64Vec3::ZERO,
            I64Vec3::new(1, -1, 2),
            I64Vec3::new(CELL_DOMAIN_MAX, -CELL_DOMAIN_MAX, 0),
        ];
        let mut checked = 0usize;
        for tier in Tier::ALL {
            for c in cells {
                for r in awkward {
                    let s = Separation {
                        cells: c,
                        residual: r,
                        tier,
                    };
                    let same = s.convert_tier(tier).expect("same rung never refuses");
                    assert_eq!(same.cells(), c, "{tier:?}");
                    assert_eq!(same.residual().to_array(), r.to_array(), "{tier:?}");
                    assert_eq!(same.tier(), tier);
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 36, "every rung × cell × residual case ran");
    }

    /// ★ A CROSS-RUNG CONVERSION PRESERVES THE DISTANCE, both directions (slice S9).
    ///
    /// The metres are what a difference means; the cell count is bookkeeping that changes with the
    /// unit by design. Refining is asserted EXACT — it multiplies the whole-number half and leaves the
    /// residual alone, so no float touches it. Coarsening folds a remainder back into the residual and
    /// is exact to within half an ulp of the destination step, which is what is asserted rather than
    /// an equality it does not have.
    #[test]
    fn a_cross_rung_conversion_preserves_the_distance() {
        let cases = [
            (I64Vec3::new(2047, -1, 3), DVec3::new(0.000_5, -0.25, 0.125)),
            (I64Vec3::new(-2048, 4096, 0), DVec3::ZERO),
            (I64Vec3::ZERO, DVec3::new(1.5, -2.5, 0.25)),
        ];
        for (c, r) in cases {
            let fine = Separation {
                cells: c,
                residual: r,
                tier: Tier::Fine,
            };
            // FINE → GALAXY, then back. The round trip is EXACT: coarsening loses nothing (the
            // remainder is kept), and refining restores the count.
            let up = fine
                .convert_tier(Tier::Galaxy)
                .expect("coarsening is total");
            assert_eq!(up.tier(), Tier::Galaxy);
            let back = up
                .convert_tier(Tier::Fine)
                .expect("and back inside the bound");
            assert_eq!(back.metres().to_array(), fine.metres().to_array(), "{c:?}");
            // The distance survives the coarse statement itself, to within the coarse step's own
            // precision — the honest bound, not an equality.
            let slack = Tier::Galaxy.cell_edge_m() * f64::EPSILON;
            assert!((up.metres() - fine.metres()).abs().max_element() <= slack);
        }
    }

    #[test]
    fn every_rung_reaches_the_totality_list() {
        // ★ THE WITNESS FOR THE ONE FAILURE THIS LADDER CANNOT SEE. `Tier::ALL` has a single reader — the
        // saved-data label's coordinate generation — and its LENGTH is not compile-forced. A rung added to
        // the enum but forgotten here would leave that generation unmoved, so stores, clients and the
        // transport tag would all agree across a unit change: exactly the disagreement the generation
        // exists to make loud.
        //
        // Three assertions, each failing for a different reason: the count, the absence of duplicates, and
        // that every variant the program can name is present.
        assert_eq!(Tier::ALL.len(), 3);
        let mut seen: Vec<i32> = Tier::ALL.iter().map(|t| t.step_exponent()).collect();
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), before, "two rungs share a step");
        // ASCENDING, which the fold's order-sensitivity makes part of the meaning.
        assert!(
            seen.windows(2).all(|w| w[0] < w[1]),
            "ALL must ascend by step"
        );
        for t in [Tier::Fine, Tier::Galaxy, Tier::Universe] {
            assert!(Tier::ALL.contains(&t), "{t:?} is not in the totality list");
        }
    }

    #[test]
    fn convert_tier_same_rung_is_the_identity_the_only_live_path() {
        // Through S8 every live frame is FINE ⇒ convert_tier is always same-rung ⇒ a pure identity. It
        // must NOT re-bucket, because the shipped path's poses ride un-normalized at cell ZERO with the
        // whole frame-local distance in the offset — the byte floor is this return, not an argument.
        let lp = LatticePos::local(DVec3::new(12345.678, -9.0, 0.001));
        let same = lp
            .convert_tier(Tier::Fine, Tier::Fine)
            .expect("same rung never fails");
        assert_eq!(same.cell(), lp.cell());
        assert_eq!(same.offset(), lp.offset());
        // …and it is the identity for an input that is NOT admissible, which proves the same-rung arm is
        // tested before admissibility rather than after it.
        let wild = LatticePos::at(I64Vec3::splat(CELL_DOMAIN_MAX), DVec3::splat(1.0e9));
        assert_eq!(
            wild.convert_tier(Tier::Galaxy, Tier::Galaxy),
            Ok(wild),
            "same rung must not inspect the position at all"
        );
    }

    #[test]
    fn sub_millimetre_offsets_survive_at_a_billion_metre_cell_via_exact_integer_rebase() {
        // THE floating-point win the lattice buys: two positions a billion metres out, half a millimetre
        // apart. Expressed as integer-mm cells their SEPARATION is exact at ANY magnitude — rebasing one onto
        // the other cancels the (equal) integer cell part and leaves the 0.5 mm offset difference to full f64
        // precision, where pure-f64 metres would have lost sub-mm resolution long before this range.
        let cell = I64Vec3::new(1_000_000_000_000, 0, 0); // 1e12 mm = 1e9 m out
        let a = LatticePos::at(cell, DVec3::new(0.25e-3, 0.0, 0.0));
        let b = LatticePos::at(cell, DVec3::new(0.75e-3, 0.0, 0.0)); // 0.5 mm further along x
        let rel = b.separation(a, Tier::Fine);
        assert_eq!(rel.cells(), I64Vec3::ZERO); // the 1e9 m cell cancels EXACTLY — zero drift
        assert!((rel.residual().x - 0.5e-3).abs() < 1e-15); // the 0.5 mm survives to full f64 precision
    }

    #[test]
    fn an_occupant_riding_a_moving_realm_is_seen_moving_by_a_pinned_client() {
        // The floating-origin LAW at the lattice level, spelled in the ONE subtraction: a realm at
        // absolute A carries its occupant at A + local; when A moves between ticks the occupant's
        // absolute moves WITH it, and a client pinned elsewhere (separation from its pin) sees
        // exactly the realm's own motion. Non-zero integer cells throughout (the floating-point
        // regime the pre-lattice f64 algebra drifted in).
        let tier = Tier::Fine;
        let star_pin = LatticePos::ORIGIN;
        let local = DVec3::new(3.0, 0.0, 0.0);
        let a0 = LatticePos::at(
            I64Vec3::new(1_000_000_000, 0, 0),
            DVec3::new(0.4e-3, 0.0, 0.0),
        );
        let a1 = LatticePos::at(
            I64Vec3::new(1_000_002_000, 0, 0),
            DVec3::new(0.9e-3, 0.0, 0.0),
        );
        let seen0 = a0
            .translated(local, tier)
            .separation(star_pin, tier)
            .metres();
        let seen1 = a1
            .translated(local, tier)
            .separation(star_pin, tier)
            .metres();
        let occupant_moved = seen1 - seen0;
        let realm_moved = a1.separation(a0, tier).metres();
        assert!((occupant_moved - realm_moved).length() < 1e-6);
        assert!(occupant_moved.length() > 1.0);
    }

    #[test]
    fn at_rest_normalizes_the_spawn_pose_at_birth() {
        // PRODUCER #4 (real-scale addendum §A4.5/H-17): the spawn/home constructor fills the
        // integer half exactly as the integrator does on the first step. Value identical.
        let p = StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::new(20.0, -1.5, 0.25),
            UniverseTick(3),
        );
        assert_eq!(p.pos.cell(), I64Vec3::new(20 * 1024, -1536, 256));
        assert_eq!(p.pos.offset(), DVec3::ZERO);
        assert_eq!(
            p.pos.delta_m(LatticePos::ORIGIN, Tier::Fine),
            DVec3::new(20.0, -1.5, 0.25)
        );
    }

    #[test]
    fn from_metres_is_the_normalized_public_constructor() {
        let lp = LatticePos::from_metres(DVec3::new(2.5, 0.0, -1.0), Tier::Fine);
        assert_eq!(lp.cell(), I64Vec3::new(2560, 0, -1024));
        assert_eq!(lp.offset(), DVec3::ZERO);
        // Value-preserving for non-dyadic inputs too.
        let v = DVec3::new(0.3, -7.77, 123.456);
        let n = LatticePos::from_metres(v, Tier::Fine);
        assert!((n.delta_m(LatticePos::ORIGIN, Tier::Fine) - v).length() < 1e-12);
        assert!(n.offset().max_element() < FINE_CELL_EDGE_M + 1e-12);
    }

    #[test]
    fn separation_metres_is_exact_inside_two_pow_53_cells_and_relative_beyond() {
        // The §A5.2 flatten bound as a measurement: at 2^53 cells the flatten is still exact; at
        // 2^61 cells (the largest shell magnitude) the error is bounded by one ulp of the ANSWER
        // (≤ 0.5 m per axis at ~2.25e15 m), never of the world.
        let exact = LatticePos::at(I64Vec3::new(1 << 53, 0, 0), DVec3::ZERO)
            .separation(LatticePos::ORIGIN, Tier::Fine)
            .metres();
        assert_eq!(exact.x, (1u64 << 53) as f64 * FINE_CELL_EDGE_M);
        let big = LatticePos::at(I64Vec3::new((1 << 61) + 1, 0, 0), DVec3::ZERO)
            .separation(LatticePos::ORIGIN, Tier::Fine)
            .metres();
        let true_m = ((1i128 << 61) + 1) as f64 * FINE_CELL_EDGE_M;
        assert!((big.x - true_m).abs() <= 1.0);
    }

    #[test]
    fn separation_chebyshev_is_the_per_axis_max_and_total_at_the_domain_ends() {
        let a = LatticePos::at(I64Vec3::new(CELL_DOMAIN_MAX, 3, -9), DVec3::ZERO);
        let b = LatticePos::at(I64Vec3::new(-CELL_DOMAIN_MAX, 0, 0), DVec3::ZERO);
        let sep = a.separation(b, Tier::Fine);
        // 2·CELL_DOMAIN_MAX = i64::MAX − 1: the difference of two in-domain cells cannot overflow.
        assert_eq!(sep.cells_chebyshev(), i64::MAX - 1);
        // H-01 driven: the square of that separation would overflow i128 — refused, not panicked.
        assert_eq!(sep.cells_sq(), None);
        // In the guarded domain the square is the exact i128 sum.
        let small = LatticePos::at(I64Vec3::new(3, -4, 12), DVec3::ZERO)
            .separation(LatticePos::ORIGIN, Tier::Fine);
        assert_eq!(small.cells_chebyshev(), 12);
        assert_eq!(small.cells_sq(), Some(9 + 16 + 144));
    }

    #[test]
    fn separation_rotated_identity_is_a_bit_exact_passthrough() {
        let sep = LatticePos::at(I64Vec3::new(1 << 61, 5, -3), DVec3::new(0.1e-3, 0.0, 0.0))
            .separation(LatticePos::ORIGIN, Tier::Fine);
        // Identity ⇒ passthrough even far beyond the rotation reach: no fold, no cost.
        let out = sep.rotated(DQuat::IDENTITY).expect("identity passthrough");
        assert_eq!(out, sep);
    }

    #[test]
    fn separation_rotated_folds_exactly_in_reach_and_refuses_beyond_it() {
        use std::f64::consts::FRAC_PI_2;
        let q = DQuat::from_rotation_z(FRAC_PI_2);
        // In reach (1000 m ≪ 2⁴² m): fold-rotate-renormalize, cost ≤ one cell (here exact: the
        // rotation of (1000,0,0) about z is (0,1000,0), a dyadic value).
        let sep = LatticePos::from_metres(DVec3::new(1000.0, 0.0, 0.0), Tier::Fine)
            .separation(LatticePos::ORIGIN, Tier::Fine);
        let out = sep.rotated(q).expect("in reach");
        assert_eq!(out.cells(), I64Vec3::new(0, 1_024_000, 0));
        assert!((out.metres() - DVec3::new(0.0, 1000.0, 0.0)).length() < 1e-9);
        // Beyond the reach (2⁵³ cells = 2× the 2⁵² cell reach): refused LOUD — R2, the P10 trigger.
        let far = LatticePos::at(I64Vec3::new(1 << 53, 0, 0), DVec3::ZERO)
            .separation(LatticePos::ORIGIN, Tier::Fine);
        assert_eq!(far.rotated(q), Err(FrameError::RotationBeyondExactReach));
        // P6 non-vacuity, both sides of the fence: AT the reach the fold costs ≤ 1 cell; at 2× the
        // reach it is refused (the fence is not a no-op).
        let at_reach = LatticePos::at(I64Vec3::new(1 << 52, 0, 0), DVec3::ZERO)
            .separation(LatticePos::ORIGIN, Tier::Fine);
        let folded = at_reach
            .rotated(q)
            .expect("exactly at the reach is admitted");
        assert!(folded.cells().y.abs_diff(1 << 52) <= 1);
    }

    #[test]
    fn rotation_exact_reach_is_tier_derived() {
        // cell_edge / ε: FINE = 2⁻¹⁰/2⁻⁵² = 2⁴² m; every other rung re-derives itself from its own step,
        // which is the point — the reach is a function of the unit, not a table.
        assert_eq!(rotation_exact_reach_m(Tier::Fine), (1u64 << 42) as f64);
        for t in Tier::ALL {
            assert_eq!(
                rotation_exact_reach_m(t),
                t.cell_edge_m() / f64::EPSILON,
                "{t:?}"
            );
        }
    }

    #[test]
    fn separation_accessors_expose_the_halves_and_the_tier() {
        let sep = LatticePos::at(I64Vec3::new(7, 0, 0), DVec3::new(0.25e-3, 0.0, 0.0)).separation(
            LatticePos::at(I64Vec3::new(3, 0, 0), DVec3::ZERO),
            Tier::Fine,
        );
        assert_eq!(sep.cells(), I64Vec3::new(4, 0, 0));
        assert_eq!(sep.residual(), DVec3::new(0.25e-3, 0.0, 0.0));
        assert_eq!(sep.tier(), Tier::Fine);
        assert!(format!("{sep:?}").contains("Separation"));
    }

    use proptest::prelude::*;

    /// The §A6.3 boundary-WEIGHTED cell domain: zero, ±1, the exact-flatten edge (±2⁵²/±2⁵³), the
    /// largest shell magnitude (±2⁶¹), the sanitize clamp (±CELL_DOMAIN_MAX) — plus a uniform fill
    /// of the interior, so both the named cliffs and the bulk are driven.
    fn domain_cell() -> impl Strategy<Value = i64> {
        prop_oneof![
            Just(0i64),
            Just(1i64),
            Just(-1i64),
            Just(1i64 << 52),
            Just(-(1i64 << 52)),
            Just(1i64 << 53),
            Just(-(1i64 << 53)),
            Just(1i64 << 61),
            Just(-(1i64 << 61)),
            Just(CELL_DOMAIN_MAX),
            Just(-CELL_DOMAIN_MAX),
            -CELL_DOMAIN_MAX..=CELL_DOMAIN_MAX,
        ]
    }

    proptest! {
        #[test]
        fn normalize_preserves_total_position_and_bounds_the_offset(
            cx in -1_000_000i64..1_000_000, cy in -1_000_000i64..1_000_000, cz in -1_000_000i64..1_000_000,
            ox in -10.0f64..10.0, oy in -10.0f64..10.0, oz in -10.0f64..10.0,
        ) {
            let edge = FINE_CELL_EDGE_M;
            let lp = LatticePos::at(I64Vec3::new(cx, cy, cz), DVec3::new(ox, oy, oz));
            let n = lp.normalize(Tier::Fine);
            // Total position (cell*edge + offset) is preserved to f64 precision.
            let before = lp.cell().as_dvec3() * edge + lp.offset();
            let after = n.cell().as_dvec3() * edge + n.offset();
            prop_assert!((before - after).length() < 1e-6);
            // Residual offset sits in [0, edge) per axis (small float slack at the boundary).
            prop_assert!(n.offset().min_element() >= -1e-12);
            prop_assert!(n.offset().max_element() <= edge + 1e-12);
        }

        #[test]
        fn separation_is_the_exact_cell_delta_and_from_origin_inverts_it_bit_for_bit(
            ax in domain_cell(), ay in domain_cell(), az in domain_cell(),
            bx in domain_cell(), by in domain_cell(), bz in domain_cell(),
            oax in -0.9e-3f64..0.9e-3, obx in -0.9e-3f64..0.9e-3,
        ) {
            // P1 + P2 (addendum §A6.3), boundary-WEIGHTED over the sanitized domain: the cell half
            // of a separation is the exact i64 subtraction (it cannot overflow — CELL_DOMAIN_MAX is
            // i64::MAX/2 exactly so the difference fits), and re-adding the origin recovers the
            // position BIT-FOR-BIT (the §A5.1 exact-rebase theorem).
            let a = LatticePos::at(I64Vec3::new(ax, ay, az), DVec3::new(oax, 0.0, 0.0));
            let b = LatticePos::at(I64Vec3::new(bx, by, bz), DVec3::new(obx, 0.0, 0.0));
            let sep = a.separation(b, Tier::Fine);
            prop_assert_eq!(sep.cells().x, ax - bx);
            prop_assert_eq!(sep.cells().y, ay - by);
            prop_assert_eq!(sep.cells().z, az - bz);
            let back = sep.from_origin(b);
            // The integer half round-trips BIT-FOR-BIT at every magnitude; the f64 residual
            // round-trips within ulp(one cell edge) ≈ 1.1e-19 m (H-32: the ulp bound, not a loose
            // Sterbenz appeal, is what carries the theorem for opposite-sign residuals).
            prop_assert_eq!(back.cell(), a.cell());
            prop_assert!((back.offset() - a.offset()).length() <= 2.5e-19);
            // And where the residual arithmetic is exact (dyadic offsets — every normalized
            // production pose is within one cell of one), the round trip IS bit-for-bit.
            let ad = LatticePos::at(a.cell(), DVec3::new(0.25e-3, 0.0, 0.0));
            let bd = LatticePos::at(b.cell(), DVec3::new(0.5e-3, 0.0, 0.0));
            prop_assert_eq!(ad.separation(bd, Tier::Fine).from_origin(bd), ad);
        }

        #[test]
        fn cells_sq_never_panics_or_wraps_over_the_whole_sanitized_domain(
            ax in domain_cell(), bx in domain_cell(),
            ay in domain_cell(), by in domain_cell(),
            az in domain_cell(), bz in domain_cell(),
        ) {
            // P8 (addendum §A4.4 driven): the comparator over ANY sanitized pair — the opposite
            // ±CELL_DOMAIN_MAX ends included — neither panics nor wraps. Where it answers, the
            // answer equals the i128 reference computed axis-by-axis.
            let a = LatticePos::at(I64Vec3::new(ax, ay, az), DVec3::ZERO);
            let b = LatticePos::at(I64Vec3::new(bx, by, bz), DVec3::ZERO);
            let sep = a.separation(b, Tier::Fine);
            let reference: Option<i128> = if sep.cells_chebyshev() <= CELL_DOMAIN_MAX {
                let sq = |v: i64| i128::from(v) * i128::from(v);
                Some(sq(ax - bx) + sq(ay - by) + sq(az - bz))
            } else {
                None
            };
            prop_assert_eq!(sep.cells_sq(), reference);
        }
    }

    /// T2's STAR realm through every naming seam at once: how it prints, which realm its own
    /// frame names, and how a player is told where they stand. One test, three arms — each was
    /// added by the taxonomy arc and each is a place a missing arm would silently mis-name a
    /// star (the `Display` feeds directory keys and logs, `realm` feeds containment, `label`
    /// feeds the player's own location line).
    #[test]
    fn a_star_realm_names_itself_everywhere_a_realm_is_named() {
        let star = RealmId::Star(0x1234_5678_9abc_def0);
        assert_eq!(star.to_string(), "star-123456789abcdef0");
        let frame = FrameRef::StarCentered {
            star_seed: 0x1234_5678_9abc_def0,
        };
        assert_eq!(frame.realm(), star);
        assert_eq!(frame.label(), "Star 1311768467463790320");
    }
}
