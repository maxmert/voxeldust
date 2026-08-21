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
    /// Galaxy space: integer light-year cells + f64 offsets keep f64 precision
    /// (the cell layout lands with the galaxy work, P10).
    GalaxySpace,
    /// A station's interior grid frame (moves with the hull, like a ship). APPENDED
    /// (discriminant 4) so the wire stays additive.
    StationLocal { station_seed: u64 },
    /// A sub-planet AREA frame: a zone WITHIN a planet, carrying the parent planet's seed
    /// (the fixed parent, no lookup) plus the area's own seed. APPENDED (discriminant 5).
    AreaLocal { planet_seed: u64, area_seed: u64 },
    /// The near-star frame: star centre at origin (the taxonomy arc T2). APPENDED
    /// (discriminant 6) so the wire stays additive.
    StarCentered { star_seed: u64 },
}

impl FrameRef {
    /// The realm whose owner is authoritative for entities expressed in this frame,
    /// when one exists (GalaxySpace has no single realm owner).
    #[must_use]
    pub fn realm(self) -> Option<RealmId> {
        match self {
            FrameRef::PlanetCentered { planet_seed } => Some(RealmId::Planet(planet_seed)),
            FrameRef::ShipLocal { ship } => Some(RealmId::Ship(ship)),
            FrameRef::SystemSpace { system_seed } => Some(RealmId::System(system_seed)),
            FrameRef::GalaxySpace => None,
            FrameRef::StationLocal { station_seed } => Some(RealmId::Station(station_seed)),
            FrameRef::AreaLocal { area_seed, .. } => Some(RealmId::Area(area_seed)),
            FrameRef::StarCentered { star_seed } => Some(RealmId::Star(star_seed)),
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
            FrameRef::GalaxySpace => "Galaxy".to_owned(),
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
            FrameRef::GalaxySpace => Tier::Coarse,
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
        RealmId::Area(area_seed) => match parent {
            Some(RealmId::Planet(planet_seed)) => Some(FrameRef::AreaLocal {
                planet_seed,
                area_seed,
            }),
            _ => None,
        },
    }
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
/// form). It also makes FINE↔COARSE an EXACT integer ratio ([`FINE_CELLS_PER_LY`]), retiring the
/// remainder-carry hack. The edge is a compile-time constant, never serialized, so this moves zero bytes.
pub const FINE_CELL_EDGE_M: f64 = 1.0 / 1024.0;

/// FINE cells per COARSE cell (per light-year), EXACT. `COARSE_CELL_EDGE_M` is the exact-integer IAU
/// light-year in metres (`9_460_730_472_580_800`) and the FINE edge is `2⁻¹⁰`, so one light-year is exactly
/// `9_460_730_472_580_800 × 1024` fine quanta — an integer that exceeds `i64::MAX` (hence `i128`). This makes
/// FINE↔COARSE tier conversion exact integer arithmetic (no float remainder carry): `coarse_cell` × this +
/// fine residual is the exact fine position.
pub const FINE_CELLS_PER_LY: i128 = 9_460_730_472_580_800_i128 * 1024;

/// The bounded cell domain enforced at wire ingress ([`StampedPose::sanitized`]): each `LatticePos.cell`
/// axis is clamped to `±CELL_DOMAIN_MAX`. Set to `i64::MAX / 2` so that a cell DIFFERENCE (`a.cell −
/// b.cell` in [`LatticePos::delta_m`] and the crossing rebase) can never overflow `i64` — the operation a
/// diverged/hostile sender would otherwise use to panic the receiver. Well beyond any real position: at the
/// FINE 2⁻¹⁰ edge it spans ±0.476 light-year (half the tier's overflow limit), and every in-domain pose
/// passes through BIT-FOR-BIT (`sanitized().cell() == self.cell()`), preserving determinism.
pub const CELL_DOMAIN_MAX: i64 = i64::MAX / 2;

/// COARSE-tier cell edge: **one light-year** (IAU julian light-year, exact metres). At the coarse tier
/// a cell counts light-years and the f64 `offset` is the sub-light-year residual. Galaxy-scale positions
/// live here so f64 stays precise across interstellar distances (pure-f64 metres drift ~131 km/ULP at
/// galaxy scale — the class this cures). **Planted, value revisable at P10** — no COARSE-tier pose is
/// produced through P3 (only `SystemSpace`/FINE is live), and the exact-integer FINE↔COARSE remainder
/// carry (the mm↔ly ratio exceeds one i64) is finalized when the galaxy tier activates (user-deferred).
pub const COARSE_CELL_EDGE_M: f64 = 9_460_730_472_580_800.0;

/// Which coordinate TIER a [`LatticePos`] cell is measured in. The cell UNIT differs by tier so the
/// f64 in-cell `offset` keeps high precision at every scale — millimetres inside a star system
/// ([`Tier::Fine`]), light-years across a galaxy ([`Tier::Coarse`], P10). Selected by
/// [`FrameRef::tier`] from a frame's KIND — a coordinate unit, never a feature branch (HR3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Star system and inward: millimetre cells ([`FINE_CELL_EDGE_M`]).
    Fine,
    /// Galaxy and out: light-year cells ([`COARSE_CELL_EDGE_M`], P10).
    Coarse,
}

impl Tier {
    /// Metres per cell edge at this tier — the exact quantum a [`LatticePos`] cell counts.
    #[must_use]
    pub fn cell_edge_m(self) -> f64 {
        match self {
            Tier::Fine => FINE_CELL_EDGE_M,
            Tier::Coarse => COARSE_CELL_EDGE_M,
        }
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

    /// Re-express this position from one tier's cell UNIT into another (e.g. a FINE mm-lattice position
    /// re-quantized into COARSE ly-cells at a SOI/warp tier crossing). Same tier ⇒ identity — the ONLY
    /// live path through P3 (all frames FINE). Cross-tier folds to total metres, then re-buckets at the
    /// target edge. **Cross-tier is the P10-deferred plant** (user decision): it is precise only where the
    /// magnitude is f64-representable — the full FINE↔COARSE ratio (mm↔ly ≈ 9.5×10¹⁸ : 1) exceeds one
    /// i64, so the exact-integer remainder carry finalizes when the COARSE unit does at P10. No COARSE
    /// pose exists before then, so this dormant arm never runs in the shipped path.
    #[must_use]
    pub fn convert_tier(self, from: Tier, to: Tier) -> LatticePos {
        if from == to {
            return self;
        }
        let metres = self.offset + self.cell.as_dvec3() * from.cell_edge_m();
        LatticePos::local(metres).normalize(to)
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
    /// overflow pre-test. Total: `|Δ| ≤ 2·CELL_DOMAIN_MAX = i64::MAX − 1` for every sanitized pair,
    /// and `abs` of that range cannot overflow.
    #[must_use]
    pub fn cells_chebyshev(self) -> i64 {
        self.cells
            .x
            .abs()
            .max(self.cells.y.abs())
            .max(self.cells.z.abs())
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
    use super::*;
    use crate::entity_kind::EntityKind;

    fn ship_id() -> EntityId {
        EntityId::pack(EntityKind::Ship, 3, 17, 0xABCDEF)
    }

    #[test]
    fn realm_mapping_per_frame() {
        assert_eq!(
            FrameRef::PlanetCentered { planet_seed: 5 }.realm(),
            Some(RealmId::Planet(5))
        );
        assert_eq!(
            FrameRef::SystemSpace { system_seed: 9 }.realm(),
            Some(RealmId::System(9))
        );
        assert_eq!(
            FrameRef::ShipLocal { ship: ship_id() }.realm(),
            Some(RealmId::Ship(ship_id()))
        );
        assert_eq!(FrameRef::GalaxySpace.realm(), None);
        assert_eq!(
            FrameRef::StationLocal { station_seed: 7 }.realm(),
            Some(RealmId::Station(7))
        );
        assert_eq!(
            FrameRef::AreaLocal {
                planet_seed: 5,
                area_seed: 8
            }
            .realm(),
            Some(RealmId::Area(8))
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
        assert_eq!(FrameRef::GalaxySpace.label(), "Galaxy");
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
            frame: FrameRef::GalaxySpace,
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
        assert_eq!(FrameRef::GalaxySpace.tier(), Tier::Coarse);
    }

    #[test]
    fn cell_edge_is_the_power_of_two_quantum_at_fine_and_a_light_year_at_coarse() {
        assert_eq!(Tier::Fine.cell_edge_m(), FINE_CELL_EDGE_M);
        assert_eq!(Tier::Coarse.cell_edge_m(), COARSE_CELL_EDGE_M);
        // FINE = 2⁻¹⁰ m (0.9765625 mm) — the largest power-of-two metre quantum ≤ 1 mm, chosen so
        // `normalize` is exactly idempotent and FINE↔COARSE is an exact integer ratio.
        assert_eq!(Tier::Fine.cell_edge_m(), 1.0 / 1024.0);
        assert_eq!(Tier::Fine.cell_edge_m(), 0.0009765625);
        // one light-year, IAU julian: 9_460_730_472_580_800 m.
        assert_eq!(Tier::Coarse.cell_edge_m(), 9_460_730_472_580_800.0);
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

    #[test]
    fn fine_cells_per_ly_is_the_exact_integer_ratio() {
        // FINE↔COARSE is exact integer arithmetic: one light-year is exactly FINE_CELLS_PER_LY fine quanta,
        // an integer that exceeds i64 (hence i128) — retiring the float remainder-carry.
        assert_eq!(FINE_CELLS_PER_LY, 9_460_730_472_580_800_i128 * 1024);
        assert_eq!(FINE_CELLS_PER_LY / 1024, COARSE_CELL_EDGE_M as i128);
        assert!(FINE_CELLS_PER_LY > i64::MAX as i128);
    }

    #[test]
    fn convert_tier_same_tier_is_the_identity_the_only_live_path() {
        // Through P3 every frame is FINE ⇒ convert_tier is always same-tier ⇒ a pure identity (it does
        // NOT re-bucket the un-normalized shipping pose) — the byte-floor for the dormant arm.
        let lp = LatticePos::local(DVec3::new(12345.678, -9.0, 0.001));
        let same = lp.convert_tier(Tier::Fine, Tier::Fine);
        assert_eq!(same.cell(), lp.cell());
        assert_eq!(same.offset(), lp.offset());
    }

    #[test]
    fn convert_tier_cross_tier_reexpresses_the_total_metres_within_f64() {
        // The P10-deferred cross-tier plant: fold to total metres, re-bucket at the target edge. A small
        // FINE position (1000 m) lands in COARSE cell 0 with the metres carried in the offset (the
        // exact-integer carry for the full mm↔ly ratio finalizes at P10 with the COARSE unit).
        let fine = LatticePos::at(I64Vec3::new(1_024_000, 0, 0), DVec3::ZERO); // 1_024_000 × 2⁻¹⁰ m = 1000 m
        let coarse = fine.convert_tier(Tier::Fine, Tier::Coarse);
        assert_eq!(coarse.cell(), I64Vec3::ZERO); // 1000 m ≪ 1 ly
        assert!((coarse.offset().x - 1000.0).abs() < 1e-6);
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
        // cell_edge / ε: FINE = 2⁻¹⁰/2⁻⁵² = 2⁴² m; COARSE re-derives itself from its own edge.
        assert_eq!(rotation_exact_reach_m(Tier::Fine), (1u64 << 42) as f64);
        assert_eq!(
            rotation_exact_reach_m(Tier::Coarse),
            COARSE_CELL_EDGE_M / f64::EPSILON
        );
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
        assert_eq!(frame.realm(), Some(star));
        assert_eq!(frame.label(), "Star 1311768467463790320");
    }
}
