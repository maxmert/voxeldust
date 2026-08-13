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
}

impl core::fmt::Display for RealmId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RealmId::Planet(seed) => write!(f, "planet-{seed:016x}"),
            RealmId::System(seed) => write!(f, "system-{seed:016x}"),
            RealmId::Ship(id) => write!(f, "ship-{id}"),
            RealmId::Station(seed) => write!(f, "station-{seed:016x}"),
            RealmId::Area(seed) => write!(f, "area-{seed:016x}"),
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
            | FrameRef::AreaLocal { .. } => Tier::Fine,
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
/// **Planted now; math owed (D-41).** Through P3 every pose is `cell == ZERO` and `offset` carries
/// the full frame-local f64 position — BEHAVIOUR-IDENTICAL to the pre-lattice `pos: DVec3`. The wire
/// SHAPE is planted now (this rides the frozen [`StampedPose`]) so the future realization — non-zero
/// cells, the normalize/cell-crossing math, the per-`FrameRef`-tier unit, and the exact inter-tier
/// (mm↔AU) conversion at the SOI/warp transfer — is PURE-ADDITIVE (no wire change) when its first
/// consumer lands (P4/P5 re-centering; P10 galaxy). It is not built yet (smallest-correct: no
/// production caller exists until then). Fields are private so all construction flows through one
/// point ([`LatticePos::local`] today) and the bounded-offset invariant has a single future home.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LatticePos {
    cell: I64Vec3,
    offset: DVec3,
}

impl LatticePos {
    /// A position expressed purely as a frame-local offset, at the cell origin (`cell == ZERO`).
    /// THE constructor used everywhere through P3 — behaviour-identical to the old `pos: DVec3`.
    #[must_use]
    pub fn local(offset: DVec3) -> LatticePos {
        LatticePos {
            cell: I64Vec3::ZERO,
            offset,
        }
    }

    /// A position at an EXPLICIT integer cell anchor + a frame-local offset — the cell-aware sibling of
    /// [`LatticePos::local`] (which pins the cell to `ZERO`). This is the single construction point for a
    /// NON-ZERO cell: the P4/P5 cross-cell re-centering and the cell-aware membership rebase in the
    /// spatial transfer trigger produce lattice positions at a real cell here, rather than reaching into
    /// the private field. Through P3 there is no production caller (every pose is cell-`ZERO` via
    /// [`LatticePos::local`]); it exists so the cell anchor can be planted and its CARRY-through
    /// verified now (the D-41 SHAPE half — the in/out DECISION flip across a cell boundary is the P4/P5
    /// rebase MATH, still owed). The bounded-offset invariant has its future home in the same one place.
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

    /// The integer CELL anchor — the authoritative, bit-deterministic coarse coordinate. Through P3
    /// this is always `ZERO` ([`LatticePos::local`] is the only public constructor and pins it), but
    /// the read accessor exists now for the (P4/P5) cell-aware membership rebase in the spatial
    /// transfer trigger: cross-cell membership is exact integer cell arithmetic, so the trigger reads
    /// the cell here rather than reaching into the private field. Mirrors [`LatticePos::offset`] — reads
    /// go through one accessor so a future integer-offset migration stays one-type-local.
    #[must_use]
    pub fn cell(self) -> I64Vec3 {
        self.cell
    }

    /// Map the frame-local offset while PRESERVING the integer cell anchor — the cell-stable in-frame
    /// move (the per-tick integrator and any local displacement). [`LatticePos::local`] resets the cell
    /// to `ZERO`; this is the cell-preserving sibling, so a mutation site OUTSIDE vd-core keeps the
    /// anchor (closing the forced-cell-zeroing footgun). Through P3 the cell is `ZERO`, so this equals
    /// the old `pos = pos + delta`; once P4/P5 re-centering lands, re-bucketing a drifted offset back
    /// into the cell is a PURE ADDITION here (a `.normalize()` on the result) — the cell is already
    /// carried, so the re-centering is genuinely additive at this site, not a clobber-and-replace.
    #[must_use]
    pub fn map_offset(self, f: impl FnOnce(DVec3) -> DVec3) -> LatticePos {
        LatticePos {
            cell: self.cell,
            offset: f(self.offset),
        }
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

    /// Express this position RELATIVE TO `origin` (both in the same frame + tier) — exact integer cell
    /// subtraction plus the f64 offset residual, then re-normalized. This is the zero-drift rebase at the
    /// heart of the floating-origin model: the client subtracts its pinned star-system absolute here, and
    /// the integer-cell part cancels EXACTLY (no ~131 km/ULP galaxy-scale f64 loss). Through P3 every cell
    /// is `ZERO`, so this reduces to `offset - origin.offset` — behaviour-identical to a plain vector
    /// subtraction; the exactness matters once S5 lights up non-zero cells.
    #[must_use]
    pub fn rebase_to(self, origin: LatticePos, _tier: Tier) -> LatticePos {
        // PURE exact rebase — integer cell subtraction + f64 offset residual, NO trailing normalize.
        // Re-bucketing here is WRONG: `normalize` forces the residual into `[0, edge)`, but a rebased
        // position (a client subtracting its pinned origin) wants the SIGNED residual AROUND the origin,
        // and re-quantizing it (a) is a pure no-op on the value it would then re-add and (b) broke
        // idempotence at the old non-power-of-two edge. Exactness needs no normalize here: the integer
        // cell difference is exact and the offset is one f64 subtract. The `_tier` is retained to mark
        // this a same-tier op (callers pass it); re-quantize explicitly via `re_anchor` at an activation
        // seam. Inverse of `compose`: `a.compose(b).rebase_to(a) == b` bit-for-bit (no normalize rounding).
        LatticePos {
            cell: self.cell - origin.cell,
            offset: self.offset - origin.offset,
        }
    }

    /// Compose `self` (a frame ORIGIN's absolute position) with a `local` position expressed IN that
    /// frame → the local's absolute position, same tier. Exact cell addition + offset sum, re-normalized.
    /// This is the once-per-tick `self_abs ∘ occupant_local` the containing realm's shard runs to author
    /// every occupant's absolute pose (author-once-ship-all). Inverse of [`LatticePos::rebase_to`]:
    /// `a.compose(b).rebase_to(a) == b` (to f64). Through P3 (`cell == ZERO`) this is `self.offset +
    /// local.offset` — a plain vector add.
    #[must_use]
    pub fn compose(self, local: LatticePos, _tier: Tier) -> LatticePos {
        // PURE exact compose — integer cell add + f64 offset sum, NO trailing normalize. `normalize`
        // here would break byte-identity at the walk floor: `identity.compose({cell 0, offset 20 m})`
        // would carry 20 m INTO the cell (`{cell 20480, offset 0}`) — different postcard bytes on every
        // walk `EntitySnap` than the `{cell 0, offset 20 m}` the wire ships today. Exactness is preserved
        // without it: the cell add is integer-exact and the offset sum is one f64 add. The `_tier` marks
        // this a same-tier op; re-quantize explicitly via `re_anchor` only at a real activation seam.
        LatticePos {
            cell: self.cell + local.cell,
            offset: self.offset + local.offset,
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

    /// This position MINUS `origin` (both same frame + tier) expressed in **metres** — the render-plane
    /// rebase the client runs at the ONE `world_pos` chokepoint (subtract the pinned origin), and the
    /// server's distance measure (containment, ghost band, demand). The integer cell difference cancels
    /// the huge galaxy-scale magnitude EXACTLY; only the small residual difference touches f64.
    ///
    /// **The honest bound.** Exact iff `|self.cell − origin.cell| ≤ 2⁵³` per axis (≈ 8.8×10¹² m ≈ 59 AU at
    /// FINE) — beyond that the `.as_dvec3()` of the integer difference loses low bits and this degrades to a
    /// plain f64 of the difference. That bound is never approached under the pin discipline (the pin is the
    /// occupant's own star system, always within one FINE cell-span of it), so in-system rebases are exact;
    /// cross-cell distances stay well inside it. `cell − origin.cell` is exact integer subtraction for every
    /// in-domain cell (sanitized to a bounded domain at wire ingress, so no overflow reaches here).
    #[must_use]
    pub fn delta_m(self, origin: LatticePos, tier: Tier) -> DVec3 {
        (self.cell - origin.cell).as_dvec3() * tier.cell_edge_m() + (self.offset - origin.offset)
    }

    /// THE single quantization decision point — re-bucket the offset into the cell, or don't, per a
    /// statically-configured [`CellAnchor`]. `Inert` returns `self` BIT-FOR-BIT (the byte-floor: every
    /// existing forest rides here, cell unchanged, wire identical); `Cell(tier)` runs [`normalize`] to carry
    /// an accumulated offset into the cell. This is the ONLY place a pose is re-quantized — the integrator,
    /// frame entry, and the generator call it, all `Inert` through P3/P4 and flipped to `Cell` only in the
    /// gated galaxy-scale fixture (A8). Idempotent on an already-anchored pose (guaranteed by the 2⁻¹⁰ edge),
    /// so a re-anchor on load is a no-op on the anchored form — the migration is a no-op, not a flag day.
    #[must_use]
    pub fn re_anchor(self, anchor: CellAnchor) -> LatticePos {
        match anchor {
            CellAnchor::Inert => self,
            CellAnchor::Cell(tier) => self.normalize(tier),
        }
    }
}

/// The quantization mode for [`LatticePos::re_anchor`] — a coordinate-unit config, NOT a realm/shard-kind
/// match (HR3). `Inert` keeps a pose at its shipped cell (the byte-floor, every forest through P4);
/// `Cell(tier)` re-buckets accumulated offset into the integer cell at that tier (the gated galaxy-scale
/// activation, A8). Planted here; flipped only by the `VD_UNIVERSE_SCALE` fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellAnchor {
    /// No re-quantization — the pose passes through bit-for-bit (byte-floor, prod through P4).
    Inert,
    /// Re-bucket the offset into the cell at this tier (galaxy-scale activation).
    Cell(Tier),
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
    /// A rest pose at a position — the common spawn/test constructor.
    #[must_use]
    pub fn at_rest(frame: FrameRef, pos: DVec3, universe_tick: UniverseTick) -> StampedPose {
        StampedPose {
            frame,
            pos: LatticePos::local(pos),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick,
        }
    }

    /// Compose `self` (a frame ORIGIN's absolute pose) with `local` (a pose expressed IN that frame) →
    /// the local's absolute pose. This is the FULL rigid compose the server runs once per tick to author
    /// every occupant's absolute (`self_abs ∘ occupant_local`): the position rides through
    /// [`LatticePos::compose`] with the lever ROTATED by the origin's orientation, the velocity carries
    /// the rotated local velocity, and the orientation composes. A position-only compose would be a
    /// DEFECT — `place_child`'s mover arm authors a real orbital velocity, so shipping an absolute
    /// position with a realm-local velocity yields a mixed-frame pose that contradicts this type's own
    /// contract (the client's no-prediction firewall drops `vel` today, but every closed-form re-advance
    /// and future consumer relies on it). The **native frame label is preserved** (`local.frame`): only
    /// the VALUE becomes absolute, so the boundary detector, the saga, and the client re-pin trigger keep
    /// reading the label they expect. Stamped at the local's tick (same-tick compose — box and rider agree
    /// in time). The `cell` rides from `local.pos` (ZERO through P4); the rotation applies to the offset
    /// residual (occupants live at offset scale, so this is exact wherever the cell is ZERO).
    #[must_use]
    pub fn compose(self, local: StampedPose, tier: Tier) -> StampedPose {
        StampedPose {
            frame: local.frame,
            pos: self.pos.compose(
                LatticePos::at(local.pos.cell(), self.orient * local.pos.offset()),
                tier,
            ),
            vel: self.vel + self.orient * local.vel,
            orient: self.orient * local.orient,
            universe_tick: local.universe_tick,
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

/// ONE PLACEMENT AS A TRANSFORM — where a CHILD frame's origin sits, and how it moves, inside its
/// PARENT's frame. It is the same content as a [`crate::frame::FramePlacement`] with the two pieces
/// nothing composes (the integer anchor and the f64 remainder) already fused into a [`LatticePos`], so a
/// CHAIN of placements can be folded into one transform and then applied to many poses.
///
/// WHY THIS TYPE EXISTS, and it is the whole point of the render-composition work. A pose arrives measured
/// from the realm it lives in; the client draws everything measured from ONE pinned realm. Turning the
/// first into the second is a walk UP from the pose's realm to the nearest common ancestor and back DOWN
/// to the pin, and every step of that walk is one of these. Folding the walk ONCE per (frame, pin) and then
/// applying the fold per entity is what keeps the per-entity cost a single vector add instead of a chain
/// re-walk; the table it lives in is bounded by realms in view, never by entities.
///
/// It is a PLACEMENT, never an absolute. Nothing here can express "where this realm is in the universe" —
/// there is no universe frame in the type at all — which is the structural half of the ground rule: only a
/// parent knows where its children are, so a placement is the only shape a position relation can take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameXform {
    /// The child frame's origin, in the PARENT's frame (integer cell + f64 remainder).
    pub pos: LatticePos,
    /// The child frame's origin velocity, in the parent's frame (m/s).
    pub vel: DVec3,
    /// The rotation taking the CHILD's axes into the PARENT's axes.
    pub orient: DQuat,
}

impl FrameXform {
    /// The child frame IS the parent: no offset, no motion, no rotation. The value every walk-scale and
    /// static forest is built entirely out of, which is what makes [`FrameXform::is_identity`] a usable
    /// whole-table short-circuit.
    pub const IDENTITY: FrameXform = FrameXform {
        pos: LatticePos {
            cell: I64Vec3::ZERO,
            offset: DVec3::ZERO,
        },
        vel: DVec3::ZERO,
        orient: DQuat::IDENTITY,
    };

    /// Is this EXACTLY the identity? Bit equality on every component, never an epsilon, and that is
    /// deliberate: this predicate gates a whole-table "skip the composition entirely" short-circuit, so a
    /// tolerance here would mean a realm rotating slowly enough to fall inside it silently STOPS ROTATING
    /// for every client. A placement that is nearly the identity is not the identity.
    #[must_use]
    pub fn is_identity(self) -> bool {
        self == FrameXform::IDENTITY
    }

    /// This placement expressed one level FURTHER OUT: `self` places a child inside its parent, `outer`
    /// places that parent inside ITS parent, and the result places the child inside the grandparent. The
    /// fold step of the up-walk.
    ///
    /// Delegates to [`StampedPose::compose`] — there is exactly ONE composition implementation in the
    /// workspace and this is not a second one. The two feeds that draw a realm box and the thing standing
    /// in it parted company once before precisely because two sites did the same arithmetic separately.
    ///
    /// # Errors
    /// [`FrameError::RotatedFrameAcrossCells`] when `outer` is rotated and `self`'s origin sits a whole
    /// number of integer cells out — see `FrameXform::rotatable` (a private helper, so plain backticks).
    pub fn then(self, outer: FrameXform, tier: Tier) -> Result<FrameXform, FrameError> {
        FrameXform::rotatable(outer.orient, self.pos.cell())?;
        let composed = outer.as_origin_pose().compose(self.as_origin_pose(), tier);
        Ok(FrameXform {
            pos: composed.pos,
            vel: composed.vel,
            orient: composed.orient,
        })
    }

    /// The placement read the other way round: `self` maps CHILD→PARENT, this maps PARENT→CHILD. The
    /// down-walk half of a pin conversion (getting from the common ancestor down to the pinned realm).
    ///
    /// Exact for every unrotated placement: the integer cell is NEGATED as an integer, so the anchor
    /// survives the round trip bit-for-bit. `_tier` marks this a same-tier operation, matching
    /// [`LatticePos::compose`] and [`LatticePos::rebase_to`]; no re-quantization happens here.
    ///
    /// # Errors
    /// [`FrameError::RotatedFrameAcrossCells`] when this placement is rotated AND its origin sits a whole
    /// number of integer cells out — see `FrameXform::rotatable` (a private helper, so plain backticks).
    pub fn inverse(self, _tier: Tier) -> Result<FrameXform, FrameError> {
        FrameXform::rotatable(self.orient, self.pos.cell())?;
        let inv = self.orient.inverse();
        Ok(FrameXform {
            // The cell negates as an INTEGER (the orientation is the identity on this arm, guarded above),
            // so `x.inverse().then(x)` returns the anchor untouched rather than through f64 metres.
            pos: LatticePos::at(-self.pos.cell(), -(inv * self.pos.offset())),
            vel: -(inv * self.vel),
            orient: inv,
        })
    }

    /// Re-express `pose` — measured in the CHILD frame this placement describes — in the PARENT frame, and
    /// RELABEL it `label`.
    ///
    /// The relabel is mandatory, not cosmetic. After composition the numbers are measured from a different
    /// realm's centre; leaving the pose's native label on them produces a value wearing the wrong frame's
    /// name, which is exactly the lie this whole arc exists to remove — a shipped absolute that still
    /// claimed to be realm-local looked correct at walk scale, where every realm sits at the origin and the
    /// two agree, and drew the player at the star everywhere else.
    ///
    /// # Errors
    /// [`FrameError::RotatedFrameAcrossCells`] when this placement is rotated and the pose's own origin
    /// sits a whole number of integer cells out — see `FrameXform::rotatable` (private, so plain backticks).
    pub fn apply(
        self,
        pose: StampedPose,
        tier: Tier,
        label: FrameRef,
    ) -> Result<StampedPose, FrameError> {
        FrameXform::rotatable(self.orient, pose.pos.cell())?;
        let mut out = self.as_origin_pose().compose(pose, tier);
        out.frame = label;
        Ok(out)
    }

    /// May a position whose integer anchor is `cell` be rotated by `orient`?
    ///
    /// A cell COUNT is measured along one frame's axes. Rotating a non-zero count into another frame's axes
    /// does not yield a count — no rotation of an integer lattice vector is generally an integer lattice
    /// vector — so the only ways to proceed are to fold the anchor into f64 metres, which is precisely the
    /// precision loss this coordinate exists to prevent, or to refuse. It refuses. EXACT quaternion
    /// equality for the same reason [`FrameXform::is_identity`] uses it: "nearly unrotated" is rotated, and
    /// a tolerance would quietly resume the folding for every slowly-spinning realm.
    ///
    /// Unreachable in production today (every placement in the tree is identity-oriented and every anchor
    /// is `ZERO`); it becomes live the first time a spinning realm is authored more than one cell-block
    /// from its parent's origin. Mirrors the identical guard in [`crate::frame::transfer_frame`] — one
    /// rule, one error, two call sites.
    ///
    /// # Errors
    /// [`FrameError::RotatedFrameAcrossCells`] on the refused combination.
    fn rotatable(orient: DQuat, cell: I64Vec3) -> Result<(), FrameError> {
        if orient != DQuat::IDENTITY && cell != I64Vec3::ZERO {
            return Err(FrameError::RotatedFrameAcrossCells);
        }
        Ok(())
    }

    /// This placement as the ORIGIN pose [`StampedPose::compose`] expects on its left-hand side. The frame
    /// label and tick are never read by `compose` (it takes both from the local operand), so they are
    /// placeholders; keeping them here rather than at each call site is what lets `then`/`apply` share the
    /// ONE compose.
    fn as_origin_pose(self) -> StampedPose {
        StampedPose {
            frame: FrameRef::GalaxySpace,
            pos: self.pos,
            vel: self.vel,
            orient: self.orient,
            universe_tick: UniverseTick(0),
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
    fn lattice_map_offset_preserves_the_cell_while_mapping_the_offset() {
        // The cell-PRESERVING in-frame move (the integrator uses this). Through P3 the cell is ZERO
        // so movement tests cannot distinguish it from `local`; this guards the cell-preservation
        // BEHAVIOUR directly with a non-zero cell, so a future refactor cannot silently re-introduce
        // the cell-zeroing foot-gun (audit wf_fd6a4b9d).
        let lp = LatticePos {
            cell: I64Vec3::new(5, -7, 11),
            offset: DVec3::new(1.0, 2.0, 3.0),
        };
        let moved = lp.map_offset(|o| o + DVec3::new(0.25, -0.5, 0.75));
        assert_eq!(moved.cell, I64Vec3::new(5, -7, 11));
        assert_eq!(moved.offset(), DVec3::new(1.25, 1.5, 3.75));
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
    fn rebase_to_is_exact_integer_cell_subtraction() {
        // Expressing one fine pose relative to another cancels the integer cell part EXACTLY (the
        // zero-drift client origin-subtraction). Offsets ZERO so the equality is bit-exact.
        let a = LatticePos::at(I64Vec3::new(1000, 2000, -3000), DVec3::ZERO);
        let origin = LatticePos::at(I64Vec3::new(1, 2, 3), DVec3::ZERO);
        let r = a.rebase_to(origin, Tier::Fine);
        assert_eq!(r.cell(), I64Vec3::new(999, 1998, -3003));
        assert_eq!(r.offset(), DVec3::ZERO);
    }

    #[test]
    fn compose_adds_cells_exactly() {
        // origin ∘ local sums the integer cells — NO normalize (offsets ZERO ⇒ exact).
        let origin = LatticePos::at(I64Vec3::new(10, 20, 30), DVec3::ZERO);
        let local = LatticePos::at(I64Vec3::new(1, 2, 3), DVec3::ZERO);
        let c = origin.compose(local, Tier::Fine);
        assert_eq!(c.cell(), I64Vec3::new(11, 22, 33));
        assert_eq!(c.offset(), DVec3::ZERO);
    }

    #[test]
    fn compose_and_rebase_at_cell_zero_are_plain_vector_add_and_subtract() {
        // THE byte-floor: the P3 shipping form is cell 0 with the offset carrying full metres. As long
        // as the result stays within one cell (sub-mm test values ⇒ no carry), compose/rebase reduce to
        // a plain add/subtract on the offset — identical to the pre-lattice DVec3 algebra, cell stays 0.
        let origin = LatticePos::local(DVec3::new(0.0004, 0.0, 0.0));
        let local = LatticePos::local(DVec3::new(0.0003, 0.0, 0.0));
        let c = origin.compose(local, Tier::Fine);
        assert_eq!(c.cell(), I64Vec3::ZERO);
        assert!((c.offset() - DVec3::new(0.0007, 0.0, 0.0)).length() < 1e-12);
        let r = c.rebase_to(origin, Tier::Fine);
        assert_eq!(r.cell(), I64Vec3::ZERO);
        assert!((r.offset() - local.offset()).length() < 1e-12);
    }

    #[test]
    fn compose_at_identity_is_bit_exact() {
        // identity ∘ x == x, BIT-for-bit. Fails on the pre-A1 normalizing compose (which would carry x's
        // whole offset into the cell). The one-line proof that compose no longer re-quantizes.
        let x = LatticePos::at(I64Vec3::new(7, -3, 11), DVec3::new(1.5, -2.25, 0.125));
        let id = LatticePos::at(I64Vec3::ZERO, DVec3::ZERO);
        assert_eq!(id.compose(x, Tier::Fine), x);
        assert_eq!(x.compose(id, Tier::Fine), x);
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
    fn stamped_compose_rotates_the_lever_carries_velocity_and_preserves_the_native_frame() {
        use std::f64::consts::FRAC_PI_2;
        // origin: a frame at (10,0,0), rotated 90° about +Z, moving +Y at 2 m/s.
        let origin = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 7 },
            pos: LatticePos::local(DVec3::new(10.0, 0.0, 0.0)),
            vel: DVec3::new(0.0, 2.0, 0.0),
            orient: DQuat::from_rotation_z(FRAC_PI_2),
            universe_tick: UniverseTick(5),
        };
        // local: an occupant 1 m along +X in the origin's frame, at rest, in a PLANET frame, at tick 9.
        let local = StampedPose {
            frame: FrameRef::PlanetCentered { planet_seed: 3 },
            pos: LatticePos::local(DVec3::new(1.0, 0.0, 0.0)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(9),
        };
        let c = origin.compose(local, Tier::Fine);
        // lever (1,0,0) rotated 90° about Z → (0,1,0), added to the origin (10,0,0) → (10,1,0).
        assert!((c.pos.offset() - DVec3::new(10.0, 1.0, 0.0)).length() < 1e-9);
        // velocity carries the origin's (the local is at rest).
        assert!((c.vel - DVec3::new(0.0, 2.0, 0.0)).length() < 1e-9);
        // the NATIVE frame label is preserved (the VALUE is absolute, the label is the local's) and the
        // stamp is the local's tick (same-tick compose).
        assert_eq!(c.frame, FrameRef::PlanetCentered { planet_seed: 3 });
        assert_eq!(c.universe_tick, UniverseTick(9));
    }

    #[test]
    fn re_anchor_inert_passes_through_and_cell_re_buckets() {
        // The byte-floor gate: Inert returns the pose BIT-FOR-BIT (every forest through P4); Cell re-buckets
        // the offset into the cell (the gated galaxy-scale activation). Both CellAnchor arms + idempotence.
        let x = LatticePos::local(DVec3::new(1.5, -2.0, 3.0));
        assert_eq!(x.re_anchor(CellAnchor::Inert), x);
        assert_eq!(
            x.re_anchor(CellAnchor::Cell(Tier::Fine)),
            x.normalize(Tier::Fine)
        );
        let anchored = x.re_anchor(CellAnchor::Cell(Tier::Fine));
        assert_eq!(anchored.re_anchor(CellAnchor::Cell(Tier::Fine)), anchored);
    }

    #[test]
    fn cell_anchor_derives_are_exercised() {
        // Cover CellAnchor's derived Debug + PartialEq (it becomes a config value at A8; the derives must
        // not sit as uncovered regions in the meantime — HR5).
        assert_eq!(CellAnchor::Inert, CellAnchor::Inert);
        assert_ne!(CellAnchor::Inert, CellAnchor::Cell(Tier::Fine));
        assert_eq!(CellAnchor::Cell(Tier::Fine), CellAnchor::Cell(Tier::Fine));
        assert!(format!("{:?}", CellAnchor::Cell(Tier::Coarse)).contains("Coarse"));
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
        let rel = b.rebase_to(a, Tier::Fine);
        assert_eq!(rel.cell(), I64Vec3::ZERO); // the 1e9 m cell cancels EXACTLY — zero drift
        assert!((rel.offset().x - 0.5e-3).abs() < 1e-15); // the 0.5 mm survives to full f64 precision
    }

    #[test]
    fn an_occupant_composed_into_a_moving_realm_rides_it_and_a_pinned_client_sees_the_motion() {
        // The floating-origin LAW that S2 (server compose) + S3 (client subtract) will wire live, proven here
        // at the lattice level: a realm at absolute A authors its occupant's absolute = A ∘ local; when A
        // MOVES between ticks the occupant's absolute moves WITH it (rides the realm). A client pinned to a
        // DIFFERENT realm (the star at the universe origin) subtracts its pin, so it renders the occupant AT
        // the moving absolute — the occupant visibly travels with its realm instead of hanging static (the
        // reported bug). Exercised with non-zero integer cells (the floating-point regime).
        let tier = Tier::Fine;
        let star_pin = LatticePos::at(I64Vec3::ZERO, DVec3::ZERO); // client's pin: the star at the origin
        let local = LatticePos::local(DVec3::new(3.0, 0.0, 0.0)); // the occupant, 3 m from its realm centre
        // Tick 0: the realm (planet) sits at absolute A0, ~1e6 m out.
        let a0 = LatticePos::at(
            I64Vec3::new(1_000_000_000, 0, 0),
            DVec3::new(0.4e-3, 0.0, 0.0),
        );
        let seen0 = a0.compose(local, tier).rebase_to(star_pin, tier);
        // Tick 1: the realm has moved (+2 m along x, plus a sub-mm step).
        let a1 = LatticePos::at(
            I64Vec3::new(1_000_002_000, 0, 0),
            DVec3::new(0.9e-3, 0.0, 0.0),
        );
        let seen1 = a1.compose(local, tier).rebase_to(star_pin, tier);
        // The occupant RODE the realm: its rendered position advanced by exactly the realm's own motion.
        let world = |p: LatticePos| p.cell().as_dvec3() * tier.cell_edge_m() + p.offset();
        let occupant_moved = world(seen1) - world(seen0);
        let realm_moved =
            (a1.cell() - a0.cell()).as_dvec3() * tier.cell_edge_m() + (a1.offset() - a0.offset());
        assert!((occupant_moved - realm_moved).length() < 1e-6);
        // And it is NOT static — the regression guard against the "occupant hangs in place" bug.
        assert!(occupant_moved.length() > 1.0);
    }

    // ===== FrameXform — one placement, folded and applied =====================================

    /// A placement 145 m out along +x with no motion and no rotation — the worked example's planet
    /// inside its star system.
    fn placed(x_m: f64) -> FrameXform {
        FrameXform {
            pos: LatticePos::local(DVec3::new(x_m, 0.0, 0.0)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
        }
    }

    fn planet_frame() -> FrameRef {
        FrameRef::PlanetCentered { planet_seed: 7 }
    }

    fn system_frame() -> FrameRef {
        FrameRef::SystemSpace { system_seed: 1 }
    }

    #[test]
    fn is_identity_is_exact_and_a_nearly_unrotated_frame_does_not_collapse() {
        assert!(FrameXform::IDENTITY.is_identity());
        // A quaternion one ulp off the identity is ROTATED. If a tolerance crept in here, a realm
        // spinning slowly enough to fall inside it would silently stop spinning for every client.
        let nearly = FrameXform {
            orient: DQuat::from_xyzw(0.0, 0.0, f64::EPSILON, 1.0),
            ..FrameXform::IDENTITY
        };
        assert!(!nearly.is_identity());
        assert!(!placed(1.0).is_identity());
        assert!(
            !FrameXform {
                vel: DVec3::new(0.0, 0.0, 1.0),
                ..FrameXform::IDENTITY
            }
            .is_identity()
        );
    }

    #[test]
    fn apply_adds_the_placement_and_relabels_to_the_parent() {
        // The worked example's middle hop: the star system holds an occupant reported 3 m from its
        // planet's centre and adds the 145 m it placed that planet at. The LABEL must move with the
        // value — a composed value still wearing the child's frame name is the exact lie this work
        // removes.
        let pose = StampedPose::at_rest(planet_frame(), DVec3::new(3.0, 0.0, 0.0), UniverseTick(9));
        let out = placed(145.0)
            .apply(pose, Tier::Fine, system_frame())
            .expect("an unrotated placement always applies");
        assert_eq!(out.pos.offset(), DVec3::new(148.0, 0.0, 0.0));
        assert_eq!(out.frame, system_frame());
        assert_eq!(out.universe_tick, UniverseTick(9));
    }

    #[test]
    fn then_folds_two_levels_into_one_add() {
        // 3 inside the planet, the planet 145 inside the system, the system 12031 inside the galaxy.
        // Folding the two placements first and applying once must equal applying them in turn.
        let folded = placed(145.0)
            .then(placed(12031.0), Tier::Fine)
            .expect("unrotated placements always fold");
        assert_eq!(folded.pos.offset(), DVec3::new(12176.0, 0.0, 0.0));
        let pose = StampedPose::at_rest(planet_frame(), DVec3::new(3.0, 0.0, 0.0), UniverseTick(0));
        let one_shot = folded
            .apply(pose, Tier::Fine, FrameRef::GalaxySpace)
            .expect("applies");
        let step_by_step = placed(12031.0)
            .apply(
                placed(145.0)
                    .apply(pose, Tier::Fine, system_frame())
                    .expect("applies"),
                Tier::Fine,
                FrameRef::GalaxySpace,
            )
            .expect("applies");
        assert_eq!(one_shot.pos.offset(), DVec3::new(12179.0, 0.0, 0.0));
        assert_eq!(one_shot.pos.offset(), step_by_step.pos.offset());
    }

    #[test]
    fn inverse_undoes_a_placement_exactly_including_the_integer_anchor() {
        // A placement one whole anchor-block out, with motion. The round trip must return the pose
        // BIT-for-bit — the cell negates as an integer, so the anchor never passes through f64.
        let x = FrameXform {
            pos: LatticePos::at(
                I64Vec3::new(1_000_000_000_000, 0, 0),
                DVec3::new(0.25, 0.0, 0.0),
            ),
            vel: DVec3::new(0.0, 7.5, 0.0),
            orient: DQuat::IDENTITY,
        };
        let pose = StampedPose {
            frame: planet_frame(),
            pos: LatticePos::at(I64Vec3::new(4096, 0, 0), DVec3::new(0.5, -0.25, 0.75)),
            vel: DVec3::new(1.0, 0.0, -2.0),
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(11),
        };
        let up = x.apply(pose, Tier::Fine, system_frame()).expect("applies");
        assert_eq!(up.pos.cell(), I64Vec3::new(1_000_000_004_096, 0, 0));
        let back = x
            .inverse(Tier::Fine)
            .expect("an unrotated placement always inverts")
            .apply(up, Tier::Fine, planet_frame())
            .expect("applies");
        assert_eq!(back.pos.cell(), pose.pos.cell());
        assert_eq!(back.pos.offset(), pose.pos.offset());
        assert_eq!(back.vel, pose.vel);
        assert_eq!(back.frame, planet_frame());
    }

    #[test]
    fn a_rotated_placement_inverts_and_applies_while_every_anchor_is_zero() {
        // The cell guard is about the INTEGER anchor, not about rotation as such: a rotated placement
        // whose origin sits inside one cell is perfectly representable and must keep working.
        let quarter = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let x = FrameXform {
            pos: LatticePos::local(DVec3::new(10.0, 0.0, 0.0)),
            vel: DVec3::ZERO,
            orient: quarter,
        };
        let pose = StampedPose::at_rest(planet_frame(), DVec3::new(2.0, 0.0, 0.0), UniverseTick(1));
        let up = x.apply(pose, Tier::Fine, system_frame()).expect("applies");
        // +x in the child's axes points along +y in the parent's after a quarter turn about z.
        assert!((up.pos.offset() - DVec3::new(10.0, 2.0, 0.0)).length() < 1e-12);
        let back = x
            .inverse(Tier::Fine)
            .expect("inverts")
            .apply(up, Tier::Fine, planet_frame())
            .expect("applies");
        assert!((back.pos.offset() - DVec3::new(2.0, 0.0, 0.0)).length() < 1e-12);
    }

    #[test]
    fn a_rotated_placement_across_integer_cells_is_refused_on_every_operation() {
        // A cell COUNT is measured along one frame's axes; rotating a non-zero count does not give a
        // count. The only alternatives are to fold the anchor into f64 metres — the precision loss this
        // coordinate exists to prevent — or to refuse. All three entry points refuse.
        let quarter = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let far = LatticePos::at(I64Vec3::new(1_000_000_000_000, 0, 0), DVec3::ZERO);
        let rotated_far = FrameXform {
            pos: far,
            vel: DVec3::ZERO,
            orient: quarter,
        };
        assert_eq!(
            rotated_far.inverse(Tier::Fine).expect_err("refused"),
            FrameError::RotatedFrameAcrossCells
        );
        let pose_far = StampedPose {
            frame: planet_frame(),
            pos: far,
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        };
        let rotated_here = FrameXform {
            pos: LatticePos::local(DVec3::ZERO),
            vel: DVec3::ZERO,
            orient: quarter,
        };
        assert_eq!(
            rotated_here
                .apply(pose_far, Tier::Fine, system_frame())
                .expect_err("refused"),
            FrameError::RotatedFrameAcrossCells
        );
        // `then` refuses on the OUTER placement's rotation against the INNER's anchor.
        let inner_far = FrameXform {
            pos: far,
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
        };
        assert_eq!(
            inner_far
                .then(rotated_here, Tier::Fine)
                .expect_err("refused"),
            FrameError::RotatedFrameAcrossCells
        );
        // …and permits it once the anchor is zero (the other side of the same guard).
        let inner_here = FrameXform {
            pos: LatticePos::local(DVec3::new(1.0, 0.0, 0.0)),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
        };
        assert!(inner_here.then(rotated_here, Tier::Fine).is_ok());
    }

    use proptest::prelude::*;

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
        fn compose_then_rebase_recovers_the_local_position(
            ax in -100_000i64..100_000, ay in -100_000i64..100_000, az in -100_000i64..100_000,
            lx in -100_000i64..100_000, ly in -100_000i64..100_000, lz in -100_000i64..100_000,
        ) {
            // origin ∘ local, then rebase back to origin, recovers `local` EXACTLY (integer cells cancel;
            // offsets ZERO ⇒ no float error): compose and rebase_to are inverses.
            let origin = LatticePos::at(I64Vec3::new(ax, ay, az), DVec3::ZERO);
            let local = LatticePos::at(I64Vec3::new(lx, ly, lz), DVec3::ZERO);
            let composed = origin.compose(local, Tier::Fine);
            let back = composed.rebase_to(origin, Tier::Fine);
            prop_assert_eq!(back.cell(), local.cell());
            prop_assert_eq!(back.offset(), local.offset());
        }
    }
}
