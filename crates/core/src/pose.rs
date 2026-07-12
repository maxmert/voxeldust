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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
            // Sanitize the local offset; the integer cell is exact (an i64 cannot be non-finite),
            // so it passes through — preserving any future non-zero cell anchor.
            pos: LatticePos {
                cell: self.pos.cell,
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
}
