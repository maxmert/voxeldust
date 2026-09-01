//! The capability DAG: a "shard type" is a VALIDATED configuration, not a codebase
//! (HR3/HR4; `docs/design/sealed_shards.md` §4).
//!
//! `ShardProfile::build` is the ONLY constructor: it derives dependent capabilities
//! (`functional_blocks ⇒ signal_graph`) and FAILS LOUD at config load on incoherent
//! requests (`functional_blocks` without a voxel realm) — never a runtime explosion
//! in a feature system querying an uninitialized table.
//!
//! Feature code dispatches by capability accessor; `match`ing on a shard-kind
//! discriminant in feature code is forbidden (G-NO-SHARD-FORK). Geometry differences
//! are confined to `FrameSpace` impls selected by `voxel().geometry` (P4/P5).
//!
//! KNOWN LIMIT (DEFERRED [[D-38]]): this is HR4's STRUCTURAL half. The NAMED
//! `assert_feature_anywhere` gate EXISTS (`stub.rs` mod tests): ONE crossing fixture
//! over a Shell AND an Aabb child region — but both runs share one `Rig::new()` shard
//! kind (the profile ties are asserts on the profile objects). The two-SHARD-KIND run
//! of one identical body is `drive_swept_crossing_feature` (D-PLACE-1). The variant
//! forcing a `reanchor()` stays owed at P5 (`FrameSpace` does not exist yet).

use serde::{Deserialize, Serialize};

/// What a node process is. Gateway/orchestrator/relay are NOT shards; every sim
/// shard is ONE binary parameterized by a validated [`ShardProfile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Gateway,
    Orchestrator,
    GalaxyRelay,
    Shard(ShardProfile),
    /// P1–P3 proving ground: entities are points in empty space; no capabilities.
    StubShard,
}

/// Voxel-realm geometry — the ONE seam where spherical planets and Cartesian ship
/// grids differ (everything above `FrameSpace` is shared write-once).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoxelGeometry {
    /// Planet surface: tangent-anchored spherical projection with re-anchoring.
    Spherical,
    /// Ship/station interior: flat grid, no re-anchoring (AnchorGen never advances).
    Cartesian,
}

/// The raw, unvalidated request (parsed from config). `ShardProfile::build` is the
/// only path from here to a usable profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapRequest {
    pub voxel: Option<VoxelGeometry>,
    pub signal_graph: bool,
    pub functional_blocks: bool,
    pub block_edit: bool,
    pub surfaces: bool,
    pub seats: bool,
    pub signal_relay: bool,
    /// May host ship-exterior hull bodies (system/planet/station shards).
    pub hull_host: bool,
    /// ★ CAN PUSH ITSELF — the realm has engines or a motor, so it may state a drive to its parent
    /// (D-MOVE-2; owner 2026-08-31, *"this is the specific of this concrete realm behavior, same as
    /// physics inside each realm"*). A ship has this; a station with thrusters has this; a star system
    /// does not steer and a map marker has no body.
    ///
    /// ⚠ **A PERSON IS NOT LISTED HERE, AND THAT IS DELIBERATE.** A person is an OCCUPANT, never a
    /// realm — SL2's own line is *"CONTACTS ARE REALMS, PEOPLE ARE SEEN"*, and `EntityKind::Player` is
    /// the first entry of the transfer registry. An occupant's push never crosses a boundary at all:
    /// the realm holding it applies the push in its own process. This switch governs the WIRE lane,
    /// which only ever carries a child REALM speaking to its parent.
    ///
    /// **THIS IS A SWITCH, NEVER A KIND TEST.** Written as "is this a ship?" a station with thrusters
    /// could not move and a person could not walk, and each new mover would need adding to a list.
    /// Written as a switch, a station turns it on and flies with the code that already exists (HR3:
    /// never match on a shard kind in a feature).
    pub self_driven: bool,
    /// ★ DOES THE PHYSICS FOR THE THINGS INSIDE IT — the realm takes the drives of what it holds, adds
    /// its own ambient, integrates, and AUTHORS the resulting placements (D-MOVE-2). A star system has
    /// this; a planet has it; a ship has it for its own crew and for anything docked inside it.
    ///
    /// **A SHIP CARRIES BOTH SWITCHES AT ONCE.** It states a drive UPWARD to the system holding it, and
    /// it integrates what is inside it. That pair is why the two switches are separate rather than one.
    ///
    /// ★ **THE CORRECTED PROOF THAT THE LANE MUST BE GENERIC (2026-08-31).** An earlier draft argued it
    /// from "a ship speaks upward and listens to its crew". That argument is WEAK, and the owner's
    /// question about player realms exposed it: a crew member is an occupant in the SAME process, so
    /// its push never touches the wire. The listening half proved nothing about the wire lane.
    ///
    /// The real proof is HR4's own gate — the identical fixture must pass on TWO REALM KINDS. A ship
    /// inside a star system and a ship inside a planet send the SAME message to two different kinds of
    /// parent, and the ruling's acceptance test is that the ship *"does not know it moved house"*. A
    /// ship-specific lane cannot pass that gate, because the second parent would need ship-specific
    /// code to receive it — and then the ship WOULD know.
    pub integrates_children: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum ProfileError {
    #[error("functional_blocks requires a voxel realm (blocks live in voxels)")]
    FunctionalBlocksNeedVoxel,
    #[error("block_edit requires a voxel realm")]
    BlockEditNeedsVoxel,
    #[error("surfaces/seats require a voxel realm")]
    SurfacesNeedVoxel,
}

/// A VALIDATED capability set. Fields are private: coherence is a construction
/// invariant, not a runtime hope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardProfile {
    voxel: Option<VoxelGeometry>,
    signal_graph: bool,
    functional_blocks: bool,
    block_edit: bool,
    surfaces: bool,
    seats: bool,
    signal_relay: bool,
    hull_host: bool,
    self_driven: bool,
    integrates_children: bool,
}

impl ShardProfile {
    /// The ONLY constructor: derives the capability lattice and rejects incoherent
    /// requests with a typed error at parse time.
    pub fn build(req: CapRequest) -> Result<ShardProfile, ProfileError> {
        // Lattice derivation: functional blocks publish/consume signals.
        let signal_graph = req.signal_graph || req.functional_blocks;

        // Voxel-dependent capabilities fail loud without a realm geometry.
        if req.functional_blocks && req.voxel.is_none() {
            return Err(ProfileError::FunctionalBlocksNeedVoxel);
        }
        if req.block_edit && req.voxel.is_none() {
            return Err(ProfileError::BlockEditNeedsVoxel);
        }
        if (req.surfaces || req.seats) && req.voxel.is_none() {
            return Err(ProfileError::SurfacesNeedVoxel);
        }

        Ok(ShardProfile {
            voxel: req.voxel,
            signal_graph,
            functional_blocks: req.functional_blocks,
            block_edit: req.block_edit,
            surfaces: req.surfaces,
            seats: req.seats,
            signal_relay: req.signal_relay,
            hull_host: req.hull_host,
            self_driven: req.self_driven,
            integrates_children: req.integrates_children,
        })
    }

    #[must_use]
    pub fn voxel(&self) -> Option<VoxelGeometry> {
        self.voxel
    }
    #[must_use]
    pub fn signal_graph(&self) -> bool {
        self.signal_graph
    }
    #[must_use]
    pub fn functional_blocks(&self) -> bool {
        self.functional_blocks
    }
    #[must_use]
    pub fn block_edit(&self) -> bool {
        self.block_edit
    }
    #[must_use]
    pub fn surfaces(&self) -> bool {
        self.surfaces
    }
    #[must_use]
    pub fn seats(&self) -> bool {
        self.seats
    }
    #[must_use]
    pub fn signal_relay(&self) -> bool {
        self.signal_relay
    }
    #[must_use]
    pub fn hull_host(&self) -> bool {
        self.hull_host
    }
    /// May this realm state a drive to its parent? See [`CapRequest::self_driven`].
    #[must_use]
    pub fn self_driven(&self) -> bool {
        self.self_driven
    }
    /// Does this realm integrate its children's drives and author their placements? See
    /// [`CapRequest::integrates_children`].
    #[must_use]
    pub fn integrates_children(&self) -> bool {
        self.integrates_children
    }

    /// Does this profile provide EVERY capability a re-home subject REQUIRES (D-37 target selection)? The
    /// target must match the required voxel GEOMETRY exactly (a ship's Cartesian realm can never re-home
    /// onto a Spherical shard, and vice-versa) AND provide every required boolean capability. An EMPTY
    /// request (`CapRequest::default()` — a bare point entity in empty space, every P3 stub subject) is
    /// satisfied by ANY profile. HR3: a capability match, never a shard-kind discriminant. Branchless
    /// bitwise `&`/`|` (HR5, no short-circuit gaps): each `!req.x | self.x` reads "req.x implies self.x".
    #[must_use]
    pub fn satisfies(&self, req: &CapRequest) -> bool {
        let voxel_ok = match req.voxel {
            None => true,                                   // no geometry requirement
            Some(geometry) => self.voxel == Some(geometry), // exact geometry match
        };
        voxel_ok
            & (!req.signal_graph | self.signal_graph)
            & (!req.functional_blocks | self.functional_blocks)
            & (!req.block_edit | self.block_edit)
            & (!req.surfaces | self.surfaces)
            & (!req.seats | self.seats)
            & (!req.signal_relay | self.signal_relay)
            & (!req.hull_host | self.hull_host)
    }
}

/// The canonical profiles as DATA (sealed_shards §4): shard types are coherent
/// configurations, not codebases. New shard types ("moon", "derelict") are new
/// values here — zero new feature code.
pub mod profiles {
    use super::{CapRequest, ProfileError, ShardProfile, VoxelGeometry};

    pub fn galaxy() -> Result<ShardProfile, ProfileError> {
        ShardProfile::build(CapRequest {
            signal_relay: true,
            // A galaxy holds the ships crossing between its systems, so it integrates their drives
            // (D-MOVE-2). It does not steer itself.
            integrates_children: true,
            ..CapRequest::default()
        })
    }

    pub fn system() -> Result<ShardProfile, ProfileError> {
        ShardProfile::build(CapRequest {
            hull_host: true,
            // The parent in the movement ruling's own worked example: it adds the star's pull to a
            // ship's push and authors where the ship now is (D-MOVE-2).
            integrates_children: true,
            ..CapRequest::default()
        })
    }

    pub fn planet() -> Result<ShardProfile, ProfileError> {
        ShardProfile::build(CapRequest {
            voxel: Some(VoxelGeometry::Spherical),
            functional_blocks: true,
            block_edit: true,
            surfaces: true,
            seats: true,
            hull_host: true,
            // The SECOND realm kind the HR4 gate needs: the identical fixture must pass with a ship
            // inside a planet as well as inside a star system, and the ship must not know the
            // difference (D-MOVE-2).
            integrates_children: true,
            ..CapRequest::default()
        })
    }

    pub fn ship() -> Result<ShardProfile, ProfileError> {
        ShardProfile::build(CapRequest {
            voxel: Some(VoxelGeometry::Cartesian),
            functional_blocks: true,
            block_edit: true,
            surfaces: true,
            seats: true,
            // ★ BOTH SWITCHES, and a ship is the realm that shows why they are two. It PUSHES ITSELF
            // against the realm holding it, and it INTEGRATES what is inside it — its crew, and
            // anything docked in its hangar (D-MOVE-2).
            self_driven: true,
            integrates_children: true,
            ..CapRequest::default()
        })
    }

    pub fn asteroid() -> Result<ShardProfile, ProfileError> {
        ShardProfile::build(CapRequest {
            voxel: Some(VoxelGeometry::Cartesian),
            block_edit: true,
            ..CapRequest::default()
        })
    }

    /// A space station: a block-built Cartesian hull like a [`ship`], PLUS `hull_host` (it
    /// hosts its own exterior body in the parent planet/system realm, like a ship exterior)
    /// and `signal_relay` (its functional blocks are a cross-shard SIGNAL source/sink — the
    /// end-goal's block-comms). DATA only — no per-realm-kind feature code (HR3).
    pub fn station() -> Result<ShardProfile, ProfileError> {
        ShardProfile::build(CapRequest {
            voxel: Some(VoxelGeometry::Cartesian),
            functional_blocks: true,
            block_edit: true,
            surfaces: true,
            seats: true,
            signal_relay: true,
            hull_host: true,
            // A station holds ships in its bays, so it integrates their drives. It does NOT push
            // itself: station-keeping is a later question, and the switch exists so that answer is
            // DATA rather than a code change (D-MOVE-2).
            integrates_children: true,
            ..CapRequest::default()
        })
    }

    /// The bare P3 empty-space subject: no capabilities, satisfied by any target. Named as
    /// DATA here (not an inline `build(default())` at the one call site) so the empty
    /// capability set has a single home like every other profile (HR3, one data row).
    pub fn stub() -> Result<ShardProfile, ProfileError> {
        ShardProfile::build(CapRequest::default())
    }
}

/// The ONE total core->sim map: a wildcard-free `ProfileKind -> ShardProfile` match onto the
/// canonical [`profiles`]. A new `ProfileKind` variant is a HARD compile error here until
/// mapped — the drift trap (a new body kind can never silently fall through to a wrong or
/// default profile). `ProfileKind` lives in vd-core (a pure tag); this map lives in vd-sim
/// because `ShardProfile` is a sim type — the tag flows DOWN the `bins->node->sim->wire->core`
/// arrow, never a reverse edge. Fallible by contract (never swallows the constructor `Result`).
pub fn profile_for(kind: vd_core::taxonomy::ProfileKind) -> Result<ShardProfile, ProfileError> {
    use vd_core::taxonomy::ProfileKind;
    match kind {
        ProfileKind::Galaxy => profiles::galaxy(),
        ProfileKind::System => profiles::system(),
        ProfileKind::Planet => profiles::planet(),
        ProfileKind::Ship => profiles::ship(),
        ProfileKind::Asteroid => profiles::asteroid(),
        ProfileKind::Station => profiles::station(),
        // An Area is presently a passive Cartesian district hull -> ship-like caps (DATA
        // reuse, HR3). DEFERRED: a signal-relaying/hull-hosting Area (a spaceport district)
        // gets its own profiles::area() — a one-arm edit, first needed at P8/P9.
        ProfileKind::Area => profiles::ship(),
        ProfileKind::Stub => profiles::stub(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_profiles_are_coherent() {
        let galaxy = profiles::galaxy().expect("galaxy");
        assert!(galaxy.signal_relay());
        assert_eq!(galaxy.voxel(), None);

        let system = profiles::system().expect("system");
        assert!(system.hull_host());
        assert!(!system.signal_graph());

        let planet = profiles::planet().expect("planet");
        assert_eq!(planet.voxel(), Some(VoxelGeometry::Spherical));
        assert!(planet.functional_blocks());
        assert!(planet.signal_graph(), "derived from functional_blocks");
        assert!(planet.block_edit());
        assert!(planet.surfaces());
        assert!(planet.seats());
        assert!(planet.hull_host());

        let ship = profiles::ship().expect("ship");
        assert_eq!(ship.voxel(), Some(VoxelGeometry::Cartesian));
        assert!(ship.signal_graph());
        assert!(!ship.hull_host());
        assert!(!ship.signal_relay());

        let asteroid = profiles::asteroid().expect("asteroid");
        assert!(asteroid.block_edit());
        assert!(!asteroid.functional_blocks());
        assert!(!asteroid.signal_graph());

        // A station is a ship-like Cartesian block hull that ALSO hull-hosts + signal-relays
        // (a first-class cross-shard signal source), coherent under build().
        let station = profiles::station().expect("station");
        assert_eq!(station.voxel(), Some(VoxelGeometry::Cartesian));
        assert!(station.functional_blocks());
        assert!(station.signal_graph());
        assert!(station.block_edit());
        assert!(station.surfaces());
        assert!(station.seats());
        assert!(station.hull_host());
        assert!(station.signal_relay());
    }

    /// The NEGATIVE gate (G-IDENTICAL's companion): incoherent requests never reach
    /// feature registration — they fail loud at build.
    #[test]
    fn incoherent_requests_fail_loud() {
        let cases = [
            (
                CapRequest {
                    functional_blocks: true,
                    ..CapRequest::default()
                },
                ProfileError::FunctionalBlocksNeedVoxel,
            ),
            (
                CapRequest {
                    block_edit: true,
                    ..CapRequest::default()
                },
                ProfileError::BlockEditNeedsVoxel,
            ),
            (
                CapRequest {
                    surfaces: true,
                    ..CapRequest::default()
                },
                ProfileError::SurfacesNeedVoxel,
            ),
            (
                CapRequest {
                    seats: true,
                    ..CapRequest::default()
                },
                ProfileError::SurfacesNeedVoxel,
            ),
        ];
        for (req, expected) in cases {
            assert_eq!(ShardProfile::build(req).expect_err("incoherent"), expected);
        }
    }

    #[test]
    fn signal_graph_standalone_is_legal_without_voxel() {
        let profile = ShardProfile::build(CapRequest {
            signal_graph: true,
            ..CapRequest::default()
        })
        .expect("signal-only profile");
        assert!(profile.signal_graph());
        assert_eq!(profile.voxel(), None);
    }

    /// Exhaustive operand combinations of the lattice's short-circuit operators —
    /// every `||` side is exercised both ways (HR5 branch coverage is honest).
    #[test]
    fn lattice_operator_combinations() {
        // signal_graph || functional_blocks: both requested explicitly.
        let both = ShardProfile::build(CapRequest {
            voxel: Some(VoxelGeometry::Cartesian),
            signal_graph: true,
            functional_blocks: true,
            ..CapRequest::default()
        })
        .expect("both signal sources");
        assert!(both.signal_graph());
        assert!(both.functional_blocks());

        // surfaces || seats: seats alone (right side), with a voxel realm (success).
        let seats_only = ShardProfile::build(CapRequest {
            voxel: Some(VoxelGeometry::Cartesian),
            seats: true,
            ..CapRequest::default()
        })
        .expect("seats with voxel");
        assert!(seats_only.seats());
        assert!(!seats_only.surfaces());

        // surfaces alone (left side), with a voxel realm (success).
        let surfaces_only = ShardProfile::build(CapRequest {
            voxel: Some(VoxelGeometry::Spherical),
            surfaces: true,
            ..CapRequest::default()
        })
        .expect("surfaces with voxel");
        assert!(surfaces_only.surfaces());
        assert!(!surfaces_only.seats());
    }

    #[test]
    fn profile_errors_display() {
        assert_eq!(
            ProfileError::FunctionalBlocksNeedVoxel.to_string(),
            "functional_blocks requires a voxel realm (blocks live in voxels)"
        );
        assert_eq!(
            ProfileError::BlockEditNeedsVoxel.to_string(),
            "block_edit requires a voxel realm"
        );
        assert_eq!(
            ProfileError::SurfacesNeedVoxel.to_string(),
            "surfaces/seats require a voxel realm"
        );
    }

    #[test]
    fn a_galaxy_coord_selects_a_signal_relay_profile_a_system_does_not() {
        // RLM Step 5a: the `VD_OWN_COORD → coord.profile_kind() → profile_for` chain PRESERVES the Galaxy
        // level (which `RealmId` collapses to `System(1)`), so a Galaxy shard's `ShardProfile` carries
        // `signal_relay` — the capability that later uncorners cross-shard Signals (P9). The System contrast
        // proves it is the Galaxy LEVEL, not a universal cap.
        use vd_core::realm_coord::RealmCoord;
        use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
        let galaxy = RealmCoord::from_path(RealmPath::from_levels(vec![
            RealmLevel::new(RealmKindTag::Universe, 0),
            RealmLevel::new(RealmKindTag::Galaxy, 1),
        ]))
        .expect("two-level galaxy lineage");
        assert!(
            profile_for(galaxy.profile_kind())
                .expect("galaxy profile")
                .signal_relay(),
            "a Galaxy coord must select a signal_relay-capable profile"
        );
        let system = RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(
            RealmKindTag::System,
            7,
        )]))
        .expect("one-level system lineage");
        assert!(
            !profile_for(system.profile_kind())
                .expect("system profile")
                .signal_relay(),
            "a System coord's profile has no signal_relay"
        );
    }

    #[test]
    fn node_kinds_roundtrip() {
        let kinds = vec![
            NodeKind::Gateway,
            NodeKind::Orchestrator,
            NodeKind::GalaxyRelay,
            NodeKind::StubShard,
            NodeKind::Shard(profiles::ship().expect("ship")),
        ];
        for kind in kinds {
            let bytes = postcard::to_allocvec(&kind).expect("encode");
            assert_eq!(
                postcard::from_bytes::<NodeKind>(&bytes).expect("decode"),
                kind
            );
        }
    }

    #[test]
    fn satisfies_matches_voxel_geometry_and_required_booleans() {
        // An EMPTY request (a bare P3 point entity in empty space) is satisfied by ANY profile.
        let stub = ShardProfile::build(CapRequest::default()).expect("empty profile is coherent");
        assert!(
            stub.satisfies(&CapRequest::default()),
            "empty req ⇒ a bare stub satisfies"
        );
        let planet = profiles::planet().expect("planet"); // Spherical voxel
        assert!(
            planet.satisfies(&CapRequest::default()),
            "empty req ⇒ a planet satisfies too"
        );

        // VOXEL GEOMETRY must match EXACTLY (a ship's Cartesian realm can never re-home onto a Spherical
        // shard, and vice-versa).
        let spherical = CapRequest {
            voxel: Some(VoxelGeometry::Spherical),
            ..CapRequest::default()
        };
        let cartesian = CapRequest {
            voxel: Some(VoxelGeometry::Cartesian),
            ..CapRequest::default()
        };
        assert!(
            !stub.satisfies(&spherical),
            "a stub (no voxel) cannot host a voxel realm"
        );
        assert!(
            planet.satisfies(&spherical),
            "a Spherical planet hosts a Spherical realm"
        );
        assert!(
            !planet.satisfies(&cartesian),
            "a Spherical planet cannot host a Cartesian realm"
        );
        assert!(
            profiles::ship().expect("ship").satisfies(&cartesian),
            "a Cartesian ship hosts a Cartesian realm"
        );

        // A required BOOLEAN capability the profile LACKS ⇒ not satisfied; PRESENT ⇒ satisfied.
        let needs_relay = CapRequest {
            signal_relay: true,
            ..CapRequest::default()
        };
        assert!(
            !planet.satisfies(&needs_relay),
            "a planet lacks signal_relay ⇒ not satisfied"
        );
        assert!(
            profiles::galaxy().expect("galaxy").satisfies(&needs_relay),
            "a galaxy relay provides signal_relay ⇒ satisfied"
        );
    }

    #[test]
    fn profile_for_accepts_every_realm_coord_profile_kind() {
        // RLM Step 1 consistency: every RealmCoord kind lowers to a ProfileKind that profile_for
        // builds (this check lives here, not in vd-core, since profile_for is a vd-sim fn). Pins the
        // Universe/Galaxy → Galaxy collapse the lifecycle relies on.
        use vd_core::realm_coord::RealmCoord;
        use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
        let coord = |kind| {
            RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(kind, 7)]))
                .expect("1-level path has a leaf")
        };
        for kind in RealmKindTag::ALL {
            let c = coord(kind);
            assert!(
                profile_for(c.profile_kind()).is_ok(),
                "{kind:?} profile builds"
            );
        }
        // Universe and Galaxy collapse to the SAME (Galaxy) profile.
        assert_eq!(
            coord(RealmKindTag::Universe).profile_kind(),
            coord(RealmKindTag::Galaxy).profile_kind()
        );
    }

    #[test]
    fn profile_for_is_total_over_profile_kind() {
        use vd_core::taxonomy::ProfileKind;
        // Every ProfileKind maps to a buildable profile (the wildcard-free match is total).
        for k in ProfileKind::ALL {
            assert!(profile_for(k).is_ok(), "{k:?} builds");
        }
        // Spot-check each arm's capability contract.
        assert!(
            profile_for(ProfileKind::Galaxy)
                .expect("galaxy")
                .signal_relay()
        );
        assert_eq!(
            profile_for(ProfileKind::Galaxy).expect("galaxy").voxel(),
            None
        );
        assert!(
            profile_for(ProfileKind::System)
                .expect("system")
                .hull_host()
        );
        assert!(
            !profile_for(ProfileKind::System)
                .expect("system")
                .signal_graph()
        );
        assert_eq!(
            profile_for(ProfileKind::Planet).expect("planet").voxel(),
            Some(VoxelGeometry::Spherical)
        );
        assert!(
            profile_for(ProfileKind::Planet)
                .expect("planet")
                .functional_blocks()
        );
        assert_eq!(
            profile_for(ProfileKind::Ship).expect("ship").voxel(),
            Some(VoxelGeometry::Cartesian)
        );
        assert!(
            profile_for(ProfileKind::Asteroid)
                .expect("asteroid")
                .block_edit()
        );
        assert!(
            !profile_for(ProfileKind::Asteroid)
                .expect("asteroid")
                .functional_blocks()
        );
        assert!(
            profile_for(ProfileKind::Station)
                .expect("station")
                .signal_relay()
        );
        assert!(
            profile_for(ProfileKind::Station)
                .expect("station")
                .hull_host()
        );
        assert_eq!(
            profile_for(ProfileKind::Area).expect("area").voxel(),
            Some(VoxelGeometry::Cartesian)
        );
    }

    #[test]
    fn profile_for_maps_every_arm_to_its_named_profile() {
        use vd_core::taxonomy::ProfileKind;
        // Each arm equals calling the named profiles::* directly — proves no drift AND that
        // every arm of the total match executes (Area reuses ship; Stub is the empty set).
        assert_eq!(
            profile_for(ProfileKind::Galaxy).expect("g"),
            profiles::galaxy().expect("g")
        );
        assert_eq!(
            profile_for(ProfileKind::System).expect("sy"),
            profiles::system().expect("sy")
        );
        assert_eq!(
            profile_for(ProfileKind::Planet).expect("p"),
            profiles::planet().expect("p")
        );
        assert_eq!(
            profile_for(ProfileKind::Ship).expect("sh"),
            profiles::ship().expect("sh")
        );
        assert_eq!(
            profile_for(ProfileKind::Asteroid).expect("a"),
            profiles::asteroid().expect("a")
        );
        assert_eq!(
            profile_for(ProfileKind::Station).expect("st"),
            profiles::station().expect("st")
        );
        assert_eq!(
            profile_for(ProfileKind::Area).expect("ar"),
            profiles::ship().expect("ar")
        );
        assert_eq!(
            profile_for(ProfileKind::Stub).expect("stub"),
            profiles::stub().expect("stub")
        );
    }
}
