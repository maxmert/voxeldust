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
//! KNOWN LIMIT (DEFERRED [[D-38]]): this is HR4's STRUCTURAL half only. The literal
//! G-IDENTICAL gate — a NAMED `assert_feature_anywhere` test running ONE identical
//! feature fixture on a Spherical AND a Cartesian profile (one forcing a `reanchor()`)
//! — is owed-at-first-feature (P6 block edits), because no capability-bearing feature
//! code exists yet to diverge. The coherence test below is the foundation, not the gate.

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
            ..CapRequest::default()
        })
    }

    pub fn system() -> Result<ShardProfile, ProfileError> {
        ShardProfile::build(CapRequest {
            hull_host: true,
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
            ..CapRequest::default()
        })
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
}
