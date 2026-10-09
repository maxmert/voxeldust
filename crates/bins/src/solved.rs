//! ★ THE HOME PLANET SOLVED IN THE GATE'S OWN PROCESS (ruling B19 item 2, 2026-09-27: "Solve in the
//! test"; shared by the moving-eye gate and the picture gate since 2026-10-01, the owner: "Wait for
//! the solve"). A gate boots a new world, and the planet's shard solves its artifact at its first
//! boot; until the solve lands the client draws the recipe's own relief. So a gate that judges the
//! ground the players see solves the planet here, by the shard's own code
//! (`crate::artifact_worker::run_solve`), seats its stands on that ground, and waits until the
//! client holds the artifact (`crate::pixel::await_window_solved`).
//!
//! **Example.** The picture gate solves the home planet (the shard's own first solve took 227 s,
//! MEASURED 2026-10-01), finds the first dry land on the circle where the star stands 15° up, and
//! seats the ground stand 1.8 m over the solved ground there; the shard solves the same artifact
//! after the login, and the stand settles on it.

use std::time::Instant;

use vd_core::glam::{DQuat, DVec3};
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::Artifact;

/// The home planet's artifact solved here, and the body carrying the solved sea — the ground every
/// stand's picture draws once the client holds the artifact.
pub struct SolvedHome {
    pub body: BodyDefinition,
    pub artifact: Artifact,
}

impl SolvedHome {
    /// The GROUND's radius along a direction at a rung — the drawn surface, and the one the stamp's
    /// altitude reads against: no water sheet stands over any column since ruling B22 (2026-09-28).
    ///
    /// # Panics
    /// Never on the solved home planet: its field covers every direction.
    #[must_use]
    pub fn ground_m(&self, dir: DVec3, rung: u8) -> f64 {
        vd_terrain::height::height_field_m(&self.body, &self.artifact, dir.to_array(), rung)
            .expect("the solved field covers every direction")
    }

    /// Whether the column along a direction is dry at rung 0: its ground stands ABOVE the solve's
    /// sea level. No water is drawn (ruling B22), so a stand on the dry sea floor would be a stand
    /// the one-volume fill later drowns; a stand stands on land.
    #[must_use]
    pub fn dry(&self, dir: DVec3) -> bool {
        self.ground_m(dir, 0) > self.body.sea_radius_m()
    }
}

impl SolvedHome {
    /// ★ THE CLIENT'S CACHE OF THIS ARTIFACT, WHOLE: the head, every pyramid level and every tile
    /// handed to the client's own receiver (`vd_client::artifact_book::ArtifactReceiver`), so a
    /// gate reads the ground the client draws once every part has landed — its altitude, surface
    /// and biome through the client's own functions.
    ///
    /// # Panics
    /// If the receiver refuses the parts it was built to take.
    #[must_use]
    pub fn client_cache(
        &self,
        realm: vd_core::pose::RealmId,
    ) -> vd_client::artifact_book::ArtifactCache {
        use vd_wire::channels::BulkMsg;
        let artifact = &self.artifact;
        let drawn = artifact.drawn_range_m(&self.body.macro_lattice().expect("a lattice"));
        let mut rx = vd_client::artifact_book::ArtifactReceiver::default();
        rx.accept(BulkMsg::ArtifactHead {
            realm,
            world_tag: 0,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            coast_parts: 0,
            low_m: drawn.0,
            high_m: drawn.1,
        });
        for (k, level) in artifact.pyramid.iter().enumerate().rev() {
            rx.accept(BulkMsg::ArtifactPyramid {
                realm,
                level: k as u32 + 1,
                part: 0,
                parts: 1,
                z_m: level.clone(),
                water_m: Vec::new(),
            });
        }
        for face in vd_seed::bend::Face::ALL {
            for tx in 0..artifact.tiles_per_edge() {
                for ty in 0..artifact.tiles_per_edge() {
                    rx.accept(BulkMsg::ArtifactTile {
                        realm,
                        face: face.index(),
                        tx,
                        ty,
                        rows: artifact.tile(face, tx, ty).to_bytes(),
                    });
                }
            }
        }
        rx.book()
            .get(realm)
            .cloned()
            .expect("the receiver holds the artifact it took")
    }
}

/// Solve the home planet here, with the same body and words its shard reads; `label` names the
/// gate in the printed line.
///
/// # Panics
/// If `body` is not the home planet, or the solve fails.
#[must_use]
pub fn solve_home(body: &BodyDefinition, label: &str) -> SolvedHome {
    let job = crate::artifact_worker::SolveJob {
        body: vd_terrain::home::home_planet(),
        words: vd_terrain::home::home_solve_words(),
    };
    assert_eq!(
        (job.body.seed(), job.body.radius_m().to_bits()),
        (body.seed(), body.radius_m().to_bits()),
        "the solve's body is the gate's home planet"
    );
    let started = Instant::now();
    let artifact = crate::artifact_worker::run_solve(&job).expect("the home planet solves");
    eprintln!(
        "{label}: the home planet's artifact solved here in {:.1} s (digest {:?}, sea {:?} m)",
        started.elapsed().as_secs_f64(),
        artifact.digest(),
        artifact.sea()
    );
    // ★ THE BODY THE CLIENT DRAWS (ruling B55): the solved sea and the band the head's drawn range
    // sizes, as the client's lane takes them from the same head.
    let wet = (*body).with_sea_m(artifact.sea());
    let drawn = artifact.drawn_range_m(&wet.macro_lattice().expect("a lattice"));
    SolvedHome {
        body: wet.with_band_m(drawn.0, drawn.1).unwrap_or(wet),
        artifact,
    }
}

/// The dry-stand search's step along its circle, how densely it reads a walker's path, and how far
/// the eye stays over the drawn ground along it.
pub const STAND_STEP_M: f64 = 500.0;
pub const WALK_SAMPLE_M: f64 = 5.0;
pub const EYE_CLEARANCE_M: f64 = 0.5;

/// ★ A DRY STAND ON THE SOLVED GROUND, UNDER THE SAME STAR (ruling B19 item 3: "Move the stand to a 15°
/// star"). The solved world is mostly sea — the sun's stand on the equator is ocean 5 490 m deep, and
/// the day side of the equator has no land (MEASURED with `examples/equator_stands`) — so the stand
/// is searched along the circle where the star stands as high as it does over `d0`: `d0` turned
/// about the sun's own direction, [`STAND_STEP_M`] a step, one way then the other, to the first stand
/// where a walker's whole path of `path_m` along the nose is dry and its eye (`eye_height_m` over
/// the ground) stays [`EYE_CLEARANCE_M`] over the ground; a still stand states a path of nought.
/// The nose is level, the star's azimuth turned `off_nose_deg` about the radial. Returns the
/// direction, the nose and the ground's radius.
///
/// **Example.** On the committed world the walker stands 1 706.5 km along the 15° circle from the
/// sun's stand, 15.8° south, on dry ground with the star 15° up over the shoulder.
///
/// # Panics
/// If no stand on the circle is dry.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the circle (the sun and its start), the walker (its path, eye and nose) and the \
              gate's label are each a fact the caller states apart"
)]
pub fn walkable_stand(
    solved: &SolvedHome,
    sun: DVec3,
    d0: DVec3,
    path_m: f64,
    eye_height_m: f64,
    off_nose_deg: f64,
    label: &str,
) -> (DVec3, DVec3, f64) {
    let radius_m = solved.body.radius_m();
    // The circle's own radius: the stand lies `90° − elevation` from the sun's direction.
    let ring_m = radius_m * sun.dot(d0).acos().sin();
    let steps = (std::f64::consts::PI * ring_m / STAND_STEP_M).ceil() as u64;
    let samples = (path_m / WALK_SAMPLE_M).ceil() as u64;
    for k in 0..=steps {
        for sign in [1.0_f64, -1.0] {
            if k == 0 && sign < 0.0 {
                continue;
            }
            let psi = sign * k as f64 * STAND_STEP_M / ring_m;
            let d = (DQuat::from_axis_angle(sun, psi) * d0).normalize();
            if !solved.dry(d) {
                continue;
            }
            let toward_sun = (sun - d * sun.dot(d)).normalize();
            let level = DQuat::from_axis_angle(d, off_nose_deg.to_radians()) * toward_sun;
            let h = solved.ground_m(d, 0);
            let eye = d * (h + eye_height_m);
            let clear = (0..=samples).all(|i| {
                let p = eye + level * (i as f64 * WALK_SAMPLE_M);
                let q = p.normalize();
                solved.dry(q) && solved.ground_m(q, 0) <= p.length() - EYE_CLEARANCE_M
            });
            if clear {
                eprintln!(
                    "{label}: the dry stand {:.1} km along the circle where the star stands {:.1}° \
                     up ({} of the start), latitude {:.2}°, the ground {:.0} m over the body's \
                     radius and {:.0} m over the sea",
                    k as f64 * STAND_STEP_M / 1000.0,
                    sun.dot(d).asin().to_degrees(),
                    if sign > 0.0 {
                        "one way"
                    } else {
                        "the other way"
                    },
                    d.z.asin().to_degrees(),
                    h - radius_m,
                    h - radius_m - f64::from(solved.body.sea_m().unwrap_or(0))
                );
                return (d, level, h);
            }
        }
    }
    panic!(
        "{label}: no dry stand on the circle where the star stands {:.1}° up",
        sun.dot(d0).asin().to_degrees()
    );
}
