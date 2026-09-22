//! ★ THE HOME PLANET'S ARTIFACT PIN (the landform arc, slice 8c stage C4; gate G-DRIFT): the solve
//! is run in full on the home planet, on this chip, and its digest is compared with the committed
//! one — red on one differing byte. The same test on every other target the game ships on is the
//! measurement SL10 asks for (the terrain legs script runs it with the golden gate).
//!
//! A minute of one core: a GATE (`just artifact-pin`), never a unit test. To record a new digest
//! after a deliberate change to the solve or the artifact (with `ARTIFACT_VERSION` or
//! `GENERATOR_VERSION` bumped): `VD_ARTIFACT_RECORD=1 cargo test --release -p vd-bins --test
//! home_artifact_pin -- --nocapture`, then commit the constant below.

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_terrain::home::{home_planet, home_solve_words};

/// The committed digest of the home planet's artifact on this build.
const HOME_ARTIFACT_DIGEST: [u64; 2] = [0x1fa6_8f27_d0d1_125b, 0xda53_d30d_7ace_278c];

#[test]
fn the_home_planets_artifact_digests_to_its_committed_words() {
    let job = SolveJob {
        body: home_planet(),
        words: home_solve_words(),
    };
    let artifact = run_solve(&job).expect("the home planet solves");
    let digest = artifact.digest();
    println!(
        "home_artifact_pin: edge {}, {} nodes, {} pyramid levels, {} bytes, digest [{:#x}, {:#x}]",
        artifact.edge,
        artifact.node_count(),
        artifact.pyramid.len(),
        artifact.bytes(),
        digest[0],
        digest[1]
    );
    if std::env::var_os("VD_ARTIFACT_RECORD").is_some() {
        println!("home_artifact_pin: RECORD — commit the digest above as HOME_ARTIFACT_DIGEST");
        return;
    }
    // ★ GATE G-SEA (slice 8c stage C5; ruling T8's cure): the sea's level the solve found over the
    // eroded field, and the share of the globe under it, pinned — read here from the artifact's own
    // rows, area-weighted by the lattice's node areas.
    let lattice = job
        .body
        .macro_lattice()
        .expect("the home planet has a lattice");
    let sea = artifact.sea().expect("the home planet has a sea");
    let (mut wet, mut all) = (0.0f64, 0.0f64);
    for (node, row) in artifact.rows.iter().enumerate() {
        let area = lattice.area_m2(node as u32) as f64;
        all += area;
        if i32::from(row.z_m) <= sea {
            wet += area;
        }
    }
    let share_q4 = (wet / all * 10_000.0).round() as u32;
    println!(
        "home_artifact_pin: G-SEA the sea stands {sea} m over the ladder radius; ocean share {:.2} % ({share_q4} / 10 000)",
        wet / all * 100.0
    );
    if std::env::var_os("VD_ARTIFACT_RECORD").is_some() {
        println!(
            "home_artifact_pin: RECORD — commit HOME_PLANET_SEA_M = {sea}, HOME_PLANET_OCEAN_SHARE_Q4 = {share_q4}"
        );
        return;
    }
    assert_eq!(
        digest, HOME_ARTIFACT_DIGEST,
        "the home planet's artifact moved: a deliberate change re-records the digest"
    );
    assert_eq!(
        sea,
        vd_terrain::home::HOME_PLANET_SEA_M,
        "the home planet's sea moved: re-record on purpose"
    );
    assert_eq!(
        share_q4,
        vd_terrain::home::HOME_PLANET_OCEAN_SHARE_Q4,
        "the home planet's ocean share moved: re-record on purpose"
    );
    // ★ THE GOLDEN FIELDS ARE THE SOLVE'S OWN ROWS (slice 8c stage C4c): every pinned row and the
    // pinned top level equal the artifact's, and the identity's measured half is the pinned word.
    let fields = vd_terrain::home::home_golden_fields();
    assert_eq!(fields.levels as usize, artifact.pyramid.len());
    assert_eq!(
        fields.top.z_m,
        artifact.pyramid[fields.top.level as usize - 1],
        "the pinned top level is not the solve's"
    );
    for (node, (z, water, facies, province)) in &fields.rows.0 {
        let row = artifact.rows[*node as usize];
        assert_eq!(
            (
                row.z_m,
                row.water_m,
                row.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT,
                row.province
            ),
            (*z, *water, *facies, *province),
            "pinned row {node} is not the solve's"
        );
    }
    assert_eq!(
        fields.sea_m, artifact.sea_m,
        "the pinned sea is not the solve's"
    );
    assert_eq!(
        vd_terrain::digest::golden_self_check(&job.body, Some(&fields)),
        Some(vd_terrain::home::HOME_IDENTITY_MEASURED),
        "the identity's measured half moved: re-record with golden_z_record"
    );
}
