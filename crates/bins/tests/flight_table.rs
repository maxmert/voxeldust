//! **G-FLIGHT-TABLE** — THE SPEED LAW's flight table, MEASURED (real-scale design §4.3 as
//! corrected by addendum §A3.2: *"the flight table was wrong at every level — the ramp was never
//! integrated"*). Each leg of the corrected table is flown here tick-by-tick with the SAME pure
//! functions the sim's integrator flies (`vd_core::flight`), at the SAME derived tuning
//! (`FlightTuning::derive` off the DEV cadences — τ = T_WAKE), and the measured time is pinned
//! inside a band DERIVED from the addendum's closed form plus the discretization terms — never a
//! literal. The gate FAILS the day the law, the tuning derivation or the world's derived
//! distances move, and it prints every leg verbatim for the owner's table.
//!
//! WHY THE NUMBERS DIFFER FROM THE ADDENDUM'S 117.2 / 248.7 / 130.2 s: every §A3 number was
//! stamped ⟨UNMEASURED — τ⟩, computed at the PLACEHOLDER τ = 2.44 s (boot p99 guessed at 122
//! ticks). The SHIPPED τ is the same T_WAKE expression at the shipped posture — the DEV cluster's
//! demand beat (25 ticks), reactive boot (`boot_ticks_p99 = 0`) and the fixed pipeline tail —
//! `(2·25 + 0 + 5)·0.02 = 1.10 s`. The ramps are `τ·ln(ratio)` per end, so the shipped legs are
//! SHORTER by exactly the τ ratio on their ramp terms; both columns are printed side by side.
//! When a measured boot p99 arms `VD_BOOT_TICKS_P99`, τ grows and this gate's derived band moves
//! WITH it — nothing here is pinned to today's τ as a number.
//!
//! THE LEG DEFINITIONS are the addendum's own (§A3.2), flown on THE WORLD THIS CLUSTER BOOTS —
//! all three legs, one world (SL5).
//!
//! ★ THE TWO IN-SYSTEM LEGS WERE RE-DERIVED 2026-08-21 (the gate-pass arc), and this is the one
//! change in this file worth reading. They used to fly the addendum's CITED target distances
//! (`TARGET_PLANET_SOI_OUTER_HOME_M` / `TARGET_SYSTEM_BOUND_HOME_M`), which are *seed 0's*
//! `System(7)` numbers, while the warp leg flew THE world's LIVE star gap and the second test
//! below booted `DEV.universe_seed`. One table, three legs, TWO DIFFERENT WORLDS. That was
//! defensible while the in-system re-solve was still owed — the citation stood in for a generator
//! that did not yet solve those shells — but the taxonomy flag day landed it: the generator now
//! SOLVES every system shell (`system_shell_r_m`), and `TARGET_SYSTEM_BOUND_HOME_M` is pinned
//! EQUAL to seed 0's solved home shell by `g_star_shell_unmoved_the_stars_clearance_arm_never_binds`.
//! The reason to cite is discharged, and the cost of citing had become MEASURABLE: at the shipped
//! default seed the home system's own shell is `5.089e12 m` against the cited `1.582e11 m` — a
//! factor of THIRTY-TWO. The gate was flying a home system nobody lives in, and departing the warp
//! leg at a ceiling 32× under the real one.
//!
//! So both in-system legs now read the home realm and its outermost world off
//! `vd_bins::boot_world(DEV.universe_seed, …)`. The CITED targets are still PRINTED beside the
//! derived distances — the same discipline this file already applies to the addendum's placeholder-τ
//! seconds — so the design citation stays visible and stays live without deciding what is flown.

use vd_bins::DEV;
use vd_core::flight::{
    FlightTuning, TRAVERSE_S, WAKE_PIPELINE_TICKS, approach_ceiling_mps, leg_time_s, ramp_cap_mps,
    realm_speed_cap_mps, throttle_axes_scale,
};
use vd_core::geometry::BoundaryTuning;
use vd_physics::worldgen::{
    TARGET_PLANET_SOI_OUTER_HOME_M, TARGET_SYSTEM_BOUND_HOME_M, UniverseConfig,
    target_system_bound_max_m,
};

/// THE world, exactly as every shard boots it (SL5: one world, no preset, no test variant).
fn world() -> UniverseConfig {
    UniverseConfig::world(DEV.move_speed, DEV.tick_dt)
}

/// The two IN-SYSTEM leg distances, read off THE world the dev cluster boots — never stated.
struct HomeLegs {
    /// The home system's own solved shell: half the "edge to edge" leg.
    system_shell_m: f64,
    /// The OUTER world of that system — the direct child with the widest orbit — and its own
    /// containment reach, which is the "planet surface out to its own shell" leg.
    outer_planet_soi_m: f64,
}

/// Read [`HomeLegs`] off the booted world. The home realm is a LINEAGE POSITION
/// (`default_home_realm`, the same call `world_roster` makes), never a stated seed; the outer world
/// is chosen by the WIDEST ORBIT the system itself authored, never by whichever child happens to
/// carry the biggest shell — "outer" is a fact about the orbit, so it is read off the orbit.
fn home_legs() -> HomeLegs {
    let world = vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
    let home = vd_core::worldgen::default_home_realm(world.regions())
        .expect("THE world names a home realm (root → galaxy → system)");
    let held = std::collections::BTreeSet::from([home]);
    let (regions, movers) = vd_bins::boot_regions_and_movers(
        DEV.universe_seed,
        &held,
        home,
        DEV.move_speed,
        DEV.tick_dt,
    );
    let extent_of = |realm| {
        regions
            .iter()
            .find(|r| r.realm == realm)
            .expect("a shard's own realm and its children are in the neighbourhood it boots with")
            .shape
            .finite_extent()
    };
    let (outer, _) = movers
        .iter()
        .max_by(|a, b| a.1.sma.total_cmp(&b.1.sma))
        .expect("THE world's home system authors movers");
    HomeLegs {
        system_shell_m: extent_of(home),
        outer_planet_soi_m: extent_of(*outer),
    }
}

/// The AoI demand beat in ticks — the shard's own expression (`aoi_recheck_cadence` at the DEV
/// posture: no self-fence recheck ⇒ the tick-derived half-second beat), spelled identically here
/// so the gate's τ and the sim's τ are the same number by the same arithmetic.
fn aoi_cadence_ticks() -> u64 {
    (((1.0 / DEV.tick_dt).round() as u64) / 2).max(1)
}

/// THE shipped flight tuning — the ONE derivation (`vd_core::flight::FlightTuning::derive`), fed
/// the DEV cluster's own numbers, exactly as the shard bin feeds its own.
fn tuning() -> FlightTuning {
    FlightTuning::derive(
        DEV.move_speed,
        DEV.tick_dt,
        aoi_cadence_ticks(),
        u32::try_from(DEV.boot_ticks_p99).expect("boot p99 fits the config type"),
    )
}

/// One leg's derived definition: a distance, the containing realm's ceiling, the departure speed
/// and the arrival ceiling (`v_end == v_cap` ⇒ no arrival body — an outward leg ends at speed).
struct Leg {
    name: &'static str,
    distance_m: f64,
    v_cap: f64,
    v_start: f64,
    v_end: f64,
    /// The addendum's §A0.6/§A3.2 prediction at ITS placeholder τ = 2.44 s — printed beside the
    /// measurement, never asserted (every §A3 number was stamped ⟨UNMEASURED — τ⟩). The WARP row's
    /// figure was additionally computed over the pre-mass-cap star gap, so it now sits beside a leg
    /// two thirds its length; both columns are history, and the print says so.
    addendum_s: f64,
    /// The addendum's CITED design target for this leg's distance, where it has one — printed
    /// beside the derived distance, never asserted. Same discipline as `addendum_s`: the citation
    /// stays visible and stays live without deciding what is flown. `None` for the warp leg, whose
    /// distance was always THE world's own.
    cited_distance_m: Option<f64>,
}

/// FLY one leg tick-by-tick with the integrator's own law: full throttle under
/// `min(realm ceiling, governor arm, ramp)` — the exact per-tick arithmetic of
/// `vd_sim::stub::integrate` at `|axes| = 1`, including the throttle map (which at full stick
/// commands the ceiling exactly). Returns the measured seconds.
fn fly(leg: &Leg, t: &FlightTuning) -> f64 {
    let mut x = 0.0_f64;
    let mut v_prev = leg.v_start;
    let mut ticks: u64 = 0;
    while x < leg.distance_m {
        // The governor's arrival arm: the ceiling at the current distance-to-go toward the
        // arrival body's bound (inert when the leg ends at speed: v_end == v_cap dominates).
        let v_allowed = leg
            .v_cap
            .min(approach_ceiling_mps(leg.v_end, leg.distance_m - x, t.tau_s));
        let ramp = ramp_cap_mps(v_prev, t.v_foot_mps, t.tick_dt_s, t.tau_s);
        let v = throttle_axes_scale(1.0, t.v_foot_mps, v_allowed.min(ramp)) * t.v_foot_mps;
        x += v * t.tick_dt_s;
        v_prev = v;
        ticks += 1;
        assert!(
            ticks < 100_000_000,
            "leg {} runaway: {x} m after {ticks} ticks",
            leg.name
        );
    }
    ticks as f64 * t.tick_dt_s
}

#[test]
fn g_flight_table_the_governed_legs_match_the_closed_form_and_are_printed_verbatim() {
    let t = tuning();
    // τ IS the wake budget in seconds — the §4.2(c) identity, asserted, not assumed.
    assert_eq!(
        t.tau_s,
        (2 * aoi_cadence_ticks() + DEV.boot_ticks_p99 + WAKE_PIPELINE_TICKS) as f64 * DEV.tick_dt,
    );
    let cfg = world();
    let v_foot = DEV.move_speed;

    // THE world's own home system and its outermost world — the two in-system legs, DERIVED.
    let home = home_legs();

    // The ceilings, each from the ONE cap expression over a derived extent.
    let cap_planet = realm_speed_cap_mps(home.outer_planet_soi_m, v_foot, TRAVERSE_S);
    let cap_home = realm_speed_cap_mps(home.system_shell_m, v_foot, TRAVERSE_S);
    let cap_max = realm_speed_cap_mps(target_system_bound_max_m(), v_foot, TRAVERSE_S);
    let cap_galaxy = realm_speed_cap_mps(cfg.scale.galaxy_r_m, v_foot, TRAVERSE_S);

    let legs = [
        // §A3.2 row 1: planet surface → its own shell. Departs at foot speed (a surface is
        // human-scale by the collision-floor law), ends AT the shell at the planet's own ceiling
        // (an outward leg — no arrival ramp; v_end == v_cap makes the governor arm inert).
        Leg {
            name: "planet surface -> its own shell",
            distance_m: home.outer_planet_soi_m,
            v_cap: cap_planet,
            v_start: v_foot,
            v_end: cap_planet,
            addendum_s: 117.2,
            cited_distance_m: Some(TARGET_PLANET_SOI_OUTER_HOME_M),
        },
        // §A3.2 row 2: the home system, edge to edge — foot speed both ends (each edge is a
        // surface-class departure/arrival in the addendum's own integration).
        Leg {
            name: "home system, edge to edge",
            distance_m: 2.0 * home.system_shell_m,
            v_cap: cap_home,
            v_start: v_foot,
            v_end: v_foot,
            addendum_s: 248.7,
            cited_distance_m: Some(2.0 * TARGET_SYSTEM_BOUND_HOME_M),
        },
        // §A3.2 row 3: THE WARP LEG — star to star down THE world's LIVE 3-D placement radius,
        // departing at the home system's ceiling, arriving onto the largest system's ceiling.
        Leg {
            name: "warp, star to star (the 3-D placement gap)",
            distance_m: cfg.stellar.galaxy_rim_r_m,
            v_cap: cap_galaxy,
            v_start: cap_home,
            v_end: cap_max,
            addendum_s: 130.2,
            cited_distance_m: None,
        },
    ];

    eprintln!(
        "[flight-table] tau = {:.3} s (T_WAKE: 2x{} beat + {} boot + {} pipeline ticks at dt {}), \
         T_TRAVERSE = {TRAVERSE_S} s, v_foot = {v_foot} m/s",
        t.tau_s,
        aoi_cadence_ticks(),
        DEV.boot_ticks_p99,
        WAKE_PIPELINE_TICKS,
        DEV.tick_dt,
    );
    for leg in &legs {
        let closed = leg_time_s(leg.distance_m, leg.v_cap, leg.v_start, leg.v_end, t.tau_s)
            .expect("every table leg holds a cruise");
        let measured = fly(leg, &t);
        // THE DERIVED BAND: the tick-flown leg may differ from the continuous closed form only by
        // the discretization it actually carries — one tick of arrival quantization per leg, one
        // tick of ramp rounding per end, and the per-tick exponential-vs-compound residue, which
        // is bounded by (dt/τ) of the two ramp times. Every term named; nothing tuned.
        let ramp_s =
            t.tau_s * (leg.v_cap / leg.v_start).ln() + t.tau_s * (leg.v_cap / leg.v_end).ln();
        let band = 3.0 * t.tick_dt_s + ramp_s * (t.tick_dt_s / t.tau_s);
        // The CITED column: the addendum's design target for this leg's distance beside the world's
        // own, so the citation is readable in the owner's table without deciding anything.
        let cited = match leg.cited_distance_m {
            Some(m) => format!(
                " | addendum's cited seed-0 target {m:.6e} m ({:.2}x the world's)",
                leg.distance_m / m,
            ),
            None => String::new(),
        };
        eprintln!(
            "[flight-table] {}: D = {:.6e} m, v_cap = {:.6e} m/s, v_start = {:.6e}, v_end = {:.6e} \
             => MEASURED {measured:.2} s | closed form {closed:.2} s | band ±{band:.2} s | \
             addendum (tau 2.44 s placeholder) {:.1} s{cited}",
            leg.name, leg.distance_m, leg.v_cap, leg.v_start, leg.v_end, leg.addendum_s,
        );
        assert!(
            (measured - closed).abs() <= band,
            "leg '{}': measured {measured:.3} s is outside the derived band ±{band:.3} s around \
             the closed form {closed:.3} s — the integrator and the law's own integral disagree",
            leg.name,
        );
        // The owner's bar, per leg: journeys in MINUTES — every leg under ten minutes, and the
        // cruise identity survives (the cruise term of a one-realm-radius leg is T/2).
        assert!(
            measured < 600.0,
            "leg '{}' takes {measured:.1} s — the owner's minutes bar is broken",
            leg.name,
        );
    }
    // The §A3.2 cruise identity, STATED GENERALLY (2026-08-20 — a proof rewrite, never a
    // weakening). The galaxy's ceiling is derived from the galaxy's own radius, so the cruise term
    // of the star-to-star leg is `(gap / R_gal) · T/2` for ANY gap; it used to be quoted as plain
    // `T/2` only because the placement radius happened to be 99.986 % of the galaxy radius. The
    // DERIVED mass cap reserved a real share of the galaxy for its largest possible child, so the
    // gap is now 66.66 % of R_gal and the coincidence is gone — the identity is not.
    let warp = &legs[2];
    let cruise = (warp.distance_m
        - t.tau_s * (warp.v_cap - warp.v_start)
        - t.tau_s * (warp.v_cap - warp.v_end))
        / warp.v_cap;
    let gap_share = warp.distance_m / cfg.scale.galaxy_r_m;
    let expected_s = gap_share * TRAVERSE_S / 2.0;
    eprintln!(
        "[flight-table] the cruise identity: gap/R_gal = {:.4} % => cruise {cruise:.2} s against          (gap/R_gal)·T/2 = {expected_s:.2} s",
        100.0 * gap_share,
    );
    assert!(
        (cruise - expected_s).abs() < 0.1 * expected_s,
        "the warp cruise term {cruise:.2} s strayed from (gap/R_gal)·T/2 = {expected_s:.2} s",
    );
}

/// Task item 6 — THE DERIVED BAND BRACKET AT GOVERNED SPEEDS: on THE world, every boundary the
/// speed law ACCELERATES (a child of an ambient realm whose ceiling exceeds the foot speed —
/// the star systems in the galaxy, the galaxy in the universe) still wakes ahead and still
/// latches:
///
/// 1. WAKE: the governed dwell inside the child's own wake band — from its `spin_up` radius down
///    to its bound at the governor's falling ceiling, `τ·ln(1 + (S−E)/(v_c·τ))` — covers the
///    whole T_WAKE budget (τ itself, by the §4.2(c) identity), so the demand loop has its full
///    pipeline before any crossing. (In-system boundaries are UNTOUCHED by the law — their
///    parents clamp to the foot ceiling, and the landed battery already proves their wake story.)
/// 2. LATCH: at the governed band-edge arrival speed (`v_c + E/τ` — §A3.5's corrected quantity,
///    never the child's bare cap) the subject spends at least `n_entry` ticks inside the child's
///    acquire depth, so a governed crossing can always latch (`crossing_flythrough` stays
///    impossible by arithmetic, not by luck).
///
/// The fly-ahead process gate (`rlm_demand_login`) proves row 1 LIVE; this pins the whole roster.
#[test]
fn g_governed_bands_bracket_every_ambient_boundary_on_the_world() {
    let t = tuning();
    let cfg = world();
    let world_view = vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
    let regions = world_view.regions();
    let n_entry = f64::from(BoundaryTuning::DEFAULT.n_entry);
    let mut governed_rows = 0u32;
    for r in regions {
        let Some(parent) = r.parent else { continue };
        let parent_extent = regions
            .iter()
            .find(|p| p.realm == parent)
            .expect("every parented region's parent is in the forest")
            .shape
            .finite_extent();
        let parent_cap = realm_speed_cap_mps(parent_extent, DEV.move_speed, TRAVERSE_S);
        if parent_cap <= DEV.move_speed {
            // An in-system boundary: the containing ceiling IS the foot speed — the law is inert
            // here by the collision-floor clamp, and the landed battery is its proof.
            continue;
        }
        governed_rows += 1;
        let extent = r.shape.finite_extent();
        let child_cap = realm_speed_cap_mps(extent, DEV.move_speed, TRAVERSE_S);
        let spin_up = r.aoi.spin_up_r_m();
        assert!(
            spin_up > extent,
            "{:?}: a live wake band must reach beyond the bound (spin {spin_up} vs extent {extent})",
            r.realm,
        );
        // Row 1 — the governed wake dwell covers the whole T_WAKE budget.
        let dwell_s = t.tau_s * (1.0 + (spin_up - extent) / (child_cap * t.tau_s)).ln();
        eprintln!(
            "[governed-bracket] {:?} in {parent:?}: spin_up {spin_up:.3e} m, extent {extent:.3e} m, \
             child cap {child_cap:.3e} m/s => governed wake dwell {dwell_s:.2} s vs T_WAKE {:.2} s",
            r.realm, t.tau_s,
        );
        assert!(
            dwell_s >= t.tau_s,
            "{:?}: the governed approach crosses the wake band in {dwell_s:.3} s, inside the \
             T_WAKE budget {:.3} s — the demand loop could not wake it ahead",
            r.realm,
            t.tau_s,
        );
        // Row 2 — the governed band-edge arrival latches: n_entry ticks inside the acquire depth.
        let acquire_depth = extent - cfg.band.inset_m;
        let v_edge = approach_ceiling_mps(child_cap, extent, t.tau_s);
        let in_band_ticks = acquire_depth / (v_edge * t.tick_dt_s);
        eprintln!(
            "[governed-bracket] {:?}: band-edge governed speed {v_edge:.1} m/s => {in_band_ticks:.1} \
             in-band ticks (n_entry {n_entry})",
            r.realm,
        );
        assert!(
            in_band_ticks >= n_entry,
            "{:?}: a governed crossing holds only {in_band_ticks:.2} in-band ticks, under \
             n_entry {n_entry} — the latch is not guaranteed",
            r.realm,
        );
    }
    // NON-VACUITY, in two statements that fail for different reasons.
    //
    // (a) THE SHAPE, derived: since the true-size re-solve EVERY parented row is a governed one —
    //     no boundary of THE world is left to the collision-floor clamp. This is what the `continue`
    //     arm above would silently swallow, so it is stated rather than counted, and it cannot go
    //     stale when the census moves.
    let parented_rows =
        u32::try_from(regions.iter().filter(|r| r.parent.is_some()).count()).expect("fits u32");
    assert_eq!(
        governed_rows, parented_rows,
        "a parented row of THE world is NOT governed — its parent's ceiling clamped to the foot \
         speed, and the collision-floor arm above skipped it without saying so",
    );
    // (b) THE CENSUS, pinned: a deliberate tripwire on the world's own roster, not a magic number.
    //     ★ RE-PINNED 40 → 56 (2026-08-20) when `DEV.universe_seed` became the HOME SEED, so this
    //     gate reads the world the cluster actually boots instead of seed 0.
    //     ★ DECOMPOSITION CORRECTED 2026-08-21 (the gate-pass arc) — MEASURED off this gate's own
    //     roster print, because the old comment attributed all 22 moons to the home star:
    //     4 System rows (the galaxy + 3 star systems) + 3 Star rows + 27 planets (9 per system) +
    //     22 moons = 56. Of those 22 moons, 19 are the home system's and 3 belong to one sibling;
    //     the other sibling holds none. (The derived mass cap did NOT move the total: seed 0 still
    //     holds 6 census moons under it, as
    //     `g_climb_the_worlds_measured_climb_at_the_true_size_resolve` measures.)
    //     ★ RE-PINNED 56 → 51 IN S12 (2026-08-28), and MEASURED off this gate's own roster print,
    //     never inferred: 1 Galaxy + 3 System + 3 Star + 27 planets + 17 moons = 51. Only the moons
    //     moved (22 → 17). The placement became a SHAPE and takes SIX seed draws where the shell
    //     took two, so every draw after them shifted by four; five planets drew a lighter mass, and
    //     a lighter planet holds no moon.
    assert_eq!(
        governed_rows, 51,
        "THE world's governed-boundary roster changed — restate this gate against the new world",
    );
}
