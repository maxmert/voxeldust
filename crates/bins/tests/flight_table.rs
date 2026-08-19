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
//! THE LEG DEFINITIONS are the addendum's own (§A3.2): the two in-system legs fly the CITED
//! target distances (`TARGET_*` — the in-system re-solve lands with the taxonomy slice; the pins
//! flip loudly if it lands different numbers), the warp leg flies THE world's LIVE derived star
//! gap (`system_ring_r_m` = the 3-D placement radius) between the two target system ceilings.

use vd_bins::DEV;
use vd_core::flight::{
    FlightTuning, TRAVERSE_S, WAKE_PIPELINE_TICKS, approach_ceiling_mps, leg_time_s, ramp_cap_mps,
    realm_speed_cap_mps, throttle_axes_scale,
};
use vd_core::geometry::BoundaryTuning;
use vd_physics::worldgen::{
    TARGET_PLANET_SOI_OUTER_HOME_M, TARGET_SYSTEM_BOUND_HOME_M, TARGET_SYSTEM_BOUND_MAX_M,
    UniverseConfig,
};

/// THE world, exactly as every shard boots it (SL5: one world, no preset, no test variant).
fn world() -> UniverseConfig {
    UniverseConfig::world(DEV.move_speed, DEV.tick_dt)
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
    /// measurement, never asserted (every §A3 number was stamped ⟨UNMEASURED — τ⟩).
    addendum_s: f64,
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

    // The ceilings, each from the ONE cap expression over a derived extent.
    let cap_planet = realm_speed_cap_mps(TARGET_PLANET_SOI_OUTER_HOME_M, v_foot, TRAVERSE_S);
    let cap_home = realm_speed_cap_mps(TARGET_SYSTEM_BOUND_HOME_M, v_foot, TRAVERSE_S);
    let cap_max = realm_speed_cap_mps(TARGET_SYSTEM_BOUND_MAX_M, v_foot, TRAVERSE_S);
    let cap_galaxy = realm_speed_cap_mps(cfg.scale.galaxy_r_m, v_foot, TRAVERSE_S);

    let legs = [
        // §A3.2 row 1: planet surface → its own shell. Departs at foot speed (a surface is
        // human-scale by the collision-floor law), ends AT the shell at the planet's own ceiling
        // (an outward leg — no arrival ramp; v_end == v_cap makes the governor arm inert).
        Leg {
            name: "planet surface -> its own shell",
            distance_m: TARGET_PLANET_SOI_OUTER_HOME_M,
            v_cap: cap_planet,
            v_start: v_foot,
            v_end: cap_planet,
            addendum_s: 117.2,
        },
        // §A3.2 row 2: the home system, edge to edge — foot speed both ends (each edge is a
        // surface-class departure/arrival in the addendum's own integration).
        Leg {
            name: "home system, edge to edge",
            distance_m: 2.0 * TARGET_SYSTEM_BOUND_HOME_M,
            v_cap: cap_home,
            v_start: v_foot,
            v_end: v_foot,
            addendum_s: 248.7,
        },
        // §A3.2 row 3: THE WARP LEG — star to star down THE world's LIVE 3-D placement radius,
        // departing at the home system's ceiling, arriving onto the largest system's ceiling.
        Leg {
            name: "warp, star to star (the 3-D placement gap)",
            distance_m: cfg.stellar.system_ring_r_m,
            v_cap: cap_galaxy,
            v_start: cap_home,
            v_end: cap_max,
            addendum_s: 130.2,
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
        eprintln!(
            "[flight-table] {}: D = {:.6e} m, v_cap = {:.6e} m/s, v_start = {:.6e}, v_end = {:.6e} \
             => MEASURED {measured:.2} s | closed form {closed:.2} s | band ±{band:.2} s | \
             addendum (tau 2.44 s placeholder) {:.1} s",
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
    // The narrowed §A3.2 identity that IS true: the warp leg's CRUISE term alone is T_TRAVERSE/2
    // to four figures (ring/R_gal = 99.986 %), asserted against the closed form's own parts.
    let warp = &legs[2];
    let cruise = (warp.distance_m
        - t.tau_s * (warp.v_cap - warp.v_start)
        - t.tau_s * (warp.v_cap - warp.v_end))
        / warp.v_cap;
    assert!(
        (cruise - TRAVERSE_S / 2.0).abs() < 0.1 * TRAVERSE_S / 2.0,
        "the warp cruise term {cruise:.2} s strayed from T/2 = {} s",
        TRAVERSE_S / 2.0,
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
    // Non-vacuity: THE world has governed boundaries — since the true-size re-solve EVERY
    // parented row is one (the galaxy + 3 systems + 27 planets + 3 stars + the 6 census moons).
    assert_eq!(
        governed_rows, 40,
        "THE world's governed-boundary roster changed — restate this gate against the new world",
    );
}
