//! ★ WHAT TODAY'S RAIN WOULD PAINT (ruling B98, the colours design §6 and §9 step 1): the land's 8 km rows read as the
//! colour laws would read them — the annual temperature, the rain, the aridity — and the shares the published cover laws
//! give: the woody-cover ceiling (Sankaran et al. 2005, `0.14 × MAP − 14.2 %` to the break at 650 mm), the dryland
//! thresholds (Berdugo et al. 2020, aridity `1 − P/PET` at 0.54 / 0.70 / 0.80). Read only; no world change.
//!
//! **Example.** On the home planet: the land's rain median, the share of the land under 250 mm, the share past Berdugo's
//! plant-cover collapse, and the woody ceiling's histogram — what the colours by law would show before the rain's cure.
//!
//! `cargo run --release -p vd-bins --example cover_census -- WORLD.art`

use vd_terrain::artifact::DRY_M;

/// The aridity byte's axis (the climate law's own: 0 at a PET/P ratio of 0.5, 255 at 20).
const WET_RATIO: f64 = 0.5;
const HYPER_RATIO: f64 = 20.0;

fn main() {
    let path = std::env::args().nth(1).expect("WORLD.art");
    let artifact = vd_bins::artifact_file::read(&path).expect("the world reads");
    let mut rain = Vec::new();
    let mut temp_c = Vec::new();
    let mut arid = Vec::new();
    for r in &artifact.rows {
        if r.water_m != DRY_M {
            continue;
        }
        // The rain class is `log2_class(P, 4)` (an exponent and four linear mantissa bits): its middle by the
        // generator's own inverse.
        let lo = vd_terrain::fine::l1::decode_class(r.rain, 4) as f64;
        let hi = vd_terrain::fine::l1::decode_class(r.rain.saturating_add(1), 4) as f64;
        let p = 0.5 * (lo + hi);
        let t = f64::from(r.temperature) + 173.0 - 273.15;
        let share = f64::from(r.aridity) / 255.0;
        let ratio = (WET_RATIO.ln() + share * (HYPER_RATIO.ln() - WET_RATIO.ln())).exp();
        rain.push(p);
        temp_c.push(t);
        arid.push(1.0 - 1.0 / ratio);
    }
    let n = rain.len() as f64;
    println!("land nodes: {} of {}", rain.len(), artifact.rows.len());
    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * q) as usize]
    };
    let (mut r2, mut t2) = (rain.clone(), temp_c.clone());
    println!(
        "rain mm/yr: p10 {:.0}  median {:.0}  p90 {:.0}; temperature °C: p10 {:.1}  median {:.1}  p90 {:.1}",
        pct(&mut r2, 0.1),
        pct(&mut r2, 0.5),
        pct(&mut r2, 0.9),
        pct(&mut t2, 0.1),
        pct(&mut t2, 0.5),
        pct(&mut t2, 0.9)
    );
    let share =
        |f: &dyn Fn(usize) -> bool| 100.0 * (0..rain.len()).filter(|&i| f(i)).count() as f64 / n;
    println!(
        "rain under 250 mm {:.1} %, 250–650 {:.1} %, 650–1000 {:.1} %, over 1000 {:.1} %",
        share(&|i| rain[i] < 250.0),
        share(&|i| (250.0..650.0).contains(&rain[i])),
        share(&|i| (650.0..1000.0).contains(&rain[i])),
        share(&|i| rain[i] >= 1000.0)
    );
    println!(
        "Berdugo aridity (1 − P/PET): under 0.54 {:.1} %, 0.54–0.70 {:.1} %, 0.70–0.80 {:.1} %, over 0.80 (plant cover collapses) {:.1} %",
        share(&|i| arid[i] < 0.54),
        share(&|i| (0.54..0.70).contains(&arid[i])),
        share(&|i| (0.70..0.80).contains(&arid[i])),
        share(&|i| arid[i] >= 0.80)
    );
    // The woody ceiling: a reading of the law's own value at each node (no cap is applied to the world).
    let ceil = |p: f64| 0.14 * p - 14.2;
    println!(
        "Sankaran woody ceiling: none (≤ 0 %) {:.1} %, 0–20 % {:.1} %, 20–40 % {:.1} %, 40–75 % {:.1} %, canopy may close (≥ 650 mm) {:.1} %",
        share(&|i| ceil(rain[i]) <= 0.0),
        share(&|i| (0.0..20.0).contains(&ceil(rain[i]))),
        share(&|i| (20.0..40.0).contains(&ceil(rain[i]))),
        share(&|i| ceil(rain[i]) >= 40.0 && rain[i] < 650.0),
        share(&|i| rain[i] >= 650.0)
    );
    println!(
        "temperature: under −10 °C {:.1} %, −10–0 {:.1} %, 0–10 {:.1} %, 10–20 {:.1} %, over 20 {:.1} %",
        share(&|i| temp_c[i] < -10.0),
        share(&|i| (-10.0..0.0).contains(&temp_c[i])),
        share(&|i| (0.0..10.0).contains(&temp_c[i])),
        share(&|i| (10.0..20.0).contains(&temp_c[i])),
        share(&|i| temp_c[i] >= 20.0)
    );
}
