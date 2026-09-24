# bevy_pbr 0.18.1 — the VOXELDUST planet-centre patch

A verbatim copy of `bevy_pbr` 0.18.1 from crates.io, wired by `[patch.crates-io]` in the workspace
root, with ONE change: the atmosphere's planet centre is STATED instead of fixed under the world origin.

**Why.** Upstream's atmosphere (Hillaire 2020) is a spherical model, but three sites nail the planet's
centre at `(0, -bottom_radius, 0)` and take world Y as up — its own comment: *"if your world is
actually spherical, this will be wrong"* (`atmosphere/functions.wgsl`). Our renderer keeps the eye at
the origin of the origin realm's frame, so upstream would read the eye as standing on the ground at
every altitude. Design: `docs/investigation/2026-09-08/landforms/slice_8s_design.md` §2.3, §3.

**The diff** (every hunk is marked `VOXELDUST PATCH (planet centre)`):
- `atmosphere/mod.rs` — `Atmosphere::planet_center: Vec3` (metres); `earthlike()` sets upstream's
  value; `ExtractedAtmosphere` carries it.
- `atmosphere/resources.rs` — `GpuAtmosphere::planet_center`; the transform basis takes the local up
  as `normalize(eye_m - planet_center)` instead of `Vec3A::Y`.
- `atmosphere/types.wgsl` — `Atmosphere.planet_center: vec3<f32>`.
- `atmosphere/functions.wgsl` — `get_view_position` subtracts the centre.
- `render/pbr_lighting.wgsl` — the ground pixel's `O` is `-planet_center`.

**Behaviour for upstream users:** with the default (`earthlike`) value the expressions are
algebraically upstream's (`p + (0, R, 0)` ≡ `p − (0, −R, 0)`; `normalize(eye − (0, −R, 0))` at the
origin is `+Y`). Bevy's own `atmosphere` example is NOT run from this workspace (it needs Bevy's
assets), so that identity is stated, not measured; the measured control is G-SKY-CONTROL (the design,
§3.3).

**The second change (2026-09-19, per-sample sky):** `atmosphere/render_sky.wgsl` — under `MULTISAMPLED` the
fragment declares `@builtin(sample_index)` and reads that sample's depth, so the pass runs per sample and a
multisampled ground/sky edge keeps its coverage (upstream reads sample 0 and writes one colour to every sample:
the edge flattens into a stair — MEASURED on the ground stand's horizon, no crossing pixel blended). The cost is
one sky evaluation per sample instead of per pixel.

**The third change (2026-09-24, the tables the raymarch overwrites):** under the raymarched mode the sky
pass read the transmittance, sky-view and aerial tables and then overwrote each value with the march's;
it now reads only its own mode's path, and the aerial table (read by nothing else) is not built.

**Tried and taken out (2026-09-24): the open sky once per pixel.** A second, per-pixel draw shaded the
pixels whose every sample sees the sky once, and the per-sample draw discarded them. MEASURED in the window at
2560 × 1440 against the same stands: no gain on foot (31.7 → 31.9 frames a second), and MORE GPU load from
orbit (87 → 92 % at 11 499 km, 71 → 78 % at 16 954 km), because a ray that misses the air already returns
early and the second full-screen draw's depth reads cost what the saved marches were worth.

**The fourth change (2026-09-24, the air over a pixel marched once):** `atmosphere/sky_air.wgsl` (new),
`atmosphere/render_sky.wgsl`, `atmosphere/resources.rs`, `atmosphere/node.rs`, `atmosphere/mod.rs` — under
multisampling every sample of a pixel marched the same ray 16 steps deep, differing only where it stops; MEASURED,
that march was about half of every window frame. A compute pass (`sky_air`) now marches each pixel ONCE before the
sky pass and stores the result in two view-sized Rgba32Float textures: an all-sky pixel's march as it is, or an
all-ground pixel's march to its samples' MEAN distance when the samples lie within one step of the march of each
other. The per-sample sky pass takes the sky's march as it is, and extends or shortens the ground's by ONE PARTIAL
STEP of the same march to its own distance (the medium at the partial step's middle, the march's own step formula),
so every sample keeps its own transmittance and light to the march's own step accuracy; any other pixel (an edge)
marches per sample as before. The pass and the sky run together or not at all; the lookup mode skips the pass.
TRIED FIRST AND MEASURED: (a) one per-pixel colour for the whole pixel at the mean distance moved single pixels by up
to 26 levels where one sample met a different surface (the drawn sea floor under the water) — the samples' own
transmittance is what the partial step restores; (b) a one-surface test (affine depths) cured that but lost most of
the saving on foot; (c) an optical-depth bound between the samples kept the tolerance by construction but only near
the nadir from orbit.

**Owed:** the upstream pull request (all four changes); the patch dies at the Bevy upgrade gate (0.19).

**Known upstream limit kept:** `write_atmosphere_buffer` takes `.single()` over cameras with an
atmosphere, so the ground term is written for ONE atmosphere camera at a time.
