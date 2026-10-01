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

**The fifth change (2026-09-29, the points pass of the temporal path's T2):** `atmosphere/mod.rs`,
`atmosphere/types.wgsl`, `atmosphere/render_sky.wgsl` — a setting `AtmosphereSettings::sun_disk_on_sky` (true
by default, upstream's behaviour) says whether the sky pass adds the sun's disc on sky pixels. Under the temporal
path the game sets it false: the reconstruction draws a jittered render picture smaller than the display, the
disc's edge would crawl with the jitter, and the points pass draws the disc after the reconstruction at the
display size instead, by the same disc law (`sample_sun_radiance`), times the transmittance table and the sky
share (the temporal path design §7.7, "difference 2"). `atmosphere/mod.rs` also makes the air's graph labels
(`AtmosphereNode`) public, so the temporal path's reactive mask can copy the picture after the sky pass and before
the transparent pass (the design §4.2 step 3).

**The sixth change (2026-09-30, the air under its ground; ruling B25):** `atmosphere/mod.rs`,
`atmosphere/resources.rs`, `atmosphere/types.wgsl`, `atmosphere/functions.wgsl`, `atmosphere/render_sky.wgsl`,
`atmosphere/sky_air.wgsl`, `render/pbr_lighting.wgsl` (every hunk marked `VOXELDUST PATCH (the air under its
ground)`). Upstream lifts an eye under the ground sphere onto it (`clamp_to_surface`) and ends every ray at the
sphere, so a ray under the eye's level met the sphere at once and took no air. MEASURED (2026-09-30, the owner low
over the dry sea floor): an eye 2 135 m under the sphere drew a hard line at its own level — the ground above it
hazed, the ground under it bare. The change:
- the component states the air UNDER the sphere (`AtmosphereBelow`, three uniform words): every term holds its
  value at the sphere (the medium's table clamps to its edge) and ONE term may grow by its exponential law at
  the coefficient the game states (the game's molecules by the pressure law under ruling B25; NOUGHT since its
  ruling B26 — no term grows under the sea's level); the default states no growth and a clear medium;
- the per-pixel march starts at the TRUE eye (`get_eye_position`; the tables' reads keep the lifted
  `get_view_position`), and `raymarch_atmosphere` names its ray (`RAY_TABLE_SKY`, upstream's sky ray of the
  tables' builders; `RAY_TO_SURFACE`; `RAY_PIXEL_SKY`): a pixel's ray is never ended by the sphere
  (`get_depth_segment`) — a ray to a drawn surface ends at its depth, a sky ray where it leaves the air or at
  the FLOOR sphere (`AtmosphereBelow::floor_radius`: no drawn ground stands lower), which reflects the
  sunlight as upstream's ground did. MEASURED (the owner's coast stand): a sky ray ended at the sea's sphere
  drew that sphere's plain ground between the sea's horizon and the far dry floor's (0.3° to 2.8° under the
  eye's level), a dark band from brightness 128 down to 99;
- a sample under the sphere adds the growth (Rayleigh phase) and takes the sunlight the sphere takes above it
  through the air between (`below_transmittance`: the held extinction plus the growth's integral straight up,
  times the chord over the depth in a cancellation-free form); the ground's own sunlight (`pbr_lighting.wgsl`)
  takes the same factor.
The tables' builders keep upstream's code (their rays run a fixed distance deep into the planet, where a growth
would overflow). Behaviour for upstream users: an eye and a surface over the sphere read exactly upstream's
values.

**The seventh change (2026-10-01, the sun at the sphere):** `atmosphere/functions.wgsl`,
`calculate_visible_sun_ratio` (marked `VOXELDUST PATCH (the sun at the sphere)`): the horizon's root takes its
argument held at nought. A ground under the sphere is lifted onto it (`clamp_to_surface`, one metre over), and at
a planet's radius a 32-bit length rounds by half a metre a coordinate, so the lifted radius can read under the
sphere; the root's argument went negative, the root NaN, and the fragment took no sunlight — a dark speck of
haze alone. MEASURED (the walk's stand, dunes 2–3 km off under the sea's level): 541 specks in 8 frames with the
sun's shadow on and 541 with it off, every one under the sphere. At the sphere the horizon is the local level.
Behaviour for upstream users: a surface over the sphere reads exactly upstream's value.

**Owed:** the upstream pull request (all seven changes); the patch dies at the Bevy upgrade gate (0.19).

**Known upstream limit kept:** `write_atmosphere_buffer` takes `.single()` over cameras with an
atmosphere, so the ground term is written for ONE atmosphere camera at a time.
