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

**Owed:** the upstream pull request (both changes); the patch dies at the Bevy upgrade gate (0.19).

**Known upstream limit kept:** `write_atmosphere_buffer` takes `.single()` over cameras with an
atmosphere, so the ground term is written for ONE atmosphere camera at a time.
