# Client Known Issues

Issues that are known, root-caused (or partly root-caused), and currently
unresolved. Each entry documents the symptom, what was ruled out during
investigation, the upstream tracking issue, and the practical workarounds
available today.

## 1. Shadow acne — wandering bright dots / spots on shadowed hull surfaces

**Status**: known, not fixed. Upstream blocker on Bevy.

### Symptom

When the directional sun is enabled and the player is inside (or near) a
voxel ship that the server is rotating each tick, small bright pixels
("dots") wander across hull surfaces that should be in shadow. Each dot
appears for ~1 second then disappears, with new ones appearing in
different places. With Bloom enabled the dots get amplified into visible
glowing halos; without Bloom they are still present but smaller.

The dots scale with shadow-map texel size:

* Multi-cascade (e.g. 4 cascades, 4096²) → small "dots" (1–2 screen pixels).
* Single cascade with the same physical span → larger "spots", because
  each shadow texel covers more world-space.

The artefact appears on **block faces** and is **oriented to the face**
(rotates with the surface as the ship rotates). It does *not* appear in
empty space or on the starfield/celestial-body billboards.

The artefact appears **only when the SYSTEM shard is connected** — that
is, only when there is a directional sun reaching the scene through
authoritative stellar state. It is not specific to system rendering;
disabling the celestial-body sphere does not help. The trigger is
"directional sun is shining at AAA-bright illuminance".

### Root cause

Bevy 0.18's directional-light cascade shadow pipeline relies on a
constant `shadow_depth_bias` / `shadow_normal_bias` per `DirectionalLight`.
The shadow comparison samples the cascade depth map at a position offset
from the receiver fragment by these biases. When the bias is too small,
fragments that should be shadow-occluded by their own geometry register
"not in shadow" → fragment receives full sun illumination → bright pixel
on an otherwise dark surface.

Two things make our scene unusually sensitive:

1. **Voxel hulls are axis-aligned**. All hull faces meet at perfect 90°
   corners with constant per-face normals (no smooth-shaded interpolation
   to spread bias error across multiple fragments). So bias errors land
   sharply on individual fragments rather than smearing into a soft band.

2. **The `ChunkSource` parent rotates each frame** (the ship's pose flows
   from `PrimaryWorldState` / `WorldStateData.entities[]` into
   `ShardOrigin.rotation`, which `rebase_shard_transforms` applies to the
   parent's `Transform.rotation`). The cascade orthographic projection
   stays anchored to the camera in world-space, so as the ship rotates,
   the same chunk fragment crosses different shadow-map texels frame to
   frame — and a fragment near the bias threshold can flip from
   "in-shadow" to "lit" between frames purely from where it lands in the
   discretised shadow map. That flip lights it for one frame at the full
   100 000-lux sun illuminance, producing a visible dot.

### Things we ruled out during investigation

The following were each tested in isolation and **none of them fixed the
artefact**:

| Hypothesis | Test | Result |
| --- | --- | --- |
| HDR pipeline / AgX tonemap | `hdr_enabled = false`, `tonemap_enabled = false` | dots persist |
| Bloom amplification | `bloom_enabled = false` | dots persist |
| Bloom threshold = 0 picking up sub-bright noise | `Bloom::OLD_SCHOOL` (threshold 0.6) instead of `Bloom::NATURAL` | dots persist |
| `Bloom::NATURAL` `low_frequency_boost` | (covered by above) | n/a |
| Specular fireflies (chunk material reflectance default 0.5) | `reflectance: 0.0`, `perceptual_roughness: 1.0` | dots persist |
| Directional-light precision noise (sun_dir f32 cast) | f64 normalisation already in place in `solar.rs` | (no regression) |
| Star sphere casting CSM shadow | `NotShadowCaster` on celestial bodies (this was the *original* "no light, hull dark" bug) | fixed that bug; dots are separate |
| Star sphere bleed through chunk-edge gaps | celestial body sphere rendering disabled | dots persist |
| Starfield billboards bleeding through depth | `StarfieldPlugin` disabled | dots persist |
| AR markers painted into HUD tile pixel buffers | `draw_ar_markers` early-return at function entry | dots persist |
| HUD tile redraw producing widget pixels | `redraw_hud_textures` system removed from schedule | dots persist |
| HUD tile entities at all | `HudPlugin` disabled, `HudTile` entities forcibly hidden | dots persist |
| Sub-grid render entities | `SubGridPlugin` disabled | dots persist |
| `SurfaceLight` sub-block discs (8 cm warm-white quads on faces) | `SurfaceLight` mesh emit disabled | dots persist |
| PCF tap pattern | `ShadowFilteringMethod::Hardware2x2` instead of `Gaussian` | dots persist |
| Cascade-seam transitions | `cascade_count = 1` (single 0.1–200 m cascade) | dots become **larger** "spots" — confirmed shadow-comparison aliasing scales with shadow texel size |
| Bias too small | `shadow_depth_bias = 1.0`, `shadow_normal_bias = 10.0` (well above the values prior iterations called the "stable floor") | dots persist |

What **does** make the artefact disappear:

* `directional_shadows_enabled = false` — kills shadows entirely. The
  hull is then evenly lit by Lambertian + ambient with no occlusion.

This is the proof that the artefact is in the cascade-shadow comparison
path itself, not in any other rendering subsystem.

### Upstream Bevy status

Bevy 0.18.1 is the latest stable as of April 2026; main is in the 0.19
development cycle. Shadow handling has not changed substantially across
recent releases.

The relevant Bevy issue is
[#16075 — "Automatic shadow biasing"](https://github.com/bevyengine/bevy/issues/16075),
which **explicitly acknowledges that constant
`shadow_depth_bias` / `shadow_normal_bias` is fundamentally hard to
tune**: "It's hard to find a good value for this that doesn't result in
objects appearing to disconnect from their shadows." The state-of-the-art
fix is **Oriented Depth Bias** from Final Fantasy XVI's renderer; the
issue is marked Ready-For-Implementation but **not yet shipped**. An
earlier PR (#10188) attempted "Receiver Plane Depth Bias" and was not
merged.

Other related Bevy issues:

* [#16185 — "Shadows in 3D scene are flickering"](https://github.com/bevyengine/bevy/issues/16185)
  (open, S-Needs-Investigation; reported on AMD RX 7900 XTX, similar
  symptom on a different platform).
* [#14021 — "Transform changes after PostUpdate are not reflected in
  render"](https://github.com/bevyengine/bevy/issues/14021) — not our
  exact case (`rebase_shard_transforms` runs in `Update`, well before
  `PostUpdate`'s transform propagation), but documents the same class of
  frame-timing/synchronisation gotcha.
* [#3315 — "macOS intel with integrated gpu — strange shadows
  artefacts"](https://github.com/bevyengine/bevy/issues/3315) — Intel
  integrated only; doesn't apply to our M-series target hardware, but
  confirms backend-specific cascade artefacts have history on macOS.

### Why a config-tuning fix isn't sufficient

We tested every reasonable point in the bias / cascade space:

* `shadow_depth_bias`: 0.02 (Bevy default), 0.2, 1.0
* `shadow_normal_bias`: 1.8 (Bevy default), 4.0, 5.0, 10.0
* `cascade_count`: 1, 4
* `cascade_max_distance_m`: 200, 500, 1200
* `first_cascade_far_bound_m`: 8, 20
* `shadow_map_size`: 4096
* `ShadowFilteringMethod`: `Gaussian`, `Hardware2x2`

None of these eliminate the artefact. Larger biases reduce its frequency
slightly but introduce visible **peter-panning** (shadows visibly
detached from their casters at corners and crates) — the trade-off
called out in the upstream issue. Smaller biases bring the dots back
even more aggressively. There is no value in this space that is
simultaneously acne-free and contact-shadow-correct on a rotating
voxel-faceted hull.

### Practical options

Three paths forward, in order of investment:

#### Option A — Architectural fix (proper, AAA-quality)

Refactor `ShardOriginPlugin` + `PlayerSyncPlugin` so that:

* The `ChunkSource` parent stays at `Transform::IDENTITY` — chunks are
  **static in Bevy world-space**.
* The camera entity rotates and translates to the player's
  ship-local position instead of the ship rotating around a static
  camera.

This is what `voxydust-next` does (see comment in
`client/src/shard/origin.rs::rebase_shard_transforms` documenting the
voxydust-next reference at `voxydust-next/src/chunk_stream.rs:626-647`).
With chunks static, the cascade orthographic projection stops shifting
relative to chunk geometry frame-to-frame, the comparison sample lands
on the same texel each frame for a given fragment, and the bias-edge
oscillation goes away. Bevy's standard biases (`0.02 / 1.8`) work fine
under that arrangement (verified empirically in voxydust-next).

This is a substantial refactor: every system that reads
`Transform`/`GlobalTransform` of a chunk-attached entity has to be
audited (raycast, block highlight, AR projector, sub-block child
transforms, HUD tile transforms, sub-grid transforms, etc.), plus the
camera rebase needs to account for the player's local position inside
the ship, plus all of the above needs to work seamlessly across
ship-to-ship transitions and EVA exits.

#### Option B — Disable directional shadows for now

`LightingFidelity::medium().directional_shadows_enabled = false`. Gives
a clean (acne-free) look at the cost of contact shadows under crates,
seats, the player's own EVA shadow, etc. Sun direction still drives the
Lambertian / ambient terms so the scene retains directional contrast
between sunlit and anti-sun faces.

Useful as the shipping default until Option A or Option C lands.

#### Option C — Wait for upstream Oriented Depth Bias

Track [Bevy #16075](https://github.com/bevyengine/bevy/issues/16075).
When it merges, drop our voxel-tuned biases (revert to Bevy defaults)
and let the per-fragment automatic bias handle the corner cases.
Probably 1–2 Bevy minor releases away, judging from the issue activity.

### Repro / verification recipe

To reproduce on a fresh checkout:

1. `./dev-cluster.sh up` — bring up galaxy/system/ship/planet shards.
2. `cargo run -p client -- --gateway 127.0.0.1:7777 --name TestPilot`.
3. Wait for SHIP primary to connect and chunks to stream.
4. Sit in the pilot seat and move the mouse to rotate the ship slowly.
5. Watch the *shadowed* hull surfaces (anti-sun side, ceiling under
   overhangs, dark corners). Bright single-pixel-ish dots will appear
   for ~1 s and disappear, with new ones in different places.

To confirm it's the shadow-comparison path:

* Edit `client/src/config/graphics.rs::medium()` and set
  `directional_shadows_enabled: false`. Restart the client. Dots
  disappear immediately, anti-sun faces stay dim from ambient only.

### File / line references

The relevant code paths if you want to inspect or experiment:

* `client/src/lighting/solar.rs` — `update_solar_light` builds the
  `DirectionalLight` and `CascadeShadowConfig` from
  `LightingFidelity`. Search for `shadow_depth_bias` /
  `shadow_normal_bias` / `CascadeShadowConfigBuilder`.
* `client/src/lighting/camera.rs` — camera lighting stack; this is
  where you'd insert `ShadowFilteringMethod` if experimenting.
* `client/src/shard/origin.rs` — `rebase_shard_transforms` applies
  ship rotation to the `ChunkSource` parent every frame. This is the
  rotating-parent that interacts badly with the cascade. The function
  doc comment also references the voxydust-next "world = ship-local"
  approach (Option A).
* `client/src/config/graphics.rs::shadow_depth_bias_voxel` and
  `shadow_normal_bias_voxel` — the values currently shipped (Bevy
  defaults `0.02` / `1.8`).
