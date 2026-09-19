# Slice 8s — THE SKY AND THE SHADOW: the implementation design

**Status:** the design for the owner's discussion, 2026-09-18. Nothing here is built. Ruling T1 item 4
places 8s inside the shape arc, after 8b and before 8c (`owner_decisions_2026-09-16_terrain.md`, T4:
*"the same exception the light (8L) got — without it no vista can be judged"*). The owner's words of
2026-09-18 bind the shape of this design: the sky must work **multi-planet**, it must be **visible from
above the planet and super realistic, not a colour sphere around the planet**, the **landing must be
believable with no line crossed**, and the owner **agreed to the patch** of Bevy's atmosphere crate
(§3) after the measurement of its flat-world assumption (§2.3).

**The one-line summary.** The sky is a VOLUME OF AIR around a sphere, summed per pixel by Hillaire's
2020 model as Bevy 0.18 ships it. Every number the volume reads is a charter word or a published law
of those words (T9). Bevy's model carries ONE flat-world assumption in three places; a patched copy of
its `bevy_pbr` crate replaces the fixed planet centre with a stated one. Nothing crosses a realm
boundary that does not cross today (SL6: no new datum, no new lane). No byte of the terrain moves.

**Every explanation carries a game-word example, as the owner asked.**

---

## 0. THE LAWS THIS DESIGN OBEYS, AND HOW

| law | how 8s obeys it |
|---|---|
| SL3 a realm draws itself | the sky is the planet realm's own look, drawn from the planet's own charter; the parent states only the placement |
| SL6 ask before new data crosses | NOTHING new crosses. The charter (scale height, mean molecular weight, surface pressure, the Rayleigh optical depth, the bond albedo, the air flag) is already stated in the surface statement (`TAG_CHARTER`, 8b). The star's luminosity and class already ride the star row. The planet's centre is the placement the client already draws |
| SL10 the client only renders | the sky derives LOOK from the seed's words and the eye's stated placement; no pose, no velocity, no state is derived |
| T3 time | the sun's direction follows the star the client already places from the tick; the sky adds no clock |
| T9 no physical number is drawn | every term is computed from a charter word by a published law with a stated calibration body (§4); the two terms whose law is a later slice's are NAMED placeholders with a ledger row and an ask (§11) |
| HR5 100 % | the laws live in `vd-client` (Tier-A, renderer-free) as pure functions with their tests; `vd-client-render` (Tier-B) only wires Bevy |
| "no magic numbers" | the LUT sizes, sample counts and the mode are operational knobs in ONE config struct (`SkyConfig`), like `TerrainConfig`; every physical number is a charter word or a law |
| the battle-tested rule (memory: Hillaire/Bevy/webgpu-sky) | Bevy's own Hillaire implementation, patched in one uniform, never a second sky pass |
| SL8 a seam is a defect | the model is continuous in altitude (§2.1); the two places a seam COULD appear (the render mode, the body handover) each get a pixel measurement against the V18 tolerance (§6, §7) |

---

## 1. THE SKY TODAY (measured from the code, 2026-09-18)

- **The sky is black.** `ClearColor` is a fixed dark colour (`client-render/src/lib.rs:991`); the
  capture camera clears to black (`lib.rs:2822`).
- **One sun (8L).** `terrain::sync_terrain` spawns ONE `DirectionalLight` from the BRIGHTEST star the
  star cloud holds (`terrain.rs:3091` `let sun_from = brightest…`), with a cascaded shadow map whose
  reach is a ladder rung (`shadow_reach_rung`, `shadow_cascades`, `shadow_map_px` in `TerrainConfig`).
  The stub world's key light retires when the sun is born. **The cascades already exist**; 8s does not
  re-invent them (§4.7).
- **The sun's illuminance is style, not physics:** `sun_lux = π / exposure` (`terrain.rs:129`), the lux
  at which a white face square to the sun renders white. MEASURED reason in the comment: a physical
  100 000 lux at the engine's default exposure rendered the ground thirty times white.
- **No HDR, no tonemapper on the window camera** unless `VD_STAR_BLOOM=1` (`lib.rs:1128`); the probe
  camera is `Tonemapping::None` and must stay so (it encodes kind/rung/cells as colours).
- **The stars** are ONE additive sprite cloud (`AlphaMode::Add`, `lib.rs:582`) at `0.9 × far`
  (`SKY_DEPTH_MARGIN`), drawn in the TRANSPARENT phase. They were measured under LDR.
- **The global ambient light** is Bevy's default (a fixed grey): a magic light nobody stated.
- **The render frame:** the eye sits at the ORIGIN (`camera_transform`, `lib.rs:1682`); the picture is
  drawn in the origin realm's own axes; a body's box is placed by an f64 subtraction narrowed to f32
  (`lib.rs:2034`). The body under the eye and the eye in the body's frame are known per frame
  (`under_eye → (realm, body, eye_body)`, `terrain.rs:3174`).
- **The terrain material** is an `ExtendedMaterial` over `StandardMaterial` (`terrain.rs:1903`), so
  Bevy's surface-lighting path (and its `ATMOSPHERE` define, §2.4) applies to the ground.
- **The charter on the client:** `BodyCharter` (`core/src/look.rs:102`) decoded from the surface
  statement (`realm_scene.rs:124`). The sky reads: `scale_height_m`, `mu_q8`, `p_surf_pa`,
  `tau_vis_q12`, `bond_albedo_q12`, `t_surface_mk`, the flag `HAS_AIR`. All present since 8b.
- **The star row** (`look.rs:537`): `class_code`, `luma_lsun`, the cell; the cloud's placement gives
  the direction and the distance in the render frame.

---

## 2. BEVY'S ATMOSPHERE, MEASURED FROM ITS SOURCE (bevy_pbr 0.18.1)

### 2.1 What it is

`bevy_pbr/src/atmosphere/mod.rs:1-34`: *"implements Hillaire's 2020 paper on real-time atmospheric
scattering … supports dynamic time-of-day, multiple directional lights … applied as a post-processing
effect on top of the existing skybox, a starry skybox would automatically show based on the time of
day. Scattering in front of terrain … is handled as well."* The planet is a sphere of `bottom_radius`
with air to `top_radius` (`Atmosphere`, `mod.rs:210`). The ray walker (`get_raymarch_segment`,
`functions.wgsl:390-420`) has THREE cases: the eye under the ground, inside the air, outside the air.
The raymarched mode's doc names *"planets seen from orbit"* (`mod.rs:446`).

**Why a landing has no line to cross (owner's requirement).** Every pixel is the SAME integral at
every altitude: the light the air between the eye and the pixel's surface scatters toward the eye,
plus the light it takes away. Example: a hull descends from orbit onto the home planet. At 2 000 km the
shell under the hull is a thin blue rim on the globe's edge and the sky above is black with stars. At
100 km the rim widens; the stars near the horizon fade first, because their light crosses the most air.
At 10 km the sky above turns dark blue, at the ground blue. Each frame is the same sum with a different
eye position. The sun's light on the GROUND passes through the same shell (`pbr_lighting.wgsl:906`), so
the ground turns orange at sunset with no separate rule.

### 2.2 The pieces, by file

| piece | where | what it does |
|---|---|---|
| `Atmosphere` component | `mod.rs:210` | `bottom_radius`, `top_radius` (m), `ground_albedo`, `medium` handle; **requires `Hdr`** |
| `AtmosphereSettings` | `mod.rs:294` | LUT sizes, sample counts, `aerial_view_lut_max_distance` (default 3.2e4 m), `scene_units_to_m`, `rendering_method` |
| `ScatteringMedium` asset | `medium.rs:69` | a list of terms: `absorption`, `scattering` (m⁻¹), `falloff` (normalised over the shell), `phase` (Rayleigh / Mie / isotropic) |
| the LUTs | `transmittance_lut`, `multiscattering_lut`, `sky_view_lut`, `aerial_view_lut` (`.wgsl`) | transmittance by (r, μ); multiple scattering; the sky dome; a frustum-fitted 3-D in-scatter table |
| `render_sky` | `render_sky.wgsl` | per pixel: sky pixels (depth 0) from the sky-view LUT, ground pixels from the aerial LUT + segment transmittance; dual-source blend `out = inscatter + transmittance × dst`; `× view.exposure` |
| the sun on surfaces | `pbr_lighting.wgsl:906-926` (`#ifdef ATMOSPHERE`) | the directional light's colour × transmittance(r, μ_light) × the visible-sun ratio |
| `SunDisk` | `bevy_light/src/directional_light.rs:256` | `angular_size` (rad), `intensity` (1.0 = physical) |
| `AtmosphereEnvironmentMapLight` | `bevy_light/src/probe.rs:153` | the sky as an environment map: ambient diffuse + reflections; `intensity` 1.0 = the atmosphere's own; `size` |
| the graph order | `mod.rs:182-201` | LUTs after the prepasses; **`RenderSky` after `MainOpaquePass`, before `MainTransparentPass`** |
| GPU needs | `mod.rs:131,140` | compute shaders; `Rgba16Float` storage; dual-source blending optional (an averaging fallback) |

### 2.3 ★ THE FLAT-WORLD ASSUMPTION, three sites, its own words

```
// functions.wgsl:304   (the eye)
var world_pos = view.world_position * settings.scene_units_to_m + vec3(0.0, atmosphere.bottom_radius, 0.0);
// functions.wgsl:308-310
// We assume the `up` vector at the view position is the y axis, since the world is locally flat/level.
// NOTE: this means that if your world is actually spherical, this will be wrong.
// pbr_lighting.wgsl:909   (the ground pixel)
let O = vec3(0.0, atmosphere.bottom_radius, 0.0);
// resources.rs:548-552   (the CPU basis)
let atmo_y = Vec3A::Y;
```

The planet's centre is FIXED at `(0, −R, 0)` under the world origin, and world Y is up. Everything
after that is spherical (`r = length(world_pos)`, `up = normalize(world_pos)`, `ray_sphere_intersect`),
so the model IS a sphere; only the centre is nailed to the origin.

**MEASURED consequence for our renderer.** Our eye sits AT the origin (§1). So `view.world_position`
is zero every frame, `world_pos = (0, R, 0)`, `r = R`: Bevy would think the eye stands on the ground
at every altitude. The sky from orbit would be the sky at the beach, the rim would never appear, and
the ground-pixel term would read the wrong `r` for every chunk. That is the whole defect, and it is
one uniform wide.

### 2.4 The lookup mode's distance limit

`aerial_view_lut_max_distance` defaults to 32 km, sliced 32 ways with 10 samples a slice
(`functions.wgsl:162-183`; `aerial_view_lut.wgsl:29-40`): beyond the limit the table CLAMPS to its last
slice. From the orbit stand (2 000 km) the sample step is 200 km, and a 79 km shell (§4.2) gets zero or
one sample: the rim is wrong in that mode, and the limb of the globe from the far stand (58 Mm) worse.
The raymarched mode walks the shell only (`segment.start` = the atmosphere entry), so it is right at
every altitude. §6 measures the cost and decides.

### 2.5 One atmosphere per camera

`prepare_atmosphere_transforms` writes one transform per view with an `ExtractedAtmosphere`
(`resources.rs:527`); the shader binds ONE `atmosphere`. So the eye gets ONE body's air a frame. §7
states the rule that picks it and what a second body with air costs.

---

## 3. ★ THE PATCH (owner: agreed, 2026-09-18)

### 3.1 The mechanism

A patched copy of `bevy_pbr` 0.18.1 under `vendor/bevy_pbr`, wired by

```toml
[patch.crates-io]
bevy_pbr = { path = "vendor/bevy_pbr" }
```

Cargo resolves every `bevy_*` crate's `bevy_pbr` edge to the copy (same version, same API). The copy
carries a `PATCH.md` with the diff and the reason, and DEFERRED gets a row: *the upstream change
(a stated planet centre) is owed as a Bevy pull request; the patch dies when 0.19's gate opens* (the
Bevy upgrade gate ruling stands: no 0.19 before S11 + S12 and a hull flown in a window).

### 3.2 The diff, minimal and behaviour-preserving for Bevy's own users

1. `Atmosphere` gains `pub planet_center: Vec3` — **the planet's centre in the camera's world, in
   metres** (atmosphere units). Default `Vec3::new(0.0, -bottom_radius, 0.0)`, which is exactly the
   fixed point today, so Bevy's examples render unchanged (a MEASURED control: the `atmosphere` example
   before and after, byte-identical).
2. `ExtractedAtmosphere` and `GpuAtmosphere` carry it; `types.wgsl`'s `Atmosphere` struct gains
   `planet_center: vec3<f32>` (padded to 16 bytes as WGSL requires).
3. `functions.wgsl:304`: `world_pos = view.world_position * scene_units_to_m - atmosphere.planet_center`.
4. `pbr_lighting.wgsl:909-911`: `P_as = P_scaled - atmosphere.planet_center`.
5. `resources.rs:548`: `atmo_y = normalize(camera_translation * scene_units_to_m - planet_center)`
   (the local up), and the tangent basis built on it instead of on `Vec3A::Y`.
6. `get_local_up` (`functions.wgsl:311`) stays: it is used in ATMOSPHERE SPACE, where the eye is
   placed at `(0, r, 0)` by construction (`sky_view_lut.wgsl:30`), so it is right once `r` is right.

**The whole diff is under thirty lines.** Every other line of the model stays Bevy's.

### 3.3 The control that must fail first (never assume)

**G-SKY-CONTROL.** Before the patch: the ground stand and the orbit stand, with Bevy's unpatched
atmosphere on and the charter's terms, must show the SAME zenith colour (the flat-world reading: `r = R`
at both stands). After the patch: the orbit stand's zenith must be black within one channel step and
the rim must stand on the limb. A patch whose control did not fail proved nothing.

### 3.4 Precision at planet scale (a bound, then a measurement)

The eye's centre-relative position in f32 at |p| = 6.34e6 m has an ulp of 0.5 m. The density's
relative change over 0.5 m is `0.5 / H` ≈ 1/17 000 (H ≈ 8.4 km on the home planet, §4.2): far under
one channel step of any LUT. From the far stand (58 Mm, ulp 4 m) the same. The ground pixel's `r` in
`pbr_lighting` carries the same ulp. **G-SKY-ZENITH (§9) measures it**: the zenith colour at the
ground stand is compared with the analytic single-scatter value; a precision defect would show there.

### 3.5 The planet's centre, on the client

`planet_center` = the body's realm-box translation in the render frame, which the scene already
places for the terrain chunks (`lib.rs:2034`, an f64 subtraction narrowed to f32) — never a value the
client derives from anything but the stated placement (SL1: one hop, read-only). When the origin realm
is a hull inside the planet's frame, the planet's box is placed from the hull's stamped placement, as
the chunks already are. **No new datum.**

---

## 4. THE LAWS THE SKY READS (T9: computed, calibrated, never drawn)

All in `vd-client` as pure functions over `BodyCharter` and the star row, 100 % covered, with the
calibration body named. `vd-client-render` maps the results onto Bevy's structs and nothing else.

### 4.1 Rayleigh (the molecules; the blue)

The charter states the Rayleigh optical depth at 550 nm, `tau_vis_q12` (drawn by 8b stage 4 as a law
of the column mass), and the scale height `scale_height_m` (`H = kT/(μ m_H g)`, a law of three charter
words). The volume's scattering coefficient at 550 nm is therefore **exact by construction**:

```
β_550 = τ_vis / H                       [m⁻¹]  — the zenith optical depth of the sky equals the charter's word
β(λ)  = β_550 · (550 nm / λ)⁴            — Rayleigh's law
```

at the three wavelengths Bevy's own Earth term is sampled at, **680 / 550 / 440 nm** (MEASURED from its
constants: `13.558/5.802 = (680/550)⁴`, `33.1/13.558 = (550/440)⁴` — Bruneton's calibration). The
absorption term is zero. The phase is Rayleigh. The falloff is exponential with `scale = H / (top − R)`
(Bevy normalises the falloff over the shell, §4.2). **Calibration check, owed in the build:** on the
home planet `β_550` against Bevy's Earth value (`13.558e-6`), printed as a ratio; the charter's
pressure is Earth-like by T9's law, so the ratio is a test of the two derivations, not a knob.

Example: on the home planet the zenith sky at noon is blue and the sunset red, because the charter
states one Earth-like optical depth. On the 1 239 Pa Mars-thin body the 8b report found, the same law
gives a sky forty times thinner: nearly black at the zenith, a faint blue-white band on the horizon, and
the stars out by day. Nothing in the code names either body.

### 4.2 The top of the shell (where the air is ignored)

```
top_radius = R + H · ln(65 536)
```

the altitude where the density falls below ONE HALF-FLOAT STEP of the surface density (the LUTs are
`Rgba16Float`): below that the term cannot be represented, so the shell ends there by the format's own
quantum, never by a typed kilometre count. Home planet: `H ≈ 8.4 km → 93 km`. Earth's Hillaire/Bevy
value is 100 km; the difference is the rule stated versus a rounded one.

### 4.3 Mie (the aerosol; the white glow round the sun) — a NAMED PLACEHOLDER, ask 1

The aerosol load is a CLIMATE quantity: dust from aridity (8c's rain shadow, 8e's biome), salt from
the sea (8o), ash from volcanoes (no slice). None of those laws exist yet. **8s uses Bevy's Earth
aerosol term** (scattering `0.444e-6`, absorption `3.996e-6`, asymmetry 0.8, scale height 1.2 km)
**scaled by the surface number density ratio** `(p / T) / (p⊕ / T⊕)` — the one physical statement
available now (an aerosol needs a gas to hang in) — and states it as a placeholder in DEFERRED, the
way 8a's roughness rides a placeholder octave with a named owed law. On an airless body the term is
absent with the whole atmosphere. **The ask:** accept the placeholder with the ledger row, or hold
the Mie term at zero until 8e (a clear sky with no sun glow — measurable, and honest).

### 4.4 Ozone (the twilight's blue) — ask 2

Ozone exists where free oxygen does. No charter word states a biosphere. **Recommendation:** the ozone
term (Bevy's Earth tent profile) is present where the census reads the body EARTH-LIKE (liquid water
and air, the predicate 8b extended under T9) and absent otherwise, ledgered against the biosphere word
8e or later gives. It is a second-order colour term (the deep blue of the zenith at twilight); the
alternative is no ozone anywhere until the word exists.

### 4.5 The sun disc

`SunDisk { angular_size = 2 · R_star / d, intensity = 1.0 }`, where `R_star = √(L / (4π σ T⁴))`
(Stefan–Boltzmann) from the star row's `luma_lsun` and the class code's temperature (the same table
the census uses — VERIFIED in the build, ask 4 if the table is not there), and `d` the star's distance
in the render frame (the cloud's placement, already computed for the direction). The disc is the
physical intensity; the sun's own glare is bloom's job (§4.8).

### 4.6 The ground albedo (multiple scattering's second bounce)

`ground_albedo = bond_albedo_q12 / 4096`, the charter's word, stated as an interim in DEFERRED: the
bond albedo includes clouds, the surface albedo is 8e's paint table. It is a second-order term.

### 4.7 The shadow and the sky light

- **The cascades stay 8L's** (`shadow_reach_rung`, `shadow_cascades`): the reach is a ladder rung,
  so the shadow's sharpness follows the terrain's detail ladder, not a metre count. 8s changes nothing
  there; the T4 phrase "a cascaded shadow map" is landed.
- **The sky light:** `AtmosphereEnvironmentMapLight { intensity: 1.0 }` — the dome itself lights the
  shadows (blue fill on the shaded side of a ridge). `GlobalAmbientLight::NONE` replaces Bevy's
  default grey, which is a magic light. The map's `size` is an operational knob (`SkyConfig`).

Example: on the hill stand at noon the ridge's shaded flank is dark blue-grey, lit by the sky; on the
airless moon the same flank is black, lit by nothing, and the stars stand over it at noon.

### 4.8 Exposure, HDR, the tonemapper (a look decision, measured once) — ask 3

The `Atmosphere` component requires `Hdr`. Bevy's sky, sun disc and ground light are all LINEAR in the
directional light's illuminance, so the sky-to-ground ratio is physical whatever the sun's absolute
scale; **the style scale `π / exposure` stands**, and the sky inherits it. Under HDR a tonemapper maps
the sky's bright range to the screen; the design proposes Bevy's default under HDR (TonyMcMapface) and
bloom `NATURAL` on the window camera, judged in the pictures on the owner's look. The probe camera
stays `Tonemapping::None` with NO atmosphere (it encodes data as colours). The star gates were measured
under LDR: their pixels change once and are re-measured (§9).

---

## 5. THE STARS AND THE FRAME ORDER

`RenderSky` runs after the opaque pass and BEFORE the transparent pass (§2.2). Our stars are additive
sprites in the transparent phase, so as things stand they would be drawn OVER the finished sky at full
brightness: stars through the noon sky, a defect.

**The fix is in the order, not in a rule.** The star cloud draws BEFORE the sky pass: an opaque-phase
pipeline with additive blend and depth write OFF (a `specialize` change on `StarSkyMaterial`; the depth
stays 0, so the sky pass treats every star pixel as SKY). The sky pass then composes
`inscatter + transmittance × stars`: the stars are dimmed by the air's transmittance and drowned by the
day's in-scatter, which is the physical result. At night the transmittance at the zenith is
`e^(−τ_vis)` ≈ 0.90, so the stars are 10 % dimmer than today through the home planet's air, and
unchanged on an airless body.

Example: a player on the ground at noon sees no star. The same player at dusk sees the brightest stars
first, near the zenith, where the air is thinnest — the order real stars appear in.

---

## 6. THE MODE AND THE DISTANCE (the first seam candidate)

**★ MEASURED AND DECIDED (S5, 2026-09-19): the raymarched mode, always; no switch.** The lookup mode with
its table stretched to the drawn radius is a WRONG picture (up to 142 channel steps against the raymarched
one at the ground stand, 136 aloft; 7 % of the day side black from orbit), and the raymarched mode costs
1.7–4.7 ms a frame more at the four measured stands. Rule 1 below applies; rule 2 never ran.

Two modes exist (§2.4). A SWITCH between them at some altitude is a seam candidate (the taxonomy's
brightness-pop). The rule:

1. **Measure the raymarched mode's cost** on the seven stands and the descent leg (frame time, the
   average machine's share). If it fits the frame budget, **one mode always, no switch**.
2. If it does not fit: the lookup mode with `aerial_view_lut_max_distance` set PER FRAME to the drawn
   radius (the stamp's `drawn_radius_m`), and the raymarched mode above the shell; the switch altitude
   is where the two modes' pictures differ by at most ONE channel step (V18's tolerance) — measured by
   rendering both at that altitude, the difference printed. A switch that pops is refused, and the cost
   is paid instead.

Knobs (`SkyConfig`): the mode, the LUT sizes, the samples, the environment-map size, `VD_SKY=0` to
draw no atmosphere for a measurement (the picture gate's airless control, §9).

---

## 7. MULTI-PLANET: WHICH AIR THE EYE GETS, AND THE HANDOVER

One atmosphere a frame (§2.5). The rule, generic for every realm kind:

- **The eye gets the air of the body with air whose SHELL SUBTENDS THE LARGEST ANGLE at the eye** —
  `top_radius / distance`, the same drawable-floor idea reach uses. A body with `HAS_AIR` clear is never
  a candidate. No body with air in the sky ⇒ no atmosphere component ⇒ the picture is what it is today.
- **Standing on an airless moon and looking at its planet:** the planet's air is the largest shell in
  view, so the eye gets IT; the ray to the moon's ground crosses none of it (in-scatter zero), and the
  planet shows its rim. Correct with no special case, because the model's "outside the air" branch does
  the work.
- **The handover between two bodies with air** (flying from the home planet to a second airy planet):
  at the switch the eye is far from both, so both shells are thin at the eye; but the rim of the body
  that LOSES the component vanishes from its globe. **G-SKY-HANDOVER measures the pixel step at the
  switch distance**; if it exceeds one channel step the fix is a second pass for the far body, owed and
  ledgered, never a fudge. The census states how many systems hold two bodies with air within sight of
  each other; the count is printed by the bins example (owed, §10).
- **Two bodies with air in ONE frame** (a double planet): the second rim is missing until that second
  pass. Ledgered.

Example: a player leaves the home planet for its Mars-thin sibling. The home planet's rim shrinks as
the hull climbs; at the handover distance the sibling's shell is the larger in the sky, the component
moves to it, and the sibling's thin blue-white limb grows as the hull closes. The gate prints how many
pixels stepped at the handover frame.

---

## 8. WHAT CROSSES (SL6): nothing new

| datum | source | status |
|---|---|---|
| scale height, μ, pressure, τ_vis, albedo, T_s, HAS_AIR | the surface statement's charter (8b) | already crosses |
| the star's luminosity and class | the star row | already crosses |
| the planet's centre in the render frame | the placement the client already draws | already crosses |
| the sun's direction | the star cloud's placement from the tick | already derived |

No new wire arm, no new lane, no new word. The gateway and every shard are untouched by 8s.

---

## 9. THE GATES

| gate | what it measures | red when |
|---|---|---|
| G-SKY-CONTROL | unpatched (`VD_SKY_FLAT=1`): the flat centre must give a WRONG picture at the ground stand; patched: the same stand right. ★ MEASURED 2026-09-18: the flat centre read the ground BLACK (lit-ground paint 0.000 — the sun under the flat model's horizon, because world Y is not up in our frame), the patch read 0.998 with a blue sky. The orbit form (ground and orbit zenith equal) never ran: the flat run stopped at the first gate | the control did not fail, or the patch did not cure it |
| G-SKY-ZENITH | the ground stand's zenith colour against the analytic single-scatter value for the charter's τ_vis (Bruneton's closed form, computed in the test in f64) | > the LUT's stated quantum |
| G-SKY-AIRLESS | a stand on a body with `HAS_AIR` clear: the picture BYTE-IDENTICAL to the same stand with `VD_SKY=0` | any pixel differs |
| G-SKY-STARS | noon at the ground stand: no star pixel brighter than its sky neighbours; night: the star gate's references re-frozen once, then held | a star through the day sky; a night drift |
| G-SKY-LANDING | the moving-eye descent (a new leg: orbit → ground at the 528 m/s hull rating): the per-frame step of the mean sky colour and of the rim's width, against the model's OWN prediction for that altitude step × 2 | a frame steps more than the model predicts (a pop) |
| G-SKY-MODE | (only if a switch exists, §6) the two modes' pictures at the switch altitude | > 1 channel step |
| G-SKY-HANDOVER | (§7) the pixel step at the handover frame | > 1 channel step |
| G-SKY-COST | frame time on the seven stands and the descent leg, both modes, printed | a report; the knob decides |
| the paint gates under the sky (★ BUILT 2026-09-18, run 7 green) | the ball's red share is its SUNLIT FRACTION `(1 + cos φ)/2` from the stand's own sun (measured 0.763 vs 0.757, 0.746 vs 0.767); above ONE SCALE HEIGHT the ground and the ball are veiled by the air and must be LIT, their hue shares reported; NIGHT IS BLACK — the stamp's star row carries the sun's body-frame direction and every ground pixel is cast onto the globe: the day side lit ≥ 0.95, the night side reported | a day pixel black; a near-ground ball off its sunlit share by more than two one-pixel bands |
| the pictures | the seven stands re-frozen on the owner's look; TWO NEW STANDS: **dusk** (the ground stand with the sun 2° under the horizon: the twilight arch and the first stars) and **high** (300 km: the shell thick under the eye, the rim wide) | as today |

---

## 10. THE ORDER OF WORK

| stage | what lands | proof |
|---|---|---|
| S1 the patch | `vendor/bevy_pbr` + `[patch]`, `planet_center`, `PATCH.md`; the DEFERRED row for the upstream PR | Bevy's `atmosphere` example byte-identical before/after; **G-SKY-CONTROL fails, then passes** |
| S2 the laws | `vd-client::sky`: Rayleigh from τ_vis and H, the shell top, the placeholder Mie, ozone under the predicate, the sun disc, the albedo; `SkyConfig` | 100 % coverage; the calibration ratios printed |
| S3 the wiring | the component on the window and capture cameras (never the probe), the centre per frame, the medium asset per body, `GlobalAmbientLight::NONE`, the sky light, HDR + the tonemapper, `VD_SKY` | G-SKY-ZENITH, G-SKY-AIRLESS |
| S4 the stars | the star pass before the sky | G-SKY-STARS; the star gates re-frozen once |
| S5 the mode | the cost table; one mode or the measured switch | G-SKY-COST, G-SKY-MODE |
| S6 the bodies | the largest-shell rule, the handover | G-SKY-HANDOVER; the census count of airy pairs |
| S7 the pictures | the descent leg, the dusk and high stands, the seven stands re-taken | G-SKY-LANDING; the owner's look; the freeze on his word |

Under the ONE-JOB rule every gate flight waits for a quiet machine; the coverage gate runs once after
S2 and once after S7.

---

## 11. THE ASKS (the owner decides; the recommendation first)

**★ ANSWERED BY THE OWNER, 2026-09-18:** (1) the aerosol placeholder — the owner asked whether it
contradicts the design choices; the answer is §11.1 below, and the placeholder stands ONLY as a law
with a calibration body and a ledger row, never as a tune; (2) ozone PRESENT — *"we need realistic
experiences only"*; (3) the tonemapper and bloom as recommended, judged on the pictures; (4) the
star's temperature as recommended; (5) the mode as recommended (measured, never an unmeasured
switch); (6) the second airy body ledgered and measured, as recommended.

### 11.1 Does the aerosol placeholder contradict T9?

T9 says every physical fact is COMPUTED by a published law with a stated calibration body, and *"a
band standing in for a missing law is a defect"*. The placeholder is lawful only in this exact form:
**the aerosol's optical density is Earth's, calibrated on Earth's air, scaled by the surface number
density `(p/T)/(p⊕/T⊕)` — a law (an aerosol hangs in a gas in proportion to it), with a stated
calibration body (Earth), and no number typed by hand.** What it lacks is the SOURCE term (dust from
aridity, salt from the sea, ash), which is 8c/8e/8o's law; DEFERRED carries that row, the way 8a's
roughness rides a placeholder octave with its owed law. It would contradict T9 the moment anyone
tuned it to look good; the gate is that its only inputs are the two charter words. The alternative
(zero aerosol) is the less physical picture: no real air is aerosol-free, and the sun would stand in
a sky with no glow, which the owner's "realistic only" refuses.

1. **The aerosol placeholder (§4.3).** Recommend: Bevy's Earth aerosol scaled by the surface number
   density, ledgered against 8c/8e's dust law. Alternative: zero aerosol until 8e.
2. **Ozone (§4.4).** Recommend: present under the earth-like predicate, ledgered against the biosphere
   word. Alternative: absent everywhere.
3. **The tonemapper and bloom (§4.8).** Recommend: Bevy's default under HDR with natural bloom, judged
   on the pictures. It is a look decision, and the pictures decide.
4. **The star's temperature.** The class code must map to a temperature for the sun disc; if the
   census holds that table the disc is a law with no ask; if not, the table is a new derived fact
   (from the class code, a published main-sequence relation), stated here so it is not a surprise.
5. **The mode (§6).** Recommend: raymarched always if the measured cost fits; the measured switch
   otherwise; never an unmeasured switch.
6. **The second airy body (§7).** Recommend: ledger the second pass, measure the census count, build
   it when a pair is reachable — the same shape as the second galaxy's differences.

---

## 12. WHAT DOES NOT CHANGE

No byte of any chunk; the recipe, the pin (`DECLARED_PIN`), the digest, the no-drift and seam gates,
the collider's shape, the wire, the gateway, every shard. The frozen pictures change COLOUR and are
re-frozen once on the owner's look, which T1's free window allows before slice 14.
